"""Train routing CLF for xRAG.

xRAG has two models (SFR retriever + LLM) that may not fit on GPU together.
Pre-computes SFR embeddings per stage into stage_cache, then keeps LLM on GPU.
compress() returns cached embeddings via self.get_cache().

It requires xRAG modelling files from https://github.com/Hannibal046/xRAG
"""

import gc
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from overflowguard import OverflowRouter, TrainConfig, train_router, llm_judge
from overflowguard.train import load_dataset_rows

sys.path.insert(0, "/workspace/xRAG")
from src.model import SFR, XMistralForCausalLM
from src.language_modeling.utils import XRAG_TOKEN

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

RAG_TEMPLATE = "[INST] Refer to the background document and answer the questions:\n\nBackground: {document}\n\nQuestion: {question} [/INST] The answer is:"


class XragRouter(OverflowRouter):

    def _load_model(self, path, **kwargs):
        self.model = XMistralForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cpu",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, add_eos_token=False, use_fast=False, padding_side="left",
        )
        self.model.set_xrag_token_id(self.tokenizer.convert_tokens_to_ids(XRAG_TOKEN))

        retriever_name = "Salesforce/SFR-Embedding-Mistral"
        self.retriever = SFR.from_pretrained(
            retriever_name, dtype=torch.bfloat16, device_map="cpu",
        ).eval()
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name)

        self.mid_layer_index = 17
        self.doc_max_length = 180

    def precompute_embeddings(self, samples, stage, output_dir="./xrag_router_ckpt"):
        """SFR on GPU → compute all embeddings → store in stage cache → SFR off."""
        cache_path = Path(output_dir) / f"emb_cache_{stage}.pt"

        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cached.get("n_samples") == len(samples):
                log.info("Loaded %d cached SFR embeddings for stage %s", len(cached["embeddings"]), stage)
                self._stage_cache[stage] = cached["embeddings"]
                return

        dev = torch.device("cuda")
        self.retriever.to(dev)

        embeddings = {}
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        task = progress.add_task(f"SFR embeddings ({stage})", total=len(samples))
        progress.start()

        try:
            with torch.no_grad():
                for i, sample in enumerate(samples):
                    chunks = self._chunk_text(sample["context"])
                    inp = self.retriever_tokenizer(
                        chunks, max_length=180, padding=True, truncation=True, return_tensors="pt",
                    ).to(dev)
                    emb = self.retriever.get_doc_embedding(
                        input_ids=inp.input_ids, attention_mask=inp.attention_mask,
                    )
                    if emb.shape[0] > 1:
                        emb = emb.mean(dim=0, keepdim=True)
                    embeddings[i] = emb.cpu()
                    progress.update(task, advance=1)
        finally:
            progress.stop()

        self.retriever.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": embeddings, "n_samples": len(samples)}, cache_path)
        self._stage_cache[stage] = embeddings
        log.info("Pre-computed %d SFR embeddings for stage %s", len(embeddings), stage)

    def compress(self, documents, query=None):
        cached = self.get_cache(self._current_sample_idx)
        if cached is not None:
            return cached.to(self.model.device)

        # fallback: live SFR computation (swap LLM off if needed)
        llm_on_gpu = next(self.model.parameters()).is_cuda
        if llm_on_gpu:
            self.model.to("cpu")
            torch.cuda.empty_cache()

        dev = torch.device("cuda")
        self.retriever.to(dev)
        inp = self.retriever_tokenizer(
            documents, max_length=180, padding=True, truncation=True, return_tensors="pt",
        ).to(dev)
        embeddings = self.retriever.get_doc_embedding(
            input_ids=inp.input_ids, attention_mask=inp.attention_mask,
        ).clone()
        self.retriever.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        if llm_on_gpu:
            self.model.cuda()

        if embeddings.shape[0] > 1:
            embeddings = embeddings.mean(dim=0, keepdim=True)
        return embeddings.unsqueeze(0)

    def generate_compressed(self, compressed_embs, query, **kw):
        dev = self.model.device
        prompt = RAG_TEMPLATE.format(document=XRAG_TOKEN, question=query)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
        out = self.model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=kw.get("max_new_tokens", 128),
            pad_token_id=self.tokenizer.pad_token_id,
            retrieval_embeds=compressed_embs[:1, :1],
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    def generate_full(self, context, query, **kw):
        dev = self.model.device
        prompt = RAG_TEMPLATE.format(document=context, question=query)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
        out = self.model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=kw.get("max_new_tokens", 128),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    def extract_clf_features(self, compressed_embs=None, query=None):
        dev = self.model.device
        prompt = RAG_TEMPLATE.format(
            document=" ".join([XRAG_TOKEN] * compressed_embs.shape[0]),
            question=query,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=False)
        input_ids = inputs["input_ids"].to(dev)
        attention_mask = inputs["attention_mask"].to(dev)

        layers = self.model.model.layers
        captured = {}

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["h"] = h.detach()

        handle = layers[self.mid_layer_index].register_forward_hook(hook)
        try:
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                retrieval_embeds=compressed_embs,
            )
        finally:
            handle.remove()

        return captured["h"][0, -1, :].float()

    def _chunk_text(self, text, chunk_tokens=None):
        chunk_tokens = chunk_tokens or self.doc_max_length
        ids = self.retriever_tokenizer(text, add_special_tokens=False).input_ids
        return [
            self.retriever_tokenizer.decode(ids[i : i + chunk_tokens], skip_special_tokens=True)
            for i in range(0, len(ids), chunk_tokens)
        ] or [text]

    def park_gpu(self):
        self.model.cuda()
        if self.clf is not None:
            self.clf.to("cuda")

    def unpark_gpu(self):
        self.model.to("cpu")
        self.retriever.to("cpu")
        if self.clf is not None:
            self.clf.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    router = XragRouter.from_pretrained("Hannibal046/xrag-7b")

    cfg = TrainConfig(
        dataset="/data/train_squad.jsonl",
        eval_dataset="/data/test_squad.jsonl",
        output_dir="./xrag_router_ckpt",
        epochs=100,
        n_folds=5,
        push_to_hub=True,
        hub_repo_id="wexumin/xrag-7b-router",
    )

    # Phase 1: precompute SFR embeddings per stage (retriever on GPU, then off)
    train_samples = load_dataset_rows(cfg)
    router.precompute_embeddings(train_samples, router.TRAIN_COLLECT, cfg.output_dir)
    if cfg.eval_dataset:
        eval_cfg = TrainConfig(**{**cfg.__dict__, "dataset": cfg.eval_dataset, "output_dir": cfg.output_dir + "/eval"})
        eval_samples = load_dataset_rows(eval_cfg)
        router.precompute_embeddings(eval_samples, router.EVAL_COLLECT, eval_cfg.output_dir)

    # Phase 2: LLM on GPU for generate + extract + CLF training
    router.park_gpu()
    result = train_router(router, cfg, evaluator=llm_judge(concurrency=30, model="deepseek-chat"))
