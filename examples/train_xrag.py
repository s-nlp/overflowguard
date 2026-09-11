"""Train routing CLF for xRAG.

xRAG has two models (SFR retriever + LLM) that may not fit on GPU together.
Pre-computes SFR embeddings per stage into stage_cache, then keeps LLM on GPU.
compress() returns cached embeddings via self.get_cache().

It requires xRAG modelling files from https://github.com/Hannibal046/xRAG

Batched collection (cfg.collect_batch_size > 1) is supported via the *_batch
methods below. NOTE: compress_batch walks the SFR cache per sample, because
collect_features sets _current_sample_idx to the BATCH START only.
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
        # Mistral has no pad token and the single-sample paths never padded;
        # batched tokenization would raise without this.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        retriever_name = "Salesforce/SFR-Embedding-Mistral"
        self.retriever = SFR.from_pretrained(
            retriever_name, dtype=torch.bfloat16, device_map="cpu",
        ).eval()
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name)

        self.mid_layer_index = 17
        self.doc_max_length = 250

    def precompute_embeddings(self, samples, stage, output_dir="./xrag_router_ckpt",
                              batch_size=64):
        """SFR on GPU → embed every context → store in stage cache → SFR off.

        One embedding per sample, shape (1, retriever_hidden). The retriever
        tokenizer truncates at doc_max_length; nothing is chunked.
        """
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
                # one embedding per sample: the context is a single document,
                # truncated by the retriever tokenizer. No chunking.
                for start in range(0, len(samples), batch_size):
                    batch = samples[start:start + batch_size]
                    inp = self.retriever_tokenizer(
                        [x["context"] for x in batch],
                        max_length=self.doc_max_length, padding=True,
                        truncation=True, return_tensors="pt",
                    ).to(dev)
                    emb = self.retriever.get_doc_embedding(
                        input_ids=inp.input_ids, attention_mask=inp.attention_mask,
                    )  # (B, retriever_hidden)
                    for j in range(len(batch)):
                        embeddings[start + j] = emb[j:j + 1].cpu()  # (1, d)
                    progress.update(task, advance=len(batch))
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
        """Read the precomputed SFR embedding for the current sample.

        The retriever is never on GPU during collection (the LLM is), so there
        is no live fallback: a cache miss is a bug in the precompute step, not
        something to paper over by swapping 7B models in and out per sample.
        """
        cached = self.get_cache(self._current_sample_idx)
        if cached is None:
            raise KeyError(
                f"no precomputed embedding for sample {self._current_sample_idx} "
                f"in stage {self.stage!r} — run precompute_embeddings() for this "
                f"split before collection"
            )
        return cached.to(self.model.device)

    def generate_compressed(self, compressed_embs, query, **kw):
        dev = self.model.device
        prompt = RAG_TEMPLATE.format(document=XRAG_TOKEN, question=query)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
        out = self.model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=kw.get("max_new_tokens", 128),
            pad_token_id=self.tokenizer.pad_token_id,
            retrieval_embeds=self._as_retrieval_embeds([compressed_embs]),
        )
        # retrieval_embeds routes generate() through inputs_embeds, which
        # returns ONLY new tokens — no prompt to slice off (unlike
        # generate_full, which passes input_ids and gets prompt + completion).
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

    # ── batched paths (used when cfg.collect_batch_size > 1) ─────────

    def _left_pad_ids(self, prompts):
        """Tokenize a batch of prompts. The tokenizer is already configured
        padding_side='left', which is what a decoder-only model needs: real
        tokens are flush right, so the last one is at index -1 for every row."""
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                             add_special_tokens=False)
        dev = self.model.device
        return enc["input_ids"].to(dev), enc["attention_mask"].to(dev)

    @torch.no_grad()
    def compress_batch(self, docs, queries=None):
        """Per-sample cache lookup, walking the batch by index.

        collect_features sets ``_current_sample_idx`` to the first index of the
        batch, so sample j of this batch is global index start + j. Cheap:
        the SFR embeddings were precomputed, this is a dict read.
        """
        start = self._current_sample_idx or 0
        out = []
        try:
            for j, per_sample_docs in enumerate(docs):
                self._current_sample_idx = start + j
                q = queries[j] if queries is not None else None
                out.append(self.compress(per_sample_docs, query=q))
        finally:
            self._current_sample_idx = start
        return out

    @torch.no_grad()
    def generate_full_batch(self, contexts, queries, max_new_tokens=128, **kw):
        prompts = [RAG_TEMPLATE.format(document=c, question=q)
                   for c, q in zip(contexts, queries)]
        input_ids, attention_mask = self._left_pad_ids(prompts)
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.batch_decode(out[:, input_ids.shape[1]:],
                                           skip_special_tokens=True)

    @torch.no_grad()
    def generate_compressed_batch(self, embs_list, queries, max_new_tokens=128, **kw):
        """One xrag token per sample, so retrieval_embeds is (B, 1, d) and rows
        line up with the prompts. Bucketing isn't needed the way it is for
        PISCO: every sample contributes exactly one embedding.

        Output is new tokens only (the inputs_embeds path), so no prompt slice.
        """
        prompts = [RAG_TEMPLATE.format(document=XRAG_TOKEN, question=q) for q in queries]
        input_ids, attention_mask = self._left_pad_ids(prompts)
        embeds = self._as_retrieval_embeds(embs_list)
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            retrieval_embeds=embeds,
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    @torch.no_grad()
    def extract_clf_features_batch(self, contexts, queries, compressed_embs=None):
        """One batched forward; mid-layer hidden state at the last real token.
        hidden_states[0] is the embedding output, so layer L is index L+1 —
        matching the hook on layers[mid_layer_index]."""
        if compressed_embs is None:
            compressed_embs = self.compress_batch(
                [self._chunk_text(c) for c in contexts], queries=queries)

        prompts = [
            RAG_TEMPLATE.format(document=" ".join([XRAG_TOKEN] * e.shape[0]), question=q)
            for e, q in zip(compressed_embs, queries)
        ]
        input_ids, attention_mask = self._left_pad_ids(prompts)
        # row-major mask assignment in prepare_inputs_embeds means embeds must
        # be concatenated in sample order, which this is.
        embeds = self._as_retrieval_embeds(compressed_embs)
        # generate() derives position_ids from the mask, but a plain forward()
        # falls back to arange — left-padded rows would get RoPE offsets shifted
        # by their pad count, so pass true positions explicitly.
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            retrieval_embeds=embeds,
            output_hidden_states=True,
        )
        return out.hidden_states[self.mid_layer_index + 1][:, -1, :].float().cpu()

    def _as_retrieval_embeds(self, embs):
        """Flatten a list of per-sample embeddings to (n_xrag_tokens, d_retriever).

        prepare_inputs_embeds does `view(-1, retriever_hidden_size)` and then a
        row-major masked assignment, so the rows here must be in sample order
        and must number exactly as many XRAG tokens as the prompts contain.
        """
        d = embs[0].shape[-1]
        return torch.cat([e.reshape(-1, d) for e in embs], dim=0).to(self.model.device)

    def _chunk_text(self, text, chunk_tokens=None):
        """xRAG does not chunk: the context is one document, and the retriever
        tokenizer truncates it at doc_max_length."""
        return [text]

    def count_tokens_compressed(self, compressed_embs, query: str) -> int:
        """One XRAG placeholder token per embedding row — not d_retriever
        values. The base implementation would multiply by the hidden size."""
        return compressed_embs.reshape(-1, compressed_embs.shape[-1]).shape[0]

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
        dataset="./triviaqa/train.jsonl",
        eval_dataset="./triviaqa/test.jsonl",
        output_dir="./xrag_mistral_router_ckpt_triviaqa",
        epochs=60,
        n_folds=5,
        collect_batch_size=32,
        push_to_hub=False,
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
