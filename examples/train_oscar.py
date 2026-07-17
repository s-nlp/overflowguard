"""Train routing CLF for OSCAR."""

import logging
from transformers import AutoModel
from compression_router import CompressRouterModule, TrainConfig, train_router, llm_judge
from jinja2.exceptions import TemplateError
import torch
import os

logging.basicConfig(level=logging.INFO)


class OscarRouter(CompressRouterModule):

    def _load_model(self, path, **kwargs):
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True).eval()
        self.tokenizer = self.model.decoder_tokenizer
        self._attach_mid_hook()

    def _attach_mid_hook(self):
        layers = self.model.decoder.model.layers
        mid = 17

        def hook(module, inp, out):
            if not self._capture_active:
                return
            h = out[0] if isinstance(out, tuple) else out
            self._captured_hs["h"] = h.detach()
            self._capture_active = False

        self._hook_handle = layers[mid].register_forward_hook(hook)

    def compress(self, documents, query=None):
        return self.model.compress_documents(
            documents=documents,
            questions=[query] * len(documents) if query else None,
        )

    def generate_compressed(self, compressed_embs, query, **kw):
        return self.model.generate_from_compressed_documents_and_questions(
            questions=[query], compressed_documents=compressed_embs,
        )[0]

    def generate_full(self, context, query, **kw):
        dec_ids, dec_mask = self._full_inputs(context, query)
        dev = self.model.decoder.device
        inputs_embeds = self.model.decoder.get_input_embeddings()(dec_ids.to(dev))
        if "decoder_adapter" in self.model.adapter_keys:
            self.model.decoder.set_adapter("decoder_adapter")
        out = self.model.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_mask.to(dev),
            do_sample=False,
            **kw,
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    def extract_clf_features(self, compressed_embs=None, query=None):

        tok = self.tokenizer
        dev = self.model.decoder.device
        self.model.generation_top_k = compressed_embs.size(0)
        prompt = self.model.blend_prompt_and_memory_tokens(query=query)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
        inputs_embeds = self.model.replace_emb(
                compressed_embs, ids["input_ids"].to(dev),
            )
        self._capture_active = True
        self._captured_hs.clear()
        _ = self.model.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=ids["attention_mask"].to(dev),
            )
        return self._captured_hs["h"][0, -1, :].float()

    def _full_inputs(self, text, query):
        tok = self.tokenizer
        prompt_system = "You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible."
        prompt_user = f"Background:\n{text}\n\nQuestion:{query}"
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ]
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TemplateError as e:
            if "System role not supported" in str(e):
                messages = [{"role": "user", "content": messages[0]["content"] + "\n" + messages[1]["content"]}]
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                raise
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
        return ids["input_ids"], ids["attention_mask"]

    def _chunk_text(self, text, chunk_tokens=None):
        chunk_tokens = chunk_tokens or self.model.doc_max_length
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return [
            self.tokenizer.decode(ids[i : i + chunk_tokens], skip_special_tokens=True)
            for i in range(0, len(ids), chunk_tokens)
        ] or [text]

    def park_gpu(self):
        self.model.cuda()

    def unpark_gpu(self):
        self.model.cpu()


router = OscarRouter.from_pretrained("naver/oscar-mistral-7B")
router.park_gpu()

cfg = TrainConfig(
    dataset="/data/train_squad.jsonl",
    eval_dataset="/data/test_squad.jsonl",
    output_dir="./oscar_router_ckpt",
    epochs=100,
    n_folds=5,
    push_to_hub=True,
    hub_repo_id="wexumin/oscar-7b-router",
)

result = train_router(router, cfg, evaluator=llm_judge(concurrency=30, model="deepseek-chat"))
