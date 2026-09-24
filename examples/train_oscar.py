import logging
import torch
from jinja2.exceptions import TemplateError
from transformers import AutoModel
from overflowguard import OverflowRouter, TrainConfig, train_router
from openai import OpenAI
import os

logging.basicConfig(level=logging.INFO)



class StopForward(Exception):
    """Raised by the mid-layer hook once the feature is captured, so the
    decoder skips every layer above it (early exit)."""

class OscarRouter(OverflowRouter):

    MID_LAYER = 17

    def _load_model(self, path, **kwargs):
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True).eval()
        self.tokenizer = self.model.decoder_tokenizer
        self.early_exit = True  # stop the extraction forward at MID_LAYER
        self._attach_mid_hook()
        self._judge_client = None
    def _get_judge(self):
        if self._judge_client is None:
            self._judge_client = OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            )
        return self._judge_client

    def _attach_mid_hook(self):
        layers = self.model.decoder.model.layers
        mid = self.MID_LAYER

        def hook(module, inp, out):
            if not self._capture_active:
                return
            h = out[0] if isinstance(out, tuple) else out
            self._captured_hs["h"] = h.detach()
            self._capture_active = False
            if self.early_exit:
                raise StopForward  # skip the layers above the probed one

        self._hook_handle = layers[mid].register_forward_hook(hook)


    def compress(self, documents, query=None):
        # returns memory embeddings (n_chunks, n_mem, d_model)
        return self.model.compress_documents(questions=[query] * len(documents) if query else None,
                                             documents=documents)

    def generate_compressed(self, compressed_embs, query, **kw):
        return self.model.generate_from_compressed_documents_and_questions(questions=[query], compressed_documents=compressed_embs)[0]

    def generate_full(self, context, query, **kw):
        dec_ids, dec_mask = self._full_inputs(context, query)
        dev = self.model.decoder.device
        inputs_embeds = self.model.decoder.get_input_embeddings()(dec_ids.to(dev))
        if "decoder_adapter" in self.model.adapter_keys:
            self.model.decoder.set_adapter("decoder_adapter")
        generate_kwargs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': dec_mask.to(dev),
            'do_sample': False,
        }
        # User kwargs override the defaults
        generate_kwargs.update(kw)

        out = self.model.decoder.generate(**generate_kwargs)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    def extract_clf_features(self, compressed_embs, query):
        tok = self.tokenizer
        dev = self.model.decoder.device

        self.model.generation_top_k = compressed_embs.size(0)
        prompt = self.model.blend_prompt_and_memory_tokens(query=query)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)

        inputs_embeds = self.model.replace_emb(
            compressed_embs, ids["input_ids"].to(dev),
        )
        # compress_documents leaves the encoder adapter active; training
        # features were read with the decoder adapter (after generate_compressed)
        if "decoder_adapter" in self.model.adapter_keys:
            self.model.decoder.set_adapter("decoder_adapter")
        self._capture_active = True
        self._captured_hs.clear()
        try:
            self.model.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=ids["attention_mask"].to(dev),
            )
        except StopForward:
            pass

        return self._captured_hs["h"][0, -1, :].float()

    # ── batched paths (used when cfg.collect_batch_size > 1) ─────────

    def _left_pad(self, rows, pad_value, device):
        """Left-pad a list of 1-D id tensors / 2-D embed tensors into a batch.

        Decoder-only models mishandle right padding, so real tokens are flush
        right: the last real token is always at index -1 for every row.
        """
        B, maxlen = len(rows), max(r.size(0) for r in rows)
        if rows[0].dim() == 1:
            out = torch.full((B, maxlen), pad_value, dtype=torch.long, device=device)
        else:
            out = pad_value.expand(B, maxlen, rows[0].size(-1)).clone().to(rows[0].dtype)
        attn = torch.zeros(B, maxlen, dtype=torch.long, device=device)
        for i, r in enumerate(rows):
            out[i, maxlen - r.size(0):] = r.to(device)
            attn[i, maxlen - r.size(0):] = 1
        return out, attn

    @property
    def _pad_id(self):
        return self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    @torch.no_grad()
    def generate_full_batch(self, contexts, queries, max_new_tokens=256, **kw):
        dev = self.model.decoder.device
        rows = [self._full_inputs(c, q)[0][0] for c, q in zip(contexts, queries)]
        ids, attn = self._left_pad(rows, self._pad_id, dev)
        inputs_embeds = self.model.decoder.get_input_embeddings()(ids)
        if "decoder_adapter" in self.model.adapter_keys:
            self.model.decoder.set_adapter("decoder_adapter")
        generate_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attn,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
        }
        generate_kwargs.update(kw)
        out = self.model.decoder.generate(**generate_kwargs)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    @torch.no_grad()
    def compress_batch(self, chunked, queries=None):
        """One compress_documents call over all chunks of all samples.

        Chunks are compressed independently, so flattening the batch is exact;
        we just split the result back per sample. Query-aware models get one
        query per CHUNK, repeated across that sample's chunks.
        """
        flat = [c for chunks in chunked for c in chunks]
        qs = [q for q, chunks in zip(queries, chunked) for _ in chunks] if queries else None
        embs = self.model.compress_documents(documents=flat, questions=qs)
        out, i = [], 0
        for chunks in chunked:
            out.append(embs[i:i + len(chunks)])
            i += len(chunks)
        return out

    @torch.no_grad()
    def generate_compressed_batch(self, embs_list, queries, max_new_tokens=256, **kw):
        """PISCO derives generation_top_k = n_embs // n_questions, so every
        sample in one call must have the SAME chunk count. Bucket by chunk
        count rather than padding with empty docs (padding would add mem slots
        and change the answer vs. the single-sample path).
        """
        buckets = {}
        for i, e in enumerate(embs_list):
            buckets.setdefault(e.size(0), []).append(i)

        answers = [None] * len(embs_list)
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"  # decoder-only: prompts must left-pad
        try:
            for idxs in buckets.values():
                out = self.model.generate_from_compressed_documents_and_questions(
                    questions=[queries[i] for i in idxs],
                    compressed_documents=torch.cat([embs_list[i] for i in idxs], dim=0),
                    max_new_tokens=max_new_tokens,
                )
                for pos, i in enumerate(idxs):
                    answers[i] = out[pos]
        finally:
            self.tokenizer.padding_side = prev_side
        return answers

    @torch.no_grad()
    def _batched_decoder_forward(self, contexts, queries, compressed_embs=None):
        """One padded decoder forward over the batch, hidden states included.

        Shared by the mid-layer feature extractor and the layer sweep — they
        differ only in which hidden states they keep.
        """
        dev = self.model.decoder.device
        if compressed_embs is None:
            compressed_embs = self.compress_batch([self._chunk_text(c) for c in contexts])

        rows = []
        for emb, q in zip(compressed_embs, queries):
            self.model.generation_top_k = emb.size(0)
            prompt = self.model.blend_prompt_and_memory_tokens(query=q)
            ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            rows.append(self.model.replace_emb(emb, ids["input_ids"].to(dev))[0])


        d = rows[0].size(-1)
        pad_emb = self.model.decoder.get_input_embeddings()(
            torch.tensor([self._pad_id], device=dev)).view(1, 1, d)
        padded, attn = self._left_pad(rows, pad_emb, dev)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)  # RoPE-correct positions

        return self.model.decoder(
            inputs_embeds=padded, attention_mask=attn, position_ids=pos,
            output_hidden_states=True,
        )

    @torch.no_grad()
    def extract_clf_features_batch(self, contexts, queries, compressed_embs=None):
        """Mid-layer hidden state at the last real token, matching
        extract_clf_features."""
        out = self._batched_decoder_forward(contexts, queries, compressed_embs)
        # hidden_states[0] is the embedding layer, so layer L == index L+1
        return out.hidden_states[self.MID_LAYER + 1][:, -1, :].float().cpu()

    def _full_inputs(self, text, query):
        prompt_system = "You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible."
        prompt_user = f"Background:\n{text}\n\nQuestion:{query}"
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TemplateError as e:
            if "System role not supported" in str(e):
                messages = [{"role": "user", "content": messages[0]["content"] + "\n" + messages[1]["content"]}]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                raise
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return ids["input_ids"], ids["attention_mask"]


if __name__ == "__main__":
    router = OscarRouter.from_pretrained("naver/oscar-qwen2-7B")
    router.park_gpu()

    cfg = TrainConfig(
        dataset="./squad/train.jsonl",
        eval_dataset="./squad/test.jsonl",
        output_dir="./oscar_qwen-7b_router_ckpt_squad",
        epochs=60,
        n_folds=5,
        push_to_hub=False,
        collect_batch_size=32,
        skip_full_wrong=False,
        hub_repo_id="wexumin/oscar-7b-router",
    )
    from overflowguard import llm_judge

    result = train_router(router, cfg, evaluator=llm_judge(concurrency=30, model="deepseek-chat"))
