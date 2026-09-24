"""Router subclasses for the demo app."""

import gc
import os
import sys

# Make the sibling `overflowguard` package importable (demo/ lives beside it).
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import torch
from transformers import AutoModel, AutoTokenizer
from overflowguard import OverflowRouter



class StopForward(Exception):
    """Raised by the mid-layer hook once the feature is captured, so the
    decoder skips every layer above it (early exit)."""

class PiscoRouter(OverflowRouter):

    def _load_model(self, path, **kwargs):
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True).eval()
        self.tokenizer = self.model.decoder_tokenizer
        self.early_exit = True  # stop the extraction forward at MID_LAYER
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
            if self.early_exit:
                raise StopForward  # skip the layers above the probed one

        self._hook_handle = layers[mid].register_forward_hook(hook)

    def compress(self, documents, query=None):
        return self.model.compress_documents(documents=documents)

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

    def _full_inputs(self, text, query):
        tok = self.tokenizer
        prompt_system = "You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible."
        prompt_user = f"Background:\n{text}\n\nQuestion:{query}"
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ]
        from jinja2.exceptions import TemplateError
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
        if self.clf is not None:
            self.clf.to("cuda")

    def unpark_gpu(self):
        self.model.cpu()
        if self.clf is not None:
            self.clf.to("cpu")


class OscarRouter(PiscoRouter):
    """OSCAR uses query-aware compression, otherwise same as PISCO."""

    def compress(self, documents, query=None):
        return self.model.compress_documents(
            documents=documents,
            questions=[query] * len(documents) if query else None,
        )


class XragRouter(OverflowRouter):

    def _load_model(self, path, **kwargs):
        import sys
        sys.path.insert(0, "/workspace/xRAG")
        from src.model import SFR, XMistralForCausalLM
        from src.language_modeling.utils import XRAG_TOKEN

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
        self._xrag_token = XRAG_TOKEN
        self._rag_template = "[INST] Refer to the background document and answer the questions:\n\nBackground: {document}\n\nQuestion: {question} [/INST] The answer is:"

    def compress(self, documents, query=None):
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
        return embeddings

    def generate_compressed(self, compressed_embs, query, **kw):
        dev = self.model.device
        prompt = self._rag_template.format(document=self._xrag_token, question=query)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
        out = self.model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=kw.get("max_new_tokens", 128),
            pad_token_id=self.tokenizer.pad_token_id,
            retrieval_embeds=compressed_embs[:1],
        )
        return self.tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    def generate_full(self, context, query, **kw):
        dev = self.model.device
        prompt = self._rag_template.format(document=context, question=query)
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
        prompt = self._rag_template.format(
            document=" ".join([self._xrag_token] * compressed_embs.shape[0]),
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
                retrieval_embeds=compressed_embs.unsqueeze(0),
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


ROUTER_CLASSES = {
    "PISCO": PiscoRouter,
    "OSCAR": OscarRouter,
    "xRAG": XragRouter,
}
