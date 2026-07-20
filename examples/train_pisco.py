import logging
from transformers import AutoModel
from overflowguard import OverflowRouter, TrainConfig, train_router
from openai import OpenAI
import os

logging.basicConfig(level=logging.INFO)


class PiscoRouter(OverflowRouter):

    def _load_model(self, path, **kwargs):
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True).eval()
        self.tokenizer = self.model.decoder_tokenizer
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
        mid = 17

        def hook(module, inp, out):
            if not self._capture_active:
                return
            h = out[0] if isinstance(out, tuple) else out
            self._captured_hs["h"] = h.detach()
            self._capture_active = False

        self._hook_handle = layers[mid].register_forward_hook(hook)


    def compress(self, documents, query=None):
        # returns memory embeddings (n_chunks, n_mem, d_model)
        return self.model.compress_documents(documents=documents)

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
        self._capture_active = True
        self._captured_hs.clear()
        _ = self.model.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=ids["attention_mask"].to(dev),
        )

        return self._captured_hs["h"][0, -1, :].float()

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
    # load bare model (no router_config.json → just loads the model)
    router = PiscoRouter.from_pretrained("naver/pisco-mistral")
    router.park_gpu()

    cfg = TrainConfig(
        dataset="/data/train_squad.jsonl",
        eval_dataset="/data/test_squad.jsonl", # {"context": ..., "query": ..., "gold": ...}
        output_dir="./pisco_router_ckpt",
        epochs=100,
        n_folds=5,
        push_to_hub=False,
        hub_repo_id="wexumin/pisco-7b-router",
    )


    # result = {"threshold": 0.62, "auc": 0.87, "accuracy": 0.91, ...}
    # default — EM/F1, no API calls

    # LLM judge — async, concurrent, with progress bar
    from overflowguard import llm_judge


    # custom settings
    result = train_router(router, cfg, evaluator=llm_judge(concurrency=30, model="deepseek-chat"))
