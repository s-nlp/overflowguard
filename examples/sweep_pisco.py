"""
Layer sweep for PISCO, reusing the labels (and judge verdicts) from a prior
training run so only the cheap forward pass is repeated.

Two changes vs train_pisco.py:
  1. extract_clf_features returns ALL layers stacked (n_layers, d).
  2. extract_clf_features_batch does the same for a batch — the padding and
     compression logic is inherited from PiscoRouter, only the layer
     selection differs.

Run after a normal train_pisco.py run has produced ./pisco_router_ckpt.
"""
import logging
import torch
from overflowguard import TrainConfig, train_router
from train_pisco import PiscoRouter

logging.basicConfig(level=logging.INFO)


class PiscoSweepRouter(PiscoRouter):

    # ── single-sample: all layers instead of just the mid hook ──
    def extract_clf_features(self, compressed_embs, query):
        dev = self.model.decoder.device
        self.model.generation_top_k = compressed_embs.size(0)
        prompt = self.model.blend_prompt_and_memory_tokens(query=query)
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs_embeds = self.model.replace_emb(compressed_embs, ids["input_ids"].to(dev))
        out = self.model.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=ids["attention_mask"].to(dev),
            output_hidden_states=True,
        )
        # (n_layers+1, d): row 0 = embeddings, row k = decoder layer k output
        return torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states], 0)

    # ── batched: same padded forward as the parent, all layers kept ──
    @torch.no_grad()
    def extract_clf_features_batch(self, contexts, queries, compressed_embs=None):
        out = self._batched_decoder_forward(contexts, queries, compressed_embs)
        # (B, n_layers+1, d) — layer dim second, as _run_sweep expects
        return torch.stack([hs[:, -1, :].float().cpu() for hs in out.hidden_states], dim=1)


if __name__ == "__main__":
    router = PiscoSweepRouter.from_pretrained("naver/pisco-mistral")
    router.park_gpu()

    cfg = TrainConfig(
        dataset="./squad/train.jsonl",
        eval_dataset="./squad/test.jsonl",
        output_dir="./pisco_router_sweep",
        sweep=True,
        collect_batch_size=32,               # batch the decoder forward
        epochs=60,
        n_folds=5,
    )
    from overflowguard import llm_judge

    result = train_router(router, cfg, evaluator=llm_judge(concurrency=30, model="deepseek-chat",  api_key="sk-b7c60a07a2a64dc6a1396eebea6d4e22"))  # evaluator unused in reuse mode
    print("best layer:", result["best_layer"])
