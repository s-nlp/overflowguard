#!/usr/bin/env python3
"""Collect PRE-GENERATION uncertainty on the OSCAR compressed path.

Perplexity, Focus and RAUQ are all evaluated on observed textual prompt tokens
that occur after the final compressed-memory position. They are computed from
ONE prompt-only decoder forward pass, before any answer tokens are appended.

Focus/RAUQ reuse the canonical lm-polygraph estimators as explicit
pre-generation adaptations.

The cached comp_answer is optional for these three scores and is used only for
"""

from __future__ import annotations

import argparse
import gc
import importlib.util

import torch
from tqdm import tqdm
from transformers import AutoModel

from ue_common import (
    as_seq_embeds,
    token_statistics,
    last_true_position,
    make_polygraph_stats,
    post_memory_span,
    run_estimators,
    select_collection_rows,
    append_jsonl,
    build_estimators,
    load_collection,
    load_done_ids,
)


def import_router(script_path):
    spec = importlib.util.spec_from_file_location("og_train_oscar", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OscarRouter


def patch_oscar_loader_for_eager_attention(OscarRouter):
    """The provided train_oscar.py ignores **kwargs in _load_model."""

    def _load_model_eager(self, path, **kwargs):
        load_kwargs = dict(kwargs)
        load_kwargs["attn_implementation"] = "eager"
        try:
            self.model = AutoModel.from_pretrained(
                path, trust_remote_code=True, **load_kwargs
            ).eval()
        except TypeError:
            load_kwargs.pop("attn_implementation", None)
            self.model = AutoModel.from_pretrained(
                path, trust_remote_code=True, **load_kwargs
            ).eval()

        self.tokenizer = self.model.decoder_tokenizer
        decoder = self.model.decoder
        if hasattr(decoder, "config"):
            try:
                decoder.config._attn_implementation = "eager"
            except Exception:
                pass
            try:
                decoder.config.attn_implementation = "eager"
            except Exception:
                pass
        if hasattr(self.model, "config"):
            try:
                self.model.config._attn_implementation = "eager"
            except Exception:
                pass
            try:
                self.model.config.attn_implementation = "eager"
            except Exception:
                pass

        self._attach_mid_hook()

    OscarRouter._load_model = _load_model_eager
    return OscarRouter


def decoder_device(decoder):
    return decoder.get_input_embeddings().weight.device


def set_decoder_adapter(router):
    if "decoder_adapter" in getattr(router.model, "adapter_keys", []):
        router.model.decoder.set_adapter("decoder_adapter")


def detect_memory_positions(decoder, prompt_ids, prompt_embeds, atol=1e-6):
    prompt_embeds = as_seq_embeds(prompt_embeds)
    ids_1d = prompt_ids[0] if prompt_ids.dim() == 2 else prompt_ids
    with torch.inference_mode():
        base = decoder.get_input_embeddings()(ids_1d.to(prompt_embeds.device))
    if base.shape != prompt_embeds.shape:
        raise RuntimeError(
            f"OSCAR prompt shape changed after replace_emb: base={tuple(base.shape)}, "
            f"replaced={tuple(prompt_embeds.shape)}"
        )
    return (prompt_embeds - base).abs().amax(dim=-1) > atol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--router-script", required=True)
    ap.add_argument("--model", default="naver/oscar-mistral-7B")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument(
        "--balance-overflow", action="store_true",
        help="Stratify the selected subset by comp_correct (about 50/50 overflow/non-overflow). Requires --max-samples.",
    )
    ap.add_argument("--sampling-seed", type=int, default=42)

    ap.add_argument("--focus-idf-path", required=True)
    ap.add_argument("--focus-tokenizer-name", default=None)
    ap.add_argument(
        "--focus-idf-dataset", default="krisbailey/RedPajama-Data-V2-100M"
    )
    ap.add_argument("--focus-text-column", default="raw_content")
    ap.add_argument("--focus-idf-size", type=int, default=-1)
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--focus-gamma", type=float, default=0.9)
    ap.add_argument("--focus-p", type=float, default=0.01)
    ap.add_argument("--rauq-alpha", type=float, default=0.2)

    ap.add_argument("--memory-diff-atol", type=float, default=1e-6)
    args = ap.parse_args()

    OscarRouter = patch_oscar_loader_for_eager_attention(
        import_router(args.router_script)
    )
    router = OscarRouter.from_pretrained(args.model, attn_implementation="eager")
    router.park_gpu()
    router.model.eval()

    tok = router.tokenizer
    decoder = router.model.decoder
    dev = decoder_device(decoder)

    focus_tokenizer_name = args.focus_tokenizer_name or tok.name_or_path
    print(f"[UE] OSCAR model: {args.model}")
    print(f"[UE] tokenizer: {type(tok).__name__} ({tok.name_or_path})")
    print(f"[UE] Focus IDF tokenizer: {focus_tokenizer_name}")
    print(
        "[UE] decoder attention:",
        getattr(
            decoder.config,
            "_attn_implementation",
            getattr(decoder.config, "attn_implementation", None),
        ),
    )

    ppl, focus, rauq = build_estimators(
        tokenizer=tok,
        focus_tokenizer_name=focus_tokenizer_name,
        focus_idf_path=args.focus_idf_path,
        focus_idf_dataset=args.focus_idf_dataset,
        focus_text_column=args.focus_text_column,
        focus_gamma=args.focus_gamma,
        focus_p=args.focus_p,
        focus_idf_dataset_size=args.focus_idf_size,
        spacy_model=args.spacy_model,
        rauq_alpha=args.rauq_alpha,
    )

    rows = load_collection(args.collection)["results"]
    selected_rows = select_collection_rows(
        rows,
        max_samples=args.max_samples,
        balance_overflow=args.balance_overflow,
        seed=args.sampling_seed,
    )
    done = load_done_ids(args.output)

    for pos, r in tqdm(selected_rows, desc="OSCAR UE"):

        sid = int(r.get("id", pos))
        if sid in done:
            continue

        query = str(r.get("query", ""))
        docs = router._chunk_text(r["context"])
        with torch.inference_mode():
            compressed = router.compress(docs, query=query)

        router.model.generation_top_k = compressed.size(0)
        prompt = router.model.blend_prompt_and_memory_tokens(query=query)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_ids = ids["input_ids"].to(dev)

        with torch.inference_mode():
            replaced = router.model.replace_emb(compressed, prompt_ids)
        prompt_embeds = as_seq_embeds(replaced)
        set_decoder_adapter(router)

        memory_positions = detect_memory_positions(
            decoder, prompt_ids, prompt_embeds, atol=args.memory_diff_atol
        )
        last_mem = last_true_position(memory_positions)
        start, end = post_memory_span(prompt_ids.shape[1], last_mem)

        prompt_mask = torch.ones(
            (1, prompt_embeds.shape[0]), dtype=torch.long, device=dev
        )
        with torch.inference_mode():
            prompt_out = decoder(
                inputs_embeds=prompt_embeds.unsqueeze(0),
                attention_mask=prompt_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        if prompt_out.attentions is None:
            raise RuntimeError(
                "OSCAR returned no attentions. Ensure eager attention is active."
            )

        one = token_statistics(
            prompt_out.logits, prompt_out.attentions, prompt_ids, start, end
        )
        poly_stats = make_polygraph_stats(one, tok, decoder)
        scores = run_estimators(poly_stats, ppl, focus, rauq)


        append_jsonl(
            args.output,
            {
                "id": sid,
                "query": r.get("query"),
                "perplexity": scores["perplexity"],
                "focus": scores["focus"],
                "rauq": scores["rauq"],
                "num_tokens": int(end - start),
            },
        )

        del prompt_out, compressed, replaced, prompt_embeds
        if torch.cuda.is_available() and (pos + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
