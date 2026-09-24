#!/usr/bin/env python3
"""Collect PRE-GENERATION uncertainty on the xRAG compressed path.

Perplexity, Focus and RAUQ are all evaluated on observed textual prompt tokens
that occur after the final XRAG retrieval placeholder. They are computed from
ONE prompt-only forward pass with retrieval_embeds, before any answer tokens
are appended.

Focus/RAUQ therefore reuse the canonical lm-polygraph estimators as explicit
pre-generation adaptations.

The cached comp_answer is optional for these three scores. It is used only for
"""

from __future__ import annotations

import argparse
import gc
import sys

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from ue_common import (
    token_statistics,
    make_polygraph_stats,
    post_memory_span,
    run_estimators,
    select_collection_rows,
    append_jsonl,
    build_estimators,
    load_collection,
    load_done_ids,
)


RAG_TEMPLATE = (
    "[INST] Refer to the background document and answer the questions:\n\n"
    "Background: {document}\n\nQuestion: {question} [/INST] The answer is:"
)


def input_device(model):
    return model.get_input_embeddings().weight.device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--emb-cache", required=True)
    ap.add_argument("--model", default="Hannibal046/xrag-moe")
    ap.add_argument("--xrag-repo", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device-map", default="auto")
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

    args = ap.parse_args()
    
    sys.path.insert(0, "/app/xRAG")
    from src.model import SFR, XMixtralForCausalLM
    from src.language_modeling.utils import XRAG_TOKEN

    rows = load_collection(args.collection)["results"]
    emb_obj = torch.load(args.emb_cache, map_location="cpu", weights_only=False)
    embeddings = emb_obj["embeddings"]

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    load_kwargs = dict(
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
    )
    try:
        model = XMixtralForCausalLM.from_pretrained(
            args.model, attn_implementation="eager", **load_kwargs
        ).eval()
    except TypeError:
        model = XMixtralForCausalLM.from_pretrained(args.model, **load_kwargs).eval()
        try:
            model.config._attn_implementation = "eager"
        except Exception:
            pass

    tok = AutoTokenizer.from_pretrained(
        args.model,
        add_eos_token=False,
        use_fast=False,
        padding_side="left",
    )
    
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    xrag_token_id = tok.convert_tokens_to_ids(XRAG_TOKEN)
    if xrag_token_id is None or xrag_token_id == tok.unk_token_id:
        raise RuntimeError(f"Could not resolve XRAG token {XRAG_TOKEN!r}")
    model.set_xrag_token_id(xrag_token_id)

    focus_tokenizer_name = args.focus_tokenizer_name or tok.name_or_path
    print(f"[UE] xRAG model: {args.model}")
    print(f"[UE] tokenizer: {type(tok).__name__} ({tok.name_or_path})")
    print(f"[UE] Focus IDF tokenizer: {focus_tokenizer_name}")

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

    selected_rows = select_collection_rows(
        rows,
        max_samples=args.max_samples,
        balance_overflow=args.balance_overflow,
        seed=args.sampling_seed,
    )
    done = load_done_ids(args.output)
    dev = input_device(model)

    for pos, r in tqdm(selected_rows, desc="xRAG UE"):

        sid = int(r.get("id", pos))
        if sid in done:
            continue
        if sid not in embeddings:
            raise KeyError(f"Embedding id {sid} missing from {args.emb_cache}")

        prompt = RAG_TEMPLATE.format(document=XRAG_TOKEN, question=r["query"])
        prompt_ids = tok(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids[0].to(dev)

        retrieval = embeddings[sid]
        retrieval = retrieval.reshape(-1, retrieval.shape[-1]).to(dev, dtype=dtype)

        xrag_positions = torch.nonzero(
            prompt_ids == int(xrag_token_id), as_tuple=False
        ).flatten()
        if xrag_positions.numel() == 0:
            raise RuntimeError(
                f"[id={sid}] XRAG placeholder absent from tokenized prompt"
            )
        last_mem = int(xrag_positions[-1].item())
        start, end = post_memory_span(prompt_ids.numel(), last_mem)

        prompt_batch = prompt_ids.unsqueeze(0)
        prompt_mask = torch.ones_like(prompt_batch)
        with torch.inference_mode():
            prompt_out = model(
                input_ids=prompt_batch,
                attention_mask=prompt_mask,
                retrieval_embeds=retrieval,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        if prompt_out.attentions is None:
            raise RuntimeError(
                "xRAG returned no attentions. Ensure eager attention is active."
            )

        one = token_statistics(
            prompt_out.logits, prompt_out.attentions, prompt_ids, start, end
        )
        poly_stats = make_polygraph_stats(one, tok, model)
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

        del prompt_out
        if torch.cuda.is_available() and (pos + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
