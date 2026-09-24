"""Shared helpers for pre-generation pre-generation uncertainty.

The selected input span is treated as a pseudo-generation sequence for the
canonical lm-polygraph Perplexity, Focus and RAUQ estimators:

  * token probabilities come from causal predictions of the observed input
    tokens;
  * attention comes from the same prompt-only forward pass;
  * only a contiguous textual span AFTER compressed memory is scored.

This is a pre-generation adaptation of Focus/RAUQ, not their original
output-generation setting.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Tuple, Iterable, List, Optional
import random

import numpy as np
import torch

import json
import math

from pathlib import Path


def as_1d_ids(ids: torch.Tensor) -> torch.Tensor:
    if ids.dim() == 2:
        if ids.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got ids={tuple(ids.shape)}")
        ids = ids[0]
    if ids.dim() != 1:
        raise ValueError(f"Expected 1-D token ids, got {tuple(ids.shape)}")
    return ids


def as_seq_embeds(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got embeds={tuple(x.shape)}")
        x = x[0]
    if x.dim() != 2:
        raise ValueError(f"Expected [seq, hidden] embeds, got {tuple(x.shape)}")
    return x


def last_true_position(mask: torch.Tensor) -> int:
    mask = mask.bool().flatten()
    pos = torch.nonzero(mask, as_tuple=False).flatten()
    if pos.numel() == 0:
        raise RuntimeError("No compressed-memory positions were detected in prompt")
    return int(pos[-1].item())


def post_memory_span(prompt_len: int, last_memory_pos: int) -> Tuple[int, int]:
    """Return [start, end) textual span after final memory position."""
    start = max(int(last_memory_pos) + 1, 1)  # token 0 cannot be scored causally
    end = int(prompt_len)
    if start >= end:
        raise RuntimeError(
            f"No scoreable input tokens after memory: start={start}, end={end}"
        )
    return start, end


def flatten_span_attentions(
    attentions: tuple,
    start: int,
    end: int,
) -> np.ndarray:
    """Return [layers*heads, span_len, span_len] for lm-polygraph."""
    per_layer = []
    for a in attentions:
        if a is None:
            continue
        # HF attention: [batch, heads, query_len, key_len]
        x = a[0, :, start:end, start:end].detach().float().cpu()
        per_layer.append(x)
    if not per_layer:
        raise RuntimeError(
            "Model returned no attention tensors. Use eager attention and "
            "output_attentions=True."
        )
    return torch.cat(per_layer, dim=0).numpy()


def token_statistics(
    logits: torch.Tensor,
    attentions: tuple,
    prompt_ids: torch.Tensor,
    start: int,
    end: int,
) -> Dict[str, Any]:
    """Build generation-shaped statistics for a contiguous observed input span.

    A prompt token at absolute position j is predicted by logits at j-1.
    Therefore tokens [start:end) use logits [start-1:end-1).
    """
    ids = as_1d_ids(prompt_ids).to(logits.device)
    start = max(int(start), 1)
    end = min(int(end), int(ids.numel()))
    if start >= end:
        raise ValueError(f"Empty pre-generation UE span [{start}, {end})")

    target_ids = ids[start:end]
    pred = logits[0, start - 1 : end - 1, :]
    if pred.shape[0] != target_ids.numel():
        raise RuntimeError(
            f"Causal alignment mismatch: pred={pred.shape[0]}, "
            f"targets={target_ids.numel()}"
        )

    log_probs = torch.log_softmax(pred.float(), dim=-1)
    token_ll = log_probs.gather(-1, target_ids[:, None]).squeeze(-1)
    attn = flatten_span_attentions(attentions, start, end)

    return {
        "greedy_log_probs": log_probs.detach().cpu().numpy(),
        "greedy_log_likelihoods": token_ll.detach().cpu().numpy(),
        "greedy_tokens": target_ids.detach().cpu().numpy(),
        "attention_all": attn,
        "start": start,
        "end": end,
    }


def make_polygraph_stats(
    one: Dict[str, Any],
    tokenizer,
    hf_model,
) -> Dict[str, Any]:
    tokens = one["greedy_tokens"]
    text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
    return {
        "greedy_log_probs": [one["greedy_log_probs"]],
        "greedy_log_likelihoods": [one["greedy_log_likelihoods"]],
        "greedy_tokens": [tokens],
        "greedy_texts": [text],
        "attention_all": [one["attention_all"]],
        "tokenizer": tokenizer,
        "model": SimpleNamespace(model=hf_model),
    }


def _scalar_or_none(estimator, stats, name: str):
    try:
        x = estimator(stats)
        value = float(np.asarray(x).reshape(-1)[0])
        if not np.isfinite(value):
            return None
        return value
    except Exception as exc:
        # A single Focus alignment edge case should not kill a multi-thousand
        # example collection run. Missing values are ignored by eval scripts.
        print(f"[UE] {name} failed for this sample: {type(exc).__name__}: {exc}")
        return None


def run_estimators(stats, ppl, focus, rauq) -> Dict[str, float | None]:
    """Apply canonical estimator implementations to observed pre-generation token stats."""
    return {
        "perplexity": _scalar_or_none(ppl, stats, "Perplexity"),
        "focus": _scalar_or_none(focus, stats, "Focus"),
        "rauq": _scalar_or_none(rauq, stats, "RAUQ"),
    }


def select_collection_rows(
    rows,
    max_samples: int | None = None,
    balance_overflow: bool = False,
    seed: int = 42,
):
    """Select collection rows while preserving original collection indices.

    Returns a list of ``(original_position, row)`` tuples.

    Default behavior is unchanged: take the first ``max_samples`` rows (or all
    rows when max_samples is None).

    With ``balance_overflow=True``:
      - ``comp_correct=False`` is treated as overflow;
      - ``comp_correct=True`` is treated as non-overflow;
      - rows with ``comp_correct=None`` are not eligible;
      - the sampler targets a 50/50 subset up to ``max_samples``;
      - if one class is too small, remaining slots are filled from the other
        class;
      - sampling is deterministic for a fixed seed.

    Balanced sampling is intended for train/debug subsets. Do not use it for a
    final test set when reporting prevalence-sensitive metrics such as accuracy,
    AUPR, compression rate, or token savings.
    """
    indexed = list(enumerate(rows))

    if not balance_overflow:
        if max_samples is None:
            return indexed
        return indexed[: min(len(indexed), int(max_samples))]

    if max_samples is None:
        raise ValueError(
            "--balance-overflow requires --max-samples so the target subset "
            "size is explicit."
        )

    overflow = []
    nonoverflow = []
    unlabeled = 0

    for pos, row in indexed:
        cc = row.get("comp_correct")
        if cc is None:
            unlabeled += 1
            continue
        if bool(cc):
            nonoverflow.append((pos, row))
        else:
            overflow.append((pos, row))

    if not overflow or not nonoverflow:
        raise RuntimeError(
            "Balanced overflow sampling requires both classes in collection.pt: "
            f"overflow={len(overflow)}, nonoverflow={len(nonoverflow)}, "
            f"unlabeled={unlabeled}."
        )

    rng = random.Random(int(seed))
    rng.shuffle(overflow)
    rng.shuffle(nonoverflow)

    target = min(int(max_samples), len(overflow) + len(nonoverflow))
    n_over = target // 2
    n_non = target - n_over

    take_over = min(n_over, len(overflow))
    take_non = min(n_non, len(nonoverflow))

    # Fill any shortage from the other class so we still approach max_samples.
    remaining = target - take_over - take_non
    if remaining > 0:
        over_spare = max(0, len(overflow) - take_over)
        add_over = min(remaining, over_spare)
        take_over += add_over
        remaining -= add_over

    if remaining > 0:
        non_spare = max(0, len(nonoverflow) - take_non)
        add_non = min(remaining, non_spare)
        take_non += add_non
        remaining -= add_non

    selected = overflow[:take_over] + nonoverflow[:take_non]
    rng.shuffle(selected)

    pct_overflow = 100.0 * take_over / max(len(selected), 1)
    print(
        "[UE sampling] "
        f"selected={len(selected)}/{len(rows)} "
        f"overflow={take_over} nonoverflow={take_non} "
        f"overflow_rate={pct_overflow:.1f}% "
        f"unlabeled_excluded={unlabeled} seed={seed}"
    )

    return selected


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(path: str | Path) -> set[int]:
    path = Path(path)
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if "id" in row:
                    done.add(int(row["id"]))
    return done


def load_collection(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def flatten_attentions_for_polygraph(
    attentions: tuple,
    start: int,
    length: int,
) -> np.ndarray:
    """
    Convert HF attentions to lm-polygraph's per-sample attention_all layout:
      (n_layers * n_heads, generated_len, generated_len)

    We slice answer-token query/key positions from a teacher-forced causal
    forward. This is the natural compressed-path analogue of generation-time
    token-to-token attention.
    """
    per_layer = []
    end = start + length
    for a in attentions:
        # a: (B, H, T, T)
        x = a[0, :, start:end, start:end].detach().float().cpu()
        per_layer.append(x)
    if not per_layer:
        raise RuntimeError(
            "Model returned no attentions. Ensure output_attentions=True is supported "
            "and, if needed, set attn_implementation='eager'."
        )
    return torch.cat(per_layer, dim=0).numpy()


def answer_statistics(
    logits: torch.Tensor,
    attentions: tuple,
    answer_ids: torch.Tensor,
    prompt_len: int,
) -> Dict[str, Any]:
    """
    Build the core lm-polygraph statistics for one cached answer.

    logits: (1, prompt_len + answer_len, vocab)
    answer_ids: (answer_len,)
    """
    answer_ids = answer_ids.to(logits.device)
    A = int(answer_ids.numel())
    if A == 0:
        raise ValueError("Cached compressed answer tokenized to zero tokens")

    # Position prompt_len-1 predicts answer token 0, etc.
    pred = logits[0, prompt_len - 1 : prompt_len - 1 + A, :]
    log_probs = torch.log_softmax(pred.float(), dim=-1)
    token_ll = log_probs.gather(-1, answer_ids[:, None]).squeeze(-1)

    attn = flatten_attentions_for_polygraph(attentions, prompt_len, A)

    return {
        "greedy_log_probs": log_probs.detach().cpu().numpy(),
        "greedy_log_likelihoods": token_ll.detach().cpu().numpy(),
        "greedy_tokens": answer_ids.detach().cpu().numpy(),
        "attention_all": attn,
        "answer_logits": pred.detach().float().cpu().numpy(),
    }


def make_polygraph_stats(
    one: Dict[str, Any],
    tokenizer,
    answer_text: str,
    hf_model,
) -> Dict[str, Any]:
    """
    lm-polygraph estimators expect batched/list statistics.
    """
    return {
        "greedy_log_probs": [one["greedy_log_probs"]],
        "greedy_log_likelihoods": [one["greedy_log_likelihoods"]],
        "greedy_tokens": [one["greedy_tokens"]],
        "greedy_texts": [answer_text],
        "attention_all": [one["attention_all"]],
    
        # actual tokenizer object
        "tokenizer": tokenizer,
    
        "model": SimpleNamespace(model=hf_model),
    }


def logit_magnitude_token_scores(
    logits: np.ndarray,
    top_k: int = 20,
) -> np.ndarray:
    """
    MetaUE paper-style positive Logit Magnitude teacher signal.

    For each generation position:
      - take top-K raw logits
      - keep positive evidence only
      - sum magnitudes

    The paper then aggregates the top-M token scores.
    """
    x = torch.as_tensor(logits, dtype=torch.float32)
    k = min(int(top_k), x.shape[-1])
    top = torch.topk(x, k=k, dim=-1).values
    return torch.relu(top).sum(dim=-1).cpu().numpy()


def adaptive_topm_logit_magnitude(
    logits: np.ndarray,
    top_k: int = 20,
    top_m: int = 5,
    patience: int = 5,
) -> tuple[float, int]:
    """
    Cached-answer analogue of MetaUE's adaptive Logit Magnitude teacher.

    We walk token scores in answer order and stop when the identity of the
    current top-M token positions has not changed for `patience` steps.
    Returns (score, number_of_answer_tokens_observed).

    This uses the already cached deterministic response, so it avoids another
    autoregressive generation pass while preserving the label-free teacher.
    """
    scores = logit_magnitude_token_scores(logits, top_k=top_k)
    if len(scores) == 0:
        return float("nan"), 0

    last_top = None
    stable = 0
    stop = len(scores)

    for t in range(1, len(scores) + 1):
        m = min(top_m, t)
        idx = tuple(sorted(np.argpartition(scores[:t], -m)[-m:].tolist()))
        if idx == last_top:
            stable += 1
        else:
            stable = 0
            last_top = idx
        if stable >= patience:
            stop = t
            break

    m = min(top_m, stop)
    chosen = np.argpartition(scores[:stop], -m)[-m:]
    return float(np.mean(scores[chosen])), int(stop)


def build_estimators(
    *,
    tokenizer,
    focus_tokenizer_name: str,
    focus_idf_path: str,
    focus_idf_dataset: str = "krisbailey/RedPajama-Data-V2-100M",
    focus_text_column: str = "raw_content",
    focus_gamma: float = 0.9,
    focus_p: float = 0.01,
    focus_idf_seed: int = 42,
    focus_idf_dataset_size: int = -1,
    spacy_model: str = "en_core_web_sm",
    rauq_alpha: float = 0.2,
):
    """
    Instantiate the official lm-polygraph implementations.

    Defaults mirror lm-polygraph's current default estimator config for
    Focus and the non-entropy RAUQ variant.
    """
    try:
        from lm_polygraph.estimators import Perplexity, Focus, RAUQ
    except Exception as e:
        raise RuntimeError(
            "lm-polygraph is required. Install with e.g. "
            "`pip install git+https://github.com/IINemo/lm-polygraph.git`"
        ) from e
        
    ppl = Perplexity()
    focus = Focus(
        gamma=focus_gamma,
        p=focus_p,
    
        # IMPORTANT:
        # this must be the underlying decoder tokenizer,
        # NOT naver/pisco-llama
        model_name=focus_tokenizer_name,
    
        path=focus_idf_path,
        idf_dataset=focus_idf_dataset,
        trust_remote_code=True,
        idf_seed=focus_idf_seed,
        idf_dataset_size=focus_idf_dataset_size,
        spacy_path=spacy_model,
        idf_dataset_text_column=focus_text_column,
    )
    
    rauq = RAUQ(
        alpha=rauq_alpha,
        use_entropy=False,
    )
    
    return ppl, focus, rauq


def run_estimators(
    poly_stats: Dict[str, Any],
    ppl,
    focus,
    rauq,
) -> Dict[str, float]:
    def scalar(est):
        x = est(poly_stats)
        return float(np.asarray(x).reshape(-1)[0])

    return {
        "perplexity": scalar(ppl),
        "focus": scalar(focus),
        "rauq": scalar(rauq),
    }
