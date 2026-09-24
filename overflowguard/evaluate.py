"""
Evaluation functions for routing correctness.

``default_evaluate`` — EM-or-F1, fast, no API calls.
``llm_judge`` — async LLM judge with concurrency + progress bar.

Pass either to ``train_router(evaluator=...)`` or ``collect_features(evaluator=...)``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import string
from collections import Counter

log = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _golds(gold) -> list[str]:
    """Gold may be a single answer or a list of accepted answers."""
    return [str(g) for g in gold] if isinstance(gold, (list, tuple)) else [str(gold)]


def em_score(prediction: str, gold: str | list[str]) -> bool:
    return any(_normalize(prediction) == _normalize(g) for g in _golds(gold))


def token_f1(prediction: str, gold: str | list[str]) -> float:
    """Best token F1 against any accepted gold answer."""
    return max(_token_f1_single(prediction, g) for g in _golds(gold))


def _token_f1_single(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def em_or_f1(prediction: str, gold: str | list[str], f1_threshold: float = 0.5) -> bool:
    return em_score(prediction, gold) or token_f1(prediction, gold) >= f1_threshold


def default_evaluate(results: list[dict]) -> None:
    """Score all results in-place using EM-or-F1 against any valid gold answer."""
    for r in results:
        golds = r["gold"] if isinstance(r["gold"], list) else [r["gold"]]
        golds = [str(g) for g in golds]

        if r.get("full_correct") is None:
            r["full_correct"] = any(
                em_or_f1(str(r["full_answer"]), gold)
                for gold in golds
            )

        if r.get("comp_correct") is None:
            r["comp_correct"] = any(
                em_or_f1(str(r["comp_answer"]), gold)
                for gold in golds
            )


# ── LLM Judge ──────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = """You are an evaluation judge.

Your task is to determine whether the MODEL PREDICTION correctly answers the QUESTION, judged against the GROUND-TRUTH answers and grounded in the BACKGROUND.

Guidelines:
- Focus on semantic meaning, not wording.
- Accept paraphrases.
- Ignore formatting, punctuation, or capitalization differences.
- Partially correct, contradictory, or hedged answers are INCORRECT.
- Additional text that contradicts or goes beyond the background is INCORRECT.
- If the prediction abstains or says it cannot determine the answer, mark as INCORRECT.
- Do NOT use outside knowledge — only the background and ground truth matter.
- Compare the prediction against each ground-truth answer individually — it only needs to match ONE to be CORRECT.

Return ONLY valid JSON:

```json
{"correct": 1}
```

or

```json
{"correct": 0}
```
"""

_JUDGE_USER_TEMPLATE = (
    "Background: {background}\n"
    "Question: {question}\n"
    "Ground truth: {answers}\n"
    "Prediction: {pred}"
)


def llm_judge(
    api_key: str | None = None,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    concurrency: int = 20,
    env_var: str = "DEEPSEEK_API_KEY",
) -> callable:
    """Return an evaluator function that uses an LLM judge.

    Usage::

        from overflowguard.evaluate import llm_judge
        train_router(model, cfg, evaluator=llm_judge())
        # or with custom settings:
        train_router(model, cfg, evaluator=llm_judge(concurrency=30, model="deepseek-chat"))
    """
    key = api_key or os.environ.get(env_var)
    if not key:
        raise ValueError(f"Provide api_key or set {env_var}")
    # the OpenAI client logs every request through httpx at INFO; quiet it only
    # when the judge is actually used, not for anyone importing overflowguard
    logging.getLogger("httpx").setLevel(logging.WARNING)

    def evaluate(results: list[dict]) -> None:
        # each verdict is judged independently: a row may be missing only one
        # of them, and verdicts that already exist must not be overwritten
        jobs = [
            (r, pred_field, correct_field)
            for r in results
            for pred_field, correct_field in _JUDGED_FIELDS
            if r.get(correct_field) is None
        ]
        if not jobs:
            return
        asyncio.run(_judge_batch(jobs, key, base_url, model, concurrency))

    return evaluate


_JUDGED_FIELDS = (("full_answer", "full_correct"), ("comp_answer", "comp_correct"))


async def _judge_batch(jobs, api_key, base_url, model, concurrency):
    """Judge each (row, pred_field, correct_field) job, setting the verdict in place.

    A job that still fails after all retries leaves its verdict None, so the
    training pipeline refuses to train on it and a rerun judges it again.
    """
    from openai import AsyncOpenAI
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(concurrency)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    task = progress.add_task("LLM judge", total=len(jobs))
    progress.start()

    async def judge_one(r, pred_field, correct_field):
        answers = r["gold"] if isinstance(r["gold"], list) else [r["gold"]]
        msg = _JUDGE_USER_TEMPLATE.format(
            background=r["context"],
            question=r["query"],
            answers=" | ".join(str(a) for a in answers),
            pred=r[pred_field],
        )
        for attempt in range(5):
            async with sem:
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        max_tokens=64,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": msg},
                        ],
                    )
                    text = resp.choices[0].message.content
                    obj = json.loads(text.strip("`").removeprefix("json").strip())
                    r[correct_field] = obj.get("correct") == 1
                    progress.update(task, advance=1)
                    return
                except Exception as e:
                    if attempt < 4:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        log.warning("Judge error id=%s field=%s: %s (left unscored)",
                                    r.get("id"), pred_field, e)
                        progress.update(task, advance=1)

    try:
        await asyncio.gather(*(judge_one(*job) for job in jobs))
    finally:
        progress.stop()
        await client.close()
