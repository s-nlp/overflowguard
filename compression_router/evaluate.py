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
logging.getLogger("httpx").setLevel(logging.WARNING)


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def em_score(prediction: str, gold: str) -> bool:
    return _normalize(prediction) == _normalize(gold)


def token_f1(prediction: str, gold: str) -> float:
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


def em_or_f1(prediction: str, gold: str, f1_threshold: float = 0.5) -> bool:
    return em_score(prediction, gold) or token_f1(prediction, gold) >= f1_threshold


def default_evaluate(results: list[dict]) -> None:
    """Score all results in-place using EM-or-F1."""
    for r in results:
        if r.get("full_correct") is None:
            r["full_correct"] = em_or_f1(r["full_answer"], r["gold"])
        if r.get("comp_correct") is None:
            r["comp_correct"] = em_or_f1(r["comp_answer"], r["gold"])


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

        from compress_router.evaluate import llm_judge
        train_router(model, cfg, evaluator=llm_judge())
        # or with custom settings:
        train_router(model, cfg, evaluator=llm_judge(concurrency=30, model="deepseek-chat"))
    """
    key = api_key or os.environ.get(env_var)
    if not key:
        raise ValueError(f"Provide api_key or set {env_var}")

    def evaluate(results: list[dict]) -> None:
        unevaluated = [r for r in results if r.get("comp_correct") is None]
        if not unevaluated:
            return
        asyncio.run(_judge_batch(unevaluated, key, base_url, model, concurrency))

    return evaluate


async def _judge_batch(results, api_key, base_url, model, concurrency):
    from openai import AsyncOpenAI
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(concurrency)

    total_calls = len(results) * 2
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    task = progress.add_task("LLM judge", total=total_calls)
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
                        log.warning("Judge error id=%s field=%s: %s", r.get("id"), pred_field, e)
                        r[correct_field] = False
                        progress.update(task, advance=1)

    tasks = []
    for r in results:
        tasks.append(judge_one(r, "full_answer", "full_correct"))
        tasks.append(judge_one(r, "comp_answer", "comp_correct"))

    try:
        await asyncio.gather(*tasks)
    finally:
        progress.stop()
        await client.close()
