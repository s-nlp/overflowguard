#!/usr/bin/env python3
"""
Threshold-free evaluation of one uncertainty baseline.

This does NOT need train scores for Perplexity / Focus / RAUQ.
It computes test AUROC/AUPR directly.

For a fair comparison to a previously recorded OverflowGuard AUC, the exact
same test population and overflow label definition must be used.

Default positive class:
    overflow = not comp_correct

Optional:
    --require-full-correct
keeps only samples with full_correct=True. Use this only if the corresponding
OverflowGuard AUC was evaluated on that same filtered population.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def read_scores(path, field):
    scores = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get(field)
            if v is not None and np.isfinite(float(v)):
                scores[int(r["id"])] = float(v)
    return scores


def load_rows(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    rows = {}
    for i, r in enumerate(obj["results"]):
        rows[int(r.get("id", i))] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--compressor", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", default=None)
    ap.add_argument("--require-full-correct", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    scores = read_scores(args.scores, args.field)
    rows = load_rows(args.collection)

    ids, x, y = [], [], []
    for sid in sorted(set(scores) & set(rows)):
        r = rows[sid]
        if r.get("comp_correct") is None:
            continue
        if args.require_full_correct:
            if r.get("full_correct") is not True:
                continue

        ids.append(sid)
        x.append(scores[sid])
        y.append(0 if bool(r["comp_correct"]) else 1)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    if len(x) == 0:
        raise RuntimeError("No usable scored/labeled examples.")
    if len(np.unique(y)) != 2:
        raise RuntimeError("Need both overflow and non-overflow samples for AUROC.")

    # Orientation is part of the estimator semantics for most UE measures.
    # For robust reporting we output both raw AUC and direction-normalized AUC.
    raw_auc = float(roc_auc_score(y, x))
    sign = 1.0 if raw_auc >= 0.5 else -1.0
    oriented = sign * x

    result = {
        "dataset": args.dataset,
        "compressor": args.compressor,
        "model": args.model,
        "method": args.method or args.field,
        "field": args.field,
        "population": (
            "full_correct_only" if args.require_full_correct else "all_labeled"
        ),
        "n": len(ids),
        "coverage_of_collection": len(ids) / max(len(rows), 1),
        "overflow_rate": float(y.mean()),
        "score_direction": "higher=overflow" if sign > 0 else "lower=overflow",
        "raw_auc": raw_auc,
        "auc": float(roc_auc_score(y, oriented)),
        "aupr": float(average_precision_score(y, oriented)),
    }

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
