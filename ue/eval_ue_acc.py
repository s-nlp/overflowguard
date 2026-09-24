#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def read_scores(path: str, field: str) -> dict[int, float]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get(field)
            if v is None:
                continue
            v = float(v)
            if np.isfinite(v):
                out[int(r["id"])] = v
    return out


def load_collection(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    rows = obj["results"]
    return {int(r.get("id", i)): r for i, r in enumerate(rows)}, len(rows)


def align_train(scores, rows, require_full_correct=False):
    ids, x, y = [], [], []
    for sid in sorted(set(scores) & set(rows)):
        r = rows[sid]
        if r.get("comp_correct") is None:
            continue
        if require_full_correct and r.get("full_correct") is not True:
            continue
        ids.append(sid)
        x.append(scores[sid])
        y.append(0 if bool(r["comp_correct"]) else 1)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(x) == 0:
        raise RuntimeError("No usable train samples.")
    if len(np.unique(y)) < 2:
        raise RuntimeError("Train split contains only one class.")
    return ids, x, y


def align_test(scores, rows, require_full_correct=False):
    ids, x, kept = [], [], []
    for sid in sorted(set(scores) & set(rows)):
        r = rows[sid]
        if r.get("comp_correct") is None or r.get("full_correct") is None:
            continue
        if require_full_correct and r.get("full_correct") is not True:
            continue
        ids.append(sid)
        x.append(scores[sid])
        kept.append(r)
    if not ids:
        raise RuntimeError("No usable test samples.")
    return ids, np.asarray(x, dtype=float), kept


def fit_threshold(x, y):
    raw_auc = float(roc_auc_score(y, x))
    sign = 1.0 if raw_auc >= 0.5 else -1.0
    xo = sign * x
    fpr, tpr, th = roc_curve(y, xo)
    threshold = float(th[np.argmax(tpr - fpr)])
    return sign, threshold, raw_auc, float(roc_auc_score(y, xo))


def evaluate(x, rows, threshold):
    comp = np.asarray([float(bool(r["comp_correct"])) for r in rows])
    full = np.asarray([float(bool(r["full_correct"])) for r in rows])
    y = 1 - comp.astype(int)

    use_comp = x <= threshold
    use_full = ~use_comp
    routed = np.where(use_comp, comp, full)

    acc_c = float(comp.mean())
    acc_r = float(routed.mean())

    result = {
        "auc": float(roc_auc_score(y, x)),
        "accuracy": acc_r,
        "acc_always_compressed": acc_c,
        "acc_always_full": float(full.mean()),
        "delta_accuracy": acc_r - acc_c,
        "pct_compressed": float(use_comp.mean()),
        "n_compressed": int(use_comp.sum()),
        "n_full": int(use_full.sum()),
        "acc_oracle": float(np.maximum(comp, full).mean()),
    }

    if all(r.get("tokens_full") is not None and r.get("tokens_compressed") is not None for r in rows):
        ft = np.asarray([float(r["tokens_full"]) for r in rows])
        ct = np.asarray([float(r["tokens_compressed"]) for r in rows])
        routed_cost = np.where(use_comp, ct, ft)
        result["token_savings"] = 1.0 - float(routed_cost.sum()) / float(ft.sum())
    else:
        result["token_savings"] = None

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-scores", required=True)
    ap.add_argument("--train-collection", required=True)
    ap.add_argument("--test-scores", required=True)
    ap.add_argument("--test-collection", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--method", default=None)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--compressor", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--require-full-correct", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tr_scores = read_scores(args.train_scores, args.field)
    te_scores = read_scores(args.test_scores, args.field)
    tr_rows, tr_total = load_collection(args.train_collection)
    te_rows, te_total = load_collection(args.test_collection)

    tr_ids, xtr, ytr = align_train(tr_scores, tr_rows, args.require_full_correct)
    te_ids, xte, te_kept = align_test(te_scores, te_rows, args.require_full_correct)

    sign, threshold, raw_auc, train_auc = fit_threshold(xtr, ytr)
    xte = sign * xte
    ev = evaluate(xte, te_kept, threshold)

    result = {
        "metadata": {
            "dataset": args.dataset,
            "compressor": args.compressor,
            "model": args.model,
            "router": args.method or args.field,
            "field": args.field,
        },
        "threshold": threshold,
        "score_direction": "higher=overflow" if sign > 0 else "lower=overflow",
        "train": {
            "raw_auc": raw_auc,
            "auc": train_auc,
            "n": len(tr_ids),
            "coverage": len(tr_ids) / max(tr_total, 1),
        },
        "eval": {
            **ev,
            "n": len(te_ids),
            "coverage": len(te_ids) / max(te_total, 1),
        },
    }

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
