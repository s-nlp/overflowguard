"""
Training pipeline for the routing classifier.

Usage (CLI):
    python -m overflowguard.train --config router_config.yaml --model my_module:MyRouter --model_path /models/my-model

Usage (Python):
    from overflowguard.train import train_router
    from my_module import MyRouter

    model = MyRouter.from_pretrained("/models/my-model")
    train_router(model, config)

Flow:
    1. Load dataset (JSONL: {"context", "query", "gold"})
    2. For each sample, run both paths, evaluate, extract features
    3. Save incrementally to collection.pt (resume-safe)
    4. K-fold CV: train CLF on each fold, collect val predictions
    5. Find optimal threshold from averaged CV predictions
    6. Retrain final CLF on all data, save with threshold
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .classifier import RouterClassifier, RouterEnsemble
from .config import TrainConfig
from .evaluate import default_evaluate

log = logging.getLogger(__name__)

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn



# ── Data loading ────────────────────────────────────────────────────


def load_dataset_rows(cfg: TrainConfig) -> list[dict]:
    path = cfg.dataset
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({
                "context": obj[cfg.column_map["context"]],
                "query": obj[cfg.column_map["query"]],
                "gold": obj[cfg.column_map["gold"]],
            })
            if cfg.max_samples and len(rows) >= cfg.max_samples:
                break
    return rows


# ── Batched fan-out helpers ────────────────────────────────────────
#
# Each falls back to the per-sample method when the router doesn't implement
# the batched one (or when the batch is a single sample), so routers that
# only define the 4 required methods keep working unchanged.


def _batch_generate_full(model, ctxs, queries, cfg):
    if len(ctxs) > 1 and hasattr(model, "generate_full_batch"):
        return model.generate_full_batch(ctxs, queries, max_new_tokens=cfg.max_new_tokens)
    return [model.generate_full(c, q, max_new_tokens=cfg.max_new_tokens)
            for c, q in zip(ctxs, queries)]


def _batch_compress(model, chunked, queries):
    """Returns a LIST of per-sample compressed tensors (not a stacked tensor):
    samples have different chunk counts, so they can't share a leading dim."""
    if len(chunked) > 1 and hasattr(model, "compress_batch"):
        return model.compress_batch(chunked, queries=queries)
    return [model.compress(ch, query=q) for ch, q in zip(chunked, queries)]


def _batch_generate_compressed(model, embs, queries, cfg):
    if len(embs) > 1 and hasattr(model, "generate_compressed_batch"):
        return model.generate_compressed_batch(embs, queries, max_new_tokens=cfg.max_new_tokens)
    return [model.generate_compressed(e, q, max_new_tokens=cfg.max_new_tokens)
            for e, q in zip(embs, queries)]


def _batch_features(model, ctxs, queries, embs):
    if len(ctxs) > 1 and hasattr(model, "extract_clf_features_batch"):
        import inspect
        params = inspect.signature(model.extract_clf_features_batch).parameters
        # routers that accept precomputed embeddings avoid compressing twice
        if "compressed_embs" in params:
            out = model.extract_clf_features_batch(ctxs, queries, compressed_embs=embs)
        else:
            out = model.extract_clf_features_batch(ctxs, queries)
        return [out[j].cpu() for j in range(len(ctxs))]
    return [model.extract_clf_features(e, q).cpu() for e, q in zip(embs, queries)]


# ── Feature collection (incremental) ───────────────────────────────


@torch.no_grad()
def collect_features(
    model,
    samples: list[dict],
    cfg: TrainConfig,
    evaluator: callable = None,
):
    """Run both paths on every sample, then evaluate with ``evaluator``.

    Steps:
        1. Generate full + compressed answers, extract CLF features.
        2. Save incrementally to collection.pt (resume-safe).
        3. Call ``evaluator(results)`` to score all predictions in-place.
        4. Optionally filter out samples where full answer is wrong.

    Args:
        evaluator: callable(results: list[dict]) → None, scores results
            in-place by setting ``full_correct`` and ``comp_correct``.
            Default: EM-or-F1.
    """
    if evaluator is None:
        evaluator = default_evaluate

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "collection.pt"

    # resume from partial
    features = []
    results = []
    start_idx = 0

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        cached_first = cached.get("first_sample")
        if cached_first and cached_first != samples[0]:
            log.info("Cache doesn't match current dataset — recomputing")
        else:
            features = list(cached["features"].unbind(0)) if cached["features"].dim() > 0 else []
            results = cached["results"]
            start_idx = cached.get("next_idx", len(results))
            if start_idx >= len(samples):
                log.info("Collection complete (%d samples), using cache", len(results))
                # re-evaluate if not yet scored
                if results and results[0].get("comp_correct") is None:
                    log.info("Scoring cached results...")
                    evaluator(results)
                return cached["features"], results
            log.info("Resuming collection from sample %d/%d", start_idx, len(samples))

    model.eval()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    task = progress.add_task("Collecting features", total=len(samples), completed=start_idx)
    progress.start()

    bs = max(1, getattr(cfg, "collect_batch_size", 1))

    try:
        for start in range(start_idx, len(samples), bs):
            batch = samples[start:start + bs]
            ctxs = [s["context"] for s in batch]
            queries = [s["query"] for s in batch]
            chunked = [model._chunk_text(c) for c in ctxs]

            model._current_sample_idx = start

            full_answers = _batch_generate_full(model, ctxs, queries, cfg)
            embs = _batch_compress(model, chunked, queries)
            comp_answers = _batch_generate_compressed(model, embs, queries, cfg)
            feats = _batch_features(model, ctxs, queries, embs)

            for j, sample in enumerate(batch):
                features.append(feats[j])
                results.append({
                    "id": start + j,
                    "query": queries[j],
                    "gold": sample["gold"],
                    "context": ctxs[j],
                    "comp_answer": comp_answers[j],
                    "full_answer": full_answers[j],
                    "comp_correct": None,
                    "full_correct": None,
                    "tokens_full": model.count_tokens_full(ctxs[j], queries[j]),
                    "tokens_compressed": model.count_tokens_compressed(embs[j], queries[j]),
                })

            torch.save(
                {"features": torch.stack(features), "results": results,
                 "first_sample": samples[0], "next_idx": start + len(batch)},
                cache_path,
            )

            progress.update(task, advance=len(batch))

    finally:
        progress.stop()

    # score all results
    _eval_stages = {model.TRAIN_COLLECT: model.TRAIN_EVALUATE, model.EVAL_COLLECT: model.EVAL_EVALUATE}
    with model.enter_stage(_eval_stages.get(model.stage, model.stage)):
        log.info("Evaluating %d samples...", len(results))
        evaluator(results)

    # filter out samples where full answer is wrong
    if cfg.skip_full_wrong:
        keep_idx = [i for i, r in enumerate(results) if r["full_correct"]]
        n_skipped = len(results) - len(keep_idx)
        if n_skipped:
            log.info("Dropped %d samples where full answer was wrong", n_skipped)
        features = [features[i] for i in keep_idx]
        results = [results[i] for i in keep_idx]

    stacked = torch.stack(features)
    torch.save({"features": stacked, "results": results,
                "first_sample": samples[0], "next_idx": len(samples)}, cache_path)
    return stacked, results


# ── CLF training ────────────────────────────────────────────────────


def train_clf(
    features: torch.Tensor,
    labels: torch.Tensor,
    cfg: TrainConfig,
    val_features: torch.Tensor | None = None,
    val_labels: torch.Tensor | None = None,
) -> RouterClassifier:
    import numpy as np

    d_input = features.shape[-1]
    clf = RouterClassifier(d_input=d_input, hidden=cfg.clf_hidden, dropout=cfg.clf_dropout)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf.to(device)

    optimizer = torch.optim.AdamW(clf.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        p = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        return max(0.01, 0.5 * (1 + np.cos(np.pi * p)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_ds = TensorDataset(features.to(device), labels.to(device))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    has_val = val_features is not None and val_labels is not None
    if has_val:
        val_ds = TensorDataset(val_features.to(device), val_labels.to(device))
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(cfg.epochs):
        clf.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = clf(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        scheduler.step()

        train_avg = train_loss / len(train_ds)

        # early stop on val loss if available, else train loss
        clf.eval()
        if has_val:
            with torch.no_grad():
                val_loss = sum(
                    loss_fn(clf(xb), yb).item() * len(yb)
                    for xb, yb in val_loader
                ) / len(val_ds)
            monitor_loss = val_loss
        else:
            monitor_loss = train_avg

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = f"Epoch {epoch+1}/{cfg.epochs} — train: {train_avg:.4f}"
            if has_val:
                msg += f", val: {val_loss:.4f}"
            msg += f" (best: {best_loss:.4f}, patience: {patience_counter}/{cfg.patience})"
            log.info(msg)

        if patience_counter >= cfg.patience:
            log.info("Early stopping at epoch %d", epoch + 1)
            break

    clf.load_state_dict(best_state)
    clf.eval()
    return clf


# ── Cross-validated threshold search ───────────────────────────────


def _cv_predict(
    features: torch.Tensor,
    labels: torch.Tensor,
    cfg: TrainConfig,
    n_folds: int = 5,
) -> tuple[torch.Tensor, list[RouterClassifier]]:
    """Train CLF on stratified k folds.

    Returns:
        oof_probs: out-of-fold probabilities (one per sample, from the fold
            that held it out — leakage-free, used for threshold search).
        models: the K trained fold models, kept for the inference ensemble.
    """
    from sklearn.model_selection import StratifiedKFold

    n = len(labels)
    oof_probs = torch.zeros(n)
    models: list[RouterClassifier] = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    task = progress.add_task("CV folds", total=n_folds)
    progress.start()

    try:
        for fold_i, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
            torch.manual_seed(cfg.seed + fold_i)  # deterministic per-fold init
            clf = train_clf(
                features[train_idx], labels[train_idx], cfg,
                val_features=features[val_idx], val_labels=labels[val_idx],
            )

            device = next(clf.parameters()).device
            with torch.no_grad():
                val_probs = clf.predict(features[val_idx].to(device)).cpu()
            oof_probs[val_idx] = val_probs
            models.append(clf)

            progress.update(task, advance=1)
    finally:
        progress.stop()

    return oof_probs, models


# ── Threshold policies ─────────────────────────────────────────────


def threshold_youden(probs, results, **kwargs):
    """Youden's J statistic via ROC curve (max TPR - FPR).

    Labels: 1 = overflow (compressed wrong), 0 = safe (compressed correct).
    High prob → likely overflow → route to full.
    """
    import numpy as np
    from sklearn.metrics import roc_curve, roc_auc_score

    probs_np = np.array(probs)
    labels = np.array([0.0 if r["comp_correct"] else 1.0 for r in results])
    comp_correct = np.array([r["comp_correct"] for r in results], dtype=float)
    full_toks = np.array([r.get("tokens_full", 1) for r in results], dtype=float)
    comp_toks = np.array([r.get("tokens_compressed", 1) for r in results], dtype=float)

    fpr, tpr, thresholds = roc_curve(labels, probs_np)
    auc = roc_auc_score(labels, probs_np)
    t_opt = float(thresholds[np.argmax(tpr - fpr)])

    use_comp = probs_np <= t_opt
    correct = np.where(use_comp, comp_correct, 1)
    acc = correct.mean()
    savings = 1.0 - np.where(use_comp, comp_toks, full_toks).sum() / full_toks.sum()

    return t_opt, {
        "auc": round(auc, 4),
        "accuracy": round(float(acc), 4),
        "token_savings": round(float(savings), 4),
        "pct_compressed": round(float(use_comp.mean()), 4),
        "n_compressed": int(use_comp.sum()),
        "n_full": int((~use_comp).sum()),
    }


def eval_at_threshold(probs, results, threshold):
    """Compute metrics at a fixed threshold (for held-out eval)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    probs_np = np.array(probs)
    labels = np.array([0.0 if r["comp_correct"] else 1.0 for r in results])
    comp_correct = np.array([r["comp_correct"] for r in results], dtype=float)
    full_toks = np.array([r.get("tokens_full", 1) for r in results], dtype=float)
    comp_toks = np.array([r.get("tokens_compressed", 1) for r in results], dtype=float)

    auc = roc_auc_score(labels, probs_np)
    use_comp = probs_np <= threshold
    correct = np.where(use_comp, comp_correct, 1)

    return {
        "auc": round(auc, 4),
        "accuracy": round(float(correct.mean()), 4),
        "token_savings": round(float(1.0 - np.where(use_comp, comp_toks, full_toks).sum() / full_toks.sum()), 4),
        "pct_compressed": round(float(use_comp.mean()), 4),
        "n_compressed": int(use_comp.sum()),
        "n_full": int((~use_comp).sum()),
    }


THRESHOLD_POLICIES = {
    "youden": threshold_youden,
}


# ── Main entry point ────────────────────────────────────────────────


@torch.no_grad()
def collect_features_only(model, output_dir: str, labels_dir: str, split_subdir: str = "",
                          batch_size: int = 1):
    """Re-extract features for an ALREADY-labeled collection — no generation,
    no judge. Reads comp_correct/full_correct from ``labels_dir`` and recomputes
    only the (multi-layer) features via ``extract_clf_features``. Resume-safe.

    This is the fast path for a layer sweep: labeling (generate both paths +
    judge) is the expensive step and was already done by a prior training run,
    so we reuse it and only pay for one cheap forward per sample.

    If ``batch_size > 1`` and the router implements ``extract_clf_features_batch
    (contexts, queries) -> (B, ...)``, samples are extracted in batches (one
    padded decoder forward per batch) — much better GPU utilization. Otherwise
    it falls back to per-sample extraction.
    """
    src = Path(labels_dir) / split_subdir / "collection.pt"
    if not src.exists():
        raise FileNotFoundError(f"labels_from set but {src} not found")
    results = torch.load(src, map_location="cpu", weights_only=False)["results"]

    out = Path(output_dir) / split_subdir
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "sweep_feats.pt"

    # tag the cache with the extraction mode so a stale cache from a different
    # code path (e.g. an earlier batched run) is invalidated, not reused.
    fingerprint = f"bs{batch_size}"

    features, start_idx = [], 0
    if cache_path.exists():
        c = torch.load(cache_path, map_location="cpu", weights_only=False)
        if c.get("n") == len(results) and c.get("fingerprint") == fingerprint:
            features = list(c["features"].unbind(0))
            start_idx = c["next_idx"]
            if start_idx >= len(results):
                return c["features"], results
            log.info("Resuming feature-only extraction from %d/%d", start_idx, len(results))
        else:
            log.info("Cache present but stale (mode/size mismatch) — recomputing")

    batched = batch_size > 1 and hasattr(model, "extract_clf_features_batch")
    if batch_size > 1 and not batched:
        log.warning("collect_batch_size>1 but %s has no extract_clf_features_batch; "
                    "falling back to per-sample", type(model).__name__)

    model.eval()
    # Re-saving the whole (growing, multi-GB) feature tensor every step is
    # disk-bound and starves the GPU — checkpoint every ~256 samples instead.
    save_every = max(batch_size, 256)
    last_saved = start_idx

    def _checkpoint(next_idx):
        torch.save({"features": torch.stack(features), "next_idx": next_idx,
                    "n": len(results), "fingerprint": fingerprint}, cache_path)

    progress = Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"),
                        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn())
    task = progress.add_task(f"Re-extracting features {split_subdir}".strip(),
                             total=len(results), completed=start_idx)
    progress.start()
    try:
        i = start_idx
        while i < len(results):
            if batched:
                batch = results[i:i + batch_size]
                feats_b = model.extract_clf_features_batch(
                    [r["context"] for r in batch], [r["query"] for r in batch])
                features.extend(feats_b[j].cpu() for j in range(len(batch)))
                i += len(batch)
            else:
                r = results[i]
                emb = model.compress(model._chunk_text(r["context"]), query=r["query"])
                features.append(model.extract_clf_features(emb, r["query"]).cpu())
                i += 1
            if i - last_saved >= save_every:
                _checkpoint(i)
                last_saved = i
            progress.update(task, completed=i)
    finally:
        progress.stop()

    stacked = torch.stack(features)
    torch.save({"features": stacked, "next_idx": len(results), "n": len(results),
                "fingerprint": fingerprint}, cache_path)
    return stacked, results


def _run_sweep(model, features, labels, results, cfg, evaluator):
    """Per-layer sweep: reuse the exact train pipeline on each layer slice.

    Expects ``features`` shaped ``(n_samples, n_layers, ...)`` — i.e. the user's
    ``extract_clf_features`` returned an all-layers stack per sample. For each
    layer we flatten the trailing dims to ``(n_samples, d)`` and run the same
    standardize → CV ensemble → threshold → eval path as normal training.
    Writes ``sweep_results.json`` and returns the ranked table.
    """
    from sklearn.metrics import roc_auc_score

    if features.dim() < 2:
        raise ValueError(
            "cfg.sweep=True but extract_clf_features returned a flat feature; "
            "return a stacked (n_layers, ...) tensor per sample for a sweep."
        )
    n_samples, n_layers = features.shape[0], features.shape[1]
    log.info("Layer sweep: %d layers × %d samples", n_layers, n_samples)

    # collect eval features once (all layers), if an eval set is given
    eval_features = eval_results = None
    if cfg.eval_dataset:
        with model.enter_stage(model.EVAL_COLLECT):
            if cfg.labels_from:
                eval_features, eval_results = collect_features_only(
                    model, cfg.output_dir, cfg.labels_from, split_subdir="eval",
                    batch_size=cfg.collect_batch_size)
            else:
                eval_cfg = TrainConfig(**{**cfg.__dict__, "dataset": cfg.eval_dataset,
                                          "output_dir": cfg.output_dir + "/eval"})
                eval_samples = load_dataset_rows(eval_cfg)
                eval_features, eval_results = collect_features(model, eval_samples, eval_cfg, evaluator=evaluator)

    policy_fn = cfg.threshold_policy
    if isinstance(policy_fn, str):
        policy_fn = THRESHOLD_POLICIES[policy_fn]

    rows = []
    for L in range(n_layers):
        Xtr = features[:, L].reshape(n_samples, -1)
        mu, sd = Xtr.mean(dim=0), Xtr.std(dim=0) + 1e-6
        oof, models = _cv_predict((Xtr - mu) / sd, labels, cfg, n_folds=cfg.n_folds)
        cv_auc = float(roc_auc_score(labels.numpy(), oof.numpy()))
        threshold, th_stats = policy_fn(oof, results, steps=cfg.threshold_steps)

        row = {"layer": L, "cv_auc": round(cv_auc, 4), "threshold": round(threshold, 4)}
        if eval_features is not None:
            ens = RouterEnsemble(models, mu, sd).eval()
            device = next(ens.parameters()).device
            ens.to(device)
            Xev = eval_features[:, L].reshape(eval_features.shape[0], -1)
            with torch.no_grad():
                ev_prob = ens.predict(Xev.to(device)).cpu()
            ev_stats = eval_at_threshold(ev_prob, eval_results, threshold)
            row["eval_auc"] = ev_stats["auc"]
            row["eval_accuracy"] = ev_stats["accuracy"]
            row["eval_token_savings"] = ev_stats["token_savings"]
        rows.append(row)
        log.info("  layer %2d: cv_auc=%.4f%s", L, cv_auc,
                 f"  eval_auc={row['eval_auc']:.4f}" if "eval_auc" in row else "")

    key = "eval_auc" if eval_features is not None else "cv_auc"
    ranked = sorted(rows, key=lambda r: r[key], reverse=True)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep_results.json").write_text(json.dumps(rows, indent=2))
    best = ranked[0]
    log.info("Best layer: %d (%s=%.4f). Results → %s",
             best["layer"], key, best[key], out / "sweep_results.json")
    return {"sweep": rows, "best_layer": best["layer"], "ranked_by": key}


def train_router(model, cfg: TrainConfig, evaluator: callable = None):
    """End-to-end: collect features → evaluate → CV threshold → train CLF → save.

    Args:
        evaluator: callable(results) that scores predictions in-place.
            Default: EM-or-F1. For LLM judge, pass your own function.
    """
    # ── collect train features ──
    log.info("Loading train dataset from %s", cfg.dataset)
    samples = load_dataset_rows(cfg)
    log.info("Loaded %d train samples", len(samples))

    with model.enter_stage(model.TRAIN_COLLECT):
        if cfg.sweep and cfg.labels_from:
            log.info("Sweep: reusing labels from %s (feature-only extraction)", cfg.labels_from)
            features, results = collect_features_only(
                model, cfg.output_dir, cfg.labels_from, batch_size=cfg.collect_batch_size)
        else:
            features, results = collect_features(model, samples, cfg, evaluator=evaluator)

    labels = torch.tensor([0.0 if r["comp_correct"] else 1.0 for r in results])
    n_overflow = int(labels.sum())
    log.info(
        "Train: %d samples — overflow (comp wrong): %d (%.1f%%)",
        len(results), n_overflow, 100 * n_overflow / max(len(results), 1),
    )

    # ── layer sweep: iterate the layer dim, report per-layer AUC ──
    if cfg.sweep:
        return _run_sweep(model, features, labels, results, cfg, evaluator)

    # standardize features (mu/sd from full train set) — baked into the
    # ensemble so inference passes raw features unchanged.
    mu = features.mean(dim=0)
    sd = features.std(dim=0) + 1e-6
    features_std = (features - mu) / sd

    # ── CV: train K fold models, collect oof probs + keep the models ──
    with model.enter_stage(model.TRAIN_THRESHOLD_SEARCH):
        log.info("Running %d-fold CV (kept as ensemble)...", cfg.n_folds)
        oof_probs, fold_models = _cv_predict(features_std, labels, cfg, n_folds=cfg.n_folds)

        policy_fn = cfg.threshold_policy
        if isinstance(policy_fn, str):
            policy_fn = THRESHOLD_POLICIES[policy_fn]
        threshold, th_stats = policy_fn(oof_probs, results, steps=cfg.threshold_steps)
        log.info("Optimal threshold (CV): %.3f — %s", threshold, th_stats)

    # ── assemble the ensemble (no retrain on all data) ──
    with model.enter_stage(model.TRAIN_CLF):
        log.info("Assembling %d-model ensemble...", len(fold_models))
        clf = RouterEnsemble(fold_models, mu, sd)
        clf.eval()

    # ── eval on held-out set ──
    eval_stats = None
    if cfg.eval_dataset:
        log.info("Loading eval dataset from %s", cfg.eval_dataset)
        eval_cfg = TrainConfig(**{**cfg.__dict__, "dataset": cfg.eval_dataset, "output_dir": cfg.output_dir + "/eval"})
        eval_samples = load_dataset_rows(eval_cfg)
        log.info("Loaded %d eval samples", len(eval_samples))

        with model.enter_stage(model.EVAL_COLLECT):
            eval_features, eval_results = collect_features(model, eval_samples, eval_cfg, evaluator=evaluator)

        with model.enter_stage(model.EVAL_CLF):
            device = next(clf.parameters()).device
            clf.to(device)  # ensure fold models + mu/sd buffers colocated
            with torch.no_grad():
                eval_probs = clf.predict(eval_features.to(device)).cpu()

            eval_stats = eval_at_threshold(eval_probs, eval_results, threshold)
            log.info("Eval metrics (threshold=%.3f): %s", threshold, eval_stats)

    # save
    with model.enter_stage(model.SAVE):
        model.clf = clf
        model.routing_threshold = threshold
        model.save_pretrained(cfg.output_dir)

        summary = {"threshold": threshold, "train": th_stats}
        if eval_stats:
            summary["eval"] = eval_stats
        (Path(cfg.output_dir) / "train_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        log.info("Saved to %s", cfg.output_dir)

    if cfg.push_to_hub and cfg.hub_repo_id:
        model.push_to_hub(cfg.hub_repo_id, output_dir=cfg.output_dir)
        log.info("Pushed to HF: %s", cfg.hub_repo_id)

    return summary


# ── CLI ─────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train a routing classifier.")
    parser.add_argument("--config", required=True, help="Path to router_config.yaml")
    parser.add_argument("--model", required=True, help="Module:ClassName (e.g. my_module:MyRouter)")
    parser.add_argument("--model_path", required=True, help="Path or HF repo for from_pretrained")
    parser.add_argument("--force", action="store_true", help="Recompute features, ignore cache")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)

    if args.force:
        cache_path = Path(cfg.output_dir) / "collection.pt"
        if cache_path.exists():
            cache_path.unlink()
            log.info("Cleared cache")

    module_name, class_name = args.model.split(":")
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)

    log.info("Loading model %s from %s", class_name, args.model_path)
    model = cls.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()

    if torch.cuda.is_available():
        model.park_gpu()

    result = train_router(model, cfg)
    log.info("Done. %s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
