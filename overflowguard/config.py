"""
Training configuration — loaded from YAML or constructed in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    # data
    dataset: str = ""
    eval_dataset: str | None = None
    column_map: dict[str, str] = field(default_factory=lambda: {
        "context": "context",
        "query": "query",
        "gold": "gold",
    })
    max_samples: int | None = None

    # classifier
    clf_hidden: int = 512
    clf_dropout: float = 0.3

    # training
    epochs: int = 60
    lr: float = 1e-4
    batch_size: int = 64
    weight_decay: float = 1e-3
    patience: int = 12
    warmup_epochs: int = 5

    # reproducibility
    seed: int = 42

    # cross-validation + threshold
    n_folds: int = 5
    threshold_steps: int = 100
    threshold_policy: str = "youden"  # "youden" (default), or pass a callable

    # evaluation
    # With the strict overflow label (comp wrong AND full right), full-wrong
    # rows are valid negatives. True excludes them from TRAINING for this run
    # only (in memory); the saved collection always keeps every row.
    skip_full_wrong: bool = False

    # layer sweep — when True, extract_clf_features must return a stacked
    # (n_layers, ...) feature per sample; training iterates the layer dim and
    # reports per-layer AUC instead of fitting one router.
    sweep: bool = False
    # batch size for feature collection. >1 uses the router's *_batch methods
    # where it implements them (generate_full_batch, compress_batch,
    # generate_compressed_batch, extract_clf_features_batch), falling back to
    # per-sample for any it doesn't.
    collect_batch_size: int = 1

    # output
    output_dir: str = "./router_checkpoint"
    push_to_hub: bool = False
    hub_repo_id: str | None = None

    # generation
    max_new_tokens: int = 64

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text())
        flat = {}
        for k, v in raw.items():
            if isinstance(v, dict) and k in ("clf", "data", "training", "output"):
                flat.update(v)
            else:
                flat[k] = v
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in flat.items() if k in known})
