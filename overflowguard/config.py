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

    # cross-validation + threshold
    n_folds: int = 5
    threshold_steps: int = 100
    threshold_policy: str = "youden"  # "youden" (default), or pass a callable

    # evaluation
    skip_full_wrong: bool = True

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
