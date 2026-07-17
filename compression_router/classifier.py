"""
RouterClassifier - small MLP that decides compressed vs full.

This is the default classifier shipped with compress_router. Users can
substitute their own ``nn.Module`` via the training config - the only
contract is ``forward(x) -> (batch,)`` logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RouterClassifier(nn.Module):
    """Two-hidden-layer MLP: d_input → hidden → hidden/4 → 1."""

    def __init__(self, d_input: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.d_input = d_input
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))
