"""
RouterClassifier - small MLP that decides compressed vs full.

This is the default classifier shipped with overflowguard. Users can
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


class RouterEnsemble(nn.Module):
    """K cross-validation fold models, averaged, with built-in standardization."""

    def __init__(self, models: list[RouterClassifier], mu: torch.Tensor, sd: torch.Tensor):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("mu", mu.view(1, -1))
        self.register_buffer("sd", sd.view(1, -1))
        self.d_input = models[0].d_input
        self.hidden = models[0].hidden

    @classmethod
    def empty(cls, n_models: int, d_input: int, hidden: int) -> "RouterEnsemble":
        """Build an untrained skeleton with the right shapes, ready for
        ``load_state_dict`` — so a saved ensemble loads in a single call."""
        models = [RouterClassifier(d_input=d_input, hidden=hidden) for _ in range(n_models)]
        return cls(models, torch.zeros(d_input), torch.ones(d_input))

    def forward(self, x):
        xs = (x - self.mu) / self.sd
        probs = torch.stack([torch.sigmoid(m(xs)) for m in self.models], dim=0)
        return probs.mean(dim=0)

    def predict(self, x):
        # mean ensemble probability on raw (unstandardized) features
        return self.forward(x)
