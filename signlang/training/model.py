"""
Sequence models for skeleton-based sign recognition.
=====================================================
Input  : (B, L, D)  fixed-length landmark-feature sequences from features.py
Output : (B, C)     class logits

Skeleton features are low-dimensional and the dataset is small, so the models
are deliberately small and regularised.  An input LayerNorm absorbs residual
feature scaling (the geometry from features.py is normalised but not unit
variance), which keeps preprocessing free of fitted state.

Two architectures (``ModelConfig.arch``):
  - "bigru"       : input MLP -> 2-layer BiGRU -> attention pool -> head  (default)
  - "transformer" : input MLP -> + learned positions -> TransformerEncoder -> pool -> head

The transformer's positional table is sized to the real sequence length
(``max_len`` = L = 64).  It used to be a fixed 4096 x hidden table — 524k of the
model's 836k parameters were positions that could never be used.

All ops are ONNX-exportable (see export.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from signlang.config import ModelConfig

ARCHS = ("bigru", "transformer")
POOLS = ("attention", "mean", "last")


class AttentionPool(nn.Module):
    """Additive attention pooling over the time axis -> (B, dim)."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):                                    # x: (B, L, dim)
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)  # (B, L)
        return torch.bmm(w.unsqueeze(1), x).squeeze(1)       # (B, dim)


class _Pool(nn.Module):
    def __init__(self, kind: str, dim: int):
        super().__init__()
        if kind not in POOLS:
            raise ValueError(f"unknown pool {kind!r}; choose from {POOLS}")
        self.kind = kind
        self.attn = AttentionPool(dim) if kind == "attention" else None

    def forward(self, x):
        if self.kind == "attention":
            return self.attn(x)
        if self.kind == "mean":
            return x.mean(dim=1)
        return x[:, -1]                                      # "last"


class SignClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, cfg: ModelConfig, max_len: int = 64):
        super().__init__()
        if cfg.arch not in ARCHS:
            raise ValueError(f"unknown arch {cfg.arch!r}; choose from {ARCHS}")
        self.cfg = cfg
        self.max_len = int(max_len)
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Sequential(nn.Linear(in_dim, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout))

        self.rnn = self.encoder = self.pos = None
        if cfg.arch == "bigru":
            self.rnn = nn.GRU(cfg.hidden, cfg.hidden, num_layers=cfg.layers, batch_first=True,
                              bidirectional=True, dropout=cfg.dropout if cfg.layers > 1 else 0.0)
            feat = cfg.hidden * 2
        else:
            self.pos = nn.Parameter(torch.zeros(1, self.max_len, cfg.hidden))
            nn.init.trunc_normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden, nhead=cfg.n_heads, dim_feedforward=cfg.ff_dim,
                dropout=cfg.dropout, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.layers)
            feat = cfg.hidden

        self.pool = _Pool(cfg.pool, feat)
        self.head = nn.Sequential(nn.LayerNorm(feat), nn.Dropout(cfg.dropout), nn.Linear(feat, n_classes))

    def forward(self, x):                                    # x: (B, L, D)
        x = self.in_proj(self.in_norm(x))
        if self.rnn is not None:
            x, _ = self.rnn(x)
        else:
            x = self.encoder(x + self.pos[:, :x.size(1)])
        return self.head(self.pool(x))


def build_model(in_dim: int, n_classes: int, cfg: ModelConfig, max_len: int = 64) -> SignClassifier:
    return SignClassifier(in_dim, n_classes, cfg, max_len=max_len)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
