"""
Sequence models for skeleton-based sign recognition.
=====================================================
Input  : (B, L, D)  fixed-length landmark-feature sequences from features.py
Output : (B, C)     class logits

Skeleton features are low-dimensional and the dataset is small, so the models
are deliberately small and regularised.  An input LayerNorm absorbs residual
feature scaling (the geometry from features.py is normalised but not unit
variance), which keeps preprocessing fitted-state-free.

Two architectures (config.model.arch):
  - "bigru"       : input MLP -> 2-layer BiGRU -> attention pool -> head  (default)
  - "transformer" : input MLP -> +pos-enc -> TransformerEncoder -> pool -> head

All ops are ONNX-exportable (see export_optimize.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ModelConfig  # noqa: E402


class AttentionPool(nn.Module):
    """Additive attention pooling over the time axis -> (B, dim)."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):                       # x: (B, L, dim)
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)   # (B, L)
        return torch.bmm(w.unsqueeze(1), x).squeeze(1)        # (B, dim)


class _Pool(nn.Module):
    def __init__(self, kind: str, dim: int):
        super().__init__()
        self.kind = kind
        self.attn = AttentionPool(dim) if kind == "attention" else None

    def forward(self, x):
        if self.kind == "attention":
            return self.attn(x)
        if self.kind == "mean":
            return x.mean(dim=1)
        return x[:, -1]                          # "last"


class SignClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout))

        if cfg.arch == "bigru":
            self.rnn = nn.GRU(cfg.hidden, cfg.hidden, num_layers=cfg.layers,
                              batch_first=True, bidirectional=True,
                              dropout=cfg.dropout if cfg.layers > 1 else 0.0)
            feat = cfg.hidden * 2
            self.encoder = None
        elif cfg.arch == "transformer":
            self.pos = nn.Parameter(torch.zeros(1, 4096, cfg.hidden))
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden, nhead=cfg.n_heads, dim_feedforward=cfg.ff_dim,
                dropout=cfg.dropout, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.layers)
            feat = cfg.hidden
            self.rnn = None
        else:
            raise ValueError(f"unknown arch: {cfg.arch}")

        self.pool = _Pool(cfg.pool, feat)
        self.head = nn.Sequential(
            nn.LayerNorm(feat), nn.Dropout(cfg.dropout), nn.Linear(feat, n_classes))

    def forward(self, x):                        # x: (B, L, D)
        x = self.in_norm(x)
        x = self.in_proj(x)
        if self.rnn is not None:
            x, _ = self.rnn(x)
        else:
            x = x + self.pos[:, :x.size(1)]
            x = self.encoder(x)
        return self.head(self.pool(x))


def build_model(in_dim: int, n_classes: int, cfg: ModelConfig) -> SignClassifier:
    return SignClassifier(in_dim, n_classes, cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from config import CONFIG
    m = build_model(346, 12, CONFIG.model)
    x = torch.randn(4, CONFIG.feature.seq_len, 346)
    y = m(x)
    print(f"arch={CONFIG.model.arch}  params={count_params(m):,}  "
          f"in (4,{CONFIG.feature.seq_len},346) -> out {tuple(y.shape)}")
