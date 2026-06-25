"""Plotting helpers (matplotlib, headless 'Agg'). Used by train.py / evaluate.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_confusion(cm, class_names, path, title="Confusion matrix", normalize=True):
    cm = np.asarray(cm, dtype=float)
    shown = cm.copy()
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        shown = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.62)))
    im = ax.imshow(shown, cmap="viridis", vmin=0, vmax=1 if normalize else shown.max())
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    thr = (shown.max() if not normalize else 1.0) * 0.6
    for i in range(n):
        for j in range(n):
            v = shown[i, j]
            if v > 0:
                txt = f"{v:.2f}" if normalize else f"{int(cm[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if v < thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cv_folds(fold_accs, path, title="Cross-validation accuracy per fold"):
    fold_accs = list(fold_accs)
    mean, std = float(np.mean(fold_accs)), float(np.std(fold_accs))
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = range(1, len(fold_accs) + 1)
    ax.bar(xs, fold_accs, color="#3b7dd8")
    ax.axhline(mean, color="#d8533b", ls="--", label=f"mean {mean:.3f} ± {std:.3f}")
    for x, a in zip(xs, fold_accs):
        ax.text(x, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_xlabel("fold"); ax.set_ylabel("val accuracy")
    ax.set_xticks(list(xs)); ax.set_title(title); ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_history(histories, path, title="Training history"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for i, h in enumerate(histories):
        ax1.plot(h["train_loss"], label=f"fold {i + 1}", alpha=0.8)
        ax2.plot(h["val_acc"], label=f"fold {i + 1}", alpha=0.8)
    ax1.set_title("train loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax2.set_title("val accuracy"); ax2.set_xlabel("epoch"); ax2.set_ylabel("acc")
    ax2.set_ylim(0, 1.05)
    if len(histories) > 1:
        ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
