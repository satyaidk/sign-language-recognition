"""
Small shared helpers: seeding, metrics, stratified k-fold, JSON IO.
No scikit-learn dependency — everything is implemented on NumPy so the pipeline
runs with just numpy + torch.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np


# Windows consoles default to cp1252 and raise UnicodeEncodeError on stray
# non-ASCII output (and worse, mangle piped output). Make stdout/stderr UTF-8
# tolerant once, on import, so logging is never the thing that crashes a run.
def _enable_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_enable_utf8_console()


# ── Reproducibility ───────────────────────────────────────────────────────────
def seed_everything(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── Cross-validation splits ───────────────────────────────────────────────────
def stratified_folds(labels, k: int, seed: int = 1337):
    """Return k arrays of indices (the validation fold for each split),
    stratified so every class is spread as evenly as possible across folds."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        for i, sample in enumerate(idx):
            folds[i % k].append(int(sample))
    return [np.array(sorted(f), dtype=int) for f in folds]


# ── Metrics (no sklearn) ──────────────────────────────────────────────────────
def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_report(y_true, y_pred, class_names) -> dict:
    """Accuracy + per-class precision/recall/F1/support + confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, n)
    per_class, macro_f1 = {}, []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        pred_pos = int(cm[:, i].sum())
        prec = tp / pred_pos if pred_pos else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[name] = {"precision": round(prec, 4), "recall": round(rec, 4),
                           "f1": round(f1, 4), "support": support}
        macro_f1.append(f1)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {"accuracy": round(acc, 4),
            "macro_f1": round(float(np.mean(macro_f1)) if macro_f1 else 0.0, 4),
            "per_class": per_class,
            "confusion_matrix": cm.tolist()}


# ── IO ────────────────────────────────────────────────────────────────────────
def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


class Tee:
    """Print to stdout AND append to a log file (best-effort)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg)
        try:
            self._fh.write(msg + "\n")
            self._fh.flush()
        except Exception:
            pass

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass
