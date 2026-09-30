"""
Small shared helpers: console encoding, seeding, stratified k-fold, metrics, IO.
===============================================================================
Metrics and folds are implemented on NumPy (no scikit-learn dependency), so the
learning pipeline needs only numpy + torch.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np


# ── Console ───────────────────────────────────────────────────────────────────
def enable_utf8_console() -> None:
    """Windows consoles default to cp1252 and crash on non-ASCII output; make
    stdout/stderr UTF-8 tolerant so logging is never what breaks a run."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


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
def stratified_folds(labels, k: int, seed: int = 1337) -> list:
    """k validation-index arrays; every sample is in exactly one fold and each
    class is spread as evenly as possible across folds."""
    labels = np.asarray(labels)
    if k < 2:
        raise ValueError(f"need at least 2 folds, got {k}")
    if len(labels) < k:
        raise ValueError(f"cannot make {k} folds from {len(labels)} samples")
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    offset = 0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        for i, sample in enumerate(idx):
            # rotate the start fold per class so small classes don't all pile into fold 0
            folds[(i + offset) % k].append(int(sample))
        offset += len(idx)
    return [np.array(sorted(f), dtype=int) for f in folds]


# ── Metrics ───────────────────────────────────────────────────────────────────
def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_report(y_true, y_pred, class_names) -> dict:
    """Accuracy, macro-F1, per-class precision/recall/F1/support, confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, n)
    per_class, f1s = {}, []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        pred_pos = int(cm[:, i].sum())
        prec = tp / pred_pos if pred_pos else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[name] = {"precision": round(prec, 4), "recall": round(rec, 4),
                           "f1": round(f1, 4), "support": support}
        f1s.append(f1)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {"accuracy": round(acc, 4),
            "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
            "per_class": per_class,
            "confusion_matrix": cm.tolist()}


# ── IO ────────────────────────────────────────────────────────────────────────
def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_csv_rows(path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path, rows, fieldnames=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def parse_clip_ref(ref: str) -> tuple:
    """``"<class>/<stem>"`` (either slash) -> ``(class, stem)``; clear error otherwise."""
    parts = ref.replace("\\", "/").split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"expected <class>/<stem>, got {ref!r}")
    return parts[0], parts[1]


class Tee:
    """Print to stdout AND append to a log file (best effort)."""

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
