"""
Dataset loading + augmentation for the sign-language recognition pipeline.
==========================================================================
- Reads the processed landmark clips listed in `processed_dataset/manifest.csv`
  (skips any FAIL verdict), labelled via `classes.json`.
- Runs `features.geometric()` ONCE per clip (cached in memory) so each epoch
  only pays for augmentation + `assemble()`.
- Augmentation (training split only) acts in the normalised coordinate space:
  temporal dropout / crop, horizontal flip WITH left/right hand+joint swap,
  rotation, scale, translation and jitter — then absent points are re-zeroed.

The horizontal flip is the high-value one for this dataset: it turns a
right-handed example into an anatomically valid left-handed one.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
from config import CLASSES_JSON, LANDMARKS_ALL, MANIFEST, TrainConfig  # noqa: E402
from utils import load_json  # noqa: E402


# ── Index the processed dataset ───────────────────────────────────────────────
def load_class_names() -> list:
    """Class names ordered by their integer label (from classes.json)."""
    classes = load_json(CLASSES_JSON)
    return [c for c, _ in sorted(classes.items(), key=lambda kv: kv[1])]


def load_index(drop_fail: bool = True) -> list:
    """Return [{'class','label','stem','path'}] for every usable clip."""
    classes = load_json(CLASSES_JSON)
    items = []
    if MANIFEST.exists():
        with open(MANIFEST, newline="") as fh:
            for row in csv.DictReader(fh):
                if drop_fail and row.get("verdict") == "FAIL":
                    continue
                cls, stem = row["class"], row["stem"]
                path = LANDMARKS_ALL / cls / f"{stem}.npy"
                if path.exists():
                    items.append({"class": cls, "label": classes[cls],
                                  "stem": stem, "path": path})
    else:  # fall back to a raw glob if the manifest is missing
        for path in sorted(LANDMARKS_ALL.glob("*/*.npy")):
            cls = path.parent.name
            items.append({"class": cls, "label": classes[cls],
                          "stem": path.stem, "path": path})
    if not items:
        raise FileNotFoundError(
            f"No clips found under {LANDMARKS_ALL}. Run extract_dataset.py first.")
    return items


def load_geometric(cfg_feature, spec, drop_fail: bool = True):
    """Load every clip and precompute its geometry once.
    Returns (clips, labels, stems) where clips[i] = (coords, vis, present_modal)."""
    index = load_index(drop_fail)
    clips, labels, stems = [], [], []
    for it in index:
        raw = np.load(it["path"])
        if raw.ndim != 2 or raw.shape[0] == 0:
            continue
        coords, vis, present = F.geometric(raw, cfg_feature, spec)
        clips.append((coords, vis, present))
        labels.append(it["label"])
        stems.append(f"{it['class']}/{it['stem']}")
    return clips, np.asarray(labels, np.int64), stems


# ── Augmentation (training only) ──────────────────────────────────────────────
def augment(coords, vis, present, cfg: TrainConfig, spec: F.FeatureSpec, rng):
    """Geometric + temporal augmentation in normalised coordinate space."""
    coords = coords.copy()
    present = {k: v.copy() for k, v in present.items()}
    T = coords.shape[0]

    # ── temporal: per-frame dropout then a random sub-window (speed warp) ──
    keep = np.arange(T)
    if cfg.aug_frame_dropout > 0 and T > 8:
        m = rng.random(T) >= cfg.aug_frame_dropout
        if m.sum() >= max(6, int(0.6 * T)):
            keep = keep[m]
    if cfg.aug_time_warp > 0 and len(keep) > 8:
        span = len(keep)
        newlen = max(8, int(span * (1.0 - rng.uniform(0.0, cfg.aug_time_warp))))
        start = int(rng.integers(0, span - newlen + 1))
        keep = keep[start:start + newlen]
    if len(keep) != T:
        coords = coords[keep]
        vis = vis[keep] if vis.shape[1] else vis
        present = {k: v[keep] for k, v in present.items()}
        T = len(keep)

    # ── horizontal flip: swap L/R hands + symmetric pose joints, negate x ──
    if cfg.aug_flip_prob > 0 and rng.random() < cfg.aug_flip_prob:
        coords = coords[:, spec.point_perm, :].copy()
        coords[..., 0] *= -1.0
        if spec.n_vis and vis.shape[1]:
            vis = vis[:, spec.vis_perm]
        if "left_hand" in present and "right_hand" in present:
            present["left_hand"], present["right_hand"] = \
                present["right_hand"], present["left_hand"]

    # presence map for re-zeroing after the geometric ops below
    pp = spec.point_present(present, T)                      # (T, P)

    # ── rotation about the origin (x, y plane) ──
    if cfg.aug_rotate_deg > 0:
        th = math.radians(rng.uniform(-cfg.aug_rotate_deg, cfg.aug_rotate_deg))
        c, s = math.cos(th), math.sin(th)
        x, y = coords[..., 0].copy(), coords[..., 1].copy()
        coords[..., 0] = c * x - s * y
        coords[..., 1] = s * x + c * y

    # ── scale, translation, jitter ──
    if cfg.aug_scale > 0:
        coords *= rng.uniform(1.0 - cfg.aug_scale, 1.0 + cfg.aug_scale)
    if cfg.aug_translate > 0:
        coords[..., 0] += rng.uniform(-cfg.aug_translate, cfg.aug_translate)
        coords[..., 1] += rng.uniform(-cfg.aug_translate, cfg.aug_translate)
    if cfg.aug_noise > 0:
        coords += rng.normal(0.0, cfg.aug_noise, coords.shape).astype(np.float32)

    coords *= pp[:, :, None]      # keep absent points exactly at the origin
    return coords.astype(np.float32), vis, present


# ── Torch dataset ─────────────────────────────────────────────────────────────
class SignDataset(Dataset):
    def __init__(self, clips, labels, cfg_feature, spec, train_cfg: TrainConfig,
                 indices=None, augment_on=False, seed=0):
        self.clips = clips
        self.labels = labels
        self.fc = cfg_feature
        self.spec = spec
        self.tc = train_cfg
        self.idx = np.arange(len(clips)) if indices is None else np.asarray(indices)
        self.augment_on = augment_on
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        coords, vis, present = self.clips[j]
        if self.augment_on and self.tc.aug:
            coords, vis, present = augment(coords, vis, present, self.tc, self.spec, self._rng)
        feats = F.assemble(coords, vis, present, self.fc, self.spec)
        return torch.from_numpy(feats), int(self.labels[j])


if __name__ == "__main__":
    from config import CONFIG
    spec = F.build_spec(CONFIG.feature)
    clips, labels, stems = load_geometric(CONFIG.feature, spec)
    print(f"loaded {len(clips)} clips, {len(set(labels.tolist()))} classes")
    ds = SignDataset(clips, labels, CONFIG.feature, spec, CONFIG.train,
                     augment_on=True, seed=0)
    x, y = ds[0]
    print(f"sample 0: {stems[0]}  x={tuple(x.shape)}  y={y}")
    import collections
    print("per class:", dict(sorted(collections.Counter(labels.tolist()).items())))
