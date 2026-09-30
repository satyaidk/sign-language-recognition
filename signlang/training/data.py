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

import math

import numpy as np
import torch
from torch.utils.data import Dataset

from signlang.config import Paths, TrainConfig
from signlang.training import features as F
from signlang.utils import load_json, read_csv_rows


# ── Index the processed dataset ───────────────────────────────────────────────
def load_class_names(paths: Paths) -> list:
    """Class names ordered by their integer label (from classes.json)."""
    if not paths.classes_json.exists():
        raise FileNotFoundError(f"{paths.classes_json} not found — run `python -m signlang extract` first")
    classes = load_json(paths.classes_json)
    labels = sorted(classes.values())
    if labels != list(range(len(labels))):
        raise ValueError(f"classes.json labels must be 0..{len(labels) - 1} without gaps, got {labels}")
    return [c for c, _ in sorted(classes.items(), key=lambda kv: kv[1])]


def load_index(paths: Paths, drop_fail: bool = True) -> list:
    """``[{'class', 'label', 'stem', 'path'}]`` for every usable clip.

    Uses manifest.csv (skipping FAIL verdicts); falls back to globbing
    ``landmarks_all`` when there is no manifest.  Rows whose .npy is missing are
    reported instead of being dropped silently.
    """
    classes = load_json(paths.classes_json)
    items, missing = [], []
    if paths.manifest.exists():
        for row in read_csv_rows(paths.manifest):
            if drop_fail and row.get("verdict") == "FAIL":
                continue
            cls, stem = row["class"], row["stem"]
            if cls not in classes:
                raise KeyError(f"manifest class {cls!r} is not in classes.json")
            path = paths.landmarks_all / cls / f"{stem}.npy"
            if path.exists():
                items.append({"class": cls, "label": classes[cls], "stem": stem, "path": path})
            else:
                missing.append(f"{cls}/{stem}")
    else:
        for path in sorted(paths.landmarks_all.glob("*/*.npy")):
            cls = path.parent.name
            if cls in classes:
                items.append({"class": cls, "label": classes[cls], "stem": path.stem, "path": path})
    if missing:
        print(f"[data] warning: {len(missing)} manifest clip(s) have no .npy "
              f"(e.g. {', '.join(missing[:3])})")
    if not items:
        raise FileNotFoundError(f"No clips found under {paths.landmarks_all}. "
                                "Run `python -m signlang extract` first.")
    return items


def load_geometric(paths: Paths, cfg_feature, spec, drop_fail: bool = True):
    """Load every clip and precompute its geometry once.
    Returns (clips, labels, stems) where clips[i] = (coords, vis, present_modal)."""
    index = load_index(paths, drop_fail)
    clips, labels, stems = [], [], []
    for it in index:
        raw = np.load(it["path"])
        if raw.ndim != 2 or raw.shape[0] == 0:
            print(f"[data] warning: skipping empty/invalid clip {it['class']}/{it['stem']} {raw.shape}")
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
