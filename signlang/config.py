"""
Central configuration: project paths + every tunable of the learning pipeline.
=============================================================================
Everything tunable lives here so features, data, model, training and inference
read ONE source of truth.  Stages may override individual fields from the CLI.

Paths
    Defaults are relative to the repository root, so nothing depends on the
    current working directory:

        dataset/             raw videos           <class>/<clip>.mp4
        processed_dataset/   landmark dataset     (extract -> verify)
        artifacts/           training outputs     (train -> finetune -> export)

    Each can be redirected with an environment variable (``SIGNLANG_DATASET_DIR``,
    ``SIGNLANG_PROCESSED_DIR``, ``SIGNLANG_ARTIFACTS_DIR``) or a CLI flag
    (``--dataset-dir`` / ``--processed-dir`` / ``--artifacts-dir``).  The tests
    use this to run the whole pipeline in a temporary folder.

This module imports nothing heavy, so it is safe to import from anywhere.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from signlang.landmarks.layout import (  # noqa: F401  (re-exported for convenience)
    BLOCK, LEFT_SHOULDER, N_FACE, N_HAND, N_POSE, RAW_TOTAL, RIGHT_SHOULDER,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Paths ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Paths:
    dataset: Path          # raw source videos
    processed: Path        # processed landmark dataset
    artifacts: Path        # checkpoints, metrics, exported models, predictions

    # processed dataset
    @property
    def landmarks_all(self) -> Path:
        return self.processed / "landmarks_all"

    @property
    def manifest(self) -> Path:
        return self.processed / "manifest.csv"

    @property
    def classes_json(self) -> Path:
        return self.processed / "classes.json"

    @property
    def feature_layout_json(self) -> Path:
        return self.processed / "FEATURE_LAYOUT.json"

    # training artifacts
    @property
    def checkpoints(self) -> Path:
        return self.artifacts / "checkpoints"

    @property
    def metrics(self) -> Path:
        return self.artifacts / "metrics"

    @property
    def exported(self) -> Path:
        return self.artifacts / "exported"

    @property
    def predictions(self) -> Path:
        return self.artifacts / "predictions"

    @property
    def logs(self) -> Path:
        return self.artifacts / "logs"

    def ensure_artifact_dirs(self) -> None:
        for d in (self.checkpoints, self.metrics, self.exported, self.predictions, self.logs):
            d.mkdir(parents=True, exist_ok=True)


def get_paths(dataset=None, processed=None, artifacts=None) -> Paths:
    """Resolve paths: explicit argument > environment variable > repo default."""
    def pick(value, env, default):
        if value:
            return Path(value)
        if os.environ.get(env):
            return Path(os.environ[env])
        return PROJECT_ROOT / default

    return Paths(dataset=pick(dataset, "SIGNLANG_DATASET_DIR", "dataset"),
                 processed=pick(processed, "SIGNLANG_PROCESSED_DIR", "processed_dataset"),
                 artifacts=pick(artifacts, "SIGNLANG_ARTIFACTS_DIR", "artifacts"))


def add_path_args(p: argparse.ArgumentParser, dataset=False, processed=False, artifacts=False):
    """Add the standard ``--*-dir`` overrides to a stage's argument parser."""
    if dataset:
        p.add_argument("--dataset-dir", help="raw videos root (default: dataset/)")
    if processed:
        p.add_argument("--processed-dir", help="processed dataset root (default: processed_dataset/)")
    if artifacts:
        p.add_argument("--artifacts-dir", help="training outputs root (default: artifacts/)")
    return p


def paths_from_args(args) -> Paths:
    return get_paths(getattr(args, "dataset_dir", None), getattr(args, "processed_dir", None),
                     getattr(args, "artifacts_dir", None))


# ── Feature selection ─────────────────────────────────────────────────────────
# Curated upper-body pose joints (the signing-relevant ones); legs are dropped —
# they are noisy / extrapolated for a seated, upper-body signer.
#   0 = nose, 11/12 shoulders, 13/14 elbows, 15/16 wrists, 17-22 hand-root points.
UPPER_POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
# MediaPipe left<->right symmetric joints (used by the horizontal-flip augment).
POSE_FLIP_PAIRS = [(11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]


@dataclass
class FeatureConfig:
    """How a raw (T, 1692) clip becomes a fixed (L, D) model input."""
    seq_len: int = 64                 # L: fixed temporal length after resampling
    use_pose: bool = True
    use_hands: bool = True
    use_face: bool = False            # face = 1434/1692 raw dims; off by default
    pose_idx: tuple = tuple(UPPER_POSE_IDX)
    use_pose_visibility: bool = True  # keep the pose visibility channel
    face_idx: tuple = ()              # subset when use_face (empty = all 478)
    add_velocity: bool = True         # append first-difference (motion) features
    add_masks: bool = True            # append per-modality presence channels
    normalize: bool = True            # centre on mid-shoulder, scale by shoulder width
    fallback_scale: float = 0.25      # used when shoulder width can't be measured


@dataclass
class ModelConfig:
    arch: str = "bigru"               # "bigru" | "transformer"
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.4
    pool: str = "attention"           # "attention" | "mean" | "last"
    n_heads: int = 4                  # transformer only
    ff_dim: int = 256                 # transformer only


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 16
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    folds: int = 5
    seed: int = 1337
    warmup_epochs: int = 5
    min_lr: float = 1e-5
    early_stop_patience: int = 35
    grad_clip: float = 5.0
    ema_decay: float = 0.999          # final fit (finetune) only
    # augmentation (training split only)
    aug: bool = True
    aug_rotate_deg: float = 13.0
    aug_scale: float = 0.12
    aug_translate: float = 0.06
    aug_noise: float = 0.012
    aug_time_warp: float = 0.15
    aug_frame_dropout: float = 0.07
    aug_flip_prob: float = 0.5


@dataclass
class Config:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def default_config() -> Config:
    """A fresh config (never mutate a shared global)."""
    return Config()


def to_dict(cfg: Config) -> dict:
    return {"feature": asdict(cfg.feature), "model": asdict(cfg.model), "train": asdict(cfg.train)}


def feature_config_from_dict(d: dict) -> FeatureConfig:
    """Rebuild a FeatureConfig from saved model meta (JSON turns tuples into lists)."""
    known = {k: v for k, v in d.items() if k in FeatureConfig.__dataclass_fields__}
    fc = FeatureConfig(**known)
    fc.pose_idx = tuple(fc.pose_idx)
    fc.face_idx = tuple(fc.face_idx)
    return fc


def model_config_from_dict(d: dict) -> ModelConfig:
    return ModelConfig(**{k: v for k, v in d.items() if k in ModelConfig.__dataclass_fields__})
