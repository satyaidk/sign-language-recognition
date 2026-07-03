"""
Central configuration for the sign-language recognition training pipeline.
============================================================================
Everything tunable lives here so the rest of the pipeline (features, data,
model, training, inference) reads ONE source of truth.  `train.py` /
`finetune.py` can override individual fields from the command line.

Paths are resolved relative to the PROJECT ROOT (two levels up from this
`src/training/` folder), so the pipeline never depends on the current working
directory.

This file is pure data + path logic — it imports nothing heavy, so it is safe
to import from anywhere (including the live inference script).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_DIR = Path(__file__).resolve().parent          # .../sign new exp/src/training
SRC_DIR = TRAIN_DIR.parent                           # .../sign new exp/src
PROJECT_ROOT = SRC_DIR.parent                        # .../sign new exp
DATAPREP_DIR = SRC_DIR / "dataprep"                  # stage-1 extraction code

PROCESSED = PROJECT_ROOT / "data" / "processed_dataset"
DATASET = PROJECT_ROOT / "data" / "dataset"
LANDMARKS_ALL = PROCESSED / "landmarks_all"
MANIFEST = PROCESSED / "manifest.csv"
CLASSES_JSON = PROCESSED / "classes.json"
FEATURE_LAYOUT_JSON = PROCESSED / "FEATURE_LAYOUT.json"

ARTIFACTS = PROJECT_ROOT / "artifacts"       # ALL pipeline outputs (mostly gitignored)
CKPT_DIR = ARTIFACTS / "checkpoints"         # per-fold + final model weights
METRICS_DIR = ARTIFACTS / "metrics"          # JSON metrics + PNG plots
EXPORT_DIR = ARTIFACTS / "exported"          # ONNX models + model_meta.json
CACHE_DIR = ARTIFACTS / "cache"              # feature cache + live temp clips
PREDICTIONS_DIR = ARTIFACTS / "predictions"  # overlay videos + prediction JSON
LIVE_TMP_DIR = ARTIFACTS / "live_tmp"        # transient live recordings (auto-deleted)
LOG_DIR = ARTIFACTS / "logs"

# ── Raw landmark layout (mirrors data/processed_dataset/FEATURE_LAYOUT.json) ───────
RAW_TOTAL = 1692
BLOCK = {                          # name -> (start, end) into the (T, 1692) vector
    "pose":       (0, 132),        # 33 x (x, y, z, visibility)
    "face":       (132, 1566),     # 478 x (x, y, z)
    "left_hand":  (1566, 1629),    # 21 x (x, y, z)
    "right_hand": (1629, 1692),    # 21 x (x, y, z)
}
N_POSE, N_FACE, N_HAND = 33, 478, 21

# Curated upper-body pose joints (the signing-relevant ones).  Legs / lower body
# are dropped — they are noisy or extrapolated for a seated upper-body signer.
#   0  = nose,  11/12 = shoulders, 13/14 = elbows, 15/16 = wrists,
#   17-22 = hand-root landmarks (pinky/index/thumb bases).
UPPER_POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
# MediaPipe left<->right symmetric joints (used by the horizontal-flip augment).
POSE_FLIP_PAIRS = [(11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]
# Pose indices used as the spatial-normalisation reference (the shoulders).
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


# ── Configuration dataclasses ─────────────────────────────────────────────────
@dataclass
class FeatureConfig:
    """How a raw (T, 1692) clip becomes a fixed (L, D) model input."""
    seq_len: int = 64                 # L: fixed temporal length after resampling
    use_pose: bool = True
    use_hands: bool = True
    use_face: bool = False            # face = 1434/1692 raw dims; off by default
    pose_idx: tuple = tuple(UPPER_POSE_IDX)
    use_pose_visibility: bool = True  # keep pose per-joint visibility channel
    face_idx: tuple = ()              # subset when use_face (empty = all 478)
    add_velocity: bool = True         # append first-difference (motion) features
    add_masks: bool = True            # append per-modality presence channels
    normalize: bool = True            # center on mid-shoulder, scale by shoulder width
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
    # augmentation (train split only)
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


CONFIG = Config()


# ── Helpers ───────────────────────────────────────────────────────────────────
def ensure_dirs() -> None:
    for d in (ARTIFACTS, CKPT_DIR, METRICS_DIR, EXPORT_DIR, CACHE_DIR,
              PREDICTIONS_DIR, LIVE_TMP_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def to_dict(cfg: Config) -> dict:
    return {"feature": asdict(cfg.feature),
            "model": asdict(cfg.model),
            "train": asdict(cfg.train)}


def feature_config_from_dict(d: dict) -> FeatureConfig:
    """Rebuild a FeatureConfig from a saved model_meta (tuples survive JSON)."""
    fc = FeatureConfig(**d)
    fc.pose_idx = tuple(fc.pose_idx)
    fc.face_idx = tuple(fc.face_idx)
    return fc
