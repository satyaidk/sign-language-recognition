"""
Shared fixtures.  Nothing here needs the real (private) dataset or a trained
model: clips are synthetic (T, 1692) landmark arrays with a known structure, and
the processed dataset is built in a temporary folder.
"""
from __future__ import annotations

import numpy as np
import pytest

from signlang.config import get_paths
from signlang.landmarks.layout import (
    BLOCK, LEFT_SHOULDER, N_POSE, POSE_DIM, RAW_TOTAL, RIGHT_SHOULDER,
)
from signlang.utils import save_json, write_csv_rows


def _mediapipe_ok() -> bool:
    try:
        from signlang.landmarks.mediapipe_compat import load_solutions
        load_solutions()
        return True
    except ImportError:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip MediaPipe-backed tests cleanly when the pinned MediaPipe is unavailable."""
    if _mediapipe_ok():
        return
    skip = pytest.mark.skip(reason="mediapipe==0.10.9 (legacy solutions API) not available")
    for item in items:
        if "mediapipe" in item.keywords:
            item.add_marker(skip)


def make_clip(rng, T: int, right_only: bool = True, shift: float = 0.0) -> np.ndarray:
    """A plausible raw clip: upper-body pose with fixed shoulders, a face, a right
    hand (and optionally a left hand).  ``shift`` moves the right hand, which is
    how the synthetic classes differ."""
    raw = np.zeros((T, RAW_TOTAL), np.float32)
    pose = rng.uniform(0.3, 0.7, (T, N_POSE, POSE_DIM)).astype(np.float32)
    pose[..., 3] = 0.9
    pose[:, LEFT_SHOULDER, :2] = [0.6, 0.5]
    pose[:, RIGHT_SHOULDER, :2] = [0.4, 0.5]
    raw[:, slice(*BLOCK["pose"])] = pose.reshape(T, -1)
    s, e = BLOCK["face"]
    raw[:, s:e] = rng.uniform(0.4, 0.6, (T, e - s))
    s, e = BLOCK["right_hand"]
    raw[:, s:e] = rng.uniform(0.3, 0.7, (T, e - s)) + shift
    if not right_only:
        s, e = BLOCK["left_hand"]
        raw[:, s:e] = rng.uniform(0.3, 0.7, (T, e - s))
    return raw


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def clip(rng):
    return make_clip(rng, 60)


def build_synthetic_dataset(base):
    """A processed dataset of 3 separable synthetic classes (10 clips each) under ``base``."""
    rng = np.random.default_rng(1)
    paths = get_paths(dataset=base / "dataset", processed=base / "processed", artifacts=base / "artifacts")
    classes = {"alpha": 0, "beta": 1, "gamma": 2}
    rows = []
    for cls, label in classes.items():
        (paths.landmarks_all / cls).mkdir(parents=True)
        for i in range(10):
            stem = f"{cls}_{i:02d}"
            clip = make_clip(rng, int(rng.integers(30, 80)), shift=0.12 * label)
            np.save(paths.landmarks_all / cls / f"{stem}.npy", clip)
            rows.append({"class": cls, "label": label, "stem": stem, "T": len(clip),
                         "verdict": "PASS", "all_npy": f"landmarks_all/{cls}/{stem}.npy", "skeleton": ""})
    save_json(classes, paths.classes_json)
    write_csv_rows(paths.manifest, rows)
    return paths


@pytest.fixture
def synthetic_dataset(tmp_path):
    return build_synthetic_dataset(tmp_path)


class FakeLandmark:
    """Duck-types a MediaPipe NormalizedLandmark."""

    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class FakeLandmarkList:
    def __init__(self, points):
        self.landmark = [FakeLandmark(*p) for p in points]
