"""
Raw landmark layout — the single source of truth for array shapes.
==================================================================
Every stage (extraction, verification, training, live inference) agrees on
these shapes, so they live in one dependency-free module.

Per frame, MediaPipe gives:

    pose   33 x (x, y, z, visibility)
    face  478 x (x, y, z)            (468 mesh + 10 iris points, refine_landmarks=True)
    hands   2 x 21 x (x, y, z)       slot 0 = signer's LEFT hand, slot 1 = RIGHT

The combined "all" vector concatenates them in the fixed order
pose | face | left_hand | right_hand  ->  (T, 1692).  A modality that was not
detected in a frame is stored as zeros; presence is recovered from the data
itself (a block is "present" when its x/y/z values are not all zero).
"""
from __future__ import annotations

import numpy as np

# ── Shapes ────────────────────────────────────────────────────────────────────
N_POSE, POSE_DIM = 33, 4
N_FACE, FACE_DIM = 478, 3
N_HAND, HAND_DIM = 21, 3

POSE_WIDTH = N_POSE * POSE_DIM     # 132
FACE_WIDTH = N_FACE * FACE_DIM     # 1434
HAND_WIDTH = N_HAND * HAND_DIM     # 63
RAW_TOTAL = POSE_WIDTH + FACE_WIDTH + 2 * HAND_WIDTH   # 1692

# name -> (start, end) columns of the full (T, 1692) vector
BLOCK = {
    "pose":       (0, POSE_WIDTH),
    "face":       (POSE_WIDTH, POSE_WIDTH + FACE_WIDTH),
    "left_hand":  (POSE_WIDTH + FACE_WIDTH, POSE_WIDTH + FACE_WIDTH + HAND_WIDTH),
    "right_hand": (POSE_WIDTH + FACE_WIDTH + HAND_WIDTH, RAW_TOTAL),
}

# ── MediaPipe Pose indices used across the project ────────────────────────────
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24

MODALITIES = ("pose", "face", "hands")
EPS = 1e-6


# ── Presence ──────────────────────────────────────────────────────────────────
def present(block: np.ndarray) -> np.ndarray:
    """(T, ..., C>=3) -> (T,) bool: True where the x/y/z values are not all zero."""
    block = np.asarray(block)
    if block.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    flat = block[..., :3].reshape(block.shape[0], -1)
    return np.abs(flat).sum(axis=1) > EPS


# ── Combined vector ───────────────────────────────────────────────────────────
def build_all_vector(arrays: dict, modalities=MODALITIES):
    """Concatenate the enabled modalities into (T, F).

    ``arrays`` holds ``pose`` (T,33,4), ``face`` (T,478,3), ``hands`` (T,2,21,3).
    Returns ``(vec, layout)`` where layout documents every column block.
    With all three modalities F == RAW_TOTAL (1692), the layout training expects.
    """
    order = []
    if "pose" in modalities:
        order.append(("pose", arrays["pose"], POSE_WIDTH,
                      "33 pose landmarks x (x,y,z,visibility)"))
    if "face" in modalities:
        order.append(("face", arrays["face"], FACE_WIDTH,
                      "478 face-mesh landmarks x (x,y,z)"))
    if "hands" in modalities:
        order.append(("left_hand", arrays["hands"][:, 0], HAND_WIDTH,
                      "21 left-hand landmarks x (x,y,z)"))
        order.append(("right_hand", arrays["hands"][:, 1], HAND_WIDTH,
                      "21 right-hand landmarks x (x,y,z)"))

    T = max((a.shape[0] for _, a, _, _ in order), default=0)
    parts, blocks, offset = [], [], 0
    for name, a, width, desc in order:
        flat = a.reshape(a.shape[0], -1) if a.shape[0] else np.zeros((T, width), np.float32)
        parts.append(flat)
        blocks.append({"name": name, "start": offset, "end": offset + width,
                       "width": width, "desc": desc})
        offset += width

    vec = (np.concatenate(parts, axis=1).astype(np.float32)
           if parts else np.zeros((T, 0), np.float32))
    return vec, {"total_features": offset, "blocks": blocks}


def split_all_vector(vec: np.ndarray) -> dict:
    """(T, 1692) -> {pose (T,33,4), face (T,478,3), left_hand (T,21,3), right_hand (T,21,3)}."""
    vec = np.asarray(vec)
    if vec.ndim != 2 or vec.shape[1] != RAW_TOTAL:
        raise ValueError(f"expected (T, {RAW_TOTAL}) raw landmarks, got {vec.shape}")
    T = vec.shape[0]
    return {
        "pose":       vec[:, slice(*BLOCK["pose"])].reshape(T, N_POSE, POSE_DIM),
        "face":       vec[:, slice(*BLOCK["face"])].reshape(T, N_FACE, FACE_DIM),
        "left_hand":  vec[:, slice(*BLOCK["left_hand"])].reshape(T, N_HAND, HAND_DIM),
        "right_hand": vec[:, slice(*BLOCK["right_hand"])].reshape(T, N_HAND, HAND_DIM),
    }


def frame_row(pose=None, face=None, left_hand=None, right_hand=None) -> np.ndarray:
    """One frame's landmarks -> a (1692,) row; ``None`` blocks stay zero."""
    row = np.zeros(RAW_TOTAL, np.float32)
    for name, pts in (("pose", pose), ("face", face),
                      ("left_hand", left_hand), ("right_hand", right_hand)):
        if pts is not None:
            s, e = BLOCK[name]
            row[s:e] = np.asarray(pts, np.float32).reshape(-1)
    return row
