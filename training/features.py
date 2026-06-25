"""
Shared preprocessing:  raw (T, 1692) landmarks  ->  fixed-length (L, D) features.
=================================================================================
THE SAME transform is used by training AND by file / live inference, so what the
model trains on is exactly what it sees at serve time (no train/serve skew).  It
has NO fitted state — the geometry is fully deterministic, and the model carries
an input LayerNorm to absorb any residual scaling.

It is split into three stages so that data augmentation (training only) can act
on the geometry in the *normalised coordinate space*:

    raw (T,1692)  --geometric-->  coords (T,P,3), vis (T,Vp), present (T,m)
                  --[augment]-->  (training only; see data.py)
                  --assemble--->  (L, D)   resample + velocity + concat

Why augment in normalised space?  `geometric` centres on the mid-shoulder and
divides by shoulder width, so scale / translation applied to the *raw* frame
would be normalised away.  Augmentation must therefore happen between the two
stages.  `to_features()` is the no-augment convenience wrapper used at inference.

Per-clip normalisation: subtract the mid-shoulder point and divide by the
shoulder width, using ONE clip-level reference (median over frames where the pose
is present) -> translation + scale invariance, robust to per-frame dropouts.
Absent blocks are re-zeroed AFTER normalisation, so a missing hand stays at the
origin instead of being shoved to -origin/scale (a "ghost" hand).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    BLOCK, FeatureConfig, LEFT_SHOULDER, N_FACE, N_HAND, N_POSE,
    POSE_FLIP_PAIRS, RIGHT_SHOULDER,
)

EPS = 1e-6


# ── Feature layout spec (point segments + flip permutations) ──────────────────
@dataclass
class FeatureSpec:
    pose_idx: list
    face_idx: list
    seg: dict                  # modality -> (start, end) in POINT-index space
    n_points: int
    point_perm: np.ndarray     # horizontal-flip permutation over points
    vis_perm: np.ndarray       # flip permutation over pose-visibility columns
    mask_perm: np.ndarray      # flip permutation over presence-mask columns
    n_vis: int
    n_mask: int

    @property
    def coord_dim(self) -> int:
        return self.n_points * 3

    def point_present(self, present_modal: dict, T: int) -> np.ndarray:
        """Expand per-modality presence (T,) booleans to a per-point mask (T, P)."""
        pp = np.zeros((T, self.n_points), np.float32)
        for name, (s, e) in self.seg.items():
            pp[:, s:e] = present_modal[name][:, None]
        return pp


def build_spec(cfg: FeatureConfig) -> FeatureSpec:
    pose_idx = list(cfg.pose_idx) if cfg.pose_idx else list(range(N_POSE))
    face_idx = list(cfg.face_idx) if cfg.face_idx else list(range(N_FACE))
    n_pose = len(pose_idx) if cfg.use_pose else 0
    n_hand = N_HAND if cfg.use_hands else 0
    n_face = len(face_idx) if cfg.use_face else 0

    seg, off = {}, 0
    if cfg.use_pose:
        seg["pose"] = (off, off + n_pose); off += n_pose
    if cfg.use_hands:
        seg["left_hand"] = (off, off + n_hand); off += n_hand
        seg["right_hand"] = (off, off + n_hand); off += n_hand
    if cfg.use_face:
        seg["face"] = (off, off + n_face); off += n_face
    n_points = off

    # Horizontal-flip permutation over points: swap L/R hands, swap symmetric
    # pose joints; face left untouched (it's off by default).
    perm = np.arange(n_points)
    if cfg.use_pose:
        pos = {idx: i for i, idx in enumerate(pose_idx)}
        s0 = seg["pose"][0]
        for a, b in POSE_FLIP_PAIRS:
            if a in pos and b in pos:
                ia, ib = s0 + pos[a], s0 + pos[b]
                perm[ia], perm[ib] = ib, ia
    if cfg.use_hands:
        (l0, l1), (r0, _) = seg["left_hand"], seg["right_hand"]
        for i in range(n_hand):
            perm[l0 + i], perm[r0 + i] = r0 + i, l0 + i

    # Visibility columns mirror the pose points; flip them the same way.
    vis_perm = np.arange(n_pose)
    if cfg.use_pose and cfg.use_pose_visibility:
        pos = {idx: i for i, idx in enumerate(pose_idx)}
        for a, b in POSE_FLIP_PAIRS:
            if a in pos and b in pos:
                vis_perm[pos[a]], vis_perm[pos[b]] = pos[b], pos[a]
    n_vis = n_pose if (cfg.use_pose and cfg.use_pose_visibility) else 0

    # Presence-mask columns, in emission order; flip swaps the two hands.
    mask_names = []
    if cfg.use_pose:
        mask_names.append("pose")
    if cfg.use_hands:
        mask_names += ["left_hand", "right_hand"]
    if cfg.use_face:
        mask_names.append("face")
    mask_perm = np.arange(len(mask_names))
    if cfg.use_hands:
        li, ri = mask_names.index("left_hand"), mask_names.index("right_hand")
        mask_perm[li], mask_perm[ri] = ri, li

    return FeatureSpec(pose_idx, face_idx, seg, n_points, perm,
                       vis_perm[:n_vis], mask_perm, n_vis, len(mask_names))


# ── Raw block slicing ─────────────────────────────────────────────────────────
def split_blocks(all_vec: np.ndarray) -> dict:
    """(T, 1692) -> dict of (T, N, C): pose(33,4), face(478,3), hands(21,3) each."""
    T = all_vec.shape[0]
    return {
        "pose":       all_vec[:, slice(*BLOCK["pose"])].reshape(T, N_POSE, 4),
        "face":       all_vec[:, slice(*BLOCK["face"])].reshape(T, N_FACE, 3),
        "left_hand":  all_vec[:, slice(*BLOCK["left_hand"])].reshape(T, N_HAND, 3),
        "right_hand": all_vec[:, slice(*BLOCK["right_hand"])].reshape(T, N_HAND, 3),
    }


def _present(block: np.ndarray) -> np.ndarray:
    """(T, N, >=3) -> (T,) bool: detected if the x/y/z block isn't all-zero."""
    flat = block[..., :3].reshape(block.shape[0], -1)
    return np.abs(flat).sum(axis=1) > EPS


def _reference(pose: np.ndarray, present_pose: np.ndarray, fallback_scale: float):
    """Clip-level origin (mid-shoulder xyz) and scale (median shoulder width)."""
    sh_l = pose[:, LEFT_SHOULDER, :3]
    sh_r = pose[:, RIGHT_SHOULDER, :3]
    mid = 0.5 * (sh_l + sh_r)                                  # (T, 3)
    width = np.linalg.norm((sh_l - sh_r)[:, :2], axis=1)      # (T,) 2D shoulder width
    valid = present_pose & (width > EPS)
    if valid.any():
        origin = np.median(mid[valid], axis=0).astype(np.float32)
        scale = float(np.median(width[valid]))
    else:
        origin = np.zeros(3, np.float32)
        scale = fallback_scale
    if scale < EPS:
        scale = fallback_scale
    return origin, float(scale)


# ── Temporal resampling ───────────────────────────────────────────────────────
def _resample_linear(arr: np.ndarray, L: int) -> np.ndarray:
    """(T, D) -> (L, D) by linear interpolation along time."""
    T = arr.shape[0]
    if arr.shape[1] == 0:
        return np.zeros((L, 0), np.float32)
    if T == 0:
        return np.zeros((L, arr.shape[1]), np.float32)
    if T == 1:
        return np.repeat(arr, L, axis=0).astype(np.float32)
    src = np.linspace(0.0, 1.0, T)
    dst = np.linspace(0.0, 1.0, L)
    out = np.empty((L, arr.shape[1]), np.float32)
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(dst, src, arr[:, j])
    return out


def _resample_nearest(arr: np.ndarray, L: int) -> np.ndarray:
    """(T, D) -> (L, D) nearest-neighbour (keeps 0/1 masks crisp)."""
    T = arr.shape[0]
    if arr.shape[1] == 0:
        return np.zeros((L, 0), np.float32)
    if T == 0:
        return np.zeros((L, arr.shape[1]), np.float32)
    idx = np.clip(np.round(np.linspace(0, T - 1, L)).astype(int), 0, T - 1)
    return arr[idx].astype(np.float32)


# ── Stage 1: geometry (normalise + select) ────────────────────────────────────
def geometric(all_vec: np.ndarray, cfg: FeatureConfig, spec: FeatureSpec):
    """raw (T,1692) -> coords (T,P,3), vis (T,Vp), present_modal {name:(T,)}.
    Normalised + re-zeroed; NOT yet resampled (so augmentation can act here)."""
    all_vec = np.asarray(all_vec, np.float32)
    if all_vec.ndim != 2 or all_vec.shape[1] != 1692:
        raise ValueError(f"expected (T, 1692) raw landmarks, got {all_vec.shape}")
    b = split_blocks(all_vec)
    pose, face, lh, rh = b["pose"], b["face"], b["left_hand"], b["right_hand"]
    present = {k: _present(b[k]) for k in ("pose", "face", "left_hand", "right_hand")}

    if cfg.normalize:
        origin, scale = _reference(pose, present["pose"], cfg.fallback_scale)
    else:
        origin, scale = np.zeros(3, np.float32), 1.0

    def norm_xyz(block, present_mask):
        xyz = (block[..., :3] - origin) / scale
        return (xyz * present_mask[:, None, None]).astype(np.float32)

    point_arrays = []
    if cfg.use_pose:
        point_arrays.append(norm_xyz(pose[:, spec.pose_idx, :], present["pose"]))
    if cfg.use_hands:
        point_arrays.append(norm_xyz(lh, present["left_hand"]))
        point_arrays.append(norm_xyz(rh, present["right_hand"]))
    if cfg.use_face:
        point_arrays.append(norm_xyz(face[:, spec.face_idx, :], present["face"]))

    T = all_vec.shape[0]
    coords = (np.concatenate(point_arrays, axis=1) if point_arrays
              else np.zeros((T, 0, 3), np.float32))

    if cfg.use_pose and cfg.use_pose_visibility:
        vis = (pose[:, spec.pose_idx, 3] * present["pose"][:, None]).astype(np.float32)
    else:
        vis = np.zeros((T, 0), np.float32)

    present_modal = {name: present[name].astype(np.float32) for name in spec.seg}
    return coords, vis, present_modal


# ── Stage 2: assemble (resample + velocity + concat) ──────────────────────────
def assemble(coords, vis, present_modal, cfg: FeatureConfig, spec: FeatureSpec):
    """coords (T,P,3), vis (T,Vp), present_modal -> (L, D) float32."""
    L = cfg.seq_len
    T = coords.shape[0]
    flat = coords.reshape(T, -1)
    coords_L = _resample_linear(flat, L)
    parts = [coords_L]

    if spec.n_vis:
        parts.append(_resample_linear(vis, L))

    if cfg.add_velocity:
        vel = np.zeros_like(coords_L)
        vel[1:] = coords_L[1:] - coords_L[:-1]
        parts.append(vel)

    if cfg.add_masks:
        cols = [present_modal[name] for name in spec.seg]
        masks = np.stack(cols, axis=1).astype(np.float32) if cols else np.zeros((T, 0), np.float32)
        parts.append(_resample_nearest(masks, L))

    return np.concatenate(parts, axis=1).astype(np.float32)


# ── Public inference entrypoint (no augmentation) ─────────────────────────────
def to_features(all_vec: np.ndarray, cfg: FeatureConfig, spec: FeatureSpec | None = None):
    """raw (T, 1692) -> (L, D) float32.  Inference path (training adds augment)."""
    if spec is None:
        spec = build_spec(cfg)
    coords, vis, present = geometric(all_vec, cfg, spec)
    return assemble(coords, vis, present, cfg, spec)


def feature_dim(cfg: FeatureConfig, spec: FeatureSpec | None = None) -> int:
    """D produced for this config (no real data needed)."""
    if spec is None:
        spec = build_spec(cfg)
    dummy = np.zeros((2, 1692), np.float32)
    dummy[:, LEFT_SHOULDER * 4:LEFT_SHOULDER * 4 + 2] = [0.4, 0.5]
    dummy[:, RIGHT_SHOULDER * 4:RIGHT_SHOULDER * 4 + 2] = [0.6, 0.5]
    return to_features(dummy, cfg, spec).shape[1]


if __name__ == "__main__":
    from config import CONFIG, LANDMARKS_ALL
    fc = CONFIG.feature
    spec = build_spec(fc)
    print(f"spec: n_points={spec.n_points} coord_dim={spec.coord_dim} "
          f"n_vis={spec.n_vis} n_mask={spec.n_mask} segs={spec.seg}")
    sample = next(LANDMARKS_ALL.glob("*/*.npy"), None) if LANDMARKS_ALL.exists() else None
    if sample is not None:
        raw = np.load(sample)
        feats = to_features(raw, fc, spec)
        print(f"clip {sample.parent.name}/{sample.stem}: raw {raw.shape} -> feats {feats.shape}")
        # flip-consistency: flipping twice == identity
        c, v, p = geometric(raw, fc, spec)
        c2 = c[:, spec.point_perm, :].copy(); c2[..., 0] *= -1
        c3 = c2[:, spec.point_perm, :].copy(); c3[..., 0] *= -1
        print(f"flip is an involution: {np.allclose(c, c3)}")
    print(f"feature_dim(cfg) = {feature_dim(fc, spec)}")
