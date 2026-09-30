"""
Landmark extraction — ONE code path for datasets and live recognition.
======================================================================
``LandmarkExtractor`` owns the MediaPipe models (FaceMesh, Hands, Pose) and the
One-Euro smoothing state, and turns one BGR frame into one frame of landmarks.

  * ``extract_video()``  (dataset building, file inference) creates a fresh
    extractor per clip, so tracking state never leaks between clips.
  * live recognition keeps ONE extractor alive and feeds it every camera frame,
    so a finished sign is already a landmark array — nothing is re-decoded.

Because both paths call the same ``process()``, a clip is converted to exactly
the same ``(T, 1692)`` layout whether it comes from a file or a webcam.

Design notes
  * Smoothing never back-fills: a frame without a detection stays zero and its
    mask stays 0 (honest data for training).
  * Hands go to anatomical slots (0 = signer's left).  Duplicate handedness
    labels are resolved geometrically (see ``hands.assign_hand_slots``).
  * Modalities the caller does not need (e.g. the face when the model ignores
    it) are skipped entirely and left as zeros, which keeps the layout intact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from signlang.landmarks import layout as L
from signlang.landmarks.hands import assign_hand_slots
from signlang.landmarks.smoothing import HAND_EURO, POSE_EURO, HandStabilizer, PointStabilizer


@dataclass
class ExtractorOptions:
    face: bool = True
    hands: bool = True
    pose: bool = True
    smooth: bool = True
    proc_width: int = 0          # downscale frames wider than this before detection (0 = off)
    mirrored: bool = False       # True only if frames are horizontally flipped (selfie view)
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    @property
    def modalities(self) -> tuple:
        return tuple(m for m, on in (("pose", self.pose), ("face", self.face),
                                     ("hands", self.hands)) if on)


@dataclass
class FrameLandmarks:
    """Landmarks of one frame.  Missing modalities are ``None`` / zero slots."""
    pose: np.ndarray | None = None                  # (33, 4)
    face: np.ndarray | None = None                  # (478, 3)
    hands: np.ndarray = field(default_factory=lambda: np.zeros((2, L.N_HAND, L.HAND_DIM), np.float32))
    hand_mask: np.ndarray = field(default_factory=lambda: np.zeros(2, np.int8))

    def row(self) -> np.ndarray:
        """This frame as one (1692,) row of the combined vector."""
        return L.frame_row(self.pose, self.face,
                           self.hands[0] if self.hand_mask[0] else None,
                           self.hands[1] if self.hand_mask[1] else None)


# ── MediaPipe result -> arrays (pure; unit-tested with fake results) ──────────
def _points(landmarks, dim: int) -> np.ndarray:
    if dim == 4:
        return np.array([(p.x, p.y, p.z, getattr(p, "visibility", 1.0)) for p in landmarks],
                        np.float32)
    return np.array([(p.x, p.y, p.z) for p in landmarks], np.float32)


def pose_from_result(pose_res) -> np.ndarray | None:
    if pose_res is None or not getattr(pose_res, "pose_landmarks", None):
        return None
    return _points(pose_res.pose_landmarks.landmark, L.POSE_DIM)


def face_from_result(face_res) -> np.ndarray | None:
    faces = getattr(face_res, "multi_face_landmarks", None) if face_res is not None else None
    if not faces:
        return None
    return _points(faces[0].landmark, L.FACE_DIM)      # first face only


def hands_from_result(hand_res) -> list:
    """-> list of ``(points (21, 3), raw_label)``."""
    if hand_res is None or not getattr(hand_res, "multi_hand_landmarks", None):
        return []
    return [(_points(lm.landmark, L.HAND_DIM), handed.classification[0].label)
            for lm, handed in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness)]


# ── The extractor ─────────────────────────────────────────────────────────────
class LandmarkExtractor:
    """Persistent MediaPipe models + smoothing: ``process(frame, t) -> FrameLandmarks``."""

    def __init__(self, opts: ExtractorOptions | None = None):
        from signlang.landmarks.mediapipe_compat import load_solutions

        self.opts = opts or ExtractorOptions()
        if not self.opts.modalities:
            raise ValueError("nothing to extract: face, hands and pose are all disabled")
        mp = load_solutions()
        o = self.opts
        conf = dict(min_detection_confidence=o.min_detection_confidence,
                    min_tracking_confidence=o.min_tracking_confidence)
        self.face_model = (mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True, **conf)
            if o.face else None)
        self.hand_model = (mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=1, **conf)
            if o.hands else None)
        self.pose_model = (mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1, smooth_landmarks=True,
            enable_segmentation=False, **conf)
            if o.pose else None)
        self.pose_stab = PointStabilizer(L.N_POSE, POSE_EURO, hold_frames=0)
        self.hand_stab = HandStabilizer(HAND_EURO)

    # context-manager support so models are always released
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        for m in (self.face_model, self.hand_model, self.pose_model):
            if m is not None:
                m.close()
        self.face_model = self.hand_model = self.pose_model = None

    def reset(self) -> None:
        """Forget smoothing state (call between unrelated clips)."""
        self.pose_stab.reset()
        self.hand_stab.reset()

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        w = self.opts.proc_width
        if w and frame.shape[1] > w:
            s = w / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return rgb

    def process(self, frame_bgr: np.ndarray, t: float) -> FrameLandmarks:
        """Detect + smooth one frame.  ``t`` is a monotonic timestamp in seconds."""
        rgb = self._prepare(frame_bgr)
        face_res = self.face_model.process(rgb) if self.face_model else None
        hand_res = self.hand_model.process(rgb) if self.hand_model else None
        pose_res = self.pose_model.process(rgb) if self.pose_model else None
        return self.from_results(face_res, hand_res, pose_res, t)

    def from_results(self, face_res, hand_res, pose_res, t: float) -> FrameLandmarks:
        """Convert raw MediaPipe results into a smoothed ``FrameLandmarks``."""
        out = FrameLandmarks()
        smooth = self.opts.smooth

        pose = pose_from_result(pose_res) if self.opts.pose else None
        if smooth and self.opts.pose:
            pose, _ = self.pose_stab.update(pose, t)
        out.pose = pose

        if self.opts.face:
            out.face = face_from_result(face_res)

        if self.opts.hands:
            hands = hands_from_result(hand_res)
            if smooth:
                hands = self.hand_stab.update(hands, t)
            for slot, points in assign_hand_slots(hands, out.pose, self.opts.mirrored):
                out.hands[slot] = points
                out.hand_mask[slot] = 1
        return out


# ── Whole-video extraction ────────────────────────────────────────────────────
def extract_video(video_path, opts: ExtractorOptions | None = None, progress=None) -> dict | None:
    """Run detection over one clip.

    Returns ``{fps, n_frames_read, n_frames_skipped, T, arrays, masks}`` with
    ``arrays`` = pose (T,33,4) / face (T,478,3) / hands (T,2,21,3) for the
    enabled modalities and matching per-frame ``masks``; ``None`` if the video
    cannot be opened.  ``progress(frame_index)`` is called every 300 frames.
    """
    opts = opts or ExtractorOptions()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = src_fps if 1.0 < src_fps <= 120.0 else 30.0

    pose_f, face_f, hand_f = [], [], []
    pose_m, face_m, hand_m = [], [], []
    read = skipped = 0
    try:
        with LandmarkExtractor(opts) as ex:          # fresh models: no cross-clip tracking
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = read / fps
                read += 1
                try:
                    fl = ex.process(frame, t)
                except Exception as exc:              # never lose a whole clip to one frame
                    print(f"      [warn] frame {read - 1} skipped: {exc}")
                    skipped += 1
                    continue
                if opts.pose:
                    pose_f.append(fl.pose if fl.pose is not None
                                  else np.zeros((L.N_POSE, L.POSE_DIM), np.float32))
                    pose_m.append(int(fl.pose is not None))
                if opts.face:
                    face_f.append(fl.face if fl.face is not None
                                  else np.zeros((L.N_FACE, L.FACE_DIM), np.float32))
                    face_m.append(int(fl.face is not None))
                if opts.hands:
                    hand_f.append(fl.hands)
                    hand_m.append(fl.hand_mask.copy())
                if progress is not None and (read - 1) % 300 == 0:
                    progress(read - 1)
    finally:
        cap.release()

    def stack(frames, shape):
        return (np.stack(frames).astype(np.float32) if frames
                else np.zeros((0,) + shape, np.float32))

    arrays, masks = {}, {}
    if opts.pose:
        arrays["pose"] = stack(pose_f, (L.N_POSE, L.POSE_DIM))
        masks["pose"] = np.asarray(pose_m, np.int8)
    if opts.face:
        arrays["face"] = stack(face_f, (L.N_FACE, L.FACE_DIM))
        masks["face"] = np.asarray(face_m, np.int8)
    if opts.hands:
        arrays["hands"] = stack(hand_f, (2, L.N_HAND, L.HAND_DIM))
        masks["hands"] = np.asarray(hand_m, np.int8).reshape(-1, 2)

    T = max((a.shape[0] for a in arrays.values()), default=0)
    return {"fps": fps, "n_frames_read": read, "n_frames_skipped": skipped,
            "T": T, "arrays": arrays, "masks": masks}


def video_to_raw(video_path, opts: ExtractorOptions | None = None) -> dict | None:
    """Extract a clip and return it in the training layout.

    ``{raw (T,1692), arrays, masks, fps, T}``.  Modalities that were skipped are
    zero-filled so the column layout always matches training.
    """
    res = extract_video(video_path, opts)
    if res is None:
        return None
    T = res["T"]
    full = {
        "pose": res["arrays"].get("pose", np.zeros((T, L.N_POSE, L.POSE_DIM), np.float32)),
        "face": res["arrays"].get("face", np.zeros((T, L.N_FACE, L.FACE_DIM), np.float32)),
        "hands": res["arrays"].get("hands", np.zeros((T, 2, L.N_HAND, L.HAND_DIM), np.float32)),
    }
    raw, _ = L.build_all_vector(full, L.MODALITIES)
    return {"raw": raw, "arrays": res["arrays"], "masks": res["masks"],
            "fps": res["fps"], "T": T}
