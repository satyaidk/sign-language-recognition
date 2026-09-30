"""
Landmark drawing + visual-verification videos.
==============================================
Renders a clean, readable skeleton:

  * a curated UPPER-BODY skeleton (torso + shoulder -> elbow -> hand arm chains),
    drawn only where both joints are confidently visible;
  * the dense face mesh (which owns the face, so pose face points are skipped);
  * both hands, labelled with the signer's anatomical side.

Arms are drawn to the matched HAND wrist (optimal 2-hand pairing), so they never
cross into an "X".  The same drawing code renders the live viewer, the dataset
skeleton videos, the verification overlays, the observation overlays and the
prediction overlays — every picture of the data comes from one implementation.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from signlang.landmarks import layout as L
from signlang.landmarks.hands import anatomical_label, match_hands_to_sides
from signlang.landmarks.mediapipe_compat import load_solutions

mp = load_solutions()
from mediapipe.framework.formats import landmark_pb2  # noqa: E402

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# ── Skeleton definition ───────────────────────────────────────────────────────
TORSO_CONNECTIONS = [
    (L.LEFT_SHOULDER, L.RIGHT_SHOULDER),
    (L.LEFT_SHOULDER, L.LEFT_HIP),
    (L.RIGHT_SHOULDER, L.RIGHT_HIP),
    (L.LEFT_HIP, L.RIGHT_HIP),
]
TORSO_JOINTS = [L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP]
ARM_SIDES = {   # side -> (shoulder, elbow, wrist)
    "Left": (L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST),
    "Right": (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST),
}

# ── Tunables ──────────────────────────────────────────────────────────────────
VIS_THRESH = 0.5          # min visibility to draw a joint / bone
HOLD_FRAMES = 12          # viewer: frames to keep pose/face through occlusion
STALE_ALPHA = 0.45        # opacity of held (stale) drawings
HAND_LINE_GAP_PX = 22     # the arm line stops short of the hand cluster

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
FACE_DOT = mp_drawing.DrawingSpec(color=(0, 220, 255), thickness=1, circle_radius=1)
FACE_CONN = mp_drawing.DrawingSpec(color=(0, 140, 255), thickness=1)
HAND_DOT = mp_drawing.DrawingSpec(color=(0, 255, 80), thickness=1, circle_radius=2)
HAND_CONN = mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=2)
TORSO_COLOR = (200, 200, 200)
ARM_LINE_COLOR = (0, 200, 255)
ARM_SHOULDER_COLOR = (255, 80, 0)
ARM_ELBOW_COLOR = (255, 180, 0)
ARM_LINE_THICKNESS = 2
JOINT_RADIUS = 4


# ── Conversions ───────────────────────────────────────────────────────────────
def to_proto(points) -> "landmark_pb2.NormalizedLandmarkList":
    """(N, 3|4) array -> MediaPipe proto (visibility = column 3, else 1.0)."""
    points = np.asarray(points)
    proto = landmark_pb2.NormalizedLandmarkList()
    has_vis = points.shape[1] >= 4
    for r in points:
        p = proto.landmark.add()
        p.x, p.y, p.z = float(r[0]), float(r[1]), float(r[2])
        p.visibility = float(r[3]) if has_vis else 1.0
    return proto


def lm_px(landmark, w: int, h: int):
    return int(landmark.x * w), int(landmark.y * h)


def _shorten_toward(p_from, p_to, gap_px):
    """Pull ``p_to`` back toward ``p_from`` by ``gap_px``; None if closer than the gap."""
    vx, vy = p_to[0] - p_from[0], p_to[1] - p_from[1]
    dist = math.hypot(vx, vy)
    if dist <= gap_px:
        return None
    s = (dist - gap_px) / dist
    return int(p_from[0] + vx * s), int(p_from[1] + vy * s)


def hand_wrists_by_side(pose_proto, hand_protos, w: int, h: int) -> dict:
    """Match each drawn hand to a body arm -> ``{"Left"|"Right": wrist_px}``."""
    if pose_proto is None or not hand_protos:
        return {}
    lw = lm_px(pose_proto.landmark[L.LEFT_WRIST], w, h)
    rw = lm_px(pose_proto.landmark[L.RIGHT_WRIST], w, h)
    wrists = [lm_px(p.landmark[0], w, h) for p in hand_protos]
    out = {}
    for idx, side in match_hands_to_sides(wrists, lw, rw).items():
        out[side] = wrists[idx]
    return out


# ── Drawing primitives ────────────────────────────────────────────────────────
def draw_upper_body(frame, pose_proto, stale, hand_wrist_by_side, show_labels=True):
    """Torso + per-side arm chains; bones only between confidently visible joints."""
    if pose_proto is None:
        return
    h, w = frame.shape[:2]
    lm = pose_proto.landmark
    overlay = frame.copy()

    def visible(i):
        return lm[i].visibility >= VIS_THRESH

    for a, b in TORSO_CONNECTIONS:
        if visible(a) and visible(b):
            cv2.line(overlay, lm_px(lm[a], w, h), lm_px(lm[b], w, h),
                     TORSO_COLOR, ARM_LINE_THICKNESS, cv2.LINE_AA)
    for i in TORSO_JOINTS:
        if visible(i):
            cv2.circle(overlay, lm_px(lm[i], w, h), JOINT_RADIUS - 1, TORSO_COLOR, -1, cv2.LINE_AA)

    for side, (sh, el, wr) in ARM_SIDES.items():
        if not (visible(sh) and visible(el)):
            continue
        p_sh, p_el = lm_px(lm[sh], w, h), lm_px(lm[el], w, h)
        cv2.line(overlay, p_sh, p_el, ARM_LINE_COLOR, ARM_LINE_THICKNESS, cv2.LINE_AA)
        # Prefer the (more accurate) hand wrist; fall back to the pose wrist.
        if side in hand_wrist_by_side:
            end = _shorten_toward(p_el, hand_wrist_by_side[side], HAND_LINE_GAP_PX)
        elif visible(wr):
            end = lm_px(lm[wr], w, h)
        else:
            end = None
        if end is not None:
            cv2.line(overlay, p_el, end, ARM_LINE_COLOR, ARM_LINE_THICKNESS, cv2.LINE_AA)
        cv2.circle(overlay, p_sh, JOINT_RADIUS, ARM_SHOULDER_COLOR, -1, cv2.LINE_AA)
        cv2.circle(overlay, p_el, JOINT_RADIUS, ARM_ELBOW_COLOR, -1, cv2.LINE_AA)
        if show_labels:
            cv2.putText(overlay, "Shoulder", (p_sh[0] + 10, p_sh[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ARM_LINE_COLOR, 1, cv2.LINE_AA)
            cv2.putText(overlay, "Elbow", (p_el[0] + 10, p_el[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ARM_LINE_COLOR, 1, cv2.LINE_AA)

    alpha = STALE_ALPHA if stale else 1.0
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_face(frame, face_protos, stale, show_indices=False):
    if not face_protos:
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for face in face_protos:
        mp_drawing.draw_landmarks(
            overlay, face, mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            overlay, face, mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=FACE_DOT, connection_drawing_spec=FACE_CONN)
        mp_drawing.draw_landmarks(
            overlay, face, mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style())
        if show_indices:
            for i, p in enumerate(face.landmark):
                if i % 20 == 0:
                    cv2.putText(overlay, str(i), lm_px(p, w, h),
                                cv2.FONT_HERSHEY_PLAIN, 0.6, (200, 200, 255), 1)
    alpha = STALE_ALPHA if stale else 1.0
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_hands(frame, hands, mirrored, show_indices=False):
    """``hands``: list of ``(proto, raw_label)``.  Pass ``mirrored=True`` when the
    labels are already anatomical side names (e.g. hands loaded from a dataset)."""
    h, w = frame.shape[:2]
    for proto, raw_label in hands:
        if proto is None:
            continue
        mp_drawing.draw_landmarks(frame, proto, mp_hands.HAND_CONNECTIONS,
                                  landmark_drawing_spec=HAND_DOT, connection_drawing_spec=HAND_CONN)
        wx, wy = lm_px(proto.landmark[0], w, h)
        cv2.putText(frame, f"{anatomical_label(raw_label, mirrored)} hand", (wx - 30, wy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150), 2)
        if show_indices:
            for i, p in enumerate(proto.landmark):
                px, py = lm_px(p, w, h)
                cv2.putText(frame, str(i), (px + 4, py - 4),
                            cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 0), 1)


def overlay_stats(frame, lines):
    """White-on-shadow text block in the top-left corner."""
    for i, line in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 3)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)


# ── Saved arrays -> pictures ──────────────────────────────────────────────────
def _frame_present(arrays, masks, name, fi, slot=None) -> bool:
    a = arrays.get(name)
    if a is None or fi >= a.shape[0]:
        return False
    if masks is not None and name in masks:
        m = masks[name][fi]
        return bool(m[slot] if slot is not None else m)
    block = a[fi] if slot is None else a[fi, slot]
    return bool(np.abs(block[..., :3]).sum() > L.EPS)


def draw_saved_landmarks(frame, arrays: dict, masks: dict | None, fi: int) -> None:
    """Draw frame ``fi`` of saved arrays (pose/face/hands) onto ``frame``.

    ``masks`` may be ``None``; presence is then read from the data itself.
    """
    h, w = frame.shape[:2]
    pose = to_proto(arrays["pose"][fi]) if _frame_present(arrays, masks, "pose", fi) else None
    hands = [(to_proto(arrays["hands"][fi, slot]), side)
             for slot, side in ((0, "Left"), (1, "Right"))
             if _frame_present(arrays, masks, "hands", fi, slot)]
    wrists = hand_wrists_by_side(pose, [p for p, _ in hands], w, h)
    if pose is not None:                                  # back -> front: skeleton, face, hands
        draw_upper_body(frame, pose, False, wrists, show_labels=False)
    if _frame_present(arrays, masks, "face", fi):
        draw_face(frame, [to_proto(arrays["face"][fi])], False)
    if hands:
        draw_hands(frame, hands, mirrored=True)          # labels are already anatomical


def render_skeleton_video(arrays: dict, masks: dict | None, size, fps: float, out_path) -> bool:
    """Black-background skeleton video rendered FROM the saved arrays."""
    W, H = size
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not vw.isOpened():
        return False
    T = max((a.shape[0] for a in arrays.values()), default=0)
    try:
        for fi in range(T):
            canvas = np.zeros((H, W, 3), np.uint8)
            draw_saved_landmarks(canvas, arrays, masks, fi)
            cv2.putText(canvas, f"{fi + 1}/{T}", (8, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
            vw.write(canvas)
    finally:
        vw.release()
    return True


def render_side_by_side(video_path, arrays: dict, masks: dict | None, out_path, width: int = 640,
                        titles=("original + saved landmarks", "skeleton from .npy")) -> int | None:
    """[original frame + saved landmarks | black skeleton] video; returns frames written."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 < fps <= 120 else 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(width * 9 / 16)
    OW, OH = width, int(H * width / W)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OW * 2, OH))
    T = max((a.shape[0] for a in arrays.values()), default=0)
    fi = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (OW, OH))
            black = np.zeros_like(frame)
            if fi < T:
                for target in (frame, black):
                    draw_saved_landmarks(target, arrays, masks, fi)
            for target, title in zip((frame, black), titles):
                cv2.putText(target, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)
            vw.write(np.hstack([frame, black]))
            fi += 1
    finally:
        cap.release()
        vw.release()
    return fi
