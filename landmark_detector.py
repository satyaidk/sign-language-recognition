"""
Facial Landmark, Hand Pose & Upper-Body Tracking Detector
==========================================================
Uses Google MediaPipe to detect, smooth and draw:
  - Face Mesh : 478 3D facial landmarks (iris-refined)
  - Hand Pose : 21 3D hand-knuckle coordinates (up to 2 hands)
  - Body Pose : 33 body landmarks -> a curated UPPER-BODY skeleton
                (shoulder line + arms: shoulder -> elbow -> hand wrist)

Why this version is robust
--------------------------
Raw MediaPipe output jitters, flickers and "collides" when joints are
occluded (e.g. a hand passing in front of the shoulder, or hands brought
up to the face).  This module fixes that with proven CV techniques:

  1. One-Euro filtering
     Every tracked point is smoothed with a One-Euro filter: almost no lag
     while moving, very steady when still.  Kills the jitter that makes
     points/lines appear to collide.

  2. Occlusion hold (persistence)
     The pose skeleton and face mesh are kept "attached" for a short window
     (HOLD_FRAMES) when detection momentarily drops behind an obstruction,
     and are drawn faded so you can see they are being held.

  3. Visibility gating
     MediaPipe reports a per-landmark ``visibility``.  A bone is drawn only
     when BOTH ends are confidently visible, so uncertain "back" joints
     (e.g. an occluded shoulder) never draw a stray colliding line.

  4. No overlapping geometry
     Pose face points (indices 0-10) are NOT drawn -- the dense face mesh
     owns the face, so the two never clash.  Only a clean upper-body
     skeleton is drawn, and the arm line stops short of the hand cluster.

  5. Correct hand<->arm matching
     Each hand is matched to the nearest pose arm using an optimal
     2-hand assignment, so the arm lines never form an X-cross.

Mirroring & handedness
----------------------
MediaPipe Hands reports handedness assuming a mirrored (selfie) image.
  - LIVE WEBCAM : frame is mirrored, so the raw label is already correct.
  - VIDEO/IMAGE : frame is shown raw, so the label is swapped in code.
A single anatomical label is used for the on-screen "Left/Right hand" text.

Dependencies
------------
    pip install mediapipe opencv-python

Usage
-----
    python landmark_detectorss.py                     # webcam (mirrored)
    python landmark_detectorss.py --source video.mp4  # video  (raw)
    python landmark_detectorss.py --source image.jpg  # image  (raw)
    python landmark_detectorss.py --no-face/--no-hands/--no-pose
    python landmark_detectorss.py --no-mirror         # don't mirror the webcam
    python landmark_detectorss.py --show-indices      # draw landmark numbers
"""

import argparse
import math
import sys
import time

import cv2

try:
    import mediapipe as mp
    from mediapipe.framework.formats import landmark_pb2
except ImportError:
    sys.exit("[ERROR] mediapipe is required:  pip install mediapipe opencv-python")

# ── MediaPipe namespaces ─────────────────────────────────────────────────────
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh      = mp.solutions.face_mesh
mp_hands          = mp.solutions.hands
mp_pose           = mp.solutions.pose

# ── Pose landmark indices ────────────────────────────────────────────────────
POSE           = mp_pose.PoseLandmark
LEFT_SHOULDER  = POSE.LEFT_SHOULDER.value   # 11
RIGHT_SHOULDER = POSE.RIGHT_SHOULDER.value  # 12
LEFT_ELBOW     = POSE.LEFT_ELBOW.value      # 13
RIGHT_ELBOW    = POSE.RIGHT_ELBOW.value     # 14
LEFT_WRIST     = POSE.LEFT_WRIST.value      # 15
RIGHT_WRIST    = POSE.RIGHT_WRIST.value     # 16
LEFT_HIP       = POSE.LEFT_HIP.value        # 23
RIGHT_HIP      = POSE.RIGHT_HIP.value       # 24

# Curated upper-body torso connections (arms are drawn separately so they can
# reach the hand wrist and never double-up with these lines).
TORSO_CONNECTIONS = [
    (LEFT_SHOULDER,  RIGHT_SHOULDER),
    (LEFT_SHOULDER,  LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP,       RIGHT_HIP),
]
TORSO_JOINTS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

# side -> (shoulder, elbow, wrist) pose indices
ARM_SIDES = {
    "Left":  (LEFT_SHOULDER,  LEFT_ELBOW,  LEFT_WRIST),
    "Right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
}

# ── Tunables ─────────────────────────────────────────────────────────────────
VIS_THRESH       = 0.5     # min landmark visibility to draw a joint/bone
HOLD_FRAMES      = 12      # frames to keep pose/face "attached" through occlusion
STALE_ALPHA      = 0.45    # opacity for held (stale) drawings
HAND_LINE_GAP_PX = 22      # gap so the arm line stops short of the hand cluster

# One-Euro filter parameters (normalised coords, ~30 fps)
POSE_EURO = dict(min_cutoff=1.2, beta=0.30, d_cutoff=1.0)
HAND_EURO = dict(min_cutoff=2.0, beta=0.80, d_cutoff=1.0)
RESET_DIST_NORM = 0.30     # jump bigger than this resets a point's filter

# ── Colours (BGR) ────────────────────────────────────────────────────────────
FACE_DOT  = mp_drawing.DrawingSpec(color=(0, 220, 255), thickness=1, circle_radius=1)
FACE_CONN = mp_drawing.DrawingSpec(color=(0, 140, 255), thickness=1)
HAND_DOT  = mp_drawing.DrawingSpec(color=(0, 255, 80),  thickness=1, circle_radius=2)
HAND_CONN = mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=2)

TORSO_COLOR        = (200, 200, 200)   # light grey torso
ARM_LINE_COLOR     = (0,  200, 255)    # cyan arm
ARM_SHOULDER_COLOR = (255, 80,  0)     # orange shoulder
ARM_ELBOW_COLOR    = (255, 180, 0)     # amber elbow
ARM_LINE_THICKNESS = 2
JOINT_RADIUS       = 4


# ════════════════════════════════════════════════════════════════════════════
# One-Euro filter
# ════════════════════════════════════════════════════════════════════════════

class OneEuroFilter:
    """Adaptive low-pass filter (Casiez et al., 2012) for a single scalar."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta       = float(beta)
        self.d_cutoff   = float(d_cutoff)
        self.x_prev  = None
        self.dx_prev = 0.0
        self.t_prev  = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = t - self.t_prev
        if dt <= 0.0:
            dt = 1e-3
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


# ════════════════════════════════════════════════════════════════════════════
# Point-set stabiliser: One-Euro smoothing + occlusion hold
# ════════════════════════════════════════════════════════════════════════════

class PointStabilizer:
    """
    Smooths a fixed-size set of landmarks and, optionally, holds the last good
    result for a few frames when detection drops (occlusion persistence).

    update(landmarks, t) -> (NormalizedLandmarkList | None, stale: bool)
    """

    def __init__(self, n, euro, hold_frames=0, reset_dist=RESET_DIST_NORM,
                 track_vis=True):
        self.n = n
        self.hold_frames = hold_frames
        self.reset_dist = reset_dist
        self.track_vis = track_vis    # Hands report no visibility -> keep them visible
        self.fx = [OneEuroFilter(**euro) for _ in range(n)]
        self.fy = [OneEuroFilter(**euro) for _ in range(n)]
        self.vis = [None] * n          # smoothed visibility (EMA)
        self.last = None               # last produced proto
        self.miss = 0

    def reset(self):
        for f in self.fx:
            f.reset()
        for f in self.fy:
            f.reset()
        self.vis = [None] * self.n
        self.last = None
        self.miss = 0

    def update(self, landmarks, t):
        if landmarks is None:
            self.miss += 1
            if self.last is not None and self.miss <= self.hold_frames:
                return self.last, True
            self.reset()
            return None, False

        proto = landmark_pb2.NormalizedLandmarkList()
        for i, lm in enumerate(landmarks):
            x, y = lm.x, lm.y
            # Reset the filter on a teleport so it doesn't smear across a jump.
            if self.reset_dist and self.fx[i].x_prev is not None:
                if math.hypot(x - self.fx[i].x_prev, y - self.fy[i].x_prev) > self.reset_dist:
                    self.fx[i].reset()
                    self.fy[i].reset()
            sx, sy = self.fx[i](x, t), self.fy[i](y, t)

            p = proto.landmark.add()
            p.x, p.y = sx, sy
            p.z = getattr(lm, "z", 0.0)
            if self.track_vis:
                v = getattr(lm, "visibility", 1.0)
                self.vis[i] = v if self.vis[i] is None else 0.6 * self.vis[i] + 0.4 * v
                p.visibility = self.vis[i]
            else:
                # MediaPipe Hands report visibility≈0; copying that in would make
                # mp_drawing.draw_landmarks() skip every point. Force them visible.
                p.visibility = 1.0

        self.last = proto
        self.miss = 0
        return proto, False


class HandStabilizer:
    """Holds one PointStabilizer per tracked hand, keyed by MediaPipe's label."""

    def __init__(self):
        self._stabs = {}

    def update(self, hand_results, t):
        """Return a list of (smoothed_proto, raw_label) for this frame."""
        out = []
        if not hand_results or not hand_results.multi_hand_landmarks:
            return out
        seen = {}
        for hand_lm, handedness in zip(hand_results.multi_hand_landmarks,
                                       hand_results.multi_handedness):
            raw = handedness.classification[0].label
            seen[raw] = seen.get(raw, 0) + 1
            key = raw if seen[raw] == 1 else f"{raw}#{seen[raw]}"   # disambiguate
            stab = self._stabs.get(key)
            if stab is None:
                stab = self._stabs[key] = PointStabilizer(
                    21, HAND_EURO, hold_frames=0, track_vis=False)
            proto, _ = stab.update(hand_lm.landmark, t)
            out.append((proto, raw))
        return out


# ── Small helpers ─────────────────────────────────────────────────────────────

def _lm_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


def _shorten_toward(p_from, p_to, gap_px):
    """Pull p_to back toward p_from by gap_px; None if they're closer than gap."""
    vx, vy = p_to[0] - p_from[0], p_to[1] - p_from[1]
    dist = math.hypot(vx, vy)
    if dist <= gap_px:
        return None
    scale = (dist - gap_px) / dist
    return int(p_from[0] + vx * scale), int(p_from[1] + vy * scale)


def anatomical_label(raw_label, mirrored):
    """Map MediaPipe handedness to the person's true side (see module docs)."""
    if mirrored:
        return raw_label
    return "Right" if raw_label == "Left" else "Left"


def match_hands_to_sides(hand_wrists, lw, rw):
    """
    Assign each hand wrist to a body side by proximity to the pose wrists.
    For two hands an optimal assignment is used so the arms never cross (X).
    Returns {hand_index: "Left"|"Right"}.
    """
    def d(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    n = len(hand_wrists)
    if n == 0:
        return {}
    if n == 1:
        return {0: "Left" if d(hand_wrists[0], lw) <= d(hand_wrists[0], rw) else "Right"}

    # Two (or more) hands: optimally pair the first two, nearest-match any extras.
    h0, h1 = hand_wrists[0], hand_wrists[1]
    if d(h0, lw) + d(h1, rw) <= d(h0, rw) + d(h1, lw):
        sides = {0: "Left", 1: "Right"}
    else:
        sides = {0: "Right", 1: "Left"}
    for i in range(2, n):
        sides[i] = "Left" if d(hand_wrists[i], lw) <= d(hand_wrists[i], rw) else "Right"
    return sides


# ════════════════════════════════════════════════════════════════════════════
# Drawing
# ════════════════════════════════════════════════════════════════════════════

def draw_upper_body(frame, pose_proto, stale, hand_wrist_by_side, show_labels=True):
    """
    Draw the curated upper-body skeleton: torso + per-side arm chains.
    Bones draw only when both endpoints are confidently visible; held (stale)
    results are faded.  Arms reach the matched hand wrist (stopping short of it)
    or fall back to the pose wrist.

    ``show_labels`` toggles the "Shoulder"/"Elbow" text (kept on for the live
    viewer; turned off for clean skeleton-only dataset renders).
    """
    if pose_proto is None:
        return
    h, w = frame.shape[:2]
    lm = pose_proto.landmark
    overlay = frame.copy()

    def visible(idx):
        return lm[idx].visibility >= VIS_THRESH

    # Torso
    for a, b in TORSO_CONNECTIONS:
        if visible(a) and visible(b):
            cv2.line(overlay, _lm_px(lm[a], w, h), _lm_px(lm[b], w, h),
                     TORSO_COLOR, ARM_LINE_THICKNESS, cv2.LINE_AA)
    for idx in TORSO_JOINTS:
        if visible(idx):
            cv2.circle(overlay, _lm_px(lm[idx], w, h), JOINT_RADIUS - 1,
                       TORSO_COLOR, -1, cv2.LINE_AA)

    # Arms
    for side, (sh_idx, el_idx, wr_idx) in ARM_SIDES.items():
        if not (visible(sh_idx) and visible(el_idx)):
            continue
        pt_sh = _lm_px(lm[sh_idx], w, h)
        pt_el = _lm_px(lm[el_idx], w, h)
        cv2.line(overlay, pt_sh, pt_el, ARM_LINE_COLOR, ARM_LINE_THICKNESS, cv2.LINE_AA)

        # Prefer the (more accurate) hand wrist; fall back to the pose wrist.
        if side in hand_wrist_by_side:
            endpoint, gap = hand_wrist_by_side[side], HAND_LINE_GAP_PX
        elif visible(wr_idx):
            endpoint, gap = _lm_px(lm[wr_idx], w, h), 0
        else:
            endpoint, gap = None, 0

        if endpoint is not None:
            end = _shorten_toward(pt_el, endpoint, gap) if gap else endpoint
            if end is not None:
                cv2.line(overlay, pt_el, end, ARM_LINE_COLOR, ARM_LINE_THICKNESS, cv2.LINE_AA)

        cv2.circle(overlay, pt_sh, JOINT_RADIUS, ARM_SHOULDER_COLOR, -1, cv2.LINE_AA)
        cv2.circle(overlay, pt_el, JOINT_RADIUS, ARM_ELBOW_COLOR,    -1, cv2.LINE_AA)
        if show_labels:
            cv2.putText(overlay, "Shoulder", (pt_sh[0] + 10, pt_sh[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ARM_LINE_COLOR, 1, cv2.LINE_AA)
            cv2.putText(overlay, "Elbow", (pt_el[0] + 10, pt_el[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ARM_LINE_COLOR, 1, cv2.LINE_AA)

    alpha = STALE_ALPHA if stale else 1.0
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_face(frame, face_list, stale, show_indices=False):
    if not face_list:
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for face_lm in face_list:
        mp_drawing.draw_landmarks(
            overlay, face_lm, mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            overlay, face_lm, mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=FACE_DOT, connection_drawing_spec=FACE_CONN,
        )
        mp_drawing.draw_landmarks(
            overlay, face_lm, mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
        if show_indices:
            for idx, p in enumerate(face_lm.landmark):
                if idx % 20 == 0:
                    cv2.putText(overlay, str(idx), _lm_px(p, w, h),
                                cv2.FONT_HERSHEY_PLAIN, 0.6, (200, 200, 255), 1)

    alpha = STALE_ALPHA if stale else 1.0
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_hands(frame, hands_smoothed, mirrored, show_indices=False):
    """hands_smoothed: list of (NormalizedLandmarkList, raw_label)."""
    h, w = frame.shape[:2]
    for proto, raw_label in hands_smoothed:
        if proto is None:
            continue
        mp_drawing.draw_landmarks(
            frame, proto, mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=HAND_DOT, connection_drawing_spec=HAND_CONN,
        )
        label = anatomical_label(raw_label, mirrored)
        wx, wy = _lm_px(proto.landmark[0], w, h)
        cv2.putText(frame, f"{label} hand", (wx - 30, wy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150), 2)
        if show_indices:
            for idx, p in enumerate(proto.landmark):
                px, py = _lm_px(p, w, h)
                cv2.putText(frame, str(idx), (px + 4, py - 4),
                            cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 0), 1)


def overlay_stats(frame, n_faces, n_hands, pose_state, face_state, mirrored, fps):
    lines = [
        f"FPS   : {fps:4.1f}",
        f"Faces : {n_faces}  {face_state}",
        f"Hands : {n_hands}",
        f"Pose  : {pose_state}",
        f"View  : {'mirrored (webcam)' if mirrored else 'raw (video/image)'}",
        "Press Q to quit",
    ]
    for i, line in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 3)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)


# ════════════════════════════════════════════════════════════════════════════
# Main loop
# ════════════════════════════════════════════════════════════════════════════

def run(source, detect_face=True, detect_hands=True, detect_pose=True,
        mirror_webcam=True, show_indices=False, max_faces=1, max_hands=2):

    is_image = isinstance(source, str) and source.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    is_webcam = isinstance(source, int)
    mirrored = is_webcam and mirror_webcam

    cap = None
    if not is_image:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            sys.exit(f"[ERROR] Cannot open source: {source}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep latency low (best-effort)

    face_model = hand_model = pose_model = None
    if detect_face:
        face_model = mp_face_mesh.FaceMesh(
            static_image_mode=is_image, max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    if detect_hands:
        hand_model = mp_hands.Hands(
            static_image_mode=is_image, max_num_hands=max_hands, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    if detect_pose:
        pose_model = mp_pose.Pose(
            static_image_mode=is_image, model_complexity=1, smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Persistence is only meaningful for streams; images are one-shot.
    hold = 0 if is_image else HOLD_FRAMES
    pose_stab = PointStabilizer(33, POSE_EURO, hold_frames=hold)
    hand_stab = HandStabilizer()
    face_hold = {"list": None, "miss": 0}

    print("[INFO] Running… Press Q (or close the window) to stop.")
    print(f"[INFO] View mode: {'mirrored webcam' if mirrored else 'raw video/image'}")
    print("[INFO] Skeleton: curated upper body (face owned by face mesh).")

    def process_frame(frame, t):
        if mirrored:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        face_res = face_model.process(rgb) if face_model else None
        hand_res = hand_model.process(rgb) if hand_model else None
        pose_res = pose_model.process(rgb) if pose_model else None
        rgb.flags.writeable = True
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # ── Pose: smooth + hold ────────────────────────────────────────────
        raw_pose = pose_res.pose_landmarks.landmark if (
            pose_res and pose_res.pose_landmarks) else None
        pose_proto, pose_stale = pose_stab.update(raw_pose, t)

        # ── Face: hold last list through brief occlusion ───────────────────
        raw_faces = (face_res.multi_face_landmarks
                     if face_res and face_res.multi_face_landmarks else None)
        if raw_faces is not None:
            face_hold["list"], face_hold["miss"] = raw_faces, 0
            face_list, face_stale = raw_faces, False
        elif face_hold["list"] is not None and face_hold["miss"] < hold:
            face_hold["miss"] += 1
            face_list, face_stale = face_hold["list"], True
        else:
            face_hold["list"], face_list, face_stale = None, None, False

        # ── Hands: smooth (no hold -> no ghost hands) ──────────────────────
        hands_smoothed = hand_stab.update(hand_res, t)

        # ── Match hands to body sides for the arm chains ───────────────────
        hand_wrist_by_side = {}
        if pose_proto is not None and hands_smoothed:
            h, w = out.shape[:2]
            lw = _lm_px(pose_proto.landmark[LEFT_WRIST],  w, h)
            rw = _lm_px(pose_proto.landmark[RIGHT_WRIST], w, h)
            wrists = [_lm_px(p.landmark[0], w, h) for p, _ in hands_smoothed if p]
            for idx, side in match_hands_to_sides(wrists, lw, rw).items():
                hand_wrist_by_side[side] = wrists[idx]   # last writer wins on conflict

        # ── Draw: skeleton (back) -> face -> hands (top) ───────────────────
        draw_upper_body(out, pose_proto, pose_stale, hand_wrist_by_side)
        draw_face(out, face_list, face_stale, show_indices)
        draw_hands(out, hands_smoothed, mirrored, show_indices)

        pose_state = ("not found" if pose_proto is None
                      else "HELD (occluded)" if pose_stale else "tracking")
        face_state = ("" if face_list is None
                      else "(HELD)" if face_stale else "")
        return out, len(face_list or []), len(hands_smoothed), pose_state, face_state

    # ── Image mode ────────────────────────────────────────────────────────────
    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            sys.exit(f"[ERROR] Cannot read image: {source}")
        out, nf, nh, ps, fs = process_frame(frame, time.monotonic())
        overlay_stats(out, nf, nh, ps, fs, mirrored, 0.0)
        cv2.imshow("Landmark Detector — image", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_path = source.rsplit(".", 1)[0] + "_landmarks.jpg"
        cv2.imwrite(save_path, out)
        print(f"[INFO] Saved → {save_path}")
        _close_models(face_model, hand_model, pose_model)
        return

    # ── Video / webcam mode ─────────────────────────────────────────────────────
    fps, miss_reads = 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            miss_reads += 1
            if is_webcam and miss_reads <= 30:    # tolerate transient camera hiccups
                continue
            break
        miss_reads = 0

        t0 = time.monotonic()
        try:
            out, nf, nh, ps, fs = process_frame(frame, t0)
        except Exception as exc:                  # never crash on a single bad frame
            print(f"[WARN] frame skipped: {exc}")
            continue

        dt = time.monotonic() - t0
        inst = 1.0 / dt if dt > 0 else 0.0
        fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst   # smoothed FPS
        overlay_stats(out, nf, nh, ps, fs, mirrored, fps)

        cv2.imshow("Landmark Detector", out)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    _close_models(face_model, hand_model, pose_model)
    print("[INFO] Done.")


def _close_models(*models):
    for m in models:
        if m:
            m.close()


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Robust face + hand + upper-body tracking via MediaPipe.")
    p.add_argument("--source", default=0,
                   help="0=webcam, device index, video path, or image path")
    p.add_argument("--no-face",  action="store_true", help="Disable face mesh")
    p.add_argument("--no-hands", action="store_true", help="Disable hand tracking")
    p.add_argument("--no-pose",  action="store_true", help="Disable pose / arm chain")
    p.add_argument("--no-mirror", action="store_true", help="Don't mirror the webcam")
    p.add_argument("--show-indices", action="store_true", help="Draw landmark numbers")
    p.add_argument("--max-faces", type=int, default=1)
    p.add_argument("--max-hands", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    run(
        source        = src,
        detect_face   = not args.no_face,
        detect_hands  = not args.no_hands,
        detect_pose   = not args.no_pose,
        mirror_webcam = not args.no_mirror,
        show_indices  = args.show_indices,
        max_faces     = args.max_faces,
        max_hands     = args.max_hands,
    )
