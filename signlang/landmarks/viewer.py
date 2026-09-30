"""
Interactive landmark viewer — face mesh, hands and upper-body tracking.
======================================================================
Shows MediaPipe landmarks live on a webcam, a video or an image, using the
project's robust smoothing + drawing:

  1. One-Euro filtering   — no jitter while still, no lag while moving.
  2. Occlusion hold       — pose / face stay attached (faded) for a few frames
                            when detection drops behind an obstruction.
  3. Visibility gating    — a bone is drawn only when both joints are visible.
  4. No overlapping geometry — the face mesh owns the face; only a curated
                            upper-body skeleton is drawn.
  5. Hand <-> arm matching — optimal 2-hand pairing, so arms never cross.

Mirroring: MediaPipe Hands labels handedness assuming a mirrored (selfie) image.
The webcam view is mirrored (labels already correct); video / image input is
shown raw and the labels are swapped (see ``hands.anatomical_label``).

Run:
    python -m signlang view                       # webcam (mirrored)
    python -m signlang view --source clip.mp4     # video (raw)
    python -m signlang view --source photo.jpg    # image (raw)
    python -m signlang view --no-face --show-indices
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2

from signlang.landmarks import drawing as D
from signlang.landmarks import layout as L
from signlang.landmarks.extractor import hands_from_result, pose_from_result
from signlang.landmarks.mediapipe_compat import load_solutions
from signlang.landmarks.smoothing import HAND_EURO, POSE_EURO, HandStabilizer, PointStabilizer

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class _Viewer:
    """Per-frame detect -> smooth/hold -> draw for the viewer."""

    def __init__(self, is_image, mirrored, face, hands, pose, max_faces, max_hands, show_indices):
        mp = load_solutions()
        conf = dict(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_model = (mp.solutions.face_mesh.FaceMesh(
            static_image_mode=is_image, max_num_faces=max_faces, refine_landmarks=True, **conf)
            if face else None)
        self.hand_model = (mp.solutions.hands.Hands(
            static_image_mode=is_image, max_num_hands=max_hands, model_complexity=1, **conf)
            if hands else None)
        self.pose_model = (mp.solutions.pose.Pose(
            static_image_mode=is_image, model_complexity=1, smooth_landmarks=True,
            enable_segmentation=False, **conf)
            if pose else None)
        self.mirrored = mirrored
        self.show_indices = show_indices
        self.hold = 0 if is_image else D.HOLD_FRAMES      # persistence only makes sense for streams
        self.pose_stab = PointStabilizer(L.N_POSE, POSE_EURO, hold_frames=self.hold)
        self.hand_stab = HandStabilizer(HAND_EURO)
        self.face_last, self.face_miss = None, 0

    def close(self):
        for m in (self.face_model, self.hand_model, self.pose_model):
            if m is not None:
                m.close()

    def _faces(self, face_res):
        """All detected faces, held through brief dropouts -> (protos, stale)."""
        faces = getattr(face_res, "multi_face_landmarks", None) if face_res else None
        if faces:
            self.face_last, self.face_miss = list(faces), 0
            return self.face_last, False
        if self.face_last is not None and self.face_miss < self.hold:
            self.face_miss += 1
            return self.face_last, True
        self.face_last = None
        return None, False

    def process(self, frame, t):
        if self.mirrored:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        face_res = self.face_model.process(rgb) if self.face_model else None
        hand_res = self.hand_model.process(rgb) if self.hand_model else None
        pose_res = self.pose_model.process(rgb) if self.pose_model else None
        out = frame.copy()
        h, w = out.shape[:2]

        pose_pts, pose_stale = self.pose_stab.update(pose_from_result(pose_res), t)
        pose_proto = D.to_proto(pose_pts) if pose_pts is not None else None
        faces, face_stale = self._faces(face_res)
        hands = [(D.to_proto(p), lbl) for p, lbl in self.hand_stab.update(hands_from_result(hand_res), t)]
        wrists = D.hand_wrists_by_side(pose_proto, [p for p, _ in hands], w, h)

        D.draw_upper_body(out, pose_proto, pose_stale, wrists)      # back -> front
        D.draw_face(out, faces, face_stale, self.show_indices)
        D.draw_hands(out, hands, self.mirrored, self.show_indices)

        pose_state = ("not found" if pose_proto is None
                      else "HELD (occluded)" if pose_stale else "tracking")
        return out, len(faces or []), len(hands), pose_state, "(HELD)" if face_stale else ""


def run(source=0, face=True, hands=True, pose=True, mirror_webcam=True,
        show_indices=False, max_faces=1, max_hands=2) -> None:
    is_image = isinstance(source, str) and source.lower().endswith(IMAGE_EXTS)
    is_webcam = isinstance(source, int)
    mirrored = is_webcam and mirror_webcam
    viewer = _Viewer(is_image, mirrored, face, hands, pose, max_faces, max_hands, show_indices)

    def stats(nf, nh, ps, fs, fps):
        return [f"FPS   : {fps:4.1f}", f"Faces : {nf}  {fs}", f"Hands : {nh}", f"Pose  : {ps}",
                f"View  : {'mirrored (webcam)' if mirrored else 'raw (video/image)'}",
                "Press Q to quit"]

    try:
        if is_image:
            frame = cv2.imread(source)
            if frame is None:
                sys.exit(f"[ERROR] cannot read image: {source}")
            out, nf, nh, ps, fs = viewer.process(frame, time.monotonic())
            D.overlay_stats(out, stats(nf, nh, ps, fs, 0.0))
            save_path = source.rsplit(".", 1)[0] + "_landmarks.jpg"
            cv2.imwrite(save_path, out)
            print(f"[INFO] saved -> {save_path}")
            cv2.imshow("Landmark viewer — image", out)
            cv2.waitKey(0)
            return

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            sys.exit(f"[ERROR] cannot open source: {source}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[INFO] running ({'mirrored webcam' if mirrored else 'raw video'}) — press Q to stop")
        fps, misses = 0.0, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    misses += 1
                    if is_webcam and misses <= 30:        # tolerate camera hiccups
                        continue
                    break
                misses = 0
                t0 = time.monotonic()
                try:
                    out, nf, nh, ps, fs = viewer.process(frame, t0)
                except Exception as exc:                  # never crash on one bad frame
                    print(f"[WARN] frame skipped: {exc}")
                    continue
                dt = time.monotonic() - t0
                inst = 1.0 / dt if dt > 0 else 0.0
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
                D.overlay_stats(out, stats(nf, nh, ps, fs, fps))
                cv2.imshow("Landmark viewer", out)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
        finally:
            cap.release()
    finally:
        viewer.close()
        cv2.destroyAllWindows()


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="signlang view",
                                description="Robust face + hand + upper-body landmark viewer.")
    p.add_argument("--source", default="0", help="webcam index, video path or image path")
    p.add_argument("--no-face", action="store_true", help="disable the face mesh")
    p.add_argument("--no-hands", action="store_true", help="disable hand tracking")
    p.add_argument("--no-pose", action="store_true", help="disable pose / arm chains")
    p.add_argument("--no-mirror", action="store_true", help="don't mirror the webcam")
    p.add_argument("--show-indices", action="store_true", help="draw landmark numbers")
    p.add_argument("--max-faces", type=int, default=1)
    p.add_argument("--max-hands", type=int, default=2)
    return p.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    src = int(a.source) if str(a.source).isdigit() else a.source
    run(src, face=not a.no_face, hands=not a.no_hands, pose=not a.no_pose,
        mirror_webcam=not a.no_mirror, show_indices=a.show_indices,
        max_faces=a.max_faces, max_hands=a.max_hands)


if __name__ == "__main__":
    main()
