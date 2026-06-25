"""
Dataset Landmark Extractor  (face + hands + pose)
=================================================
Walks a sign-language video dataset laid out as::

    dataset/
        <class_1>/  *.mp4
        <class_2>/  *.mp4
        ...

and, for every clip, runs the SAME MediaPipe detection used by the live
``landmark_detector.py`` viewer, then SAVES the numeric landmarks to ``.npy``
files plus a clean black-background skeleton ``.mp4``.

What it produces (under ``--out``, default ``processed_dataset/``)::

    landmarks_face/<class>/<stem>.npy     (T, 478, 3)    x, y, z
    landmarks_hands/<class>/<stem>.npy    (T, 2, 21, 3)  slot0=Left, slot1=Right
    landmarks_pose/<class>/<stem>.npy     (T, 33, 4)     x, y, z, visibility
    landmarks_all/<class>/<stem>.npy      (T, F)         every enabled part, concatenated
    skeleton_videos/<class>/<stem>.mp4    black bg, skeleton only
    metadata/<class>/<stem>.json          per-clip QA (shapes, detection rates, ranges)
    FEATURE_LAYOUT.json                   exact column layout of the "all" vector
    classes.json                          class -> integer label
    manifest.csv                          one row per clip (paths, frames, verdict)
    extraction_report.json                dataset-wide QA summary

Coordinates are MediaPipe's normalised image coordinates: x, y in [0, 1]
(fraction of width / height), z a relative depth in roughly the same scale.
These ARE the spatial features — translation within the frame is preserved in
x/y, depth in z, and (for pose) per-joint ``visibility`` in [0, 1].

The three modality folders are produced separately AND merged into
``landmarks_all`` in one pass, so nothing is missing between them.  The
``--no-face / --no-hands / --no-pose`` flags drop a modality everywhere
(its folder is not created and it is excluded from ``landmarks_all``).

Because the skeleton video is rendered FROM the saved arrays (not a second
detection pass), watching it is a direct visual proof of what the ``.npy``
file contains.  Use ``verify_landmarks.py`` for the full cross-check.

Usage
-----
    python extract_dataset.py                          # whole dataset/ -> processed_dataset/
    python extract_dataset.py --dataset dataset --out processed_dataset
    python extract_dataset.py --classes hello you      # only some classes
    python extract_dataset.py --limit 2                # first 2 clips per class (quick test)
    python extract_dataset.py --no-face                # skip face everywhere
    python extract_dataset.py --no-skeleton            # don't render skeleton videos
    python extract_dataset.py --no-smooth              # store raw (unfiltered) landmarks
    python extract_dataset.py --proc-width 720         # downscale input for speed
    python extract_dataset.py --skip-existing          # resume an interrupted run
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Reuse the proven detection / smoothing / drawing from the live viewer.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import landmark_detector as ld                       # noqa: E402
from mediapipe.framework.formats import landmark_pb2  # noqa: E402

# ── Landmark shapes ──────────────────────────────────────────────────────────
N_POSE, POSE_DIM = 33, 4     # x, y, z, visibility
N_FACE, FACE_DIM = 478, 3    # x, y, z   (refine_landmarks=True -> includes irises)
N_HAND, HAND_DIM = 21, 3     # x, y, z   (per hand; 2 slots: Left, Right)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}


# ════════════════════════════════════════════════════════════════════════════
# MediaPipe model lifecycle  (recreated per clip -> no cross-clip tracking leak)
# ════════════════════════════════════════════════════════════════════════════

def build_models(opts):
    face = hands = pose = None
    if opts.face:
        face = ld.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    if opts.hands:
        hands = ld.mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    if opts.pose:
        pose = ld.mp_pose.Pose(
            static_image_mode=False, model_complexity=1, smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return face, hands, pose


# ════════════════════════════════════════════════════════════════════════════
# Per-clip extraction
# ════════════════════════════════════════════════════════════════════════════

def _raw_hands(hand_res):
    """Hands without smoothing -> list of (landmark_seq, raw_label)."""
    out = []
    if hand_res and hand_res.multi_hand_landmarks:
        for hand_lm, handed in zip(hand_res.multi_hand_landmarks,
                                   hand_res.multi_handedness):
            out.append((hand_lm.landmark, handed.classification[0].label))
    return out


def extract_clip(video_path, opts, progress=None):
    """
    Run detection over one clip and return a dict of stacked arrays + QA.

    ``progress(frame_index)`` is called every ~300 frames if given (useful for
    long observation videos). Returns None if the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = src_fps if 1.0 < src_fps <= 120.0 else 30.0

    face_model, hand_model, pose_model = build_models(opts)

    # Smoothing state (fresh per clip).  Hold is OFF so missing frames stay
    # honestly empty (mask=0) instead of being back-filled with stale points.
    pose_stab = ld.PointStabilizer(N_POSE, ld.POSE_EURO, hold_frames=0) if opts.smooth else None
    hand_stab = ld.HandStabilizer() if opts.smooth else None

    pose_frames, face_frames, hand_frames = [], [], []
    pose_mask, face_mask, hand_mask = [], [], []
    skipped = 0
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = f / fps  # monotonic per-frame timestamp for the One-Euro filter

        try:
            if opts.proc_width and frame.shape[1] > opts.proc_width:
                scale = opts.proc_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            face_res = face_model.process(rgb) if face_model else None
            hand_res = hand_model.process(rgb) if hand_model else None
            pose_res = pose_model.process(rgb) if pose_model else None
        except Exception as exc:                       # never lose a whole clip
            print(f"      [warn] frame {f} skipped: {exc}")
            skipped += 1
            f += 1
            continue

        # ── Pose ───────────────────────────────────────────────────────────
        p_arr = np.zeros((N_POSE, POSE_DIM), np.float32)
        p_present = 0
        if opts.pose:
            raw_pose = (pose_res.pose_landmarks.landmark
                        if pose_res and pose_res.pose_landmarks else None)
            if opts.smooth:
                proto, _ = pose_stab.update(raw_pose, t)
                seq = proto.landmark if proto is not None else None
            else:
                seq = raw_pose
            if seq is not None:
                for i, lm in enumerate(seq):
                    p_arr[i] = (lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0))
                p_present = 1
            pose_frames.append(p_arr)
            pose_mask.append(p_present)

        # ── Face (first face only) ───────────────────────────────────────────
        fa_arr = np.zeros((N_FACE, FACE_DIM), np.float32)
        fa_present = 0
        if opts.face:
            faces = (face_res.multi_face_landmarks
                     if face_res and face_res.multi_face_landmarks else None)
            if faces:
                for i, lm in enumerate(faces[0].landmark):
                    fa_arr[i] = (lm.x, lm.y, lm.z)
                fa_present = 1
            face_frames.append(fa_arr)
            face_mask.append(fa_present)

        # ── Hands (anatomical Left=slot0, Right=slot1) ───────────────────────
        h_arr = np.zeros((2, N_HAND, HAND_DIM), np.float32)
        h_present = [0, 0]
        if opts.hands:
            items = (hand_stab.update(hand_res, t) if opts.smooth
                     else _raw_hands(hand_res))
            for entry in items:
                seq, raw_label = (entry[0].landmark, entry[1]) if opts.smooth else entry
                if seq is None:
                    continue
                side = ld.anatomical_label(raw_label, mirrored=False)  # video = raw frame
                slot = 0 if side == "Left" else 1
                for i, lm in enumerate(seq):
                    h_arr[slot, i] = (lm.x, lm.y, lm.z)
                h_present[slot] = 1
            hand_frames.append(h_arr)
            hand_mask.append(h_present)

        if progress is not None and f % 300 == 0:
            progress(f)
        f += 1

    cap.release()
    for m in (face_model, hand_model, pose_model):
        if m:
            m.close()

    T = f - skipped if (f - skipped) > 0 else max(
        len(pose_frames), len(face_frames), len(hand_frames))

    result = {"fps": fps, "n_frames_read": f, "n_frames_skipped": skipped,
              "T": 0, "arrays": {}, "masks": {}}

    def _stack(frames, shape):
        return (np.stack(frames).astype(np.float32) if frames
                else np.zeros((0,) + shape, np.float32))

    if opts.pose:
        result["arrays"]["pose"] = _stack(pose_frames, (N_POSE, POSE_DIM))
        result["masks"]["pose"] = np.asarray(pose_mask, np.int8)
    if opts.face:
        result["arrays"]["face"] = _stack(face_frames, (N_FACE, FACE_DIM))
        result["masks"]["face"] = np.asarray(face_mask, np.int8)
    if opts.hands:
        result["arrays"]["hands"] = _stack(hand_frames, (2, N_HAND, HAND_DIM))
        result["masks"]["hands"] = np.asarray(hand_mask, np.int8).reshape(-1, 2)

    result["T"] = max((a.shape[0] for a in result["arrays"].values()), default=0)
    return result


# ════════════════════════════════════════════════════════════════════════════
# "all" feature vector  (concatenation of enabled modalities)
# ════════════════════════════════════════════════════════════════════════════

def build_all_vector(arrays, opts):
    """Flatten + concatenate enabled modalities into (T, F); return (vec, layout)."""
    parts, layout, offset = [], [], 0
    order = []
    if opts.pose:
        order.append(("pose", arrays["pose"], N_POSE * POSE_DIM,
                      "33 pose landmarks x (x,y,z,visibility)"))
    if opts.face:
        order.append(("face", arrays["face"], N_FACE * FACE_DIM,
                      "478 face-mesh landmarks x (x,y,z)"))
    if opts.hands:
        order.append(("left_hand", arrays["hands"][:, 0], N_HAND * HAND_DIM,
                      "21 left-hand landmarks x (x,y,z)"))
        order.append(("right_hand", arrays["hands"][:, 1], N_HAND * HAND_DIM,
                      "21 right-hand landmarks x (x,y,z)"))

    T = max((a.shape[0] for _, a, _, _ in order), default=0)
    for name, a, width, desc in order:
        flat = a.reshape(a.shape[0], -1) if a.shape[0] else np.zeros((T, width), np.float32)
        parts.append(flat)
        layout.append({"name": name, "start": offset, "end": offset + width,
                       "width": width, "desc": desc})
        offset += width

    vec = (np.concatenate(parts, axis=1).astype(np.float32)
           if parts else np.zeros((T, 0), np.float32))
    return vec, {"total_features": offset, "blocks": layout}


# ════════════════════════════════════════════════════════════════════════════
# Skeleton render  (FROM the saved arrays -> direct proof of the .npy contents)
# ════════════════════════════════════════════════════════════════════════════

def _proto(rows, with_vis):
    proto = landmark_pb2.NormalizedLandmarkList()
    for r in rows:
        p = proto.landmark.add()
        p.x, p.y, p.z = float(r[0]), float(r[1]), float(r[2])
        p.visibility = float(r[3]) if with_vis else 1.0
    return proto


def render_skeleton(arrays, masks, size, fps, out_path, opts):
    W, H = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not vw.isOpened():
        return False
    T = max((a.shape[0] for a in arrays.values()), default=0)

    for fi in range(T):
        canvas = np.zeros((H, W, 3), np.uint8)

        pose_proto = None
        if opts.pose and fi < arrays["pose"].shape[0] and masks["pose"][fi]:
            pose_proto = _proto(arrays["pose"][fi], with_vis=True)

        hands_drawn = []
        if opts.hands and fi < arrays["hands"].shape[0]:
            for slot, side in ((0, "Left"), (1, "Right")):
                if masks["hands"][fi, slot]:
                    hands_drawn.append((_proto(arrays["hands"][fi, slot], False), side))

        # Arms reach toward the detected hand wrist, exactly like the viewer.
        hand_wrist_by_side = {}
        if pose_proto is not None and hands_drawn:
            lw = ld._lm_px(pose_proto.landmark[ld.LEFT_WRIST], W, H)
            rw = ld._lm_px(pose_proto.landmark[ld.RIGHT_WRIST], W, H)
            wrists = [ld._lm_px(p.landmark[0], W, H) for p, _ in hands_drawn]
            for idx, side in ld.match_hands_to_sides(wrists, lw, rw).items():
                hand_wrist_by_side[side] = wrists[idx]

        # back -> front: skeleton, face, hands  (same order as the live viewer)
        if pose_proto is not None:
            ld.draw_upper_body(canvas, pose_proto, False, hand_wrist_by_side,
                               show_labels=False)
        if opts.face and fi < arrays["face"].shape[0] and masks["face"][fi]:
            ld.draw_face(canvas, [_proto(arrays["face"][fi], False)], False)
        if hands_drawn:
            ld.draw_hands(canvas, hands_drawn, mirrored=True)  # raw==side -> correct text

        cv2.putText(canvas, f"{fi + 1}/{T}", (8, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
        vw.write(canvas)

    vw.release()
    return True


# ════════════════════════════════════════════════════════════════════════════
# QA on one clip
# ════════════════════════════════════════════════════════════════════════════

def qa_clip(arrays, masks, T, opts):
    """Per-clip detection rates, coordinate ranges and a PASS/WARN/FAIL verdict."""
    qa = {"detection_rate": {}, "ranges": {}, "issues": []}

    def rate(mask):
        return float(mask.mean()) if mask.size else 0.0

    if opts.pose:
        qa["detection_rate"]["pose"] = rate(masks["pose"])
    if opts.face:
        qa["detection_rate"]["face"] = rate(masks["face"])
    if opts.hands:
        qa["detection_rate"]["left_hand"] = rate(masks["hands"][:, 0])
        qa["detection_rate"]["right_hand"] = rate(masks["hands"][:, 1])
        qa["detection_rate"]["any_hand"] = rate((masks["hands"].sum(1) > 0))

    has_nan = has_inf = False
    for name, a in arrays.items():
        if a.size == 0:
            continue
        has_nan |= bool(np.isnan(a).any())
        has_inf |= bool(np.isinf(a).any())
        if a[..., 0].any():
            qa["ranges"][name] = {
                "x_min": float(a[..., 0][a[..., 0] != 0].min(initial=0.0)),
                "x_max": float(a[..., 0].max()),
                "y_min": float(a[..., 1][a[..., 1] != 0].min(initial=0.0)),
                "y_max": float(a[..., 1].max()),
            }

    # Verdict
    if has_nan or has_inf:
        qa["issues"].append("NaN/Inf present")
    if T == 0:
        qa["issues"].append("no frames")
    if opts.pose and qa["detection_rate"].get("pose", 0) < 0.5:
        qa["issues"].append("pose < 50%")
    # NOTE: pose x/y can legitimately exceed [0,1] — MediaPipe extrapolates
    # off-screen joints (e.g. hips below an upper-body frame). Only face/hands,
    # which are reported only when visible, are range-checked here.
    for name in ("face", "hands"):
        r = qa["ranges"].get(name)
        if r and (r["x_max"] > 1.5 or r["y_max"] > 1.5 or r["x_min"] < -0.5 or r["y_min"] < -0.5):
            qa["issues"].append(f"{name} coords far outside [0,1]")

    if T == 0 or has_nan or has_inf:
        qa["verdict"] = "FAIL"
    elif qa["issues"]:
        qa["verdict"] = "WARN"
    else:
        qa["verdict"] = "PASS"
    return qa


# ════════════════════════════════════════════════════════════════════════════
# Dataset walk + orchestration
# ════════════════════════════════════════════════════════════════════════════

def discover(dataset_dir, only_classes):
    classes = {}
    for cls_dir in sorted(p for p in Path(dataset_dir).iterdir() if p.is_dir()):
        if only_classes and cls_dir.name not in only_classes:
            continue
        vids = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if vids:
            classes[cls_dir.name] = vids
    return classes


def main():
    opts = parse_args()
    dataset = Path(opts.dataset)
    if not dataset.is_dir():
        sys.exit(f"[ERROR] dataset folder not found: {dataset}")

    out = Path(opts.out)
    folders = {}
    if opts.face:
        folders["face"] = out / "landmarks_face"
    if opts.hands:
        folders["hands"] = out / "landmarks_hands"
    if opts.pose:
        folders["pose"] = out / "landmarks_pose"
    folders["all"] = out / "landmarks_all"
    meta_dir = out / "metadata"
    skel_dir = out / "skeleton_videos"
    for d in list(folders.values()) + [meta_dir] + ([skel_dir] if not opts.no_skeleton else []):
        d.mkdir(parents=True, exist_ok=True)

    classes = discover(dataset, set(opts.classes) if opts.classes else None)
    if not classes:
        sys.exit(f"[ERROR] no class sub-folders with videos under {dataset}")

    class_index = {c: i for i, c in enumerate(sorted(classes))}
    skel_size = tuple(opts.skeleton_size)

    print(f"[INFO] dataset : {dataset}")
    print(f"[INFO] output  : {out}")
    print(f"[INFO] modalities: face={opts.face} hands={opts.hands} pose={opts.pose}"
          f"  smooth={opts.smooth}  skeleton={not opts.no_skeleton}")
    total = sum(len(v[:opts.limit] if opts.limit else v) for v in classes.values())
    print(f"[INFO] {len(classes)} classes, {total} clips to process\n")

    manifest, report = [], {"clips": [], "verdicts": {"PASS": 0, "WARN": 0, "FAIL": 0}}
    layout_doc = None
    t_start = time.time()
    done = 0

    for cls, vids in classes.items():
        if opts.limit:
            vids = vids[:opts.limit]
        for vp in vids:
            stem = vp.stem
            done += 1
            all_path = folders["all"] / cls / f"{stem}.npy"
            if opts.skip_existing and all_path.exists():
                print(f"[{done}/{total}] skip (exists)  {cls}/{stem}")
                continue

            t0 = time.time()
            res = extract_clip(vp, opts)
            if res is None:
                print(f"[{done}/{total}] FAIL open    {cls}/{stem}")
                report["verdicts"]["FAIL"] += 1
                continue

            arrays, masks, T = res["arrays"], res["masks"], res["T"]

            # Save the separate modality folders.
            for name in ("face", "hands", "pose"):
                if name in folders:
                    (folders[name] / cls).mkdir(parents=True, exist_ok=True)
                    np.save(folders[name] / cls / f"{stem}.npy", arrays[name])

            # Save the combined "all" vector.
            all_vec, layout = build_all_vector(arrays, opts)
            (folders["all"] / cls).mkdir(parents=True, exist_ok=True)
            np.save(all_path, all_vec)
            layout_doc = layout

            # Skeleton video (rendered from the saved arrays).
            skel_path = ""
            if not opts.no_skeleton:
                (skel_dir / cls).mkdir(parents=True, exist_ok=True)
                skel_path = skel_dir / cls / f"{stem}.mp4"
                render_skeleton(arrays, masks, skel_size, res["fps"], skel_path, opts)

            # QA + per-clip metadata.
            qa = qa_clip(arrays, masks, T, opts)
            meta = {
                "class": cls, "label": class_index[cls], "stem": stem,
                "source": str(vp), "source_fps": res["fps"],
                "frames_read": res["n_frames_read"],
                "frames_skipped": res["n_frames_skipped"],
                "T": T, "smoothed": opts.smooth, "proc_width": opts.proc_width,
                "shapes": {n: list(a.shape) for n, a in arrays.items()},
                "all_shape": list(all_vec.shape),
                "masks": {n: m.tolist() for n, m in masks.items()},
                "qa": qa,
            }
            (meta_dir / cls).mkdir(parents=True, exist_ok=True)
            with open(meta_dir / cls / f"{stem}.json", "w") as fh:
                json.dump(meta, fh, indent=2)

            report["verdicts"][qa["verdict"]] += 1
            report["clips"].append({"class": cls, "stem": stem, "T": T,
                                    "verdict": qa["verdict"],
                                    "detection_rate": qa["detection_rate"],
                                    "issues": qa["issues"]})
            manifest.append({
                "class": cls, "label": class_index[cls], "stem": stem,
                "T": T, "verdict": qa["verdict"],
                "all_npy": str(all_path),
                "skeleton": str(skel_path),
            })

            dt = time.time() - t0
            rates = " ".join(f"{k}={v:.0%}" for k, v in qa["detection_rate"].items())
            print(f"[{done}/{total}] {qa['verdict']:4s} {cls}/{stem}  "
                  f"T={T}  {rates}  ({dt:.1f}s)")

    # ── Dataset-level outputs ───────────────────────────────────────────────
    with open(out / "classes.json", "w") as fh:
        json.dump(class_index, fh, indent=2)

    if layout_doc is not None:
        layout_doc["note"] = ("Column layout of landmarks_all/*.npy (T, total_features). "
                              "Normalised image coords: x,y in [0,1], z relative depth.")
        with open(out / "FEATURE_LAYOUT.json", "w") as fh:
            json.dump(layout_doc, fh, indent=2)

    report["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["dataset"] = str(dataset)
    report["modalities"] = {"face": opts.face, "hands": opts.hands, "pose": opts.pose}
    report["smoothed"] = opts.smooth
    report["n_clips"] = len(report["clips"])
    if report["clips"]:
        agg = {}
        for c in report["clips"]:
            for k, v in c["detection_rate"].items():
                agg.setdefault(k, []).append(v)
        report["mean_detection_rate"] = {k: round(sum(v) / len(v), 3)
                                         for k, v in agg.items()}
    with open(out / "extraction_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    if manifest:
        with open(out / "manifest.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
            w.writeheader()
            w.writerows(manifest)

    _write_readme(out, opts, class_index)

    v = report["verdicts"]
    print(f"\n[DONE] {report['n_clips']} clips in {time.time() - t_start:.0f}s  "
          f"-> PASS={v['PASS']} WARN={v['WARN']} FAIL={v['FAIL']}")
    if report.get("mean_detection_rate"):
        print("[DONE] mean detection rate:",
              " ".join(f"{k}={x:.0%}" for k, x in report["mean_detection_rate"].items()))
    print(f"[DONE] verify with:  python verify_landmarks.py --out {out}")


def _write_readme(out, opts, class_index):
    txt = f"""# Processed landmark dataset

Generated by `extract_dataset.py` from `{opts.dataset}`.

## Layout
- `landmarks_pose/<class>/<stem>.npy`  -> (T, 33, 4)  x, y, z, visibility
- `landmarks_face/<class>/<stem>.npy`  -> (T, 478, 3) x, y, z
- `landmarks_hands/<class>/<stem>.npy` -> (T, 2, 21, 3) slot0=Left, slot1=Right; x, y, z
- `landmarks_all/<class>/<stem>.npy`   -> (T, F) all enabled parts concatenated
                                          (see FEATURE_LAYOUT.json for columns)
- `skeleton_videos/<class>/<stem>.mp4` -> black background, skeleton only
- `metadata/<class>/<stem>.json`       -> per-clip QA (shapes, detection rates, masks)
- `classes.json`, `manifest.csv`, `extraction_report.json`

## Coordinates / spatial features
MediaPipe normalised image coordinates: x, y in [0, 1] (fraction of width/height),
z a relative depth, and pose `visibility` in [0, 1].  Missing landmarks in a
frame are stored as zeros and flagged in the metadata `masks`.

## Verify the .npy files
    python verify_landmarks.py --out {out}                       # whole-dataset QA + consistency
    python verify_landmarks.py --out {out} --video <class>/<stem> --dataset {opts.dataset} --overlay
The overlay re-projects the SAVED points back onto the ORIGINAL video, so you can
see they track the signer (the strongest correctness check).

Classes: {", ".join(class_index)}
"""
    with open(out / "README.md", "w") as fh:
        fh.write(txt)


def parse_args():
    p = argparse.ArgumentParser(description="Extract face/hand/pose landmarks from a video dataset.")
    p.add_argument("--dataset", default="dataset", help="root folder of <class>/<video> clips")
    p.add_argument("--out", default="processed_dataset", help="output root folder")
    p.add_argument("--classes", nargs="*", help="only these class names")
    p.add_argument("--limit", type=int, default=0, help="max clips per class (0 = all)")
    p.add_argument("--no-face", action="store_true", help="skip face mesh everywhere")
    p.add_argument("--no-hands", action="store_true", help="skip hands everywhere")
    p.add_argument("--no-pose", action="store_true", help="skip pose everywhere")
    p.add_argument("--no-skeleton", action="store_true", help="don't render skeleton videos")
    p.add_argument("--no-smooth", action="store_true", help="store raw (unfiltered) landmarks")
    p.add_argument("--proc-width", type=int, default=0,
                   help="downscale input to this width before detection (0 = full res)")
    p.add_argument("--skip-existing", action="store_true", help="skip clips already extracted")
    p.add_argument("--skeleton-size", type=int, nargs=2, default=[960, 540],
                   metavar=("W", "H"), help="skeleton video size (default 960 540)")
    a = p.parse_args()
    # Friendly derived flags
    a.face = not a.no_face
    a.hands = not a.no_hands
    a.pose = not a.no_pose
    a.smooth = not a.no_smooth
    if not (a.face or a.hands or a.pose):
        sys.exit("[ERROR] nothing to extract: --no-face --no-hands --no-pose all set")
    return a


if __name__ == "__main__":
    main()
