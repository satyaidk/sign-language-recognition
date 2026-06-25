"""
STAGE 4 — Verify the model on VIDEO FILES (the safe check before going live).
=============================================================================
This is the model-level analogue of the dataset's overlay-video QA: it runs the
trained model on a whole clip, then renders an overlay video with the landmarks
drawn on the original frames AND the predicted sign shown as a subtitle, so you
can SEE whether the recognised word matches what the signer is doing.

It produces:
  - a whole-clip prediction (+ top-k),
  - a sliding-window "subtitle" timeline (sign per time window),
  - an overlay .mp4 and a .json under training/artifacts/predictions/.

Run:
    python infer_video.py --video good/good_03          # a dataset source clip
    python infer_video.py --path "C:/some/clip.mp4"     # any video file
    python infer_video.py --video hello/hello --backend torch --no-overlay
    python infer_video.py --scan-dataset --limit 3      # batch sanity check

The same video->raw->features->model path is used by infer_live.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET, PREDICTIONS_DIR, PROJECT_ROOT, ensure_dirs  # noqa: E402
from predictor import SignPredictor, video_to_clip  # noqa: E402
from utils import save_json  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
import extract_dataset as ed  # noqa: E402
import landmark_detector as ld  # noqa: E402

CONF_THRESH = 0.45   # below this a window is shown as uncertain ("...")


# ── Draw saved landmarks onto an original frame (same look as verify overlay) ──
def draw_frame(frame, arrays, masks, fi):
    if "pose" in arrays and fi < arrays["pose"].shape[0] and masks["pose"][fi]:
        pose_proto = ed._proto(arrays["pose"][fi], with_vis=True)
    else:
        pose_proto = None
    hands = []
    if "hands" in arrays and fi < arrays["hands"].shape[0]:
        for slot, side in ((0, "Left"), (1, "Right")):
            if masks["hands"][fi, slot]:
                hands.append((ed._proto(arrays["hands"][fi, slot], False), side))
    wrist_by_side = {}
    if pose_proto is not None and hands:
        H, W = frame.shape[:2]
        lw = ld._lm_px(pose_proto.landmark[ld.LEFT_WRIST], W, H)
        rw = ld._lm_px(pose_proto.landmark[ld.RIGHT_WRIST], W, H)
        wrists = [ld._lm_px(p.landmark[0], W, H) for p, _ in hands]
        for idx, s in ld.match_hands_to_sides(wrists, lw, rw).items():
            wrist_by_side[s] = wrists[idx]
    if pose_proto is not None:
        ld.draw_upper_body(frame, pose_proto, False, wrist_by_side, show_labels=False)
    if "face" in arrays and fi < arrays["face"].shape[0] and arrays["face"][fi].any():
        ld.draw_face(frame, [ed._proto(arrays["face"][fi], False)], False)
    if hands:
        ld.draw_hands(frame, hands, mirrored=True)


def _banner(frame, top_text, bottom_text):
    H, W = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (W, 44), (0, 0, 0), -1)
    cv2.rectangle(ov, (0, H - 46), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, top_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (80, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, bottom_text, (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)


# ── Sliding-window "subtitle" timeline ────────────────────────────────────────
def timeline(raw, fps, predictor, window, stride):
    T = raw.shape[0]
    win = min(window, T)
    segs = []
    starts = list(range(0, max(1, T - win + 1), stride)) or [0]
    if starts[-1] != T - win and T > win:
        starts.append(T - win)
    for a in starts:
        b = min(a + win, T)
        out = predictor.predict(raw[a:b])
        name = out["name"] if out["conf"] >= CONF_THRESH else "..."
        segs.append({"start_frame": a, "end_frame": b,
                     "start_s": round(a / fps, 2), "end_s": round(b / fps, 2),
                     "sign": name, "conf": round(out["conf"], 3)})
    return segs


def sign_at_frame(segs, fi):
    cur = None
    for s in segs:
        if s["start_frame"] <= fi:
            cur = s
    return cur


def render_overlay(video_path, clip, whole, segs, out_path, true_label=None):
    arrays, masks, fps = clip["arrays"], clip["masks"], clip["fps"]
    cap = cv2.VideoCapture(str(video_path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    T = clip["T"]
    fi = 0
    top = f"PRED: {whole['name']}  ({whole['conf']:.2f})"
    if true_label:
        top += f"   [true: {true_label}]"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi < T:
            draw_frame(frame, arrays, masks, fi)
        seg = sign_at_frame(segs, fi)
        live = f"now: {seg['sign']} ({seg['conf']:.2f})" if seg else "now: ..."
        _banner(frame, top, live)
        vw.write(frame)
        fi += 1
    cap.release()
    vw.release()
    return fi


def resolve(args):
    """Return (video_path, true_label, stem)."""
    if args.path:
        p = Path(args.path)
        return p, None, p.stem
    cls, stem = args.video.replace("\\", "/").split("/", 1)
    p = DATASET / cls / f"{stem}.mp4"
    if not p.exists():
        cand = list((DATASET / cls).glob(f"{stem}.*"))
        if cand:
            p = cand[0]
    return p, cls, stem


def run_one(args, predictor, video_path, true_label, stem):
    clip = video_to_clip(video_path, proc_width=args.proc_width)
    if clip is None:
        print(f"[infer] cannot open {video_path}")
        return None
    whole = predictor.predict(clip["raw"])
    segs = timeline(clip["raw"], clip["fps"], predictor, args.window, args.stride)
    mark = "" if true_label is None else (" OK" if whole["name"] == true_label else " <-- MISMATCH")
    print(f"\n[{stem}]  T={clip['T']}  fps={clip['fps']:.0f}")
    print(f"  whole-clip : {whole['name']}  conf={whole['conf']:.3f}"
          + (f"   (true={true_label}){mark}" if true_label else ""))
    print("  top-3      : " + ", ".join(f"{n}={c:.2f}" for n, c in whole["topk"]))
    if len(segs) > 1:
        print("  timeline   : " + " | ".join(f"{s['start_s']}-{s['end_s']}s:{s['sign']}" for s in segs))

    result = {"stem": stem, "true_label": true_label, "prediction": whole["name"],
              "conf": whole["conf"], "topk": whole["topk"], "timeline": segs,
              "correct": (None if true_label is None else whole["name"] == true_label)}
    if not args.no_overlay:
        ensure_dirs()
        out_mp4 = PREDICTIONS_DIR / f"{stem}_pred.mp4"
        n = render_overlay(video_path, clip, whole, segs, out_mp4, true_label)
        save_json(result, PREDICTIONS_DIR / f"{stem}_pred.json")
        print(f"  overlay    : {out_mp4}  ({n} frames)")
    return result


def main():
    args = parse_args()
    predictor = SignPredictor(backend=args.backend, int8=args.int8)
    print(f"[infer] backend={predictor.backend} classes={len(predictor.classes)} "
          f"L={predictor.seq_len} D={predictor.in_dim}")

    if args.scan_dataset:
        from data import load_index
        items = load_index()
        by_class = {}
        for it in items:
            by_class.setdefault(it["class"], []).append(it)
        correct = total = 0
        for cls, lst in by_class.items():
            for it in lst[:args.limit]:
                vp = DATASET / cls / f"{it['stem']}.mp4"
                if not vp.exists():
                    continue
                r = run_one(args, predictor, vp, cls, it["stem"])
                if r:
                    total += 1
                    correct += int(r["correct"])
        if total:
            print(f"\n[scan] {correct}/{total} correct  ({correct / total:.1%}) "
                  f"on the first {args.limit}/class source videos")
        return

    video_path, true_label, stem = resolve(args)
    if not Path(video_path).exists():
        sys.exit(f"[ERROR] video not found: {video_path}")
    run_one(args, predictor, video_path, true_label, stem)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 4: run the model on a video file + overlay.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--video", help="dataset clip as <class>/<stem>, e.g. good/good_03")
    g.add_argument("--path", help="path to any video file")
    p.add_argument("--scan-dataset", action="store_true", help="batch-test source videos")
    p.add_argument("--limit", type=int, default=2, help="clips/class for --scan-dataset")
    p.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    p.add_argument("--int8", action="store_true", help="use the int8 ONNX model")
    p.add_argument("--no-overlay", action="store_true", help="skip rendering the overlay video")
    p.add_argument("--window", type=int, default=48, help="sliding-window length (frames)")
    p.add_argument("--stride", type=int, default=12, help="sliding-window stride (frames)")
    p.add_argument("--proc-width", type=int, default=0, help="downscale width for detection (0=full)")
    if len(sys.argv) == 1:
        p.print_help(); sys.exit(0)
    return p.parse_args()


if __name__ == "__main__":
    main()
