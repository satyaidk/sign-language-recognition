"""
STAGE 4 — Recognise signs in a video FILE (the safe check before going live).
=============================================================================
Runs the trained model on a whole clip and renders an overlay video — landmarks
drawn on the original frames plus the predicted sign as a subtitle — so you can
SEE whether the recognised word matches what the signer does.

Produces
  * a whole-clip prediction (+ top-k),
  * a sliding-window "subtitle" timeline (sign per time window),
  * ``predictions/<stem>_pred.json`` (always) and ``<stem>_pred.mp4`` (unless --no-overlay).

The same extractor -> features -> model path is used by live recognition.

Run:
    python -m signlang video --path "C:/some/clip.mp4"
    python -m signlang video --clip good/good_03            # a clip from dataset/
    python -m signlang video --scan-dataset --limit 2       # batch sanity check over dataset/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from signlang.config import Paths, add_path_args, paths_from_args
from signlang.dataset.extract import discover
from signlang.dataset.verify import find_source_video
from signlang.inference.predictor import SignPredictor, load_predictor
from signlang.landmarks.extractor import video_to_raw
from signlang.utils import parse_clip_ref, save_json

UNCERTAIN = 0.45   # below this confidence a timeline window is shown as "..."


def timeline(raw, fps: float, predictor: SignPredictor, window: int, stride: int) -> list:
    """Sliding-window predictions -> ``[{start_frame, end_frame, start_s, end_s, sign, conf}]``."""
    T = raw.shape[0]
    if T == 0:
        return []
    win = min(window, T)
    starts = list(range(0, T - win + 1, max(1, stride)))
    if starts[-1] != T - win:
        starts.append(T - win)
    segs = []
    for a in starts:
        b = a + win
        out = predictor.predict(raw[a:b])
        segs.append({"start_frame": a, "end_frame": b, "start_s": round(a / fps, 2),
                     "end_s": round(b / fps, 2), "conf": round(out["conf"], 3),
                     "sign": out["name"] if out["conf"] >= UNCERTAIN else "..."})
    return segs


def segment_at(segs: list, fi: int):
    """The latest window that has started by frame ``fi``."""
    cur = None
    for s in segs:
        if s["start_frame"] <= fi:
            cur = s
    return cur


def _banner(frame, top_text, bottom_text):
    H, W = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (W, 44), (0, 0, 0), -1)
    cv2.rectangle(ov, (0, H - 46), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, top_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, bottom_text, (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def render_overlay(video_path, clip, whole, segs, out_path, true_label=None) -> int:
    from signlang.landmarks.drawing import draw_saved_landmarks

    cap = cv2.VideoCapture(str(video_path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), clip["fps"], (W, H))
    top = f"PRED: {whole['name']}  ({whole['conf']:.2f})" + (f"   [true: {true_label}]" if true_label else "")
    fi = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fi < clip["T"]:
                draw_saved_landmarks(frame, clip["arrays"], clip["masks"], fi)
            seg = segment_at(segs, fi)
            _banner(frame, top, f"now: {seg['sign']} ({seg['conf']:.2f})" if seg else "now: ...")
            vw.write(frame)
            fi += 1
    finally:
        cap.release()
        vw.release()
    return fi


def run_one(video_path: Path, predictor: SignPredictor, paths: Paths, args, true_label=None) -> dict | None:
    clip = video_to_raw(video_path, predictor.extractor_options(proc_width=args.proc_width))
    if clip is None or clip["T"] == 0:
        print(f"[video] cannot read frames from {video_path}")
        return None
    stem = Path(video_path).stem
    whole = predictor.predict(clip["raw"])
    segs = timeline(clip["raw"], clip["fps"], predictor, args.window, args.stride)
    mark = "" if true_label is None else ("  OK" if whole["name"] == true_label else "  <-- MISMATCH")
    print(f"\n[{stem}]  T={clip['T']}  fps={clip['fps']:.0f}")
    print(f"  whole-clip : {whole['name']}  conf={whole['conf']:.3f}"
          + (f"   (true={true_label}){mark}" if true_label else ""))
    print("  top-3      : " + ", ".join(f"{n}={c:.2f}" for n, c in whole["topk"]))
    if len(segs) > 1:
        print("  timeline   : " + " | ".join(f"{s['start_s']}-{s['end_s']}s:{s['sign']}" for s in segs))

    result = {"stem": stem, "source": str(video_path), "true_label": true_label,
              "prediction": whole["name"], "conf": whole["conf"], "topk": whole["topk"],
              "timeline": segs, "correct": None if true_label is None else whole["name"] == true_label}
    paths.predictions.mkdir(parents=True, exist_ok=True)
    save_json(result, paths.predictions / f"{stem}_pred.json")      # always saved (was overlay-only)
    if not args.no_overlay:
        out_mp4 = paths.predictions / f"{stem}_pred.mp4"
        n = render_overlay(video_path, clip, whole, segs, out_mp4, true_label)
        print(f"  overlay    : {out_mp4}  ({n} frames)")
    return result


def scan_dataset(predictor: SignPredictor, paths: Paths, args) -> tuple:
    """Run the first ``--limit`` clips of every class in dataset/ -> (correct, total)."""
    if not paths.dataset.is_dir():
        raise SystemExit(f"[ERROR] dataset folder not found: {paths.dataset}")
    correct = total = 0
    for cls, videos in discover(paths.dataset).items():
        for vp in videos[:args.limit]:
            r = run_one(vp, predictor, paths, args, true_label=cls)
            if r:
                total += 1
                correct += int(r["correct"])
    if total:
        print(f"\n[scan] {correct}/{total} correct ({correct / total:.1%}) on the first {args.limit}/class videos"
              "  (in-sample if these clips were used for training)")
    return correct, total


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="signlang video", description="Stage 4: recognise signs in a video file.")
    add_path_args(p, dataset=True, artifacts=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--path", help="any video file")
    g.add_argument("--clip", help="a dataset clip as <class>/<stem>, e.g. good/good_03")
    g.add_argument("--scan-dataset", action="store_true", help="batch-test the dataset/ videos")
    p.add_argument("--limit", type=int, default=2, help="clips per class for --scan-dataset")
    p.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    p.add_argument("--int8", action="store_true", help="use the int8 ONNX model if exported")
    p.add_argument("--no-overlay", action="store_true", help="skip rendering the overlay video")
    p.add_argument("--window", type=int, default=48, help="timeline window length (frames)")
    p.add_argument("--stride", type=int, default=12, help="timeline window stride (frames)")
    p.add_argument("--proc-width", type=int, default=0, help="downscale width for detection (0 = full)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    paths = paths_from_args(args)
    predictor = load_predictor(paths, backend=args.backend, int8=args.int8)
    print(f"[video] backend={predictor.backend} classes={len(predictor.classes)} "
          f"L={predictor.seq_len} D={predictor.in_dim}")
    if args.scan_dataset:
        scan_dataset(predictor, paths, args)
        return
    if args.clip:
        try:
            cls, stem = parse_clip_ref(args.clip)
        except ValueError as exc:
            sys.exit(f"[ERROR] {exc}")
        video = find_source_video(paths.dataset, cls, stem)
        if video is None:
            sys.exit(f"[ERROR] video for {cls}/{stem} not found under {paths.dataset}")
        run_one(video, predictor, paths, args, true_label=cls)
        return
    video = Path(args.path)
    if not video.exists():
        sys.exit(f"[ERROR] video not found: {video}")
    run_one(video, predictor, paths, args)


if __name__ == "__main__":
    main()
