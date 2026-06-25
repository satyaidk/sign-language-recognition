"""
STAGE 5 — Live translation by chunked record -> process -> delete.
==================================================================
Implements the method requested: grab a few seconds of video, write it to a
TEMP file, run the project's landmark extraction on that recording, classify the
sign, show it like a subtitle, then DELETE the temp file (it is a cache file).
Repeat continuously so it reads as live translation.

Why a temp file instead of a pure in-memory buffer?  It reuses the EXACT,
already-verified `extract_dataset.extract_clip` path (smoothing, anatomical hand
assignment, the same (T,1692) layout) that the dataset and `infer_video.py` use
— so the live result matches the offline result, with no second code path.

Run (webcam):
    python infer_live.py                      # 0 = default camera, ~2.5s chunks
    python infer_live.py --seconds 2.0 --proc-width 480 --int8

Test WITHOUT a camera (simulate "live" from a file — exercises the exact
record->temp->process->delete loop chunk by chunk):
    python infer_live.py --source ../good.mp4 --seconds 2.0
    python infer_live.py --source ../asl.mp4 --no-display

Controls (webcam window): press Q to stop.  Temp recordings live under
training/artifacts/live_tmp/ and are removed right after each chunk is read
(use --keep-temp to inspect them).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LIVE_TMP_DIR, PREDICTIONS_DIR, ensure_dirs  # noqa: E402
from predictor import SignPredictor, video_to_clip  # noqa: E402
from utils import save_json  # noqa: E402

FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


def _banner(frame, lines):
    H, W = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, H - 70), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame, txt, (12, H - 42 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, col, 2, cv2.LINE_AA)


def process_chunk(tmp_path, predictor, conf_thresh, proc_width, keep_temp):
    """Run the verified extraction+model on a temp clip, then delete it."""
    try:
        clip = video_to_clip(tmp_path, proc_width=proc_width)
        if clip is None or clip["T"] < 3:
            return None
        out = predictor.predict(clip["raw"])
        out["T"] = clip["T"]
        # presence: only trust the chunk if hands were actually seen
        hands_seen = float(clip["masks"]["hands"].any(axis=1).mean()) if "hands" in clip["masks"] else 0.0
        out["hands_rate"] = round(hands_seen, 2)
        out["ok"] = out["conf"] >= conf_thresh and hands_seen > 0.15
        return out
    finally:
        if not keep_temp:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ── Webcam: record a chunk to a temp file, with live preview ──────────────────
def run_webcam(args, predictor):
    cap = cv2.VideoCapture(args.device_index)
    if not cap.isOpened():
        sys.exit(f"[ERROR] cannot open camera index {args.device_index}. "
                 f"Use --source <video> to test without a camera.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1.0 < fps <= 120.0 else 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    chunk_frames = max(8, int(args.seconds * fps))
    print(f"[live] camera {args.device_index}  {W}x{H}@{fps:.0f}  chunk={args.seconds}s "
          f"({chunk_frames} frames)  proc_width={args.proc_width}")
    print("[live] press Q in the window to stop.")

    last = ("listening...", (200, 200, 200))
    transcript = []
    stop = False
    while not stop:
        tmp = LIVE_TMP_DIR / f"chunk_{uuid.uuid4().hex[:8]}.mp4"
        vw = cv2.VideoWriter(str(tmp), FOURCC, fps, (W, H))
        for _ in range(chunk_frames):
            ok, frame = cap.read()
            if not ok:
                stop = True
                break
            vw.write(frame)
            disp = cv2.flip(frame, 1)
            _banner(disp, [(f"Sign: {last[0]}", last[1]),
                           ("recording... (Q to quit)", (180, 180, 180))])
            if not args.no_display:
                cv2.imshow("Live sign translation", disp)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    stop = True
                    break
        vw.release()

        out = process_chunk(tmp, predictor, args.conf, args.proc_width, args.keep_temp)
        if out is None:
            last = ("...", (160, 160, 160))
        elif out["ok"]:
            last = (f"{out['name']}  ({out['conf']:.2f})", (80, 255, 120))
            transcript.append({"t": round(time.time(), 1), "sign": out["name"],
                               "conf": round(out["conf"], 3)})
            print(f"[live] -> {out['name']:16s} conf={out['conf']:.2f} "
                  f"hands={out['hands_rate']}  (temp deleted)")
        else:
            last = ("...", (160, 160, 160))
            print(f"[live] -> (uncertain) best={out['name']} conf={out['conf']:.2f} "
                  f"hands={out['hands_rate']}  (temp deleted)")

    cap.release()
    cv2.destroyAllWindows()
    _save_transcript(transcript)


# ── Simulate "live" from a video file (testable without a camera) ─────────────
def run_source(args, predictor):
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] cannot open source video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1.0 < fps <= 120.0 else 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
    chunk_frames = max(8, int(args.seconds * fps))
    print(f"[live-sim] source={args.source}  {W}x{H}@{fps:.0f}  chunk={args.seconds}s "
          f"({chunk_frames} frames)")

    transcript, chunk_id, done = [], 0, False
    while not done:
        tmp = LIVE_TMP_DIR / f"chunk_{uuid.uuid4().hex[:8]}.mp4"
        vw = cv2.VideoWriter(str(tmp), FOURCC, fps, (W, H))
        n = 0
        for _ in range(chunk_frames):
            ok, frame = cap.read()
            if not ok:
                done = True
                break
            vw.write(frame)
            n += 1
            if not args.no_display:
                disp = cv2.resize(frame, (min(W, 960), int(H * min(W, 960) / W)))
                cv2.imshow("Live sign translation (sim)", disp)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    done = True
                    break
        vw.release()
        if n < 3:
            try:
                os.remove(tmp)
            except OSError:
                pass
            break

        t0 = chunk_id * args.seconds
        out = process_chunk(tmp, predictor, args.conf, args.proc_width, args.keep_temp)
        chunk_id += 1
        if out and out["ok"]:
            transcript.append({"t_start": round(t0, 1), "sign": out["name"],
                               "conf": round(out["conf"], 3)})
            print(f"[{t0:5.1f}s] {out['name']:16s} conf={out['conf']:.2f} "
                  f"hands={out['hands_rate']} T={out['T']}  (temp deleted)")
        else:
            best = out["name"] if out else "-"
            conf = out["conf"] if out else 0.0
            print(f"[{t0:5.1f}s] (uncertain)      best={best} conf={conf:.2f}  (temp deleted)")

    cap.release()
    cv2.destroyAllWindows()
    _save_transcript(transcript)


def _save_transcript(transcript):
    leftover = list(LIVE_TMP_DIR.glob("chunk_*.mp4"))
    for f in leftover:                       # safety net: never leave cache behind
        try:
            f.unlink()
        except OSError:
            pass
    if transcript:
        path = PREDICTIONS_DIR / "live_transcript.json"
        save_json({"transcript": transcript}, path)
        print(f"\n[live] transcript ({len(transcript)} signs) -> {path}")
        print("[live] " + "  ".join(s["sign"] for s in transcript))
    else:
        print("\n[live] no confident signs recognised.")


def main():
    args = parse_args()
    ensure_dirs()
    predictor = SignPredictor(backend=args.backend, int8=args.int8)
    print(f"[live] backend={predictor.backend} classes={len(predictor.classes)} "
          f"L={predictor.seq_len} conf_thresh={args.conf}")
    if args.source:
        run_source(args, predictor)
    else:
        run_webcam(args, predictor)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 5: live chunked sign translation.")
    p.add_argument("--seconds", type=float, default=2.5, help="chunk length to record/process")
    p.add_argument("--source", help="simulate live from this video file (no camera needed)")
    p.add_argument("--device-index", type=int, default=0, help="webcam index")
    p.add_argument("--proc-width", type=int, default=480, help="downscale width for live speed (0=full)")
    p.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    p.add_argument("--int8", action="store_true", help="use the int8 ONNX model")
    p.add_argument("--conf", type=float, default=0.5, help="min confidence to emit a sign")
    p.add_argument("--no-display", action="store_true", help="headless (no preview window)")
    p.add_argument("--keep-temp", action="store_true", help="don't delete temp chunk files")
    return p.parse_args()


if __name__ == "__main__":
    main()
