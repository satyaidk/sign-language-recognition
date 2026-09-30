"""
STAGE 5 — Live translation by MOTION-GATED record -> process -> delete.
=======================================================================
Same cache method as before — grab video, write a TEMP file, run the project's
verified landmark extraction on it, classify, show it like a subtitle, then
DELETE the temp file — but the *recording is no longer a fixed 2 s chunk*.

What changed and why
--------------------
The old loop recorded fixed 2.5 s windows.  That had three problems you hit
while testing:

  - a fixed window cuts a sign in half, so a window held the tail of one sign +
    the start of the next, and the model saw a blend;
  - the model has no "idle" class, so a window where you were just resting was
    still force-classified — a still hand kept matching the PREVIOUS sign, so the
    old word "stuck" even after you changed signs;
  - repeating one sign printed the same word many times.

Now a sign is recorded for *exactly as long as you sign it*:

  - `MotionSegmenter` (see segmenter.py) watches a cheap per-frame motion signal.
    It OPENS a recording when your hands start moving and CLOSES it when you go
    still — the gap between signs becomes an explicit LISTENING state instead of
    being classified.
  - the closed segment is classified with `predict_robust` (whole-clip + sliding
    windows averaged) and only emitted if it clears BOTH a confidence threshold
    and a top1-top2 `margin` (so look-alike signs that tie are shown as "?").
  - `SignDebouncer` collapses an immediately-repeated sign to one word.

The extraction + model path is byte-for-byte the one infer_video.py uses, so a
live result matches the offline result.

Run (webcam):
    python infer_live.py                         # 0 = default camera
    python infer_live.py --proc-width 480 --int8
    python infer_live.py --margin 0.2 --still 0.6 # stricter / longer end-of-sign

Test WITHOUT a camera (simulate "live" from a file — exercises the exact
record->temp->process->delete loop, motion-gated):
    python infer_live.py --source ../good.mp4
    python infer_live.py --source ../good.mp4 --no-display

Controls (webcam window): press Q to stop.  Temp recordings live under
training/artifacts/live_tmp/ and are removed right after each segment is read
(use --keep-temp to inspect them).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LIVE_TMP_DIR, PREDICTIONS_DIR, ensure_dirs  # noqa: E402
from predictor import SignPredictor, video_to_clip  # noqa: E402
from segmenter import MotionSegmenter, SignDebouncer  # noqa: E402
from utils import save_json  # noqa: E402

FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
GREY = (170, 170, 170)
GREEN = (90, 240, 130)
ORANGE = (60, 170, 250)
DIM = (140, 140, 140)


# ── on-screen feedback ────────────────────────────────────────────────────────
def _overlay(disp, ev, rec_secs, transcript, flash, now):
    """Draw an unambiguous state so a *stale* word never looks like a live one.

    LISTENING (grey)  while idle   ·   ● REC + motion bar  while recording
    a recognised sign flashes GREEN for ~1 s   ·   transcript along the bottom.
    """
    H, W = disp.shape[:2]
    ov = disp.copy()
    cv2.rectangle(ov, (0, 0), (W, 44), (0, 0, 0), -1)
    cv2.rectangle(ov, (0, H - 38), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, disp, 0.45, 0, disp)

    if flash and now < flash[2]:
        cv2.putText(disp, f"= {flash[0]}  ({flash[1]:.2f})", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, GREEN, 2, cv2.LINE_AA)
    elif ev.state == "recording":
        cv2.putText(disp, f"REC  {rec_secs:4.1f}s", (40, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, ORANGE, 2, cv2.LINE_AA)
        cv2.circle(disp, (20, 24), 8, (60, 60, 240), -1)        # red dot
        bar = int(min(1.0, ev.level / 3.0) * (W - 220))         # motion strength
        cv2.rectangle(disp, (200, 18), (200 + max(2, bar), 30), ORANGE, -1)
    else:
        cv2.putText(disp, "LISTENING", (40, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, GREY, 2, cv2.LINE_AA)
        cv2.circle(disp, (20, 24), 8, (90, 90, 90), 2)
        cv2.putText(disp, "(sign when ready - Q to quit)", (210, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, DIM, 1, cv2.LINE_AA)

    tail = " ".join(s["sign"] for s in transcript[-8:]) or "-"
    cv2.putText(disp, f"transcript: {tail}", (12, H - 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)


# ── classify ONE closed segment (record -> temp -> extract -> model -> delete) ─
def process_segment(frames, fps, size, predictor, args):
    if not frames:
        return None
    tmp = LIVE_TMP_DIR / f"seg_{uuid.uuid4().hex[:8]}.mp4"
    vw = cv2.VideoWriter(str(tmp), FOURCC, fps, size)
    for f in frames:
        vw.write(f)
    vw.release()
    try:
        clip = video_to_clip(tmp, proc_width=args.proc_width)
        if clip is None or clip["T"] < 3:
            return None
        out = predictor.predict_robust(clip["raw"])
        out["T"] = clip["T"]
        hands_seen = (float(clip["masks"]["hands"].any(axis=1).mean())
                      if "hands" in clip["masks"] else 0.0)
        out["hands_rate"] = round(hands_seen, 2)
        out["ok"] = (out["conf"] >= args.conf and out["margin"] >= args.margin
                     and hands_seen > 0.15)
        return out
    finally:
        if not args.keep_temp:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _emit(out, predictor, deb, transcript, now_wall):
    """Apply the dedup gate and append to the transcript; return a status str."""
    if out is None:
        return None, "open"
    if not out["ok"]:
        why = "low-conf" if out["conf"] < 0.5 else "tie"
        return None, f"{out['name']}? ({out['conf']:.2f}/m{out['margin']:.2f}) {why}"
    if not deb.accept(out["name"], now_wall):
        return None, f"{out['name']} (repeat suppressed)"
    transcript.append({"t": round(now_wall, 1), "sign": out["name"],
                       "conf": round(out["conf"], 3)})
    return out["name"], f"{out['name']}  conf={out['conf']:.2f} m={out['margin']:.2f} " \
                        f"hands={out['hands_rate']} T={out['T']} ({out['n_windows']}w)"


def _make_segmenter(fps, args):
    return MotionSegmenter(
        fps, start_mult=args.motion_start, stop_mult=args.motion_stop,
        abs_floor=args.floor, still_hold_s=args.still, min_sign_s=args.min_sign,
        max_sign_s=args.max_sign, preroll_s=args.preroll)


# ── Webcam ────────────────────────────────────────────────────────────────────
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

    seg = _make_segmenter(fps, args)
    deb = SignDebouncer(window_s=args.repeat_window, enabled=not args.no_dedup)
    print(f"[live] camera {args.device_index}  {W}x{H}@{fps:.0f}  proc_width={args.proc_width}")
    print(f"[live] motion-gated: start>{seg.start_thresh:.4f} stop<{seg.stop_thresh:.4f} "
          f"still={args.still}s sign={args.min_sign}-{args.max_sign}s  "
          f"conf>={args.conf} margin>={args.margin}  dedup={'off' if args.no_dedup else f'{args.repeat_window}s'}")
    print("[live] press Q in the window to stop.")

    transcript, flash, rec_frames, stop = [], None, 0, False
    while not stop:
        ok, frame = cap.read()
        if not ok:
            break
        ev = seg.feed(frame)
        rec_frames = rec_frames + 1 if ev.state == "recording" else 0
        now = time.time()

        if not args.no_display:
            disp = cv2.flip(frame, 1)
            _overlay(disp, ev, rec_frames / fps, transcript, flash, now)
            cv2.imshow("Live sign translation", disp)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break

        if ev.segment is not None:
            out = process_segment(ev.segment, fps, (W, H), predictor, args)
            name, msg = _emit(out, predictor, deb, transcript, time.time())
            if name:
                flash = (name, out["conf"], time.time() + 1.2)
            print(f"[live] -> {msg}  (temp deleted)")
            seg.reset_motion()                  # avoid a false spike after the stall

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

    seg = _make_segmenter(fps, args)
    deb = SignDebouncer(window_s=args.repeat_window, enabled=not args.no_dedup)
    print(f"[live-sim] source={args.source}  {W}x{H}@{fps:.0f}  "
          f"motion-gated (start>{seg.start_thresh:.4f})")

    transcript, n_seg, fi, done = [], 0, 0, False

    def handle(segment):
        nonlocal n_seg
        n_seg += 1
        out = process_segment(segment, fps, (W, H), predictor, args)
        _, msg = _emit(out, predictor, deb, transcript, time.time() + n_seg)
        print(f"[seg {n_seg}] {msg}")

    while not done:
        ok, frame = cap.read()
        if not ok:
            break
        ev = seg.feed(frame)
        fi += 1
        if not args.no_display:
            disp = cv2.resize(frame, (min(W, 960), int(H * min(W, 960) / W)))
            _overlay(disp, ev, 0.0, transcript, None, time.time())
            cv2.imshow("Live sign translation (sim)", disp)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                done = True
        if ev.segment is not None:
            handle(ev.segment)
            seg.reset_motion()

    tail = seg.flush()
    if tail:
        handle(tail)
    if n_seg == 0:                              # never triggered: classify whole clip
        print("[live-sim] no motion segment detected — classifying the whole clip.")
        clip = video_to_clip(args.source, proc_width=args.proc_width)
        if clip is not None and clip["T"] >= 3:
            out = predictor.predict_robust(clip["raw"])
            out.update(T=clip["T"], hands_rate=1.0, ok=out["conf"] >= args.conf)
            _, msg = _emit(out, predictor, deb, transcript, time.time())
            print(f"[whole] {msg}")

    cap.release()
    cv2.destroyAllWindows()
    _save_transcript(transcript)


def _save_transcript(transcript):
    for f in list(LIVE_TMP_DIR.glob("seg_*.mp4")):    # never leave cache behind
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
          f"L={predictor.seq_len}")
    if args.source:
        run_source(args, predictor)
    else:
        run_webcam(args, predictor)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 5: motion-gated live sign translation.")
    p.add_argument("--source", help="simulate live from this video file (no camera needed)")
    p.add_argument("--device-index", type=int, default=0, help="webcam index")
    p.add_argument("--proc-width", type=int, default=480, help="downscale width for live speed (0=full)")
    p.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    p.add_argument("--int8", action="store_true", help="use the int8 ONNX model")
    p.add_argument("--no-display", action="store_true", help="headless (no preview window)")
    p.add_argument("--keep-temp", action="store_true", help="don't delete temp segment files")
    # emit gating
    p.add_argument("--conf", type=float, default=0.55, help="min top-1 confidence to emit")
    p.add_argument("--margin", type=float, default=0.15,
                   help="min top1-top2 gap to emit (rejects look-alike ties)")
    p.add_argument("--repeat-window", type=float, default=4.0,
                   help="seconds within which an identical sign is treated as a repeat")
    p.add_argument("--no-dedup", action="store_true", help="emit every segment, even repeats")
    # motion gating (segmenter)
    p.add_argument("--motion-start", type=float, default=2.6,
                   help="start a recording when motion > this x the idle noise floor")
    p.add_argument("--motion-stop", type=float, default=1.7,
                   help="treat as still when motion < this x the idle noise floor")
    p.add_argument("--floor", type=float, default=0.0025, help="absolute motion floor (anti-noise)")
    p.add_argument("--still", type=float, default=0.5, help="seconds of stillness that ends a sign")
    p.add_argument("--min-sign", type=float, default=0.5, help="discard segments shorter than this")
    p.add_argument("--max-sign", type=float, default=5.0, help="force-close a sign after this long")
    p.add_argument("--preroll", type=float, default=0.25, help="seconds kept before the onset")
    p.add_argument("--seconds", type=float, help=argparse.SUPPRESS)  # deprecated: ignored
    args = p.parse_args()
    if args.seconds is not None:
        print("[live] note: --seconds is deprecated and ignored "
              "(recording is now motion-gated; see --min-sign/--max-sign/--still).")
    return args


if __name__ == "__main__":
    main()
