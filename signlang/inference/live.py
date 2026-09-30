"""
STAGE 5 — Live sign recognition with streaming landmarks.
=========================================================
Pipeline per camera frame:

    frame ─► LandmarkExtractor (persistent models, One-Euro)  ─► landmark row (1692)
          └► MotionSegmenter (frame-difference energy)         ─► idle | recording
                                  on "sign finished" (stillness)  │
                                                                  ▼
             predict_robust(segment rows)  ─► gates (confidence, top-1/top-2 margin,
             hands present)  ─► SignDebouncer  ─► caption + transcript + session log

Why streaming?  The first live version recorded each sign to a temporary MP4,
re-opened it, built three new MediaPipe models and re-extracted every frame
AFTER the sign ended — ~40 ms per frame, so a 2 s sign froze the preview for
~2.5 s and signs made meanwhile were lost.  Now landmarks are computed while
the person signs, so when a sign ends only the tiny model runs (a few ms), no
temp files are written, and the preview never freezes.  Face detection is
skipped when the model does not use the face.

The frame -> sign logic lives in ``LiveRecognizer`` (camera- and model-agnostic,
unit-tested with fakes).  ``--source video.mp4`` runs the identical code path on
a file, using the video's own clock so results are reproducible.

Run:
    python -m signlang live                          # default webcam
    python -m signlang live --device-index 1 --int8
    python -m signlang live --margin 0.2 --still 0.6 # stricter / longer end-of-sign
    python -m signlang live --source clip.mp4 --no-display    # no camera needed

Every run writes predictions/live_transcript.json (latest) and a timestamped
predictions/sessions/<time>.json with one diagnostic record per segment.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from signlang.config import add_path_args, paths_from_args
from signlang.inference.predictor import load_predictor
from signlang.inference.segmenter import MotionSegmenter, SignDebouncer
from signlang.utils import save_json

GREY, GREEN, ORANGE, DIM = (170, 170, 170), (90, 240, 130), (60, 170, 250), (140, 140, 140)
STALL_S = 0.2   # a classification slower than this resets the motion baseline


# ── Emit decision (pure) ──────────────────────────────────────────────────────
@dataclass
class EmitGate:
    conf: float = 0.55           # min top-1 probability
    margin: float = 0.15         # min top-1 minus top-2 probability (rejects look-alike ties)
    min_hands_rate: float = 0.15 # min fraction of frames with a hand (rejects non-signing motion)

    def decide(self, conf: float, margin: float, hands_rate: float) -> tuple:
        """-> (ok, reason).  Reasons are checked in order and name the real cause."""
        if hands_rate <= self.min_hands_rate:
            return False, "no-hands"
        if conf < self.conf:
            return False, "low-conf"
        if margin < self.margin:
            return False, "tie"
        return True, "ok"


@dataclass
class LiveRecognizer:
    """Frames in, recognised signs out.  Owns no camera and no window."""
    predictor: object                  # SignPredictor-like: predict_robust(raw) -> dict
    extractor: object                  # LandmarkExtractor-like: process(frame, t) -> FrameLandmarks
    segmenter: MotionSegmenter
    debouncer: SignDebouncer
    gate: EmitGate = field(default_factory=EmitGate)
    keep_history: bool = False                      # file mode: keep every frame for a fallback
    records: list = field(default_factory=list)      # one diagnostic record per closed segment
    transcript: list = field(default_factory=list)   # emitted signs only
    history: list = field(default_factory=list)

    def push(self, frame, t: float):
        """Process one frame at stream time ``t`` (s) -> ``(SegEvent, record | None)``."""
        fl = self.extractor.process(frame, t)
        payload = (fl.row(), bool(np.any(fl.hand_mask)))
        if self.keep_history:
            self.history.append(payload)
        ev = self.segmenter.feed(frame, payload=payload)
        record = self.classify(ev.segment, t, ev.reason) if ev.segment is not None else None
        return ev, record

    def flush(self, t: float):
        """End of stream: classify a sign still being recorded.  In file mode, if
        no segment was ever detected (e.g. a short single-sign clip that starts
        moving inside the segmenter's calibration window), classify the whole clip."""
        segment = self.segmenter.flush()
        if segment:
            return self.classify(segment, t, "end")
        if self.keep_history and not self.records and self.history:
            return self.classify(self.history, t, "whole-clip")
        return None

    def classify(self, segment, t: float, close_reason: str = "") -> dict:
        t0 = time.perf_counter()
        rows = np.stack([row for row, _ in segment]).astype(np.float32)
        hands_rate = float(np.mean([has_hand for _, has_hand in segment]))
        out = self.predictor.predict_robust(rows)
        ok, reason = self.gate.decide(out["conf"], out["margin"], hands_rate)
        decision = "emit" if ok else "abstain"
        if ok and not self.debouncer.accept(out["name"], t):
            decision, reason = "repeat", "repeat suppressed"
        record = {"t": round(t, 3), "sign": out["name"], "conf": round(out["conf"], 4),
                  "margin": round(out["margin"], 4), "top3": out["topk"], "T": int(len(rows)),
                  "hands_rate": round(hands_rate, 3), "n_windows": out.get("n_windows", 1),
                  "close": close_reason, "decision": decision, "reason": reason,
                  "classify_ms": round((time.perf_counter() - t0) * 1000, 2)}
        self.records.append(record)
        if decision == "emit":
            self.transcript.append({"t": record["t"], "sign": out["name"], "conf": record["conf"]})
        return record


def describe(record: dict) -> str:
    """One console line for a segment decision."""
    if record["decision"] == "emit":
        return (f"{record['sign']}  conf={record['conf']:.2f} m={record['margin']:.2f} "
                f"hands={record['hands_rate']:.2f} T={record['T']} ({record['n_windows']}w, "
                f"{record['classify_ms']:.0f} ms)")
    if record["decision"] == "repeat":
        return f"{record['sign']} (repeat suppressed)"
    return f"{record['sign']}? ({record['conf']:.2f}/m{record['margin']:.2f}) {record['reason']}"


# ── On-screen feedback ────────────────────────────────────────────────────────
def draw_overlay(disp, ev, rec_secs, transcript, flash, now) -> None:
    """Unambiguous state, so a stale word never looks like a live one:
    LISTENING (grey) · REC + motion bar · recognised sign flashes GREEN · transcript."""
    H, W = disp.shape[:2]
    ov = disp.copy()
    cv2.rectangle(ov, (0, 0), (W, 44), (0, 0, 0), -1)
    cv2.rectangle(ov, (0, H - 38), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, disp, 0.45, 0, disp)
    if flash and now < flash[2]:
        cv2.putText(disp, f"= {flash[0]}  ({flash[1]:.2f})", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, GREEN, 2, cv2.LINE_AA)
    elif ev.state == "recording":
        cv2.putText(disp, f"REC  {rec_secs:4.1f}s", (40, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ORANGE, 2, cv2.LINE_AA)
        cv2.circle(disp, (20, 24), 8, (60, 60, 240), -1)
        bar = int(min(1.0, ev.level / 3.0) * (W - 220))
        cv2.rectangle(disp, (200, 18), (200 + max(2, bar), 30), ORANGE, -1)
    else:
        cv2.putText(disp, "LISTENING", (40, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2, cv2.LINE_AA)
        cv2.circle(disp, (20, 24), 8, (90, 90, 90), 2)
        cv2.putText(disp, "(sign when ready - Q to quit)", (210, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, DIM, 1, cv2.LINE_AA)
    tail = " ".join(s["sign"] for s in transcript[-8:]) or "-"
    cv2.putText(disp, f"transcript: {tail}", (12, H - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (235, 235, 235), 1, cv2.LINE_AA)


# ── Assembly ──────────────────────────────────────────────────────────────────
def build_recognizer(predictor, fps: float, args, extractor=None, keep_history=False) -> LiveRecognizer:
    if extractor is None:
        from signlang.landmarks.extractor import LandmarkExtractor
        extractor = LandmarkExtractor(predictor.extractor_options(proc_width=args.proc_width))
    seg = MotionSegmenter(fps, start_mult=args.motion_start, stop_mult=args.motion_stop,
                          abs_floor=args.floor, still_hold_s=args.still, min_sign_s=args.min_sign,
                          max_sign_s=args.max_sign, preroll_s=args.preroll)
    return LiveRecognizer(predictor, extractor, seg,
                          SignDebouncer(window_s=args.repeat_window, enabled=not args.no_dedup),
                          EmitGate(conf=args.conf, margin=args.margin), keep_history=keep_history)


def save_session(rec: LiveRecognizer, predictions_dir, meta: dict) -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    save_json({"transcript": rec.transcript}, predictions_dir / "live_transcript.json")
    save_json({**meta, "segments": rec.records, "transcript": rec.transcript},
              predictions_dir / "sessions" / f"{stamp}.json")
    if rec.transcript:
        print(f"\n[live] transcript ({len(rec.transcript)} signs): "
              + "  ".join(s["sign"] for s in rec.transcript))
    else:
        print("\n[live] no confident signs recognised.")
    print(f"[live] session log -> {predictions_dir / 'sessions' / (stamp + '.json')}")


def _open_capture(args):
    src = args.source if args.source else args.device_index
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        what = f"source video {args.source}" if args.source else f"camera index {args.device_index}"
        sys.exit(f"[ERROR] cannot open {what}. Use --source <video> to test without a camera.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, (fps if 1.0 < fps <= 120.0 else 30.0)


def run(args) -> LiveRecognizer:
    paths = paths_from_args(args)
    predictor = load_predictor(paths, backend=args.backend, int8=args.int8)
    cap, fps = _open_capture(args)
    is_file = bool(args.source)
    if not is_file:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    rec = build_recognizer(predictor, fps, args, keep_history=is_file)
    print(f"[live] backend={predictor.backend} classes={len(predictor.classes)} "
          f"face={'on' if predictor.fcfg.use_face else 'skipped'}  source={'file' if is_file else 'camera'}")
    print(f"[live] gates: conf>={args.conf} margin>={args.margin}  "
          f"dedup={'off' if args.no_dedup else f'{args.repeat_window}s'}  still={args.still}s")

    flash, rec_frames, fi = None, 0, 0
    t_start = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = fi / fps if is_file else time.monotonic() - t_start   # stream clock
            fi += 1
            t0 = time.perf_counter()
            ev, record = rec.push(frame, t)
            if time.perf_counter() - t0 > STALL_S:
                rec.segmenter.reset_motion()      # a slow step must not look like a motion spike
            rec_frames = rec_frames + 1 if ev.state == "recording" else 0
            if record is not None:
                print(f"[live] -> {describe(record)}")
                if record["decision"] == "emit":
                    flash = (record["sign"], record["conf"], time.monotonic() + 1.2)
            if not args.no_display:
                disp = frame.copy() if is_file else cv2.flip(frame, 1)
                draw_overlay(disp, ev, rec_frames / fps, rec.transcript, flash, time.monotonic())
                cv2.imshow("Live sign recognition", disp)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
        tail = rec.flush(fi / fps if is_file else time.monotonic() - t_start)
        if tail is not None:
            print(f"[live] -> {describe(tail)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rec.extractor.close()

    paths.predictions.mkdir(parents=True, exist_ok=True)
    save_session(rec, paths.predictions, {
        "source": args.source or f"camera:{args.device_index}", "fps": fps, "frames": fi,
        "backend": predictor.backend, "classes": predictor.classes,
        "gates": {"conf": args.conf, "margin": args.margin, "repeat_window": args.repeat_window},
        "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    return rec


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="signlang live", description="Stage 5: live sign recognition.")
    add_path_args(p, artifacts=True)
    p.add_argument("--source", help="run on this video file instead of the camera")
    p.add_argument("--device-index", type=int, default=0, help="webcam index")
    p.add_argument("--proc-width", type=int, default=480, help="downscale frames for detection (0 = full)")
    p.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    p.add_argument("--int8", action="store_true", help="use the int8 ONNX model if one was exported")
    p.add_argument("--no-display", action="store_true", help="headless (no preview window)")
    g = p.add_argument_group("emit gating")
    g.add_argument("--conf", type=float, default=0.55, help="min top-1 confidence to emit")
    g.add_argument("--margin", type=float, default=0.15, help="min top1-top2 gap to emit")
    g.add_argument("--repeat-window", type=float, default=4.0, help="seconds in which a repeat is suppressed")
    g.add_argument("--no-dedup", action="store_true", help="emit every segment, even repeats")
    m = p.add_argument_group("motion segmentation")
    m.add_argument("--motion-start", type=float, default=2.6, help="start when motion > this x noise floor")
    m.add_argument("--motion-stop", type=float, default=1.7, help="still when motion < this x noise floor")
    m.add_argument("--floor", type=float, default=0.0025, help="absolute motion floor (anti-noise)")
    m.add_argument("--still", type=float, default=0.5, help="seconds of stillness that end a sign")
    m.add_argument("--min-sign", type=float, default=0.5, help="discard segments shorter than this")
    m.add_argument("--max-sign", type=float, default=5.0, help="force-close a sign after this long")
    m.add_argument("--preroll", type=float, default=0.25, help="seconds kept before the onset")
    return p.parse_args(argv)


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
