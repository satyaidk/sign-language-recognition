"""
Motion-gated sign segmentation + repeat de-duplication for live translation.
=============================================================================
The live translator used to record FIXED 2.5 s chunks, run the model, repeat.
Three things went wrong with that:

  1. A fixed window cuts a sign in half — a chunk ends up holding the *tail* of
     one sign and the *start* of the next, so the model sees a blend.
  2. The model has no "idle / no-sign" class, so EVERY chunk is force-classified
     — even a chunk where the signer is just resting between signs.  A held,
     still hand keeps matching the previous sign, so the old word "sticks".
  3. Repeating one sign emits the same word many times.

This module fixes (1)+(2) at the *segmentation* level and (3) at the *emit*
level, WITHOUT touching the verified extraction/model path:

  - `MotionSegmenter` watches a cheap per-frame motion signal (grayscale
    frame-difference energy) and only opens a recording when motion rises, then
    closes it when the signer goes still.  The recording is therefore exactly as
    long as the sign — variable length, not a fixed 2 s — and the gaps between
    signs become an explicit IDLE state instead of being force-classified.
  - `SignDebouncer` collapses an immediately-repeated sign to a single word
    (within a time window), so signing "no, no, no" reads as one "no".

Both are pure, dependency-light, and unit-testable (see `__main__`).  Only the
motion measurement uses OpenCV; the decision logic (`_step`) is fed a scalar so
it can be tested with a synthetic motion trace and dummy frames.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 always present at serve time
    cv2 = None


@dataclass
class SegEvent:
    """What the segmenter reports for one fed frame."""
    state: str                       # "idle" | "recording"
    motion: float                    # smoothed motion energy (0..~0.1)
    level: float                     # motion / start_threshold  (UI bar, capped at caller)
    started: bool                    # True on the exact frame a recording opened
    segment: Optional[List[np.ndarray]]  # frames to classify (set once, when a sign closes)
    reason: str                      # "", "still" (normal end), "max" (length cap), "short" (discarded)


class MotionSegmenter:
    """Open a recording when the signer moves, close it when they go still.

    Thresholds are relative to an auto-learned *noise floor* (the residual
    frame-diff seen while idle), so it adapts to camera/lighting without manual
    tuning.  Hysteresis (start_mult > stop_mult) stops the state from chattering
    on the boundary.
    """

    def __init__(self, fps: float, *, proc_h: int = 120,
                 start_mult: float = 2.6, stop_mult: float = 1.7,
                 abs_floor: float = 0.0025, still_hold_s: float = 0.5,
                 min_sign_s: float = 0.5, max_sign_s: float = 5.0,
                 preroll_s: float = 0.25, keep_tail_s: float = 0.15,
                 smooth: float = 0.5, calib_s: float = 0.7):
        self.fps = float(fps)
        self.proc_h = int(proc_h)
        self.start_mult, self.stop_mult = float(start_mult), float(stop_mult)
        self.abs_floor = float(abs_floor)
        self.smooth = float(smooth)

        f = self.fps
        self.still_hold = max(1, int(round(still_hold_s * f)))
        self.min_frames = max(2, int(round(min_sign_s * f)))
        self.max_frames = max(self.min_frames + 1, int(round(max_sign_s * f)))
        self.keep_tail = max(0, int(round(keep_tail_s * f)))
        self.calib_frames = max(1, int(round(calib_s * f)))
        self.preroll = deque(maxlen=max(1, int(round(preroll_s * f))))

        self._prev = None            # previous small grayscale frame
        self._m = 0.0                # smoothed motion
        self._noise = self.abs_floor  # running idle noise floor
        self._seen = 0
        self.state = "idle"
        self._buf: List[np.ndarray] = []
        self._still = 0

    # ── motion measurement ────────────────────────────────────────────────────
    def _gray(self, frame):
        h = self.proc_h
        w = max(1, int(round(frame.shape[1] * h / frame.shape[0])))
        g = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        return g.astype(np.float32) / 255.0

    def _raw_motion(self, frame) -> float:
        g = self._gray(frame)
        if self._prev is None or self._prev.shape != g.shape:
            self._prev = g
            return 0.0
        m = float(np.mean(np.abs(g - self._prev)))
        self._prev = g
        return m

    # ── thresholds (relative to the learned noise floor) ──────────────────────
    @property
    def start_thresh(self) -> float:
        return max(self.abs_floor, self._noise * self.start_mult)

    @property
    def stop_thresh(self) -> float:
        return max(self.abs_floor * 0.6, self._noise * self.stop_mult)

    # ── public API ────────────────────────────────────────────────────────────
    def feed(self, frame) -> SegEvent:
        """Measure motion for `frame` and advance the state machine."""
        m = self._raw_motion(frame)
        self._m = (self.smooth * self._m + (1.0 - self.smooth) * m) if self._seen else m
        return self._step(self._m, frame)

    def reset_motion(self) -> None:
        """Drop the motion baseline (call after a processing stall so the next
        frame-diff isn't a false spike).  Keeps the learned noise floor."""
        self._prev = None
        self._m = 0.0

    def flush(self) -> Optional[List[np.ndarray]]:
        """End-of-stream: close an open recording (used by the file simulator)."""
        if self.state == "recording":
            seg = self._buf if len(self._buf) >= self.min_frames else None
            self.state = "idle"
            self._buf, self._still = [], 0
            return seg
        return None

    # ── decision logic (pure: scalar motion + frame in, event out) ────────────
    def _step(self, m: float, frame) -> SegEvent:
        self._seen += 1
        # Track the *quiet baseline*: follow motion DOWN fast but rise only very
        # slowly, so a sign (high motion) barely moves the floor while a lull
        # resets it.  A plain average would let the sign inflate its own
        # threshold (it did — a small-in-frame signer never crossed it).
        if m < self._noise:
            self._noise = 0.6 * self._noise + 0.4 * m          # follow a lull down fast
        else:
            self._noise = 0.998 * self._noise + 0.002 * m      # rise very slowly
        self._noise = max(self._noise, 1e-5)

        self.preroll.append(frame)
        started = False
        segment: Optional[List[np.ndarray]] = None
        reason = ""

        if self.state == "idle":
            if self._seen > self.calib_frames and m > self.start_thresh:
                self.state = "recording"
                self._buf = list(self.preroll)   # pre-roll so the onset isn't clipped
                self._still = 0
                started = True
        else:  # recording
            self._buf.append(frame)
            self._still = self._still + 1 if m < self.stop_thresh else 0
            full = len(self._buf) >= self.max_frames
            if self._still >= self.still_hold or full:
                segment = self._close(full)
                reason = "max" if full else ("still" if segment is not None else "short")
                self.state = "idle"
                self._buf, self._still = [], 0

        level = m / (self.start_thresh + 1e-9)
        return SegEvent(self.state, m, level, started, segment, reason)

    def _close(self, full: bool) -> Optional[List[np.ndarray]]:
        buf = self._buf
        # Trim the trailing stillness that confirmed the end (keep a short tail).
        if not full and self._still > self.keep_tail:
            buf = buf[: len(buf) - (self._still - self.keep_tail)]
        return buf if len(buf) >= self.min_frames else None


class SignDebouncer:
    """Suppress an immediately-repeated sign within `window_s` seconds.

    A *different* sign always passes through; the same sign passes again only
    after a clear pause longer than the window (an intentional repeat).
    """

    def __init__(self, window_s: float = 4.0, enabled: bool = True):
        self.window_s = float(window_s)
        self.enabled = bool(enabled)
        self.last: Optional[str] = None
        self.last_t = -1e9

    def accept(self, name: str, t: float) -> bool:
        if self.enabled and name == self.last and (t - self.last_t) < self.window_s:
            self.last_t = t                       # refresh: a run keeps collapsing
            return False
        self.last, self.last_t = name, t
        return True


# ── self-test: synthetic motion trace, no camera needed ───────────────────────
def _self_test():
    fps = 30.0
    seg = MotionSegmenter(fps, calib_s=0.3, still_hold_s=0.3, min_sign_s=0.3,
                          preroll_s=0.1)
    dummy = np.zeros((4, 4, 3), np.uint8)

    # idle (learn floor) -> burst -> idle -> burst -> idle
    trace = ([0.001] * 15 + [0.05] * 30 + [0.001] * 20
             + [0.04] * 25 + [0.001] * 15)
    segments, starts = [], 0
    for m in trace:
        ev = seg._step(m, dummy.copy())     # feed scalar motion directly
        starts += int(ev.started)
        if ev.segment is not None:
            segments.append(len(ev.segment))
    tail = seg.flush()

    print(f"[self-test] starts={starts}  segments_closed={len(segments)} "
          f"lengths={segments}  tail={'yes' if tail else 'no'}")
    assert starts == 2, f"expected 2 sign onsets, got {starts}"
    assert len(segments) == 2, f"expected 2 closed segments, got {len(segments)}"

    deb = SignDebouncer(window_s=4.0)
    seq = [("no", 0.0), ("no", 1.0), ("no", 2.0), ("yes", 3.0), ("no", 9.0)]
    emitted = [name for name, t in seq if deb.accept(name, t)]
    print(f"[self-test] dedup {[s[0] for s in seq]} -> {emitted}")
    assert emitted == ["no", "yes", "no"], emitted

    deb2 = SignDebouncer(enabled=False)
    assert all(deb2.accept("no", t) for t in range(5)), "disabled dedup must pass all"
    print("[self-test] OK")


if __name__ == "__main__":
    _self_test()
