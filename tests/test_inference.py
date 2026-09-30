"""L3: motion segmenter, debouncer, emit gates and the live recogniser (fakes; no camera, no model)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from signlang.inference.live import EmitGate, LiveRecognizer, describe
from signlang.inference.segmenter import MotionSegmenter, SignDebouncer
from signlang.landmarks.extractor import FrameLandmarks

FPS = 30.0


# ── segmenter ─────────────────────────────────────────────────────────────────
def burst_trace():
    """idle (learn floor) -> burst -> idle -> burst -> idle."""
    return [0.001] * 15 + [0.05] * 30 + [0.001] * 20 + [0.04] * 25 + [0.001] * 15


def test_segmenter_finds_two_signs_in_two_bursts():
    seg = MotionSegmenter(FPS, calib_s=0.3, still_hold_s=0.3, min_sign_s=0.3, preroll_s=0.1)
    starts, lengths = 0, []
    for m in burst_trace():
        ev = seg.step(m, "frame")
        starts += int(ev.started)
        if ev.segment is not None:
            lengths.append(len(ev.segment))
    assert starts == 2 and len(lengths) == 2
    assert all(20 <= n <= 40 for n in lengths)              # ~ burst length + pre-roll + short tail
    assert seg.flush() is None


def test_segmenter_buffers_the_payload_not_the_frame():
    seg = MotionSegmenter(FPS, calib_s=0.1, still_hold_s=0.2, min_sign_s=0.2, preroll_s=0.0)
    got = None
    for i, m in enumerate([0.001] * 5 + [0.05] * 12 + [0.001] * 10):
        ev = seg.step(m, ("row", i))
        got = ev.segment or got
    assert got and all(isinstance(p, tuple) and p[0] == "row" for p in got)


def test_segmenter_force_closes_long_signs_and_drops_twitches():
    seg = MotionSegmenter(FPS, calib_s=0.1, still_hold_s=0.2, min_sign_s=0.5, max_sign_s=1.0)
    events = [seg.step(m, 0) for m in [0.001] * 5 + [0.05] * 45]
    assert any(e.reason == "max" and e.segment for e in events)

    seg = MotionSegmenter(FPS, calib_s=0.1, still_hold_s=0.2, min_sign_s=0.5, preroll_s=0.0)
    events = [seg.step(m, 0) for m in [0.001] * 5 + [0.05] * 3 + [0.001] * 12]
    assert not any(e.segment for e in events) and any(e.reason == "short" for e in events)


def test_segmenter_flush_returns_an_open_recording():
    seg = MotionSegmenter(FPS, calib_s=0.1, min_sign_s=0.2)
    for m in [0.001] * 5 + [0.05] * 20:
        seg.step(m, 0)
    assert seg.state == "recording" and len(seg.flush()) >= 6 and seg.state == "idle"


def test_segmenter_measures_motion_from_frames():
    seg = MotionSegmenter(FPS, calib_s=0.1)
    still = np.zeros((120, 160, 3), np.uint8)
    moving = still.copy()
    moving[40:80, 60:100] = 255
    seg.feed(still)
    assert seg.feed(still).motion == pytest.approx(0.0, abs=1e-6)
    assert seg.feed(moving).motion > 0.01


def test_debouncer():
    deb = SignDebouncer(window_s=4.0)
    seq = [("no", 0.0), ("no", 1.0), ("no", 2.0), ("yes", 3.0), ("no", 9.0)]
    assert [n for n, t in seq if deb.accept(n, t)] == ["no", "yes", "no"]
    off = SignDebouncer(enabled=False)
    assert all(off.accept("no", t) for t in range(5))


# ── emit gate ─────────────────────────────────────────────────────────────────
def test_emit_gate_names_the_real_reason():
    g = EmitGate(conf=0.55, margin=0.15, min_hands_rate=0.15)
    assert g.decide(0.9, 0.5, 0.9) == (True, "ok")
    assert g.decide(0.9, 0.5, 0.0) == (False, "no-hands")
    assert g.decide(0.52, 0.5, 0.9) == (False, "low-conf")   # old code called this a "tie" (>=0.5)
    assert g.decide(0.7, 0.05, 0.9) == (False, "tie")


# ── live recogniser (fake extractor + fake model) ─────────────────────────────
class FakeExtractor:
    """Pretends a right hand is visible in every frame."""

    def process(self, frame, t):
        fl = FrameLandmarks()
        fl.hands[1] = 0.5
        fl.hand_mask[1] = 1
        return fl

    def close(self):
        pass


class FakePredictor:
    """Returns the next scripted answer for every classified segment."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.seen_lengths = []

    def predict_robust(self, raw):
        self.seen_lengths.append(len(raw))
        name, conf, margin = self.answers.pop(0)
        return {"name": name, "conf": conf, "margin": margin, "topk": [(name, conf)], "n_windows": 1}


def frames_with_bursts(n_bursts, idle=25, burst=25):
    """Synthetic video: a white square moves during each burst, the scene is still otherwise."""
    frames, still = [], np.zeros((120, 160, 3), np.uint8)
    frames += [still] * idle
    for _ in range(n_bursts):
        for i in range(burst):
            f = still.copy()
            x = 10 + (i * 5) % 120
            f[40:80, x:x + 30] = 255
            frames.append(f)
        frames += [still] * idle
    return frames


def make_recognizer(answers, keep_history=False):
    seg = MotionSegmenter(FPS, calib_s=0.3, still_hold_s=0.3, min_sign_s=0.3)
    return LiveRecognizer(FakePredictor(answers), FakeExtractor(), seg, SignDebouncer(window_s=4.0),
                          EmitGate(), keep_history=keep_history)


def run_stream(rec, frames):
    for i, f in enumerate(frames):
        rec.push(f, i / FPS)
    rec.flush(len(frames) / FPS)
    return rec


def test_live_recognizer_emits_each_sign_once_with_gates():
    rec = run_stream(make_recognizer([("hello", 0.9, 0.6), ("good", 0.6, 0.05), ("thank_you", 0.95, 0.9)]),
                     frames_with_bursts(3))
    assert [r["decision"] for r in rec.records] == ["emit", "abstain", "emit"]
    assert rec.records[1]["reason"] == "tie"
    assert [s["sign"] for s in rec.transcript] == ["hello", "thank_you"]
    assert all(r["hands_rate"] == 1.0 and r["T"] > 20 for r in rec.records)


def test_live_recognizer_suppresses_quick_repeats_on_stream_time():
    rec = run_stream(make_recognizer([("no", 0.9, 0.6)] * 2), frames_with_bursts(2, idle=20))
    assert [r["decision"] for r in rec.records] == ["emit", "repeat"]
    assert len(rec.transcript) == 1


def test_file_mode_falls_back_to_the_whole_clip():
    # a clip in which the segmenter never fires -> classify all frames as one sign
    frames = [np.zeros((120, 160, 3), np.uint8)] * 20
    rec = run_stream(make_recognizer([("yes", 0.9, 0.7)], keep_history=True), frames)
    assert [r["close"] for r in rec.records] == ["whole-clip"]
    assert rec.predictor.seen_lengths == [len(frames)]


def test_describe_lines():
    assert "conf=0.90" in describe({"decision": "emit", "sign": "hi", "conf": 0.9, "margin": 0.5,
                                    "hands_rate": 1.0, "T": 30, "n_windows": 2, "classify_ms": 3.0})
    assert "repeat" in describe({"decision": "repeat", "sign": "hi"})
    assert "tie" in describe({"decision": "abstain", "sign": "hi", "conf": 0.6, "margin": 0.01, "reason": "tie"})


def test_live_cli_parses_defaults():
    from signlang.inference.live import parse_args
    a = parse_args([])
    assert (a.conf, a.margin, a.still, a.proc_width) == (0.55, 0.15, 0.5, 480)
    assert SimpleNamespace(**vars(a)).no_dedup is False
