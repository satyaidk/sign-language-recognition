"""L0: landmark layout, smoothing and hand-side logic (pure; no MediaPipe needed)."""
from __future__ import annotations

import numpy as np
import pytest

from signlang.landmarks import layout as L
from signlang.landmarks.hands import anatomical_label, assign_hand_slots, match_hands_to_sides
from signlang.landmarks.smoothing import (
    HAND_EURO, POSE_EURO, HandStabilizer, OneEuroFilter, PointStabilizer,
)


# ── layout ────────────────────────────────────────────────────────────────────
def test_raw_layout_is_contiguous_and_1692_wide():
    assert L.RAW_TOTAL == 1692
    ends = [L.BLOCK[n] for n in ("pose", "face", "left_hand", "right_hand")]
    assert ends[0][0] == 0 and ends[-1][1] == 1692
    assert all(a[1] == b[0] for a, b in zip(ends, ends[1:]))


def test_build_all_vector_full_layout_roundtrips_through_split(rng):
    T = 7
    arrays = {"pose": rng.random((T, 33, 4), dtype=np.float32),
              "face": rng.random((T, 478, 3), dtype=np.float32),
              "hands": rng.random((T, 2, 21, 3), dtype=np.float32)}
    vec, layout = L.build_all_vector(arrays)
    assert vec.shape == (T, 1692) and layout["total_features"] == 1692
    parts = L.split_all_vector(vec)
    np.testing.assert_array_equal(parts["pose"], arrays["pose"])
    np.testing.assert_array_equal(parts["face"], arrays["face"])
    np.testing.assert_array_equal(parts["left_hand"], arrays["hands"][:, 0])
    np.testing.assert_array_equal(parts["right_hand"], arrays["hands"][:, 1])


def test_build_all_vector_partial_modalities(rng):
    arrays = {"pose": rng.random((4, 33, 4), dtype=np.float32),
              "hands": rng.random((4, 2, 21, 3), dtype=np.float32)}
    vec, layout = L.build_all_vector(arrays, modalities=("pose", "hands"))
    assert vec.shape == (4, 132 + 63 + 63)
    assert [b["name"] for b in layout["blocks"]] == ["pose", "left_hand", "right_hand"]


def test_split_rejects_wrong_width():
    with pytest.raises(ValueError):
        L.split_all_vector(np.zeros((3, 100)))


def test_present_and_frame_row():
    block = np.zeros((3, 21, 3), np.float32)
    block[1] = 0.5
    assert L.present(block).tolist() == [False, True, False]
    row = L.frame_row(right_hand=np.full((21, 3), 0.25))
    s, e = L.BLOCK["right_hand"]
    assert row.shape == (1692,) and np.all(row[s:e] == 0.25) and not row[:s].any()


# ── smoothing ─────────────────────────────────────────────────────────────────
def test_one_euro_reduces_jitter_on_a_still_point():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(**POSE_EURO)
    raw = 0.5 + rng.normal(0, 0.01, 300)
    out = np.array([f(x, i / 30) for i, x in enumerate(raw)])
    assert out[50:].std() < 0.5 * raw[50:].std()


def test_one_euro_follows_a_step_quickly():
    f = OneEuroFilter(**HAND_EURO)
    xs = [0.2] * 30 + [0.6] * 30
    out = [float(f(x, i / 30)) for i, x in enumerate(xs)]
    assert abs(out[-1] - 0.6) < 0.01          # converged after one second
    assert out[31] > 0.3                      # and already moved most of the way


def test_one_euro_is_elementwise_for_arrays():
    f = OneEuroFilter()
    a = f(np.array([0.1, 0.9]), 0.0)
    np.testing.assert_allclose(a, [0.1, 0.9])
    b = f(np.array([0.1, 0.9]), 1 / 30)
    np.testing.assert_allclose(b, [0.1, 0.9])


def test_point_stabilizer_resets_a_teleporting_point():
    st = PointStabilizer(2, POSE_EURO)
    st.update(np.array([[0.1, 0.1, 0, 1], [0.5, 0.5, 0, 1]]), 0.0)
    out, stale = st.update(np.array([[0.9, 0.9, 0, 1], [0.51, 0.5, 0, 1]]), 1 / 30)
    assert not stale
    np.testing.assert_allclose(out[0, :2], [0.9, 0.9])   # jump > 0.3 -> taken as-is
    assert out[1, 0] < 0.51                              # small move -> smoothed


def test_point_stabilizer_hold_then_drop():
    st = PointStabilizer(1, POSE_EURO, hold_frames=2)
    st.update(np.array([[0.4, 0.4, 0, 1]]), 0.0)
    assert st.update(None, 0.1)[1] is True
    assert st.update(None, 0.2)[1] is True
    assert st.update(None, 0.3) == (None, False)


def test_point_stabilizer_validates_point_count():
    with pytest.raises(ValueError):
        PointStabilizer(33, POSE_EURO).update(np.zeros((21, 3)), 0.0)


def test_hand_stabilizer_forgets_unseen_hands():
    hs = HandStabilizer()
    pts = np.full((21, 3), 0.5, np.float32)
    hs.update([(pts, "Left"), (pts, "Left")], 0.0)       # duplicate labels are kept apart
    assert set(hs._stabs) == {"Left", "Left#2"}
    hs.update([(pts, "Right")], 1 / 30)
    assert set(hs._stabs) == {"Right"}


# ── hands ─────────────────────────────────────────────────────────────────────
def test_anatomical_label():
    assert anatomical_label("Left", mirrored=True) == "Left"
    assert anatomical_label("Left", mirrored=False) == "Right"


def test_match_hands_to_sides_never_crosses():
    sides = match_hands_to_sides([(10, 0), (0, 0)], lw=(9, 0), rw=(1, 0))
    assert sides == {0: "Left", 1: "Right"}


def _hand(x):
    pts = np.zeros((21, 3), np.float32)
    pts[:, 0], pts[:, 1] = x, 0.5
    return pts


def test_assign_hand_slots_normal_labels():
    # raw video: MediaPipe "Right" == the signer's LEFT hand
    out = dict(assign_hand_slots([(_hand(0.7), "Right"), (_hand(0.3), "Left")]))
    assert out[0][0, 0] == pytest.approx(0.7) and out[1][0, 0] == pytest.approx(0.3)


def test_assign_hand_slots_resolves_duplicate_labels_by_position():
    # Both hands labelled "Left": the old extractor wrote both into one slot.
    out = dict(assign_hand_slots([(_hand(0.3), "Left"), (_hand(0.7), "Left")]))
    assert set(out) == {0, 1}
    assert out[1][0, 0] == pytest.approx(0.3)   # image-left hand = signer's RIGHT (raw video)
    assert out[0][0, 0] == pytest.approx(0.7)


def test_assign_hand_slots_resolves_duplicates_with_pose_wrists():
    pose = np.zeros((33, 4), np.float32)
    pose[:, 3] = 1.0
    pose[L.LEFT_WRIST, :2] = [0.72, 0.5]      # signer's left wrist is on the image right
    pose[L.RIGHT_WRIST, :2] = [0.28, 0.5]
    out = dict(assign_hand_slots([(_hand(0.7), "Right"), (_hand(0.3), "Right")], pose=pose))
    assert out[0][0, 0] == pytest.approx(0.7) and out[1][0, 0] == pytest.approx(0.3)


def test_assign_hand_slots_empty():
    assert assign_hand_slots([]) == []


# ── MediaPipe guard ───────────────────────────────────────────────────────────
def test_mediapipe_guard_explains_a_too_new_version(monkeypatch):
    import sys
    import types

    from signlang.landmarks.mediapipe_compat import MediaPipeUnavailable, load_solutions
    fake = types.ModuleType("mediapipe")
    fake.__version__ = "0.10.31"                  # no `solutions` attribute, like the real 0.10.31+
    monkeypatch.setitem(sys.modules, "mediapipe", fake)
    with pytest.raises(MediaPipeUnavailable, match="0.10.9"):
        load_solutions()
