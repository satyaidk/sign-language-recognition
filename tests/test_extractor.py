"""L0: MediaPipe result conversion (with fakes) and real extraction on a synthetic video."""
from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from signlang.landmarks.extractor import (
    ExtractorOptions, FrameLandmarks, LandmarkExtractor, face_from_result, hands_from_result,
    pose_from_result,
)
from signlang.landmarks.smoothing import HandStabilizer, PointStabilizer, POSE_EURO
from tests.conftest import FakeLandmarkList


def _hand_result(labels_and_x):
    return SimpleNamespace(
        multi_hand_landmarks=[FakeLandmarkList([(x, 0.5, 0.0)] * 21) for _, x in labels_and_x],
        multi_handedness=[SimpleNamespace(classification=[SimpleNamespace(label=lbl)])
                          for lbl, _ in labels_and_x])


def _extractor_without_models(**opts) -> LandmarkExtractor:
    """A LandmarkExtractor whose MediaPipe models are never built (conversion tests)."""
    ex = object.__new__(LandmarkExtractor)
    ex.opts = ExtractorOptions(**opts)
    ex.pose_stab = PointStabilizer(33, POSE_EURO)
    ex.hand_stab = HandStabilizer()
    return ex


def test_result_converters_handle_missing_detections():
    assert pose_from_result(None) is None
    assert pose_from_result(SimpleNamespace(pose_landmarks=None)) is None
    assert face_from_result(SimpleNamespace(multi_face_landmarks=None)) is None
    assert hands_from_result(SimpleNamespace(multi_hand_landmarks=None)) == []


def test_result_converters_shapes():
    pose = pose_from_result(SimpleNamespace(pose_landmarks=FakeLandmarkList([(0.5, 0.5, 0.0, 0.9)] * 33)))
    face = face_from_result(SimpleNamespace(multi_face_landmarks=[FakeLandmarkList([(0.5, 0.4, 0.0)] * 478)]))
    hands = hands_from_result(_hand_result([("Left", 0.3)]))
    assert pose.shape == (33, 4) and pose[0, 3] == pytest.approx(0.9)
    assert face.shape == (478, 3)
    assert len(hands) == 1 and hands[0][0].shape == (21, 3) and hands[0][1] == "Left"


def test_from_results_places_hands_in_anatomical_slots():
    ex = _extractor_without_models(smooth=False)
    fl = ex.from_results(None, _hand_result([("Right", 0.7)]), None, 0.0)
    # raw video: MediaPipe "Right" is the signer's LEFT hand -> slot 0
    assert fl.hand_mask.tolist() == [1, 0]
    assert fl.pose is None and fl.face is None


def test_from_results_keeps_both_hands_when_labels_collide():
    ex = _extractor_without_models(smooth=True)
    fl = ex.from_results(None, _hand_result([("Left", 0.3), ("Left", 0.7)]), None, 0.0)
    assert fl.hand_mask.tolist() == [1, 1]           # the old code lost one of them


def test_frame_landmarks_row_layout():
    fl = FrameLandmarks(pose=np.full((33, 4), 0.5, np.float32))
    fl.hands[1] = 0.25
    fl.hand_mask[1] = 1
    row = fl.row()
    assert row.shape == (1692,)
    assert np.all(row[:132] == 0.5) and np.all(row[1629:] == 0.25) and not row[1566:1629].any()


def test_skipped_modalities_are_not_extracted():
    ex = _extractor_without_models(face=False, smooth=False)
    face_res = SimpleNamespace(multi_face_landmarks=[FakeLandmarkList([(0.5, 0.5, 0.0)] * 478)])
    assert ex.from_results(face_res, None, None, 0.0).face is None


def test_extractor_requires_a_modality():
    with pytest.raises(ValueError):
        LandmarkExtractor(ExtractorOptions(face=False, hands=False, pose=False))


@pytest.mark.mediapipe
def test_extract_video_on_synthetic_clip(tmp_path):
    """Real MediaPipe models on a short noise video: correct shapes, nothing detected."""
    from signlang.landmarks.extractor import extract_video, video_to_raw

    path = tmp_path / "noise.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    rng = np.random.default_rng(0)
    for _ in range(8):
        vw.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    vw.release()

    res = extract_video(path, ExtractorOptions(proc_width=160))
    assert res["T"] == res["n_frames_read"] == 8
    assert res["arrays"]["pose"].shape == (8, 33, 4)
    assert res["arrays"]["face"].shape == (8, 478, 3)
    assert res["masks"]["hands"].shape == (8, 2)

    clip = video_to_raw(path, ExtractorOptions(face=False))
    assert clip["raw"].shape == (8, 1692)
    assert not clip["raw"][:, 132:1566].any()        # skipped face -> zero block, layout intact


def test_extract_video_missing_file_returns_none(tmp_path):
    from signlang.landmarks.extractor import extract_video
    assert extract_video(tmp_path / "missing.mp4") is None
