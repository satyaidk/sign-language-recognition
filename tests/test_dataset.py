"""L1: extraction bookkeeping, QA, verification and pruning (no MediaPipe: a fake extractor)."""
from __future__ import annotations

import numpy as np
import pytest

from signlang.dataset import extract as E
from signlang.dataset.prune import prune, refresh_reports
from signlang.dataset.qa import qa_clip, verify_clip
from signlang.dataset.verify import verify_dataset
from signlang.landmarks.extractor import ExtractorOptions
from signlang.utils import load_json, read_csv_rows, save_json, write_csv_rows


def fake_extract(video, opts):
    """Stand-in for extract_video: deterministic arrays, no MediaPipe."""
    if video.stat().st_size == 0:
        return None                                   # empty file = "cannot open"
    return fake_result(seed=len(video.stem))


def fake_result(seed=0):
    T = 6
    rng = np.random.default_rng(seed)
    hands = rng.uniform(0.2, 0.8, (T, 2, 21, 3)).astype(np.float32)
    hands[:, 0] = 0                                   # left hand never detected
    return {"fps": 30.0, "n_frames_read": T, "n_frames_skipped": 0, "T": T,
            "arrays": {"pose": rng.uniform(0.2, 0.8, (T, 33, 4)).astype(np.float32),
                       "face": rng.uniform(0.3, 0.7, (T, 478, 3)).astype(np.float32),
                       "hands": hands},
            "masks": {"pose": np.ones(T, np.int8), "face": np.ones(T, np.int8),
                      "hands": np.tile(np.array([0, 1], np.int8), (T, 1))}}


@pytest.fixture
def video_tree(tmp_path):
    ds = tmp_path / "dataset"
    for cls, n in (("hello", 3), ("no", 2), ("yes", 2)):
        (ds / cls).mkdir(parents=True)
        for i in range(n):
            (ds / cls / f"{cls}_{i}.mp4").write_bytes(b"x")
    (ds / "notes").mkdir()                            # a folder without videos is ignored
    return ds, tmp_path / "processed"


def run(ds, out, **kw):
    return E.extract_dataset(ds, out, ExtractorOptions(), skeleton_size=None, extract_fn=fake_extract, **kw)


def test_discover_and_merge_helpers(video_tree):
    ds, _ = video_tree
    found = E.discover(ds)
    assert list(found) == ["hello", "no", "yes"]
    assert E.merge_class_index({"no": 0}, ["yes", "hello"]) == {"no": 0, "hello": 1, "yes": 2}
    rows = E.upsert([{"class": "a", "stem": "1", "v": 1}], [{"class": "a", "stem": "1", "v": 2},
                                                            {"class": "b", "stem": "1", "v": 3}])
    assert sorted(r["v"] for r in rows) == [2, 3]


def test_full_extraction_writes_consistent_dataset(video_tree):
    ds, out = video_tree
    report = run(ds, out)
    assert report["n_clips"] == 7 and report["verdicts"]["PASS"] == 7
    assert load_json(out / "classes.json") == {"hello": 0, "no": 1, "yes": 2}
    manifest = read_csv_rows(out / "manifest.csv")
    assert len(manifest) == 7 and manifest[0]["all_npy"].startswith("landmarks_all/")
    assert np.load(out / "landmarks_all" / "hello" / "hello_0.npy").shape == (6, 1692)
    assert (out / "FEATURE_LAYOUT.json").exists() and (out / "README.md").exists()


def test_partial_run_merges_instead_of_overwriting(video_tree):
    """Regression: --classes used to rewrite classes.json (renumbering labels) and
    truncate manifest.csv to the processed subset."""
    ds, out = video_tree
    run(ds, out)
    run(ds, out, classes=["yes"], limit=1)
    assert load_json(out / "classes.json") == {"hello": 0, "no": 1, "yes": 2}
    assert len(read_csv_rows(out / "manifest.csv")) == 7
    assert load_json(out / "extraction_report.json")["n_clips"] == 7


def test_new_class_is_appended_after_existing_labels(video_tree):
    ds, out = video_tree
    run(ds, out)
    (ds / "again").mkdir()
    (ds / "again" / "again_0.mp4").write_bytes(b"x")
    run(ds, out, classes=["again"])
    assert load_json(out / "classes.json")["again"] == 3          # not re-sorted to index 0


def test_skip_existing_keeps_skipped_clips_in_the_manifest(video_tree):
    ds, out = video_tree
    run(ds, out)
    (out / "manifest.csv").unlink()
    run(ds, out, skip_existing=True)
    assert len(read_csv_rows(out / "manifest.csv")) == 7


def test_unreadable_video_is_reported_not_manifested(video_tree):
    ds, out = video_tree
    (ds / "no" / "broken.mp4").write_bytes(b"")
    report = run(ds, out)
    broken = [c for c in report["clips"] if c["stem"] == "broken"]
    assert broken and broken[0]["verdict"] == "FAIL"
    assert all(r["stem"] != "broken" for r in read_csv_rows(out / "manifest.csv"))


# ── QA ────────────────────────────────────────────────────────────────────────
def test_qa_clip_verdicts():
    good = fake_result()
    assert qa_clip(good["arrays"], good["masks"], good["T"])["verdict"] == "PASS"
    bad = {k: v.copy() for k, v in good["arrays"].items()}
    bad["face"][0, 0, 0] = np.nan
    assert qa_clip(bad, good["masks"], good["T"])["verdict"] == "FAIL"
    low_pose = dict(good["masks"], pose=np.zeros(6, np.int8))
    q = qa_clip(good["arrays"], low_pose, good["T"])
    assert q["verdict"] == "WARN" and any("pose" in i for i in q["issues"])
    assert qa_clip({}, {}, 0)["verdict"] == "FAIL"


def test_verify_passes_extracted_data_and_catches_corruption(video_tree):
    ds, out = video_tree
    run(ds, out)
    layout = load_json(out / "FEATURE_LAYOUT.json")
    assert verify_clip(out, "hello", "hello_0", layout)["verdict"] == "PASS"

    all_path = out / "landmarks_all" / "hello" / "hello_1.npy"
    a = np.load(all_path)
    a[0, 0] += 0.5
    np.save(all_path, a)
    rep = verify_clip(out, "hello", "hello_1", layout)
    assert rep["verdict"] == "FAIL" and any("all != parts" in i for i in rep["issues"])

    np.save(out / "landmarks_pose" / "no" / "no_0.npy", np.zeros((3, 33, 4), np.float32))
    assert verify_clip(out, "no", "no_0", layout)["verdict"] == "FAIL"   # frame counts differ
    assert verify_clip(out, "no", "missing", layout)["verdict"] == "FAIL"

    summary = verify_dataset(out, quiet=True)
    assert summary["n_clips"] == 7 and summary["verdicts"]["FAIL"] == 2
    assert (out / "verification_report.json").exists()


# ── prune ─────────────────────────────────────────────────────────────────────
def test_prune_moves_source_deletes_derived_and_refreshes_reports(video_tree, tmp_path):
    ds, out = video_tree
    run(ds, out)
    verify_dataset(out, quiet=True)
    excluded = tmp_path / "excluded_clips"

    dry = prune([("no", "no_0")], out, ds, excluded, dry_run=True)
    assert dry["moved"] and (ds / "no" / "no_0.mp4").exists()               # dry run changes nothing

    res = prune([("no", "no_0")], out, ds, excluded)
    refresh_reports(out, {("no", "no_0")})
    assert (excluded / "no" / "no_0.mp4").exists() and not (ds / "no" / "no_0.mp4").exists()
    assert not (out / "landmarks_all" / "no" / "no_0.npy").exists() and len(res["removed"]) >= 5
    assert len(read_csv_rows(out / "manifest.csv")) == 6
    for name in ("extraction_report.json", "verification_report.json"):
        assert load_json(out / name)["n_clips"] == 6


def test_prune_requires_explicit_clips():
    from signlang.dataset.prune import parse_args
    with pytest.raises(SystemExit):
        parse_args([])                                 # the old default pruned 2 real clips


def test_refresh_reports_tolerates_missing_files(tmp_path):
    refresh_reports(tmp_path, {("a", "b")})           # nothing to refresh -> no error
    write_csv_rows(tmp_path / "manifest.csv", [{"class": "a", "stem": "b", "label": 0}])
    save_json({"clips": [{"class": "a", "stem": "b", "verdict": "PASS"}]}, tmp_path / "extraction_report.json")
    refresh_reports(tmp_path, {("a", "b")})
    assert load_json(tmp_path / "extraction_report.json")["n_clips"] == 0
