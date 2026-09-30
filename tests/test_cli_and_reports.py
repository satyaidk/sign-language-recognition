"""CLI wiring and the dataset infographics (report must reflect measured numbers only)."""
from __future__ import annotations

import pytest

from signlang import __main__ as cli
from signlang.dataset import report as R
from signlang.utils import save_json


def test_cli_lists_every_command(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for name in cli.COMMANDS:
        assert name in out


def test_cli_unknown_command_fails(capsys):
    assert cli.main(["dance"]) == 2
    assert "unknown command" in capsys.readouterr().err


@pytest.mark.parametrize("command", sorted(cli.COMMANDS))
def test_every_command_has_working_help(command):
    with pytest.raises(SystemExit) as exc:
        cli.main([command, "--help"])
    assert exc.value.code == 0


def _metadata(out, cls, stem, verdict="PASS"):
    save_json({"class": cls, "stem": stem, "T": 40, "all_shape": [40, 1692],
               "qa": {"verdict": verdict, "detection_rate": {
                   "pose": 1.0, "face": 0.9, "left_hand": 0.2, "right_hand": 0.8, "any_hand": 0.85}}},
              out / "metadata" / cls / f"{stem}.json")


def test_report_facts_come_from_files_not_constants(tmp_path):
    processed = tmp_path / "processed_dataset"
    assert R.verification_facts(processed)["ran"] is False          # never claims unverified results
    save_json({"verdicts": {"FAIL": 1}, "clips": [
        {"all_vs_parts_max_diff": 0.0, "issues": []},
        {"all_vs_parts_max_diff": 0.25, "issues": ["face: NaN/Inf"]}]}, processed / "verification_report.json")
    facts = R.verification_facts(processed)
    assert facts == {"ran": True, "max_diff": 0.25, "bad_values": 1, "fails": 1}


def test_excluded_clips_are_discovered(tmp_path):
    (tmp_path / "excluded_clips" / "no").mkdir(parents=True)
    (tmp_path / "excluded_clips" / "no" / "no_13.mp4").write_bytes(b"x")
    assert R.excluded_clips(tmp_path / "dataset") == [("no", "no_13")]


def test_report_renders_all_charts(tmp_path):
    processed = tmp_path / "processed_dataset"
    for cls in ("hello", "no"):
        for i in range(3):
            _metadata(processed, cls, f"{cls}_{i}")
    save_json({"total_features": 1692, "blocks": [
        {"name": "pose", "start": 0, "end": 132, "width": 132},
        {"name": "face", "start": 132, "end": 1566, "width": 1434},
        {"name": "left_hand", "start": 1566, "end": 1629, "width": 63},
        {"name": "right_hand", "start": 1629, "end": 1692, "width": 63}]}, processed / "FEATURE_LAYOUT.json")
    R.main(["--processed-dir", str(processed), "--dataset-dir", str(tmp_path / "dataset")])
    assert len(list((processed / "reports").glob("*.png"))) == 6


def test_observation_timeline_helpers():
    import numpy as np
    from signlang.dataset.observe import runs, timeline_rows
    assert list(runs([0, 1, 1, 0, 1])) == [(1, 2), (4, 1)]
    assert list(runs([])) == []
    rows = timeline_rows({"pose": np.ones(3), "hands": np.zeros((3, 2))})   # face not extracted
    assert [name for name, _ in rows] == ["pose", "left_hand", "right_hand"]
