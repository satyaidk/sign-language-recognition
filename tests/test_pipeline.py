"""End-to-end: synthetic dataset -> k-fold CV -> final fit -> ONNX export -> predictor.

This is the whole learning + serving path on data with a known answer, run in a
temporary folder.  Takes ~20-40 s on a laptop CPU.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from signlang.config import default_config
from signlang.inference.predictor import SignPredictor
from signlang.inference.video import segment_at, timeline
from signlang.training.export import export
from signlang.training.finetune import fit_final
from signlang.training.train import run_cv
from tests.conftest import build_synthetic_dataset, make_clip


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train, fine-tune and export once for every test in this module."""
    paths = build_synthetic_dataset(tmp_path_factory.mktemp("pipeline"))
    cfg = default_config()
    cfg.model.hidden, cfg.train.epochs, cfg.train.folds = 32, 15, 2
    cfg.train.batch_size, cfg.train.warmup_epochs, cfg.train.early_stop_patience = 8, 1, 0
    report = run_cv(cfg, paths, make_plots=True)
    meta = fit_final(cfg, paths)
    exported = export(paths, quantize=True)
    return SimpleNamespace(paths=paths, report=report, meta=meta, exported=exported)


def test_cross_validation_learns_the_synthetic_classes(trained):
    r = trained.report
    assert r["n_clips"] == 30 and len(r["fold_val_acc"]) == 2
    assert r["oof_accuracy"] >= 0.9
    for name in ("cv_report.json", "confusion_matrix.png", "cv_folds.png", "history.png"):
        assert (trained.paths.metrics / name).exists()
    assert (trained.paths.checkpoints / "fold1.pt").exists()


def test_final_model_meta_is_self_describing(trained):
    meta = json.loads((trained.paths.checkpoints / "model_meta.json").read_text())
    assert meta["classes"] == ["alpha", "beta", "gamma"] and meta["in_dim"] == 346 and meta["seq_len"] == 64
    assert meta["cv_oof_accuracy"] == trained.report["oof_accuracy"]
    assert meta["weights"] in ("ema", "raw")


def test_onnx_export_matches_pytorch(trained):
    assert trained.exported["onnx_vs_torch_max_diff"] < 1e-3
    assert (trained.paths.exported / "sign_model.onnx").exists()
    assert (trained.paths.exported / "model_meta.json").exists()


def test_predictor_backends_agree_and_are_correct(trained):
    onnx = SignPredictor(trained.paths, backend="onnx")
    torch_ = SignPredictor(trained.paths, backend="torch")
    assert onnx.extractor_options().face is False           # model ignores the face -> skip it
    rng = np.random.default_rng(42)
    correct = 0
    for label in range(3):
        clip = make_clip(rng, 50, shift=0.12 * label)
        a, b = onnx.predict(clip), torch_.predict(clip)
        assert a["name"] == b["name"] and abs(a["conf"] - b["conf"]) < 1e-3
        correct += a["label"] == label
        robust = onnx.predict_robust(make_clip(rng, 120, shift=0.12 * label))
        assert robust["n_windows"] > 1 and 0 <= robust["margin"] <= 1
    assert correct == 3


def test_timeline_windows_cover_the_clip(trained):
    pred = SignPredictor(trained.paths)
    segs = timeline(make_clip(np.random.default_rng(0), 100), 30.0, pred, window=48, stride=12)
    assert segs[0]["start_frame"] == 0 and segs[-1]["end_frame"] == 100
    assert segment_at(segs, 50)["start_frame"] <= 50
    assert timeline(np.zeros((0, 1692), np.float32), 30.0, pred, 48, 12) == []


def test_missing_model_gives_a_clear_error(tmp_path):
    from signlang.config import get_paths
    with pytest.raises(FileNotFoundError):
        SignPredictor(get_paths(artifacts=tmp_path))
