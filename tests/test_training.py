"""L2: config, utils, feature transform, augmentation and model (synthetic data)."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from signlang.config import default_config, feature_config_from_dict, get_paths, model_config_from_dict
from signlang.training import features as F
from signlang.training.data import SignDataset, augment, load_class_names, load_geometric, load_index
from signlang.training.model import build_model, count_params
from signlang.utils import classification_report, parse_clip_ref, save_json, stratified_folds
from tests.conftest import make_clip


# ── config / paths ────────────────────────────────────────────────────────────
def test_paths_resolution_order(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNLANG_PROCESSED_DIR", str(tmp_path / "env"))
    p = get_paths(artifacts=tmp_path / "arts")
    assert p.processed == tmp_path / "env"                  # env var
    assert p.artifacts == tmp_path / "arts"                 # explicit argument
    assert p.dataset.name == "dataset"                      # repo default
    assert p.manifest == tmp_path / "env" / "manifest.csv"


def test_configs_roundtrip_through_json_and_ignore_unknown_keys():
    fc = feature_config_from_dict({"seq_len": 32, "pose_idx": [0, 11, 12], "not_a_field": 1})
    assert fc.seq_len == 32 and fc.pose_idx == (0, 11, 12)
    assert model_config_from_dict({"arch": "transformer", "bogus": True}).arch == "transformer"


def test_default_config_is_fresh_each_time():
    a, b = default_config(), default_config()
    a.train.epochs = 1
    assert b.train.epochs == 120


# ── utils ─────────────────────────────────────────────────────────────────────
def test_stratified_folds_partition_and_balance():
    labels = np.repeat(np.arange(12), 11)                   # 12 classes x 11 clips (like the dataset)
    folds = stratified_folds(labels, 5, seed=3)
    flat = np.concatenate(folds)
    assert sorted(flat.tolist()) == list(range(len(labels)))   # every clip exactly once
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 1                        # was 36 vs 24 before the fix
    for f in folds:                                            # every class in every fold
        assert len(set(labels[f].tolist())) == 12


def test_stratified_folds_validates_arguments():
    with pytest.raises(ValueError):
        stratified_folds([0, 1, 0], 1)
    with pytest.raises(ValueError):
        stratified_folds([0, 1], 5)


def test_classification_report_against_hand_computed_values():
    rep = classification_report([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], ["a", "b", "c"])
    assert rep["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert rep["per_class"]["b"] == {"precision": pytest.approx(2 / 3, abs=1e-4), "recall": 1.0,
                                     "f1": pytest.approx(0.8, abs=1e-4), "support": 2}
    assert rep["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_parse_clip_ref():
    assert parse_clip_ref("hello\\hello_01") == ("hello", "hello_01")
    with pytest.raises(ValueError):
        parse_clip_ref("hello")


# ── features ──────────────────────────────────────────────────────────────────
@pytest.fixture
def fc():
    return default_config().feature


def test_feature_spec_default_dimensions(fc):
    spec = F.build_spec(fc)
    assert (spec.n_points, spec.n_vis, spec.n_mask) == (55, 13, 3)
    assert F.feature_dim(fc, spec) == 346               # 165 coords + 13 vis + 165 vel + 3 masks


def test_feature_dim_follows_the_config(fc):
    no_vel = copy.deepcopy(fc)
    no_vel.add_velocity = False
    assert F.feature_dim(no_vel) == 346 - 165
    face = copy.deepcopy(fc)
    face.use_face, face.face_idx = True, (0, 1, 2)
    assert F.feature_dim(face) == 346 + 2 * 9 + 1       # +3 points x3 (coords, velocity) +1 mask


def test_transform_shape_determinism_and_lengths(fc, rng, clip):
    a, b = F.to_features(clip, fc), F.to_features(clip, fc)
    assert a.shape == (64, 346) and a.dtype == np.float32 and np.array_equal(a, b)
    for T in (1, 14, 286, 600):
        assert F.to_features(make_clip(rng, T), fc).shape == (64, 346)
    assert F.to_features(np.zeros((0, 1692), np.float32), fc).shape == (64, 346)


def test_transform_rejects_partial_layouts(fc):
    with pytest.raises(ValueError):
        F.to_features(np.zeros((10, 132 + 126), np.float32), fc)


def test_normalisation_is_translation_and_scale_invariant(fc, clip):
    moved = clip.copy()
    T = len(clip)
    for s, e, ch in ((0, 132, 4), (132, 1566, 3), (1566, 1692, 3)):
        blk = moved[:, s:e].reshape(T, -1, ch)
        present = np.abs(blk[..., :3]).sum(-1, keepdims=True) > 0
        blk[..., :2] = np.where(present, blk[..., :2] * 1.5 + 0.1, 0)
        moved[:, s:e] = blk.reshape(T, -1)
    a = F.to_features(clip, fc)[:, :165].reshape(64, 55, 3)[..., :2]
    b = F.to_features(moved, fc)[:, :165].reshape(64, 55, 3)[..., :2]
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_missing_hand_is_zero_not_a_ghost(fc, clip):
    feats = F.to_features(clip, fc)                     # the clip has no left hand
    spec = F.build_spec(fc)
    s, e = spec.seg["left_hand"]
    assert not feats[:, s * 3:e * 3].any()
    assert feats[:, -2].sum() == 0 and feats[:, -1].sum() == 64    # left / right presence masks


# ── augmentation ──────────────────────────────────────────────────────────────
def test_absent_hand_stays_zero_after_augmentation(fc, clip):
    spec = F.build_spec(fc)
    tc = copy.deepcopy(default_config().train)
    tc.aug_flip_prob = 0.0
    co, vi, pr = F.geometric(clip, fc, spec)
    s, e = spec.seg["left_hand"]
    for seed in range(30):
        co2, _, _ = augment(co, vi, pr, tc, spec, np.random.default_rng(seed))
        assert not co2[:, s:e].any()


def test_flip_is_an_involution_and_swaps_hands(fc, clip):
    spec = F.build_spec(fc)
    co, vi, pr = F.geometric(clip, fc, spec)
    once = co[:, spec.point_perm].copy()
    once[..., 0] *= -1
    twice = once[:, spec.point_perm].copy()
    twice[..., 0] *= -1
    np.testing.assert_allclose(co, twice)

    tc = copy.deepcopy(default_config().train)
    tc.aug_flip_prob = 1.0
    flipped, _, pres = augment(co, vi, pr, tc, spec, np.random.default_rng(1))
    s, e = spec.seg["left_hand"]
    assert np.abs(flipped[:, s:e]).sum() > 0 and pres["left_hand"].sum() > 0 and pres["right_hand"].sum() == 0


# ── dataset loading ───────────────────────────────────────────────────────────
def test_load_index_and_geometry(synthetic_dataset, fc):
    assert load_class_names(synthetic_dataset) == ["alpha", "beta", "gamma"]
    assert len(load_index(synthetic_dataset)) == 30
    clips, labels, stems = load_geometric(synthetic_dataset, fc, F.build_spec(fc))
    assert len(clips) == 30 and sorted(set(labels.tolist())) == [0, 1, 2] and stems[0] == "alpha/alpha_00"


def test_load_class_names_rejects_gapped_labels(synthetic_dataset):
    save_json({"a": 0, "b": 2}, synthetic_dataset.classes_json)
    with pytest.raises(ValueError):
        load_class_names(synthetic_dataset)


def test_sign_dataset_items(synthetic_dataset, fc):
    spec = F.build_spec(fc)
    clips, labels, _ = load_geometric(synthetic_dataset, fc, spec)
    ds = SignDataset(clips, labels, fc, spec, default_config().train, augment_on=True, seed=0)
    x, y = ds[0]
    assert tuple(x.shape) == (64, 346) and x.dtype == torch.float32 and y == 0


# ── model ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("arch,pool", [("bigru", "attention"), ("bigru", "mean"), ("transformer", "attention"),
                                       ("transformer", "last")])
def test_model_shapes(arch, pool):
    cfg = default_config().model
    cfg.arch, cfg.pool = arch, pool
    m = build_model(346, 12, cfg, max_len=64).eval()
    assert tuple(m(torch.randn(3, 64, 346)).shape) == (3, 12)


def test_default_bigru_size_and_small_transformer():
    cfg = default_config().model
    assert count_params(build_model(346, 12, cfg)) == 543_553
    cfg.arch = "transformer"
    assert count_params(build_model(346, 12, cfg, max_len=64)) < 400_000   # was 836k with a 4096-row table


def test_model_rejects_unknown_options():
    cfg = default_config().model
    cfg.arch = "lstm"
    with pytest.raises(ValueError):
        build_model(346, 12, cfg)
    cfg.arch, cfg.pool = "bigru", "max"
    with pytest.raises(ValueError):
        build_model(346, 12, cfg)


# ── schedule + EMA ────────────────────────────────────────────────────────────
def test_scheduler_warms_up_then_decays_to_the_floor():
    from signlang.training.train import make_scheduler
    tc = default_config().train
    tc.epochs, tc.warmup_epochs, tc.lr, tc.min_lr = 20, 4, 1e-3, 1e-5
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=tc.lr)
    sched = make_scheduler(opt, tc, tc.epochs)
    lrs = []
    for _ in range(tc.epochs):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(tc.lr / 4) and lrs[3] == pytest.approx(tc.lr)   # linear warm-up
    assert all(a >= b for a, b in zip(lrs[3:], lrs[4:]))                            # then non-increasing
    assert min(lrs) >= tc.min_lr - 1e-12


def test_ema_tracks_the_model_with_decay_warmup():
    from signlang.training.finetune import EMA
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = EMA(model, decay=0.999)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model, step=1)                      # early steps use a small decay (2/11)
    assert ema.state_dict()["weight"].mean().item() == pytest.approx(1 - 2 / 11, abs=1e-6)
    for step in range(2, 400):
        ema.update(model, step)
    assert ema.state_dict()["weight"].mean().item() > 0.99
