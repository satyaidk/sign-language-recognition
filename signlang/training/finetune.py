"""
STAGE 2 — Fit the deployment model on 100% of the data (EMA weights).
=====================================================================
Cross-validation (stage 1) measured how well the recipe generalises.  The model
you ship should learn from every clip, so this trains on ALL data with the same
augmentation and schedule and keeps an Exponential Moving Average (EMA) of the
weights.  EMA gives a smoother solution than the last optimiser step — which
matters because there is no held-out set to early-stop on here.

Run:
    python -m signlang finetune
    python -m signlang finetune --epochs 90 --init-from fold3     # warm-start from a CV fold

Outputs:
    checkpoints/model_final.pt      EMA (or raw) weights + full meta (self-describing)
    checkpoints/model_meta.json     the same meta without weights
"""
from __future__ import annotations

import argparse
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from signlang.config import Config, Paths, add_path_args, default_config, paths_from_args, to_dict
from signlang.training import features as F
from signlang.training.data import SignDataset, load_class_names, load_geometric
from signlang.training.model import build_model, count_params
from signlang.training.train import add_config_args, apply_overrides, evaluate, get_device, make_scheduler
from signlang.utils import Tee, load_json, save_json, seed_everything


class EMA:
    """Exponential moving average of model parameters, with decay warm-up.

    A fixed decay of 0.999 needs ~1000 steps to forget the random init; a run on
    this small dataset is only a few hundred steps, so the EMA would lag badly.
    The warm-up ``min(decay, (1 + step) / (10 + step))`` tracks closely early on
    and settles toward ``decay`` later.  Non-float buffers are copied as-is.
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model, step: int) -> None:
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self) -> dict:
        return self.shadow


def build_meta(cfg: Config, in_dim: int, classes: list, extra: dict | None = None) -> dict:
    meta = {"kind": "final", "framework": "pytorch", "classes": classes, "n_classes": len(classes),
            "in_dim": in_dim, "seq_len": cfg.feature.seq_len, "raw_dim": 1692,
            "config": to_dict(cfg), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    meta.update(extra or {})
    return meta


def fit_final(cfg: Config, paths: Paths, init_from: str | None = None) -> dict:
    paths.ensure_artifact_dirs()
    log = Tee(paths.logs / "finetune.log")
    device = get_device()
    seed_everything(cfg.train.seed)
    try:
        class_names = load_class_names(paths)
        spec = F.build_spec(cfg.feature)
        in_dim = F.feature_dim(cfg.feature, spec)
        clips, labels, _ = load_geometric(paths, cfg.feature, spec)
        log("=" * 70)
        log(f"[finetune] device={device} arch={cfg.model.arch} clips={len(clips)} D={in_dim} "
            f"L={cfg.feature.seq_len} epochs={cfg.train.epochs} ema={cfg.train.ema_decay}")

        full_ds = SignDataset(clips, labels, cfg.feature, spec, cfg.train, augment_on=True, seed=cfg.train.seed)
        probe_ds = SignDataset(clips, labels, cfg.feature, spec, cfg.train, augment_on=False)
        loader = DataLoader(full_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)
        probe = DataLoader(probe_ds, batch_size=64, shuffle=False, num_workers=0)

        def new_model():
            return build_model(in_dim, len(class_names), cfg.model, max_len=cfg.feature.seq_len).to(device)

        model = new_model()
        if init_from:
            ck_path = paths.checkpoints / f"{init_from}.pt"
            if not ck_path.exists():
                raise FileNotFoundError(f"{ck_path} not found — run `python -m signlang train` first")
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["state_dict"])
            log(f"[finetune] warm-started from {init_from} (val_acc {ck.get('val_acc')})")
        log(f"[finetune] params={count_params(model):,}")

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        sched = make_scheduler(opt, cfg.train, cfg.train.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
        ema = EMA(model, decay=cfg.train.ema_decay)

        t0, step = time.time(), 0
        for epoch in range(cfg.train.epochs):
            model.train()
            run, seen = 0.0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                if cfg.train.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                opt.step()
                step += 1
                ema.update(model, step)
                run += loss.item() * x.size(0)
                seen += x.size(0)
            sched.step()
            if epoch % 10 == 0 or epoch == cfg.train.epochs - 1:
                shadow = new_model()
                shadow.load_state_dict(ema.state_dict())
                log(f"    epoch {epoch:3d}  loss {run / max(1, seen):.3f}  "
                    f"raw_acc {evaluate(model, probe, device)[3]:.3f}  "
                    f"ema_acc {evaluate(shadow, probe, device)[3]:.3f}  lr {sched.get_last_lr()[0]:.2e}")

        # Ship whichever of {EMA, raw} fits the (un-augmented) training clips better;
        # EMA usually wins, this guarantees the shipped model is never the worse one.
        ema_model = new_model()
        ema_model.load_state_dict(ema.state_dict())
        ema_acc = evaluate(ema_model, probe, device)[3]
        raw_acc = evaluate(model, probe, device)[3]
        if ema_acc >= raw_acc:
            final_state, fit_acc, chosen = ema.state_dict(), ema_acc, "ema"
        else:
            final_state, fit_acc, chosen = copy.deepcopy(model.state_dict()), raw_acc, "raw"
        log(f"[finetune] training-fit acc: raw={raw_acc:.4f} ema={ema_acc:.4f} -> using {chosen} "
            "(fit on training clips — NOT a generalisation metric; see cv_report.json)")

        meta = build_meta(cfg, in_dim, class_names,
                          extra={"fit_train_acc": round(fit_acc, 4), "weights": chosen,
                                 "init_from": init_from, "elapsed_sec": round(time.time() - t0, 1)})
        cv = paths.metrics / "cv_report.json"
        if cv.exists():
            rep = load_json(cv)
            meta["cv_mean_acc"] = rep.get("cv_mean_acc")
            meta["cv_oof_accuracy"] = rep.get("oof_accuracy")
        torch.save({"state_dict": {k: v.cpu() for k, v in final_state.items()}, **meta},
                   paths.checkpoints / "model_final.pt")
        save_json(meta, paths.checkpoints / "model_meta.json")
        log(f"[finetune] saved -> {paths.checkpoints / 'model_final.pt'}")
        log("[finetune] next: python -m signlang export")
        return meta
    finally:
        log.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="signlang finetune", description="Stage 2: fit the final model on all data.")
    add_path_args(p, processed=True, artifacts=True)
    add_config_args(p)
    p.add_argument("--init-from", help="warm-start from a fold checkpoint, e.g. fold3")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        fit_final(apply_overrides(default_config(), args), paths_from_args(args), args.init_from)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
