"""
STAGE 2 — Fit the deployment model on 100% of the data.
=======================================================
Cross-validation (train.py) proved the recipe generalises (~0.91 OOF acc).  The
model you actually ship should use every clip, so this trains on ALL data with
the same augmentation + schedule, and keeps an Exponential Moving Average (EMA)
of the weights as the final model.  EMA gives a smoother, better-generalising
solution than the last raw step — important here because there is no held-out
set to early-stop on.

Run:
    python finetune.py                    # uses config defaults / CV settings
    python finetune.py --epochs 90 --init-from fold3   # warm-start from a fold

Outputs:
    checkpoints/model_final.pt            EMA weights + full meta (self-describing)
    checkpoints/model_meta.json           same meta without weights (for inference)
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
from config import (CONFIG, CKPT_DIR, LOG_DIR, METRICS_DIR, ensure_dirs,  # noqa: E402
                    to_dict)
from data import SignDataset, load_class_names, load_geometric  # noqa: E402
from model import build_model, count_params  # noqa: E402
from train import apply_overrides, evaluate, get_device, make_scheduler  # noqa: E402
from utils import Tee, load_json, save_json, seed_everything  # noqa: E402


class EMA:
    """Exponential moving average of model parameters, with decay warmup.

    A fixed decay of 0.999 needs ~1000 steps to forget the random init; on this
    small dataset a run is only a few hundred steps, so the EMA would lag badly.
    The warmup `min(decay, (1+step)/(10+step))` makes the average track closely
    early on and settle toward `decay` later."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return self.shadow


def build_meta(cfg, in_dim, classes, extra=None):
    meta = {"kind": "final", "framework": "pytorch",
            "classes": classes, "n_classes": len(classes),
            "in_dim": in_dim, "seq_len": cfg.feature.seq_len,
            "raw_dim": 1692, "config": to_dict(cfg),
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        meta.update(extra)
    return meta


def fit_final(cfg, args):
    ensure_dirs()
    log = Tee(LOG_DIR / "finetune.log")
    device = get_device()
    seed_everything(cfg.train.seed)

    class_names = load_class_names()
    spec = F.build_spec(cfg.feature)
    in_dim = F.feature_dim(cfg.feature, spec)
    clips, labels, stems = load_geometric(cfg.feature, spec)

    log("=" * 70)
    log(f"[finetune] device={device} arch={cfg.model.arch} clips={len(clips)} "
        f"D={in_dim} L={cfg.feature.seq_len} epochs={cfg.train.epochs} ema=0.999")

    full_ds = SignDataset(clips, labels, cfg.feature, spec, cfg.train,
                          augment_on=True, seed=cfg.train.seed)
    probe_ds = SignDataset(clips, labels, cfg.feature, spec, cfg.train,
                           augment_on=False)
    loader = DataLoader(full_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)
    probe = DataLoader(probe_ds, batch_size=64, shuffle=False, num_workers=0)

    model = build_model(in_dim, len(class_names), cfg.model).to(device)
    if args.init_from:
        ck = torch.load(CKPT_DIR / f"{args.init_from}.pt", map_location=device)
        model.load_state_dict(ck["state_dict"])
        log(f"[finetune] warm-started from {args.init_from} (val_acc {ck.get('val_acc')})")
    log(f"[finetune] params={count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    sched = make_scheduler(opt, cfg.train, cfg.train.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    ema = EMA(model, decay=0.999)

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
            run += loss.item() * x.size(0); seen += x.size(0)
        sched.step()
        if epoch % 10 == 0 or epoch == cfg.train.epochs - 1:
            shadow = build_model(in_dim, len(class_names), cfg.model).to(device)
            shadow.load_state_dict(ema.state_dict())
            _, _, _, ema_acc = evaluate(shadow, probe, device)
            _, _, _, raw_acc = evaluate(model, probe, device)
            log(f"    epoch {epoch:3d}  loss {run / max(1, seen):.3f}  "
                f"raw_acc {raw_acc:.3f}  ema_acc {ema_acc:.3f}  "
                f"lr {sched.get_last_lr()[0]:.2e}")

    # Pick whichever of {EMA, raw} fits the (no-aug) training clips better — the
    # EMA usually wins, but this guarantees the shipped model is never worse.
    ema_model = build_model(in_dim, len(class_names), cfg.model).to(device)
    ema_model.load_state_dict(ema.state_dict())
    _, _, _, ema_acc = evaluate(ema_model, probe, device)
    _, _, _, raw_acc = evaluate(model, probe, device)
    if ema_acc >= raw_acc:
        final_state, fit_acc, chosen = ema.state_dict(), ema_acc, "ema"
    else:
        final_state, fit_acc, chosen = copy.deepcopy(model.state_dict()), raw_acc, "raw"
    log(f"[finetune] probe fit acc: raw={raw_acc:.4f}  ema={ema_acc:.4f}  -> using {chosen}")

    meta = build_meta(cfg, in_dim, class_names,
                      extra={"fit_train_acc": round(fit_acc, 4), "weights": chosen,
                             "init_from": args.init_from,
                             "elapsed_sec": round(time.time() - t0, 1)})
    cv = METRICS_DIR / "cv_report.json"
    if cv.exists():
        rep = load_json(cv)
        meta["cv_mean_acc"] = rep.get("cv_mean_acc")
        meta["cv_oof_accuracy"] = rep.get("oof_accuracy")

    torch.save({"state_dict": final_state, **meta}, CKPT_DIR / "model_final.pt")
    save_json(meta, CKPT_DIR / "model_meta.json")
    log(f"[finetune] fit train acc = {fit_acc:.4f}  ({chosen} weights)")
    log(f"[finetune] saved -> {CKPT_DIR / 'model_final.pt'}")
    log(f"[finetune] meta  -> {CKPT_DIR / 'model_meta.json'}")
    log("[finetune] next: python export_optimize.py")
    log.close()
    return meta


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: fit final model on all data.")
    p.add_argument("--epochs", type=int)
    p.add_argument("--folds", type=int)        # accepted/ignored for parity
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seq-len", type=int)
    p.add_argument("--arch", choices=["bigru", "transformer"])
    p.add_argument("--hidden", type=int)
    p.add_argument("--use-face", action="store_true")
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--init-from", help="warm-start from a fold checkpoint name, e.g. fold3")
    return p.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    fit_final(apply_overrides(CONFIG, _args), _args)
