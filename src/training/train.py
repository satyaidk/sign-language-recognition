"""
STAGE 1 — Proper training via stratified k-fold cross-validation.
=================================================================
With ~11 clips per class a single train/val/test split is unreliable, so we use
stratified k-fold CV: every clip is in exactly one validation fold, and the
out-of-fold (OOF) predictions give an honest accuracy + confusion matrix over
the WHOLE dataset.  Per-fold best weights are saved so you can inspect them;
the deployment model is trained on 100% of the data by `finetune.py`.

Run:
    python src/training/train.py                       # 5-fold CV with config defaults
    python src/training/train.py --folds 5 --epochs 120 --arch bigru
    python src/training/train.py --use-face            # include the face block
    python src/training/train.py --no-aug              # ablate augmentation

Outputs (under artifacts/):
    checkpoints/fold{k}.pt        best weights per fold
    metrics/cv_report.json        OOF accuracy, per-class P/R/F1, per-fold accs
    metrics/confusion_matrix.png  OOF confusion matrix
    metrics/cv_folds.png          per-fold accuracy bars
    metrics/history.png           loss / val-acc curves
    logs/train.log
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import viz  # noqa: E402
from config import (CONFIG, CKPT_DIR, LOG_DIR, METRICS_DIR, ensure_dirs,  # noqa: E402
                    to_dict)
from data import SignDataset, load_class_names, load_geometric  # noqa: E402
from model import build_model, count_params  # noqa: E402
from utils import (Tee, classification_report, save_json, seed_everything,  # noqa: E402
                   stratified_folds)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_scheduler(optimizer, tcfg, total_epochs):
    warmup, base_lr, min_lr = tcfg.warmup_epochs, tcfg.lr, tcfg.min_lr
    floor = min_lr / base_lr

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        prog = (epoch - warmup) / max(1, total_epochs - warmup)
        return max(floor, 0.5 * (1.0 + math.cos(math.pi * prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    preds, targets, total_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * x.size(0)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(y.cpu().numpy())
        n += x.size(0)
    preds = np.concatenate(preds) if preds else np.array([], int)
    targets = np.concatenate(targets) if targets else np.array([], int)
    acc = float((preds == targets).mean()) if n else 0.0
    return preds, targets, (total_loss / n if n else 0.0), acc


def train_model(train_ds, val_ds, in_dim, n_classes, cfg, device, log,
                seed=0, tag="", probe_ds=None):
    """Train one model. Returns (best_state, history, best_val_acc, val_preds).
    If val_ds is None, trains for the full schedule and returns the last state
    (used by finetune.py); `probe_ds` is an optional no-shuffle monitor set."""
    seed_everything(seed)
    tcfg = cfg.train
    model = build_model(in_dim, n_classes, cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = make_scheduler(opt, tcfg, tcfg.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.label_smoothing)

    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True,
                              drop_last=False, num_workers=0)
    val_loader = (DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
                  if val_ds is not None else None)
    monitor = val_loader or (DataLoader(probe_ds, batch_size=64, shuffle=False)
                             if probe_ds is not None else None)

    history = {"train_loss": [], "val_acc": []}
    best_acc, best_state, best_preds, patience = -1.0, None, None, 0

    for epoch in range(tcfg.epochs):
        model.train()
        run_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            if tcfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            run_loss += loss.item() * x.size(0); seen += x.size(0)
        sched.step()
        train_loss = run_loss / max(1, seen)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            vp, _, vloss, vacc = evaluate(model, val_loader, device, criterion)
            history["val_acc"].append(vacc)
            improved = vacc > best_acc + 1e-4
            if improved:
                best_acc, best_state, best_preds, patience = vacc, \
                    copy.deepcopy(model.state_dict()), vp, 0
            else:
                patience += 1
            if epoch % 10 == 0 or epoch == tcfg.epochs - 1:
                log(f"    {tag} epoch {epoch:3d}  loss {train_loss:.3f}  "
                    f"val_acc {vacc:.3f}  best {best_acc:.3f}  lr {sched.get_last_lr()[0]:.2e}")
            if tcfg.early_stop_patience and patience >= tcfg.early_stop_patience:
                log(f"    {tag} early stop at epoch {epoch} (best val_acc {best_acc:.3f})")
                break
        else:
            if monitor is not None and (epoch % 10 == 0 or epoch == tcfg.epochs - 1):
                _, _, _, macc = evaluate(model, monitor, device)
                log(f"    {tag} epoch {epoch:3d}  loss {train_loss:.3f}  train_acc {macc:.3f}")

    if val_loader is None:
        best_state = copy.deepcopy(model.state_dict())
        best_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
    return best_state, history, best_acc, best_preds


def run_cv(cfg, args):
    ensure_dirs()
    log = Tee(LOG_DIR / "train.log")
    device = get_device()
    seed_everything(cfg.train.seed)

    class_names = load_class_names()
    spec = F.build_spec(cfg.feature)
    in_dim = F.feature_dim(cfg.feature, spec)
    clips, labels, stems = load_geometric(cfg.feature, spec)

    log("=" * 70)
    log(f"[train] device={device}  arch={cfg.model.arch}  clips={len(clips)}  "
        f"classes={len(class_names)}")
    log(f"[train] feature D={in_dim}  L={cfg.feature.seq_len}  "
        f"pose={cfg.feature.use_pose} hands={cfg.feature.use_hands} "
        f"face={cfg.feature.use_face} vel={cfg.feature.add_velocity}")
    log(f"[train] params={count_params(build_model(in_dim, len(class_names), cfg.model)):,}  "
        f"folds={cfg.train.folds}  epochs={cfg.train.epochs}  aug={cfg.train.aug}")

    folds = stratified_folds(labels, cfg.train.folds, seed=cfg.train.seed)
    oof_pred = -np.ones(len(clips), dtype=int)
    fold_accs, histories = [], []
    t0 = time.time()

    for k, val_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(clips)), val_idx)
        train_ds = SignDataset(clips, labels, cfg.feature, spec, cfg.train,
                               indices=train_idx, augment_on=True, seed=cfg.train.seed + k)
        val_ds = SignDataset(clips, labels, cfg.feature, spec, cfg.train,
                             indices=val_idx, augment_on=False)
        log(f"\n[fold {k + 1}/{cfg.train.folds}]  train={len(train_idx)}  val={len(val_idx)}")
        state, hist, best_acc, val_preds = train_model(
            train_ds, val_ds, in_dim, len(class_names), cfg, device, log,
            seed=cfg.train.seed + k, tag=f"f{k + 1}")
        oof_pred[val_idx] = val_preds
        fold_accs.append(best_acc)
        histories.append(hist)
        torch.save({"state_dict": state, "in_dim": in_dim,
                    "classes": class_names, "config": to_dict(cfg),
                    "fold": k, "val_acc": best_acc},
                   CKPT_DIR / f"fold{k + 1}.pt")
        log(f"[fold {k + 1}] best val_acc = {best_acc:.4f}  -> {CKPT_DIR / f'fold{k + 1}.pt'}")

    # ── Out-of-fold aggregate (honest whole-dataset evaluation) ──
    rep = classification_report(labels, oof_pred, class_names)
    mean, std = float(np.mean(fold_accs)), float(np.std(fold_accs))
    log("\n" + "=" * 70)
    log(f"[CV] per-fold val acc: {[round(a, 3) for a in fold_accs]}")
    log(f"[CV] mean {mean:.4f} +/- {std:.4f}   |   OOF accuracy {rep['accuracy']:.4f}   "
        f"macro-F1 {rep['macro_f1']:.4f}")
    log("[CV] per-class F1: " + "  ".join(
        f"{c}={rep['per_class'][c]['f1']:.2f}" for c in class_names))

    report = {"oof_accuracy": rep["accuracy"], "oof_macro_f1": rep["macro_f1"],
              "cv_mean_acc": round(mean, 4), "cv_std_acc": round(std, 4),
              "fold_val_acc": [round(a, 4) for a in fold_accs],
              "per_class": rep["per_class"], "confusion_matrix": rep["confusion_matrix"],
              "classes": class_names, "in_dim": in_dim, "config": to_dict(cfg),
              "n_clips": len(clips), "elapsed_sec": round(time.time() - t0, 1)}
    save_json(report, METRICS_DIR / "cv_report.json")
    viz.plot_confusion(rep["confusion_matrix"], class_names,
                       METRICS_DIR / "confusion_matrix.png",
                       title=f"OOF confusion (acc {rep['accuracy']:.2f})")
    viz.plot_cv_folds(fold_accs, METRICS_DIR / "cv_folds.png")
    viz.plot_history(histories, METRICS_DIR / "history.png")
    log(f"[CV] wrote metrics -> {METRICS_DIR}")
    log(f"[CV] done in {report['elapsed_sec']}s. Next: python src/training/finetune.py")
    log.close()
    return report


def apply_overrides(cfg, args):
    if args.epochs is not None: cfg.train.epochs = args.epochs
    if args.folds is not None: cfg.train.folds = args.folds
    if args.batch_size is not None: cfg.train.batch_size = args.batch_size
    if args.lr is not None: cfg.train.lr = args.lr
    if args.seq_len is not None: cfg.feature.seq_len = args.seq_len
    if args.arch is not None: cfg.model.arch = args.arch
    if args.hidden is not None: cfg.model.hidden = args.hidden
    if args.use_face: cfg.feature.use_face = True
    if args.no_aug: cfg.train.aug = False
    if args.seed is not None: cfg.train.seed = args.seed
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: k-fold CV training.")
    p.add_argument("--epochs", type=int)
    p.add_argument("--folds", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seq-len", type=int)
    p.add_argument("--arch", choices=["bigru", "transformer"])
    p.add_argument("--hidden", type=int)
    p.add_argument("--use-face", action="store_true")
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--seed", type=int)
    return p.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    run_cv(apply_overrides(CONFIG, _args), _args)
