"""
STAGE 1 — Honest evaluation via stratified k-fold cross-validation.
===================================================================
With ~11 clips per class a single train/val/test split is unreliable, so every
clip is placed in exactly one validation fold and the out-of-fold (OOF)
predictions give an accuracy + confusion matrix over the WHOLE dataset.  The
per-fold best weights are saved for inspection; the model you deploy is trained
on 100% of the data by ``finetune`` (stage 2).

Run:
    python -m signlang train                          # 5-fold CV, config defaults
    python -m signlang train --folds 5 --epochs 120 --arch bigru
    python -m signlang train --use-face               # include the face block
    python -m signlang train --no-aug                 # ablate augmentation

Outputs (under artifacts/):
    checkpoints/fold{k}.pt        best weights per fold
    metrics/cv_report.json        OOF accuracy, per-class P/R/F1, per-fold accuracy
    metrics/confusion_matrix.png  OOF confusion matrix
    metrics/cv_folds.png          per-fold accuracy bars
    metrics/history.png           loss / val-accuracy curves
    logs/train.log
"""
from __future__ import annotations

import argparse
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from signlang.config import Config, Paths, add_path_args, default_config, paths_from_args, to_dict
from signlang.training import features as F
from signlang.training import viz
from signlang.training.data import SignDataset, load_class_names, load_geometric
from signlang.training.model import build_model, count_params
from signlang.utils import Tee, classification_report, save_json, seed_everything, stratified_folds


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_scheduler(optimizer, tcfg, total_epochs):
    """Linear warm-up, then cosine decay down to ``min_lr``."""
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


def train_model(train_ds, val_ds, in_dim, n_classes, cfg: Config, device, log, seed=0, tag=""):
    """Train one model with early stopping on validation accuracy.
    Returns ``(best_state, history, best_val_acc, best_val_preds)``."""
    seed_everything(seed)
    tcfg = cfg.train
    model = build_model(in_dim, n_classes, cfg.model, max_len=cfg.feature.seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = make_scheduler(opt, tcfg, tcfg.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.label_smoothing)
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

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
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)
        sched.step()
        history["train_loss"].append(run_loss / max(1, seen))

        vp, _, _, vacc = evaluate(model, val_loader, device, criterion)
        history["val_acc"].append(vacc)
        if vacc > best_acc + 1e-4:
            best_acc, best_state, best_preds, patience = vacc, copy.deepcopy(model.state_dict()), vp, 0
        else:
            patience += 1
        if epoch % 10 == 0 or epoch == tcfg.epochs - 1:
            log(f"    {tag} epoch {epoch:3d}  loss {history['train_loss'][-1]:.3f}  "
                f"val_acc {vacc:.3f}  best {best_acc:.3f}  lr {sched.get_last_lr()[0]:.2e}")
        if tcfg.early_stop_patience and patience >= tcfg.early_stop_patience:
            log(f"    {tag} early stop at epoch {epoch} (best val_acc {best_acc:.3f})")
            break
    return best_state, history, best_acc, best_preds


def run_cv(cfg: Config, paths: Paths, make_plots: bool = True) -> dict:
    """k-fold CV over the whole dataset -> OOF report (also written to disk)."""
    paths.ensure_artifact_dirs()
    log = Tee(paths.logs / "train.log")
    device = get_device()
    seed_everything(cfg.train.seed)
    try:
        class_names = load_class_names(paths)
        spec = F.build_spec(cfg.feature)
        in_dim = F.feature_dim(cfg.feature, spec)
        clips, labels, stems = load_geometric(paths, cfg.feature, spec)

        log("=" * 70)
        log(f"[train] device={device}  arch={cfg.model.arch}  clips={len(clips)}  classes={len(class_names)}")
        log(f"[train] feature D={in_dim}  L={cfg.feature.seq_len}  pose={cfg.feature.use_pose} "
            f"hands={cfg.feature.use_hands} face={cfg.feature.use_face} vel={cfg.feature.add_velocity}")
        n_params = count_params(build_model(in_dim, len(class_names), cfg.model, cfg.feature.seq_len))
        log(f"[train] params={n_params:,}  folds={cfg.train.folds}  epochs={cfg.train.epochs}  "
            f"aug={cfg.train.aug}")

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
            torch.save({"state_dict": state, "in_dim": in_dim, "seq_len": cfg.feature.seq_len,
                        "classes": class_names, "config": to_dict(cfg), "fold": k, "val_acc": best_acc},
                       paths.checkpoints / f"fold{k + 1}.pt")
            log(f"[fold {k + 1}] best val_acc = {best_acc:.4f}")

        rep = classification_report(labels, oof_pred, class_names)
        mean, std = float(np.mean(fold_accs)), float(np.std(fold_accs))
        log("\n" + "=" * 70)
        log(f"[CV] per-fold val acc: {[round(a, 3) for a in fold_accs]}")
        log(f"[CV] mean {mean:.4f} +/- {std:.4f}   |   OOF accuracy {rep['accuracy']:.4f}   "
            f"macro-F1 {rep['macro_f1']:.4f}")
        log("[CV] per-class F1: " + "  ".join(f"{c}={rep['per_class'][c]['f1']:.2f}" for c in class_names))

        report = {"oof_accuracy": rep["accuracy"], "oof_macro_f1": rep["macro_f1"],
                  "cv_mean_acc": round(mean, 4), "cv_std_acc": round(std, 4),
                  "fold_val_acc": [round(a, 4) for a in fold_accs],
                  "per_class": rep["per_class"], "confusion_matrix": rep["confusion_matrix"],
                  "classes": class_names, "in_dim": in_dim, "config": to_dict(cfg),
                  "n_clips": len(clips), "elapsed_sec": round(time.time() - t0, 1)}
        save_json(report, paths.metrics / "cv_report.json")
        if make_plots:
            viz.plot_confusion(rep["confusion_matrix"], class_names, paths.metrics / "confusion_matrix.png",
                               title=f"OOF confusion (acc {rep['accuracy']:.2f})")
            viz.plot_cv_folds(fold_accs, paths.metrics / "cv_folds.png")
            viz.plot_history(histories, paths.metrics / "history.png")
        log(f"[CV] wrote metrics -> {paths.metrics}")
        log(f"[CV] done in {report['elapsed_sec']}s. Next: python -m signlang finetune")
        return report
    finally:
        log.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def add_config_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Command-line overrides shared by train and finetune."""
    p.add_argument("--epochs", type=int)
    p.add_argument("--folds", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seq-len", type=int)
    p.add_argument("--arch", choices=["bigru", "transformer"])
    p.add_argument("--hidden", type=int)
    p.add_argument("--use-face", action="store_true", help="include the face block")
    p.add_argument("--no-aug", action="store_true", help="disable augmentation")
    p.add_argument("--seed", type=int)
    return p


def apply_overrides(cfg: Config, args) -> Config:
    for attr, section, field_ in (("epochs", "train", "epochs"), ("folds", "train", "folds"),
                                  ("batch_size", "train", "batch_size"), ("lr", "train", "lr"),
                                  ("seed", "train", "seed"), ("seq_len", "feature", "seq_len"),
                                  ("arch", "model", "arch"), ("hidden", "model", "hidden")):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(getattr(cfg, section), field_, value)
    if getattr(args, "use_face", False):
        cfg.feature.use_face = True
    if getattr(args, "no_aug", False):
        cfg.train.aug = False
    return cfg


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="signlang train", description="Stage 1: k-fold cross-validation.")
    add_path_args(p, processed=True, artifacts=True)
    return add_config_args(p).parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        run_cv(apply_overrides(default_config(), args), paths_from_args(args))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
