# 01 — Overview

## What this layer does

It turns the **verified landmark dataset** (`processed_dataset/`) into a working
**sign-language recogniser**: a small sequence model that reads a clip of
skeleton landmarks and outputs one of 12 word classes, plus everything needed to
run it on a video file or a live webcam.

The extraction project answered *"are the landmarks correct?"*. This project
answers *"given correct landmarks, what sign is being made?"* — without touching
the extraction format. Every clip is read as a raw `(T, 1692)` landmark array
and run through one shared feature transform, so **what the model trains on is
exactly what it sees at serve time** (no train/serve skew).

## The five stages

The pipeline is deliberately split into numbered, runnable stages:

| Stage | Script | Job | Key output |
|-------|--------|-----|------------|
| 1 | `train.py` | Stratified **k-fold cross-validation** — honest whole-dataset accuracy | `metrics/cv_report.json`, confusion matrix |
| 2 | `finetune.py` | Fit the **deployment model on 100%** of the data (EMA weights) | `checkpoints/model_final.pt` |
| 3 | `export_optimize.py` | Export to **ONNX** + int8, verify parity, measure latency | `exported/sign_model.onnx` |
| 4 | `infer_video.py` | Run the model on a **video file**, render a prediction overlay | `predictions/*_pred.mp4` |
| 5 | `infer_live.py` | **Live** webcam translation (record → process → delete chunks) | `predictions/live_transcript.json` |

Stages 1–3 are offline (training). Stages 4–5 are inference and share one
`SignPredictor` so the serving path is identical everywhere.

## The model

- **Input:** a fixed-length `(L, D)` feature sequence — `L = 64` frames,
  `D = 346` features/frame (default config).
- **Network:** a small **2-layer BiGRU** with attention pooling (a Transformer
  encoder is available via config). ~0.5 M parameters.
- **Output:** 12-way softmax over the word classes.
- **Measured:** ~**0.91** out-of-fold accuracy / macro-F1 (5-fold CV).

Skeleton features are low-dimensional, so the model is intentionally small and
heavily regularised (dropout, label smoothing, augmentation, EMA).

## The dataset (consumed read-only)

- **12 word classes**: `good, hello, help, hi, how_are_you, me,
  nice_to_meet_you, no, sorry, thank_you, yes, you`.
- ~**11 clips per class** (~130 usable clips after pruning).
- Source: `processed_dataset/landmarks_all/<class>/<stem>.npy`, each `(T, 1692)`,
  indexed by `manifest.csv` (rows with `verdict == FAIL` are skipped).

With so few clips per class, a single train/val/test split is unreliable — hence
k-fold CV for evaluation and a separate full-data fit for deployment.

## Tech stack

| Layer | Tool | Notes |
|-------|------|-------|
| Language | Python | 3.11 |
| Tensors / training | PyTorch | BiGRU / Transformer, AdamW, cosine schedule |
| Arrays / metrics | NumPy | folds, confusion matrix — no scikit-learn dependency |
| Deployment runtime | ONNX Runtime | CPU inference (fp32 + dynamic int8) |
| Detection (inference) | MediaPipe + OpenCV | reused from the extraction project |
| Charts | matplotlib | confusion / fold / history plots |

## Directory tree

```
training/
├── config.py            # ONE source of truth: paths, layout, all hyperparameters
├── features.py          # raw (T,1692) -> fixed (L,D) — shared by train AND inference
├── data.py              # dataset index, in-memory cache, augmentation, torch Dataset
├── model.py             # SignClassifier (BiGRU / Transformer) + pooling
├── utils.py             # seeding, stratified folds, metrics, JSON IO, logging
├── viz.py               # confusion / fold / history plots
│
├── train.py             # STAGE 1 — k-fold cross-validation
├── finetune.py          # STAGE 2 — fit final model on all data (EMA)
├── export_optimize.py   # STAGE 3 — export + optimize to ONNX
├── predictor.py         # shared inference: load model, landmarks -> sign
├── infer_video.py       # STAGE 4 — run on a video file + overlay
├── infer_live.py        # STAGE 5 — live webcam translation
│
├── artifacts/           # ALL outputs (gitignored, regenerable)
│   ├── checkpoints/     #   fold{k}.pt, model_final.pt, model_meta.json
│   ├── metrics/         #   cv_report.json + confusion / fold / history PNGs
│   ├── exported/        #   sign_model.onnx, sign_model.int8.onnx, model_meta.json
│   ├── predictions/     #   *_pred.mp4 / *_pred.json, live_transcript.json
│   ├── cache/           #   feature cache
│   ├── live_tmp/        #   transient live recordings (auto-deleted)
│   └── logs/            #   train.log, finetune.log
│
└── docs/                # THIS DOCUMENTATION
    └── *.md
```
