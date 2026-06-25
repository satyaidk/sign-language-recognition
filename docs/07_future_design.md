# 07 — Future System Design

The current system ends at a verified landmark dataset. This is the suggested
design for the next phases: **training** and **real-time inference**. None of it
requires changing the extraction format — it consumes `landmarks_all/*.npy` and
`manifest.csv` read-only.

## Proposed end-to-end system

```
            TRAINING (offline)                         INFERENCE (real-time)
 ┌───────────────────────────────────┐      ┌───────────────────────────────────┐
 │ landmarks_all/*.npy  +  manifest  │      │ webcam frame                       │
 │            │                       │      │     │ landmark_detector (live)     │
 │            ▼                       │      │     ▼                              │
 │  preprocessing                     │      │  same preprocessing                │
 │   • normalise (center+scale)       │      │   (identical transform)            │
 │   • temporal resample -> fixed L   │      │     │                              │
 │   • augment (train only)           │      │     ▼                              │
 │            │                       │      │  sliding-window buffer (L frames)  │
 │            ▼                       │      │     │                              │
 │  sequence model (GRU/Transformer)  │ ───► │  same trained model -> softmax     │
 │            │                       │ wts  │     │                              │
 │            ▼                       │      │     ▼                              │
 │  train/val/test (manifest split)   │      │  predicted sign + confidence       │
 └───────────────────────────────────┘      └───────────────────────────────────┘
```

The key property: **training and inference share one preprocessing function and
one model**, so what you train on is exactly what you serve.

## 1. Preprocessing (new module, e.g. `features.py`)

A single function `to_features(all_vec) -> (L, D)` used by both training and
inference:

- **Spatial normalisation** (translation/scale invariance): per frame, subtract
  a reference point (mid-shoulder from pose) and divide by a reference length
  (shoulder width). Makes signs invariant to where/how big the signer is.
- **Temporal resampling**: clips are 14–286 frames; resample each to a fixed
  length `L` (e.g. 48) by uniform sampling or interpolation. Alternatively pad +
  mask for a variable-length model.
- **Mask channel**: append the presence mask so the model can tell "hand absent"
  from "hand at origin".
- Optionally **drop the face block** (1434 of 1692 dims) for a lighter model if
  facial expression isn't needed for the target signs.

## 2. Model

- Baseline: **2-layer BiGRU** or **small Transformer encoder** over `(L, D)` →
  mean/attention pool → linear → 12-way softmax. Skeleton-based recognition needs
  far less data/compute than pixel models.
- Loss: cross-entropy; report per-class accuracy + confusion matrix (the heatmap
  already shows which signs share hand structure and may confuse).

## 3. Data splitting & balancing

- Drive the split from `manifest.csv` (stratify by `class`). With ~10–14 clips
  per class, use **k-fold cross-validation** rather than a single split.
- Augmentation (train only): small rotations/scales/jitter on coordinates, time
  warping, horizontal flip **with L/R hand-slot swap** (turns a right-handed
  example into a left-handed one).

## 4. Real-time inference

Reuse `landmark_detector.run()`'s per-frame detection, feed each frame through
the **same** `to_features`, keep a sliding window of `L` frames, and classify
continuously. Smoothing and hand-side matching are already solved in the core
library.

## 5. Scaling & robustness (when the dataset grows)

- **Parallel extraction**: `extract_clip` is independent per clip — shard by
  class across processes; `--skip-existing` makes runs resumable.
- **Quality gates**: promote QA to a hard gate (e.g. auto-exclude pose < 50%) via
  `prune_clips.py` driven by `extraction_report.json`.
- **More signers / lighting**: collect varied data; the normalisation step keeps
  features comparable across sources.
- **Two-handed coverage**: current data is mostly right-handed; flip-augmentation
  and balanced collection will improve left-hand generalisation.

## Suggested next deliverables

1. `features.py` — shared preprocessing (normalise + resample + mask).
2. `train.py` — k-fold training over `landmarks_all`, writes a model + metrics.
3. `infer_live.py` — real-time classifier built on `landmark_detector`.
4. Extend `dataset_report.py` with a confusion-matrix panel once a model exists.
