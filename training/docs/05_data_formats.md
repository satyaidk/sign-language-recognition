# 05 — Data Formats

## Input: the raw landmark clip

Read-only from the extraction project:

| File | Shape | Contents |
|------|-------|----------|
| `processed_dataset/landmarks_all/<class>/<stem>.npy` | `(T, 1692)` | per-frame `pose|face|left|right`, `float32` |
| `processed_dataset/manifest.csv` | — | one row per clip; `verdict == FAIL` rows are skipped |
| `processed_dataset/classes.json` | — | `{ "good": 0, "hello": 1, ... }` class → label |

The `(T, 1692)` layout (from `config.BLOCK`): pose `0–131` (33×4), face
`132–1565` (478×3), left hand `1566–1628` (21×3), right hand `1629–1691` (21×3).

## The model input: the `(L, D)` feature vector

`features.to_features()` turns each clip into a fixed `(L, D)` tensor. With the
default `FeatureConfig` (`seq_len=64`, pose+hands, no face, velocity + masks):

| Channel group | Width | Source |
|---------------|-------|--------|
| coordinates | 165 | 55 points × 3 (13 pose joints + 21 left + 21 right) |
| pose visibility | 13 | one per curated pose joint |
| velocity | 165 | first-difference of the coordinates |
| presence masks | 3 | pose / left_hand / right_hand |
| **Total `D`** | **346** | |

`L = 64` (fixed temporal length). So every clip → `(64, 346)`, regardless of its
original frame count. `D` changes if you toggle modalities (`--use-face`, etc.) —
`features.feature_dim(cfg)` computes it for any config; the actual value used is
recorded in `model_meta.json` (`in_dim`).

### Coordinate space (after the transform)
- Centered on the clip's **mid-shoulder** and scaled by **shoulder width** →
  translation + scale invariant. (Not unit-variance; the model's input
  `LayerNorm` handles the rest.)
- Absent points sit exactly at the **origin** (re-zeroed after normalisation).
- `x` is negated under horizontal-flip augmentation, with L/R points swapped.

## Checkpoints

| File | Contents |
|------|----------|
| `checkpoints/fold{k}.pt` | Per-fold best weights + `in_dim`, `classes`, `config`, `val_acc`. |
| `checkpoints/model_final.pt` | Deployment weights (`state_dict`) + full self-describing meta. |
| `checkpoints/model_meta.json` | The meta without weights (for inference). |

## `model_meta.json` (self-describing)

Written by `finetune.py` and copied beside the ONNX by `export_optimize.py`.
Inference rebuilds the entire feature pipeline from this file alone.

```jsonc
{
  "kind": "final", "framework": "pytorch",
  "classes": ["good","hello", ... ],          // 12 classes
  "n_classes": 12,
  "in_dim": 346, "seq_len": 64, "raw_dim": 1692,
  "config": { "feature": {...}, "model": {...}, "train": {...} },
  "created": "2026-06-25 16:45:04",
  "fit_train_acc": 1.0, "weights": "ema",      // EMA vs raw chosen at fit time
  "cv_mean_acc": 0.9105, "cv_oof_accuracy": 0.9104,
  "onnx": "sign_model.onnx", "onnx_int8": "sign_model.int8.onnx",
  "onnx_vs_torch_max_diff": 1.67e-06
}
```

## Exported models

| File | Notes |
|------|-------|
| `exported/sign_model.onnx` | fp32 ONNX, opset 17, dynamic batch, fixed `L`. |
| `exported/sign_model.int8.onnx` | Dynamic int8 quantized (smaller / faster on CPU). |
| `exported/model_meta.json` | Same meta + `onnx` / `onnx_int8` filenames + parity diff. |

## Metrics

| File | Contents |
|------|----------|
| `metrics/cv_report.json` | OOF accuracy + macro-F1, per-fold accs, per-class P/R/F1, confusion matrix, config. |
| `metrics/confusion_matrix.png` | Row-normalised OOF confusion matrix. |
| `metrics/cv_folds.png` | Per-fold accuracy bars. |
| `metrics/history.png` | Loss / val-acc curves per fold. |

`cv_report.json` shape:

```jsonc
{
  "oof_accuracy": 0.9104, "oof_macro_f1": 0.9091,
  "cv_mean_acc": 0.9105, "cv_std_acc": 0.0291,
  "fold_val_acc": [0.9062, 0.8929, 0.9615, 0.875, 0.9167],
  "per_class": { "good": {"precision":0.9,"recall":0.818,"f1":0.857,"support":11}, ... },
  "confusion_matrix": [[...], ...],
  "classes": [...], "in_dim": 346, "n_clips": 130, "config": {...}
}
```

## Predictions

| File | Contents |
|------|----------|
| `predictions/<stem>_pred.mp4` | Overlay video: landmarks + predicted-sign subtitle. |
| `predictions/<stem>_pred.json` | `{prediction, conf, topk, timeline, correct}`. |
| `predictions/live_transcript.json` | `{transcript: [{t, sign, conf}, ...]}` from a live session. |

## Transient

- `live_tmp/chunk_*.mp4` — recorded webcam chunks, **deleted immediately** after
  each chunk is processed (a safety net also clears leftovers on exit).
- `cache/` — feature cache scratch space.

All of `training/artifacts/` is gitignored and regenerable by re-running the
stages.
