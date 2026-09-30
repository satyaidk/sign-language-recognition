# Data formats

## Raw landmark arrays (`processed_dataset/`)

| File | Shape | Content |
|------|-------|---------|
| `landmarks_pose/<class>/<stem>.npy` | (T, 33, 4) float32 | x, y, z, visibility |
| `landmarks_face/<class>/<stem>.npy` | (T, 478, 3) float32 | x, y, z (468 mesh + 10 iris points) |
| `landmarks_hands/<class>/<stem>.npy` | (T, 2, 21, 3) float32 | slot 0 = signer's **left** hand, slot 1 = **right**; x, y, z |
| `landmarks_all/<class>/<stem>.npy` | (T, 1692) float32 | the three above, concatenated (what training reads) |

Coordinates are MediaPipe-normalised: x, y in [0, 1] as a fraction of image width
and height, z relative depth, visibility in [0, 1]. **An undetected modality in a
frame is all zeros.** Presence is recovered from the data (`layout.present`) and is
also stored in the per-clip metadata masks.

### Column layout of the (T, 1692) vector (`FEATURE_LAYOUT.json`)

| Block | Columns | Width |
|-------|---------|-------|
| pose | 0 – 131 | 132 = 33 × 4 |
| face | 132 – 1565 | 1434 = 478 × 3 |
| left_hand | 1566 – 1628 | 63 = 21 × 3 |
| right_hand | 1629 – 1691 | 63 = 21 × 3 |

Defined once in `signlang/landmarks/layout.py`.

## Dataset-level files

| File | Content |
|------|---------|
| `classes.json` | `{class_name: label}`. Labels are 0..N-1 without gaps; the order defines the model's outputs. Partial extractions only append. |
| `manifest.csv` | one row per clip: `class, label, stem, T, verdict, all_npy, skeleton` (paths relative to `processed_dataset/`). Training skips `FAIL` rows. |
| `metadata/<class>/<stem>.json` | source path, fps, frames read/skipped, shapes, smoothing, processing width, per-frame masks, QA (detection rates, coordinate ranges, issues, verdict) |
| `extraction_report.json` | per-clip verdicts + dataset-wide counts and mean detection rates |
| `verification_report.json` | the independent verifier's per-clip results (incl. `all_vs_parts_max_diff`) |
| `reports/*.png` | the six QA infographics |
| `skeleton_videos/`, `overlay_check/` | visual proof videos (git-ignored) |

## Model input features

`(L=64, D=346)` float32 per clip, produced by `training/features.to_features`:
165 normalised coordinates, 13 pose visibilities, 165 velocities, 3 presence masks.
D changes with the feature config (e.g. `--use-face`); it is always stored in the model meta.

## Training artifacts (`artifacts/`)

| File | Content |
|------|---------|
| `checkpoints/fold{k}.pt` | best weights of CV fold k + config + classes + `val_acc` |
| `checkpoints/model_final.pt` | deployment weights (EMA or raw) + full meta |
| `checkpoints/model_meta.json`, `exported/model_meta.json` | `classes`, `n_classes`, `in_dim`, `seq_len`, `raw_dim`, `config {feature, model, train}`, `cv_mean_acc`, `cv_oof_accuracy`, `fit_train_acc` (training-fit, *not* accuracy), `weights`, `created`; the exported copy adds `onnx`, `onnx_int8`, `onnx_vs_torch_max_diff`, `onnx_latency_ms` |
| `exported/sign_model.onnx` | input `features` [batch, 64, 346] → output `logits` [batch, classes] |
| `metrics/cv_report.json` | `oof_accuracy`, `oof_macro_f1`, `cv_mean_acc`, `cv_std_acc`, `fold_val_acc`, `per_class`, `confusion_matrix`, `classes`, `config`, `n_clips` |
| `metrics/*.png` | confusion matrix, per-fold accuracy, training curves |
| `logs/train.log`, `logs/finetune.log` | full training logs |

## Prediction outputs (`artifacts/predictions/`)

| File | Content |
|------|---------|
| `<stem>_pred.json` | whole-clip `prediction`, `conf`, `topk`, sliding-window `timeline`, `true_label`, `correct` |
| `<stem>_pred.mp4` | original video + landmarks + predicted-sign banner |
| `live_transcript.json` | `{"transcript": [{t, sign, conf}]}` for the latest live run |
| `sessions/<YYYYmmdd-HHMMSS>.json` | source, fps, frames, backend, classes, gates and one record per segment: `t, sign, conf, margin, top3, T, hands_rate, n_windows, close, decision (emit/abstain/repeat), reason, classify_ms` |

## What is (and isn't) in git

Kept: code, docs, `classes.json`, `manifest.csv`, `FEATURE_LAYOUT.json`, metadata, JSON
reports, PNG charts, model/metric JSON. Ignored: videos, `.npy/.npz`, `.pt/.pth`, `.onnx`,
logs, session logs, caches. Those are large, regenerable, or personal video data.
