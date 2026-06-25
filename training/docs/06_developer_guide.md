# 06 — Developer Guide

## Setup

```bash
pip install torch onnxruntime numpy matplotlib mediapipe opencv-python
```
Python 3.11 recommended. The extraction project must already have produced
`processed_dataset/` (run `python extract_dataset.py` first if not). Inference
also imports `extract_dataset.py` and `landmark_detector.py` from the project
root, so keep `training/` next to them.

Paths are resolved relative to the project root, so the commands below work from
anywhere; the examples assume you run from the project root.

## End-to-end run

```bash
python training/train.py            # 1. 5-fold CV  -> metrics/cv_report.json
python training/finetune.py         # 2. fit on 100% -> checkpoints/model_final.pt
python training/export_optimize.py  # 3. ONNX + int8 -> exported/sign_model.onnx
python training/infer_video.py --video good/good_03   # 4. file test + overlay
python training/infer_live.py                          # 5. live webcam
```

Each stage prints the suggested next command when it finishes.

## Everyday commands

```bash
# Train variants
python training/train.py --arch transformer --epochs 150
python training/train.py --use-face            # include the 1434-dim face block
python training/train.py --no-aug              # ablate augmentation
python training/train.py --seq-len 48 --hidden 96

# Deployment fit
python training/finetune.py --epochs 90
python training/finetune.py --init-from fold3  # warm-start from a fold checkpoint

# Export
python training/export_optimize.py --no-quant  # skip the int8 variant

# File inference
python training/infer_video.py --video hello/hello --backend torch --no-overlay
python training/infer_video.py --path "C:/clips/test.mp4"
python training/infer_video.py --scan-dataset --limit 3   # batch sanity check

# Live (or simulate from a file without a camera)
python training/infer_live.py --seconds 2.0 --proc-width 480 --int8
python training/infer_live.py --source good.mp4 --no-display

# Module self-tests (quick smoke checks)
python training/features.py        # feature spec + a sample clip + flip involution
python training/data.py            # dataset load + per-class counts
python training/model.py           # param count + forward shape
python training/predictor.py       # load model + predict one sample
```

## Common tasks

### Change a hyperparameter
Edit the dataclass defaults in `config.py` (one source of truth), or pass a CLI
flag for the common ones. The active config is recorded in `model_meta.json`, so
exported models stay self-describing.

### Add / remove a modality
Toggle `use_pose / use_hands / use_face` (or `--use-face`). The feature dimension
`D`, flip permutations, and mask channels all adjust automatically — retrain and
re-export. Inference picks up the new `D` from `model_meta.json`.

### Add a new class
1. Add clips and re-run the extraction pipeline so `processed_dataset/` includes
   the new class (`classes.json`, `manifest.csv`, `landmarks_all/`).
2. Re-run `train.py` → `finetune.py` → `export_optimize.py`. The class list is
   read from `classes.json`, so no code change is needed.

### Switch inference backend
`--backend onnx` (default, fast CPU) or `--backend torch`; add `--int8` for the
quantized ONNX. If export hasn't run, `SignPredictor` falls back to the torch
checkpoint automatically.

## Debugging guide

| Symptom | Likely cause / action |
|---------|----------------------|
| `No clips found under landmarks_all` | Extraction hasn't run, or `manifest.csv` is missing. Run `extract_dataset.py`. |
| `model_final.pt not found` (export) | Run `finetune.py` first. |
| ONNX export `MISMATCH` | A non-exportable op crept in; the legacy exporter handles GRU/Transformer — check custom layers. |
| int8 "quantization skipped" | `onnxruntime.quantization` missing/incompatible; use the fp32 ONNX or `--no-quant`. |
| Live always says "..." | Confidence below `--conf`, or hands not seen (>15% needed). Improve lighting/framing or lower `--conf`. |
| Low CV accuracy on one class | Inspect `confusion_matrix.png` — confusable signs share hand shape; collect more clips or enable face. |
| Slow live loop | Lower `--proc-width` (e.g. 320) and/or use `--int8`; shorten `--seconds`. |
| `UnicodeEncodeError` on Windows | Handled — `utils._enable_utf8_console()` runs on import. |

### Inspect a model meta / features by hand
```python
import sys; sys.path.insert(0, "training")
import numpy as np, features as F
from config import CONFIG, LANDMARKS_ALL
spec = F.build_spec(CONFIG.feature)
raw = np.load(next(LANDMARKS_ALL.glob("*/*.npy")))
print(raw.shape, "->", F.to_features(raw, CONFIG.feature, spec).shape)  # (T,1692) -> (64,346)
```

## Performance notes

- **Geometry is cached**: `load_geometric()` runs the expensive `geometric()`
  once per clip; each epoch only re-augments + assembles. CV over the whole
  dataset takes ~1–2 min on CPU.
- **CPU is the target**: there's no GPU here, so the int8 ONNX + ONNX Runtime is
  the fast path for live use (latency is reported by `export_optimize.py`).
- **`--proc-width`** is the main live speed knob; normalised coordinates are
  resolution-independent, so downscaling input barely affects accuracy.

## Conventions for contributors

- Keep **all** feature logic in `features.py` and **all** model logic in
  `model.py`; both training and inference must import them, never re-implement.
- Never duplicate the video→landmark path — always go through
  `predictor.video_to_clip` (which reuses the verified extractor).
- Never hand-edit anything under `training/artifacts/` — regenerate it.
- Keep `model_meta.json` the single contract between training and inference.
