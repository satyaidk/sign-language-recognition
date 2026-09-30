# 07 — Results & Future Work

## Measured results

From the latest 5-fold cross-validation (`metrics/cv_report.json`, 134 clips,
default BiGRU config, `D = 346`, `L = 64`):

- **Out-of-fold accuracy: 0.910**, **macro-F1: 0.909**.
- **Per-fold val accuracy:** `0.906, 0.893, 0.962, 0.875, 0.917`
  (mean `0.911 ± 0.029`).
- The deployment model (`finetune.py`, EMA weights) fits the training clips at
  `1.00` and inherits the CV numbers in its meta.
- ONNX matches PyTorch to `~1.7e-6` max logit diff; int8 keeps top-1 agreement.

### Per-class (out-of-fold)

| Class | P | R | F1 | n |
|-------|----|----|----|---|
| good | 0.90 | 0.82 | 0.86 | 11 |
| hello | 0.91 | 0.91 | 0.91 | 11 |
| help | 0.75 | 0.90 | 0.82 | 10 |
| hi | 0.85 | 1.00 | 0.92 | 11 |
| how_are_you | 0.92 | 0.92 | 0.92 | 13 |
| me | 0.82 | 0.90 | 0.86 | 10 |
| nice_to_meet_you | 1.00 | 0.91 | 0.95 | 11 |
| no | 1.00 | 0.92 | 0.96 | 13 |
| sorry | 0.89 | 0.80 | 0.84 | 10 |
| thank_you | 1.00 | 1.00 | 1.00 | 10 |
| yes | 1.00 | 0.92 | 0.96 | 12 |
| you | 0.92 | 0.92 | 0.92 | 12 |

The weakest classes (`help`, `sorry`, `good`) are the ones whose hand shape /
motion overlaps with others — see `confusion_matrix.png`. These are the first
candidates for more data or richer features.

> Numbers reflect the run that produced the committed artifacts. Re-running the
> stages regenerates them; expect small variation with seed/config changes.

## Where to go next

### Data
- **More clips per class.** ~11/class is the main limiter; even a handful more
  per class measurably stabilises the weak classes.
- **More signers / lighting / backgrounds.** Normalisation keeps features
  comparable across sources, so added variety mostly helps.
- **Balanced two-handed coverage.** Current data skews right-handed;
  flip-augmentation already mitigates this, but balanced collection helps more.

### Features
- **Try the face block** (`--use-face`) for signs where expression matters — it
  is off by default to keep the model light (face is 1434/1692 raw dims).
- **Richer motion**: acceleration (second difference), or per-hand
  joint-angle features, on top of the current velocity channel.
- **Subset the face** via `face_idx` (e.g. mouth/brow only) to add expression
  cheaply.

### Model
- **Transformer encoder** (`--arch transformer`) once the dataset is larger — it
  has more capacity than the BiGRU but needs more data to pay off.
- **Temperature calibration** so the live confidence threshold is more meaningful
  across classes.

### Inference / product
- **Continuous segmentation.** *Done (first pass):* the live loop is now
  **motion-gated** (`segmenter.py`) — it records one sign at a time between
  stillness boundaries instead of fixed chunks. A learned sequence segmenter (or
  CTC-style decoding) would still handle back-to-back signs with no pause.
- **Hand-velocity gating.** The current motion signal is whole-frame
  frame-difference; gating on hand-landmark velocity would ignore head/body
  motion and tighten the boundaries.
- **Streaming model.** A causal/streaming variant would cut the per-segment
  latency of the record→process→delete loop.
- **On-device export.** The int8 ONNX is already CPU-friendly; ORT-Mobile / WebGPU
  would extend this to phones / browsers.

### Evaluation
- **Confusion-driven collection loop**: use `confusion_matrix.png` to target
  exactly the pairs the model confuses.
- **Held-out signer test** once multiple signers exist — the honest measure of
  generalisation beyond k-fold over a single signer.
