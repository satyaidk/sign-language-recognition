# Results

Two kinds of numbers are reported here, kept separate:
1. **The original training run** (June 2026), recorded in the committed reports. The
   recorded videos and trained weights were not retained (personal video data), so
   these are historical results. They can't be re-run from this repository.
2. **Engineering measurements of the current code** (September 2026), reproducible
   with the test suite and the commands shown.

## 1. Original run: 12 signs, 134 clips

**Dataset** (`processed_dataset/extraction_report.json`, `verification_report.json`):

| | |
|---|---|
| Classes | 12: good, hello, help, hi, how_are_you, me, nice_to_meet_you, no, sorry, thank_you, yes, you |
| Clips | 134 (136 recorded; 2 framing outliers pruned), 10–13 per class, 14–286 frames (median 60) |
| Signers | 1 |
| Verification | 134 PASS / 0 WARN / 0 FAIL |
| Mean detection rate | pose 99.4% · face 93.9% · right hand 86.1% · left hand 22.2% · any hand 88% |

The low left-hand rate reflects the recordings: most signs were performed with the
right (dominant) hand. That's why the left↔right flip augmentation matters.

**Model quality**: 5-fold stratified cross-validation, BiGRU, L=64, D=346
(`artifacts/metrics/cv_report.json`):

| Metric | Value |
|--------|-------|
| Out-of-fold accuracy | **0.910** |
| Out-of-fold macro-F1 | **0.909** |
| Per-fold accuracy | 0.906 · 0.893 · 0.962 · 0.875 · 0.917 (mean 0.911 ± 0.029) |

| Class | F1 | | Class | F1 |
|-------|----|-|-------|----|
| thank_you | 1.00 | | hello | 0.91 |
| no | 0.96 | | good | 0.86 |
| yes | 0.96 | | me | 0.86 |
| nice_to_meet_you | 0.95 | | sorry | 0.84 |
| how_are_you | 0.92 | | help | 0.82 |
| hi / you | 0.92 | | | |

Most confused pairs: *good → help* (2 of 11), *help → good*, *me → sorry*,
*sorry → me / you*, *hello → how_are_you*. These are genuine look-alikes
(similar hand shapes and paths), and they're the reason live mode uses a
top-1/top-2 margin gate.

![Confusion matrix](../artifacts/metrics/confusion_matrix.png)

**Deployment model:** ONNX vs PyTorch max logit difference **1.67e-6**. A held-in
check on `good_03` gave `good` with 0.959 confidence (a sanity check, not an
accuracy claim).

### How to read the 0.91

- It is **within-signer**: every clip came from one person. It shows the pipeline
  learns the signs, but not how well it generalises to new people. Testing on
  held-out signers is the missing measurement.
- The classes were **mostly ASL with a few ISL variants**. A clean dataset should
  keep one sign language per label set.
- The final model's `fit_train_acc = 1.00` is fit on the training clips and says
  nothing about generalisation. It is never reported as accuracy.

## 2. Measurements of the current code

Reference machine: Intel i5-11320H (4C/8T), 7.8 GB RAM, Windows 11, Python 3.11,
torch 2.11 (CPU), onnxruntime 1.24, mediapipe 0.10.9.

| Measurement | Result | How |
|-------------|--------|-----|
| Test suite | all passing, ~20 s | `python -m pytest -q` |
| End-to-end CLI smoke test (extract → verify → report → prune → observe → train → finetune → export → video → live) | all stages pass | synthetic videos + synthetic landmark dataset |
| Model latency (ONNX, 1 clip, CPU) | ~1.1–1.2 ms | `export` |
| Feature transform | ~1.4 ms per clip (T = 90) | timed |
| Streaming landmark extraction (720p frames → 480 px, persistent models) | ~25 ms per frame (~40 fps) | timed on synthetic frames |
| Live classification after a sign ends | **~2 ms** (was ~2.5 s of frozen preview for a 2 s sign) | session log `classify_ms` |
| Synthetic 3-class dataset (end-to-end test) | ≥ 0.9 out-of-fold accuracy; ONNX parity < 1e-3; ONNX and PyTorch agree | `tests/test_pipeline.py` |
| Int8 ONNX | ~6% smaller, ~10% faster (GRU layers are not quantised) | `export` |
