# Sign Language Recognition from Body Landmarks

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-BiGRU-EE4C2C?logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-landmarks-0097A7)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU%20~1%20ms-005CED?logo=onnx&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-2E7D32)

An end-to-end system that **recognises isolated sign-language signs from a webcam**,
running in real time on a laptop CPU. It turns raw sign videos into a verified
skeleton-landmark dataset, trains a small sequence model on it, exports it to ONNX,
and captions signs live, staying quiet when it's unsure instead of guessing.

The original run used 12 everyday signs (*hello, thank you, sorry, help, yes, no, …*)
and reached **91% out-of-fold accuracy** on 134 clips from one signer. Every stage is
documented, tested and measured.

![End-to-end pipeline](docs/diagrams/pipeline.png)

---

## Contents
[The problem](#the-problem) · [How it works](#how-it-works) · [Architecture](#architecture) ·
[Features](#features) · [Results](#results) · [Quick start](#quick-start) ·
[Project structure](#project-structure) · [Engineering challenges](#engineering-challenges-and-what-i-learned) ·
[Limitations and roadmap](#limitations-and-roadmap) · [Documentation](#documentation)

## The problem

Sign languages are full visual languages. Recognising even *isolated* signs by
computer is hard: a sign is defined by hand shape, hand position, movement and
sometimes facial expression, all changing over time. Typical constraints make it
harder still:

- **Very little data.** A self-recorded dataset has on the order of 10 examples per sign, far too few for video models that learn from pixels.
- **Real-time on ordinary hardware.** It has to run from a laptop webcam, on a CPU.
- **Live input has no boundaries.** The camera never says when a sign starts or ends, and a person spends much of the time *not* signing.
- **Honesty.** A system that confidently prints the wrong word is worse than one that says "?".

**My approach:** don't learn from pixels. Use Google MediaPipe to turn every frame
into a skeleton of **553 body, hand and face landmarks**, then learn from how that
skeleton moves. This removes background, lighting and clothing from the problem,
shrinks the input from millions of pixels to a few hundred numbers per frame, and
makes a tiny model trainable on a tiny dataset.

## How it works

| Stage | Command | What happens |
|-------|---------|--------------|
| **1. Extract** | `python -m signlang extract` | MediaPipe FaceMesh + Hands + Pose on every frame → One-Euro smoothing → hands mapped to the signer's anatomical left/right → `(T, 1692)` landmark array per clip, a skeleton video rendered from the saved data, and per-clip QA |
| **2. Verify** | `verify`, `report` | Independent re-check of the saved files (NaN/Inf, frame counts, "combined vector == its parts"), an overlay of the saved points on the original video, 6 QA infographics |
| **3. Features** | *(inside training and inference)* | Normalise to the signer's shoulders (position/scale invariant) → select 13 upper-body joints + both hands → resample to 64 frames → add velocity + presence masks → **(64, 346)** |
| **4. Train** | `train` | Stratified **5-fold cross-validation** with heavy augmentation (incl. a left↔right mirror) → honest out-of-fold accuracy and confusion matrix |
| **5. Fit and export** | `finetune`, `export` | Final model on all data with **EMA** weights → **ONNX** with an automatic PyTorch-parity check → ~1 ms per prediction on CPU |
| **6. Recognise** | `video`, `live` | File mode: prediction + timeline + overlay video. Live mode: **streaming landmarks → motion-gated segmentation → confidence/margin/hands gates → repeat suppression → captions** |

The key design rule: **one feature transform (`features.to_features`) is shared by
training, file inference and live inference**, and it has no fitted state. The model
file describes exactly how its inputs were built, so there is no train/serve skew.

## Architecture

![Layered architecture](docs/diagrams/architecture.png)

One Python package, four layers, one CLI. Each layer only depends on the layers below it.

| Layer | Package | Responsibility |
|-------|---------|----------------|
| L0 | `signlang.landmarks` | MediaPipe detection, One-Euro smoothing (pure NumPy), hand-side logic, skeleton drawing, live viewer |
| L1 | `signlang.dataset` | Video dataset → verified landmark dataset; QA, verification, pruning, reports, continuous-video analysis |
| L2 | `signlang.training` | Feature transform, augmentation, BiGRU/Transformer model, k-fold CV, EMA fit, ONNX export |
| L3 | `signlang.inference` | Shared predictor, motion segmenter + de-duplication, file and live recognition |

**Model:** LayerNorm → Linear(346→128) → 2-layer bidirectional GRU → attention pooling
→ classifier. 543k parameters, regularised with dropout, label smoothing, augmentation and EMA.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/PIPELINE.md](docs/PIPELINE.md)

## Features

- **Robust landmark tracking:** One-Euro filtering (no jitter, no lag), visibility-gated bones, occlusion hold in the viewer, arms matched to the correct hand (no crossed "X" skeletons).
- **Correct handedness:** MediaPipe's mirrored-image assumption is corrected, and duplicate hand labels are resolved geometrically, so both hands are always kept.
- **Self-verifying dataset:** QA verdicts per clip, an independent verifier, skeleton and overlay videos rendered from the saved numbers, reversible pruning, infographics computed only from real data.
- **Data-efficient learning:** landmark features, shoulder-normalised coordinates, left↔right mirror augmentation, small regularised model, EMA.
- **Honest evaluation:** out-of-fold metrics, per-class F1 and confusion matrix. Training-fit accuracy is never reported as quality.
- **Fast deployment:** ONNX Runtime on CPU (~1 ms per prediction); export fails if ONNX doesn't match PyTorch.
- **Real-time live mode:** streaming landmarks (~25 ms per frame), motion-gated sign segmentation, "?" instead of guesses on look-alike signs, repeat suppression, per-session diagnostic logs.
- **Engineering quality:** one CLI, configurable paths, pinned dependencies, ~100 pytest tests that need no private data, an end-to-end CLI smoke test, and documented decisions.

## Results

**Original run:** 12 signs, 134 clips, 1 signer, 5-fold stratified cross-validation.

| Metric | Value |
|--------|-------|
| Out-of-fold accuracy | **0.910** |
| Out-of-fold macro-F1 | **0.909** |
| Per-fold accuracy | 0.906 · 0.893 · 0.962 · 0.875 · 0.917 (0.911 ± 0.029) |
| Best / hardest classes | *thank_you* 1.00 F1 · *help* 0.82 F1 |
| Dataset QA | 134 / 134 PASS · pose detected in 99.4% of frames |
| ONNX vs PyTorch | max logit difference 1.7e-6 |
| Inference | ~1.1 ms per prediction (ONNX, CPU); ~2 ms to classify a finished live sign |

<p align="center"><img src="artifacts/metrics/confusion_matrix.png" width="560" alt="Out-of-fold confusion matrix"></p>

The remaining errors are genuine look-alike pairs (*good ↔ help*, *me ↔ sorry*,
*hello ↔ how_are_you*). That's why live mode refuses to commit when the top two
predictions are too close.

**Reading this honestly:** every clip came from one signer, so 91% is a
*within-signer* number. It shows the pipeline learns the signs; accuracy on new
people is unmeasured. The recorded videos and trained weights aren't part of the
repository (personal video data); the committed reports are the record of the run.
More in [docs/RESULTS.md](docs/RESULTS.md).

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt                       # MediaPipe is pinned to 0.10.9 (see docs)
python -m signlang --help

# 1. put clips in dataset/<sign>/<clip>.mp4, then:
python -m signlang extract          # videos -> landmark dataset
python -m signlang verify           # independent check  (+ `report` for infographics)
python -m signlang train            # 5-fold cross-validation
python -m signlang finetune         # final model on all data
python -m signlang export           # ONNX
python -m signlang live             # real-time webcam recognition (Q to quit)

python -m pytest -q                 # run the tests (no data needed)
```

Full guide with every option, outputs and troubleshooting: **[docs/README.md](docs/README.md)**.

## Project structure

```
sign-language-translation/
├── signlang/                 the package (python -m signlang <command>)
│   ├── landmarks/            L0  extractor · smoothing · hands · layout · drawing · viewer
│   ├── dataset/              L1  extract · qa · verify · prune · report · observe
│   ├── training/             L2  features · data · model · train · finetune · export · viz
│   ├── inference/            L3  predictor · segmenter · video · live
│   ├── config.py             paths + every hyper-parameter
│   └── utils.py              folds, metrics, IO
├── tests/                    pytest suite (synthetic data, fakes for camera / model / MediaPipe)
├── docs/                     how-to-run guide, architecture, pipeline, live recognition, results
├── processed_dataset/        dataset metadata, QA reports and infographics (landmark arrays git-ignored)
├── artifacts/                training metrics, confusion matrix, model meta (weights git-ignored)
├── build_docs/               product/engineering plan for the next version (see roadmap)
├── requirements.txt · requirements-dev.txt · pyproject.toml · CHANGELOG.md
```

## Engineering challenges and what I learned

**1. Jittery, colliding skeletons.** Raw MediaPipe points shake from frame to frame
and arms cross into an "X" when hands pass each other. I added a **One-Euro filter**
(adaptive smoothing that only filters when a point is slow), visibility-gated bones,
and an **optimal two-hand assignment** that pairs each hand with the nearest arm.
*Lesson: clean input beats a clever model. Smoothing also made the velocity features far less noisy.*

**2. Left is right.** MediaPipe labels handedness as if the image were a mirrored
selfie, so on normal video every hand was on the wrong side. Later, while writing
tests, I found that when both hands got the *same* label, one hand was silently
overwritten. Both are now handled explicitly, and pinned by tests.
*Lesson: read the model's assumptions, and test the edge cases of third-party output.*

**3. Learning from 134 clips.** Pixels were out of the question, so I learned from
landmarks, normalised to the signer's shoulders. I made the data go further with
augmentation, especially a **left↔right mirror** that also swaps hand slots and
joints, and kept the model small and regularised. I evaluated with **k-fold
cross-validation** because a single split of ~11 clips per class is noise.
*Lesson: with small data, the evaluation protocol matters as much as the model.*

**4. The "ghost hand".** Normalising coordinates (subtract the shoulder centre,
divide by shoulder width) turned *missing* hands, stored as zeros, into a fake
hand at a fixed offset, a pattern the model could learn from. Missing points are
now re-zeroed after normalisation and after every augmentation.
*Lesson: "absent" needs an explicit representation, not a coincidental zero.*

**5. Train/serve skew.** Training and live inference used to be separate code
paths. I made **one feature transform with no fitted state** and a
**self-describing model file**, so the deployed model rebuilds its exact input pipeline.

**6. Live recognition took three iterations.**
- *Fixed 2.5 s windows* cut signs in half, kept repeating the previous word, and
  spammed repeats.
- *Motion-gated segmentation* recorded from "hands start moving" to "hands go still",
  with an adaptive noise floor, a confidence + top-2-margin gate that shows "?"
  rather than guessing, and repeat suppression.
- A review then showed that each finished sign was re-encoded to a temp video and
  re-processed by three freshly built MediaPipe models *on the camera thread*: a
  2-second sign froze the preview for ~2.5 s.
  I rebuilt it to **stream landmarks** with persistent models, so a finished sign is
  already in memory and classifies in **~2 ms**.

*Lesson: define the unit of work (the sign, not the clock), and measure where the time goes.*

**7. Honest numbers.** I found my own QA infographic printing hard-coded "all checks
passed" text, and the model metadata carrying a training-fit accuracy of 1.00 that
could easily be mistaken for real accuracy. Both are fixed: every reported number is
now computed and labelled.
I also learned the limits of my own result: one signer, and a label set that mixed
ASL with a few ISL signs. So 91% is within-signer, and the next dataset must keep one
sign language per label set.

**8. Living with dependency churn.** MediaPipe removed the API this project uses (≥ 0.10.31),
PyTorch deprecated the ONNX exporter I rely on, and ONNX Runtime doesn't quantise GRUs
(int8 gave only ~6%). I pinned versions, added a clear failure message, isolated MediaPipe
behind one extractor class, and measured instead of assuming.

**9. Refactoring into a tested package.** Turning a folder of scripts into a layered
package with ~100 tests found **real bugs**: labels renumbered by partial runs,
manifests silently truncated, a verifier crash, a feature-transform crash on empty
clips, and wasted Transformer parameters. All are listed in [CHANGELOG.md](CHANGELOG.md).
*Lesson: tests aren't just a safety net; writing them is a code review.*

## Limitations and roadmap

- **Generalisation is unmeasured.** The data has one signer. Next: several signers, and a signer-independent test split.
- **One language per label set.** The original classes mixed ASL with a few ISL signs.
- **No "not signing" class.** Thresholds reduce, but don't eliminate, false captions from non-sign motion.
- **Isolated signs only.** Continuous signing needs a sequence model or sign spotting.
- **Legacy MediaPipe API.** A port to the MediaPipe Tasks API is contained to one module.

[`build_docs/`](build_docs/) holds a full product and engineering plan (PRD, TRD,
architecture, rules, implementation plan, tasks, tests) for the next version: a
**language-agnostic** platform that trains one sign language at a time and is designed
to co-train several sign languages with a shared encoder and one output head per language.

## Documentation

| | |
|---|---|
| [docs/README.md](docs/README.md) | How to run: install, every command, outputs, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layers, module map, design decisions, limitations |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Extraction, QA, feature transform, augmentation, model, training, export |
| [docs/LIVE_RECOGNITION.md](docs/LIVE_RECOGNITION.md) | Live mode: three iterations, gates, tuning |
| [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md) | Every file the pipeline reads and writes |
| [docs/RESULTS.md](docs/RESULTS.md) | Measured results and how to read them |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Tests, conventions, extending the project |
| [CHANGELOG.md](CHANGELOG.md) | What changed and which bugs were fixed |

---

Built by **Satyanarayana Nikadi**. Tools: Python · MediaPipe · OpenCV · NumPy · PyTorch · ONNX Runtime · Matplotlib · pytest.
