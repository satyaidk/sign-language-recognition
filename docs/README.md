# How to run the project

This guide takes you from a folder of sign videos to live webcam recognition.
Every stage is a sub-command of one CLI: `python -m signlang <command>`.

| Documentation | What it covers |
|---|---|
| **This guide** | Install, data layout, every command, outputs, troubleshooting |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Package layers, module map, design decisions |
| [PIPELINE.md](PIPELINE.md) | How each stage works: extraction, QA, features, training, export |
| [LIVE_RECOGNITION.md](LIVE_RECOGNITION.md) | Real-time recognition: segmentation, gating, streaming, and the bugs it fixed |
| [DATA_FORMATS.md](DATA_FORMATS.md) | Every file the pipeline reads or writes |
| [RESULTS.md](RESULTS.md) | Measured results and their limits |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Tests, conventions, extending the project |
| [archive/original/](archive/original/) | The original (pre-refactor) design docs and PDFs, kept for history |

---

## 1. Install

Requirements: **Python 3.10–3.12** (tested on 3.11, Windows 11), a webcam for live mode.
A GPU is not needed.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # + pytest (and reportlab for the archived PDFs)
python -m signlang --help              # lists every command
```

> **MediaPipe must be 0.10.9.** The project uses MediaPipe's legacy `mp.solutions`
> API (FaceMesh / Hands / Pose), which newer MediaPipe versions (0.10.31+) removed.
> If you see `mediapipe ... no longer ships the legacy mp.solutions API`, re-run
> `pip install -r requirements.txt`. Install only `opencv-contrib-python`; having
> `opencv-python` installed at the same time breaks `cv2`.

## 2. Prepare the data

Put one folder per sign, one short clip per recording:

```
dataset/
├── hello/   hello_01.mp4  hello_02.mp4 ...
├── thank_you/  ...
└── yes/     ...
```

- One clip = one sign, with a little stillness before and after.
- Upper body and both hands in frame; any resolution / frame rate (the pipeline reads the fps).
- Supported extensions: `.mp4 .mov .avi .mkv .m4v .wmv`.
- Around 10+ clips per sign is the minimum to train anything; more signers means better generalisation.

Default locations (repo root): `dataset/`, `processed_dataset/`, `artifacts/`. Override
them per command with `--dataset-dir`, `--processed-dir`, `--artifacts-dir`, or globally with
the environment variables `SIGNLANG_DATASET_DIR`, `SIGNLANG_PROCESSED_DIR`, `SIGNLANG_ARTIFACTS_DIR`.

## 3. Run the pipeline

### L1 — Build and check the landmark dataset

```bash
python -m signlang extract                    # dataset/ -> processed_dataset/ (all clips)
python -m signlang extract --classes hello yes --limit 2      # quick partial run (merges, never overwrites)
python -m signlang extract --skip-existing    # resume an interrupted run
python -m signlang verify                     # independent cross-check -> verification_report.json
python -m signlang verify --clip hello/hello_01 --overlay     # eyes-on: original video + saved landmarks
python -m signlang report                     # 6 QA infographics -> processed_dataset/reports/
python -m signlang prune no/no_13 --dry-run   # preview dropping a bad clip
python -m signlang prune no/no_13             # move it to excluded_clips/, delete derived files, refresh reports
```

Useful `extract` options: `--proc-width 720` (faster detection on large videos),
`--no-skeleton` (skip skeleton videos), `--no-smooth` (store raw, unfiltered landmarks).
Training needs all three modalities (face, hands, pose), so keep `--no-face/--no-hands/--no-pose` for inspection only.

### L2 — Train, fit and export the model

```bash
python -m signlang train                      # stage 1: 5-fold cross-validation -> honest accuracy
python -m signlang finetune                   # stage 2: final model on 100% of the data (EMA)
python -m signlang export                     # stage 3: ONNX + parity check + CPU latency
```

Common training options (both `train` and `finetune`): `--epochs 120 --folds 5 --lr 1.5e-3
--batch-size 16 --seq-len 64 --arch {bigru,transformer} --hidden 128 --use-face --no-aug --seed 1337`.
Every other hyper-parameter lives in [`signlang/config.py`](../signlang/config.py).
`finetune --init-from fold3` warm-starts from a cross-validation checkpoint.

### L3 — Recognise signs

```bash
python -m signlang video --path my_clip.mp4               # stage 4: prediction + timeline + overlay video
python -m signlang video --clip hello/hello_01            # a clip from dataset/, with its true label
python -m signlang video --scan-dataset --limit 2         # batch sanity check (in-sample!)
python -m signlang live                                   # stage 5: webcam (press Q to quit)
python -m signlang live --device-index 1 --margin 0.2 --still 0.6
python -m signlang live --source my_clip.mp4 --no-display # the live pipeline on a file, no camera needed
```

Live tuning (see [LIVE_RECOGNITION.md](LIVE_RECOGNITION.md)):

| Flag | Default | Effect |
|------|---------|--------|
| `--conf` | 0.55 | minimum top-1 probability to show a sign |
| `--margin` | 0.15 | minimum gap between the top-2 signs (raise to reject look-alikes) |
| `--still` | 0.5 s | stillness that ends a sign (raise if your signs have internal pauses) |
| `--min-sign` / `--max-sign` | 0.5 / 5.0 s | discard twitches / force-close very long segments |
| `--repeat-window` | 4.0 s | an identical sign inside this window counts as a repeat |
| `--motion-start` / `--motion-stop` | 2.6 / 1.7 | start / stop thresholds as multiples of the learned noise floor |
| `--proc-width` | 480 | detection resolution (smaller = faster) |

### Other tools

```bash
python -m signlang view                                   # live landmark viewer (webcam, mirrored)
python -m signlang view --source clip.mp4 --show-indices  # video or image
python -m signlang observe --video conversation.mp4       # continuous-signing analysis + detection timeline
```

## 4. Outputs at a glance

| Command | Writes |
|---------|--------|
| `extract` | `processed_dataset/landmarks_{pose,face,hands,all}/<class>/<stem>.npy`, `skeleton_videos/`, `metadata/`, `classes.json`, `manifest.csv`, `FEATURE_LAYOUT.json`, `extraction_report.json` |
| `verify` | `processed_dataset/verification_report.json` (+ `overlay_check/*.mp4`) |
| `report` | `processed_dataset/reports/01…06_*.png` |
| `train` | `artifacts/metrics/cv_report.json`, `confusion_matrix.png`, `cv_folds.png`, `history.png`, `artifacts/checkpoints/fold*.pt`, `artifacts/logs/train.log` |
| `finetune` | `artifacts/checkpoints/model_final.pt`, `model_meta.json` |
| `export` | `artifacts/exported/sign_model.onnx` (+ `.int8.onnx` if it agrees), `model_meta.json` |
| `video` | `artifacts/predictions/<stem>_pred.json` (+ `_pred.mp4`) |
| `live` | `artifacts/predictions/live_transcript.json`, `artifacts/predictions/sessions/<time>.json` |

Large files (videos, `.npy`, weights, ONNX) are git-ignored; JSON reports and PNG charts are kept.

## 5. Test

```bash
python -m pytest -q          # ~100 tests, ~20 s, no private data needed
```

## 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `mediapipe ... no longer ships the legacy mp.solutions API` | `pip install mediapipe==0.10.9 protobuf==3.20.3` |
| `ImportError` / odd `cv2` errors | `pip uninstall opencv-python` and keep only `opencv-contrib-python` |
| `No clips found under processed_dataset/landmarks_all` | Run `python -m signlang extract` first (or point `--processed-dir` at your data) |
| `model_meta.json not found` | Train a model: `train` -> `finetune` -> `export` |
| `expected (T, 1692) raw landmarks` | The dataset was extracted with a modality disabled; re-extract with all three |
| Live mode never shows a sign | Check the log line reason: `no-hands` (hands out of frame / too dark), `low-conf`, or `tie` (look-alike signs; sign more distinctly or lower `--margin`) |
| Live mode cuts a sign in two | Raise `--still` (e.g. 0.7) so internal pauses don't end the sign |
| Garbled console characters on Windows | Already handled (UTF-8 console); use Windows Terminal for best results |
