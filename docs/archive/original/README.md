# Original design documentation (pre-refactor)

These are the documents written while the project was first built (June–July 2026),
kept unchanged as a record of the design process:

- `dataset/`: the landmark extraction and verification system (7 chapters + diagrams + `TECHNICAL_DOCUMENTATION.pdf`)
- `training/`: the training and inference system (7 chapters + diagrams + `TRAINING_DOCUMENTATION.pdf`), and
  `LIVE_FIX.md` / `LIVE_SEGMENTATION_FIX.pdf`, the write-up of the live-segmentation redesign

**They describe the original file layout** (`extract_dataset.py`, `landmark_detector.py`,
`training/train.py`, `training/infer_live.py`, …). In September 2026 the code was reorganised
into the `signlang` package. The concepts are unchanged, but file names, function names
and commands differ.

| Original | Now |
|----------|-----|
| `landmark_detector.py` | `signlang/landmarks/{viewer,smoothing,hands,drawing,layout}.py` |
| `extract_dataset.py` | `signlang/landmarks/extractor.py` + `signlang/dataset/extract.py` |
| `verify_landmarks.py` | `signlang/dataset/{verify,qa}.py` |
| `prune_clips.py` · `dataset_report.py` · `observe_video.py` | `signlang/dataset/{prune,report,observe}.py` |
| `training/config.py` · `utils.py` | `signlang/config.py` · `signlang/utils.py` |
| `training/{features,data,model,train,finetune,viz}.py` | `signlang/training/…` (same names) |
| `training/export_optimize.py` | `signlang/training/export.py` |
| `training/{predictor,segmenter}.py` | `signlang/inference/…` (same names) |
| `training/infer_video.py` · `infer_live.py` | `signlang/inference/video.py` · `live.py` |
| `training/artifacts/` | `artifacts/` |
| `python extract_dataset.py …` | `python -m signlang extract …` (see [../../README.md](../../README.md)) |

For the current documentation, start at [docs/README.md](../../README.md).
