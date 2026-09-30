# Changelog

## 0.2.0 — 2026-09-30 · Refactor and hardening

The whole codebase was reorganised into the `signlang` package, bugs found in review
were fixed, and a test suite was added. The pipeline's behaviour and the model
design are unchanged, except where a change is listed below.

### Structure
- Six root-level scripts and a `training/` folder of scripts became one package with four
  layers (`landmarks`, `dataset`, `training`, `inference`) and one CLI (`python -m signlang <command>`).
- Removed every `sys.path` hack. Paths are configurable (`--*-dir` flags or `SIGNLANG_*_DIR`
  environment variables), so the whole pipeline can run in a temporary folder.
- One implementation of the landmark layout, drawing and QA replaced up to four copies
  (skeleton render, verification overlay, observation overlay, prediction overlay).
- Smoothing and hand logic are now pure NumPy, independent of MediaPipe protobufs, and unit-tested.
- Added `requirements.txt` (pinned; one OpenCV wheel only), `requirements-dev.txt`, `pyproject.toml`.
- Deleted the duplicate `training/finetune (2).py`.

### Bug fixes
- **Extraction renumbered labels:** `--classes` rebuilt `classes.json` from the processed subset, so
  existing labels changed. Labels are now append-only.
- **Extraction dropped clips from the manifest:** `--classes`, `--limit` and `--skip-existing` rewrote
  `manifest.csv` / `extraction_report.json` with only the clips processed in that run. They now merge.
- **Lost hands:** when MediaPipe labelled both hands the same side, both were written to one slot and
  one hand disappeared. Duplicates are now resolved by pose-wrist proximity or image position.
- **Prune had a dangerous default:** running it without arguments pruned two hard-coded clips. Clips
  must now be named; `--dry-run` added. It also refreshes `verification_report.json`.
- **Report showed hard-coded "results":** the infographics always displayed "NaN/Inf: none",
  "all == parts: max diff 0.0" and a fixed list of excluded clips. Every value is now computed from
  the dataset files; unverified checks say so.
- **Verification crashed** (instead of reporting FAIL) when a clip's modality files had different frame counts.
- **Feature transform crashed** on an empty (0-frame) clip.
- **Live mode froze** after every sign (temp MP4 + three new MediaPipe models + re-extraction on the
  capture thread, ~2.5 s for a 2 s sign). It now streams landmarks with persistent models, so only the
  model runs when a sign ends (~2 ms).
- **Live mode reported the wrong abstain reason** ("tie" whenever confidence ≥ 0.5). It now reports
  `no-hands`, `low-conf` or `tie` correctly.
- **Live file simulation wasn't reproducible:** de-duplication used the wall clock. It now uses video time.
- **Whole-clip fallback bypassed the hands gate** (hard-coded `hands_rate = 1.0`).
- **File inference didn't save its JSON** with `--no-overlay`.
- **Transformer option wasted parameters:** a fixed 4096-row positional table (524k of 836k parameters).
  It's now sized to the sequence length (~320k parameters total).
- **Unbalanced CV folds:** every class started filling fold 0 (e.g. 36 vs 24 clips). Folds are now balanced.
- **Observation crashed** with `--no-face/--no-hands/--no-pose` (charts assumed every modality) and defaulted to a hard-coded `asl.mp4`.
- Unknown `pool`/`arch` values silently fell back to a default; they now raise.
- File handles opened without closing / without an encoding (Windows cp1252 issues).

### Added
- Streaming `LiveRecognizer` + `EmitGate`, testable without a camera; face detection skipped when
  the model doesn't use it; per-session diagnostic logs (`predictions/sessions/<time>.json`).
- Clear error when MediaPipe is missing or too new for the legacy solutions API.
- Export refuses to write a model that fails the ONNX/PyTorch parity check; int8 is kept only if it agrees.
- ~100 pytest tests (synthetic data; no private data or model needed) + an end-to-end CLI smoke test.
- New documentation (`docs/`) and diagrams; the original design docs are archived in `docs/archive/original/`.

## 0.1.0 — 2026-06 / 2026-07 · Original prototype

- MediaPipe landmark viewer with One-Euro smoothing, occlusion hold and hand↔arm matching.
- Dataset extraction, verification, pruning and QA infographics (134 clips, 12 signs, all PASS).
- Training pipeline: shared feature transform, augmentation, BiGRU, 5-fold CV (0.910 out-of-fold
  accuracy, within-signer), EMA final fit, ONNX export (parity 1.7e-6).
- File inference with overlay; live recognition redesigned from fixed windows to motion-gated
  segments with margin gating and repeat de-duplication.
