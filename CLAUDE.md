# CLAUDE.md

## What this repository is
An end-to-end **isolated sign-language recognition** system built on MediaPipe
landmarks. It has a dataset pipeline (extract → verify → report), training (5-fold CV →
EMA final fit → ONNX) and inference (file + real-time webcam), all in one Python package,
`signlang`, with one CLI: `python -m signlang <command>`. It is the author's portfolio
project; the root `README.md` tells its story.

State (2026-09-30): **v0.2.0**. The original scripts were refactored into the package,
bugs were fixed, and a test suite was added (see `CHANGELOG.md`). The original 12-sign
dataset and trained weights are **not available** (only their JSON reports and charts are
committed), so real-data stages can only be exercised with new data or synthetic data.

## Where things are
- `signlang/landmarks` (L0) · `signlang/dataset` (L1) · `signlang/training` (L2) · `signlang/inference` (L3)
- `signlang/config.py`: paths (`--*-dir` flags / `SIGNLANG_*_DIR` env vars) and every hyper-parameter
- `tests/`: pytest; synthetic data and fakes, no private data needed
- `docs/`: `README.md` (how to run), `ARCHITECTURE.md`, `PIPELINE.md`, `LIVE_RECOGNITION.md`,
  `DATA_FORMATS.md`, `RESULTS.md`, `DEVELOPMENT.md`; `docs/archive/original/` = pre-refactor docs (history only)
- `build_docs/`: the plan for the **next version** (a language-agnostic, multi-sign-language platform).
  Read it (start with `build_docs/README.md` and `08_MEMORY.md`) only when working on that next version.

## Commands
```bash
python -m signlang --help               # all commands
python -m pytest -q                     # 100 tests, ~20 s — run before calling any change done
python docs/make_diagrams.py            # regenerate docs/diagrams/*.png
```

## Rules
- Keep the layering: `landmarks` → `dataset` → `training` → `inference`; never import upward.
- One source of truth: shapes in `landmarks/layout.py`, config in `config.py`, the feature
  transform in `training/features.py` (shared by training AND inference — never duplicate it).
- The `(T, 1692)` landmark layout and `classes.json` label order are contracts; changing them
  invalidates every model.
- MediaPipe stays pinned to **0.10.9** (legacy `mp.solutions`; removed in ≥ 0.10.31). Only
  `landmarks/extractor.py` and `landmarks/viewer.py` create MediaPipe models.
- New logic gets a unit test; a bug fix gets a regression test. Keep pure logic separate from I/O.
- Honest numbers only: never present training-fit accuracy as quality; reports compute what they show.
- Never commit videos, `.npy`, weights or `.onnx` (see `.gitignore`). Commit or push only when asked.

## Environment
Windows 11, Python 3.11. **The repo path contains spaces**, so quote paths in shell commands.
Reference laptop: i5-11320H, 7.8 GB RAM, CPU-only torch. Style: module docstrings explaining
what and why, type hints, `pathlib`, dataclass configs, `# ── Section ──` dividers,
actionable `[ERROR] … run X first` messages.
