# Development guide

## Tests

```bash
python -m pytest -q                    # everything (~100 tests, ~20 s)
python -m pytest tests/test_pipeline.py -q      # end-to-end: train -> finetune -> export -> predict
python -m pytest -k segmenter -q       # by keyword
```

No test needs the private dataset or a trained model. Clips are synthetic `(T, 1692)`
arrays with a known structure, datasets are built in temporary folders, and camera,
MediaPipe and model are replaced by fakes where they aren't under test. Tests marked
`mediapipe` run real MediaPipe on a synthetic video and skip automatically if the
pinned version isn't installed.

| File | Covers |
|------|--------|
| `tests/test_landmarks.py` | layout round-trips, One-Euro behaviour (jitter, step response, teleport reset), stabiliser hold, hand-slot assignment incl. duplicate labels |
| `tests/test_extractor.py` | MediaPipe result conversion (fakes), anatomical slots, skipped modalities, real extraction on a synthetic video |
| `tests/test_dataset.py` | extraction bookkeeping (merge, append-only labels, skip-existing, unreadable video), QA verdicts, verification catching corruption, pruning + report refresh |
| `tests/test_training.py` | paths/config, stratified folds, metrics vs hand-computed values, feature dims/invariance/ghost-hand, augmentation, data loading, model shapes |
| `tests/test_inference.py` | segmenter (bursts, payloads, max/short, flush, frame motion), debouncer, emit-gate reasons, `LiveRecognizer` end-to-end with fakes |
| `tests/test_pipeline.py` | synthetic dataset → CV → final fit → ONNX export (parity) → predictor (ONNX ≡ torch), timeline |
| `tests/test_cli_and_reports.py` | every CLI command's `--help`, unknown commands, report facts read from files |

Rules: new pure logic gets a unit test; a bug fix gets a regression test that fails
before the fix. Several tests are named after the bug they pin down.

## Conventions

- **Layers:** `landmarks` → `dataset` → `training` → `inference`; a layer never imports from a layer above it.
- **One source of truth:** shapes in `landmarks/layout.py`, paths and hyper-parameters in `config.py`, the feature transform in `training/features.py`. Never duplicate them.
- **Pure logic separate from I/O**, so it can be tested without a camera, MediaPipe or data.
- **Every stage module** exposes `main(argv=None)` and `parse_args(argv=None)` and is registered in `signlang/__main__.py`.
- **Paths** come from `config.get_paths()` / `--*-dir` flags; no hard-coded drives or working-directory assumptions (the repo path may contain spaces).
- **Heavy imports are lazy** (MediaPipe only in L0 detection/drawing; stage modules imported on demand by the CLI).
- **Honest outputs:** reports compute every number they show; training-fit accuracy is never presented as model quality.

## Common tasks

**Add a sign:** record ≈10+ clips into `dataset/<new_sign>/`, then run
`extract --classes <new_sign>`. The new class gets the next label and existing labels
never change. Then `verify`, `train`, `finetune`, `export`.

**Try the face features:** `train --use-face` (adds the 478-point face block; D grows accordingly).

**Try the Transformer:** `train --arch transformer`.

**Tune live thresholds:** run `live --source recording.mp4 --no-display` on a recorded session and
read `artifacts/predictions/sessions/<time>.json`. Every segment records its confidence,
margin, hands rate, decision and reason.

**Regenerate the documentation diagrams:** `python docs/make_diagrams.py`.

## Upgrading MediaPipe (future)

MediaPipe ≥ 0.10.31 removed the legacy `mp.solutions` API. Only two places create
MediaPipe models: `landmarks/extractor.py` (`LandmarkExtractor`) and
`landmarks/viewer.py`. Everything else consumes `FrameLandmarks` / numpy arrays. A port
to the MediaPipe Tasks API (`PoseLandmarker`, `HandLandmarker`, `FaceLandmarker`) is
therefore local. Note that Tasks' HandLandmarker also assumes a mirrored image for
handedness. After porting, re-extract the dataset and retrain.
