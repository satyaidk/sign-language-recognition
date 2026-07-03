# Sign-Language Recognition Training — Documentation

Technical documentation for the **training + inference** layer that sits on top
of the verified landmark dataset. This is the system sketched as "future work"
in the main project docs (`../dataprep/07_future_design.md`), now built.

It consumes `data/processed_dataset/landmarks_all/*.npy` + `manifest.csv` **read-only**
and produces a trained sign classifier, an ONNX export, and file / live inference.

Read in order, or jump to what you need.

| # | Document | What it covers |
|---|----------|----------------|
| 01 | [Overview](01_overview.md) | What the training layer does, the 5 stages, dataset, tech stack, directory tree |
| 02 | [Architecture](02_architecture.md) | Layered design, module responsibilities, design principles |
| 03 | [Pipeline](03_pipeline.md) | Feature transform + the 5 stages, step by step |
| 04 | [File Reference](04_file_reference.md) | Every `.py` file explained symbol by symbol |
| 05 | [Data Formats](05_data_formats.md) | The `(L, D)` feature vector, checkpoints, `model_meta.json`, artifacts |
| 06 | [Developer Guide](06_developer_guide.md) | How to run end-to-end, extend, and debug |
| 07 | [Results & Future Work](07_results_and_future.md) | Measured accuracy and where to go next |

> This documentation set is **scoped to `training/`** and lives entirely inside
> `training/docs/`. The original pipeline docs (extraction / verification /
> reporting) remain untouched under the project-root `docs/` folder.

Diagrams live in [`diagrams/`](diagrams). The combined PDF is
`TRAINING_DOCUMENTATION.pdf` (regenerate with `python docs/training/build_pdf.py`).

A focused write-up of the **live-translation fix** (motion-gated recording,
margin gating, de-duplication) is in [`LIVE_FIX.md`](LIVE_FIX.md) →
`LIVE_SEGMENTATION_FIX.pdf` (regenerate with
`python docs/training/build_live_fix_pdf.py`).

## Regenerating the docs assets
```
python docs/training/make_diagrams.py   # rebuild architecture / feature / stages diagrams
python src/training/train.py                # (re)generate the metrics plots embedded in the PDF appendix
python docs/training/build_pdf.py       # rebuild TRAINING_DOCUMENTATION.pdf from these markdown files
```

## Quick start

```bash
# from the project root, with the processed dataset already built
python src/training/train.py            # 1. k-fold cross-validation (honest accuracy)
python src/training/finetune.py         # 2. fit the deployment model on 100% of data
python src/training/export_optimize.py  # 3. export to ONNX (+ int8)
python src/inference/infer_video.py --video good/good_03   # 4. test on a video file
python src/inference/infer_live.py                          # 5. live webcam translation
```
