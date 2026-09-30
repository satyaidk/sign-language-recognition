# 02 — Technical Requirements Document (TRD)

The technical capabilities SignX v1 needs to meet the PRD. Each has a priority
(M/S/C) and a status against the **v0 reference code** (✅ reusable as-is ·
🔁 port/adapt from v0 · 🆕 new work · 🔬 needs measurement). Traceability to the
PRD is in §15.

---

## 1. Environments

### 1.1 Reference machine (measured 2026-09-30)

| Item | Value |
|------|-------|
| OS | Windows 11 Home (10.0.26300); PowerShell + Git Bash |
| CPU / RAM | Intel i5-11320H (4C/8T) / 7.8 GB |
| GPU | NVIDIA RTX 3050 Laptop, **4 GB**, driver 591.74 (unused so far: torch is `+cpu`) |
| Disk | **C: 31 GB free**, **E: 100 GB free** |
| Python | 3.11.0 |

### 1.2 The v0 environment (installed today; reference only)
`mediapipe==0.10.9` (legacy `mp.solutions`), `protobuf==3.20.3`, `numpy==2.4.4`,
`torch==2.11.0+cpu`, `onnx==1.21.0`, `onnxruntime==1.24.4`, `opencv-python` **and**
`opencv-contrib-python==4.13.0.92` (duplicate `cv2` wheels, a conflict risk),
`matplotlib==3.10.8`, `reportlab==5.0.0`. Not installed: `pytest`, `onnxscript`.

### 1.3 v1 environment requirements

| ID | Requirement | Pri | Status |
|----|-------------|-----|--------|
| TR-ENV-1 | A **new virtual environment** for v1 (`.venv-signx`), separate from the v0 packages | M | 🆕 |
| TR-ENV-2 | Python **3.11** (3.12 allowed; MediaPipe wheels support 3.9–3.12 per [PyPI](https://pypi.org/project/mediapipe/)) | M | ✅ |
| TR-ENV-3 | Exact pins in `requirements.txt` (runtime) and `requirements-dev.txt` (pytest, onnxscript if used). Only **one** OpenCV wheel (`opencv-contrib-python` if MediaPipe requires it) | M | 🆕 |
| TR-ENV-4 | MediaPipe version pinned **after** the Tasks-backend spike (T-1.3). Candidate: latest 1.0.x | M | 🆕 |
| TR-ENV-5 | Optional CUDA build of torch for training on the RTX 3050 (4 GB VRAM is ample for ≤ 10 M-param landmark models). Inference stays on CPU/ONNX Runtime | S | 🆕 |
| TR-ENV-6 | `python -m signx doctor`: versions vs pins, GPU/CUDA, camera, `SIGNX_DATA_ROOT` present and writable, free disk, backup-manifest status | S | 🆕 |

---

## 2. Language layer (TR-LANG): the foundation for multilingual support

| ID | Requirement | Pri | Status |
|----|-------------|-----|--------|
| TR-LANG-1 | **Language registry** `registry/languages.toml`: `code` (ISO 639-3), `name`, `region`, `notes`. Codes are verified against the [SIL ISO 639-3 tables](https://iso639-3.sil.org/code_tables/639/data) when a language is registered | M | 🆕 |
| TR-LANG-2 | **Vocabulary per language** `registry/vocab/<code>.csv`: `sign_id` (`<code>:<GLOSS>`), `gloss`, `concept_en` (optional meaning, for analysis only), `vs_group` (visually-similar group, optional), `sources` (datasets providing it), `status` (active/retired) | M | 🆕 |
| TR-LANG-3 | **Label invariants**: a `sign_id` belongs to exactly one language; class indices per head come from the *ordered active vocabulary* of that language in a **frozen, versioned** label map saved with each model; appending signs creates a new label-map version; indices are never re-numbered within a version | M | 🆕 |
| TR-LANG-4 | Cross-dataset **gloss mapping** per language (`registry/gloss_map/<dataset>.csv`: dataset gloss → `sign_id`, or `DROP`), reviewed by a language expert where possible | M | 🆕 |
| TR-LANG-5 | A special background class per language: `<code>:_BACKGROUND` | M | 🆕 |
| TR-LANG-6 | Nothing outside `registry/` and configs may hard-code a language code, gloss, or class count (enforced by a test that greps the package) | M | 🆕 |

Examples of codes: `ase` American SL, `ins` Indian SL, `bfi` British SL, `bzs` Brazilian SL (Libras).
**Verify each code in the SIL table when registering (R-2.9).**

---

## 3. Landmark detection (TR-DET)

| ID | Requirement | Pri | Status |
|----|-------------|-----|--------|
| TR-DET-1 | **`DetectorBackend` interface**: `detect(rgb, t_ms) -> FrameLandmarks` (pose/face/left/right, each optional), `close()`, `info() -> {name, version, models, running_mode}` | M | 🆕 |
| TR-DET-2 | **Tasks backend** (primary): MediaPipe Tasks `PoseLandmarker` + `HandLandmarker` (2 hands) + `FaceLandmarker`, or `HolisticLandmarker`, in `VIDEO` mode (files) / `VIDEO` mode with a live clock (webcam). The choice is made in the spike (T-1.3) on latency, detection rate, and API stability (Python Holistic was removed in 0.10.32 and re-added in 0.10.33 per the [changelog](https://data.safetycli.com/packages/pypi/mediapipe/changelog)) | M | 🆕 |
| TR-DET-3 | **Anatomical handedness**: slot 0 = signer's left. Tasks' HandLandmarker "assumes the input image is mirrored" ([docs](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)), so labels are swapped for raw (non-mirrored) frames. Covered by a test with a known-handed clip | M | 🔁 v0 `anatomical_label` |
| TR-DET-4 | **Landmark schema v1** (`schema_id = "mp-1692-v1"`): pose 33×(x,y,z,visibility) · face 478×(x,y,z) · left hand 21×(x,y,z) · right hand 21×(x,y,z) = **1692 floats/frame**, absent = zeros, plus a 4-bit presence mask per frame. It keeps v0's proven layout, and matches the Tasks outputs (Pose 33, Face 478 incl. iris, Hand 21) | M | 🔁 v0 layout |
| TR-DET-5 | Schema adapters for foreign landmark sources (e.g. legacy Holistic 543 points = face 468 + hands + pose): map into `mp-1692-v1`, zero-fill missing points (e.g. iris 468–477), and record `schema_source` | S | 🆕 |
| TR-DET-6 | One-Euro smoothing as a separate, backend-agnostic stage with explicit timestamps; no occlusion back-fill in datasets | M | 🔁 v0 `PointStabilizer`/`HandStabilizer` |
| TR-DET-7 | **Extractor provenance** on every sample: backend name + MediaPipe version + model file hashes + smoothing params + processing resolution. **A training set must use exactly one extractor configuration** | M | 🆕 |
| TR-DET-8 | **Bulk extraction**: resumable, idempotent (skip done by checksum), multi-process across videos (≤ physical cores − 1), progress + ETA, per-video error isolation | M | 🔁 partly v0 `--skip-existing` |
| TR-DET-9 | Modality selection (e.g. skip face when the model doesn't use it) with zero-fill that keeps the schema | S | 🆕 |
| TR-DET-10 | Performance: live per-frame detection ≤ 33 ms at the serving resolution on the reference CPU, or frame-skipping to hold ≥ 15 fps landmarks with a verified accuracy impact | S | 🔬 v0 legacy: ~40 ms/frame (synthetic) |
| TR-DET-11 | **The same processing resolution for extraction and serving** (v0 mixed 1080p training with 480 px serving) | M | 🆕 |

---

## 4. Data layer (TR-DATA)

| ID | Requirement | Pri | Status |
|----|-------------|-----|--------|
| TR-DATA-1 | **Data root outside the repo**: `SIGNX_DATA_ROOT` (default proposal `E:\signx-data`, which has 100 GB free). The repo holds code, registries, configs, manifests, and checksums only | M | 🆕 |
| TR-DATA-2 | Layout under the root: `raw/<source>/…` (videos as delivered) · `landmarks/<schema>/<extractor_id>/<source>/<sample_id>.npz` · `datasets/<dataset_version>/manifest.csv` · `cache/features/<feature_hash>/` · `models/<model_version>/` · `sessions/` | M | 🆕 |
| TR-DATA-3 | **Sample file** `.npz`: `landmarks` (T, 1692) float16, `mask` (T, 4) uint8, `fps`, `t_ms` (T,) int64; `meta` JSON (sample_id, extractor provenance) | M | 🆕 |
| TR-DATA-4 | **Canonical manifest** columns: `sample_id, language, sign_id, gloss, source, source_split, signer_id, session_id, handedness, fps, frames, width, height, extractor_id, qa_verdict, split, video_sha256, npz_sha256, licence_id, consent_id` | M | 🆕 |
| TR-DATA-5 | **Dataset versions are immutable**: `dataset_version = <lang>-<name>-v<N>`, with a content hash over the manifest + npz hashes. Training references a version, never "whatever is on disk" | M | 🆕 |
| TR-DATA-6 | **Adapters** (`signx/data/adapters/`): (a) `folder` (`<root>/<gloss>/*.mp4` + a signer CSV), (b) one adapter per public dataset used (parses its metadata, signer IDs, and official splits), (c) `recordings` (own tool output). Each adapter yields manifest rows + video paths | M | 🆕 |
| TR-DATA-7 | **Splits**: use the dataset's official signer-independent split when it has one; otherwise grouped by `signer_id` (e.g. 70/15/15 by signer) with a fixed seed. **No signer in more than one split** (tested) | M | 🆕 |
| TR-DATA-8 | QA: per-sample verdict (detection rates per modality, frame count bounds, hands-present rate), a dataset report, reversible exclusion list | M | 🔁 v0 `qa_clip`, `verify_landmarks`, `prune_clips` |
| TR-DATA-9 | **Backup**: `backups/BACKUP_MANIFEST.json` in git (path, bytes, sha256) + a second physical copy + a `verify` command + a documented **restore drill** (restore to a temp dir and verify) | M | 🆕 |
| TR-DATA-10 | **Recording tool** (`signx record`): vocabulary-driven prompts, countdown, N reps, preview, auto-naming `raw/recordings/<lang>/<signer>/<session>/<sign_id>_<rep>.mp4`, writes the manifest stub with `consent_id` | M | 🆕 |
| TR-DATA-11 | **Background protocol** per language: rest, talking, face-touch, adjusting, reaching, transitions; target ≈ 2× the median class count | M | 🆕 |
| TR-DATA-12 | **Disk budget**: bulk video sources are downloaded and processed **in chunks** (download → extract → verify → move the video to archive storage or delete if re-downloadable, per licence), so C: never fills | M | 🆕 |
| TR-DATA-13 | Licence + consent registry `registry/sources.toml`: source id, licence name/URL, commercial use yes/no, citation, download date | M | 🆕 |

### 4.1 Data sources. **Chosen for v1.0 (D-017/D-018): ASL Citizen (primary) + own recordings (background class, live/RW tests); WLASL optional later.** Other rows are for future languages. Verify each licence before use.

| Language (code) | Dataset | Size (as published) | Signer-independent split | Licence (as found) |
|-----------------|---------|---------------------|--------------------------|--------------------|
| ASL (`ase`) | [ASL Citizen](https://arxiv.org/abs/2304.05934) | 83,399 videos · 2,731 signs · 52 signers | ✅ official (35/6/11 signers) | [MSR licence](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/): **non-commercial research only** |
| ASL (`ase`) | [WLASL](https://github.com/dxli94/WLASL) | 2,000 glosses · 21,083 videos · 119 signers | per dataset | **C-UDA**, academic/computational use |
| ASL (`ase`) | [MS-ASL](https://www.microsoft.com/en-us/research/publication/ms-asl-a-large-scale-data-set-and-benchmark-for-understanding-american-sign-language/) | 1,000 signs · 25k+ videos · 222 signers | ✅ | check licence |
| ASL (`ase`) | [Kaggle GISLR](https://www.kaggle.com/c/asl-signs/data) | 250 signs, **landmarks only** (legacy Holistic, 543 points) | per competition | competition terms; needs TR-DET-5 |
| ISL (`ins`) | [INCLUDE](https://dl.acm.org/doi/10.1145/3394171.3413528) | 263 signs · 4,287 videos | check | check licence |
| Turkish SL | [AUTSL](https://arxiv.org/abs/2008.00932) | 226 signs · 38,336 videos · 43 signers | ✅ | check licence |
| Argentine SL | [LSA64](https://arxiv.org/abs/2310.17429) | 64 signs · 3,200 videos · 10 signers | by signer | check licence |
| Russian SL | [Logos](https://arxiv.org/abs/2505.10481) | 2,863 glosses · 199,668 videos · 381 signers | ✅ | check licence |
| any | **Own recordings** (TR-DATA-10) | as recruited | by signer | own consent; commercial use possible if the consent allows |

---

## 5. Features (TR-FEAT)

| ID | Requirement | Pri | Status |
|----|-------------|-----|--------|
| TR-FEAT-1 | **One shared transform** `raw (T,1692) + mask → (L, D)` for training, file inference, and live inference | M | 🔁 v0 `features.py` |
| TR-FEAT-2 | Deterministic, **no fitted state**; per-sample normalisation (median mid-shoulder origin, median shoulder width) | M | 🔁 v0 |
| TR-FEAT-3 | Configurable **point subsets**: upper-body pose, both hands, and named face subsets (`lips`, `eyes`, `brows`, `nose`). The Kaggle ISLR 1st place used lips + eyes + nose + hands ([solution](https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution)); non-manual features matter more as vocabularies grow | M | 🔁 v0 has pose/hands/full-face only |
| TR-FEAT-4 | Channels: normalised coordinates, velocity, optional acceleration, optional **wrist-relative hand shape**, visibility, presence masks | M | 🔁 partly |
| TR-FEAT-5 | Temporal: resample to L (default 64; configurable); velocity in **real-time units** (per second, using `t_ms`) as an option, so it doesn't depend on clip length | S | 🆕 |
| TR-FEAT-6 | Augmentation in normalised space (flip with L/R swap, rotation, scale, shift, jitter, time warp/crop, frame drop); absent points stay exactly zero | M | 🔁 v0 `data.augment` |
| TR-FEAT-7 | **Feature cache** on disk keyed by `(dataset_version, feature_config_hash)`, so large datasets aren't held in RAM (7.8 GB machine) | M | 🆕 (v0 cached in memory) |
| TR-FEAT-8 | The feature config is stored in model meta and inference rebuilds it from meta only | M | 🔁 v0 |

---

## 6. Models (TR-MODEL)

| ID | Requirement | Pri | Status |
|----|-------------|-----|--------|
| TR-MODEL-1 | **Encoder registry**: `bigru` (v0), `transformer` (v0), `cnn_transformer` (depthwise temporal conv blocks + attention, the ISLR-winner pattern) | M | 🔁 + 🆕 |
| TR-MODEL-2 | **Per-language heads**: `ModuleDict{lang: Linear(d, n_classes_lang)}` behind a shared encoder, from day one (one head in v1.0) | M | 🆕 |
| TR-MODEL-3 | Budget: ≤ 10 M params; ≤ 10 ms per classification with ONNX Runtime on the reference CPU | M | 🔬 v0 BiGRU 0.54 M / ~1.2 ms |
| TR-MODEL-4 | Background class inside each head (TR-LANG-5) | M | 🆕 |
| TR-MODEL-5 | Temperature calibration per head, stored in meta | S | 🆕 |
| TR-MODEL-6 | Regularisation toolbox: dropout, label smoothing, EMA, optional late dropout / drop-path | S | 🔁 partly |

---

## 7. Training (TR-TRAIN)

| ID | Requirement | Pri | Status |
|----|-------------|-----|--------|
| TR-TRAIN-1 | **Config files** (TOML, parsed with stdlib `tomllib` into dataclasses; no new dependency): dataset_version, languages, feature config, encoder, heads, optimiser, schedule, augmentation, seed | M | 🆕 (v0: dataclasses + CLI flags) |
| TR-TRAIN-2 | Training on the train split, model selection on the **val split (held-out signers)**, a final report on the **test split (held-out signers)**; test used once per candidate | M | 🆕 |
| TR-TRAIN-3 | Optional grouped k-fold by signer for small datasets | S | 🔁 v0 stratified k-fold |
| TR-TRAIN-4 | Class-balanced sampling or loss weighting; AdamW + warmup + cosine; grad clipping; EMA | M | 🔁 v0 |
| TR-TRAIN-5 | Device-agnostic (CPU / CUDA), optional AMP on CUDA; deterministic seeds | M | 🔁 partly |
| TR-TRAIN-6 | **Mixed-language batches** (future): each sample routed to its language's head; per-language loss weights | C (v2) | 🆕 |
| TR-TRAIN-7 | Checkpoint/resume; logging to file; **experiment registry** line per run (date, git SHA + dirty flag, config hash, dataset_version hash, metrics, duration, device) | M | 🆕 |

## 8. Evaluation (TR-EVAL)

| ID | Requirement | Pri | Status |
|----|-------------|-----|--------|
| TR-EVAL-1 | Metrics: top-1, top-5, macro-F1, per-class P/R/F1, confusion matrix + ranked confused pairs | M | 🔁 v0 (no top-5) |
| TR-EVAL-2 | Per-signer accuracy (mean, std, worst signer) | M | 🆕 |
| TR-EVAL-3 | Calibration (ECE, reliability plot); abstention/precision curve at the live thresholds | S | 🆕 |
| TR-EVAL-4 | **Live-path evaluator**: session videos → the exact live pipeline → transcript aligned to a script (correct/sub/ins/del, sign error rate, abstentions, latency) | M | 🆕 |
| TR-EVAL-5 | Per-language reports once > 1 language exists (same metrics per head) | S | 🆕 |
| TR-EVAL-6 | Visually-similar group view: accuracy when confusions within a `vs_group` are counted correct (diagnostic only) | C | 🆕 |

## 9. Export (TR-EXPORT)

| ID | Requirement | Pri | Status |
|----|-------------|-----|--------|
| TR-EXPORT-1 | ONNX per language: encoder + that language's head; input `features` [B, L, D] float32 → `logits` [B, C_lang] | M | 🔁 v0 (single head) |
| TR-EXPORT-2 | Parity < 1e-3 vs PyTorch on random + real inputs, at batch 1 and > 1 | M | 🔁 v0 (1.67e-6) |
| TR-EXPORT-3 | Exporter: the `dynamo=True` path (with `onnxscript`, the default since PyTorch 2.9, [docs](https://docs.pytorch.org/docs/stable/onnx)) or a pinned torch with the legacy exporter (deprecated; warns under 2.11). GRU under dynamo has open issues ([#164834](https://github.com/pytorch/pytorch/issues/164834)), which favours non-RNN encoders for export | M | 🆕 |
| TR-EXPORT-4 | Quantized variants only when measurably better (ORT dynamic quantization skips GRU, [#9796](https://github.com/microsoft/onnxruntime/issues/9796); v0 int8 was only 6% smaller) | C | — |
| TR-EXPORT-5 | **Model package** = `model_version/` with the ONNX file(s) + `model_meta.json` (languages, label maps with version, feature config, landmark schema + extractor id, calibration, metrics with evaluation type, dataset_version, git SHA) | M | 🔁 v0 meta |

## 10. Live serving (TR-SERVE)

| ID | Requirement | Pri | Status |
|----|-------------|-----|--------|
| TR-SERVE-1 | Threads: capture/UI (never blocks) · landmark worker (persistent backend, per-frame rows into a pre-roll ring buffer) · classifier/emitter. Bounded queues, drop-oldest, drop counter | M | 🆕 |
| TR-SERVE-2 | Segmentation: v0 `MotionSegmenter` (frame-diff, adaptive floor, hysteresis) + **hand-velocity** source from landmarks; chosen by evidence | M | 🔁 + 🆕 |
| TR-SERVE-3 | At segment close only features + model run (target ≤ 50 ms for ~10 robust windows) | M | 🆕 |
| TR-SERVE-4 | Gates: background class, calibrated confidence, top-1/top-2 margin, hands-present rate; debounce repeats | M | 🔁 partly |
| TR-SERVE-5 | `--lang` selects the head/ONNX; the UI shows the language and model version | M | 🆕 |
| TR-SERVE-6 | Timestamped session logs: per segment t_start/t_end, T, top-3, conf, margin, decision, latency breakdown, drops | M | 🆕 |
| TR-SERVE-7 | `--source <video>` exercises the identical live code path | M | 🔁 v0 |

**Latency budget:** stillness confirmation 500 ms (UX parameter) + queued landmark frames ≤ 100 ms + features/model ≤ 50 ms + render ≤ 16 ms → **≤ 700 ms p95 end-of-motion → caption**.

## 11. Performance (TR-PERF)

| ID | Target | Status |
|----|--------|--------|
| TR-PERF-1 | Live capture/display ≥ 20 fps | 🔬 |
| TR-PERF-2 | Live RSS ≤ 1.5 GB | 🔬 |
| TR-PERF-3 | Bulk extraction throughput measured and recorded (videos/hour) to plan dataset processing time | 🔬 |
| TR-PERF-4 | Training fits in 7.8 GB RAM via the on-disk feature cache | 🔬 |

## 12. Quality (TR-QA)

| ID | Requirement | Pri |
|----|-------------|-----|
| TR-QA-1 | `pytest` suite with synthetic fixtures, running without private data in < 60 s | M |
| TR-QA-2 | `@pytest.mark.data` / `model` / `slow` markers; skip cleanly when resources are absent | M |
| TR-QA-3 | Contract tests: schema offsets, label-map invariants, split leakage, language hard-coding grep (TR-LANG-6), ONNX parity | M |
| TR-QA-4 | Multilingual dry-run test (ST-8): two tiny synthetic "languages" train on two heads and export | M |
| TR-QA-5 | Optional ruff lint/format (owner decision) | C |

## 13. Security, privacy, licensing (TR-PRIV)

| ID | Requirement | Pri |
|----|-------------|-----|
| TR-PRIV-1 | No network calls at runtime; downloads only via explicit `signx data fetch` commands | M |
| TR-PRIV-2 | Videos of people never in git and never uploaded to third parties without consent | M |
| TR-PRIV-3 | Consent records for own recordings, stored outside git; deletion procedure (remove raw + npz + manifest rows → new dataset version → retrain) | M |
| TR-PRIV-4 | Licence registry (TR-DATA-13); a model's meta lists every source and whether it permits commercial use | M |
| TR-PRIV-5 | Model card + datasheet per released model | M |

## 14. Packaging (TR-PKG)

| ID | Requirement | Pri |
|----|-------------|-----|
| TR-PKG-1 | Installable package `signx/` (`pip install -e .` with a minimal `pyproject.toml`) | S |
| TR-PKG-2 | CLI `python -m signx {doctor, data, extract, qa, build-dataset, train, eval, export, live, video, record, backup}` | S |
| TR-PKG-3 | Windows executable of the live app | C |

---

## 15. Traceability: PRD → TRD

| PRD | TRD |
|-----|-----|
| G1 language-agnostic / G6 multilingual-ready | TR-LANG-*, TR-MODEL-2, TR-TRAIN-6, TR-EVAL-5, TR-QA-3, TR-QA-4 |
| G2 strong L1 recogniser | TR-DATA-6/7, TR-FEAT-3/4, TR-MODEL-1/3, TR-TRAIN-2, TR-EVAL-1/2 |
| G3 reproducible, safe data | TR-DATA-1..5, TR-DATA-9, TR-DET-7, TR-TRAIN-7 |
| G4 live | TR-DET-9/10/11, TR-SERVE-*, TR-PERF-1/2 |
| G5 measured quality | TR-EVAL-*, TR-QA-*, TR-SERVE-6 |
| FR-2 label namespace | TR-LANG-2/3/6 |
| FR-6 data safety | TR-DATA-1, TR-DATA-9, TR-DATA-12 |
| C2 licences | TR-DATA-13, TR-PRIV-4 |

## 16. References (checked 2026-09-30)

- MediaPipe: [PyPI](https://pypi.org/project/mediapipe/) · [mp.solutions removed (#6204)](https://github.com/google-ai-edge/mediapipe/issues/6204) · [#6241](https://github.com/google-ai-edge/mediapipe/issues/6241) · [changelog](https://data.safetycli.com/packages/pypi/mediapipe/changelog) · [HolisticLandmarker (Python)](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HolisticLandmarker) · [HandLandmarker guide, handedness note](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
- Export: [torch.onnx](https://docs.pytorch.org/docs/stable/onnx) · [GRU + dynamo issue](https://github.com/pytorch/pytorch/issues/164834) · [ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) · [GRU not quantized](https://github.com/microsoft/onnxruntime/issues/9796)
- Modelling: [Kaggle ISLR 1st place](https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution) · [landmark subsets: more accurate and 5× faster (LIBRAS)](https://arxiv.org/abs/2510.24887) · [Logos: multi-dataset co-training with language-specific heads, visually-similar sign groups](https://arxiv.org/abs/2505.10481) · [Collaborative multilingual CSLR](https://ieeexplore.ieee.org/iel7/6046/4456689/09954921.pdf) · [Online CSLR with a background class](https://arxiv.org/abs/2401.05336)
- Data: [ASL Citizen](https://arxiv.org/abs/2304.05934) ([licence](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/)) · [WLASL](https://github.com/dxli94/WLASL) · [MS-ASL](https://www.microsoft.com/en-us/research/publication/ms-asl-a-large-scale-data-set-and-benchmark-for-understanding-american-sign-language/) · [GISLR](https://www.kaggle.com/c/asl-signs/data) · [INCLUDE](https://dl.acm.org/doi/10.1145/3394171.3413528) · [AUTSL](https://arxiv.org/abs/2008.00932) · [LSA64](https://arxiv.org/abs/2310.17429)
- Tooling worth evaluating: [pose-format (.pose files, MIT)](https://arxiv.org/abs/2310.09066) · [sign-language-processing/datasets loaders](https://github.com/sign-language-processing/datasets)
- Ethics: [Deaf-led critique](https://aclanthology.org/2024.signlang-1.6/) · [bias in SL models](https://arxiv.org/abs/2410.05206) · [ISO 639-3 tables](https://iso639-3.sil.org/code_tables/639/data)
