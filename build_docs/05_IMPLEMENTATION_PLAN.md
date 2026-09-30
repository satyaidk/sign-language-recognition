# 05 — Implementation Plan

How we build **SignX v1**, a language-agnostic sign-recognition platform proven on
one sign language (L1), from the v0 reference code. The concrete work items are in
`06_TASKS.md` (IDs `T-<phase>.<n>`). "Done" is defined in `01_PRD.md` §9.

Sizes: **S** ≤ ½ day · **M** 1–2 days · **L** 3–5 days · **XL** depends on people or data acquisition.

---

## 1. Guiding principles

1. **Foundations first, in this order:** data safety → landmark layer → language/data layer → training → serving. Each layer is tested before the next one depends on it.
2. **Port, don't reinvent.** v0 has proven pieces (smoothing, feature transform, augmentation, segmenter, predictor logic). Port them with their tests; rewrite only what v0 got wrong (03_ARCHITECTURE §B3).
3. **Language-agnostic from the first line.** Multilingual *support* is built into the data and model structures immediately (it's cheap). Multilingual *training* waits until after v1.0.
4. **Decide L1 and the data licence early.** *Done:* ASL, research/non-commercial, 100 signs (D-017…D-019). ASL Citizen is usable as the primary source.
5. **Recruiting runs in parallel.** Own recordings and Deaf/fluent testers take calendar time, so start as soon as the recording tool exists.
6. **Every phase ends with numbers** recorded in `07_TEST.md`.

## 2. Phase overview

| Phase | Name | Outcome | Depends on | Size | Milestone |
|-------|------|---------|-----------|------|-----------|
| **P0** | Reset & foundation | v0 archived; v1 package, venv, tests, data root, backup system | — | M | **M0** Safe foundation |
| **P1** | Landmark layer | Schema v1, `DetectorBackend`, MediaPipe Tasks backend, bulk extraction with provenance | P0 | L | **M1** Extraction works |
| **P2** | Language & data layer | Registries, manifest, adapters, splits, recording tool; **L1 chosen**; L1 dataset v1 built and backed up | P1 (+ owner decisions) | L + XL | **M2** L1 dataset v1 |
| **P3** | Training platform | Features, augmentation, disk cache, encoders + language heads, config-driven training, metrics, registry, calibration, multilingual dry run | P0 (can start on synthetic data); real runs need P2 | L | **M3** Platform trains |
| **P4** | L1 model | Baseline → ablations → encoder choice → background class → test-split report | P2, P3 | L | **M4** L1 model meets offline targets |
| **P5** | Export & live serving | Per-language ONNX package, predictor, threaded streaming live app, session scoring, benchmarks | P3 (P4 for real models) | L | **M5** Responsive live app |
| **P6** | Real-world validation & release | RW protocol, error loop, soak, model card, guide, v1.0 | P4, P5 | M–L | **M6** v1.0 |
| **P7** | Multilingual (v2) | Add L2+, co-train with shared encoder + per-language heads | v1.0 | — | v2.0 |

### Dependency graph

```
P0 ─► P1 ─► P2 ─────────────► P4 ─► P6 ─► v1.0 ─► P7
 │           ▲  (L1 decision,  ▲       ▲
 │           │   licences,     │       │
 │           │   recruiting)   │       │
 └────────► P3 ────────────────┘       │
             └──────────► P5 ──────────┘
CRITICAL PATH: P0 → P1 → P2 (L1 data acquisition) → P4 → P6
Parallel:      P3 on synthetic data during P1/P2; P5 skeleton on synthetic models
```

---

## 3. Phases in detail

### P0 — Reset & foundation (M0)

**Why:** v1 is a new product built on lessons from v0. It needs a clean home, a
reproducible environment, and **data safety before any data exists**.

1. **T-0.1 Archive v0:** `git mv` all v0 code, docs, and `processed_dataset/` metadata into `reference/v0/` (history preserved). Delete `training/finetune (2).py`. Confirm the v0 self-tests still run from the new location. Update `CLAUDE.md` commands.
2. **T-0.2 New venv + pins:** `.venv-signx`, `requirements.txt` (numpy, opencv-contrib-python, torch, onnx, onnxruntime; MediaPipe is added by T-1.3), `requirements-dev.txt` (pytest).
3. **T-0.3 Package skeleton:** `pyproject.toml`, `signx/__init__.py`, `signx/__main__.py` (argparse sub-commands), `signx/paths.py` (`SIGNX_DATA_ROOT`, layout helpers), `signx/config.py` (dataclasses + TOML loader), `tests/conftest.py` (markers, synthetic fixtures).
4. **T-0.4 Data root + backup system:** `signx backup write|verify` → `backups/BACKUP_MANIFEST.json` (path, bytes, sha256). Choose the second-copy location (Q-13). Write the restore drill procedure.
5. **T-0.5 `signx doctor`.**

**Exit (M0):** `pip install -e .` in the new venv; `python -m pytest -q` green; `python -m signx doctor` green; the backup write/verify round-trip is tested on a temp dir; v0 is archived and still runnable.

### P1 — Landmark layer (M1)

1. **T-1.1 Schema:** `landmarks/schema.py`: `mp-1692-v1` offsets, `to_row()`, masks, float16 packing; contract tests.
2. **T-1.2 Backend protocol + smoothing:** `landmarks/backend.py` (`FrameLandmarks`, `DetectorBackend`); port the One-Euro filter and stabilizers from v0 into `landmarks/smoothing.py`, with tests.
3. **T-1.3 Tasks backend spike → decision:** implement `TasksBackend` twice: (a) Pose + Hand + Face landmarkers, (b) HolisticLandmarker. Compare them on ≥ 10 real sign videos (own quick recordings or licensed samples): detection rates, per-frame latency at 480/720/1080 widths, handedness correctness (a mirrored and a non-mirrored clip), API stability. Pin the MediaPipe version and model files (download script + sha256). Record the decision as D-n.
4. **T-1.4 Extraction engine:** `data/extract.py`: video → frames → backend → smoothing → npz with `t_ms`, masks, provenance, `extractor_id`; resumable (skip by checksum); multi-process; per-video error isolation; per-sample QA metrics. Port `draw.py` (skeleton/overlay rendering) for visual checks.
5. **T-1.5 QA & verification:** port v0's structural and consistency checks; an overlay spot-check command; a QA report.
6. **T-1.6 Processing resolution:** choose one width for extraction **and** serving from T-1.3 measurements (accuracy proxy = detection rates; latency) → D-n.

**Exit (M1):** extraction of a folder of videos produces valid npz files with one `extractor_id`; QA passes; throughput is measured (TR-PERF-3).

### P2 — Language & data layer (M2)

1. **T-2.1 Language code modules** (`signx/lang/`): languages, vocab, label maps (versioned, append-only), sources; `registry/` files; contract tests, including the "no hard-coded language" grep test.
2. **T-2.2 Manifest + dataset builder:** manifest schema (03_ARCHITECTURE §A4.3); `build-dataset`: gloss mapping, QA filter, background class, signer-independent splits (official or grouped), `dataset.json` + content hash; a leakage test.
3. **T-2.3 Adapters:** `folder` and `recordings` adapters, with tests on a synthetic folder.
4. **T-2.4 Recording tool + protocol + consent template:** `signx record`; `RECORDING_PROTOCOL.md` (framing, distance, lighting, backgrounds, pace, pauses, handedness) and `CONSENT_TEMPLATE.md` ("adapt before use; not legal advice").
5. **T-2.5 Background protocol** (per language).
6. **T-2.6 Register ASL and pick the 100 signs** (decided: `ase`, research/non-commercial, 100 + background). Output: the `ase` entry in `registry/languages.toml`; `sources.toml` with the ASL Citizen licence (`commercial_ok = false`); `vocab/ase.csv` with 100 signs chosen for everyday usefulness, enough videos per sign across signers, and a few deliberate look-alike pairs to stress-test. ASL Citizen maps to ASL-LEX entries (handshape etc.), which helps build `vs_group`s.
7. **T-2.7 ASL Citizen adapter:** parse its metadata (gloss, signer/user ID, official train/val/test split) → staging rows. Check the download size and packaging first, then **chunked** extract → verify → archive the videos off C: (disk budget R-5.6). About 30 videos per sign on average, so roughly 3,000 videos for 100 signs.
8. **T-2.8 Own recordings:** recruit ≥ 5 signers (≥ 2 Deaf/fluent, ≥ 1 left-handed), including background clips (runs in parallel; XL).
9. **T-2.9 Build `ase-core100-v1`** (+ backup write/verify + second copy).

**Exit (M2):** an immutable L1 dataset version with signer-independent splits, QA-passed, backed up in two places; licences recorded.

### P3 — Training platform (M3)  (starts on synthetic data during P1/P2)

1. **T-3.1 Feature transform** port + extensions (face subsets, acceleration, wrist-relative hands, real-time velocity); port the v0 synthetic tests (CT-1…CT-7).
2. **T-3.2 Augmentation** port + tests.
3. **T-3.3 Feature cache** on disk keyed by (dataset hash, feature-config hash).
4. **T-3.4 Models:** `encoders.py` (port bigru/transformer, new cnn_transformer), `heads.py` (per-language ModuleDict), `classifier.py`; shape/param tests.
5. **T-3.5 Training run:** TOML config → train/val loop (CPU/CUDA, AMP, EMA, early stopping on val signers, checkpoint/resume) → test-split report on request.
6. **T-3.6 Metrics & reports:** top-1/top-5/macro-F1/per-class/per-signer/confused pairs/ECE; JSON + PNG.
7. **T-3.7 Experiment registry.**
8. **T-3.8 Temperature calibration.**
9. **T-3.9 Multilingual dry run (ST-8):** two synthetic languages → two heads → train → per-language report → per-language ONNX. Proves G6 with no real multilingual data.
10. **T-3.10 (S) CUDA torch** in the venv, if the owner approves the download (~2–3 GB).

**Exit (M3):** a synthetic end-to-end run (data → features → train → report → registry) passes in CI-style tests; ST-8 passes.

### P4 — L1 model (M4)

1. **T-4.1 Baseline** on `<L1>-v1` (bigru, v0 feature set) → the first honest signer-independent numbers.
2. **T-4.2 Feature ablations** (face subsets lips/eyes/brows, acceleration, wrist-relative hands, velocity units) × 3 seeds, on val.
3. **T-4.3 Encoder comparison:** bigru vs transformer vs cnn_transformer (≤ 10 M params, ≤ 10 ms ONNX).
4. **T-4.4 Background class** training + gate thresholds tuned on val signers.
5. **T-4.5 Error analysis with a language expert:** confused pairs → `vs_group` annotations; gloss-map fixes → possibly dataset v2.
6. **T-4.6 Candidate selection → test-split report (once)** → `model_version 1.0.0-rc1`.

**Exit (M4):** PRD §9 offline targets met on the test split; the report is recorded; the registry is complete.

### P5 — Export & live serving (M5)  (skeleton can start on a synthetic model)

1. **T-5.1 Export path decision + per-language ONNX** with parity tests (dynamo + onnxscript vs pinned legacy exporter).
2. **T-5.2 Model package + predictor** (loads meta, label maps, calibration; `predict_robust`).
3. **T-5.3 Segmenter + debouncer port** (+ v0 self-test as pytest) + hand-velocity source.
4. **T-5.4 Threaded streaming live app** `signx live --lang <L1>` (UI states ported from v0).
5. **T-5.5 Session logs + latency instrumentation.**
6. **T-5.6 File mode** `signx video` (overlay + timeline).
7. **T-5.7 Session scorer** `signx eval-session` (script alignment).
8. **T-5.8 Live benchmark** (fps, drops, latency p50/p95, RSS).

**Exit (M5):** ≥ 20 fps preview during classification; p95 ≤ 700 ms; streaming ≡ file-mode equivalence on recorded clips; no temp files.

### P6 — Real-world validation & release (M6)

1. **T-6.1** Scripted session videos from held-out signers.
2. **T-6.2** RW protocol with ≥ 5 people (07_TEST §6).
3. **T-6.3** Error analysis → fix loop.
4. **T-6.4** 30-minute soak test.
5. **T-6.5** Model card + dataset datasheet (sources, licences, signer stats as consented, metrics by evaluation type, limits, intended use/non-use).
6. **T-6.6** README quick start + user guide (install, data root, add a language, train, run live).
7. **T-6.7** (C) Windows packaging.
8. **T-6.8** v1.0 release checklist (mirrors PRD §9) → tag (owner).

### P7 — Multilingual (v2; after v1.0)

1. **T-7.1** Register L2; adapter + vocab + gloss map; extract with the **same `extractor_id`**; build `<L2>-v1`.
2. **T-7.2** Mixed-language batching with per-language loss weights (TR-TRAIN-6).
3. **T-7.3** Evaluate per language: L1-only vs L2-only vs co-trained (does co-training help the smaller language? [Logos](https://arxiv.org/abs/2505.10481) reports it does).
4. **T-7.4** (optional) Language-ID head / automatic language selection.
5. **T-7.5** Serve multiple languages from one package (`--lang` or auto).

---

## 4. Suggested order for the next sessions

| # | Tasks | Why |
|---|-------|-----|
| 1 | ~~Owner decisions Q-10…Q-12~~ **done** (ASL, research/non-commercial, 100 signs). Q-13 (backup paths) still open | Everything data-related depends on them |
| 2 | T-0.1, T-0.2, T-0.3 | Clean, testable home for v1 |
| 3 | T-0.4 (needs the Q-13 paths), T-0.5 | Data safety before data exists |
| 4 | T-1.1, T-1.2, T-3.1, T-3.2 (ports with tests) | Proven pieces first; no data needed |
| 5 | T-1.3 (record ~10 quick clips of yourself for the spike) | Unblocks extraction and pins MediaPipe |
| 6 | T-1.4, T-2.1, T-2.2, T-2.3 | The pipeline can now build a dataset |
| 7 | T-2.4, T-2.5, then start recruiting (T-2.8) | Longest lead time |
| 8+ | T-2.7 / T-2.9 → P3 real runs → P4 → P5 → P6 | |

## 5. Risk register

| Risk | L | I | Mitigation / trigger |
|------|---|---|---------------------|
| Data loss (again) | M | Critical | T-0.4 before any data; backup verify after every data session; second copy off the laptop; restore drill each milestone |
| Use changes to commercial later | L | High | v1 is research/non-commercial (D-018); a commercial version needs retraining on commercially licensed data (own recordings with suitable consent) |
| MediaPipe Tasks gaps or bugs (API churn) | M | M | Spike both variants (T-1.3); pin exact versions + model hashes; backend interface isolates the rest |
| Disk space for public video datasets (C: 31 GB) | H | M | Data root on E: (100 GB); chunked processing; keep landmarks (small), archive videos externally |
| Laptop compute for bulk extraction or training | M | M | Multi-process extraction; measure throughput early (T-1.4); CUDA for training (T-3.10); smaller vocabulary first |
| Signer-independent accuracy below target | M | H | Face subsets, cnn_transformer, more signers, background class, expert-reviewed gloss maps; revisit the 100-sign vocabulary with the owner (D-019) |
| Gloss-mapping errors across sources | M | M | Reviewed `gloss_map` files; `DROP` when unsure; per-source accuracy breakdown |
| Multilingual scope creep before L1 works | M | H | P7 only after v1.0; structural support only (ST-8) |
