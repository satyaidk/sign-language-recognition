# 06 — Tasks

The single list of work items for **SignX v1**. The why and the order are in
`05_IMPLEMENTATION_PLAN.md`. A task is `DONE` only when its acceptance criteria
are met **with evidence** (04_RULES §12).

**Status:** `TODO` · `IN-PROGRESS (date)` · `BLOCKED (reason)` · `DONE (date)` · `DROPPED (reason)`
**Pri:** M must · S should · C could    **Size:** S ≤ ½ day · M 1–2 days · L 3–5 days · XL people/data-bound
**Owner:** `agent` (Claude/dev can do it) · `owner` (needs the project owner's decision or action)

---

## Board

| ID | Task | Pri | Size | Owner | Status | Depends on |
|----|------|-----|------|-------|--------|-----------|
| **D-Q** | Owner decisions: Q-10 ✅ ASL · Q-11 ✅ 100 signs + background · Q-12 ✅ research/non-commercial · **Q-13 ⏳ exact data-root + backup paths** | M | S | owner | IN-PROGRESS (2026-09-30) | — |
| T-0.1 | Archive v0 → `reference/v0/`; delete duplicate `finetune (2).py`; update CLAUDE.md | M | S | agent | SUPERSEDED (2026-09-30): v0 was refactored in place into `signlang/`; duplicate deleted; CLAUDE.md updated. Decide at restart whether v1 evolves `signlang/` or starts fresh | — |
| T-0.2 | New venv `.venv-signx` + pinned requirements | M | S | agent | PARTLY DONE (2026-09-30): `requirements.txt` / `requirements-dev.txt` pinned for the v0.2 env (MediaPipe 0.10.9); v1 still needs its own env with the Tasks API | — |
| T-0.3 | Package skeleton (`pyproject`, `signx/`, CLI, paths, config) + pytest scaffold | M | M | agent | PARTLY DONE (2026-09-30): `signlang` package, `pyproject.toml`, CLI, configurable paths and a 100-test pytest suite exist; the v1 language layer does not | T-0.2 |
| T-0.4 | Data root + `signx backup write/verify` + restore drill doc | M | M | agent | TODO | T-0.3, Q-13 |
| T-0.5 | `signx doctor` | S | S | agent | TODO | T-0.3 |
| T-1.1 | Landmark schema `mp-1692-v1` + `to_row` + tests | M | S | agent | TODO | T-0.3 |
| T-1.2 | `DetectorBackend` protocol + One-Euro smoothing port + tests | M | M | agent | TODO | T-1.1 |
| T-1.3 | Tasks backend spike (landmarkers vs Holistic) → pin MediaPipe | M | M | agent (+owner: ~10 clips) | TODO | T-1.2 |
| T-1.4 | Bulk extraction engine (npz, provenance, resumable, multi-process) + draw port | M | L | agent | TODO | T-1.3 |
| T-1.5 | QA + verification + overlay spot-check | M | M | agent | TODO | T-1.4 |
| T-1.6 | Processing-resolution decision | M | S | agent | TODO | T-1.3 |
| T-2.1 | `signx/lang/` + `registry/` files + contract tests | M | M | agent | TODO | T-0.3 |
| T-2.2 | Manifest schema + `build-dataset` (gloss map, QA filter, splits, hash) | M | L | agent | TODO | T-2.1, T-1.4 |
| T-2.3 | `folder` + `recordings` adapters | M | M | agent | TODO | T-2.2 |
| T-2.4 | `signx record` + recording protocol + consent template | M | M | agent | TODO | T-2.1, T-1.4 |
| T-2.5 | Background-class protocol | M | S | agent | TODO | T-2.4 |
| T-2.6 | Register `ase`, ASL Citizen licence, choose the 100-sign vocabulary | M | S | owner + agent | TODO | T-2.1 |
| T-2.7 | ASL Citizen adapter + chunked extract (official splits) | M | L | agent | TODO | T-2.6, T-2.2 |
| T-2.8 | Own recordings: background class (≥ 5 people) + held-out ASL test signers (≥ 5; ≥ 2 Deaf/fluent; ≥ 1 left-handed) | M | XL | owner | TODO | T-2.4, T-2.5 |
| T-2.9 | Build `ase-core100-v1` + backup (2 copies) | M | M | agent | TODO | T-2.7, T-2.8 (background) |
| T-3.1 | Feature transform port + extensions + tests | M | M | agent | TODO | T-1.1 |
| T-3.2 | Augmentation port + tests | M | S | agent | TODO | T-3.1 |
| T-3.3 | On-disk feature cache | M | S | agent | TODO | T-3.1, T-2.2 |
| T-3.4 | Encoders (bigru, transformer, cnn_transformer) + language heads | M | M | agent | TODO | T-0.3 |
| T-3.5 | Config-driven training run (TOML, CPU/CUDA, EMA, early stop, resume) | M | L | agent | TODO | T-3.2, T-3.4 |
| T-3.6 | Metrics + reports (top-1/5, macro-F1, per-signer, confusions, ECE) | M | M | agent | TODO | T-3.5 |
| T-3.7 | Experiment registry | M | S | agent | TODO | T-3.5 |
| T-3.8 | Temperature calibration | S | S | agent | TODO | T-3.6 |
| T-3.9 | Multilingual dry run ST-8 (2 synthetic languages, 2 heads) | M | M | agent | TODO | T-3.5, T-2.1 |
| T-3.10 | CUDA torch in venv (optional) | S | S | agent (+owner OK) | TODO | T-0.2 |
| T-4.1 | L1 baseline (signer-independent) | M | M | agent | TODO | T-2.9, T-3.6 |
| T-4.2 | Feature ablations × 3 seeds | S | L | agent | TODO | T-4.1 |
| T-4.3 | Encoder comparison | S | L | agent | TODO | T-4.1 |
| T-4.4 | Background class + gate thresholds (val) | M | M | agent | TODO | T-4.1 |
| T-4.5 | Error analysis with a language expert; `vs_group`; gloss-map fixes | M | M | owner + agent | TODO | T-4.1 |
| T-4.6 | Candidate selection → test-split report → `1.0.0-rc1` | M | S | agent | TODO | T-4.2..T-4.5 |
| T-5.1 | Export path decision + per-language ONNX + parity tests | M | M | agent | TODO | T-3.4 |
| T-5.2 | Model package + predictor (meta, label maps, calibration, robust windows) | M | M | agent | TODO | T-5.1 |
| T-5.3 | Segmenter + debouncer port (+ hand-velocity source) | M | M | agent | TODO | T-0.3 |
| T-5.4 | Threaded streaming live app `signx live` | M | L | agent | TODO | T-5.2, T-5.3, T-1.4 |
| T-5.5 | Session logs + latency instrumentation | M | S | agent | TODO | T-5.4 |
| T-5.6 | File mode `signx video` | S | M | agent | TODO | T-5.2 |
| T-5.7 | Session scorer `signx eval-session` | M | M | agent | TODO | T-5.4 |
| T-5.8 | Live benchmark (fps, latency p95, RSS) | M | S | agent | TODO | T-5.4 |
| T-6.1 | Scripted session videos (held-out signers) | M | M | owner | TODO | T-2.8 |
| T-6.2 | RW protocol with ≥ 5 people | M | L | owner + agent | TODO | T-4.6, T-5.8, T-6.1 |
| T-6.3 | Error analysis → fix loop | M | M | agent | TODO | T-6.2 |
| T-6.4 | 30-minute soak test | M | S | agent | TODO | T-5.4 |
| T-6.5 | Model card + dataset datasheet | M | S | agent | TODO | T-6.2 |
| T-6.6 | README quick start + user guide | M | S | agent | TODO | T-5.4 |
| T-6.7 | Windows packaging | C | M | agent | TODO | T-6.6 |
| T-6.8 | v1.0 release checklist + tag | M | S | owner | TODO | all M tasks |
| T-7.x | Multilingual v2 (see plan P7) | — | — | — | NOT PLANNED YET | v1.0 |

---

## Task cards

### Owner decisions

**D-Q Owner decisions** — see `08_MEMORY.md` §6. Answered 2026-09-30: **Q-10 → ASL (`ase`)** (D-017), **Q-12 → research/non-commercial** (D-018), **Q-11 → 100 signs + background** (D-019). **Still open: Q-13**: the owner chose "Other location" for data and backups; they need to give the exact data-root path and second-copy location. *Accept:* the Q-13 paths are recorded in 08_MEMORY §3.

### P0 — Reset & foundation

**T-0.1 Archive v0**
- *Do:* `git mv` everything v0 (`*.py` at the root, `training/`, `docs/`, `processed_dataset/`) into `reference/v0/`, keeping relative structure so v0 still runs. Delete `training/finetune (2).py` (verified identical to `finetune.py`, 2026-09-30) in its own commit. Update the command paths in `CLAUDE.md`.
- *Accept:* `python reference/v0/training/segmenter.py` → `[self-test] OK`; `python reference/v0/training/model.py` runs; the repo root contains only v1 items (+ `reference/`); history preserved (`git log --follow` works on a moved file).

**T-0.2 New venv + pins**
- *Do:* `py -3.11 -m venv .venv-signx`; `requirements.txt`: `numpy`, `opencv-contrib-python` (only this OpenCV wheel), `torch` (CPU wheel; index URL in a comment), `onnx`, `onnxruntime`, pinned to exact versions that install together (start from v0's verified set: numpy 2.4.4, opencv-contrib-python 4.13.0.92, torch 2.11.0, onnx 1.21.0, onnxruntime 1.24.4). MediaPipe is added by T-1.3. `requirements-dev.txt`: `-r requirements.txt`, `pytest`. Add `.venv-signx/` to `.gitignore`.
- *Accept:* clean install from scratch; `pip check` clean; versions recorded in 07_TEST §4.

**T-0.3 Package skeleton + test scaffold**
- *Do:* minimal `pyproject.toml` (name `signx`, packages `signx*`, Python ≥ 3.11); `signx/__init__.py` (version), `signx/__main__.py` (argparse sub-commands; UTF-8 console fix), `signx/paths.py` (read `SIGNX_DATA_ROOT`; layout helpers per 03_ARCHITECTURE A2/TR-DATA-2; clear error if unset), `signx/config.py` (dataclasses + `load_config(toml_path, overrides)` rejecting unknown keys); `tests/conftest.py` (markers `data`, `model`, `slow`; tmp data-root fixture).
- *Accept:* `pip install -e .`; `python -m signx --help` lists commands; `pytest -q` passes with tests for paths (unset/set root) and config (defaults, override, unknown-key error).

**T-0.4 Data root + backup**
- *Do:* `signx backup write` (walk `SIGNX_DATA_ROOT`, excluding `cache/`; write `backups/BACKUP_MANIFEST.json` in the repo with relative path, bytes, sha256, mtime); `signx backup verify [--root <copy>]` (reports missing/mismatched/extra); `docs`: backup + **restore drill** procedure in `build_docs/08_MEMORY.md` §3.
- *Accept:* tests on a temp root: write → verify OK → corrupt one byte → verify fails naming the file → delete a file → verify reports it missing. The data root is created at the agreed location (Q-13; proposal `E:\signx-data`).

**T-0.5 `signx doctor`** — Python/package versions vs pins, CUDA availability, camera 0 opens, data root exists and is writable, free disk on the data drive and C:, age of the last backup verify. *Accept:* runs in < 5 s; non-zero exit on a missing requirement; a test with a mocked environment.

### P1 — Landmark layer

**T-1.1 Schema** — `landmarks/schema.py`: block offsets, `FrameLandmarks → (1692,) + (4,) mask`, stack to (T, …), float16 pack/unpack. *Accept:* UT-10 offsets match A4.1; round-trip float16 error < 1e-3 on [0,1] coordinates; absent → zeros + mask 0.

**T-1.2 Backend protocol + smoothing** — `landmarks/backend.py` (`FrameLandmarks`, `DetectorBackend`, `Smoothed` wrapper), port `OneEuroFilter`/`PointStabilizer`/`HandStabilizer` from v0 into `landmarks/smoothing.py`. *Accept:* unit tests: the One-Euro filter reduces jitter on a noisy synthetic sine while keeping a step response within N frames; a fake backend + `Smoothed` produce the expected rows; no occlusion back-fill (hold = 0).

**T-1.3 Tasks backend spike → pin**
- *Do:* the owner records ~10 short clips (a few signs, both hands, one mirrored webcam clip and one normal video; any language is fine for this test). Implement `TasksBackend` variant (a) Pose + Hand (2) + Face landmarkers and (b) HolisticLandmarker, `VIDEO` mode, with a `mirrored` flag. A download script for `.task` model files with sha256 pins, stored under `SIGNX_DATA_ROOT/models/mediapipe/`.
- *Accept:* a comparison table in 07_TEST (detection rate per modality, ms/frame at 480/720/1080, handedness correct on both clips, crashes/warnings) → decision D-n (variant + exact MediaPipe version) → `requirements.txt` updated; UT-13 handedness test passes.

**T-1.4 Extraction engine** — `data/extract.py` + `signx extract`: video → frames (at the chosen processing width) → backend → smoothing → npz (A4.2) with `t_ms`, provenance, `extractor_id`; skip if the npz exists with a matching video sha; `--workers N` (spawn-safe); per-video try/except with an error log; per-sample QA metrics; port `draw.py` for skeleton/overlay rendering. *Accept:* on the T-1.3 clips: all npz valid (CT-12); a re-run skips everything; `--workers 3` speeds up vs 1 (recorded); throughput recorded (videos/hour, frames/s).

**T-1.5 QA + verification** — port v0 checks (shape, value ranges, mask consistency, hand-slot sanity, frame count), dataset QA report, `signx qa overlay <sample_id>` renders an original+landmarks side-by-side for eyeballing. *Accept:* the checks catch synthetic corruptions (CT-13); report generated for the spike clips.

**T-1.6 Processing resolution** — *Accept:* D-n records the width used for extraction and live, with evidence (detection rates, latency) from T-1.3.

### P2 — Language & data layer

**T-2.1 Language layer** — `signx/lang/{languages,vocab,labelmap,sources}.py`; `registry/languages.toml`, `registry/sources.toml`, `registry/vocab/`, `registry/gloss_map/` (empty templates + a `README.md` in `registry/` explaining formats); label maps versioned and append-only. *Accept:* UT-14 (sign_id format and uniqueness; one language per sign_id), UT-15 (append-only label map; re-numbering raises), UT-16 (grep test: no language codes/glosses hard-coded in `signx/`).

**T-2.2 Manifest + dataset builder** — `data/manifest.py` (schema A4.3, validation), `data/splits.py` (official or grouped-by-signer with seed), `data/build.py` + `signx build-dataset <config>` (apply gloss_map; drop unmapped; QA filter; background class; splits; `dataset.json` with counts + content hash; refuse mixed `extractor_id`). *Accept:* UT-17 split leakage = 0; UT-18 deterministic hash; CT-14 build from a synthetic source → valid manifest; refusal tests (mixed extractor, unknown language, missing consent for recordings).

**T-2.3 Adapters** — `data/adapters/base.py`, `folder.py` (`<root>/<gloss>/*.mp4` + `signers.csv`), `recordings.py` (reads the `signx record` output). *Accept:* CT-15 on synthetic folders.

**T-2.4 Recording tool + protocol + consent** — `signx record --lang X --signer S --session N --reps 5 [--signs file]`: shows the gloss (+ optional reference image/video path from vocab), 3-2-1 countdown, records a fixed duration or until a key press, preview, retake, saves `raw/recordings/<lang>/<signer>/<session>/<sign_id>_<rep>.mp4` + a sidecar JSON (consent_id, handedness, camera, lighting). `build_docs/RECORDING_PROTOCOL.md`, `build_docs/CONSENT_TEMPLATE.md`. *Accept:* manual test recording 2 signs × 2 reps; the files validate with the `recordings` adapter.

**T-2.5 Background protocol** — a section in `RECORDING_PROTOCOL.md`: behaviours (rest, talking, face-touch, adjusting, reaching, transitions), durations, target count ≈ 2× the median class count. *Accept:* reviewed by the owner.

**T-2.6 Register ASL + choose 100 signs** — the `ase` entry in `languages.toml` (verify the code, R-2.9); `sources.toml` entry `asl_citizen` (licence URL, `commercial_ok = false`, citation, download date); `vocab/ase.csv` with **100 signs + `ase:_BACKGROUND`**. Selection criteria: everyday usefulness; ≥ ~20 videos per sign spread across signers in ASL Citizen's train split and present in its val/test splits; a few deliberate look-alike pairs; reviewed by an ASL signer if possible (Q-16). *Accept:* validation tests pass; the per-sign video/signer counts table is saved in `registry/vocab/ase_selection.md`.

**T-2.7 ASL Citizen adapter** — first confirm the download size/packaging and plan disk use (R-5.6). Then `adapters/asl_citizen.py`: parse the metadata (gloss → `ase:<GLOSS>` via a reviewed `gloss_map/asl_citizen.csv`, user/signer ID, **official split**), filter to the 100 chosen signs, chunked extraction. *Accept:* ≥ 1 chunk processed end to end; official split preserved (UT-17 leakage = 0); C: never below 10 GB (logged); counts per split/sign/signer reported.

**T-2.8 Own recordings** (owner) — ASL Citizen already provides signer diversity for the 100 signs, so own recordings serve three purposes: (1) the **`ase:_BACKGROUND` class** (≥ 5 people, protocol T-2.5; ASL Citizen has no non-sign clips); (2) **held-out live/RW test signers** (≥ 5; ≥ 2 Deaf/fluent ASL signers, ≥ 1 left-handed; 2 lighting setups), **never used for training**; (3) optionally extra training signers, if consent allows. Consent on file for all. *Accept:* counts per person/purpose/class recorded in 08_MEMORY.

**T-2.9 Build `ase-core100-v1`** — *Accept:* `datasets/ase-core100-v1/` with manifest + `dataset.json`; QA report; split counts (signers per split); `signx backup write/verify` OK on both copies; recorded in 08_MEMORY.

### P3 — Training platform

**T-3.1 Feature transform** — port v0 `build_spec/geometric/assemble/to_features`; add named face subsets (`lips`, `eyes`, `brows`, `nose`; index lists documented with their source), acceleration, wrist-relative hand coordinates, velocity per second from `t_ms`; `spec()` returns the column map. *Accept:* CT-1…CT-7 ported and passing; new tests for each extension; D computed from the config (no hard-coded 346).

**T-3.2 Augmentation** — port v0 `augment` (flip with L/R swap for the new point sets incl. face subsets, rotation, scale, shift, jitter, time warp/crop, frame drop). *Accept:* CT-4/5/6 ported; the flip permutation is correct for face subsets (symmetry pairs tested).

**T-3.3 Feature cache** — *Accept:* cache hit/miss keyed by (dataset hash, feature hash); training on a synthetic 5k-sample dataset stays under 2 GB RSS.

**T-3.4 Models** — `encoders.py` (bigru + attention pool, transformer, **cnn_transformer**: depthwise temporal conv blocks + 1–2 attention blocks), `heads.py` (`ModuleDict` per language), `classifier.py` (`forward(x, lang)`). *Accept:* UT shape tests per encoder; param counts logged; all ≤ 10 M params at the default size.

**T-3.5 Training run** — `signx train <config.toml>`: loads the dataset version + cache; class-balanced sampling option; AdamW + warmup-cosine; grad clip; EMA; early stopping on val-signer macro-F1; checkpoint/resume; CPU/CUDA + AMP; writes `runs/<run_id>/`. *Accept:* CT-8 (tiny synthetic fit ≥ 0.9); resume reproduces the same loss curve within tolerance; effective config saved.

**T-3.6 Metrics & reports** — *Accept:* UT for top-k, macro-F1, per-signer, ECE against hand-computed examples; report JSON + confusion PNG + top confused pairs.

**T-3.7 Experiment registry** — `experiments.jsonl` under `SIGNX_DATA_ROOT/runs/` + a copy of the summary in the repo (`reports/experiments.csv`, small). *Accept:* each run appends one line with git SHA/dirty, config hash, dataset hash, metrics, device, duration.

**T-3.8 Calibration** — *Accept:* temperature fitted on val logits (NLL); ECE drops on synthetic miscalibrated logits (UT); stored in meta.

**T-3.9 Multilingual dry run (ST-8)** — two synthetic "languages" using ISO 639-3's **reserved local-use codes** (`qaa`, `qab`, from the `qaa–qtz` range), registered only in a test registry with different class counts → build → train both heads in one run → per-language report → per-language ONNX. *Accept:* passes with **no code changes** beyond test fixtures.

**T-3.10 CUDA torch (optional)** — *Accept:* `torch.cuda.is_available()`; the same config trains on GPU; speed-up recorded.

### P4 — L1 model

**T-4.1 Baseline** — bigru, the v0-equivalent feature set (pose-upper + hands + velocity + masks). *Accept:* val and test (once) metrics recorded with evaluation type; per-signer table.
**T-4.2 Feature ablations** — face subsets / accel / wrist-relative / velocity units; 3 seeds on val. *Accept:* results table; adopt only > 1 std improvements (D-n).
**T-4.3 Encoder comparison** — bigru vs transformer vs cnn_transformer; 3 seeds; ONNX latency. *Accept:* choice recorded (D-n).
**T-4.4 Background + gates** — *Accept:* background recall and false-emit estimate on val-signer background clips; thresholds (τ_bg, conf, margin) chosen on val (D-n).
**T-4.5 Expert error analysis** — *Accept:* confused pairs reviewed; `vs_group` filled; gloss-map fixes → dataset v2 if needed.
**T-4.6 Selection → test report → rc1** — *Accept:* PRD §9 offline targets checked on test signers; `model_version 1.0.0-rc1`; registry entry.

### P5 — Export & live serving

**T-5.1 Export** — try `dynamo=True` (+ `onnxscript`) and the legacy exporter (pinned torch); choose (D-n). `EncoderWithHead(lang)` → `<lang>.onnx`. *Accept:* CT-9 parity < 1e-3 at batch 1 and 4 for every encoder type; latency per encoder recorded.
**T-5.2 Package + predictor** — *Accept:* the predictor loads only from the package; the label map/feature config come from meta; `predict_robust` ported with tests; calibration applied.
**T-5.3 Segmenter + debouncer** — port v0 as-is + a `hands` motion source using landmark velocity. *Accept:* UT-1/UT-2 ported to pytest; hand-velocity segmenter unit-tested on synthetic landmark traces.
**T-5.4 Live app** — threads per 03_ARCHITECTURE A8; `signx live --lang X [--source video]`. *Accept:* ST-6 equivalence (streaming vs file mode ≥ 95% same top-1 on recorded clips); no temp files; clean exit ≤ 1 s.
**T-5.5 Session logs** — *Accept:* the A4.7 schema is written; the latency breakdown is present.
**T-5.6 File mode** — *Accept:* overlay + timeline JSON for a video.
**T-5.7 Session scorer** — *Accept:* alignment unit-tested; the scorer outputs correct/sub/ins/del/abstain/latency.
**T-5.8 Live benchmark** — *Accept:* ST-4 row in 07_TEST: fps, drops, latency p50/p95, RSS, on the reference laptop.

### P6 — Real-world validation & release

**T-6.1…T-6.8** — as described in 05_IMPLEMENTATION_PLAN P6. The release checklist mirrors PRD §9 item by item.

---

## Dropped / superseded (from the first version of this plan)

| Old ID | Task | Why dropped |
|--------|------|-------------|
| old T-0.4 | Restore v0 data + weights | **Data permanently lost** (owner, 2026-09-30) |
| old T-0.6 | Reproduce the v0 baseline (0.91) | No data; v0 is not retrained (owner decision) |
| old T-1.2 | Byte-identical legacy refactor | Nothing to be identical to; v1 uses the Tasks backend |
| old T-3.1 | Backfill signer IDs for v0 clips | v0 data gone |
| old P4 items | Retrain the 12-class v0 model with a background class | v1 builds new vocabularies per language |

## Completed work (history)

| ID | Work | Date | Evidence |
|----|------|------|----------|
| H-1…H-11 | v0 prototype: landmark viewer, extraction + QA (134 PASS), verification, pruning, reports, docs; training pipeline (5-fold within-signer OOF 0.910); EMA final model; ONNX (parity 1.67e-6); file inference; motion-gated live mode with margin gate + de-dup | 2026-06-24 → 07-03 | `reference/v0/` (after T-0.1); v0 JSON reports; 07_TEST historical table |
| H-12 | `build_docs/` created; v0 self-tests + synthetic end-to-end verified; latency measurements | 2026-09-30 | 07_TEST §4 |
| H-13 | Direction change: language-agnostic v1; v0 = reference; `CLAUDE.md` added; build_docs rewritten | 2026-09-30 | 08_MEMORY D-012…D-016 |
| H-14 | **v0.2 refactor:** code reorganised into the layered `signlang` package + CLI; 15+ bugs fixed; streaming live mode; 100 tests; new docs + portfolio README | 2026-09-30 | root `CHANGELOG.md`; 08_MEMORY D-020 |
