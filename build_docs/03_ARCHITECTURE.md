# 03 — Architecture

**Part A** is the **v1 target architecture**: what we build. **Part B** summarises
the **v0 reference** (what exists, what to port, what to avoid). **Part C** holds the
architecture decisions (ADRs).

---

# Part A — SignX v1 target architecture

## A1. Principles

1. **Language is data, not code.** Languages, vocabularies, gloss mappings, and licences live in registry files. Code never names a language or gloss.
2. **One path from pixels to prediction.** One landmark schema, one detection backend configuration per dataset version, one feature transform: shared by dataset building, training, file inference, and live serving.
3. **Immutable, checksummed data.** Datasets are versioned artifacts outside the repo, with a backup manifest in the repo.
4. **Shared encoder, per-language heads.** Built for N languages from day one; v1.0 trains N = 1.
5. **Stream at serve time.** Per-frame landmarks while signing; the UI never waits for the model.
6. **Every stage is measurable.** Each stage writes a machine-readable report; each training run appends to a registry.

## A2. System overview

```
                         ┌──────────────── registry/ (in git) ────────────────┐
                         │ languages.toml · sources.toml (licences)           │
                         │ vocab/<lang>.csv · gloss_map/<source>.csv          │
                         └───────────────┬────────────────────────────────────┘
                                         │ drives
 SOURCES                 INGEST                  LANDMARKS                 DATASETS
┌──────────────┐   ┌──────────────────┐   ┌──────────────────────┐   ┌─────────────────────────┐
│ public sets  │──►│ adapters/        │──►│ extract (bulk,       │──►│ build-dataset           │
│ (ASL Citizen,│   │  folder          │   │ resumable, multi-    │   │  gloss_map → sign_id    │
│  INCLUDE, …) │   │  recordings      │   │ process)             │   │  QA filter, splits      │
│ own          │   │  <public>        │   │ DetectorBackend      │   │  (by signer)            │
│ recordings ──┼──►│ → manifest rows  │   │ (MediaPipe Tasks)    │   │ → datasets/<ver>/       │
│ (signx record)   │   + video paths  │   │ + One-Euro smoothing │   │    manifest.csv + hash  │
└──────────────┘   └──────────────────┘   │ → landmarks/*.npz    │   └───────────┬─────────────┘
                                          └──────────────────────┘               │
   TRAINING                                                                      ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ features.transform (+augment in train) → cache/features/<hash>/                           │
│ Encoder (bigru | transformer | cnn_transformer) → embedding → heads{lang}                  │
│ train on train-signers · select on val-signers · report on test-signers                   │
│ → runs/<run_id>/ (ckpt, report.json, plots) + experiments.jsonl                           │
└───────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                ▼ export
                          models/<model_version>/ {<lang>.onnx, model_meta.json}
                                                │
   SERVING                                      ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ Capture/UI thread ─frames─► Landmark worker (persistent backend, ring buffer, segmenter)  │
│        ▲                         │ closed segment (T,1692)                                │
│        └──── captions ◄──── Classifier (features → ONNX[lang] → calibrate → gates →       │
│                              debounce) ──► sessions/<ts>.json                             │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

## A3. Repository layout (v1)

```
sign-language-translation/
├── CLAUDE.md                     agent entry point
├── build_docs/                   this operating manual
├── pyproject.toml                minimal; `pip install -e .` (T-0.3)
├── requirements.txt / requirements-dev.txt
├── registry/                     ← DATA FILES in git (small, human-reviewed)
│   ├── languages.toml            code, name, region
│   ├── sources.toml              source id, licence, commercial_ok, url, citation
│   ├── vocab/<lang>.csv          sign_id, gloss, concept_en, vs_group, sources, status
│   └── gloss_map/<source>.csv    source_gloss → sign_id | DROP
├── configs/
│   └── experiments/*.toml        one file per experiment
├── signx/                        ← THE PACKAGE
│   ├── __main__.py               CLI dispatch: python -m signx <command>
│   ├── paths.py                  SIGNX_DATA_ROOT resolution + layout helpers
│   ├── config.py                 dataclasses + TOML loading (tomllib)
│   ├── lang/                     languages.py · vocab.py · labelmap.py · sources.py
│   ├── landmarks/                schema.py · backend.py · tasks_backend.py · smoothing.py · draw.py
│   ├── data/                     adapters/{base,folder,recordings,<public>}.py · extract.py · qa.py
│   │                             manifest.py · splits.py · build.py · backup.py · record.py
│   ├── features/                 transform.py · augment.py · cache.py
│   ├── models/                   encoders.py · heads.py · classifier.py
│   ├── train/                    run.py · loop.py · calibrate.py · registry.py
│   ├── eval/                     metrics.py · report.py · session_score.py
│   ├── export/                   onnx_export.py · package.py
│   └── serve/                    predictor.py · segmenter.py · debounce.py · live.py · video.py · ui.py
├── tests/                        unit · component (synthetic) · contract · data/model-marked
├── tools/                        one-off scripts not part of the product (optional)
└── reference/v0/                 archived v0 code + docs (read-only; T-0.1)
```

Create modules **only when a task needs them** (R-3.6). The layout above is the map, not a scaffold to generate up front.

## A4. Data contracts (versioned; changing one needs an ADR)

### A4.1 Landmark schema `mp-1692-v1`

| Block | Columns | Per frame | Source (Tasks API) |
|-------|---------|-----------|--------------------|
| pose | 0–131 | 33 × (x, y, z, visibility) | PoseLandmarker |
| face | 132–1565 | 478 × (x, y, z) | FaceLandmarker (468 mesh + 10 iris) |
| left_hand | 1566–1628 | 21 × (x, y, z) | HandLandmarker, **anatomical** left |
| right_hand | 1629–1691 | 21 × (x, y, z) | HandLandmarker, anatomical right |

Coordinates are normalised image coordinates. **Absent = zeros.** `mask` (T, 4) uint8 per
block [pose, face, left, right]. This is the same layout as v0 (proven), now with an
explicit schema id.

### A4.2 Sample file: `landmarks/<schema>/<extractor_id>/<source>/<sample_id>.npz`

| Key | Type | Notes |
|-----|------|-------|
| `landmarks` | float16 (T, 1692) | float16 halves storage; the transform upcasts to float32 |
| `mask` | uint8 (T, 4) | presence per block |
| `t_ms` | int64 (T,) | frame timestamps (enables real-time velocity) |
| `fps` | float32 scalar | source fps |
| `meta` | JSON string | sample_id, video_sha256, extractor provenance (backend, mediapipe version, model hashes, smoothing params, processing width) |

`extractor_id` = short hash of the provenance. **A dataset version contains exactly one `extractor_id`.**

### A4.3 Manifest `datasets/<dataset_version>/manifest.csv`

`sample_id, language, sign_id, gloss, source, source_split, signer_id, session_id, handedness, fps, frames, width, height, extractor_id, qa_verdict, split, video_sha256, npz_sha256, licence_id, consent_id`

- `sample_id` is globally unique: `<source>:<source_sample_key>`.
- `split` ∈ {train, val, test, excluded}; **a signer appears in exactly one of train/val/test**.
- `datasets/<dataset_version>/dataset.json`: name, languages, label-map version per language, extractor_id, created, content hash, counts per split/class/signer.

### A4.4 Registries (in git)

- `languages.toml`: `[ase] name="American Sign Language" region="US/CA"` …
- `vocab/<lang>.csv`: `sign_id` = `<lang>:<GLOSS>` (upper-case gloss, `_` for spaces); includes `<lang>:_BACKGROUND`.
- **Label map** (generated, frozen per model): `{language, version, classes: [sign_id…]}`, where index = logit position.
- `sources.toml`: `[asl_citizen] licence="MSR-…" commercial_ok=false url=… citation=…`.

### A4.5 Feature layout

Computed from `FeatureConfig` (point subsets, channels), never hard-coded. `features.spec(config)`
returns the exact column map, which is stored in model meta.

### A4.6 Model package `models/<model_version>/`

- `<lang>.onnx`: input `features` [B, L, D] float32 → output `logits` [B, C_lang].
- `model_meta.json`: `model_version` (semver), `languages`, `label_maps{lang}`, `feature_config`, `feature_spec`, `landmark_schema`, `extractor_id`, `calibration{lang: temperature}`, `metrics{lang: {eval_type, top1, top5, macro_f1, per_signer…}}`, `dataset_version`, `sources` + `commercial_ok`, `git_sha`, `created`, `onnx_parity`.

### A4.7 Session log `sessions/<YYYYmmdd-HHMMSS>.json`

Model version, language, thresholds, and per segment: `t_start, t_end, T, top3, conf, margin, hands_rate, decision (emit | abstain | background | repeat), latency {landmarks_ms, features_ms, model_ms, emit_ms}, dropped_frames`.

## A5. Detection backend

```python
@dataclass
class FrameLandmarks:
    pose: np.ndarray | None        # (33, 4)
    face: np.ndarray | None        # (478, 3)
    left_hand: np.ndarray | None   # (21, 3)  anatomical
    right_hand: np.ndarray | None  # (21, 3)

class DetectorBackend(Protocol):
    def detect(self, rgb: np.ndarray, t_ms: int) -> FrameLandmarks: ...
    def info(self) -> dict: ...          # provenance → extractor_id
    def close(self) -> None: ...

def to_row(fl: FrameLandmarks) -> tuple[np.ndarray, np.ndarray]:  # (1692,), (4,) per A4.1
```

- `TasksBackend`: MediaPipe Tasks landmarkers in `VIDEO` running mode with monotonically increasing `t_ms`; `mirrored` flag drives the handedness swap (Tasks assumes mirrored input).
- Smoothing (`OneEuro`) wraps any backend: `Smoothed(backend, params)`.
- Modality selection: `modalities={"pose","hands","face"}`. Skipped blocks are zero-filled and masked 0.

## A6. Dataset build flow

```
signx data fetch <source>        (optional; only for sources with a download procedure)
signx ingest <adapter> …         → staging rows (video path, gloss, signer, source split)
signx extract --backend tasks    → landmarks/*.npz (resumable, parallel), per-sample QA
signx build-dataset <config>     → gloss_map → sign_id; drop unmapped; QA filter;
                                   splits (official or by signer); background class;
                                   manifest.csv + dataset.json + content hash
signx backup write|verify        → backups/BACKUP_MANIFEST.json
```

## A7. Model

```
features (B, L, D)
  → LayerNorm → Linear(D→d) → GELU → Dropout
  → Encoder:  bigru(d, layers)  |  transformer(d, heads, layers)  |  cnn_transformer(d, k, blocks)
  → Pool: attention | mean
  → embedding e (B, d_e)
  → heads: ModuleDict{ lang: Dropout → Linear(d_e → C_lang) }
forward(x, lang) → logits for that language's head
```

- **Training, one language (v1.0):** standard cross-entropy on `heads[L1]`.
- **Training, several languages (v2):** a batch is grouped by `language`; each group goes through the shared encoder and its own head; the loss is a weighted sum over languages (weights ∝ data share or tuned). This is the multi-head co-training scheme reported to help low-resource languages ([Logos](https://arxiv.org/abs/2505.10481)).
- **Export:** a thin wrapper `EncoderWithHead(model, lang)` → one ONNX per language (small; the serving path picks by `--lang`).

## A8. Serving pipeline

| Thread | Work | Never does |
|--------|------|-----------|
| **A Capture/UI** | read camera → push frame (drop-oldest queue) → draw overlay + latest caption at camera fps | model inference, landmark detection |
| **B Landmark worker** | persistent `Smoothed(TasksBackend)` → `to_row` → ring buffer (pre-roll) → segmenter step (frame-diff or hand-velocity) → on close, send `(T,1692)` + masks + t_ms | UI drawing |
| **C Classifier** | `features.transform` → `predict_robust` (whole + windows) with ONNX[lang] → temperature → gates (background, conf, margin, hands) → debounce → caption event + session record | touching the camera |

State shown to the user: LISTENING / REC (motion bar) / result flash / "?" / transcript.

## A9. Adding a new sign language (the multilingual path)

1. Add the code to `registry/languages.toml` (verified ISO 639-3).
2. Create `registry/vocab/<lang>.csv` (with `_BACKGROUND`) and `gloss_map/<source>.csv`.
3. Add the source to `registry/sources.toml` (licence!). Write an adapter only if the source format is new.
4. `signx extract` with **the same backend configuration** (`extractor_id`) as the other languages you'll co-train with.
5. `signx build-dataset` → `<lang>-<name>-v1`.
6. Train: a config with `languages = ["ase", "<lang>"]`. Heads are created from the label maps automatically.
7. Evaluate per language; export `<lang>.onnx`; serve with `--lang <lang>`.

**No code change is expected.** Test ST-8 enforces this with two synthetic languages.

## A10. Configuration

- `signx/config.py`: dataclasses `DataConfig`, `FeatureConfig`, `ModelConfig`, `TrainConfig`, `ServeConfig` with defaults.
- `configs/experiments/<name>.toml` overrides the defaults and is loaded with stdlib `tomllib` (no dependency). Unknown keys are an error.
- CLI flags override config values for quick runs. The **effective** config is saved with every run and model.

---

# Part B — v0 reference (as built; archived to `reference/v0/` by T-0.1)

## B1. What v0 is

12 mixed ASL/ISL signs, 134 clips, one signer. Landmarks via legacy `mp.solutions`
(FaceMesh + Hands + Pose, 1080p), One-Euro smoothing. `features.py` (L=64, D=346).
BiGRU + attention (543,553 params). 5-fold within-signer CV: **0.910** OOF accuracy.
ONNX ~1.2 ms. Motion-gated live loop that records → writes a temp MP4 → re-extracts →
classifies (blocking). **Data and weights are lost; the code works (self-tests pass).**

## B2. Port map: v0 → v1

| v0 file | What's worth keeping | v1 destination | Action |
|---------|----------------------|----------------|--------|
| `landmark_detector.py` | One-Euro filter, `PointStabilizer`, `HandStabilizer`, `anatomical_label`, `match_hands_to_sides`, drawing | `landmarks/smoothing.py`, `landmarks/draw.py` | **Port** (drop the `mp.solutions` model setup) |
| `extract_dataset.py` | per-clip loop, error isolation, `build_all_vector` layout, `qa_clip`, skeleton rendering | `data/extract.py`, `data/qa.py`, `landmarks/schema.py` | **Rewrite** on `DetectorBackend`; add npz/t_ms/provenance |
| `verify_landmarks.py` | structural + consistency checks | `data/qa.py` | Port the checks |
| `prune_clips.py` | reversible exclusion | `split = excluded` in the manifest | Replace |
| `dataset_report.py` | QA charts | `eval/report.py` (data section) | Port later (S) |
| `observe_video.py` | continuous-video extraction + timeline | — | Keep in reference (future continuous work) |
| `training/config.py` | dataclass config, `feature_config_from_dict` | `signx/config.py` | Port + TOML |
| `training/features.py` | `build_spec`, `geometric`/`assemble` split, normalisation, flip permutations, ghost-hand re-zeroing | `features/transform.py` | **Port** + face subsets + t_ms velocity + accel + wrist-relative |
| `training/data.py` | augmentations (flip with L/R swap, rotate, scale, shift, jitter, time warp, frame drop) | `features/augment.py`; dataset from the cache | Port |
| `training/model.py` | `AttentionPool`, BiGRU/Transformer encoder | `models/encoders.py`, `models/heads.py` | Port + `cnn_transformer` + heads |
| `training/utils.py` | seeding, metrics without sklearn, UTF-8 console fix | `train/`, `eval/metrics.py` | Port + top-5, per-signer |
| `training/train.py` | train loop, warmup-cosine schedule, early stopping | `train/loop.py`, `train/run.py` | Rewrite for split-based + registry |
| `training/finetune.py` | EMA with decay warm-up | `train/loop.py` (EMA option) | Port |
| `training/export_optimize.py` | parity check, latency measurement, meta beside ONNX | `export/onnx_export.py`, `export/package.py` | Rewrite (per-language, exporter choice) |
| `training/predictor.py` | `predict_robust` (whole + windows averaging, margin) | `serve/predictor.py` | Port; **drop** `video_to_clip` temp-file path |
| `training/segmenter.py` | `MotionSegmenter`, `SignDebouncer` + self-test | `serve/segmenter.py`, `serve/debounce.py` | **Port as-is** + hand-velocity source |
| `training/infer_video.py` | overlay rendering, timeline | `serve/video.py` | Port |
| `training/infer_live.py` | overlay/UI states, gates, `--source` simulation | `serve/live.py`, `serve/ui.py` | **Rewrite** as threaded streaming |
| `training/finetune (2).py` | — (duplicate) | — | **Delete** |

## B3. v0 lessons (don't repeat)

| # | v0 issue | Evidence | v1 answer |
|---|----------|----------|-----------|
| L-1 | ASL + ISL signs mixed in one label set | owner, 2026-09-30 | ADR-101 language-namespaced labels |
| L-2 | One signer; within-signer CV only | manifest has no signer_id | ADR-109 signer-independent splits |
| L-3 | Data/weights only on one laptop → lost | owner, 2026-09-30 | ADR-105 data root + backup manifest + second copy |
| L-4 | Legacy `mp.solutions` (removed upstream) | `landmark_detector.py:75-79` | ADR-104 Tasks backend |
| L-5 | Blocking live loop; re-extraction after each sign (~40 ms/frame); fresh models per segment | `infer_live.process_segment` | ADR-107 streaming threads |
| L-6 | Face extracted live though unused | `video_to_clip(face=True)` | TR-DET-9 modality selection |
| L-7 | Train 1080p vs serve 480 px | CLI defaults | TR-DET-11 |
| L-8 | No background class | `classes.json` | ADR-108 |
| L-9 | Legacy TorchScript ONNX exporter; int8 useless for GRU | 07_TEST 2026-09-30 | ADR-111 / TR-EXPORT-3/4 |
| L-10 | Features cached in RAM | `data.load_geometric` | TR-FEAT-7 on-disk cache |
| L-11 | Velocity in resampled-index units | `features.assemble` | TR-FEAT-5 real-time units option |

---

# Part C — Architecture Decision Records

Format: **ID — decision.** Why. *Status.*

### Inherited from v0 (still valid)
- **ADR-001 — One shared feature transform for train and serve.** Prevents skew; augmentation sits between geometry and assembly. *Accepted.*
- **ADR-002 — Per-sample normalisation (median mid-shoulder, median shoulder width); no fitted statistics.** *Accepted.*
- **ADR-006 — ONNX Runtime on CPU for serving.** *Accepted.*
- **ADR-007 — Motion-gated segmentation for isolated signs.** *Accepted; add a hand-velocity source.*
- **ADR-008 — Abstain (confidence + margin) and de-duplicate repeats.** *Accepted.*
- **ADR-010 — No scikit-learn; numpy + torch utilities.** *Accepted.*

### Superseded v0 decisions
- ADR-003 (face always off) → replaced by ADR-103 (configurable face subsets, chosen by evaluation).
- ADR-004 (BiGRU only) → replaced by an encoder registry chosen by evaluation (TR-MODEL-1).
- ADR-005 (within-signer k-fold as the headline metric) → replaced by ADR-109.
- ADR-009 (artifacts gitignored) → strengthened by ADR-105.

### New for v1 (proposed 2026-09-30; become *Accepted* when the owner confirms)
- **ADR-101 — Language-namespaced labels (`<iso639-3>:<GLOSS>`) and file-based registries.** Prevents v0's language mixing; makes languages data rather than code. *Proposed.*
- **ADR-102 — Shared encoder + one classification head per language, from day one.** Cheap now; enables co-training later without migration. *Proposed.*
- **ADR-103 — Keep the 1692-float landmark layout as schema `mp-1692-v1`; face subsets are selected in features, not at extraction.** Proven layout; matches Tasks outputs; extraction stays model-agnostic. *Proposed.*
- **ADR-104 — Detection behind `DetectorBackend`; MediaPipe Tasks is the primary backend; exactly one `extractor_id` per dataset version.** The legacy API is gone upstream; mixed extractors create domain shift. *Proposed.*
- **ADR-105 — Data root outside the repo; immutable, content-hashed dataset versions; backup manifest in git; a second physical copy; a restore drill.** v0's data loss. *Proposed.*
- **ADR-106 — Experiment configs in TOML via stdlib `tomllib`.** Readable, no dependency. *Proposed.*
- **ADR-107 — Threaded streaming live pipeline (no temp files).** v0 froze for about 1.2× the sign duration. *Proposed.*
- **ADR-108 — A background class per language + gates.** Rejects non-sign motion. *Proposed.*
- **ADR-109 — Signer-independent held-out splits are the headline metric; within-signer numbers are diagnostic only.** *Proposed.*
- **ADR-110 — Archive v0 under `reference/v0/` (git mv, history preserved); v1 lives in `signx/`.** Clean separation; v0 stays runnable for comparison. *Proposed.*
- **ADR-111 — One ONNX per language (encoder + head); the export path (dynamo vs pinned legacy exporter) is chosen in T-5.1.** *Proposed.*
- **ADR-112 — Per-sample `.npz` (float16) + CSV manifest as the storage format; evaluate `pose-format` for interop later.** Simple, dependency-free, inspectable. *Proposed.*
