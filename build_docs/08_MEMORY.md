# 08 — Project Memory

The project's long-term memory: **current state, decisions, gotchas, open
questions, session log.** Read it first every session; update it last (04_RULES §1).
Keep the snapshot current. The decision log and session log are **append-only**.

---

## 1. Current state snapshot (as of 2026-09-30)

| Area | State |
|------|-------|
| **Product** | **SignX v1**: a language-agnostic sign-recognition platform; v1.0 = **ASL (`ase`), 100 signs + `ase:_BACKGROUND`, research/non-commercial use**; multilingual co-training later (v2) |
| **Repo** | `https://github.com/satyaidk/sign-language-translation` · `main` · last commit `d8c1b62` "signx". **Uncommitted:** `build_docs/`, `CLAUDE.md` |
| **Phase** | **v1 plan PARKED (owner, 2026-09-30).** The current work was a refactor of the existing project into `signlang` v0.2.0 (D-020). When v1 restarts: T-0.1 superseded, T-0.2/T-0.3 partly done; T-0.4 still needs **Q-13** |
| **Primary data source (planned)** | **ASL Citizen** (Microsoft Research licence, non-commercial research; official signer-independent split), plus **own recordings** for the background class and live/real-world tests; WLASL optional later |
| **v0 code** | Refactored into the `signlang` package (L0 landmarks · L1 dataset · L2 training · L3 inference) with CLI `python -m signlang`. Original docs archived in `docs/archive/original/` |
| **v0 data / weights** | **Permanently lost** (owner deleted them after leaving the internship; unrecoverable). Only v0 JSON reports/metadata remain in git |
| **Data root** | Not created yet. Proposal: `E:\signx-data` (E: has 100 GB free; C: 31 GB) |
| **Backups** | None yet. T-0.4 builds the system before any data exists |
| **Environment** | Global Python 3.11 env; `requirements.txt` pins it (mediapipe 0.10.9 legacy API). pytest 9.1.1 installed. The v1 venv is not created yet |
| **Tests** | `python -m pytest -q`: **100 passed** (2026-09-30, ~17 s), plus an end-to-end CLI smoke test of all 11 commands on synthetic data. v1: none yet |

## 2. Key facts to remember

- **v0 lesson set** (03_ARCHITECTURE §B3): mixed ASL+ISL labels → confusion; one signer → misleading 0.91; data lost; legacy MediaPipe API removed upstream; live loop blocked ~1.2× sign duration; no background class; train 1080p vs serve 480 px.
- **v0 numbers** (diagnostic only): 12 classes, 134 clips, BiGRU 543,553 params, D = 346, L = 64, within-signer OOF 0.910, ONNX ~1.1–1.2 ms/clip, feature transform ~1.4 ms, legacy MediaPipe ~40 ms/frame (synthetic video).
- **Landmark schema to keep:** 1692 floats = pose 33×4 + face 478×3 + left hand 21×3 + right hand 21×3 (`mp-1692-v1`).
- **Label format:** `<iso639-3>:<GLOSS>` (e.g. `ase:HELLO`); background `<lang>:_BACKGROUND`; test-only languages use ISO local-use codes `qaa`/`qab`.
- **Reference machine:** i5-11320H, 7.8 GB RAM, RTX 3050 **4 GB** (driver 591.74; torch currently CPU-only), C: 31 GB free, E: 100 GB free, Windows 11, Python 3.11.0.

## 3. Environment, data root, backups

**v0 global env (verified 2026-09-30):** `mediapipe==0.10.9`, `protobuf==3.20.3`, `numpy==2.4.4`, `torch==2.11.0+cpu`,
`onnx==1.21.0`, `onnxruntime==1.24.4`, `opencv-python==4.13.0.92` + `opencv-contrib-python==4.13.0.92` (duplicate!),
`matplotlib==3.10.8`, `reportlab==5.0.0`. Not installed: `pytest`, `onnxscript`.

**v1 venv:** `.venv-signx` (T-0.2). Pins recorded here once created.

**Data root:** `SIGNX_DATA_ROOT` = *(to be set in T-0.4; proposal `E:\signx-data`)*.
**Second copy:** *(Q-13)*.
**Backup procedure:** *(T-0.4 fills this in: write → verify → sync the second copy → verify the copy.)*
**Restore drill:** *(T-0.4; run at every milestone; log the result in 07_TEST DT-8.)*

## 4. Decision log (append-only)

| ID | Date | Decision | Why | Ref |
|----|------|----------|-----|-----|
| D-001…D-007 | 2026-06-24 → 07-03 | v0 decisions: MediaPipe landmarks + One-Euro; anatomical hand slots; pruning 2 outliers; shared transform + per-clip normalisation + face off; BiGRU + k-fold + EMA; ONNX CPU; motion-gated live + margin + de-dup | v0 prototype | 03_ARCH Part C (inherited ADRs) |
| D-008 | 2026-09-30 | `build_docs/` is the project operating manual | single source of truth for agents | README |
| D-009 | 2026-09-30 | Signer-independent evaluation is the headline metric | the within-signer 0.91 over-estimated | ADR-109 (**still valid**) |
| D-010 | 2026-09-30 | ~~Keep mediapipe 0.10.9 pinned until Tasks parity~~ → **superseded by D-016**: v1 uses the Tasks API directly; 0.10.9 stays only in the v0 reference env | no v0 data left to stay compatible with | ADR-104 |
| D-011 | 2026-09-30 | Streaming threaded live pipeline; background class | fix freeze; reject non-sign motion | ADR-107, ADR-108 (**still valid**) |
| **D-012** | 2026-09-30 | **v0 data and weights are permanently lost**; no recovery attempts | owner: deleted after the internship ended, months ago | owner message |
| **D-013** | 2026-09-30 | **Do not retrain v0 or reuse its 12-class label set.** v0 code is a reference for pipeline, architecture, tests, training, and serving patterns | owner direction | ADR-110, R-0.7 |
| **D-014** | 2026-09-30 | **SignX v1 is language-agnostic.** v1.0 trains one sign language; the future goal is multi-sign-language recognition (supervised, labelled datasets from each country) with a shared encoder + per-language heads | owner direction | PRD §1, §10; ADR-101/102 |
| **D-015** | 2026-09-30 | v0's 12 signs were **ASL with some ISL mixed in**, which explains confusions → every label is namespaced by language and languages are never mixed in one head | owner information | ADR-101, R-2.2/2.3 |
| **D-016** | 2026-09-30 | **(proposed)** Adopt ADR-101…112: language registry + namespaced labels; per-language heads; schema `mp-1692-v1`; MediaPipe Tasks via `DetectorBackend`, one `extractor_id` per dataset; data root outside the repo + immutable dataset versions + backup manifest + second copy; TOML configs; threaded live; background class; signer-independent splits; archive v0 in `reference/v0/`; per-language ONNX; npz + CSV storage | architecture for the new direction | 03_ARCHITECTURE Part C |
| **D-017** | 2026-09-30 | **L1 = American Sign Language (`ase`)** | owner choice; largest public data with official signer-independent splits and mostly Deaf signers (ASL Citizen) | Q-10 |
| **D-018** | 2026-09-30 | **Use is research / non-commercial.** Non-commercial-licensed sources (ASL Citizen, WLASL C-UDA) are allowed; every model meta and model card states `commercial_ok = false` and lists the source licences | owner choice | Q-12, R-5.2, TR-PRIV-4 |
| **D-019** | 2026-09-30 | **v1.0 vocabulary = 100 ASL signs + `ase:_BACKGROUND`**; the PRD §9 targets apply | owner choice | Q-11 |
| **D-020** | 2026-09-30 | **Refine the existing project first (v0.2), park the v1 plan.** Reorganised into the layered `signlang` package with one CLI; fixed the review bugs; streaming live mode; 100 tests; new docs + portfolio README. `build_docs/` stays as the plan for the next version | owner: portfolio project; wants a clean, well-organised, tested codebase now | root `CHANGELOG.md` |

## 5. Gotchas and lessons learned

- **Data safety first:** v0 lost everything because data lived only on one laptop. Nothing enters `SIGNX_DATA_ROOT` without the backup system in place (T-0.4 before T-1.4).
- **`mp.solutions` is gone in MediaPipe ≥ 0.10.31.** The v0 code only runs in the old env. v1 uses Tasks. Python HolisticLandmarker was removed in 0.10.32 and re-added in 0.10.33; pin exact versions and model-file hashes.
- **Tasks HandLandmarker assumes a mirrored (selfie) image** for handedness. For raw video, swap the labels to get anatomical slots.
- **Two OpenCV wheels** are in the global env. Install only `opencv-contrib-python` in the v1 venv.
- **Legacy ONNX exporter** is deprecated in torch ≥ 2.9 (works in 2.11 with warnings). The dynamo exporter needs `onnxscript` and has GRU issues, which is another reason to consider non-RNN encoders.
- **ORT int8 doesn't quantize GRU** (v0: 6% smaller). Don't ship quantized models without measured benefit.
- **Licences:** ASL Citizen is non-commercial research only; WLASL is C-UDA (academic). Check `commercial_ok` before choosing sources (Q-12).
- **Disk:** C: has 31 GB free. Keep data on E:, and process public video datasets in chunks.
- **RAM 7.8 GB:** cache features on disk; don't load whole datasets into memory (v0 did, which was fine for 134 clips).
- **Windows:** repo path has spaces; cp1252 console; multiprocessing uses spawn (guard `__main__`).
- The v0 MediaPipe timing (~40 ms/frame) was measured on a **no-person noise video**. Re-measure on real clips in T-1.3.

## 6. Open questions for the owner

| ID | Question | Why it matters | Blocks | Status |
|----|----------|----------------|--------|--------|
| Q-1 | ASL or ISL? | datasets, vocab | — | **Answered:** v0 was ASL with some ISL mixed; v1 is language-agnostic (D-014/015) |
| Q-2 | Where's the v0 backup? | restore | — | **Answered:** lost (D-012) |
| Q-3 | Harden 12 or expand? | scope | — | **Superseded** by D-013 |
| Q-4 | One signer for v0? | backfill | — | **Moot** (data lost) |
| Q-5 | Who can be recruited as signers (≥ 5; ≥ 2 Deaf/fluent; ≥ 1 left-handed), on what timeline? | G2, RW tests | T-2.8, T-6.2 | Open |
| Q-7 | Is a desktop Python app enough for v1.0, or packaged/web? | P6 scope | T-6.7 | Open |
| Q-8 | OK to add `pytest` (dev) and possibly `ruff`? | tests | T-0.2 | Open (pytest assumed yes) |
| Q-9 | What is the "signx experiment 2 folder" from commit `9c5c8ba`? Is there work elsewhere? | might hold useful code | — | Open |
| Q-10 | Which sign language is L1? | everything data-related | T-2.6 | **Answered: ASL (`ase`)** (D-017) |
| Q-11 | v1 vocabulary size and targets | effort, data needs | T-2.6, P4 exit | **Answered: 100 signs + background; PRD §9 targets** (D-019) |
| Q-12 | Commercial or non-commercial/research use? | allowed datasets | T-2.6, T-2.7 | **Answered: research / non-commercial** (D-018) |
| **Q-13** | **Exact data root and second backup location.** The owner chose "Other location" (not E: + external/cloud) but hasn't said where yet | data safety | **T-0.4** | **Open: need the paths** |
| Q-16 | Access to Deaf/fluent **ASL** signers for vocabulary review, own recordings (background + live tests), and RW testing (possibly remote)? | G2, T-4.5, T-6.2 | T-2.8, T-4.5, T-6.2 | Open |
| Q-14 | OK to download CUDA torch (~2–3 GB) for GPU training on the RTX 3050? | training speed | T-3.10 | Open |

**Decision support for Q-10 / Q-12** (kept for the record; **ASL + research/non-commercial chosen**, D-017/D-018):

| Option | Strength | Watch out |
|--------|----------|-----------|
| **ASL** with ASL Citizen (+ WLASL / MS-ASL) | Largest public data; Deaf signers; **official signer-independent split**; published baselines to beat | Non-commercial research only (ASL Citizen) and academic-only C-UDA (WLASL). Fine for research; not for a commercial product |
| **ISL** with INCLUDE + own recordings | Useful if the owner has access to ISL signers (local impact) | Much less public data (263 signs, 4,287 videos); licence to check; relies more on own recordings |
| **Own recordings only** (any language) | Full control; commercial use possible under your consent terms | Needs many signers and sessions: slowest path to a strong model |

## 7. Session log (append-only; newest first)

### 2026-09-30 (d): v0.2 refactor (portfolio clean-up)
- **Owner said:** don't start v1; refine the existing codebase: organise files and folders, fix bugs, errors and conflicts, rewrite robust logic with tests, keep a how-to-run README in `docs/`, and write a professional portfolio README.
- **Did:** moved everything (with `git mv`) into `signlang/{landmarks,dataset,training,inference}` + `python -m signlang` CLI; made paths configurable; pinned requirements (one OpenCV wheel); made smoothing/hand logic pure NumPy; unified drawing/QA; fixed the bugs listed in `CHANGELOG.md` (label renumbering, manifest truncation, lost duplicate-label hands, prune default, hard-coded report claims, verifier and empty-clip crashes, live freeze, wrong abstain reason, non-reproducible file runs, hands-gate bypass, JSON not saved, Transformer positional waste, unbalanced folds, observe crash); built streaming `LiveRecognizer` + session logs; wrote 100 tests + a CLI smoke test; wrote new docs + diagrams; archived the original docs; wrote README + CHANGELOG; updated CLAUDE.md.
- **Measured:** tests 100/100; CLI smoke 11/11 commands; live classification ~2 ms after a sign; streaming extraction ~25 ms/frame (synthetic 720p→480).
- **Branch:** `refactor/clean-architecture` (nothing committed).

### 2026-09-30 (c): owner decisions
- **Owner answered:** L1 = **ASL**; use = **research / non-commercial**; vocabulary = **100 signs + background**; backups = "Other location" (paths not given yet → Q-13 still open).
- **Did:** recorded D-017…D-019; updated PRD, TRD §4.1, plan, tasks (T-2.6/T-2.7 now target ASL Citizen), and CLAUDE.md.
- **Planning note:** ASL Citizen averages ~30 videos per sign (83,399 / 2,731), so a 100-sign subset is roughly 3,000 videos from 52 signers (actual per-sign counts vary; confirm in T-2.7). The download size and packaging of the full archive still need checking before download (disk budget, R-5.6).
- **Next:** owner gives the Q-13 paths (and Q-5/Q-16 when possible). Agent starts **T-0.1 → T-0.2 → T-0.3**.

### 2026-09-30 (b): direction change → language-agnostic v1 (Claude Code session)
- **Owner said:** v0 data is permanently lost; don't retrain the 12-sign model; use v0 only as a reference to build a strong pipeline, architecture, requirements, tests, training, and serving; v0 mixed ASL with some ISL; v1 must work for any sign language, start with one, and later co-train languages from several countries.
- **Did:** researched multilingual SLR (shared encoder + language-specific heads; Logos multi-dataset co-training; collaborative multilingual CSLR), per-country datasets and licences (ASL Citizen non-commercial; WLASL C-UDA; MS-ASL; INCLUDE; AUTSL; LSA64; Logos), ISO 639-3 conventions (incl. the `qaa–qtz` local-use range), and the Tasks HandLandmarker handedness convention. Measured disk (C: 31 GB, E: 100 GB) and GPU (RTX 3050 4 GB). Rewrote `build_docs/` 01–08 for v1 and added `CLAUDE.md`.
- **Next:** owner answers **Q-10, Q-11, Q-12, Q-13** (and ideally Q-5, Q-14). Agent can start **T-0.1 → T-0.2 → T-0.3** right away (no decisions needed).
- Nothing committed.

### 2026-09-30 (a): build_docs created
- Explored the v0 codebase; ran v0 self-tests and a synthetic end-to-end check (ALL PASS); measured latencies; researched MediaPipe status, ISLR approaches, datasets, ethics; wrote the first version of `build_docs/`.
