# 01 — Product Requirements Document (PRD)

| | |
|---|---|
| **Product** | **SignX v1**: a language-agnostic sign-language recognition platform (data → landmarks → training → live recognition) |
| **Scope of v1.0** | One sign language: **L1 = American Sign Language (`ase`)**. **100 isolated signs + a background class**, local CPU inference. **Research / non-commercial use** (D-017…D-019) |
| **Future scope** | Multiple sign languages from different countries, co-trained in one model (§10) |
| **Predecessor** | v0 prototype (12 mixed ASL/ISL signs, one signer). Data and weights lost; **code kept as a reference only** |
| **Last updated** | 2026-09-30 |

---

## 1. Summary

SignX is a **pipeline and runtime for training and running isolated-sign
recognisers for any sign language**. Given labelled videos of signs (public
datasets and/or our own recordings), it:

1. extracts body, hand, and face **landmarks** with MediaPipe in one versioned format;
2. organises them in a **language-aware dataset** (every sample tagged with its sign language, signer, and source);
3. **trains** small, fast sequence models with honest, signer-independent evaluation;
4. **exports** them to ONNX and runs them **live on a laptop webcam**, captioning recognised signs and staying quiet when unsure or idle.

v1.0 proves the whole platform on **one sign language**. The design lets a second,
third, … sign language be added later **without rewriting anything**: new data
adapter, new vocabulary, new classification head on a shared encoder.

## 2. Background: what v0 taught us

The v0 prototype (June–July 2026) proved the approach end to end:
- a landmark pipeline with QA;
- a shared train/serve feature transform;
- a small BiGRU reaching 91% within-signer accuracy on 12 signs;
- ONNX export at ~1 ms per classification;
- a motion-gated live mode with abstention and de-duplication.

It also taught us what **not** to do again:

| v0 problem | v1 requirement |
|------------|----------------|
| ASL and ISL signs mixed in one 12-class label set → confusions | Every sample and label carries its **sign language code**; one label namespace per language (FR-2) |
| All data from one signer → the 91% says nothing about new people | **Signer-independent evaluation** is the headline metric (FR-12) |
| Data and weights stored only on one laptop, now lost | **Data root outside the repo + checksummed backup manifest + a second copy** (FR-6) |
| Legacy MediaPipe API, now removed upstream | **Detection backend interface**, MediaPipe Tasks (FR-3) |
| Live mode froze while classifying (~40 ms/frame of re-extraction after each sign) | **Streaming, threaded live pipeline** (FR-17) |
| Idle or non-sign motion forced into a sign | **Background (non-sign) class** + abstention (FR-19) |

## 3. Honest framing

- SignX recognises **isolated signs from a finite vocabulary**. That is not sign-language *translation*: sign languages have their own grammar, non-manual markers, classifiers, and spatial reference. SignX must never be described as a translator or as a replacement for interpreters.
- Deaf researchers have documented systemic problems in sign-language AI: hearing-led agendas, unrepresentative data, and a "fixing communication" framing ([Desai et al., 2024](https://aclanthology.org/2024.signlang-1.6/)). SignX commits to: honest scope; **Deaf and fluent signers** as contributors and testers; vocabulary chosen with signers of the target language; respecting dataset consent and licence terms.

## 4. Goals and non-goals

### Goals for v1.0 (single language)

| ID | Goal |
|----|------|
| **G1** | **Language-agnostic platform**: nothing in the code is specific to one sign language, vocabulary, or dataset. Adding a language means data + config, not code changes. |
| **G2** | **A strong ASL recogniser** that generalises to **people it has never seen** (signer-independent), on **100 signs + background**. |
| **G3** | **Reproducible, safe data**: every dataset version is rebuildable from recorded sources, checksummed, and backed up in two places. |
| **G4** | **Live recognition that feels immediate and honest**: no UI freeze, captions shortly after each sign, "?" when unsure, nothing when idle. |
| **G5** | **Measured quality**: every accuracy or latency claim is backed by a recorded test (07_TEST). |
| **G6** | **Multilingual-ready by design**: data schema, label registry, model heads, and evaluation already support N languages, even though v1.0 trains only one. |

### Non-goals for v1.0

- Training more than one sign language (that's the next major version, §10). v1.0 only has to *support* it structurally.
- Continuous or sentence-level recognition, fingerspelling, grammar, and translation to spoken or written language.
- Mobile or web apps, cloud services, accounts.
- Retraining or recovering the v0 12-sign model.

## 5. Users and personas

| Persona | Needs |
|---------|-------|
| **P1 Signer** (Deaf/HoH signer or learner of L1) | Sign naturally in front of a webcam, left- or right-handed; get a correct caption or an honest "?" |
| **P2 Viewer** (partner, teacher, demo audience) | Readable captions and transcript; can always tell listening / recording / result apart |
| **P3 ML developer** | Add datasets and languages via adapters; one command per stage; configs; honest metrics; reproducible runs |
| **P4 Data contributor** | Clear consent; a simple recording tool; their data stored privately and removable on request |
| **P5 Language expert** (fluent/Deaf signer of L1) | Review the vocabulary, gloss names, look-alike groups, and error reports |

## 6. User stories

1. *As a developer (P3)*, I point an adapter at a public dataset or a folder of my own recordings and get a validated, language-tagged landmark dataset with signer-independent splits.
2. *As a developer*, I add a second sign language by writing one adapter and one vocabulary file. No model or pipeline code changes.
3. *As a developer*, I launch training from a config file and get a report with signer-independent top-1/top-5 accuracy, macro-F1, per-signer and per-class metrics, calibration, and a registry entry.
4. *As a signer (P1)*, I select my sign language, sign a sequence with short pauses, and see correct captions within a fraction of a second after each sign.
5. *As a signer*, when I rest, talk, or scratch my head, no caption appears.
6. *As a signer*, when my sign is ambiguous, I see "?" rather than a wrong word.
7. *As a contributor (P4)*, a guided tool prompts each sign, records my repetitions, and files them with my pseudonymous signer ID and consent reference.
8. *As a language expert (P5)*, I can review which signs are confused with which, and mark visually-similar sign groups.

## 7. Functional requirements

Priority: **M** must (v1.0) · **S** should · **C** could. "v0 ref" means v0 has a working reference implementation to port.

### Data and languages
| ID | Requirement | Pri | Status |
|----|-------------|-----|--------|
| FR-1 | **Language registry**: every supported sign language is registered with its ISO 639-3 code, name, and region; the registry drives everything else | M | ❌ |
| FR-2 | **Label namespace**: every class is `"<lang>:<GLOSS>"` (e.g. `ase:HELLO`). The same meaning in two languages is two different classes. An optional `concept` field (e.g. English "hello") links equivalents across languages for analysis only | M | ❌ |
| FR-3 | **Landmark extraction** through a detection-backend interface (MediaPipe Tasks) into one **versioned landmark schema**; per-frame presence masks; One-Euro smoothing; anatomical handedness | M | v0 ref (legacy API) |
| FR-4 | **Dataset adapters**: generic "folder of videos per gloss" adapter, plus one adapter per public dataset used; each outputs the canonical manifest | M | ❌ |
| FR-5 | **Canonical manifest** per dataset version: sample_id, language, sign_id, gloss, signer_id, source, split, fps, frames, extractor version, QA verdict, checksums | M | ❌ (v0 had a minimal one) |
| FR-6 | **Data safety**: data root outside the repo (`SIGNX_DATA_ROOT`); checksummed backup manifest in git; verified second copy; restore test | M | ❌ (v0 data was lost) |
| FR-7 | **Signer-independent splits** (official splits where the dataset has them; otherwise grouped by signer), stored in the manifest and never changed silently | M | ❌ |
| FR-8 | Dataset QA: per-sample verdicts, detection rates, outlier report, reversible pruning | M | v0 ref |
| FR-9 | **Recording tool** for own data: prompts, countdown, N repetitions, auto-naming, signer and consent IDs | M | ❌ |
| FR-10 | **Background (non-sign) class** collected per language by protocol | M | ❌ |

### Training and evaluation
| ID | Requirement | Pri | Status |
|----|-------------|-----|--------|
| FR-11 | Config-driven training (one config file per experiment); encoder choice; **per-language classification heads** (one head used in v1.0) | M | v0 ref (single head) |
| FR-12 | Evaluation report: **signer-independent** top-1, top-5, macro-F1, per-class and per-signer metrics, confusion analysis, calibration (ECE) | M | partial v0 ref |
| FR-13 | **Experiment registry**: every run logged with git SHA, config, dataset version hash, metrics | M | ❌ |
| FR-14 | Self-describing model package: ONNX + meta (languages, label lists per head, feature config, landmark schema version, calibration, metrics, model_version) | M | v0 ref |
| FR-15 | Probability calibration (temperature) per head | S | ❌ |
| FR-16 | Visually-similar-sign groups (from confusion analysis + expert review) usable as an evaluation view and optional grouped labels | C | ❌ |

### Inference and live
| ID | Requirement | Pri | Status |
|----|-------------|-----|--------|
| FR-17 | **Streaming live pipeline**: per-frame landmarks while recording; capture/UI never blocks on inference | M | ❌ (v0 blocks) |
| FR-18 | Automatic sign start/end detection (motion gating; hand-velocity option) | M | v0 ref |
| FR-19 | Emit gates: background class, confidence, top-1/top-2 margin, hands present; "?" on abstain | M | partial v0 ref |
| FR-20 | Repeat de-duplication | M | v0 ref |
| FR-21 | Language selection at start-up (`--lang ase`); only that language's head is used | M | ❌ |
| FR-22 | On-screen state (LISTENING / REC / result) + transcript; timestamped session log with per-sign diagnostics and latency | M | partial v0 ref |
| FR-23 | File mode: video → predictions + timeline + overlay | M | v0 ref |
| FR-24 | Privacy by default: no video kept from live mode; no network calls | M | v0 ref |
| FR-25 | Single CLI entry point `python -m signx <command>` | S | ❌ |
| FR-26 | Automatic language identification (no `--lang`) | C (future) | ❌ |
| FR-27 | Windows packaged build | C | ❌ |

## 8. Non-functional requirements

| ID | Requirement | Target (proposed) |
|----|-------------|-------------------|
| NFR-1 | Inference on a mid-range laptop **CPU**, offline | Reference: i5-11320H, 8 GB RAM, Windows 11 |
| NFR-2 | Caption latency, end of sign motion → caption | p95 ≤ 700 ms (including the 0.5 s stillness confirmation) |
| NFR-3 | Preview fps while classifying | ≥ 20 fps, never frozen |
| NFR-4 | Model size / speed | ≤ 10 M params; ≤ 10 ms per classification on CPU (ONNX) |
| NFR-5 | Training hardware | Runs on the reference laptop (CPU; optional CUDA on the RTX 3050 4 GB); no cloud required |
| NFR-6 | Storage | Landmark datasets stored compactly (float16 allowed); raw videos may live on an external/secondary drive |
| NFR-7 | Reproducibility | A dataset version + config + git SHA reproduces metrics within the documented seed variance |
| NFR-8 | Stability | 30-min live session, no crash, memory growth < 200 MB |
| NFR-9 | Privacy / licensing | Consent for every own-recorded signer; licences recorded and respected per dataset |

## 9. Success metrics and release criteria

### Metrics (vocabulary and use confirmed 2026-09-30: 100 ASL signs + background, research/non-commercial)

| Metric | Test | v1.0 target |
|--------|------|-------------|
| Signer-independent top-1 / macro-F1 on the 100-sign ASL vocabulary | DT-4 | **≥ 0.85 top-1** on ASL Citizen's official **test signers**, restricted to the 100 chosen signs |
| Signer-independent top-5 | DT-4 | ≥ 0.95 |
| Worst-class F1 | DT-4 | ≥ 0.60, with every class < 0.60 reviewed by a language expert |
| Live sign accuracy (scripted sessions, new signers) | RW-1 | ≥ 85% correct, ≤ 10% "?" |
| False emits during idle/non-sign behaviour | RW-2 | ≤ 1 per 60 s |
| Caption latency p95 | ST-4 | ≤ 700 ms |
| Left-handed gap | RW-3 | ≤ 10 points |
| Adding a second language (dry run with a small dataset) | ST-8 | no code changes outside a new adapter + configs |

### v1.0 release criteria

1. All **M** requirements in §7 are ✅, with evidence in 07_TEST.
2. The metric targets above are met on L1, measured on signers **not** in training.
3. The multilingual dry run (ST-8) passes: a second language's small dataset trains on a second head with no pipeline code changes.
4. Data safety: backup manifest verified; restore test passed on a second copy.
5. Real-world protocol run with ≥ 5 people (≥ 1 left-handed, ≥ 2 Deaf/fluent L1 signers).
6. A model card and dataset datasheet (sources, licences, signer demographics as consented, metrics, limitations, intended use/non-use).

## 10. Multilingual roadmap (after v1.0)

| Stage | What | Key technique |
|-------|------|---------------|
| v1.x | Add L2 (and L3…) datasets via adapters | Language-tagged manifests; per-language vocabularies |
| v2.0 | **Co-train** languages: a shared landmark encoder + **one classification head per language**; mixed-language batches routed to their heads | Shown to help low-resource languages in multi-dataset co-training ([Logos, 2025](https://arxiv.org/abs/2505.10481)); shared-encoder + language-specific modules in multilingual CSLR ([Collaborative Multilingual CSLR](https://ieeexplore.ieee.org/iel7/6046/4456689/09954921.pdf)) |
| v2.x | Language selection → optional **language-ID** head (FR-26); cross-language concept links for analysis | |
| later | Continuous signing (sign spotting with a background class + de-duplication, [arXiv 2401.05336](https://arxiv.org/abs/2401.05336)) | |

## 11. Assumptions and constraints

- **A1** Signer roughly faces the camera, upper body and hands in frame, adequate light.
- **A2** Signs in live mode are separated by short pauses (isolated recognition).
- **C1** Laptop-class hardware (CPU inference; 4 GB GPU for optional training); disk: C: 31 GB free, E: 100 GB free (2026-09-30), so large public video datasets must be processed in chunks and/or stored off the system drive.
- **C2** Many public sign datasets are **non-commercial, research-only** (e.g. [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/); WLASL under C-UDA). **v1 is research / non-commercial (D-018)**, so these may be used. Every model states `commercial_ok = false`, and a future commercial version would need retraining on commercially licensed data.
- **C3** MediaPipe's API changes between versions (legacy removed; Holistic removed and re-added in Python), so the detection layer must be behind an interface with a pinned, tested version.

## 12. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Losing data again | Catastrophic | FR-6: data root + backup manifest + second copy + restore test, from P0 |
| Licence blocks intended use | Model unusable for the product | Decided research/non-commercial (D-018); licences recorded per source; `commercial_ok=false` in model meta; a commercial version would retrain on commercially licensed data |
| Too little signer diversity for L1 | G2 fails | Public signer-independent datasets + recruited Deaf/fluent signers |
| Label noise across sources (same gloss, different sign variants) | Accuracy ceiling | Per-source QA; language-expert review; visually-similar groups (FR-16) |
| Scope creep into multilingual before L1 works | Nothing ships | v1.0 = one language; multilingual only structurally (G6) |
| Over-claiming ("translator") | Harm, credibility | §3 framing in the UI, docs, and model card |
