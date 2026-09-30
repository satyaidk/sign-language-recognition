# 04 — Rules

These rules bind every contributor, human or AI agent. They exist because the
project mixes **ML** (where silent mistakes look like success), **multiple sign
languages** (where label mix-ups are easy), **personal video data**, and fragile
dependencies. v0 lost all its data and mixed two sign languages in one label set.
These rules are how v1 avoids repeating that. When a rule conflicts with a
request, stop and ask the project owner (§11).

---

## 0. Prime directives

- **R-0.1 Honest metrics.** Only report numbers you measured, or that you can trace to a committed report. Always state the evaluation type: `signer-independent test`, `signer-independent val`, `grouped CV`, `within-signer (diagnostic)`, `live session`. Never report training-fit accuracy as quality.
- **R-0.2 Language is data.** Never hard-code a language code, gloss, class name, or class count in `signx/` code. They come from `registry/` and configs (a test enforces this).
- **R-0.3 Protect the data.** Never delete, move, or overwrite anything under `SIGNX_DATA_ROOT` without a **verified** backup (`signx backup verify` passes and a second copy exists).
- **R-0.4 One code path.** One landmark schema, one detection backend configuration per dataset version, one feature transform: shared by dataset building, training, and serving. No "quick" second copies in scripts, notebooks, or tests.
- **R-0.5 Don't break contracts** (03_ARCHITECTURE §A4) without an ADR, a version bump, and owner approval.
- **R-0.6 Scope honesty.** SignX recognises isolated signs. Never call it a translator.
- **R-0.7 v0 is reference only.** Read and port from `reference/v0/`; never import from it inside `signx/`, and never retrain the v0 model or reuse its 12-class label set.

## 1. Session protocol

**Start**
1. Read `08_MEMORY.md` → `06_TASKS.md` → this file.
2. Pick **one** task whose dependencies are `DONE` (or resume an `IN-PROGRESS` one). Mark it `IN-PROGRESS (YYYY-MM-DD)`.
3. Read the PRD/TRD/ARCHITECTURE sections the task references.
4. Check the environment you need: venv active, `SIGNX_DATA_ROOT` set, the data/models the task needs exist (`python -m signx doctor` once T-0.5 is done).

**End**
1. Run the relevant tests (07_TEST §2); append the results (date, command, pass/fail, key numbers) to 07_TEST §4.
2. Update the task in `06_TASKS.md`: status, **evidence** (test IDs, numbers, paths), and any follow-up tasks you discovered.
3. Append a session-log entry to `08_MEMORY.md`, plus any new decision (D-n) or gotcha. Refresh the snapshot if the state changed.
4. If you stop mid-task, write down exactly where you stopped and the next command to run.

## 2. Contracts, languages, and data invariants

- **R-2.1** Landmarks follow schema `mp-1692-v1`: pose[0:132], face[132:1566], left_hand[1566:1629], right_hand[1629:1692]; absent = zeros; `mask` (T,4); hands are **anatomical**.
- **R-2.2** Every sample has `language` (ISO 639-3) and `sign_id = "<lang>:<GLOSS>"`. **A sign_id belongs to exactly one language.** The same meaning in two languages is two classes, linked only via the optional `concept_en` field.
- **R-2.3** Never merge data from different sign languages into one head. Never relabel a sample into another language.
- **R-2.4** Label maps are **frozen per model**. New signs create a new label-map version (append only). Never re-number within a version.
- **R-2.5** A dataset version is **immutable** and contains exactly **one `extractor_id`**. Changing the backend, MediaPipe version, model files, smoothing, or processing resolution means re-extraction and a new dataset version.
- **R-2.6** **Signer-independent splits:** a `signer_id` appears in exactly one of train/val/test. Use a dataset's official split when it has one. Splits never change inside a dataset version.
- **R-2.7** Inference rebuilds the feature config and label maps **from the model package meta**, never from defaults.
- **R-2.8** Absent-modality points stay exactly zero through normalisation and augmentation.
- **R-2.9** Verify a language code against the [SIL ISO 639-3 tables](https://iso639-3.sil.org/code_tables/639/data) before adding it to `registry/languages.toml`.
- **R-2.10** Every gloss mapping from a public dataset (`gloss_map/<source>.csv`) is reviewed. Unclear glosses map to `DROP` rather than being guessed. Prefer review by a fluent/Deaf signer.

## 3. Code style (match v0's proven style)

- **R-3.1** Every module starts with a docstring explaining **what it does and why** (see `reference/v0/training/features.py`, `segmenter.py`).
- **R-3.2** `from __future__ import annotations`; type hints on public functions; `pathlib.Path`; `@dataclass` for configs and records.
- **R-3.3** Section dividers in the house style: `# ── Section name ─────`.
- **R-3.4** v1 is a package: absolute imports (`from signx.features.transform import to_features`). **No `sys.path` hacks** in `signx/`.
- **R-3.5** CLI commands live in `signx/__main__.py` (argparse sub-commands), each calling a plain function that is testable without the CLI.
- **R-3.6** No speculative scaffolding: create modules from 03_ARCHITECTURE §A3 only when a task needs them.
- **R-3.7** Comment the *why*. Keep comment density like the surrounding code.
- **R-3.8** Pure logic (transforms, state machines, gates, split logic, label maps) is separated from I/O so it can be unit-tested with synthetic inputs.
- **R-3.9** Fail loudly and actionably: `[ERROR] <what> — run <command> first`. Don't catch broad exceptions except to isolate one bad video or frame in bulk extraction (log it and continue).
- **R-3.10** Paths come from `signx/paths.py` (`SIGNX_DATA_ROOT`). Never hard-code drive letters or user directories.

## 4. Dependencies and environment

- **R-4.1** Runtime dependencies are pinned exactly in `requirements.txt`; dev tools go in `requirements-dev.txt`. A new runtime dependency needs a reason, a D-n entry, and owner OK if it's heavy (> 50 MB or native).
- **R-4.2** MediaPipe is pinned to the version validated in T-1.3. Upgrading it means a new `extractor_id` → re-extraction → a new dataset version. Do it only as a planned task.
- **R-4.3** After any torch or onnxruntime change, re-run the export parity tests and record the result.
- **R-4.4** Install only **one** OpenCV wheel in the v1 venv.
- **R-4.5** The v1 venv is separate from the v0 environment. Don't install v1 dependencies into the global Python that v0 uses.

## 5. Data, privacy, licensing, ethics

- **R-5.1** Never commit videos, `.npy/.npz`, weights (`.pt/.pth/.ckpt`), `.onnx`, or feature caches. Only registries, configs, manifests, reports (JSON/CSV/PNG), and the backup manifest go in git.
- **R-5.2** Before downloading a public dataset: add it to `registry/sources.toml` with licence name, URL, `commercial_ok`, and citation. If the licence is unclear, stop and ask.
- **R-5.3** Never upload videos of people to external services (including AI tools, trackers, chats) without the owner's decision and the contributors' consent.
- **R-5.4** Own recordings require a consent record (`consent_id` in the manifest) **before** the clips enter a dataset. Deletion requests: remove the raw video, npz, and manifest rows → new dataset version → retrain the affected models.
- **R-5.5** Backups: after every data-changing session, run `signx backup write` + `verify`, and sync the second copy. Record where the copies are in 08_MEMORY.
- **R-5.6** Watch the disk budget: process large sources in chunks. Never let C: drop below 10 GB free during extraction.
- **R-5.7** Live mode stores no video unless explicitly asked (`--save-video`), and that is for debugging only.

## 6. ML and evaluation integrity

- **R-6.1** No leakage: hyperparameters, thresholds, calibration, and model selection use **val signers** only. The **test split is touched once per candidate model**, for the final report.
- **R-6.2** Report top-1, top-5, macro-F1, per-class, and per-signer metrics. Name the evaluation type (R-0.1).
- **R-6.3** Compare experiments under the same dataset version, split, and seeds. Treat a difference smaller than the seed standard deviation as noise; use ≥ 3 seeds for claims.
- **R-6.4** Every training run appends to the experiment registry (git SHA, dirty flag, config hash, dataset hash, metrics).
- **R-6.5** A new deployable model gets a new `model_version` (semver). Keep the previous package until the new one passes the ST and RW tests.
- **R-6.6** Changing a live threshold (`conf`, `margin`, `still`, …) is a behaviour change: record a D-n with the evidence.
- **R-6.7** Mixed-language training (future) reports **per language**. An aggregate across languages alone is not acceptable.

## 7. Testing

- **R-7.1** The fast suite (`python -m pytest -q`, 07_TEST §2.1) passes before any task is `DONE`.
- **R-7.2** New pure logic gets unit tests with synthetic inputs that run without private data.
- **R-7.3** Tests that need data or models are marked (`data`, `model`, `slow`) and skip cleanly when the resources are absent.
- **R-7.4** A bug fix comes with a regression test that fails before the fix and passes after.
- **R-7.5** Performance claims (latency, fps, memory, throughput) are recorded in 07_TEST §4 with machine, input, and conditions.

## 8. Git workflow

- **R-8.1** One branch per task: `task/T-<p>.<n>-short-name`. Commit or push only when the owner asks.
- **R-8.2** Meaningful messages: `T-1.1: add mp-1692-v1 schema and to_row`. Don't write messages like "changes" or "signx".
- **R-8.3** Never force-push `main`; never rewrite published history.
- **R-8.4** Moves and renames use `git mv` (history preserved), in their own commit.

## 9. Documentation

- **R-9.1** Update `06_TASKS.md`, `07_TEST.md`, and `08_MEMORY.md` every session.
- **R-9.2** New CLI commands and config keys are documented in their docstrings, and in the user guide once it exists (T-6.6).
- **R-9.3** Keep `CLAUDE.md` accurate when commands, paths, or the environment change.

## 10. Windows specifics

- **R-10.1** The repo path contains spaces: quote paths in shell commands; use `pathlib` in code.
- **R-10.2** The console is cp1252: reconfigure stdout/stderr to UTF-8 at CLI start-up (v0's `utils._enable_utf8_console` pattern).
- **R-10.3** `multiprocessing` on Windows uses spawn: guard entry points with `if __name__ == "__main__":` and keep worker functions importable at module top level.
- **R-10.4** OpenCV windows need `cv2.waitKey`; headless runs use `--no-display`.

## 11. Stop and ask the owner before…

- deleting or moving data, or anything outside the repo;
- downloading a dataset, or using one whose licence is unclear or non-commercial when the intended use is commercial;
- recording or importing data of people;
- changing a contract (§2, ARCHITECTURE §A4), a label map of a released model, or the language list;
- adding heavy dependencies, or upgrading MediaPipe/torch/onnxruntime;
- committing, pushing, opening PRs, or publishing artifacts;
- when a task's acceptance criteria can't be met as written.

## 12. Definition of Done (any task)

1. The acceptance criteria in `06_TASKS.md` are met **with evidence** (test IDs, numbers, paths).
2. The fast suite passes; data/model tests pass if the task touched those layers and the resources exist.
3. Contracts are intact, or versioned under an ADR.
4. Docs are updated (§9); there's a session entry in 08_MEMORY; backups are updated if data changed.
5. No stray artifacts: temp files, debug prints, commented-out code, duplicates.
