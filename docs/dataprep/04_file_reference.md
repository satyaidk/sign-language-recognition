# 04 — File Reference

Every Python file, what it contains, and the logic of its key pieces.

---

## `landmark_detector.py` — core library + live viewer (~637 lines)

The original, reused everywhere. Top-level constants define MediaPipe namespaces,
pose landmark indices, the curated upper-body connections, tunables
(`VIS_THRESH`, `HOLD_FRAMES`, One-Euro params) and drawing colours.

| Symbol | Role |
|--------|------|
| `OneEuroFilter` | Adaptive low-pass filter for a single scalar (`__call__(x, t)`). |
| `PointStabilizer` | Smooths a fixed-size landmark set; optional occlusion hold; tracks visibility. Returns a `NormalizedLandmarkList`. |
| `HandStabilizer` | One `PointStabilizer` per hand, keyed by handedness label. |
| `anatomical_label(raw, mirrored)` | Map MediaPipe handedness to the person's true side. |
| `match_hands_to_sides(wrists, lw, rw)` | Optimal hand→body-side assignment (no arm crossing). |
| `draw_upper_body(frame, proto, stale, wrists, show_labels=True)` | Torso + arm chains. `show_labels` added for clean dataset renders. |
| `draw_face` / `draw_hands` | Face mesh and hand skeletons via `mp_drawing`. |
| `run(...)` | The live viewer loop (webcam / video / image). Guarded by `__main__`. |

> This module **does not save landmark data** — it only detects and draws.
> The extraction layer was built on top of it for that reason.

---

## `extract_dataset.py` — extraction pipeline (~621 lines)

| Symbol | Role |
|--------|------|
| `build_models(opts)` | Create the enabled MediaPipe models for one clip. |
| `_raw_hands(hand_res)` | Hands without smoothing (used when `--no-smooth`). |
| `extract_clip(video_path, opts)` | Core: run detection over a clip → stacked arrays + masks + meta. |
| `build_all_vector(arrays, opts)` | Concatenate enabled modalities into `(T, F)`; return the layout. |
| `_proto(rows, with_vis)` | Rebuild a `NormalizedLandmarkList` from an array (for rendering). |
| `render_skeleton(arrays, masks, size, fps, out, opts)` | Black-bg skeleton video from saved arrays. |
| `qa_clip(arrays, masks, T, opts)` | Detection rates, ranges, NaN/Inf, `PASS/WARN/FAIL` verdict. |
| `discover(dir, only)` | Find `<class>/<video>` clips. |
| `main()` | Orchestrate: folders, loop, save, aggregate reports + README. |

**CLI flags:** `--dataset --out --classes --limit --no-face --no-hands --no-pose
--no-skeleton --no-smooth --proc-width --skip-existing --skeleton-size`.

---

## `verify_landmarks.py` — verification + overlay (~313 lines)

| Symbol | Role |
|--------|------|
| `_proto(rows, with_vis)` | Rebuild a proto from an array. |
| `_present(block)` | Per-frame boolean: a block is "detected" if not all-zero. |
| `load_clip(out, cls, stem)` | Load whichever modality arrays exist. |
| `rebuild_all(arr, layout)` | Reconstruct the combined vector from the separate files. |
| `verify_clip(...)` | The 3 checks (structure, consistency, detection) + verdict. |
| `make_overlay(out, dataset, cls, stem)` | Side-by-side original+landmarks / skeleton video. |
| `main()` | Single-clip mode (`--video`, optional `--overlay`) or whole-dataset scan. |

**Consistency check** is the strongest structural guarantee: it rebuilds
`landmarks_all` from `landmarks_{pose,face,hands}` and asserts an exact match
(`max diff` reported; observed `0.0`).

---

## `prune_clips.py` — reversible curation (~115 lines)

| Symbol | Role |
|--------|------|
| `prune(pairs, out, dataset, excluded)` | Move source mp4 → `data/excluded_clips/`, delete processed artifacts. |
| `refresh_reports(out, dropped)` | Drop rows from `manifest.csv`; recompute `extraction_report.json`. |
| `main()` | CLI: `python src/dataprep/prune_clips.py <class>/<stem> ...` (default: the 2 close-up clips). |

Nothing is deleted irreversibly — sources are preserved in `data/excluded_clips/`.

---

## `dataset_report.py` — infographics (~295 lines)

| Symbol | Role |
|--------|------|
| `load(out)` | Read all per-clip `metadata/*.json`. |
| `fig_overview` | Clips/class, verdict donut, mean detection, key numbers. |
| `fig_heatmap` | Class × modality detection-rate heatmap. |
| `fig_per_class` | Grouped detection bars per class. |
| `fig_frames` | Frames-per-clip histogram. |
| `fig_layout` | Composition of the `(T, 1692)` vector. |
| `fig_scorecard` | One-page verification scorecard (reads `verification_report.json`). |
| `main()` | Render all six PNGs into `data/processed_dataset/reports/`. |

---

## `docs/dataprep/make_diagrams.py` & `docs/dataprep/build_pdf.py`

Documentation tooling: `make_diagrams.py` renders the architecture / data-flow /
verification diagrams; `build_pdf.py` parses these markdown files (plus the
diagrams and infographics) into `TECHNICAL_DOCUMENTATION.pdf`.
