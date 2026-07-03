# 04 — File Reference

Every Python file in `src/training/` and `src/inference/`, what it contains,
and the logic of its key pieces.

---

## `config.py` — single source of truth (~140 lines)

Pure data + path logic; imports nothing heavy, so it is safe to import anywhere
(including live inference). Paths resolve relative to the **project root** (two
levels up from `src/training/`), never the current working directory.

| Symbol | Role |
|--------|------|
| Path constants | `PROCESSED`, `LANDMARKS_ALL`, `MANIFEST`, `CLASSES_JSON`, and the `artifacts/` subdirs (`CKPT_DIR`, `METRICS_DIR`, `EXPORT_DIR`, `PREDICTIONS_DIR`, `LIVE_TMP_DIR`, `LOG_DIR`, `CACHE_DIR`). |
| `RAW_TOTAL`, `BLOCK` | The `(T, 1692)` raw layout (mirrors `data/processed_dataset/FEATURE_LAYOUT.json`). |
| `UPPER_POSE_IDX` | The 13 curated signing-relevant pose joints (nose, shoulders, elbows, wrists, hand roots). |
| `POSE_FLIP_PAIRS`, `LEFT/RIGHT_SHOULDER` | L/R symmetric joints (for flip) and the normalisation reference. |
| `FeatureConfig` | How a clip becomes `(L,D)`: `seq_len`, `use_pose/hands/face`, `add_velocity`, `add_masks`, `normalize`, etc. |
| `ModelConfig` | `arch` (`bigru`/`transformer`), `hidden`, `layers`, `dropout`, `pool`, transformer dims. |
| `TrainConfig` | Optimisation + all augmentation knobs (`epochs`, `lr`, `folds`, `aug_*`, early stop). |
| `Config`, `CONFIG` | Composite of the three; `CONFIG` is the shared default instance. |
| `ensure_dirs()` | Create all artifact dirs. |
| `to_dict()` / `feature_config_from_dict()` | Round-trip config to/from JSON (rebuilds tuple fields). |

---

## `features.py` — shared preprocessing (~294 lines)

Raw `(T, 1692)` → fixed `(L, D)`. No fitted state. Split into `geometric` →
`assemble` so augmentation can act in normalised space. **Used by training AND
inference.**

| Symbol | Role |
|--------|------|
| `FeatureSpec` | Precomputed layout: point segments, flip permutations (points/vis/masks), counts. `point_present()` expands per-modality presence to per-point. |
| `build_spec(cfg)` | Build the `FeatureSpec` (segments + flip permutations) for a config. |
| `split_blocks(all_vec)` | `(T,1692)` → dict of `(T,N,C)` arrays per modality. |
| `_present(block)` | Per-frame "detected" = block not all-zero. |
| `_reference(...)` | Clip-level origin (median mid-shoulder) + scale (median shoulder width). |
| `_resample_linear / _resample_nearest` | `(T,D)` → `(L,D)` time resampling. |
| `geometric(all_vec, cfg, spec)` | Stage 1: normalise + select → `coords, vis, present_modal`. |
| `assemble(coords, vis, present, cfg, spec)` | Stage 2: resample + velocity + masks → `(L,D)`. |
| `to_features(all_vec, cfg, spec)` | Inference wrapper (geometric→assemble, no augment). |
| `feature_dim(cfg, spec)` | Compute `D` for a config (no real data needed). |

Self-test (`__main__`) prints the spec, transforms a sample clip, and asserts the
flip is an involution.

---

## `data.py` — dataset + augmentation (~173 lines)

| Symbol | Role |
|--------|------|
| `load_class_names()` | Class names ordered by integer label (from `classes.json`). |
| `load_index(drop_fail)` | One dict per usable clip from `manifest.csv` (skips `FAIL`); falls back to a glob if no manifest. |
| `load_geometric(...)` | Load every clip, run `geometric()` **once**, cache `(coords, vis, present)` in memory. |
| `augment(coords, vis, present, ...)` | Training-only: temporal dropout/warp, horizontal flip (+L/R swap), rotation, scale, translation, jitter; re-zero absent points. |
| `SignDataset` | torch `Dataset`; indexes the cache, augments if enabled, runs `assemble()` → `(features, label)`. |

---

## `model.py` — sequence models (~110 lines)

Input `(B, L, D)` → logits `(B, C)`. Small + regularised; all ops ONNX-exportable.

| Symbol | Role |
|--------|------|
| `AttentionPool` | Additive attention pooling over time → `(B, dim)`. |
| `_Pool` | Pooling selector: `attention` / `mean` / `last`. |
| `SignClassifier` | input `LayerNorm` → MLP → **BiGRU** *or* **Transformer encoder** → pool → head. |
| `build_model(in_dim, n_classes, cfg)` | Construct a `SignClassifier`. |
| `count_params(model)` | Trainable parameter count. |

The input `LayerNorm` is what lets preprocessing stay fitted-state-free — it
absorbs any residual feature scaling.

---

## `utils.py` — shared helpers (~122 lines)

No scikit-learn dependency — metrics are NumPy.

| Symbol | Role |
|--------|------|
| `_enable_utf8_console()` | Make stdout/stderr UTF-8 tolerant on Windows (runs on import). |
| `seed_everything(seed)` | Seed `random` / NumPy / torch. |
| `stratified_folds(labels, k)` | k validation-index arrays, each class spread evenly. |
| `confusion_matrix` / `classification_report` | Accuracy + per-class P/R/F1 + confusion matrix. |
| `save_json` / `load_json` | JSON IO (creates parent dirs). |
| `Tee` | Print to stdout AND append to a log file. |

---

## `viz.py` — plots (~71 lines)

Headless matplotlib (`Agg`).

| Symbol | Role |
|--------|------|
| `plot_confusion(cm, names, path)` | Row-normalised confusion-matrix heatmap. |
| `plot_cv_folds(fold_accs, path)` | Per-fold accuracy bars + mean line. |
| `plot_history(histories, path)` | Train-loss + val-accuracy curves per fold. |

---

## `train.py` — STAGE 1: k-fold CV (~252 lines)

| Symbol | Role |
|--------|------|
| `get_device()` | CUDA if available, else CPU. |
| `make_scheduler(...)` | Linear warmup → cosine decay to `min_lr`. |
| `evaluate(model, loader, ...)` | Preds / targets / loss / accuracy (no grad). |
| `train_model(train_ds, val_ds, ...)` | Train one model; best-val-acc weights + history; supports a no-val "monitor" mode used by finetune. |
| `run_cv(cfg, args)` | Loop folds, gather OOF predictions, write report + plots + per-fold checkpoints. |
| `apply_overrides` / `parse_args` | CLI: `--epochs --folds --batch-size --lr --seq-len --arch --hidden --use-face --no-aug --seed`. |

---

## `finetune.py` — STAGE 2: deployment fit (~188 lines)

| Symbol | Role |
|--------|------|
| `EMA` | Exponential moving average of weights, with decay warmup for short runs. |
| `build_meta(...)` | Self-describing meta dict (classes, dims, config, timestamp). |
| `fit_final(cfg, args)` | Train on all data with EMA; ship the better of EMA/raw; write `model_final.pt` + `model_meta.json` (and fold in CV accuracy if present). |
| `parse_args` | Same overrides as train + `--init-from <fold>` warm-start. |

---

## `export_optimize.py` — STAGE 3: ONNX export (~144 lines)

| Symbol | Role |
|--------|------|
| `load_final(device)` | Load `model_final.pt`, rebuild network + feature config. |
| `_real_clip_features(...)` | A real clip's features for the parity check. |
| `export(args)` | ONNX export (opset 17, dynamic batch) → verify vs PyTorch → int8 quantize → verify agreement → write meta → report sizes + latency. |
| `parse_args` | `--no-quant` to skip the int8 variant. |

---

## `predictor.py` — shared inference (~118 lines)

| Symbol | Role |
|--------|------|
| `softmax(z)` | Numerically stable softmax. |
| `SignPredictor` | Reads `model_meta.json`, rebuilds the EXACT `FeatureConfig`, runs ONNX (default) or torch. `features()`, `predict_features()`, `predict()` → `{name, conf, probs, topk}`. |
| `video_to_clip(video_path, proc_width)` | Reuse `extract_dataset.extract_clip` + `build_all_vector` → the same `(T,1692)` raw vector. |

Used by **both** inference CLIs so the serving path is identical.

---

## `infer_video.py` — STAGE 4: file inference (~228 lines)

| Symbol | Role |
|--------|------|
| `draw_frame(...)` | Draw saved landmarks on an original frame (matches verify-overlay look; reuses `landmark_detector` draw fns). |
| `_banner(...)` | Top (whole-clip pred) + bottom (current window) subtitle bars. |
| `timeline(raw, fps, ...)` | Sliding-window per-segment predictions. |
| `render_overlay(...)` | Write the prediction overlay `.mp4`. |
| `resolve(args)` | `<class>/<stem>` or an arbitrary `--path` → video path + true label. |
| `run_one` / `main` / `parse_args` | Single clip, `--scan-dataset` batch test; `--backend onnx/torch --int8 --no-overlay --window --stride --proc-width`. |

---

## `infer_live.py` — STAGE 5: live translation (motion-gated)

| Symbol | Role |
|--------|------|
| `_overlay(disp, ev, ...)` | State-aware overlay: **LISTENING** vs **● REC** + motion bar, flashes a recognised sign, draws the transcript. |
| `process_segment(frames, ...)` | Write one motion-gated segment to a temp clip, run extractor + `predict_robust`, gate on confidence + margin + hands-seen, then **delete** the temp. |
| `_emit(out, ...)` | Apply the de-dup gate and append to the transcript. |
| `run_webcam(args, predictor)` | Motion-gated record→process→delete loop with live preview (Q to quit). |
| `run_source(args, predictor)` | Same loop simulated from a video file (no camera needed); whole-clip fallback if no segment triggers. |
| `_save_transcript(...)` | Write `live_transcript.json`; clean up any leftover temp segments. |
| `main` / `parse_args` | `--source --device-index --proc-width --backend --int8 --no-display --keep-temp` plus gating (`--conf --margin --repeat-window --no-dedup`) and motion (`--motion-start --motion-stop --floor --still --min-sign --max-sign --preroll`). |

## `segmenter.py` — live sign segmentation + de-duplication

| Symbol | Role |
|--------|------|
| `MotionSegmenter` | Frame-difference motion energy + a hysteresis state machine (auto-calibrated noise floor, pre-roll, still-hold end detection, min/max duration). `feed(frame)` → `SegEvent`; emits the segment frames once a sign closes. |
| `SignDebouncer` | Suppress an immediately-repeated sign within a time window (`accept(name, t) → bool`). |
| `SegEvent` | Per-frame report: `state`, `motion`, `level`, `started`, `segment`, `reason`. |
| `__main__` | Self-test on a synthetic motion trace (no camera): asserts 2 onsets / 2 segments and the de-dup behaviour. |
