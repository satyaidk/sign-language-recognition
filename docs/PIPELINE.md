# How each stage works

## 1. Landmark extraction (`extract`, `landmarks/extractor.py`)

For every clip a **fresh** `LandmarkExtractor` is created, so MediaPipe's tracking
state never leaks from one clip into the next. Then, per frame:

1. Optionally downscale to `--proc-width` (landmarks are normalised image
   coordinates, so this only changes speed and detection quality).
2. Run **FaceMesh** (478 points incl. irises), **Hands** (up to 2 × 21 points) and
   **Pose** (33 points with visibility).
3. **Smooth** pose and hands with a One-Euro filter (below). Smoothing never
   back-fills: a frame without a detection stays zero and its mask stays 0, so
   the training data is honest about what was actually seen.
4. Put each hand into its **anatomical slot** (0 = signer's left, 1 = right).
5. Stack all frames into `pose (T,33,4)`, `face (T,478,3)`, `hands (T,2,21,3)` and
   the combined **`all (T,1692)`** = pose | face | left hand | right hand.

The skeleton video is rendered **from the saved arrays**, not from a second
detection pass, so watching it shows exactly what the `.npy` file contains.

### One-Euro filter (`landmarks/smoothing.py`)
An adaptive low-pass filter (Casiez et al., 2012). Its cutoff frequency rises with
speed: `cutoff = min_cutoff + beta * |velocity|`. A still hand gets heavy smoothing
(no jitter) and a fast hand gets almost none (no lag). The implementation is
vectorised over all points of a hand or body. A point that jumps more than 0.3
(normalised units) is reset instead of smeared. Parameters: pose `min_cutoff=1.2,
beta=0.3`; hands `min_cutoff=2.0, beta=0.8`.

### Hand sides (`landmarks/hands.py`)
MediaPipe labels handedness **as if the image were a mirrored selfie**. For raw
video the label is swapped to get the signer's true side. The label is sometimes
wrong, most often by returning "Left" for both hands. The original extractor
then wrote both hands into one slot and lost one. `assign_hand_slots` now
resolves duplicates geometrically:
- If the body is tracked, it uses the nearest pose wrist (optimal 2-hand pairing).
- Otherwise it uses image position (in a raw frame the signer's right hand appears on the image's left).

The same optimal pairing makes the drawn arms connect to the right hand, so they never cross into an "X".

## 2. Quality control (`dataset/qa.py`, `verify`, `report`, `prune`)

| Check | Where | Verdict |
|-------|-------|---------|
| NaN / Inf values | extraction + verification | FAIL |
| No frames | extraction | FAIL |
| Modalities have different frame counts | verification | FAIL |
| `all` vector ≠ concatenation of the separate pose/face/hands files | verification | FAIL |
| Pose detected in < 50% of frames | both | WARN |
| Face / hand x,y far outside the image (< −0.5 or > 1.5) | both | WARN |

Pose is not range-checked: MediaPipe legitimately extrapolates off-screen joints
such as the hips in an upper-body shot. `verify --overlay` renders the **original
video with the saved landmarks** next to the skeleton. That is the eyes-on proof
that the numbers track the signer. `report` draws six charts, and every number on
them comes from the metadata and verification report; a check that hasn't run
shows as "not verified". `prune` moves a bad clip's video to `excluded_clips/`
(reversible), deletes its derived files and refreshes every dataset-level file.

Partial runs are safe: `extract --classes/--limit/--skip-existing` **merges**
`classes.json` (existing label numbers never change), `manifest.csv` and
`extraction_report.json`.

## 3. Feature transform (`training/features.py`)

A clip `(T, 1692)` becomes a fixed model input `(L=64, D=346)`. The same function
is used in training, file inference and live inference.

```
raw (T,1692) --geometric--> coords (T,55,3), visibility (T,13), presence {pose, L, R}
             --[augment]--> (training only)
             --assemble---> resample to 64 frames + velocity + masks -> (64, 346)
```

| Block | Width | Content |
|-------|-------|---------|
| coordinates | 165 | 55 points × (x, y, z): 13 upper-body pose joints (nose, shoulders, elbows, wrists, hand roots) + 21 left-hand + 21 right-hand |
| visibility | 13 | pose visibility of those joints |
| velocity | 165 | first difference of the resampled coordinates (motion) |
| masks | 3 | pose / left hand / right hand present in that frame |

- **Normalisation:** subtract the mid-shoulder point and divide by the shoulder
  width, using one clip-level reference (the median over frames where the pose is
  present). This makes the features invariant to where the signer stands and how
  big they appear. Absent hands are re-zeroed *after* normalisation. Without that
  step, an absent hand would sit at `-origin/scale`, a "ghost hand" the model
  could learn from.
- **Resampling:** linear interpolation to 64 frames (nearest-neighbour for the
  0/1 masks), so clips of 14–286 frames share one length.
- **No fitted state:** nothing is learned from the dataset (no mean/std). The
  model's input LayerNorm absorbs the remaining scale, and the saved feature
  config fully describes the transform.

## 4. Augmentation (`training/data.py`, training only)

Applied in the *normalised* space, between `geometric` and `assemble`. Scaling
the raw frame would simply be normalised away.

| Augmentation | Setting |
|--------------|---------|
| Horizontal flip **with left/right swap** of hands, pose joints, visibility and masks | p = 0.5 |
| Rotation | ±13° |
| Scale / translation / jitter | ±12% / ±0.06 / σ = 0.012 |
| Time warp (random sub-window) / frame dropout | up to 15% / 7% |

The flip is the most valuable one: it turns a right-handed example into a valid
left-handed one. Absent points are re-zeroed after every geometric operation.

## 5. Model (`training/model.py`)

```
(B, 64, 346) -> LayerNorm -> Linear(346->128) + GELU + Dropout
             -> 2-layer bidirectional GRU (128 per direction)
             -> additive attention pooling over time
             -> LayerNorm -> Dropout -> Linear(256 -> classes)
```

543,553 parameters. A Transformer encoder (`--arch transformer`, learned positions
sized to the sequence length) is available for larger datasets.

## 6. Training (`train`, `finetune`)

- **Stage 1, cross-validation:** stratified 5-fold. Every clip is validated exactly
  once and the out-of-fold predictions give accuracy, macro-F1, per-class metrics and
  a confusion matrix over the whole dataset. Fold sizes are balanced (class start
  folds rotate).
- **Optimisation:** AdamW (lr 1.5e-3, weight decay 1e-4), 5-epoch warm-up then cosine
  decay to 1e-5, label smoothing 0.05, gradient clipping 5.0, batch 16, up to 120
  epochs, early stopping (patience 35) on validation accuracy.
- **Stage 2, deployment fit:** the same recipe on 100% of the data, with an
  **exponential moving average** of the weights (decay 0.999 with warm-up
  `min(0.999, (1+step)/(10+step))`, because a run is only a few hundred steps).
  The EMA or raw weights, whichever fits the training clips better, are saved with
  a self-describing meta. That training-fit number is logged as such and is never
  presented as accuracy.

## 7. Export (`export`)

- ONNX opset 17, input `features [batch, 64, 346]` → output `logits [batch, classes]`.
- Uses the TorchScript exporter (`dynamo=False`). The newer dynamo exporter needs
  `onnxscript` and has open GRU issues. PyTorch ≥ 2.9 warns about the legacy
  exporter, and export silences that expected warning.
- **Parity check:** ONNX Runtime vs PyTorch on random inputs (batch 1 and 3) and a
  real clip. Export fails if the max logit difference is ≥ 1e-3 (typically ~1e-6).
- **Int8:** dynamic quantisation is tried and kept only if it agrees with fp32.
  ONNX Runtime does not quantise GRU layers, so for this model the int8 file is
  only a few percent smaller. The number is measured and reported, not assumed.

## 8. Inference

`SignPredictor` loads `model_meta.json`, rebuilds the exact feature config used in
training, and runs ONNX (default) or the PyTorch checkpoint.
- `predict(raw)`: one whole-clip pass.
- `predict_robust(raw)`: averages the softmax of the whole clip and overlapping
  48-frame windows. This stabilises look-alike signs and gives a meaningful
  top-1 − top-2 margin.

File mode (`video`) adds a sliding-window timeline and an overlay video. Live mode
is described in [LIVE_RECOGNITION.md](LIVE_RECOGNITION.md).
