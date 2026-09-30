# 06 — Developer Guide

## Setup

```
pip install mediapipe opencv-python numpy matplotlib reportlab
```
Python 3.11 recommended (MediaPipe 0.10.9 wheels).

## Everyday commands

```bash
# Extract the whole dataset (downscale to 960px wide for speed)
python extract_dataset.py --out processed_dataset --proc-width 960

# Extract just some classes / a few clips (quick test)
python extract_dataset.py --classes hello you --limit 2

# Verify every .npy file (structure + consistency + detection rates)
python verify_landmarks.py --out processed_dataset

# Eyes-on proof for one clip (landmarks reprojected on the original video)
python verify_landmarks.py --out processed_dataset --video hello/hello --dataset dataset --overlay

# Generate overlays for ALL clips
python verify_landmarks.py --out processed_dataset --dataset dataset --overlay

# Drop clips (reversible) and refresh reports
python prune_clips.py no/no_13 yes/yes_12

# Rebuild the infographics
python dataset_report.py

# Live viewer (sanity check the detector itself)
python landmark_detector.py --source dataset/hello/hello.mp4
```

## Common tasks

### Re-extract only one modality
```bash
python extract_dataset.py --no-face --no-pose      # hands only
```
The combined `landmarks_all` and `FEATURE_LAYOUT.json` adjust automatically.

### Add a new class
1. Drop `dataset/<new_class>/*.mp4`.
2. `python extract_dataset.py --classes <new_class>` (or re-run all).
3. `python verify_landmarks.py --out processed_dataset`.
4. `python dataset_report.py`.

### Resume an interrupted run
```bash
python extract_dataset.py --skip-existing
```

### Store raw (unsmoothed) landmarks
```bash
python extract_dataset.py --no-smooth
```

## Debugging guide

| Symptom | Likely cause / action |
|---------|----------------------|
| A clip is `WARN` for "pose < 50%" | Body/face out of frame (close-up). Inspect with `--overlay`; prune if it's an outlier. |
| `left_hand` near 0% | Expected for one-handed (right-handed) signs — see the heatmap. Not a bug. |
| Pose `x/y` > 1 | By design — MediaPipe extrapolates off-screen joints. Pose is not range-checked. |
| `FAIL` for "all != parts" | Corruption or a format change. Re-extract that clip. |
| Slow extraction | Lower `--proc-width` (e.g. 640); it does not affect normalised coordinates. |
| Hands swapped L/R | Check `anatomical_label` / `match_hands_to_sides`; raw video is *not* mirrored. |

### Inspect a `.npy` by hand
```python
import numpy as np
a = np.load("processed_dataset/landmarks_pose/hello/hello.npy")
print(a.shape, a.dtype, np.isnan(a).any())
print("pose detected frames:", int((np.abs(a).reshape(len(a),-1).sum(1) > 0).sum()), "/", len(a))
```

## Performance notes

- Models are **recreated per clip** (≈0.5–1 s each) to avoid cross-clip tracking
  leakage; this dominates the per-clip cost on short videos.
- `--proc-width` is the main speed knob. Normalised landmark coordinates are
  resolution-independent, so downscaling input is essentially free accuracy-wise
  for upper-body framing.
- Skeleton/overlay rendering reads only the saved arrays (no detection), so it is
  fast; disable with `--no-skeleton` if not needed.

## Conventions for contributors

- Keep detection/drawing logic in `landmark_detector.py`; tools should import it,
  not re-implement it.
- Never hand-edit files under `processed_dataset/` — regenerate them.
- Prune by **moving** sources to `excluded_clips/`, never deleting `dataset/`.
