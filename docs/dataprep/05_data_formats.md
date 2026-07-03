# 05 — Data Formats

## Output folders

| Folder | File | Shape | Contents |
|--------|------|-------|----------|
| `landmarks_pose/` | `<class>/<stem>.npy` | `(T, 33, 4)` | `x, y, z, visibility` |
| `landmarks_face/` | `<class>/<stem>.npy` | `(T, 478, 3)` | `x, y, z` |
| `landmarks_hands/` | `<class>/<stem>.npy` | `(T, 2, 21, 3)` | slot 0 = Left, slot 1 = Right; `x, y, z` |
| `landmarks_all/` | `<class>/<stem>.npy` | `(T, 1692)` | combined per-frame feature vector |
| `skeleton_videos/` | `<class>/<stem>.mp4` | — | black-bg skeleton from the arrays |
| `overlay_check/` | `<class>/<stem>.mp4` | — | original + landmarks ∣ skeleton |
| `metadata/` | `<class>/<stem>.json` | — | per-clip QA |

`T` is the number of frames in the clip. `dtype` is `float32` everywhere.

## The combined vector `(T, 1692)`

Per frame, in this exact order (`FEATURE_LAYOUT.json` is the machine-readable
source of truth):

| Block | Columns | Width | Source |
|-------|---------|-------|--------|
| `pose` | 0 – 131 | 132 | 33 × (x, y, z, visibility) |
| `face` | 132 – 1565 | 1434 | 478 × (x, y, z) |
| `left_hand` | 1566 – 1628 | 63 | 21 × (x, y, z) |
| `right_hand` | 1629 – 1691 | 63 | 21 × (x, y, z) |

Total **1692** features/frame. If a modality is disabled with `--no-*`, its block
is omitted everywhere and the layout shrinks accordingly.

```python
import numpy as np, json
all_vec = np.load("data/processed_dataset/landmarks_all/hello/hello.npy")  # (T, 1692)
layout  = json.load(open("data/processed_dataset/FEATURE_LAYOUT.json"))
pose_block = all_vec[:, 0:132].reshape(-1, 33, 4)   # == landmarks_pose/hello/hello.npy
```

## Coordinate system

- `x, y` — normalised image coordinates, `[0, 1]` = fraction of width / height.
- `z` — relative depth, same scale as `x` (negative = closer to camera).
- `visibility` (pose only) — `[0, 1]` confidence the joint is visible.
- **Missing landmarks** in a frame are stored as **zeros**; presence is recorded
  in the metadata `masks` (and is recoverable as "block is not all-zero").

## `metadata/<class>/<stem>.json`

```jsonc
{
  "class": "hello", "label": 1, "stem": "hello",
  "source": "data/dataset/hello/hello.mp4", "source_fps": 30.0,
  "frames_read": 65, "frames_skipped": 0, "T": 65,
  "smoothed": true, "proc_width": 960,
  "shapes": { "pose": [65,33,4], "face": [65,478,3], "hands": [65,2,21,3] },
  "all_shape": [65, 1692],
  "masks": { "pose": [1,1,...], "face": [...], "hands": [[0,1],...] },
  "qa": {
    "detection_rate": { "pose": .94, "face": .94, "left_hand": 0, "right_hand": .46, "any_hand": .46 },
    "ranges": { ... }, "issues": [], "verdict": "PASS"
  }
}
```

## Dataset-wide files

- **`classes.json`** — `{ "good": 0, "hello": 1, ... }` (class → integer label).
- **`manifest.csv`** — one row per clip: `class, label, stem, T, verdict,
  all_npy, skeleton`. Use this to drive a train/val/test split.
- **`extraction_report.json`** — `n_clips`, `verdicts`, `mean_detection_rate`,
  per-clip summaries (produced by extraction).
- **`verification_report.json`** — independent re-check: per-clip
  `all_vs_parts_max_diff`, detection rates, verdict (produced by verification).

## Verdicts

| Verdict | Meaning |
|---------|---------|
| `PASS` | No issues: valid floats, consistent, pose detected in ≥ 50% of frames. |
| `WARN` | Usable but flagged (e.g. pose < 50%, or face/hands far outside `[0,1]`). |
| `FAIL` | Unusable: empty clip, or NaN/Inf, or `all` ≠ parts. |
