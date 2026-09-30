# 01 — Overview

## What this system does

It turns a folder of sign-language **video clips** into a **machine-learning-ready
landmark dataset**, and then **proves** that the extracted data is correct.

For every clip it runs Google MediaPipe to detect three things per frame —
the **face mesh** (478 points), **both hands** (21 points each) and the
**upper-body pose** (33 points) — smooths them, and saves them as NumPy arrays.
It also renders a black-background skeleton video and an overlay video so the
numeric data can be checked by eye, and it produces QA reports and infographics.

The original detector (`landmark_detector.py`) only *displayed* landmarks live;
this project adds the **dataset extraction, verification and reporting** layer on
top of it.

## Why each piece exists

- **Extraction** (`extract_dataset.py`) — the data you actually train on.
- **Verification** (`verify_landmarks.py`) — answers "are the `.npy` files really
  correct?" with structural checks, a consistency proof, and an eyes-on overlay.
- **Pruning** (`prune_clips.py`) — removes bad/outlier clips reversibly.
- **Reporting** (`dataset_report.py`) — infographics that summarise quality.

## The dataset

- **12 word classes**: `good, hello, help, hi, how_are_you, me, nice_to_meet_you,
  no, sorry, thank_you, yes, you`.
- **134 clips** after pruning (136 filmed; 2 close-up outliers excluded).
- Source clips are 1920×1080 @ 30 fps, ~14–286 frames each.

## Tech stack

| Layer | Tool | Version (tested) |
|-------|------|------------------|
| Language | Python | 3.11 |
| Detection | MediaPipe | 0.10.9 |
| Video / image I/O | OpenCV (`opencv-python`) | 4.13 |
| Arrays | NumPy | 2.4 |
| Charts | matplotlib | 3.10 |
| PDF docs | reportlab | 5.0 |

## Directory tree

```
sign new exprmt/
├── landmark_detector.py        # CORE library: detection, smoothing, drawing (also a live viewer)
├── extract_dataset.py          # Pipeline: dataset/ -> processed_dataset/
├── verify_landmarks.py         # Cross-checks the .npy files (+ overlay videos)
├── prune_clips.py              # Reversibly drop clips from the dataset
├── dataset_report.py           # Verification infographics
│
├── dataset/                    # SOURCE videos (input)
│   ├── good/  *.mp4
│   ├── hello/ *.mp4
│   └── ... (12 classes)
│
├── excluded_clips/             # Clips pruned out (preserved, not deleted)
│   ├── no/no_13.mp4
│   └── yes/yes_12.mp4
│
├── processed_dataset/          # ALL OUTPUTS
│   ├── landmarks_pose/<class>/<stem>.npy     # (T, 33, 4)
│   ├── landmarks_face/<class>/<stem>.npy     # (T, 478, 3)
│   ├── landmarks_hands/<class>/<stem>.npy    # (T, 2, 21, 3)
│   ├── landmarks_all/<class>/<stem>.npy      # (T, 1692) combined
│   ├── skeleton_videos/<class>/<stem>.mp4    # black-bg skeleton
│   ├── overlay_check/<class>/<stem>.mp4      # original + landmarks, side by side
│   ├── metadata/<class>/<stem>.json          # per-clip QA
│   ├── reports/*.png                         # infographics
│   ├── FEATURE_LAYOUT.json                   # layout of the (T,1692) vector
│   ├── classes.json                          # class -> integer label
│   ├── manifest.csv                          # one row per clip
│   ├── extraction_report.json                # dataset-wide extraction QA
│   ├── verification_report.json              # dataset-wide verification result
│   └── README.md
│
└── docs/                       # THIS DOCUMENTATION
    ├── *.md
    ├── diagrams/*.png
    └── TECHNICAL_DOCUMENTATION.pdf
```
