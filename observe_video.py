"""
Observation Extractor — full-length / continuous-signing video
==============================================================
Applies the EXACT same pipeline used on the dataset (the functions in
extract_dataset.py + the drawing in landmark_detector.py) to one long video
such as a sentence/paragraph of continuous signing, and writes a
``processed_dataset``-style folder for it (default ``observation/``).

Why a separate tool: the dataset is *isolated* signs (one short clip = one
sign, clean start/end). A real reference is *continuous* (many signs run
together with no boundaries). This produces the same artifacts — landmark
``.npy``, a black-bg skeleton video, and an overlay cross-check — plus a
**detection timeline** so you can see the rhythm of a long conversation:
when each hand / the face / the body is present over the whole 7-minute stream.

Outputs (under ``--out``, default ``observation/``)::

    landmarks_pose/<stem>.npy     (T, 33, 4)
    landmarks_face/<stem>.npy     (T, 478, 3)
    landmarks_hands/<stem>.npy    (T, 2, 21, 3)
    landmarks_all/<stem>.npy      (T, 1692)
    skeleton_video/<stem>.mp4     black-bg skeleton (whole video)
    overlay_check/<stem>.mp4      original + saved landmarks | skeleton
    metadata/<stem>.json          QA (same schema as the dataset)
    reports/detection_timeline.png
    reports/observation_summary.png
    FEATURE_LAYOUT.json · README.md

Usage
-----
    python observe_video.py                       # asl.mp4 -> observation/
    python observe_video.py --video asl.mp4 --out observation
    python observe_video.py --no-overlay          # skip the (slow) overlay render
    python observe_video.py --proc-width 480      # smaller = faster
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import landmark_detector as ld                       # noqa: E402
import extract_dataset as ed                         # noqa: E402  (reuse the pipeline)

C_POSE, C_FACE, C_LH, C_RH = "#2E86C1", "#E67E22", "#27AE60", "#C0392B"
INK, MUTED = "#1B2631", "#5D6D7E"


# ── overlay (same drawing as the dataset overlays) ──────────────────────────────
def render_overlay(video_path, arrays, masks, opts, out_path, width=640):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if 1 < fps <= 120 else 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = width / W; OW, OH = int(W * scale), int(H * scale)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OW * 2, OH))

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (OW, OH)); black = np.zeros_like(frame)

        pose_proto = None
        if opts.pose and fi < arrays["pose"].shape[0] and masks["pose"][fi]:
            pose_proto = ed._proto(arrays["pose"][fi], with_vis=True)
        hands = []
        if opts.hands and fi < arrays["hands"].shape[0]:
            for slot, side in ((0, "Left"), (1, "Right")):
                if masks["hands"][fi, slot]:
                    hands.append((ed._proto(arrays["hands"][fi, slot], False), side))

        wrist_by_side = {}
        if pose_proto is not None and hands:
            lw = ld._lm_px(pose_proto.landmark[ld.LEFT_WRIST], OW, OH)
            rw = ld._lm_px(pose_proto.landmark[ld.RIGHT_WRIST], OW, OH)
            wr = [ld._lm_px(p.landmark[0], OW, OH) for p, _ in hands]
            for idx, s in ld.match_hands_to_sides(wr, lw, rw).items():
                wrist_by_side[s] = wr[idx]

        for target in (frame, black):
            if pose_proto is not None:
                ld.draw_upper_body(target, pose_proto, False, wrist_by_side, show_labels=False)
            if opts.face and fi < arrays["face"].shape[0] and masks["face"][fi]:
                ld.draw_face(target, [ed._proto(arrays["face"][fi], False)], False)
            if hands:
                ld.draw_hands(target, hands, mirrored=True)

        t = fi / fps
        cv2.putText(frame, f"original + landmarks   t={t:5.1f}s", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(black, "skeleton from .npy", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(np.hstack([frame, black])); fi += 1

    cap.release(); vw.release()
    return fi


# ── detection timeline (the key visual for a long continuous video) ─────────────
def _runs(mask):
    """Yield (start, length) for each run of 1s in a binary mask."""
    m = np.asarray(mask).astype(int)
    if m.size == 0:
        return
    pad = np.concatenate([[0], m, [0]])
    d = np.diff(pad)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    for s, e in zip(starts, ends):
        yield s, e - s


def fig_timeline(masks, fps, out_path):
    rows = [("pose", masks["pose"], C_POSE), ("face", masks["face"], C_FACE),
            ("left_hand", masks["hands"][:, 0], C_LH), ("right_hand", masks["hands"][:, 1], C_RH)]
    T = len(masks["pose"]); dur = T / fps
    fig, ax = plt.subplots(figsize=(15, 4.2))
    for i, (name, m, col) in enumerate(rows):
        y = len(rows) - 1 - i
        bars = [(s / fps, ln / fps) for s, ln in _runs(m)]
        ax.broken_barh(bars, (y + 0.1, 0.8), facecolors=col)
        ax.text(-dur * 0.005, y + 0.5, name, ha="right", va="center",
                fontsize=10, color=INK, fontweight="bold")
        ax.text(dur * 1.005, y + 0.5, f"{np.mean(m)*100:.0f}%", ha="left", va="center",
                fontsize=9, color=MUTED)
    ax.set_xlim(0, dur); ax.set_ylim(0, len(rows)); ax.set_yticks([])
    ax.set_xlabel("time (seconds)")
    ax.set_title(f"Detection timeline — {dur:.0f}s of continuous signing  "
                 f"(coloured = landmark present)", fontsize=13, color=INK, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def fig_summary(meta, masks, fps, out_path):
    nhands = masks["hands"].sum(1)  # 0/1/2 hands per frame
    frac = {k: float(np.mean(nhands == k)) for k in (0, 1, 2)}
    dr = meta["qa"]["detection_rate"]
    fig = plt.figure(figsize=(12, 5.2)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, color=INK))
    ax.text(0.5, 0.94, "CONTINUOUS-VIDEO OBSERVATION", ha="center", va="center",
            color="white", fontsize=18, fontweight="bold")
    facts = [
        ("Source", Path(meta["source"]).name),
        ("Duration / frames", f"{meta['T']/fps:.1f} s  ({meta['T']} frames @ {fps:.1f} fps)"),
        ("Pose detected", f"{dr['pose']*100:.0f}%"),
        ("Face detected", f"{dr['face']*100:.0f}%"),
        ("Right / Left hand", f"{dr['right_hand']*100:.0f}%  /  {dr['left_hand']*100:.0f}%"),
        ("Frames with 0 / 1 / 2 hands", f"{frac[0]*100:.0f}%  /  {frac[1]*100:.0f}%  /  {frac[2]*100:.0f}%"),
        ("Feature vector", f"{meta['all_shape'][1]} dims / frame"),
        ("Verdict", meta["qa"]["verdict"]),
    ]
    y = 0.76
    for k, v in facts:
        ax.text(0.06, y, k, fontsize=12, color=MUTED)
        ax.text(0.52, y, v, fontsize=12, fontweight="bold", color=INK)
        y -= 0.092
    fig.savefig(out_path, dpi=150); plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    a = parse_args()
    video = Path(a.video)
    if not video.exists():
        sys.exit(f"[ERROR] video not found: {video}")
    out = Path(a.out); stem = video.stem

    # Same modality flags object the dataset pipeline expects.
    opts = argparse.Namespace(
        face=not a.no_face, hands=not a.no_hands, pose=not a.no_pose,
        smooth=not a.no_smooth, proc_width=a.proc_width)

    folders = {"pose": out / "landmarks_pose", "face": out / "landmarks_face",
               "hands": out / "landmarks_hands", "all": out / "landmarks_all",
               "skeleton": out / "skeleton_video", "overlay": out / "overlay_check",
               "meta": out / "metadata", "reports": out / "reports"}
    for d in folders.values():
        d.mkdir(parents=True, exist_ok=True)

    total = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
    t0 = time.time()

    def progress(f):
        pct = 100 * f / total if total else 0
        print(f"  [extract] frame {f}/{total}  ({pct:4.1f}%)  {time.time()-t0:5.0f}s",
              flush=True)

    print(f"[OBSERVE] {video}  ->  {out}", flush=True)
    print(f"[OBSERVE] {total} frames; extracting with the dataset pipeline...", flush=True)
    res = ed.extract_clip(video, opts, progress=progress)
    if res is None:
        sys.exit(f"[ERROR] cannot open {video}")
    arrays, masks, T, fps = res["arrays"], res["masks"], res["T"], res["fps"]

    # Save landmark arrays (same shapes/semantics as the dataset).
    for name in ("pose", "face", "hands"):
        if name in arrays:
            np.save(folders[name] / f"{stem}.npy", arrays[name])
    all_vec, layout = ed.build_all_vector(arrays, opts)
    np.save(folders["all"] / f"{stem}.npy", all_vec)
    layout["note"] = "Same layout as processed_dataset/FEATURE_LAYOUT.json"
    json.dump(layout, open(out / "FEATURE_LAYOUT.json", "w"), indent=2)

    # Skeleton video (reuses ed.render_skeleton — same as the dataset).
    print("[OBSERVE] rendering skeleton video...", flush=True)
    ed.render_skeleton(arrays, masks, tuple(a.skeleton_size), fps,
                       folders["skeleton"] / f"{stem}.mp4", opts)

    # Overlay cross-check.
    if not a.no_overlay:
        print("[OBSERVE] rendering overlay cross-check (this is the slow part)...", flush=True)
        render_overlay(video, arrays, masks, opts, folders["overlay"] / f"{stem}.mp4",
                       width=a.overlay_width)

    # QA + metadata (same qa_clip the dataset uses).
    qa = ed.qa_clip(arrays, masks, T, opts)
    meta = {"source": str(video), "stem": stem, "source_fps": fps, "T": T,
            "smoothed": opts.smooth, "proc_width": opts.proc_width,
            "shapes": {n: list(arr.shape) for n, arr in arrays.items()},
            "all_shape": list(all_vec.shape),
            "masks": {n: m.tolist() for n, m in masks.items()}, "qa": qa}
    json.dump(meta, open(folders["meta"] / f"{stem}.json", "w"), indent=2)

    # Reports tailored to a long continuous video.
    print("[OBSERVE] building timeline + summary...", flush=True)
    fig_timeline(masks, fps, folders["reports"] / "detection_timeline.png")
    fig_summary(meta, masks, fps, folders["reports"] / "observation_summary.png")
    _write_readme(out, stem, video, T, fps)

    dr = qa["detection_rate"]
    print(f"\n[DONE] {T} frames in {time.time()-t0:.0f}s  verdict={qa['verdict']}", flush=True)
    print("[DONE] detection: " + " ".join(f"{k}={v:.0%}" for k, v in dr.items()), flush=True)
    print(f"[DONE] outputs in {out}/  (see reports/detection_timeline.png)", flush=True)


def _write_readme(out, stem, video, T, fps):
    txt = f"""# Observation — {video.name}

Continuous-signing reference processed with the SAME pipeline as the dataset
(`extract_dataset.extract_clip` + `landmark_detector` drawing).

- Duration: {T/fps:.1f} s ({T} frames @ {fps:.1f} fps)
- `landmarks_{{pose,face,hands,all}}/{stem}.npy` — same shapes as processed_dataset
- `skeleton_video/{stem}.mp4` — skeleton on black
- `overlay_check/{stem}.mp4` — original + saved landmarks | skeleton (cross-check)
- `reports/detection_timeline.png` — when each modality is present over time
- `reports/observation_summary.png` — headline stats
- `metadata/{stem}.json` — QA (same schema as the dataset)

This is *continuous* signing (sentences), unlike the *isolated* one-sign dataset
clips — useful for understanding what real inference input looks like and why a
sliding-window / sequence model is needed at inference time.
"""
    open(out / "README.md", "w", encoding="utf-8").write(txt)


def parse_args():
    p = argparse.ArgumentParser(description="Apply the dataset pipeline to one long video.")
    p.add_argument("--video", default="asl.mp4")
    p.add_argument("--out", default="observation")
    p.add_argument("--no-face", action="store_true")
    p.add_argument("--no-hands", action="store_true")
    p.add_argument("--no-pose", action="store_true")
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--no-overlay", action="store_true", help="skip the slow overlay render")
    p.add_argument("--proc-width", type=int, default=0, help="downscale input width (0 = full)")
    p.add_argument("--overlay-width", type=int, default=640, help="overlay panel width")
    p.add_argument("--skeleton-size", type=int, nargs=2, default=[960, 540], metavar=("W", "H"))
    return p.parse_args()


if __name__ == "__main__":
    main()
