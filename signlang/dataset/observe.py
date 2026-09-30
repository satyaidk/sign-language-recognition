"""
Observation — the dataset pipeline applied to one long, continuous-signing video.
================================================================================
The dataset is *isolated* signs (one short clip = one sign, clean start/end).
Real signing is *continuous* (many signs with no boundaries).  This tool runs
the SAME extractor, QA and drawing on a long video and adds a **detection
timeline**, so you can see when each hand / the face / the body is present over
the whole stream — the key picture for designing live segmentation.

Outputs (under ``--out``, default ``observation/``)::

    landmarks_{pose,face,hands,all}/<stem>.npy   same shapes as processed_dataset
    skeleton_video/<stem>.mp4                    skeleton on black
    overlay_check/<stem>.mp4                     original + saved landmarks | skeleton
    metadata/<stem>.json                         QA (same schema as the dataset)
    reports/detection_timeline.png · reports/observation_summary.png

Run:
    python -m signlang observe --video conversation.mp4
    python -m signlang observe --video conversation.mp4 --no-overlay --proc-width 480
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from signlang.dataset.qa import qa_clip  # noqa: E402
from signlang.landmarks.extractor import ExtractorOptions, extract_video  # noqa: E402
from signlang.landmarks.layout import build_all_vector  # noqa: E402
from signlang.utils import save_json  # noqa: E402

COLORS = {"pose": "#2E86C1", "face": "#E67E22", "left_hand": "#27AE60", "right_hand": "#C0392B"}
INK, MUTED = "#1B2631", "#5D6D7E"


def runs(mask):
    """Yield ``(start, length)`` for each run of 1s in a binary mask."""
    m = np.asarray(mask).astype(int)
    if m.size == 0:
        return
    d = np.diff(np.concatenate([[0], m, [0]]))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        yield int(s), int(e - s)


def timeline_rows(masks: dict) -> list:
    """``[(name, mask)]`` for whichever modalities were extracted."""
    rows = []
    if "pose" in masks:
        rows.append(("pose", masks["pose"]))
    if "face" in masks:
        rows.append(("face", masks["face"]))
    if "hands" in masks:
        rows += [("left_hand", masks["hands"][:, 0]), ("right_hand", masks["hands"][:, 1])]
    return rows


def fig_timeline(masks, fps, out_path):
    rows = timeline_rows(masks)
    T = len(rows[0][1]) if rows else 0
    dur = max(T / fps, 1e-6)
    fig, ax = plt.subplots(figsize=(15, 1.2 + 0.75 * len(rows)))
    for i, (name, m) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.broken_barh([(s / fps, ln / fps) for s, ln in runs(m)], (y + 0.1, 0.8),
                       facecolors=COLORS[name])
        ax.text(-dur * 0.005, y + 0.5, name, ha="right", va="center", fontsize=10,
                color=INK, fontweight="bold")
        ax.text(dur * 1.005, y + 0.5, f"{np.mean(m) * 100:.0f}%", ha="left", va="center",
                fontsize=9, color=MUTED)
    ax.set_xlim(0, dur); ax.set_ylim(0, max(1, len(rows))); ax.set_yticks([])
    ax.set_xlabel("time (seconds)")
    ax.set_title(f"Detection timeline — {dur:.0f}s of signing (coloured = landmark present)",
                 fontsize=13, color=INK, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def fig_summary(meta, masks, fps, out_path):
    dr = meta["qa"]["detection_rate"]
    pct = lambda k: f"{dr[k] * 100:.0f}%" if k in dr else "not extracted"  # noqa: E731
    facts = [("Source", Path(meta["source"]).name),
             ("Duration / frames", f"{meta['T'] / fps:.1f} s  ({meta['T']} frames @ {fps:.1f} fps)"),
             ("Pose detected", pct("pose")), ("Face detected", pct("face")),
             ("Right / Left hand", f"{pct('right_hand')}  /  {pct('left_hand')}")]
    if "hands" in masks and len(masks["hands"]):
        n = np.asarray(masks["hands"]).sum(1)
        facts.append(("Frames with 0 / 1 / 2 hands",
                      "  /  ".join(f"{np.mean(n == k) * 100:.0f}%" for k in (0, 1, 2))))
    facts += [("Feature vector", f"{meta['all_shape'][1]} dims / frame"),
              ("Verdict", meta["qa"]["verdict"])]

    fig = plt.figure(figsize=(12, 5.2)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, color=INK))
    ax.text(0.5, 0.94, "CONTINUOUS-VIDEO OBSERVATION", ha="center", va="center",
            color="white", fontsize=18, fontweight="bold")
    y = 0.76
    for k, v in facts:
        ax.text(0.06, y, k, fontsize=12, color=MUTED)
        ax.text(0.52, y, v, fontsize=12, fontweight="bold", color=INK)
        y -= 0.085
    fig.savefig(out_path, dpi=150); plt.close(fig)


def observe(video: Path, out: Path, opts: ExtractorOptions, overlay=True,
            overlay_width=640, skeleton_size=(960, 540)) -> dict:
    stem = video.stem
    for sub in ("landmarks_pose", "landmarks_face", "landmarks_hands", "landmarks_all",
                "skeleton_video", "overlay_check", "metadata", "reports"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    t0 = time.time()

    def progress(f):
        pct = 100 * f / total if total else 0
        print(f"  [extract] frame {f}/{total} ({pct:4.1f}%)  {time.time() - t0:5.0f}s", flush=True)

    print(f"[OBSERVE] {video} -> {out}  ({total} frames)", flush=True)
    res = extract_video(video, opts, progress=progress)
    if res is None:
        raise FileNotFoundError(f"cannot open {video}")
    arrays, masks, T, fps = res["arrays"], res["masks"], res["T"], res["fps"]

    for name, folder in (("pose", "landmarks_pose"), ("face", "landmarks_face"),
                         ("hands", "landmarks_hands")):
        if name in arrays:
            np.save(out / folder / f"{stem}.npy", arrays[name])
    all_vec, layout = build_all_vector(arrays, opts.modalities)
    np.save(out / "landmarks_all" / f"{stem}.npy", all_vec)
    layout["note"] = "Same layout as processed_dataset/FEATURE_LAYOUT.json"
    save_json(layout, out / "FEATURE_LAYOUT.json")

    from signlang.landmarks.drawing import render_side_by_side, render_skeleton_video
    print("[OBSERVE] rendering skeleton video...", flush=True)
    render_skeleton_video(arrays, masks, skeleton_size, fps, out / "skeleton_video" / f"{stem}.mp4")
    if overlay:
        print("[OBSERVE] rendering overlay cross-check (slow)...", flush=True)
        render_side_by_side(video, arrays, masks, out / "overlay_check" / f"{stem}.mp4",
                            width=overlay_width)

    qa = qa_clip(arrays, masks, T)
    meta = {"source": str(video), "stem": stem, "source_fps": fps, "T": T,
            "smoothed": opts.smooth, "proc_width": opts.proc_width,
            "shapes": {n: list(a.shape) for n, a in arrays.items()},
            "all_shape": list(all_vec.shape),
            "masks": {n: np.asarray(m).tolist() for n, m in masks.items()}, "qa": qa}
    save_json(meta, out / "metadata" / f"{stem}.json")
    fig_timeline(masks, fps, out / "reports" / "detection_timeline.png")
    fig_summary(meta, masks, fps, out / "reports" / "observation_summary.png")

    print(f"\n[DONE] {T} frames in {time.time() - t0:.0f}s  verdict={qa['verdict']}")
    print("[DONE] detection: " + " ".join(f"{k}={v:.0%}" for k, v in qa["detection_rate"].items()))
    print(f"[DONE] outputs in {out}  (see reports/detection_timeline.png)")
    return meta


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="signlang observe",
                                description="Apply the dataset pipeline to one long video.")
    p.add_argument("--video", required=True, help="continuous-signing video to analyse")
    p.add_argument("--out", default="observation", help="output folder")
    p.add_argument("--no-face", action="store_true")
    p.add_argument("--no-hands", action="store_true")
    p.add_argument("--no-pose", action="store_true")
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--no-overlay", action="store_true", help="skip the slow overlay render")
    p.add_argument("--proc-width", type=int, default=0, help="downscale input width (0 = full)")
    p.add_argument("--overlay-width", type=int, default=640, help="overlay panel width")
    p.add_argument("--skeleton-size", type=int, nargs=2, default=[960, 540], metavar=("W", "H"))
    return p.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    video = Path(a.video)
    if not video.exists():
        sys.exit(f"[ERROR] video not found: {video}")
    opts = ExtractorOptions(face=not a.no_face, hands=not a.no_hands, pose=not a.no_pose,
                            smooth=not a.no_smooth, proc_width=a.proc_width)
    if not opts.modalities:
        sys.exit("[ERROR] nothing to extract: --no-face --no-hands --no-pose all set")
    try:
        observe(video, Path(a.out), opts, overlay=not a.no_overlay,
                overlay_width=a.overlay_width, skeleton_size=tuple(a.skeleton_size))
    except FileNotFoundError as exc:
        sys.exit(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
