"""
Render the documentation diagrams (architecture, data flow, verification flow)
to docs/diagrams/*.png.  Pure matplotlib so it always works.

    python docs/make_diagrams.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED = "#1B2631", "#5D6D7E"
C_CORE, C_TOOL, C_MP, C_DATA, C_OUT = "#2E86C1", "#8E44AD", "#16A085", "#7F8C8D", "#E67E22"


def box(ax, x, y, w, h, text, color, fg="white", fs=10, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2",
                                linewidth=1.2, edgecolor="white", facecolor=color, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=fg,
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=3, wrap=True)


def arrow(ax, p0, p1, color=INK, style="-|>", lw=1.6, rad=0.0):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"), zorder=1)


def canvas(w=14, h=9, title=""):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    if title:
        ax.text(50, 97, title, ha="center", va="top", fontsize=17,
                fontweight="bold", color=INK)
    return fig, ax


# ── 1. System architecture (layered) ───────────────────────────────────────────
def architecture():
    fig, ax = canvas(title="System Architecture")

    ax.text(50, 90, "Entry-point tools (CLI)", ha="center", color=MUTED, fontsize=11)
    tools = [("extract_dataset.py\nextraction pipeline", 3),
             ("verify_landmarks.py\ncross-check + overlay", 27.5),
             ("prune_clips.py\nreversible drop", 52),
             ("dataset_report.py\ninfographics", 74)]
    for t, x in tools:
        box(ax, x, 79, 22, 9, t, C_TOOL, fs=9.5)

    box(ax, 20, 55, 60, 13,
        "landmark_detector.py  —  CORE LIBRARY\n"
        "OneEuroFilter · PointStabilizer · HandStabilizer\n"
        "draw_upper_body / draw_face / draw_hands · hand-side matching",
        C_CORE, fs=10)

    ax.text(50, 47, "Google MediaPipe Solutions", ha="center", color=MUTED, fontsize=11)
    for t, x in [("FaceMesh\n478 pts", 22), ("Hands\n21 pts x2", 42), ("Pose\n33 pts", 62)]:
        box(ax, x, 35, 16, 9, t, C_MP, fs=9.5)

    box(ax, 6, 12, 26, 12, "dataset/\n<class>/<clip>.mp4\n(136 source videos)", C_DATA, fs=9.5)
    box(ax, 40, 12, 54, 12,
        "processed_dataset/\nlandmarks_{pose,face,hands,all} · skeleton_videos\n"
        "overlay_check · metadata · reports · *.json/csv", C_OUT, fs=9.5)

    # dependency arrows (tools -> core)
    for _, x in tools:
        arrow(ax, (x + 11, 79), (50, 68), MUTED, lw=1.1, rad=0.05)
    # core -> mediapipe
    for x in (30, 50, 70):
        arrow(ax, (50, 55), (x, 44), MUTED, lw=1.1, rad=0.0)
    # data flow
    arrow(ax, (19, 18), (40, 18), C_DATA, lw=2.2)
    arrow(ax, (14, 79), (19, 24), C_TOOL, lw=1.4, rad=-0.25)   # extract reads dataset
    arrow(ax, (14, 79), (60, 24), C_TOOL, lw=1.4, rad=0.25)    # extract writes processed
    ax.text(29.5, 20, "read", ha="center", color=C_DATA, fontsize=9, fontweight="bold")

    fig.savefig(OUT / "architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 2. Extraction data flow (per clip) ──────────────────────────────────────────
def dataflow():
    fig, ax = canvas(w=15, h=9, title="Extraction Data Flow  (per clip)")
    steps = [
        ("video.mp4", C_DATA, 4, 70),
        ("read frame\n(OpenCV)", C_CORE, 20, 70),
        ("resize\n--proc-width", C_CORE, 36, 70),
        ("BGR -> RGB", C_CORE, 52, 70),
        ("MediaPipe\nFace / Hands / Pose", C_MP, 70, 70),
        ("One-Euro\nsmoothing", C_CORE, 70, 50),
        ("per-frame arrays\n+ presence masks", C_CORE, 50, 50),
        ("stack over T frames", C_CORE, 28, 50),
    ]
    for t, c, x, y in steps:
        box(ax, x, y, 14, 9, t, c, fs=9)
    chain = [(0, 1), (1, 2), (2, 3), (3, 4)]
    for a, b in chain:
        arrow(ax, (steps[a][2] + 14, steps[a][3] + 4.5), (steps[b][2], steps[b][3] + 4.5))
    arrow(ax, (77, 70), (77, 59))                  # mp -> smoothing
    arrow(ax, (70, 54.5), (64, 54.5))              # smoothing -> arrays
    arrow(ax, (50, 54.5), (42, 54.5))              # arrays -> stack

    # outputs
    outs = [
        ("landmarks_pose/  (T,33,4)", 4, 33),
        ("landmarks_face/  (T,478,3)", 4, 25),
        ("landmarks_hands/ (T,2,21,3)", 4, 17),
        ("landmarks_all/   (T,1692)", 4, 9),
    ]
    for t, x, y in outs:
        box(ax, x, y, 34, 6.2, t, C_OUT, fs=9)
    box(ax, 44, 25, 24, 8, "render skeleton\n-> skeleton_videos/*.mp4", C_CORE, fs=9)
    box(ax, 44, 13, 24, 8, "QA checks\n-> metadata/*.json", C_TOOL, fs=9)
    box(ax, 74, 17, 22, 12, "aggregate ->\nextraction_report.json\nmanifest.csv\nclasses.json",
        C_OUT, fs=9)

    arrow(ax, (35, 50), (20, 39.2), MUTED, rad=0.1)   # stack -> outputs
    arrow(ax, (35, 50), (52, 33), MUTED, rad=-0.1)    # stack -> skeleton
    arrow(ax, (35, 50), (52, 21), MUTED, rad=-0.2)    # stack -> QA
    arrow(ax, (68, 21), (74, 23), MUTED)              # QA -> report
    ax.text(50, 4, "One MediaPipe pass per clip feeds every output — the separate folders "
            "and the combined vector never diverge.", ha="center", color=MUTED, fontsize=9.5)

    fig.savefig(OUT / "dataflow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 3. Verification flow ────────────────────────────────────────────────────────
def verification():
    fig, ax = canvas(w=14, h=8.5, title="Verification & Reporting Flow")
    box(ax, 4, 70, 22, 10, "processed_dataset/\nlandmarks_*.npy", C_OUT, fs=9.5)

    checks = [
        ("1. STRUCTURE\nshape · dtype · NaN/Inf · ranges", 34, 78),
        ("2. CONSISTENCY\nrebuild all == face+hands+pose", 34, 64),
        ("3. DETECTION\nrates recomputed from arrays", 34, 50),
    ]
    for t, x, y in checks:
        box(ax, x, y, 34, 9, t, C_CORE, fs=9)
    for _, x, y in checks:
        arrow(ax, (26, 75), (x, y + 4.5), MUTED, rad=0.05)

    box(ax, 74, 64, 22, 10, "verdict\nPASS / WARN / FAIL", C_TOOL, fs=10)
    for _, x, y in checks:
        arrow(ax, (68, y + 4.5), (74, 69), MUTED, rad=0.0)

    box(ax, 34, 33, 34, 9, "4. OVERLAY (optional)\nreproject saved pts onto original video",
        C_MP, fs=9)
    arrow(ax, (21, 70), (40, 42), MUTED, rad=-0.2)
    box(ax, 74, 33, 22, 9, "overlay_check/\n*.mp4 (eyes-on proof)", C_OUT, fs=9)
    arrow(ax, (68, 37.5), (74, 37.5), MUTED)

    box(ax, 30, 14, 26, 9, "verification_report.json", C_OUT, fs=9.5)
    arrow(ax, (85, 64), (50, 23), MUTED, rad=0.2)
    box(ax, 62, 14, 30, 9, "dataset_report.py ->\nreports/*.png infographics", C_TOOL, fs=9)
    arrow(ax, (56, 18.5), (62, 18.5), MUTED)

    fig.savefig(OUT / "verification_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture(); dataflow(); verification()
    print(f"[OK] diagrams -> {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print("   ", p)
