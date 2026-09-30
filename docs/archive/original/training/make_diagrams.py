"""
Render the training-docs diagrams (architecture, feature transform, pipeline
stages) to training/docs/diagrams/*.png.  Pure matplotlib so it always works.

    python training/docs/make_diagrams.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED = "#1B2631", "#5D6D7E"
# core/shared, train stages, export, inference, data-in, data-out
C_CORE, C_TRAIN, C_EXP, C_INFER, C_DATA, C_OUT = \
    "#2E86C1", "#8E44AD", "#16A085", "#C0392B", "#7F8C8D", "#E67E22"


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


# ── 1. Training & inference architecture (layered) ──────────────────────────────
def architecture():
    fig, ax = canvas(title="Training & Inference Architecture")

    ax.text(50, 90, "Training stages (offline)", ha="center", color=MUTED, fontsize=11)
    box(ax, 8, 79, 24, 9, "train.py\nStage 1 - k-fold CV", C_TRAIN, fs=9.5)
    box(ax, 38, 79, 24, 9, "finetune.py\nStage 2 - full fit + EMA", C_TRAIN, fs=9.5)
    box(ax, 68, 79, 24, 9, "export_optimize.py\nStage 3 - ONNX + int8", C_EXP, fs=9.5)

    box(ax, 16, 55, 68, 13,
        "SHARED FOUNDATION\n"
        "config.py (one source of truth) - features.py (raw -> (L,D))\n"
        "model.py (BiGRU / Transformer) - utils.py - viz.py - data.py",
        C_CORE, fs=10)

    ax.text(50, 47, "Inference (one shared serving path)", ha="center", color=MUTED, fontsize=11)
    box(ax, 14, 35, 24, 9, "predictor.py\nlandmarks -> sign", C_INFER, fs=9.5)
    box(ax, 41, 35, 21, 9, "infer_video.py\nStage 4 - file", C_INFER, fs=9.5)
    box(ax, 65, 35, 21, 9, "infer_live.py\nStage 5 - webcam", C_INFER, fs=9.5)

    box(ax, 4, 12, 26, 12, "processed_dataset/\nlandmarks_all/*.npy\nmanifest.csv (read-only)", C_DATA, fs=9)
    box(ax, 34, 12, 30, 12, "training/artifacts/\ncheckpoints - exported\nmetrics - predictions", C_OUT, fs=9)
    box(ax, 68, 12, 28, 12, "reuse (project root)\nextract_dataset.py\nlandmark_detector.py", C_DATA, fs=9)

    # stages -> foundation
    for x in (20, 50, 80):
        arrow(ax, (x, 79), (50, 68), MUTED, lw=1.1, rad=0.04)
    # foundation -> inference
    for x in (26, 51, 75):
        arrow(ax, (50, 55), (x, 44), MUTED, lw=1.1, rad=0.0)
    # data read / write
    arrow(ax, (17, 24), (20, 79), C_DATA, lw=1.4, rad=-0.25)     # read dataset
    arrow(ax, (62, 79), (45, 24), C_OUT, lw=1.4, rad=0.2)        # write artifacts
    arrow(ax, (75, 35), (80, 24), MUTED, lw=1.2)                 # inference -> reuse
    ax.text(11, 52, "read", ha="center", color=C_DATA, fontsize=9, fontweight="bold")

    fig.savefig(OUT / "architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 2. Feature transform (raw -> model input) ───────────────────────────────────
def feature_pipeline():
    fig, ax = canvas(w=15, h=8.5, title="Feature Transform  (raw landmarks -> model input)")
    steps = [
        ("landmarks_all\n(T, 1692)", C_DATA, 4, 72),
        ("geometric()\nnormalise + select\n-> coords, vis, present", C_CORE, 26, 72),
        ("[ augment ]\ntraining only:\nflip / rotate / warp", C_TRAIN, 50, 72),
        ("assemble()\nresample L + velocity\n+ masks", C_CORE, 74, 72),
    ]
    for t, c, x, y in steps:
        box(ax, x, y, 20, 13, t, c, fs=9)
    arrow(ax, (24, 78.5), (26, 78.5))
    arrow(ax, (46, 78.5), (50, 78.5))
    arrow(ax, (70, 78.5), (74, 78.5))

    box(ax, 36, 49, 28, 9, "model input  (L, D)\nL = 64   D = 346", C_INFER, fs=11)
    arrow(ax, (84, 72), (60, 58), MUTED, rad=-0.15)

    comp = [("coordinates\n165", 8), ("pose visibility\n13", 31),
            ("velocity\n165", 54), ("presence masks\n3", 77)]
    for t, x in comp:
        box(ax, x, 28, 16, 9, t, C_OUT, fs=9, bold=True)
    for x in (16, 39, 62, 85):
        arrow(ax, (50, 49), (x, 37), MUTED, lw=1.0, rad=0.0)

    ax.text(50, 18, "55 points x 3 (13 pose joints + 21 left + 21 right) + 13 vis "
            "+ 165 velocity + 3 masks", ha="center", color=MUTED, fontsize=9)
    ax.text(50, 13, "Augmentation acts in NORMALISED space (between geometric and assemble) "
            "so global scale/translation isn't cancelled away.", ha="center",
            color=MUTED, fontsize=9.5)

    fig.savefig(OUT / "feature_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 3. Pipeline stages (train -> deploy -> infer) ───────────────────────────────
def stages():
    fig, ax = canvas(w=15, h=8.5, title="Pipeline Stages")
    box(ax, 4, 70, 20, 11, "processed_dataset/\nlandmarks_all/*.npy", C_DATA, fs=9)

    stage_boxes = [
        ("Stage 1\ntrain.py\nk-fold CV", C_TRAIN, 30, 70, "cv_report.json\nconfusion_matrix.png"),
        ("Stage 2\nfinetune.py\nfull fit + EMA", C_TRAIN, 54, 70, "model_final.pt\nmodel_meta.json"),
        ("Stage 3\nexport_optimize.py\nONNX + int8", C_EXP, 78, 70, "sign_model.onnx\nsign_model.int8.onnx"),
    ]
    for t, c, x, y, out in stage_boxes:
        box(ax, x, y, 18, 11, t, c, fs=9)
        box(ax, x, y - 16, 18, 9, out, C_OUT, fs=8)
        arrow(ax, (x + 9, y), (x + 9, y - 7), MUTED, lw=1.1)
    arrow(ax, (24, 75.5), (30, 75.5), C_DATA, lw=2.0)
    arrow(ax, (48, 75.5), (54, 75.5), MUTED)
    arrow(ax, (72, 75.5), (78, 75.5), MUTED)

    ax.text(50, 40, "Inference  (shared predictor.py)", ha="center", color=MUTED, fontsize=11)
    box(ax, 14, 22, 30, 11, "Stage 4 - infer_video.py\nfile + prediction overlay", C_INFER, fs=9.5)
    box(ax, 56, 22, 30, 11, "Stage 5 - infer_live.py\nrecord -> process -> delete", C_INFER, fs=9.5)
    box(ax, 14, 8, 30, 8, "predictions/*_pred.mp4 / .json", C_OUT, fs=8.5)
    box(ax, 56, 8, 30, 8, "predictions/live_transcript.json", C_OUT, fs=8.5)
    arrow(ax, (29, 22), (29, 16), MUTED, lw=1.1)
    arrow(ax, (71, 22), (71, 16), MUTED, lw=1.1)
    # exported model feeds inference
    arrow(ax, (87, 54), (71, 33), MUTED, rad=0.2)
    arrow(ax, (87, 54), (29, 33), MUTED, rad=-0.25)
    ax.text(50, 48, "exported model + model_meta.json", ha="center", color=C_EXP,
            fontsize=9, fontweight="bold")

    fig.savefig(OUT / "stages.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture(); feature_pipeline(); stages()
    print(f"[OK] diagrams -> {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print("   ", p)
