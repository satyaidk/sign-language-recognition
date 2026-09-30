"""
Render the documentation diagrams to docs/diagrams/*.png (pure matplotlib).

    python docs/make_diagrams.py

    architecture.png   the signlang package: four layers + CLI
    pipeline.png       end-to-end data flow, videos -> deployed model -> captions
    live_pipeline.png  streaming live recognition, frame by frame
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "diagrams"
INK, MUTED = "#1B2631", "#5D6D7E"
C_L0, C_L1, C_L2, C_L3, C_DATA, C_OUT = "#16A085", "#2E86C1", "#8E44AD", "#C0392B", "#7F8C8D", "#E67E22"


def box(ax, x, y, w, h, text, color, fg="white", fs=10, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2",
                                linewidth=1.2, edgecolor="white", facecolor=color, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=fg, fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, p0, p1, color=INK, lw=1.6, rad=0.0, text=None):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                                      connectionstyle=f"arc3,rad={rad}"), zorder=1)
    if text:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 1.5, text, ha="center", fontsize=8.5, color=MUTED)


def canvas(w=14, h=9, title=""):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    if title:
        ax.text(50, 98, title, ha="center", va="top", fontsize=17, fontweight="bold", color=INK)
    return fig, ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ── 1. package architecture ───────────────────────────────────────────────────
def architecture():
    fig, ax = canvas(14, 9.5, "signlang — layered architecture")
    box(ax, 2, 84, 96, 7, "python -m signlang  <view | extract | verify | prune | report | observe | "
        "train | finetune | export | video | live>", INK, fs=10.5)

    layers = [
        (66, C_L3, "L3  inference", [("predictor.py", "ONNX / torch model\n+ robust window voting"),
                                     ("segmenter.py", "motion-gated segments\n+ repeat de-dup"),
                                     ("video.py", "stage 4: file\n+ prediction overlay"),
                                     ("live.py", "stage 5: streaming\nwebcam recognition")]),
        (46, C_L2, "L2  training", [("features.py", "raw (T,1692) -> (64,346)\nshared train + serve"),
                                    ("data.py + model.py", "augmentation\nBiGRU / Transformer"),
                                    ("train.py / finetune.py", "stage 1 k-fold CV\nstage 2 full fit + EMA"),
                                    ("export.py", "stage 3 ONNX\nparity + latency")]),
        (26, C_L1, "L1  dataset", [("extract.py", "videos -> .npy +\nskeletons + manifest"),
                                   ("qa.py / verify.py", "PASS/WARN/FAIL\nall == parts check"),
                                   ("prune.py / report.py", "reversible drops\ninfographics"),
                                   ("observe.py", "continuous video\ndetection timeline")]),
        (6, C_L0, "L0  landmarks", [("extractor.py", "persistent MediaPipe\nframe -> landmarks"),
                                    ("smoothing.py", "One-Euro filter\n(pure NumPy)"),
                                    ("hands.py / layout.py", "anatomical hand slots\n1692-column layout"),
                                    ("drawing.py / viewer.py", "skeleton rendering\nlive viewer")]),
    ]
    for y, color, title, mods in layers:
        box(ax, 2, y, 14, 14, title, color, fs=11)
        for i, (name, desc) in enumerate(mods):
            box(ax, 18 + i * 20.3, y, 18.8, 14, f"{name}\n{desc}", color, fs=8.6)
    for y in (20, 40, 60, 80):
        arrow(ax, (50, y + 0.2), (50, y + 5.6), color=MUTED, lw=1.2)
    ax.text(50, 1.5, "config.py (paths + all hyper-parameters)  ·  utils.py (folds, metrics, IO)  ·  "
            "tests/ (pytest on synthetic data - no private data needed)", ha="center", fontsize=9.5, color=MUTED)
    save(fig, "architecture.png")


# ── 2. end-to-end data flow ───────────────────────────────────────────────────
def pipeline():
    fig, ax = canvas(15, 7.5, "End-to-end pipeline: sign videos -> deployed model -> live captions")
    y = 58
    steps = [(1, "dataset/\n<class>/*.mp4", C_DATA), (15.5, "extract\nMediaPipe +\nOne-Euro", C_L1),
             (30, "processed_\ndataset/ (T,1692)\n.npy + QA\n+ manifest", C_OUT), (44.5, "verify\n+ report", C_L1),
             (59, "train\n5-fold CV\n(honest acc.)", C_L2), (73.5, "finetune\n100% data\n+ EMA", C_L2),
             (88, "export\nONNX\n(~1 ms/clip)", C_L2)]
    for x, t, c in steps:
        box(ax, x, y, 11.5, 20, t, c, fs=9.3)
    for (x0, _, _), (x1, _, _) in zip(steps, steps[1:]):
        arrow(ax, (x0 + 11.9, y + 10), (x1 - 0.4, y + 10))

    box(ax, 30, 20, 18, 16, "video\nfile -> prediction\n+ timeline overlay", C_L3, fs=9.5)
    box(ax, 58, 20, 18, 16, "live\nwebcam -> captions\n+ session log", C_L3, fs=9.5)
    box(ax, 84, 20, 15, 16, "artifacts/exported/\nsign_model.onnx\nmodel_meta.json", C_OUT, fs=8.8)
    arrow(ax, (93.5, y - 0.4), (91.5, 36.4))
    arrow(ax, (84, 28), (76.4, 28))
    arrow(ax, (91.5, 19.6), (39, 19.6), rad=-0.25)
    ax.text(50, 88, "features.to_features() is the SAME function in training, file inference and live "
            "inference  ->  no train/serve skew", ha="center", fontsize=10.5, color=INK, fontweight="bold")
    save(fig, "pipeline.png")


# ── 3. streaming live recognition ─────────────────────────────────────────────
def live_pipeline():
    fig, ax = canvas(15, 7.8, "Live recognition: streaming landmarks + motion-gated segments")
    y = 60
    row = [(1, "camera frame\n(30 fps)", C_DATA),
           (19, "LandmarkExtractor\npersistent models\n~25 ms / frame", C_L0),
           (37, "landmark row\n(1692 floats)\n-> segment buffer", C_OUT),
           (55, "MotionSegmenter\nidle / recording\nstillness = end", C_L3)]
    for x, t, c in row:
        box(ax, x, y, 13.5, 20, t, c, fs=9.1)
    for (x0, _, _), (x1, _, _) in zip(row, row[1:]):
        arrow(ax, (x0 + 13.9, y + 10), (x1 - 0.4, y + 10))
    box(ax, 73, y, 25, 20, "sign finished\n-> (T, 1692) already in memory\n(no temp video, no re-extraction)",
        C_OUT, fs=9)
    arrow(ax, (68.9, y + 10), (72.6, y + 10))

    low = [(72, "predict_robust\nwhole clip + windows\n~2 ms", C_L2),
           (50, "gates\nhands present?\nconfidence? margin?", C_L3),
           (28, "SignDebouncer\nrepeat within 4 s\n-> suppressed", C_L3),
           (6, "caption + transcript\n+ sessions/<time>.json", C_OUT)]
    for x, t, c in low:
        box(ax, x, 22, 18, 20, t, c, fs=9.2)
    arrow(ax, (85, y - 0.4), (81, 42.4))
    for (x0, _, _), (x1, _, _) in zip(low, low[1:]):
        arrow(ax, (x0 - 0.4, 32), (x1 + 18.4, 32))
    ax.text(50, 8, 'uncertain -> "?" (abstain: no-hands / low-conf / tie)      '
            "the preview never waits for the model", ha="center", fontsize=10.5, color=INK, fontweight="bold")
    save(fig, "live_pipeline.png")


if __name__ == "__main__":
    architecture()
    pipeline()
    live_pipeline()
