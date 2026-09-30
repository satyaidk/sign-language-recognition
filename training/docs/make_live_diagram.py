"""
Render the live-fix diagram (motion-gated segmentation state machine + the
motion-vs-time trace that drives it) to diagrams/live_segmentation.png.

    python training/docs/make_live_diagram.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED = "#1B2631", "#5D6D7E"
C_IDLE, C_REC, C_EMIT = "#5D6D7E", "#C0392B", "#16A085"
C_START, C_STOP, C_MOT = "#C0392B", "#E67E22", "#2E86C1"


def _box(ax, x, y, w, h, text, color, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2",
                                linewidth=1.4, edgecolor="white", facecolor=color, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white",
            fontsize=fs, fontweight="bold", zorder=3)


def _arrow(ax, p0, p1, color=INK, rad=0.0, lw=1.8):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=16, lw=lw,
                                 color=color, zorder=1))


def state_machine(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 95, "Motion-gated segmentation — state machine", ha="center",
            va="top", fontsize=15, fontweight="bold", color=INK)

    _box(ax, 8, 45, 30, 18, "LISTENING\n(idle, not classified)", C_IDLE, fs=12)
    _box(ax, 62, 45, 30, 18, "RECORDING\n(buffering the sign)", C_REC, fs=12)

    _arrow(ax, (38, 58), (62, 58), C_REC, rad=-0.25)
    ax.text(50, 71, "motion > start_threshold\n(open + pre-roll)", ha="center",
            color=C_REC, fontsize=9.5, fontweight="bold")

    _arrow(ax, (62, 50), (38, 50), C_IDLE, rad=-0.25)
    ax.text(50, 36, "still for the still-time\n(or max-sign) -> close segment",
            ha="center", color=MUTED, fontsize=9.5, fontweight="bold")

    # self-loops
    ax.text(23, 40, "learn noise floor", ha="center", color=MUTED, fontsize=8.5)
    ax.text(77, 40, "count still frames", ha="center", color=MUTED, fontsize=8.5)

    _box(ax, 62, 12, 30, 14, "classify segment\n(predict_robust + margin)\n-> dedup -> emit",
         C_EMIT, fs=9.5)
    _arrow(ax, (77, 45), (77, 26), C_EMIT)
    _box(ax, 8, 12, 30, 14, "record -> temp .mp4\nextract -> model -> DELETE\n(verified cache path)",
         "#34495E", fs=9)
    _arrow(ax, (62, 19), (38, 19), MUTED, rad=0.0)


def motion_trace(ax):
    rng = np.random.default_rng(3)
    fps = 30
    t = np.arange(0, 6, 1 / fps)
    noise = 0.0025
    m = noise + rng.normal(0, 0.0006, t.size).clip(min=0)
    # one sign burst between 1.4s and 3.6s, plus pre/idle/idle tails
    burst = (t > 1.4) & (t < 3.6)
    m[burst] += 0.03 * np.sin((t[burst] - 1.4) / 2.2 * np.pi) ** 2 + 0.012
    m = np.convolve(m, np.ones(3) / 3, mode="same")

    start_th, stop_th = noise * 2.6, noise * 1.7
    ax.plot(t, m, color=C_MOT, lw=1.8, label="motion energy")
    ax.axhline(start_th, color=C_START, ls="--", lw=1.3, label="start threshold")
    ax.axhline(stop_th, color=C_STOP, ls=":", lw=1.3, label="stop threshold")

    # recording span (with pre-roll) and emit point
    rec0, rec1 = 1.4 - 0.25, 3.6 + 0.5
    ax.axvspan(rec0, rec1, color=C_REC, alpha=0.10)
    ax.axvspan(rec0, 1.4, color=C_EMIT, alpha=0.12)
    ax.annotate("pre-roll", (rec0 + 0.12, m.max() * 0.9), color=C_EMIT, fontsize=8.5)
    ax.annotate("RECORDING", ((1.4 + 3.6) / 2, m.max() * 1.02), color=C_REC,
                fontsize=10, fontweight="bold", ha="center")
    ax.annotate("still -> END\n(emit sign)", (rec1 + 0.05, stop_th + 0.004),
                color=INK, fontsize=8.5)
    ax.scatter([rec1], [m[np.argmin(np.abs(t - rec1))]], color=C_EMIT, zorder=5, s=30)

    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_ylabel("motion", fontsize=9)
    ax.set_title("What drives it — record only while the sign is moving",
                 fontsize=12, color=INK, fontweight="bold")
    ax.set_ylim(0, m.max() * 1.18)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    fig = plt.figure(figsize=(12, 9))
    state_machine(fig.add_axes([0.02, 0.52, 0.96, 0.46]))
    motion_trace(fig.add_axes([0.10, 0.07, 0.82, 0.36]))
    out = OUT / "live_segmentation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] diagram -> {out}")


if __name__ == "__main__":
    main()
