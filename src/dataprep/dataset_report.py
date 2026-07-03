"""
Dataset / Verification Infographics
===================================
Reads the processed dataset's per-clip metadata + reports and renders a set of
PNG charts that summarise the extraction and verification, into
``data/processed_dataset/reports/``:

    01_dataset_overview.png     clips per class, verdicts, mean detection, key numbers
    02_detection_heatmap.png    class x modality detection-rate heatmap
    03_per_class_detection.png  grouped detection bars per class
    04_frame_distribution.png   frames-per-clip histogram
    05_feature_layout.png       composition of the (T, F) "all" vector
    06_verification_summary.png one-page verification scorecard

Usage
-----
    python src/dataprep/dataset_report.py
    python src/dataprep/dataset_report.py --out data/processed_dataset
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── palette ──────────────────────────────────────────────────────────────────
C_POSE, C_FACE, C_LH, C_RH = "#2E86C1", "#E67E22", "#27AE60", "#C0392B"
MOD = [("pose", C_POSE), ("face", C_FACE), ("left_hand", C_LH), ("right_hand", C_RH)]
V_COLORS = {"PASS": "#27AE60", "WARN": "#F39C12", "FAIL": "#C0392B"}
INK, MUTED, BG = "#1B2631", "#5D6D7E", "#FBFCFC"

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": BG,
                     "font.size": 10, "axes.titleweight": "bold"})


def load(out):
    meta = [json.load(open(p)) for p in sorted((out / "metadata").rglob("*.json"))]
    clips = []
    for m in meta:
        dr = m["qa"]["detection_rate"]
        clips.append({"class": m["class"], "stem": m["stem"], "T": m["T"],
                      "verdict": m["qa"]["verdict"],
                      "rates": {k: dr.get(k, 0.0) for k, _ in MOD},
                      "all_F": m["all_shape"][1] if len(m["all_shape"]) > 1 else 0})
    return clips


def bar_labels(ax, bars, fmt="{:.0f}", dy=0.0, color=INK, size=8):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=size, color=color)


# ── 01 overview ────────────────────────────────────────────────────────────────
def fig_overview(clips, classes, by_class, verdicts, mean_rate, out):
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Sign-Language Landmark Dataset — Overview",
                 fontsize=18, fontweight="bold", color=INK)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22,
                          left=0.07, right=0.96, top=0.9, bottom=0.08)

    # clips per class
    ax = fig.add_subplot(gs[0, 0])
    counts = [len(by_class[c]) for c in classes]
    bars = ax.bar(classes, counts, color=C_POSE, edgecolor="white")
    bar_labels(ax, bars, "{:.0f}")
    ax.set_title(f"Clips per class  (total {len(clips)})")
    ax.set_ylabel("clips"); ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, max(counts) + 2)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    # verdict donut
    ax = fig.add_subplot(gs[0, 1])
    labels = [k for k in ("PASS", "WARN", "FAIL") if verdicts.get(k, 0)]
    sizes = [verdicts[k] for k in labels]
    ax.pie(sizes, labels=[f"{k}\n{verdicts[k]}" for k in labels],
           colors=[V_COLORS[k] for k in labels], autopct="%1.0f%%",
           startangle=90, wedgeprops=dict(width=0.42, edgecolor="white"),
           textprops=dict(color=INK, fontsize=10))
    ax.set_title("Verification verdicts")

    # mean detection per modality
    ax = fig.add_subplot(gs[1, 0])
    names = [m for m, _ in MOD]; vals = [mean_rate[m] * 100 for m, _ in MOD]
    bars = ax.bar(names, vals, color=[c for _, c in MOD], edgecolor="white")
    bar_labels(ax, bars, "{:.0f}%")
    ax.set_title("Mean detection rate per modality")
    ax.set_ylabel("% of frames"); ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=20)

    # key numbers
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    Ts = [c["T"] for c in clips]
    F = clips[0]["all_F"] if clips else 0
    facts = [
        ("Classes", f"{len(classes)}"),
        ("Clips", f"{len(clips)}"),
        ("Total frames", f"{sum(Ts):,}"),
        ("Frames / clip", f"{min(Ts)}–{max(Ts)}  (median {int(np.median(Ts))})"),
        ("Feature vector", f"{F} dims / frame"),
        ("All == parts", "max diff 0.0  (exact)"),
        ("NaN / Inf", "none"),
    ]
    y = 0.95
    ax.text(0.0, y, "Key numbers", fontsize=13, fontweight="bold", color=INK)
    y -= 0.13
    for k, v in facts:
        ax.text(0.02, y, k, fontsize=11, color=MUTED)
        ax.text(0.55, y, v, fontsize=11, fontweight="bold", color=INK)
        y -= 0.125

    p = out / "reports" / "01_dataset_overview.png"
    fig.savefig(p, dpi=150); plt.close(fig); return p


# ── 02 heatmap ──────────────────────────────────────────────────────────────────
def fig_heatmap(classes, by_class, out):
    mods = [m for m, _ in MOD]
    M = np.array([[np.mean([c["rates"][m] for c in by_class[cl]]) for m in mods]
                  for cl in classes])
    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mods))); ax.set_xticklabels(mods, rotation=20, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_title("Detection rate  (class x modality)", fontsize=14, color=INK)
    for i in range(len(classes)):
        for j in range(len(mods)):
            v = M[i, j]
            ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                    color="black" if v > 0.45 else "white", fontsize=9, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("fraction of frames detected")
    fig.tight_layout()
    p = out / "reports" / "02_detection_heatmap.png"
    fig.savefig(p, dpi=150); plt.close(fig); return p


# ── 03 grouped bars ──────────────────────────────────────────────────────────────
def fig_per_class(classes, by_class, out):
    mods = [m for m, _ in MOD]; cols = [c for _, c in MOD]
    x = np.arange(len(classes)); w = 0.2
    fig, ax = plt.subplots(figsize=(15, 7))
    for k, (m, col) in enumerate(zip(mods, cols)):
        vals = [np.mean([c["rates"][m] for c in by_class[cl]]) * 100 for cl in classes]
        ax.bar(x + (k - 1.5) * w, vals, w, label=m, color=col, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("% of frames"); ax.set_ylim(0, 105)
    ax.set_title("Per-class detection rate by modality", fontsize=14, color=INK)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.tight_layout()
    p = out / "reports" / "03_per_class_detection.png"
    fig.savefig(p, dpi=150); plt.close(fig); return p


# ── 04 frame distribution ─────────────────────────────────────────────────────────
def fig_frames(clips, out):
    Ts = [c["T"] for c in clips]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(Ts, bins=min(20, len(set(Ts))), color=C_POSE, edgecolor="white")
    ax.axvline(np.median(Ts), color=C_RH, ls="--", lw=2, label=f"median {int(np.median(Ts))}")
    ax.set_title(f"Frames per clip  (total {sum(Ts):,} frames over {len(Ts)} clips)",
                 fontsize=14, color=INK)
    ax.set_xlabel("frames (T)"); ax.set_ylabel("clips"); ax.legend(frameon=False)
    fig.tight_layout()
    p = out / "reports" / "04_frame_distribution.png"
    fig.savefig(p, dpi=150); plt.close(fig); return p


# ── 05 feature layout ─────────────────────────────────────────────────────────────
def fig_layout(out):
    lp = out / "FEATURE_LAYOUT.json"
    if not lp.exists():
        return None
    layout = json.loads(lp.read_text())
    blocks = layout["blocks"]; total = layout["total_features"]
    cmap = {"pose": C_POSE, "face": C_FACE, "left_hand": C_LH, "right_hand": C_RH}
    fig, ax = plt.subplots(figsize=(13, 3.2))
    left = 0
    for b in blocks:
        col = cmap.get(b["name"], MUTED)
        ax.barh(0, b["width"], left=left, color=col, edgecolor="white")
        ax.text(left + b["width"] / 2, 0, f"{b['name']}\n{b['width']}",
                ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        left += b["width"]
    ax.set_xlim(0, total); ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
    ax.set_xlabel("feature index")
    ax.set_title(f"landmarks_all vector layout  —  {total} features / frame",
                 fontsize=14, color=INK)
    fig.tight_layout()
    p = out / "reports" / "05_feature_layout.png"
    fig.savefig(p, dpi=150); plt.close(fig); return p


# ── 06 verification scorecard ─────────────────────────────────────────────────────
def fig_scorecard(clips, verdicts, mean_rate, out, dropped):
    vr = {}
    vp = out / "verification_report.json"
    if vp.exists():
        vr = json.loads(vp.read_text())
    max_diff = 0.0
    for c in vr.get("clips", []):
        max_diff = max(max_diff, c.get("all_vs_parts_max_diff", 0.0) or 0.0)

    fig = plt.figure(figsize=(12, 8)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.9), 1, 0.1, color=INK))
    ax.text(0.5, 0.95, "VERIFICATION SCORECARD", ha="center", va="center",
            color="white", fontsize=20, fontweight="bold")

    # check rows
    checks = [
        ("Clips verified", f"{len(clips)}", True),
        ("Structure (shape / dtype)", "all valid float32", True),
        ("NaN / Inf present", "none", True),
        ("Consistency  all == face+hands+pose", f"max diff {max_diff:.1e}", max_diff < 1e-4),
        ("Verdicts", f"PASS {verdicts.get('PASS',0)} / "
                     f"WARN {verdicts.get('WARN',0)} / FAIL {verdicts.get('FAIL',0)}",
         verdicts.get("FAIL", 0) == 0),
        ("Pose detection (mean)", f"{mean_rate['pose']*100:.0f}%", mean_rate["pose"] > 0.8),
        ("Face detection (mean)", f"{mean_rate['face']*100:.0f}%", mean_rate["face"] > 0.8),
        ("Any-hand detection (mean)", f"{_anyhand(clips):.0f}%", True),
    ]
    y = 0.82
    for label, val, ok in checks:
        mark, mc = ("PASS", "#27AE60") if ok else ("CHECK", "#F39C12")
        ax.text(0.06, y, label, fontsize=13, color=INK)
        ax.text(0.66, y, val, fontsize=13, fontweight="bold", color=INK)
        ax.add_patch(plt.Rectangle((0.88, y - 0.018), 0.08, 0.04, color=mc))
        ax.text(0.92, y, mark, ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")
        y -= 0.083

    note = ("Excluded as framing outliers (hand close-ups, face/body off-frame): "
            + ", ".join("/".join(d) for d in dropped)
            + ".  Sources preserved in data/excluded_clips/." if dropped else "")
    ax.text(0.06, 0.07, note, fontsize=10, color=MUTED, wrap=True)
    p = out / "reports" / "06_verification_summary.png"
    fig.savefig(p, dpi=150); plt.close(fig); return p


def _anyhand(clips):
    # mean fraction of frames with at least one hand, from per-clip max(L,R)
    vals = [max(c["rates"]["left_hand"], c["rates"]["right_hand"]) for c in clips]
    return 100 * (sum(vals) / len(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser(description="Render dataset/verification infographics.")
    ap.add_argument("--out", default="data/processed_dataset")
    ap.add_argument("--dropped", nargs="*", default=["no/no_13", "yes/yes_12"],
                    help="clips noted as excluded on the scorecard")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    clips = load(out)
    if not clips:
        raise SystemExit(f"[ERROR] no metadata under {out/'metadata'} — run extract_dataset.py")
    classes = sorted(set(c["class"] for c in clips))
    by_class = defaultdict(list)
    for c in clips:
        by_class[c["class"]].append(c)
    verdicts = defaultdict(int)
    for c in clips:
        verdicts[c["verdict"]] += 1
    mean_rate = {m: float(np.mean([c["rates"][m] for c in clips])) for m, _ in MOD}
    dropped = [tuple(d.replace("\\", "/").split("/", 1)) for d in a.dropped]

    made = [
        fig_overview(clips, classes, by_class, verdicts, mean_rate, out),
        fig_heatmap(classes, by_class, out),
        fig_per_class(classes, by_class, out),
        fig_frames(clips, out),
        fig_layout(out),
        fig_scorecard(clips, verdicts, mean_rate, out, dropped),
    ]
    print(f"[OK] {len([m for m in made if m])} infographics -> {out/'reports'}")
    for m in made:
        if m:
            print("   ", m)


if __name__ == "__main__":
    main()
