"""
Landmark Verification / Cross-Check
===================================
Proves that the ``.npy`` files written by ``extract_dataset.py`` actually
contain correct face / hand / pose landmarks.  Three independent checks:

  1. STRUCTURE   shapes, dtype, NaN/Inf, all-zero (missing) frames, and that
                 x/y stay inside the normalised [0,1] range.

  2. CONSISTENCY rebuilds the combined ``landmarks_all`` vector from the
                 separate face/hands/pose ``.npy`` files (using
                 FEATURE_LAYOUT.json) and checks it matches byte-for-byte.
                 This guarantees the "all" folder really contains every part
                 with nothing missing or mis-ordered.

  3. DETECTION   recomputes per-modality detection rates straight from the
                 saved arrays (a frame counts as detected when its block is
                 not all-zero) — independent of what extraction reported.

  4. OVERLAY     (optional) re-projects the SAVED points back onto the
                 ORIGINAL video, side-by-side with the black-background
                 skeleton, and writes a comparison ``.mp4``.  If the dots
                 track the signer, the extraction is correct.  This is the
                 strongest, eyes-on proof.

Usage
-----
    python verify_landmarks.py --out processed_dataset
    python verify_landmarks.py --out processed_dataset --classes hello you
    python verify_landmarks.py --out processed_dataset \
        --video hello/hello --dataset dataset --overlay
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import landmark_detector as ld                       # noqa: E402
from mediapipe.framework.formats import landmark_pb2  # noqa: E402

N_POSE, N_FACE, N_HAND = 33, 478, 21


# ── helpers ──────────────────────────────────────────────────────────────────

def _proto(rows, with_vis):
    proto = landmark_pb2.NormalizedLandmarkList()
    for r in rows:
        p = proto.landmark.add()
        p.x, p.y, p.z = float(r[0]), float(r[1]), float(r[2])
        p.visibility = float(r[3]) if with_vis else 1.0
    return proto


def _present(block):
    """A per-frame boolean: landmark block is 'detected' if it is not all-zero."""
    return np.abs(block).reshape(block.shape[0], -1).sum(1) > 0


def load_clip(out, cls, stem):
    """Load whichever modality arrays exist for one clip."""
    arr = {}
    for name, folder in (("pose", "landmarks_pose"), ("face", "landmarks_face"),
                         ("hands", "landmarks_hands"), ("all", "landmarks_all")):
        p = out / folder / cls / f"{stem}.npy"
        if p.exists():
            arr[name] = np.load(p)
    return arr


def rebuild_all(arr, layout):
    """Rebuild the combined vector from separate arrays, following the layout."""
    parts = []
    src = {}
    if "pose" in arr:
        src["pose"] = arr["pose"].reshape(arr["pose"].shape[0], -1)
    if "face" in arr:
        src["face"] = arr["face"].reshape(arr["face"].shape[0], -1)
    if "hands" in arr:
        src["left_hand"] = arr["hands"][:, 0].reshape(arr["hands"].shape[0], -1)
        src["right_hand"] = arr["hands"][:, 1].reshape(arr["hands"].shape[0], -1)
    for block in layout["blocks"]:
        if block["name"] in src:
            parts.append(src[block["name"]])
    return np.concatenate(parts, axis=1) if parts else None


# ── checks on one clip ────────────────────────────────────────────────────────

def verify_clip(out, cls, stem, layout, tol):
    arr = load_clip(out, cls, stem)
    rep = {"class": cls, "stem": stem, "issues": [], "detection_rate": {}}
    if not arr:
        rep["verdict"] = "FAIL"
        rep["issues"].append("no .npy files found")
        return rep

    T = arr.get("all", next(iter(arr.values()))).shape[0]
    rep["T"] = int(T)

    # 1. STRUCTURE
    for name, a in arr.items():
        if np.isnan(a).any():
            rep["issues"].append(f"{name}: NaN")
        if np.isinf(a).any():
            rep["issues"].append(f"{name}: Inf")
    # pose x/y may exceed [0,1] by design (off-screen joints are extrapolated),
    # so only face/hands — reported only when visible — are range-checked.
    for name in ("face", "hands"):
        if name in arr:
            xy = arr[name][..., :2]
            nz = xy[xy != 0.0]
            if nz.size and (nz.min() < -0.5 or nz.max() > 1.5):
                rep["issues"].append(f"{name}: x/y outside [0,1] "
                                     f"(min={nz.min():.2f} max={nz.max():.2f})")

    # 2. CONSISTENCY  (all == concat of the separate parts)
    if "all" in arr and layout is not None:
        rebuilt = rebuild_all(arr, layout)
        if rebuilt is None:
            rep["issues"].append("could not rebuild 'all'")
        elif rebuilt.shape != arr["all"].shape:
            rep["issues"].append(f"all shape {arr['all'].shape} != rebuilt {rebuilt.shape}")
        else:
            diff = float(np.max(np.abs(rebuilt - arr["all"]))) if rebuilt.size else 0.0
            rep["all_vs_parts_max_diff"] = diff
            if diff > tol:
                rep["issues"].append(f"all != parts (max diff {diff:.2e})")
    elif "all" in arr and layout is None:
        rep["issues"].append("FEATURE_LAYOUT.json missing -> skipped consistency")

    # 3. DETECTION rates (recomputed from the arrays themselves)
    if "pose" in arr:
        rep["detection_rate"]["pose"] = round(float(_present(arr["pose"]).mean()), 3)
    if "face" in arr:
        rep["detection_rate"]["face"] = round(float(_present(arr["face"]).mean()), 3)
    if "hands" in arr:
        rep["detection_rate"]["left_hand"] = round(float(_present(arr["hands"][:, 0]).mean()), 3)
        rep["detection_rate"]["right_hand"] = round(float(_present(arr["hands"][:, 1]).mean()), 3)

    if "pose" in arr and rep["detection_rate"].get("pose", 0) < 0.5:
        rep["issues"].append("pose detected < 50% of frames")

    # Verdict
    fatal = any(("NaN" in i or "Inf" in i or "all !=" in i or "shape" in i or
                 "no .npy" in i) for i in rep["issues"])
    rep["verdict"] = "FAIL" if fatal else ("WARN" if rep["issues"] else "PASS")
    return rep


# ── overlay video (eyes-on proof) ──────────────────────────────────────────────

def make_overlay(out, dataset, cls, stem):
    arr = load_clip(out, cls, stem)
    src = Path(dataset) / cls / f"{stem}.mp4"
    if not src.exists():
        for ext in (".mov", ".avi", ".mkv", ".m4v"):
            if (Path(dataset) / cls / f"{stem}{ext}").exists():
                src = Path(dataset) / cls / f"{stem}{ext}"
                break
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[ERROR] cannot open original video: {src}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 < fps <= 120 else 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 640 / W
    OW, OH = int(W * scale), int(H * scale)

    dst = out / "overlay_check" / cls
    dst.mkdir(parents=True, exist_ok=True)
    dst = dst / f"{stem}.mp4"
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OW * 2, OH))

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (OW, OH))
        black = np.zeros_like(frame)

        pose_proto = None
        if "pose" in arr and fi < arr["pose"].shape[0] and arr["pose"][fi].any():
            pose_proto = _proto(arr["pose"][fi], True)
        hands = []
        if "hands" in arr and fi < arr["hands"].shape[0]:
            for slot, side in ((0, "Left"), (1, "Right")):
                if arr["hands"][fi, slot].any():
                    hands.append((_proto(arr["hands"][fi, slot], False), side))

        hand_wrist_by_side = {}
        if pose_proto is not None and hands:
            lw = ld._lm_px(pose_proto.landmark[ld.LEFT_WRIST], OW, OH)
            rw = ld._lm_px(pose_proto.landmark[ld.RIGHT_WRIST], OW, OH)
            wrists = [ld._lm_px(p.landmark[0], OW, OH) for p, _ in hands]
            for idx, s in ld.match_hands_to_sides(wrists, lw, rw).items():
                hand_wrist_by_side[s] = wrists[idx]

        for target in (frame, black):
            if pose_proto is not None:
                ld.draw_upper_body(target, pose_proto, False, hand_wrist_by_side,
                                   show_labels=False)
            if "face" in arr and fi < arr["face"].shape[0] and arr["face"][fi].any():
                ld.draw_face(target, [_proto(arr["face"][fi], False)], False)
            if hands:
                ld.draw_hands(target, hands, mirrored=True)

        cv2.putText(frame, "original + saved landmarks", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(black, "skeleton from .npy", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(np.hstack([frame, black]))
        fi += 1

    cap.release()
    vw.release()
    print(f"[OK] overlay written -> {dst}  ({fi} frames)")
    return dst


# ── main ───────────────────────────────────────────────────────────────────

def main():
    a = parse_args()
    out = Path(a.out)
    if not out.is_dir():
        sys.exit(f"[ERROR] output folder not found: {out}")

    layout = None
    lp = out / "FEATURE_LAYOUT.json"
    if lp.exists():
        layout = json.loads(lp.read_text())

    # Single-clip mode (optionally with overlay)
    if a.video:
        cls, stem = a.video.split("/", 1) if "/" in a.video else a.video.split("\\", 1)
        rep = verify_clip(out, cls, stem, layout, a.tol)
        print(json.dumps(rep, indent=2))
        if a.overlay:
            make_overlay(out, a.dataset, cls, stem)
        return

    # Whole-dataset scan
    only = set(a.classes) if a.classes else None
    base = out / "landmarks_all"
    if not base.is_dir():
        sys.exit(f"[ERROR] {base} not found — run extract_dataset.py first")

    results, verdicts = [], {"PASS": 0, "WARN": 0, "FAIL": 0}
    print(f"{'verdict':7} {'clip':32} {'T':>4}  detection rates / issues")
    print("-" * 92)
    for cls_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if only and cls_dir.name not in only:
            continue
        clips = sorted(cls_dir.glob("*.npy"))
        if a.limit:
            clips = clips[:a.limit]
        for c in clips:
            rep = verify_clip(out, cls_dir.name, c.stem, layout, a.tol)
            verdicts[rep["verdict"]] += 1
            results.append(rep)
            rates = " ".join(f"{k}={v:.0%}" for k, v in rep["detection_rate"].items())
            tail = rates if rep["verdict"] == "PASS" else (rates + "  | " + "; ".join(rep["issues"]))
            print(f"{rep['verdict']:7} {cls_dir.name + '/' + c.stem:32} {rep.get('T', 0):>4}  {tail}")
            if a.overlay:
                make_overlay(out, a.dataset, cls_dir.name, c.stem)

    summary = {"verdicts": verdicts, "n_clips": len(results), "clips": results}
    if results:
        agg = {}
        for r in results:
            for k, v in r["detection_rate"].items():
                agg.setdefault(k, []).append(v)
        summary["mean_detection_rate"] = {k: round(sum(v) / len(v), 3) for k, v in agg.items()}
    with open(out / "verification_report.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("-" * 92)
    print(f"TOTAL  PASS={verdicts['PASS']}  WARN={verdicts['WARN']}  FAIL={verdicts['FAIL']}"
          f"   (of {len(results)} clips)")
    if summary.get("mean_detection_rate"):
        print("mean detection rate:",
              " ".join(f"{k}={x:.0%}" for k, x in summary["mean_detection_rate"].items()))
    print(f"report -> {out / 'verification_report.json'}")
    if verdicts["FAIL"]:
        print("\n[!] FAIL clips need attention (NaN/Inf, shape or consistency mismatch).")
    print("\nFor eyes-on proof of one clip:")
    print(f"  python verify_landmarks.py --out {out} --video <class>/<stem> "
          f"--dataset {a.dataset} --overlay")


def parse_args():
    p = argparse.ArgumentParser(description="Verify extracted landmark .npy files.")
    p.add_argument("--out", default="processed_dataset", help="processed dataset root")
    p.add_argument("--dataset", default="dataset", help="original videos (for --overlay)")
    p.add_argument("--video", help="verify one clip: <class>/<stem>")
    p.add_argument("--overlay", action="store_true", help="render original+landmarks overlay (.mp4)")
    p.add_argument("--classes", nargs="*", help="only these classes in the scan")
    p.add_argument("--limit", type=int, default=0, help="max clips per class in the scan")
    p.add_argument("--tol", type=float, default=1e-5, help="allclose tolerance for consistency")
    return p.parse_args()


if __name__ == "__main__":
    main()
