"""
Landmark verification — prove the saved .npy files are correct.
===============================================================
Independent of what extraction reported, re-reads the saved arrays and checks:

  1. STRUCTURE    NaN/Inf, equal frame counts across modalities, face/hand
                  coordinates inside the image range.
  2. CONSISTENCY  rebuilds ``landmarks_all`` from the separate pose/face/hands
                  files (via FEATURE_LAYOUT.json) and requires an exact match,
                  so the combined vector really holds every part in order.
  3. DETECTION    recomputes per-modality detection rates from the data.
  4. OVERLAY      (optional, needs the source videos) re-projects the SAVED
                  points onto the ORIGINAL video next to the skeleton — if the
                  dots track the signer, the extraction is right.

Run:
    python -m signlang verify
    python -m signlang verify --classes hello you
    python -m signlang verify --clip hello/hello_03 --overlay
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from signlang.config import add_path_args, paths_from_args
from signlang.dataset.extract import VIDEO_EXTS
from signlang.dataset.qa import count_verdicts, load_clip_arrays, mean_rates, verify_clip
from signlang.utils import load_json, parse_clip_ref, save_json


def find_source_video(dataset: Path, cls: str, stem: str) -> Path | None:
    for p in sorted((Path(dataset) / cls).glob(f"{stem}.*")):
        if p.suffix.lower() in VIDEO_EXTS:
            return p
    return None


def make_overlay(processed: Path, dataset: Path, cls: str, stem: str) -> Path | None:
    """Side-by-side [original + saved landmarks | skeleton] video for one clip."""
    from signlang.landmarks.drawing import render_side_by_side   # needs MediaPipe drawing

    src = find_source_video(dataset, cls, stem)
    if src is None:
        print(f"[ERROR] original video for {cls}/{stem} not found under {dataset}")
        return None
    arrays = {k: v for k, v in load_clip_arrays(processed, cls, stem).items() if k != "all"}
    dst = Path(processed) / "overlay_check" / cls / f"{stem}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = render_side_by_side(src, arrays, None, dst)
    if n is None:
        print(f"[ERROR] cannot open original video: {src}")
        return None
    print(f"[OK] overlay written -> {dst}  ({n} frames)")
    return dst


def verify_dataset(processed: Path, classes=None, limit: int = 0, tol: float = 1e-5,
                   quiet: bool = False) -> dict:
    """Verify every clip under ``landmarks_all`` and write verification_report.json."""
    processed = Path(processed)
    base = processed / "landmarks_all"
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found — run `python -m signlang extract` first")
    layout_path = processed / "FEATURE_LAYOUT.json"
    layout = load_json(layout_path) if layout_path.exists() else None

    results = []
    if not quiet:
        print(f"{'verdict':7} {'clip':32} {'T':>4}  detection rates / issues")
        print("-" * 92)
    for cls_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if classes and cls_dir.name not in classes:
            continue
        clips = sorted(cls_dir.glob("*.npy"))
        for c in (clips[:limit] if limit else clips):
            rep = verify_clip(processed, cls_dir.name, c.stem, layout, tol)
            results.append(rep)
            if not quiet:
                rates = " ".join(f"{k}={v:.0%}" for k, v in rep["detection_rate"].items())
                tail = rates if rep["verdict"] == "PASS" else f"{rates}  | {'; '.join(rep['issues'])}"
                print(f"{rep['verdict']:7} {cls_dir.name + '/' + c.stem:32} {rep.get('T', 0):>4}  {tail}")

    summary = {"verdicts": count_verdicts(results), "n_clips": len(results),
               "mean_detection_rate": mean_rates(results), "clips": results}
    save_json(summary, processed / "verification_report.json")
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="signlang verify", description="Verify extracted landmark .npy files.")
    add_path_args(p, dataset=True, processed=True)
    p.add_argument("--clip", help="verify one clip: <class>/<stem>")
    p.add_argument("--overlay", action="store_true",
                   help="also render original+landmarks overlay video(s) (needs the source videos)")
    p.add_argument("--classes", nargs="*", help="only these classes")
    p.add_argument("--limit", type=int, default=0, help="max clips per class")
    p.add_argument("--tol", type=float, default=1e-5, help="tolerance for the consistency check")
    return p.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    paths = paths_from_args(a)
    if not paths.processed.is_dir():
        sys.exit(f"[ERROR] processed dataset not found: {paths.processed}")

    if a.clip:
        try:
            cls, stem = parse_clip_ref(a.clip)
        except ValueError as exc:
            sys.exit(f"[ERROR] {exc}")
        layout_path = paths.processed / "FEATURE_LAYOUT.json"
        rep = verify_clip(paths.processed, cls, stem,
                          load_json(layout_path) if layout_path.exists() else None, a.tol)
        print(json.dumps(rep, indent=2))
        if a.overlay:
            make_overlay(paths.processed, paths.dataset, cls, stem)
        return

    try:
        summary = verify_dataset(paths.processed, set(a.classes) if a.classes else None, a.limit, a.tol)
    except FileNotFoundError as exc:
        sys.exit(f"[ERROR] {exc}")
    v = summary["verdicts"]
    print("-" * 92)
    print(f"TOTAL  PASS={v['PASS']}  WARN={v['WARN']}  FAIL={v['FAIL']}   (of {summary['n_clips']} clips)")
    if summary["mean_detection_rate"]:
        print("mean detection rate:",
              " ".join(f"{k}={x:.0%}" for k, x in summary["mean_detection_rate"].items()))
    print(f"report -> {paths.processed / 'verification_report.json'}")
    if a.overlay:
        for r in summary["clips"]:
            make_overlay(paths.processed, paths.dataset, r["class"], r["stem"])
    if v["FAIL"]:
        print("\n[!] FAIL clips need attention (NaN/Inf, frame-count or consistency mismatch).")


if __name__ == "__main__":
    main()
