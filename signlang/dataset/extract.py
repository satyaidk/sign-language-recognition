"""
Dataset landmark extraction (face + hands + pose).
==================================================
Walks a sign-language video dataset laid out as::

    dataset/
        <class_1>/ *.mp4
        <class_2>/ *.mp4

runs the shared ``LandmarkExtractor`` on every clip and writes, under the
processed-dataset folder::

    landmarks_pose/<class>/<stem>.npy     (T, 33, 4)     x, y, z, visibility
    landmarks_face/<class>/<stem>.npy     (T, 478, 3)    x, y, z
    landmarks_hands/<class>/<stem>.npy    (T, 2, 21, 3)  slot 0 = Left, slot 1 = Right
    landmarks_all/<class>/<stem>.npy      (T, 1692)      every modality, concatenated
    skeleton_videos/<class>/<stem>.mp4    skeleton rendered FROM the saved arrays
    metadata/<class>/<stem>.json          per-clip QA
    FEATURE_LAYOUT.json · classes.json · manifest.csv · extraction_report.json

Partial runs are safe.  ``--classes``, ``--limit`` and ``--skip-existing``
MERGE into the existing dataset files instead of overwriting them:

  * ``classes.json`` keeps every existing class -> label index and only appends
    new classes (the old version rebuilt it from the processed subset, which
    silently renumbered every label);
  * ``manifest.csv`` and ``extraction_report.json`` are upserted per clip (the
    old version wrote only the clips processed in *this* run);
  * clips skipped by ``--skip-existing`` are carried over from their metadata.

Run:
    python -m signlang extract                              # dataset/ -> processed_dataset/
    python -m signlang extract --classes hello you --limit 2
    python -m signlang extract --proc-width 720 --skip-existing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from signlang.config import add_path_args, paths_from_args
from signlang.dataset.qa import count_verdicts, mean_rates, qa_clip
from signlang.landmarks.extractor import ExtractorOptions, extract_video
from signlang.landmarks.layout import MODALITIES, build_all_vector
from signlang.utils import load_json, read_csv_rows, save_json, write_csv_rows

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}
MODALITY_DIRS = {"pose": "landmarks_pose", "face": "landmarks_face", "hands": "landmarks_hands"}
MANIFEST_FIELDS = ["class", "label", "stem", "T", "verdict", "all_npy", "skeleton"]


# ── Discovery + dataset-level bookkeeping (pure, unit-tested) ─────────────────
def discover(dataset_dir: Path, only_classes=None) -> dict:
    """``{class: [video paths]}`` for every class folder that contains videos."""
    classes = {}
    for cls_dir in sorted(p for p in Path(dataset_dir).iterdir() if p.is_dir()):
        if only_classes and cls_dir.name not in only_classes:
            continue
        vids = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if vids:
            classes[cls_dir.name] = vids
    return classes


def merge_class_index(existing: dict, classes) -> dict:
    """Keep existing class -> label indices; append unseen classes (sorted) after them."""
    merged = dict(existing)
    next_label = max(merged.values(), default=-1) + 1
    for c in sorted(classes):
        if c not in merged:
            merged[c] = next_label
            next_label += 1
    return merged


def upsert(rows: list, new_rows: list) -> list:
    """Merge per-clip rows keyed by (class, stem); new rows win."""
    by_key = {(r["class"], r["stem"]): r for r in rows}
    for r in new_rows:
        by_key[(r["class"], r["stem"])] = r
    return list(by_key.values())


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


# ── One clip ──────────────────────────────────────────────────────────────────
def process_clip(video: Path, cls: str, label: int, out: Path, opts: ExtractorOptions,
                 skeleton_size=None, extract_fn=extract_video) -> tuple:
    """Extract, save and QA one clip -> ``(manifest_row | None, report_entry, layout | None)``."""
    stem = video.stem
    res = extract_fn(video, opts)
    if res is None:
        return None, {"class": cls, "stem": stem, "T": 0, "verdict": "FAIL",
                      "detection_rate": {}, "issues": ["cannot open video"]}, None

    arrays, masks, T = res["arrays"], res["masks"], res["T"]
    for name, folder in MODALITY_DIRS.items():
        if name in arrays:
            (out / folder / cls).mkdir(parents=True, exist_ok=True)
            np.save(out / folder / cls / f"{stem}.npy", arrays[name])

    all_vec, layout = build_all_vector(arrays, opts.modalities)
    all_path = out / "landmarks_all" / cls / f"{stem}.npy"
    all_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(all_path, all_vec)

    skel = ""
    if skeleton_size:
        from signlang.landmarks.drawing import render_skeleton_video   # needs MediaPipe drawing
        skel_path = out / "skeleton_videos" / cls / f"{stem}.mp4"
        skel_path.parent.mkdir(parents=True, exist_ok=True)
        if render_skeleton_video(arrays, masks, skeleton_size, res["fps"], skel_path):
            skel = _rel(skel_path, out)

    qa = qa_clip(arrays, masks, T)
    save_json({
        "class": cls, "label": label, "stem": stem, "source": str(video),
        "source_fps": res["fps"], "frames_read": res["n_frames_read"],
        "frames_skipped": res["n_frames_skipped"], "T": T,
        "smoothed": opts.smooth, "proc_width": opts.proc_width,
        "shapes": {n: list(a.shape) for n, a in arrays.items()},
        "all_shape": list(all_vec.shape),
        "masks": {n: np.asarray(m).tolist() for n, m in masks.items()},
        "qa": qa,
    }, out / "metadata" / cls / f"{stem}.json")

    row = {"class": cls, "label": label, "stem": stem, "T": T, "verdict": qa["verdict"],
           "all_npy": _rel(all_path, out), "skeleton": skel}
    entry = {"class": cls, "stem": stem, "T": T, "verdict": qa["verdict"],
             "detection_rate": qa["detection_rate"], "issues": qa["issues"]}
    return row, entry, layout


def _carry_over(out: Path, cls: str, stem: str, label: int):
    """Rebuild manifest/report rows for a clip skipped by --skip-existing."""
    meta_path = out / "metadata" / cls / f"{stem}.json"
    if not meta_path.exists():
        return None, None
    m = load_json(meta_path)
    qa = m.get("qa", {})
    skel = out / "skeleton_videos" / cls / f"{stem}.mp4"
    row = {"class": cls, "label": label, "stem": stem, "T": m.get("T", 0),
           "verdict": qa.get("verdict", "FAIL"),
           "all_npy": f"landmarks_all/{cls}/{stem}.npy",
           "skeleton": _rel(skel, out) if skel.exists() else ""}
    entry = {"class": cls, "stem": stem, "T": m.get("T", 0), "verdict": row["verdict"],
             "detection_rate": qa.get("detection_rate", {}), "issues": qa.get("issues", [])}
    return row, entry


# ── Whole dataset ─────────────────────────────────────────────────────────────
def extract_dataset(dataset: Path, out: Path, opts: ExtractorOptions, classes=None, limit: int = 0,
                    skip_existing: bool = False, skeleton_size=(960, 540),
                    extract_fn=extract_video) -> dict:
    """Extract every clip, then merge the dataset-level files.  Returns the report."""
    dataset, out = Path(dataset), Path(out)
    if not dataset.is_dir():
        raise FileNotFoundError(f"dataset folder not found: {dataset}")
    found = discover(dataset, set(classes) if classes else None)
    if not found:
        raise FileNotFoundError(f"no class sub-folders with videos under {dataset}")
    out.mkdir(parents=True, exist_ok=True)

    class_index = merge_class_index(load_json(out / "classes.json")
                                    if (out / "classes.json").exists() else {}, found)
    total = sum(len(v[:limit] if limit else v) for v in found.values())
    print(f"[INFO] dataset : {dataset}\n[INFO] output  : {out}")
    print(f"[INFO] modalities: {', '.join(opts.modalities)}  smooth={opts.smooth}  "
          f"skeleton={bool(skeleton_size)}  proc_width={opts.proc_width or 'full'}")
    print(f"[INFO] {len(found)} classes, {total} clips\n")

    rows, entries, layout = [], [], None
    t_start, done = time.time(), 0
    for cls, vids in found.items():
        for vp in (vids[:limit] if limit else vids):
            done += 1
            label = class_index[cls]
            if skip_existing and (out / "landmarks_all" / cls / f"{vp.stem}.npy").exists():
                row, entry = _carry_over(out, cls, vp.stem, label)
                if row is not None:
                    rows.append(row)
                    entries.append(entry)
                print(f"[{done}/{total}] skip (exists)  {cls}/{vp.stem}")
                continue
            t0 = time.time()
            row, entry, clip_layout = process_clip(vp, cls, label, out, opts,
                                                   skeleton_size, extract_fn)
            entries.append(entry)
            if row is None:
                print(f"[{done}/{total}] FAIL open    {cls}/{vp.stem}")
                continue
            rows.append(row)
            layout = clip_layout
            rates = " ".join(f"{k}={v:.0%}" for k, v in entry["detection_rate"].items())
            print(f"[{done}/{total}] {entry['verdict']:4s} {cls}/{vp.stem}  T={entry['T']}  "
                  f"{rates}  ({time.time() - t0:.1f}s)")

    # ── dataset-level files (merged, never truncated) ──
    save_json(class_index, out / "classes.json")
    if layout is not None:
        layout["note"] = ("Column layout of landmarks_all/*.npy (T, total_features). "
                          "Normalised image coords: x,y in [0,1], z relative depth.")
        save_json(layout, out / "FEATURE_LAYOUT.json")

    manifest = upsert(read_csv_rows(out / "manifest.csv") if (out / "manifest.csv").exists() else [],
                      [{k: str(v) for k, v in r.items()} for r in rows])
    manifest.sort(key=lambda r: (int(r["label"]), r["stem"]))
    write_csv_rows(out / "manifest.csv", manifest, MANIFEST_FIELDS)

    old = load_json(out / "extraction_report.json") if (out / "extraction_report.json").exists() else {}
    clips = upsert(old.get("clips", []), entries)
    clips.sort(key=lambda c: (class_index.get(c["class"], 1 << 30), c["stem"]))
    report = {"clips": clips, "verdicts": count_verdicts(clips),
              "created": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": str(dataset),
              "modalities": {m: m in opts.modalities for m in MODALITIES},
              "smoothed": opts.smooth, "n_clips": len(clips),
              "mean_detection_rate": mean_rates(clips)}
    save_json(report, out / "extraction_report.json")
    _write_readme(out, class_index)

    v = report["verdicts"]
    print(f"\n[DONE] {done} clips in {time.time() - t_start:.0f}s  -> dataset now has "
          f"{report['n_clips']} clips: PASS={v['PASS']} WARN={v['WARN']} FAIL={v['FAIL']}")
    if report["mean_detection_rate"]:
        print("[DONE] mean detection rate:",
              " ".join(f"{k}={x:.0%}" for k, x in report["mean_detection_rate"].items()))
    print(f"[DONE] verify with:  python -m signlang verify --processed-dir \"{out}\"")
    return report


def _write_readme(out: Path, class_index: dict) -> None:
    txt = f"""# Processed landmark dataset

Generated by `python -m signlang extract`.

## Layout
- `landmarks_pose/<class>/<stem>.npy`  -> (T, 33, 4)   x, y, z, visibility
- `landmarks_face/<class>/<stem>.npy`  -> (T, 478, 3)  x, y, z
- `landmarks_hands/<class>/<stem>.npy` -> (T, 2, 21, 3) slot0 = Left, slot1 = Right
- `landmarks_all/<class>/<stem>.npy`   -> (T, 1692) all parts concatenated (see FEATURE_LAYOUT.json)
- `skeleton_videos/<class>/<stem>.mp4` -> black-background skeleton rendered from the arrays
- `metadata/<class>/<stem>.json`       -> per-clip QA (shapes, detection rates, masks)
- `classes.json`, `manifest.csv`, `extraction_report.json`, `verification_report.json`

## Coordinates
MediaPipe normalised image coordinates: x, y in [0, 1], z relative depth, pose
visibility in [0, 1].  Undetected landmarks are zeros and flagged in the metadata masks.

## Verify
    python -m signlang verify                                    # whole-dataset cross-check
    python -m signlang verify --clip <class>/<stem> --overlay    # eyes-on overlay video

Classes: {", ".join(class_index)}
"""
    (out / "README.md").write_text(txt, encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="signlang extract",
                                description="Extract face/hand/pose landmarks from a video dataset.")
    add_path_args(p, dataset=True, processed=True)
    p.add_argument("--classes", nargs="*", help="only these class names")
    p.add_argument("--limit", type=int, default=0, help="max clips per class (0 = all)")
    p.add_argument("--no-face", action="store_true", help="skip the face mesh")
    p.add_argument("--no-hands", action="store_true", help="skip hands")
    p.add_argument("--no-pose", action="store_true", help="skip pose")
    p.add_argument("--no-skeleton", action="store_true", help="don't render skeleton videos")
    p.add_argument("--no-smooth", action="store_true", help="store raw (unfiltered) landmarks")
    p.add_argument("--proc-width", type=int, default=0,
                   help="downscale frames to this width before detection (0 = full resolution)")
    p.add_argument("--skip-existing", action="store_true", help="resume: skip clips already extracted")
    p.add_argument("--skeleton-size", type=int, nargs=2, default=[960, 540], metavar=("W", "H"))
    return p.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    paths = paths_from_args(a)
    opts = ExtractorOptions(face=not a.no_face, hands=not a.no_hands, pose=not a.no_pose,
                            smooth=not a.no_smooth, proc_width=a.proc_width)
    if not opts.modalities:
        sys.exit("[ERROR] nothing to extract: --no-face --no-hands --no-pose all set")
    if len(opts.modalities) < 3:
        print("[WARN] training expects all three modalities (the (T, 1692) layout); "
              "a partial extraction is fine for inspection only.")
    try:
        extract_dataset(paths.dataset, paths.processed, opts, classes=a.classes, limit=a.limit,
                        skip_existing=a.skip_existing,
                        skeleton_size=None if a.no_skeleton else tuple(a.skeleton_size))
    except FileNotFoundError as exc:
        sys.exit(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
