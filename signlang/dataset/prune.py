"""
Prune clips from the dataset — reversibly.
==========================================
For each ``<class>/<stem>``:

  * MOVES the source video from ``dataset/<class>/`` to ``excluded_clips/<class>/``
    (kept, just no longer picked up by extraction);
  * DELETES the clip's derived files (4 landmark .npy, skeleton + overlay video,
    metadata json);
  * REFRESHES manifest.csv, extraction_report.json and verification_report.json.

Clips must be named explicitly — the old version pruned two hard-coded clips
when run without arguments.  ``--dry-run`` shows what would happen.

Run:
    python -m signlang prune no/no_13 yes/yes_12
    python -m signlang prune hi/hi_03 --dry-run

Undo: move the video back from excluded_clips/ and re-run extraction for it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from signlang.config import add_path_args, paths_from_args
from signlang.dataset.extract import VIDEO_EXTS
from signlang.dataset.qa import count_verdicts, mean_rates
from signlang.utils import load_json, parse_clip_ref, read_csv_rows, save_json, write_csv_rows

DERIVED = [("landmarks_all", "npy"), ("landmarks_face", "npy"), ("landmarks_hands", "npy"),
           ("landmarks_pose", "npy"), ("skeleton_videos", "mp4"), ("overlay_check", "mp4"),
           ("metadata", "json")]


def prune(pairs, processed: Path, dataset: Path, excluded: Path, dry_run: bool = False) -> dict:
    """Move sources + delete derived files for ``[(class, stem), ...]``."""
    moved, removed, notes = [], [], []
    for cls, stem in pairs:
        sources = [p for p in sorted((dataset / cls).glob(f"{stem}.*"))
                   if p.suffix.lower() in VIDEO_EXTS] if (dataset / cls).is_dir() else []
        if not sources:
            notes.append(f"source video for {cls}/{stem} not found (already moved?)")
        for src in sources:
            target = excluded / cls / src.name
            moved.append(f"{src} -> {target}")
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(target))
        for folder, ext in DERIVED:
            p = processed / folder / cls / f"{stem}.{ext}"
            if p.exists():
                removed.append(str(p))
                if not dry_run:
                    p.unlink()
    return {"moved": moved, "removed": removed, "notes": notes}


def refresh_reports(processed: Path, dropped: set) -> None:
    """Remove the dropped clips from every dataset-level file and recount."""
    mf = processed / "manifest.csv"
    if mf.exists():
        rows = read_csv_rows(mf)
        if rows:
            write_csv_rows(mf, [r for r in rows if (r["class"], r["stem"]) not in dropped],
                           list(rows[0].keys()))
    for name in ("extraction_report.json", "verification_report.json"):
        rp = processed / name
        if rp.exists():
            r = load_json(rp)
            r["clips"] = [c for c in r.get("clips", []) if (c["class"], c["stem"]) not in dropped]
            r["n_clips"] = len(r["clips"])
            r["verdicts"] = count_verdicts(r["clips"])
            r["mean_detection_rate"] = mean_rates(r["clips"])
            save_json(r, rp)


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="signlang prune",
                                description="Reversibly drop clips from the processed dataset.")
    p.add_argument("clips", nargs="+", help="clips as <class>/<stem>")
    add_path_args(p, dataset=True, processed=True)
    p.add_argument("--excluded-dir", default=None, help="where pruned videos go (default: excluded_clips/)")
    p.add_argument("--dry-run", action="store_true", help="show what would happen, change nothing")
    return p.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    paths = paths_from_args(a)
    try:
        pairs = [parse_clip_ref(c) for c in a.clips]
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")
    excluded = Path(a.excluded_dir) if a.excluded_dir else paths.dataset.parent / "excluded_clips"

    res = prune(pairs, paths.processed, paths.dataset, excluded, dry_run=a.dry_run)
    if not a.dry_run:
        refresh_reports(paths.processed, set(pairs))
    tag = "[DRY-RUN]" if a.dry_run else "[PRUNE]"
    print(f"{tag} {len(pairs)} clip(s): " + ", ".join("/".join(x) for x in pairs))
    for m in res["moved"]:
        print(f"  move    {m}")
    for r in res["removed"]:
        print(f"  delete  {r}")
    for n in res["notes"]:
        print(f"  note    {n}")
    if not a.dry_run:
        print(f"{tag} reports refreshed; sources preserved in {excluded}")


if __name__ == "__main__":
    main()
