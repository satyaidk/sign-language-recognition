"""
Prune clips from the processed dataset  (reversible)
====================================================
Drops one or more clips from the dataset cleanly:

  * MOVES the source video out of ``dataset/<class>/`` into
    ``excluded_clips/<class>/`` (preserved, not deleted, and no longer picked
    up by ``extract_dataset.py``).
  * DELETES the clip's processed artifacts (the 4 landmark .npy files, the
    skeleton video, the overlay video, and the metadata json).
  * REFRESHES ``manifest.csv`` and ``extraction_report.json`` to match.

Usage
-----
    python prune_clips.py                      # default: no/no_13 yes/yes_12
    python prune_clips.py no/no_13 yes/yes_12  # explicit
    python prune_clips.py --out processed_dataset --dataset dataset hi/hi_03

To undo: move the file(s) back from excluded_clips/ into dataset/ and re-run
extract_dataset.py for those clips.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ART_NPY = ["landmarks_all", "landmarks_face", "landmarks_hands", "landmarks_pose"]
ART_MISC = [("skeleton_videos", "mp4"), ("overlay_check", "mp4"), ("metadata", "json")]


def prune(pairs, out, dataset, excluded):
    moved, removed, missing = [], [], []
    for cls, stem in pairs:
        src = dataset / cls / f"{stem}.mp4"
        if src.exists():
            (excluded / cls).mkdir(parents=True, exist_ok=True)
            target = excluded / cls / f"{stem}.mp4"
            shutil.move(str(src), str(target))
            moved.append(f"{cls}/{stem}.mp4 -> {target}")
        else:
            missing.append(f"source {cls}/{stem}.mp4 (already moved?)")

        for fol in ART_NPY:
            p = out / fol / cls / f"{stem}.npy"
            if p.exists():
                p.unlink(); removed.append(str(p))
        for sub, ext in ART_MISC:
            p = out / sub / cls / f"{stem}.{ext}"
            if p.exists():
                p.unlink(); removed.append(str(p))
    return moved, removed, missing


def refresh_reports(out, dropped):
    # manifest.csv — drop matching rows
    mf = out / "manifest.csv"
    if mf.exists():
        rows = list(csv.DictReader(open(mf, newline="")))
        if rows:
            keep = [r for r in rows if (r["class"], r["stem"]) not in dropped]
            with open(mf, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(keep)

    # extraction_report.json — recompute counts + mean rates from remaining clips
    rp = out / "extraction_report.json"
    if rp.exists():
        r = json.load(open(rp))
        r["clips"] = [c for c in r.get("clips", []) if (c["class"], c["stem"]) not in dropped]
        r["n_clips"] = len(r["clips"])
        verdicts = {"PASS": 0, "WARN": 0, "FAIL": 0}
        agg = {}
        for c in r["clips"]:
            verdicts[c["verdict"]] = verdicts.get(c["verdict"], 0) + 1
            for k, v in c.get("detection_rate", {}).items():
                agg.setdefault(k, []).append(v)
        r["verdicts"] = verdicts
        if agg:
            r["mean_detection_rate"] = {k: round(sum(v) / len(v), 3) for k, v in agg.items()}
        json.dump(r, open(rp, "w"), indent=2)


def main():
    p = argparse.ArgumentParser(description="Reversibly drop clips from the processed dataset.")
    p.add_argument("clips", nargs="*", default=["no/no_13", "yes/yes_12"],
                   help="clips as <class>/<stem> (default: the 2 close-up WARN clips)")
    p.add_argument("--out", default="processed_dataset")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--excluded", default="excluded_clips")
    a = p.parse_args()

    clips = a.clips or ["no/no_13", "yes/yes_12"]
    pairs = [tuple(c.replace("\\", "/").split("/", 1)) for c in clips]
    if any(len(x) != 2 for x in pairs):
        sys.exit("[ERROR] clips must be <class>/<stem>")

    moved, removed, missing = prune(pairs, Path(a.out), Path(a.dataset), Path(a.excluded))
    refresh_reports(Path(a.out), set(pairs))

    print(f"[PRUNE] dropped {len(pairs)} clip(s): " + ", ".join("/".join(x) for x in pairs))
    for m in moved:
        print(f"  moved   {m}")
    for r in removed:
        print(f"  removed {r}")
    for m in missing:
        print(f"  note    {m}")
    print("[PRUNE] refreshed manifest.csv + extraction_report.json")
    print(f"[PRUNE] sources preserved in {a.excluded}/ (move back + re-extract to undo)")


if __name__ == "__main__":
    main()
