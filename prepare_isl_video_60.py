"""Create a leakage-safe manifest for the 60-class ISL video dataset.

Each source recording has left/right tilt augmentations. All three variants are
kept in the same split so validation never sees an augmented copy of training data.
"""

import argparse
import csv
import hashlib
from pathlib import Path
import re

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUGMENTATION_SUFFIX = re.compile(r"_(left|right)_tilt$", re.IGNORECASE)


def source_stem(path):
    return AUGMENTATION_SUFFIX.sub("", Path(path).stem)


def stable_rank(label, stem):
    digest = hashlib.sha1(f"{label}/{stem}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def prepare(root, output, test_fraction=0.2):
    root = Path(root).resolve()
    rows = []
    per_label_groups = {}
    for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        videos = sorted(path for path in label_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)
        groups = {}
        for video in videos:
            groups.setdefault(source_stem(video), []).append(video)
        per_label_groups[label_dir.name] = len(groups)

        # Rank-based selection is reproducible and keeps every class close to 80/20.
        ranked = sorted(groups, key=lambda stem: stable_rank(label_dir.name, stem))
        test_count = min(max(1, round(len(ranked) * test_fraction)), max(len(ranked) - 1, 1))
        test_stems = set(ranked[:test_count])
        assignments = {stem: stem in test_stems for stem in groups}

        for stem, variants in groups.items():
            split = "test" if assignments[stem] else "train"
            group_id = f"{label_dir.name}/{stem}"
            for video in variants:
                rows.append({"path": str(video), "label": label_dir.name, "split": split, "signer": group_id})

    with Path(output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split", "signer"])
        writer.writeheader()
        writer.writerows(rows)
    return rows, per_label_groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--output", default="video_manifest.csv")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    rows, groups = prepare(args.root, args.output, args.test_fraction)
    train = sum(row["split"] == "train" for row in rows)
    test = len(rows) - train
    print(f"Classes: {len(groups)} | source recordings: {sum(groups.values())}")
    print(f"Videos: {len(rows)} | train: {train} | test: {test}")
    print(f"Saved leakage-safe manifest to {args.output}")
