"""Build a standard manifest from INCLUDE, Amrita, or another labelled video tree."""

import argparse
import csv
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def build_manifest(root, output, split="", signer_from_filename=False, append=False):
    root = Path(root).resolve()
    videos = sorted(path for path in root.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES)
    rows = []
    for path in videos:
        label = path.parent.name.replace("_", " ").strip()
        signer = path.stem.split("_")[0] if signer_from_filename and "_" in path.stem else ""
        rows.append({"path": str(path), "label": label, "split": split, "signer": signer})
    output_path = Path(output)
    write_header = not append or not output_path.exists()
    with output_path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split", "signer"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Root containing one directory per sign label")
    parser.add_argument("--output", default="video_manifest.csv")
    parser.add_argument("--split", choices=("", "train", "validation", "test"), default="")
    parser.add_argument("--signer-from-filename", action="store_true")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    count = build_manifest(args.root, args.output, args.split, args.signer_from_filename, args.append)
    print(f"Indexed {count} videos in {args.output}")
