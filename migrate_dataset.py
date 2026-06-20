"""Migrate the legacy one-hand (63 feature) CSV to the two-hand 126 format."""

import argparse
import csv
from pathlib import Path

TARGET_FEATURES = 126


def migrate(source, destination):
    source = Path(source)
    destination = Path(destination)
    with source.open("r", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError("Dataset is empty")

    output = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) not in (64, 127):
            raise ValueError(f"Row {row_number} has {len(row) - 1} features")
        output.append(row + ["0"] * (127 - len(row)))

    header = ["label"]
    for hand in range(1, 3):
        for landmark in range(21):
            header.extend([f"h{hand}_x{landmark}", f"h{hand}_y{landmark}", f"h{hand}_z{landmark}"])
    with destination.open("w", newline="") as handle:
        csv.writer(handle).writerows([header, *output])
    return len(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="isl_custom_dataset.csv")
    parser.add_argument("destination", nargs="?", default="isl_custom_dataset_126.csv")
    args = parser.parse_args()
    count = migrate(args.source, args.destination)
    print(f"Migrated {count} samples to {args.destination}")
