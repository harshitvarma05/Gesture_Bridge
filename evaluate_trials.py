"""Generate recognition metrics from manually observed participant trials."""

import argparse
import csv
import json
from pathlib import Path


def evaluate(rows):
    usable = [row for row in rows if row.get("expected", "").strip()]
    if not usable:
        raise ValueError("No labelled trials found")
    labels = sorted({row["expected"].strip() for row in usable} | {row.get("predicted", "").strip() for row in usable if row.get("predicted", "").strip()})
    matrix = {expected: {predicted: 0 for predicted in labels + ["<none>"]} for expected in labels}
    correct = false_sos = 0
    latencies = []
    participants = set()
    for row in usable:
        expected = row["expected"].strip()
        predicted = row.get("predicted", "").strip() or "<none>"
        matrix[expected].setdefault(predicted, 0)
        matrix[expected][predicted] += 1
        correct += expected.lower() == predicted.lower()
        false_sos += row.get("false_sos", "").strip().lower() in ("1", "true", "yes")
        participants.add(row.get("participant", "unknown").strip() or "unknown")
        try:
            latencies.append(float(row.get("response_time_seconds", "")))
        except ValueError:
            pass
    return {
        "trials": len(usable),
        "participants": len(participants),
        "accuracy": round(correct / len(usable), 4),
        "average_response_time_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "false_sos_count": false_sos,
        "confusion_matrix": matrix,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", nargs="?", default="evaluation_trials.csv")
    parser.add_argument("--output", default="evaluation_report.json")
    args = parser.parse_args()
    with open(args.csv_file, newline="", encoding="utf-8") as handle:
        report = evaluate(list(csv.DictReader(handle)))
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
