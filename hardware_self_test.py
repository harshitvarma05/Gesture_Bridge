"""Raspberry Pi/caregiver integration readiness test.

No external message or buzzer is triggered unless its explicit flag is supplied.
"""

import argparse
import csv
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import time

from gesture_bridge.alerts import AlertManager, LocationProvider
from gesture_bridge.temporal import DESCRIPTOR_VERSION


def check_dataset(path="isl_custom_dataset.csv"):
    try:
        with open(path, newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            labels = Counter(row[0] for row in reader if row)
        return {
            "ok": len(header) == 127 and len(labels) >= 2,
            "features": len(header) - 1,
            "samples": sum(labels.values()),
            "classes": dict(labels),
        }
    except (OSError, StopIteration):
        return {"ok": False, "error": "dataset unavailable"}


def check_model(path="isl_landmark_model.pkl"):
    try:
        with open(path, "rb") as handle:
            model = pickle.load(handle)
        return {
            "ok": int(model.n_features_in_) == 126,
            "features": int(model.n_features_in_),
            "classes": [str(item) for item in model.classes_],
        }
    except (OSError, AttributeError, pickle.UnpicklingError, EOFError) as error:
        return {"ok": False, "error": type(error).__name__}


def check_temporal_model(path="isl_temporal_model.pkl"):
    try:
        with open(path, "rb") as handle:
            bundle = pickle.load(handle)
        report = bundle.get("report", {})
        return {
            "ok": bundle.get("descriptor_version") == DESCRIPTOR_VERSION,
            "classes": [str(item) for item in bundle["model"].classes_],
            "validation_accuracy": report.get("accuracy"),
            "split_method": report.get("split_method"),
        }
    except (OSError, KeyError, AttributeError, pickle.UnpicklingError, EOFError) as error:
        return {"ok": False, "optional": True, "error": type(error).__name__}


def check_camera(index, frames):
    try:
        import cv2
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            return {"ok": False, "error": "camera did not open"}
        start = time.perf_counter()
        received = 0
        for _ in range(frames):
            success, _ = capture.read()
            received += int(success)
        elapsed = time.perf_counter() - start
        capture.release()
        return {"ok": received == frames, "frames": received, "fps": round(received / max(elapsed, 0.001), 1)}
    except (ImportError, OSError) as error:
        return {"ok": False, "error": type(error).__name__}


def provider_readiness():
    twilio_keys = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM", "GESTURE_BRIDGE_CAREGIVER_TO")
    webhook = bool(os.getenv("GESTURE_BRIDGE_ALERT_WEBHOOK", "").strip())
    twilio = all(os.getenv(key, "").strip() for key in twilio_keys)
    return {"ok": webhook or twilio, "webhook_configured": webhook, "twilio_configured": twilio}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--buzzer", action="store_true", help="physically pulse the configured GPIO buzzer")
    parser.add_argument("--send-test-alert", action="store_true", help="send a real provider test if live mode is enabled")
    parser.add_argument("--output", default="self_test_report.json")
    args = parser.parse_args()

    report = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": check_dataset(),
        "model": check_model(),
        "temporal_model": check_temporal_model(),
        "camera": check_camera(args.camera_index, args.frames),
        "providers": provider_readiness(),
        "location": {"value": LocationProvider().get()},
        "gpio": {"requested": args.buzzer},
    }

    if args.buzzer:
        AlertManager._pulse_buzzer()
        report["gpio"]["result"] = "pulse requested; confirm audibly"

    if args.send_test_alert:
        manager = AlertManager()
        payload = manager.trigger("Integration self-test", "Gesture-Bridge test alert—no emergency.", silent=True)
        report["test_alert"] = {"alert_id": payload["alert_id"], "mode": payload["mode"], "status": manager.last_status}

    critical = (report["dataset"]["ok"], report["model"]["ok"], report["camera"]["ok"])
    report["ready_for_camera_demo"] = all(critical)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
