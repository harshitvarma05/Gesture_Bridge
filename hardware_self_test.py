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
from gesture_bridge.camera import open_camera
from gesture_bridge.config import load_env_file
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


def check_model(path="isl_landmark_model.pkl", metrics_path="model_metrics.json"):
    try:
        with open(path, "rb") as handle:
            model = pickle.load(handle)
        validation_reliable = False
        try:
            validation_reliable = bool(json.loads(Path(metrics_path).read_text(encoding="utf-8")).get("validation_reliable"))
        except (OSError, ValueError, TypeError):
            pass
        return {
            "ok": int(model.n_features_in_) == 126,
            "live_enabled": validation_reliable,
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


def check_pretrained_model(path="gesture_recognizer.task"):
    model_path = Path(path)
    try:
        size = model_path.stat().st_size
        return {"ok": size > 1_000_000, "path": str(model_path), "size_bytes": size}
    except OSError as error:
        return {"ok": False, "error": type(error).__name__}


def check_camera(index, frames):
    try:
        camera = open_camera(index, 640, 360, fps=30)
        if camera is None:
            return {"ok": False, "error": "no readable webcam found", "requested_index": index}
        capture = camera.capture
        start = time.perf_counter()
        received = 0
        for _ in range(frames):
            success, _ = capture.read()
            received += int(success)
        elapsed = time.perf_counter() - start
        capture.release()
        return {
            "ok": received == frames,
            "index": camera.index,
            "backend": camera.backend,
            "resolution": f"{camera.width}x{camera.height}",
            "frames": received,
            "fps": round(received / max(elapsed, 0.001), 1),
        }
    except (ImportError, OSError) as error:
        return {"ok": False, "error": type(error).__name__}


def provider_readiness():
    twilio_keys = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM", "GESTURE_BRIDGE_CAREGIVER_TO")
    webhook = bool(os.getenv("GESTURE_BRIDGE_ALERT_WEBHOOK", "").strip())
    ntfy = bool(os.getenv("GESTURE_BRIDGE_NTFY_URL", "").strip())
    twilio = all(os.getenv(key, "").strip() for key in twilio_keys)
    live_mode = os.getenv("GESTURE_BRIDGE_LIVE_ALERTS", "0") == "1"
    return {
        "ok": (webhook or ntfy or twilio) if live_mode else True,
        "mode": "live" if live_mode else "demo",
        "webhook_configured": webhook,
        "phone_notification_configured": ntfy,
        "twilio_configured": twilio,
    }


def main():
    load_env_file(Path(__file__).resolve().parent / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=-1, help="-1 automatically scans USB webcams")
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
        "pretrained_gesture_model": check_pretrained_model(),
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
        manager.wait_for_delivery(payload, timeout=30)
        report["test_alert"] = {
            "alert_id": payload["alert_id"],
            "mode": payload["mode"],
            "state": payload["state"],
            "status": manager.last_status,
        }

    critical = (report["pretrained_gesture_model"]["ok"], report["camera"]["ok"], report["providers"]["ok"])
    report["ready_for_camera_demo"] = all(critical)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
