"""Privacy-conscious aggregate session telemetry for prototype validation."""

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time
import uuid


class SessionTelemetry:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.session_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.started_wall = datetime.now().astimezone().isoformat(timespec="seconds")
        self.started = time.monotonic()
        self.frames = 0
        self.hand_frames = 0
        self.unknown_frames = 0
        self.fps_total = 0.0
        self.methods = Counter()
        self.confirmations = Counter()
        self.alerts = []
        self.closed = False

    def frame(self, hand_visible, raw_detection, method, fps):
        self.frames += 1
        self.hand_frames += int(bool(hand_visible))
        self.unknown_frames += int(raw_detection in ("Unknown", "No hand detected"))
        self.fps_total += max(float(fps), 0.0)
        if hand_visible:
            self.methods[str(method)] += 1

    def confirmed(self, label):
        self.confirmations[str(label)] += 1

    def alert(self, reason, alert_id=None, state="TRIGGERED"):
        self.alerts.append({"reason": str(reason), "alert_id": alert_id, "state": state})

    def summary(self):
        duration = max(time.monotonic() - self.started, 0.001)
        return {
            "session_id": self.session_id,
            "started_at": self.started_wall,
            "duration_seconds": round(duration, 2),
            "frames": self.frames,
            "average_fps": round(self.fps_total / max(self.frames, 1), 2),
            "hand_visibility_percent": round(100 * self.hand_frames / max(self.frames, 1), 2),
            "unknown_frame_percent": round(100 * self.unknown_frames / max(self.frames, 1), 2),
            "recognition_methods": dict(self.methods),
            "confirmed_gestures": dict(self.confirmations),
            "alerts": list(self.alerts),
        }

    def close(self):
        if self.closed:
            return None
        self.closed = True
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"{self.session_id}.json"
        destination.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return destination
