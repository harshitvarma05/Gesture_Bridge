"""Auditable emergency delivery, acknowledgement, cancellation and Pi outputs."""

import base64
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import threading
import time
import urllib.parse
import urllib.request
import uuid


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class LocationProvider:
    """Prefer GPSD when configured, with a privacy-safe static fallback."""

    def __init__(self):
        self.static = os.getenv("GESTURE_BRIDGE_LOCATION", "Location unavailable (offline)")
        self.use_gpsd = os.getenv("GESTURE_BRIDGE_GPSD", "0") == "1"

    def get(self):
        if not self.use_gpsd:
            return self.static
        try:
            with socket.create_connection(("127.0.0.1", 2947), timeout=1.5) as connection:
                connection.sendall(b'?WATCH={"enable":true,"json":true};\n')
                deadline = time.monotonic() + 1.5
                buffer = b""
                while time.monotonic() < deadline:
                    buffer += connection.recv(4096)
                    for line in buffer.splitlines():
                        report = json.loads(line)
                        if report.get("class") == "TPV" and "lat" in report and "lon" in report:
                            lat, lon = float(report["lat"]), float(report["lon"])
                            return f"{lat:.6f}, {lon:.6f} | https://maps.google.com/?q={lat},{lon}"
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return self.static


class AlertManager:
    TERMINAL_STATES = {"ACKNOWLEDGED", "CANCELLED"}

    def __init__(self, log_path="emergency_alerts.jsonl"):
        self.log_path = Path(log_path)
        self.webhook_url = os.getenv("GESTURE_BRIDGE_ALERT_WEBHOOK", "").strip()
        self.demo_mode = os.getenv("GESTURE_BRIDGE_LIVE_ALERTS", "0") != "1"
        self.location_provider = LocationProvider()
        self.max_attempts = max(1, int(os.getenv("GESTURE_BRIDGE_ALERT_RETRIES", "3")))
        self.active_alert = None
        self.last_status = "Ready (demo mode)" if self.demo_mode else "Ready"
        self._lock = threading.Lock()

    def trigger(self, reason, message, silent=False):
        payload = {
            "alert_id": uuid.uuid4().hex[:12],
            "timestamp": _now(),
            "reason": reason,
            "message": message,
            "location": self.location_provider.get(),
            "silent": silent,
            "mode": "demo" if self.demo_mode else "live",
            "state": "RECORDED" if self.demo_mode else "QUEUED",
            "attempt": 0,
        }
        with self._lock:
            self.active_alert = payload
            self._write_event(payload)

        if not silent:
            self._pulse_buzzer()
        if self.demo_mode:
            self.last_status = f"Alert {payload['alert_id']} recorded (demo)"
        else:
            self.last_status = f"Alert {payload['alert_id']} queued"
            threading.Thread(target=self._deliver_with_retry, args=(payload,), daemon=True).start()
        return payload

    def acknowledge(self):
        return self._transition("ACKNOWLEDGED", "Caregiver acknowledged")

    def cancel(self):
        return self._transition("CANCELLED", "Alert cancelled by user")

    def _transition(self, state, status):
        with self._lock:
            if not self.active_alert:
                self.last_status = "No active alert"
                return False
            self.active_alert["state"] = state
            self.active_alert["updated_at"] = _now()
            self._write_event(self.active_alert)
            self.last_status = f"{status}: {self.active_alert['alert_id']}"
            return True

    def _deliver_with_retry(self, payload):
        configured = bool(self.webhook_url or self._twilio_configured())
        if not configured:
            self._set_delivery_state(payload, "FAILED", "No delivery provider configured")
            return

        for attempt in range(1, self.max_attempts + 1):
            with self._lock:
                if payload.get("state") in self.TERMINAL_STATES:
                    return
                payload["attempt"] = attempt
            try:
                deliveries = []
                if self.webhook_url:
                    self._post_webhook(payload)
                    deliveries.append("webhook")
                if self._twilio_configured():
                    self._post_twilio(payload)
                    deliveries.append("Twilio")
                self._set_delivery_state(payload, "DELIVERED", f"Delivered via {', '.join(deliveries)}")
                return
            except (OSError, TimeoutError, ValueError) as error:
                if attempt == self.max_attempts:
                    self._set_delivery_state(payload, "FAILED", f"Delivery failed: {type(error).__name__}")
                    return
                self._set_delivery_state(payload, "RETRYING", f"Retry {attempt}/{self.max_attempts}")
                time.sleep(min(2 ** (attempt - 1), 8))

    def _set_delivery_state(self, payload, state, status):
        with self._lock:
            if payload.get("state") in self.TERMINAL_STATES:
                return
            payload["state"] = state
            payload["updated_at"] = _now()
            self._write_event(payload)
            self.last_status = f"{status}: {payload['alert_id']}"

    def _post_webhook(self, payload):
        request = urllib.request.Request(
            self.webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()

    @staticmethod
    def _twilio_configured():
        keys = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM", "GESTURE_BRIDGE_CAREGIVER_TO")
        return all(os.getenv(key, "").strip() for key in keys)

    @staticmethod
    def _post_twilio(payload):
        sid = os.environ["TWILIO_ACCOUNT_SID"]
        token = os.environ["TWILIO_AUTH_TOKEN"]
        destination = os.environ["GESTURE_BRIDGE_CAREGIVER_TO"]
        sender = os.environ["TWILIO_FROM"]
        body = f"{payload['message']} Reason: {payload['reason']}. Location: {payload['location']} Alert: {payload['alert_id']}"
        data = urllib.parse.urlencode({"From": sender, "To": destination, "Body": body}).encode()
        credentials = base64.b64encode(f"{sid}:{token}".encode()).decode()
        request = urllib.request.Request(
            f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
            data=data,
            headers={"Authorization": f"Basic {credentials}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            response.read()

    def _write_event(self, payload):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self.log_path.stat().st_size >= 5 * 1024 * 1024:
                archived = self.log_path.with_name(self.log_path.name + ".1")
                os.replace(self.log_path, archived)
        except FileNotFoundError:
            pass
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload)) + "\n")

    @staticmethod
    def _pulse_buzzer():
        try:
            from gpiozero import Buzzer
            buzzer = Buzzer(int(os.getenv("GESTURE_BRIDGE_BUZZER_PIN", "17")))
            buzzer.beep(on_time=0.2, off_time=0.1, n=3, background=True)
        except (ImportError, RuntimeError, OSError, ValueError):
            pass
