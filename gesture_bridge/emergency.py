"""Deterministic pre-alert countdown for cancelable, non-silent emergencies."""

from dataclasses import dataclass
import time


@dataclass
class PendingAlert:
    reason: str
    message: str
    armed_at: float
    deadline: float


class EmergencyController:
    def __init__(self, alert_manager, delay_seconds=5.0):
        self.alert_manager = alert_manager
        self.delay_seconds = max(float(delay_seconds), 1.0)
        self.pending = None
        self.last_result = None

    def arm(self, reason, message, now=None):
        if self.pending:
            return False
        now = time.monotonic() if now is None else float(now)
        self.pending = PendingAlert(reason, message, now, now + self.delay_seconds)
        return True

    def cancel(self):
        if not self.pending:
            return False
        self.last_result = "Cancelled before delivery"
        self.pending = None
        return True

    def remaining(self, now=None):
        if not self.pending:
            return None
        now = time.monotonic() if now is None else float(now)
        return max(0.0, self.pending.deadline - now)

    def tick(self, now=None):
        if not self.pending:
            return None
        now = time.monotonic() if now is None else float(now)
        if now < self.pending.deadline:
            return None
        pending = self.pending
        self.pending = None
        payload = self.alert_manager.trigger(pending.reason, pending.message, silent=False)
        self.last_result = f"Alert sent: {payload['alert_id']}"
        return payload

