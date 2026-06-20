"""Motion-based distress and covert SOS detection.

The analyzer only uses MediaPipe hand landmarks, so it works offline on a
Raspberry Pi. Face and body cues are deliberately reported as unavailable
until the corresponding landmark models are enabled.
"""

from collections import deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import statistics
import time


def _distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


@dataclass
class SafetyState:
    level: str = "CALM"
    score: float = 0.0
    speed: float = 0.0
    tremor: float = 0.0
    repetition: int = 0
    abruptness: float = 0.0
    sos_pattern: str | None = None
    reasons: list[str] = field(default_factory=list)


class SafetyAnalyzer:
    """Maintains a short rolling window of normalized hand motion features."""

    def __init__(self, window_seconds=3.0, profile_path="gesture_profile.json"):
        self.window_seconds = window_seconds
        self.profile_path = Path(profile_path)
        self.calibration_status = "Default thresholds"
        self.profile = self._load_profile()
        self.samples = deque()
        self.gesture_events = deque()
        self.pinch_events = deque()
        self.pose_events = deque()
        self.last_pinch = False
        self.last_pose = None
        self.last_sos_at = 0.0
        self.last_confirmed = None

    def reset(self):
        self.__init__(self.window_seconds, str(self.profile_path))

    def _load_profile(self):
        defaults = {
            "speed_reference": 5.0,
            "tremor_reference": 5.0,
            "abrupt_reference": 12.0,
            "elevated_threshold": 0.42,
            "high_threshold": 0.68,
        }
        try:
            with self.profile_path.open("r", encoding="utf-8") as handle:
                stored = json.load(handle)
            defaults.update({key: float(value) for key, value in stored.items() if key in defaults})
            self.calibration_status = "Personal calibration loaded"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return defaults

    def calibrate_from_window(self):
        """Learn conservative motion references from a calm three-second sample."""
        velocities = self._velocities()
        if len(velocities) < 15:
            self.calibration_status = "Calibration needs a visible hand for 3 seconds"
            return False
        mean = statistics.fmean(velocities)
        deviation = statistics.pstdev(velocities)
        self.profile.update({
            "speed_reference": max(mean * 3.0, 0.8),
            "tremor_reference": max(deviation * 3.5, 0.8),
            "abrupt_reference": max(max(velocities) * 2.5, 2.0),
        })
        with self.profile_path.open("w", encoding="utf-8") as handle:
            json.dump(self.profile, handle, indent=2)
        self.calibration_status = "Personal calm-motion baseline saved"
        return True

    def _velocities(self):
        velocities = []
        for previous, current in zip(self.samples, list(self.samples)[1:]):
            dt = max(current[0] - previous[0], 0.001)
            avg_scale = max((current[3] + previous[3]) / 2, 0.025)
            velocities.append(math.hypot(current[1] - previous[1], current[2] - previous[2]) / avg_scale / dt)
        return velocities

    @staticmethod
    def _features(hand):
        wrist = hand[0]
        palm_scale = max(_distance(hand[5], hand[17]), 0.025)
        cx = sum(p.x for p in hand) / len(hand)
        cy = sum(p.y for p in hand) / len(hand)
        pinch = _distance(hand[4], hand[8]) / palm_scale < 0.38
        finger_spread = sum(_distance(hand[0], hand[i]) for i in (8, 12, 16, 20)) / (4 * palm_scale)
        pose = "open" if finger_spread > 2.05 else "fist" if finger_spread < 1.35 else "other"
        thumb_rub = min(_distance(hand[4], hand[8]), _distance(hand[4], hand[12])) / palm_scale < 0.34
        return cx, cy, palm_scale, pinch, pose, thumb_rub

    def note_confirmed_gesture(self, gesture, now=None):
        now = now or time.monotonic()
        if gesture and gesture not in ("Unknown", "No hand detected"):
            if gesture != self.last_confirmed or not self.gesture_events or now - self.gesture_events[-1][0] > 0.45:
                self.gesture_events.append((now, gesture))
                self.last_confirmed = gesture

    def update(self, hands, now=None):
        now = now or time.monotonic()
        if not hands:
            self._trim(now)
            return SafetyState()

        cx, cy, scale, pinch, pose, thumb_rub = self._features(hands[0])
        self.samples.append((now, cx, cy, scale, thumb_rub))

        if pinch and not self.last_pinch:
            self.pinch_events.append(now)
        self.last_pinch = pinch

        if pose != self.last_pose and pose in ("open", "fist"):
            self.pose_events.append((now, pose))
            self.last_pose = pose

        self._trim(now)
        sos = self._detect_sos(now)
        return self._distress_state(sos)

    def _trim(self, now):
        cutoff = now - self.window_seconds
        for queue in (self.samples, self.gesture_events, self.pinch_events, self.pose_events):
            while queue:
                event_time = queue[0][0] if isinstance(queue[0], tuple) else queue[0]
                if event_time >= cutoff:
                    break
                queue.popleft()

    def _detect_sos(self, now):
        if now - self.last_sos_at < 5.0:
            return None

        # Three deliberate thumb/index taps inside 2.2 seconds.
        recent_taps = [t for t in self.pinch_events if now - t <= 2.2]
        if len(recent_taps) >= 3:
            self.pinch_events.clear()
            self.last_sos_at = now
            return "3 finger taps"

        # Closed-open fist twice: fist, open, fist, open.
        poses = [p for t, p in self.pose_events if now - t <= 3.0]
        if len(poses) >= 4 and poses[-4:] == ["fist", "open", "fist", "open"]:
            self.pose_events.clear()
            self.last_sos_at = now
            return "closed-open fist pattern"

        # Rapid thumb rubbing generates several near/far transitions (tap proxy).
        rub_values = [rub for _, _, _, _, rub in self.samples]
        transitions = sum(a != b for a, b in zip(rub_values, rub_values[1:]))
        if len(rub_values) >= 12 and transitions >= 6:
            self.samples.clear()
            self.last_sos_at = now
            return "thumb rub pattern"

        # Open-palm depth pulse: repeated scale expansion/contraction.
        scales = [s for _, _, _, s, _ in self.samples]
        if len(scales) >= 18:
            mean_scale = statistics.fmean(scales)
            pulse_range = (max(scales) - min(scales)) / max(mean_scale, 0.001)
            direction_changes = sum(
                (b - a) * (c - b) < 0 for a, b, c in zip(scales, scales[1:], scales[2:])
            )
            if pulse_range > 0.28 and direction_changes >= 4:
                self.samples.clear()
                self.last_sos_at = now
                return "palm pulse gesture"
        return None

    def _distress_state(self, sos):
        if len(self.samples) < 3:
            return SafetyState(sos_pattern=sos, level="SOS" if sos else "CALM", score=1.0 if sos else 0.0)

        velocities = self._velocities()

        speed = min(statistics.fmean(velocities) / self.profile["speed_reference"], 1.0)
        if len(velocities) >= 5:
            recent = velocities[-10:]
            tremor = min(statistics.pstdev(recent) / self.profile["tremor_reference"], 1.0)
        else:
            tremor = 0.0
        abruptness = min(max(velocities) / self.profile["abrupt_reference"], 1.0)

        labels = [label for _, label in self.gesture_events]
        repetition = max((labels.count(label) for label in set(labels)), default=0)
        repeat_score = min(max(repetition - 1, 0) / 3.0, 1.0)
        score = min(0.32 * speed + 0.34 * tremor + 0.18 * abruptness + 0.16 * repeat_score, 1.0)

        reasons = []
        if speed > 0.55:
            reasons.append("rapid motion")
        if tremor > 0.42:
            reasons.append("hand trembling")
        if repetition >= 3:
            reasons.append("repeated gesture")
        if abruptness > 0.65:
            reasons.append("abrupt movement")

        if sos:
            level = "SOS"
            score = 1.0
            reasons = [sos]
        elif score >= self.profile["high_threshold"]:
            level = "HIGH"
        elif score >= self.profile["elevated_threshold"]:
            level = "ELEVATED"
        else:
            level = "CALM"
        return SafetyState(level, score, speed, tremor, repetition, abruptness, sos, reasons)
