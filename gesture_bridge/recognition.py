"""Time-based smoothing for reliable, frame-rate-independent gesture events."""

from collections import Counter, deque
from dataclasses import dataclass


INVALID_GESTURES = {"Unknown", "No hand detected", "None", ""}


@dataclass(frozen=True)
class RecognitionState:
    active: str
    candidate: str
    progress: float
    event: str | None = None


class GestureDebouncer:
    """Confirm a confident majority, then require release before repeating it."""

    def __init__(self, hold_seconds=0.28, release_seconds=0.18, window_seconds=0.65,
                 min_samples=3, confidence_threshold=0.55, agreement_threshold=0.68):
        self.hold_seconds = hold_seconds
        self.release_seconds = release_seconds
        self.window_seconds = window_seconds
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        self.agreement_threshold = agreement_threshold
        self.samples = deque()
        self.active = "Unknown"
        self.release_started = None

    def reset(self):
        self.samples.clear()
        self.active = "Unknown"
        self.release_started = None

    def update(self, label, confidence, now):
        valid = label not in INVALID_GESTURES and confidence >= self.confidence_threshold
        if not valid:
            if self.release_started is None:
                self.release_started = now
            if now - self.release_started >= self.release_seconds:
                self.active = "Unknown"
                self.samples.clear()
            return RecognitionState(self.active, "Unknown", 0.0)

        self.release_started = None
        if self.active not in INVALID_GESTURES and label != self.active:
            if self.samples and self.samples[-1][1] == self.active:
                self.samples.clear()
        self.samples.append((now, label, float(confidence)))
        cutoff = now - self.window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

        counts = Counter(item[1] for item in self.samples)
        candidate, count = counts.most_common(1)[0]
        candidate_samples = [item for item in self.samples if item[1] == candidate]
        span = candidate_samples[-1][0] - candidate_samples[0][0]
        agreement = count / len(self.samples)
        average_confidence = sum(item[2] for item in candidate_samples) / count
        progress = min(
            span / max(self.hold_seconds, 0.001),
            count / self.min_samples,
            agreement / self.agreement_threshold,
            average_confidence / self.confidence_threshold,
            1.0,
        )

        event = None
        if (
            count >= self.min_samples
            and span >= self.hold_seconds
            and agreement >= self.agreement_threshold
            and candidate != self.active
        ):
            self.active = candidate
            event = candidate
            self.samples.clear()
        return RecognitionState(self.active, candidate, progress, event)
