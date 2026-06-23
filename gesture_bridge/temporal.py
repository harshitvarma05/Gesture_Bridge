"""Shared motion-sequence representation for training and live inference."""

from collections import deque
import os
import pickle

import numpy as np


SEQUENCE_STEPS = 12
SHAPE_FEATURES = 126
FRAME_FEATURES = 130
DESCRIPTOR_VERSION = "landmarks-v2-12step-shape-trajectory-stats"


def landmarks_to_vector(hand_landmarks_list):
    """Encode normalized hand shape plus absolute centroid/scale trajectory."""
    hands = sorted(
        hand_landmarks_list[:2],
        key=lambda hand: sum(point.x for point in hand) / len(hand),
    )
    points = [point for hand in hands for point in hand]
    if not points:
        return np.zeros(FRAME_FEATURES, dtype=np.float32)
    origin = np.array([
        np.mean([point.x for point in points]),
        np.mean([point.y for point in points]),
        np.mean([point.z for point in points]),
    ], dtype=np.float32)
    values = []
    for hand in hands:
        for point in hand:
            values.extend(np.asarray([point.x, point.y, point.z], dtype=np.float32) - origin)
    values.extend([0.0] * (SHAPE_FEATURES - len(values)))
    shape = np.asarray(values[:SHAPE_FEATURES], dtype=np.float32)
    scale = float(np.max(np.abs(shape)))
    if scale > 0:
        shape = shape / scale
    trajectory = np.asarray([origin[0], origin[1], scale, len(hands) / 2.0], dtype=np.float32)
    return np.concatenate([shape, trajectory])


def sequence_descriptor(sequence, steps=SEQUENCE_STEPS):
    """Preserve ordered motion while remaining small enough for Raspberry Pi inference."""
    frames = np.asarray(sequence, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != FRAME_FEATURES:
        raise ValueError(f"Expected [frames, {FRAME_FEATURES}] sequence, got {frames.shape}")
    if len(frames) < 2:
        raise ValueError("At least two frames are required")

    positions = np.linspace(0, len(frames) - 1, steps)
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    weight = (positions - lower).reshape(-1, 1)
    sampled = frames[lower] * (1.0 - weight) + frames[upper] * weight
    deltas = np.diff(frames, axis=0)
    summary = np.concatenate([
        frames.mean(axis=0),
        frames.std(axis=0),
        deltas.mean(axis=0),
        deltas.std(axis=0),
    ])
    return np.concatenate([sampled.reshape(-1), summary]).astype(np.float32)


class TemporalRecognizer:
    def __init__(self, model_path="isl_temporal_model.pkl", window_frames=24, min_frames=10, prediction_interval=2):
        self.window = deque(maxlen=window_frames)
        self.min_frames = min_frames
        self.prediction_interval = max(1, int(prediction_interval))
        self._update_count = 0
        self._last_prediction = None
        self.bundle = None
        self.last_debug = None
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as handle:
                    candidate = pickle.load(handle)
                if candidate.get("descriptor_version") == DESCRIPTOR_VERSION:
                    self.bundle = candidate
            except (OSError, EOFError, AttributeError, pickle.UnpicklingError):
                pass

    @property
    def available(self):
        return self.bundle is not None

    def reset(self):
        self.window.clear()
        self.last_debug = None
        self._update_count = 0
        self._last_prediction = None

    def update(self, hands):
        if not hands:
            self.reset()
            return None
        self.window.append(landmarks_to_vector(hands))
        if not self.available or len(self.window) < self.min_frames:
            return None
        self._update_count += 1
        if self._update_count % self.prediction_interval:
            return self._last_prediction
        descriptor = sequence_descriptor(list(self.window))
        frames = np.asarray(self.window, dtype=np.float32)
        centroid_motion = float(np.mean(np.linalg.norm(np.diff(frames[:, 126:128], axis=0), axis=1)))
        shape_motion = float(np.mean(np.abs(np.diff(frames[:, :126], axis=0))))
        model = self.bundle["model"]
        probabilities = model.predict_proba([descriptor])[0]
        still_matches = np.where(np.char.lower(model.classes_.astype(str)) == "still")[0]
        motion_energy = centroid_motion + shape_motion
        if len(still_matches) and motion_energy >= 0.025:
            probabilities = probabilities.copy()
            probabilities[still_matches[0]] = 0.0
            total = probabilities.sum()
            if total > 0:
                probabilities /= total
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        ordered = np.sort(probabilities)
        margin = confidence - float(ordered[-2]) if len(ordered) > 1 else confidence
        raw_label = str(model.classes_[index])
        self.last_debug = {
            "candidate": raw_label.title(),
            "confidence": confidence,
            "margin": margin,
            "radius_ratio": None,
            "rejection": None,
            "motion_energy": motion_energy,
        }
        ood = self.bundle.get("ood")
        if ood:
            standardized = (descriptor - ood["mean"]) / ood["std"]
            distance = float(np.linalg.norm(standardized - ood["centroids"][raw_label]))
            radius = float(ood["radii"][raw_label])
            radius_ratio = distance / max(radius, 1e-6)
            # Public-dataset backgrounds differ sharply from live webcams. Keep a
            # generous OOD boundary while still rejecting random landmark sequences.
            limit = radius * max(float(ood.get("radius_multiplier", 1.25)), 6.0)
            minimum_margin = float(ood.get("minimum_probability_margin", 0.12))
            self.last_debug["radius_ratio"] = radius_ratio
            if distance > limit:
                self.last_debug["rejection"] = "unfamiliar motion"
                self._last_prediction = None
                return None
            if margin < minimum_margin:
                self.last_debug["rejection"] = "ambiguous classes"
                self._last_prediction = None
                return None
        self._last_prediction = (raw_label.title(), confidence)
        return self._last_prediction
