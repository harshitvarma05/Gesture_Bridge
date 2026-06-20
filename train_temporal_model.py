"""Extract video landmark sequences and train an offline temporal sign recognizer."""

import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_bridge.temporal import DESCRIPTOR_VERSION, landmarks_to_vector, sequence_descriptor


class VideoSequenceExtractor:
    def __init__(self, task_file="hand_landmarker.task"):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=task_file,
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.clock_ms = 0

    def close(self):
        self.landmarker.close()

    def extract(self, video_path, max_frames=96):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return None
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, total // max_frames) if total > 0 else 1
        sequence = []
        frame_index = 0
        last_timestamp = self.clock_ms
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = self.clock_ms + int(frame_index * 1000 / fps)
                last_timestamp = timestamp
                result = self.landmarker.detect_for_video(image, timestamp)
                if result.hand_landmarks:
                    sequence.append(landmarks_to_vector(result.hand_landmarks))
            frame_index += 1
        capture.release()
        self.clock_ms = last_timestamp + 1000
        return np.asarray(sequence, dtype=np.float32) if len(sequence) >= 6 else None


def cache_path(cache_dir, video_path, max_frames):
    digest = hashlib.sha1(
        f"{Path(video_path).resolve()}|frames={max_frames}|{DESCRIPTOR_VERSION}".encode()
    ).hexdigest()
    return Path(cache_dir) / f"{digest}.npy"


def load_rows(manifest):
    with open(manifest, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Manifest needs path and label columns")
    return rows


def build_features(rows, cache_dir, task_file, max_frames=48):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    extractor = None
    features, labels, splits, signers, kept = [], [], [], [], []
    try:
        for index, row in enumerate(rows, start=1):
            cached = cache_path(cache_dir, row["path"], max_frames)
            if cached.exists():
                sequence = np.load(cached)
            else:
                if extractor is None:
                    extractor = VideoSequenceExtractor(task_file)
                sequence = extractor.extract(row["path"], max_frames=max_frames)
                if sequence is not None:
                    np.save(cached, sequence)
            if sequence is None or len(sequence) < 6:
                print(f"Skip {index}/{len(rows)}: insufficient hand frames in {row['path']}")
                continue
            features.append(sequence_descriptor(sequence))
            labels.append(row["label"])
            splits.append(row.get("split", "").lower())
            signers.append(row.get("signer", "") or f"video-{index}")
            kept.append(row["path"])
            if index % 25 == 0:
                print(f"Processed {index}/{len(rows)} videos")
    finally:
        if extractor is not None:
            extractor.close()
    return np.asarray(features), np.asarray(labels), np.asarray(splits), np.asarray(signers), kept


def split_indices(labels, splits, signers):
    from sklearn.model_selection import GroupShuffleSplit
    explicit_train = np.where(splits == "train")[0]
    explicit_test = np.where(np.isin(splits, ["validation", "test"]))[0]
    if len(explicit_train) and len(explicit_test):
        return explicit_train, explicit_test, "manifest-defined grouped split"
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train, test = next(splitter.split(np.zeros(len(labels)), labels, signers))
    return train, test, "signer-grouped split" if any(not value.startswith("video-") for value in signers) else "video split (provisional)"


def train(args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    rows = load_rows(args.manifest)
    if args.labels:
        requested = {label.strip().lower() for label in args.labels.split(",") if label.strip()}
        rows = [row for row in rows if row["label"].lower() in requested]
        found = {row["label"].lower() for row in rows}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Requested labels not found in manifest: {missing}")
    features, labels, splits, signers, paths = build_features(
        rows, args.cache_dir, args.task_file, max_frames=args.max_frames
    )
    counts = Counter(labels)
    too_small = {label: count for label, count in counts.items() if count < args.min_videos}
    if len(counts) < 2 or too_small:
        raise ValueError(f"Need at least two classes and {args.min_videos} videos/class. Too small: {too_small}")
    train_idx, test_idx, split_method = split_indices(labels, splits, signers)
    if set(labels[train_idx]) != set(labels) or set(labels[test_idx]) != set(labels):
        raise ValueError("Evaluation split omitted a class; add signer/video diversity or official splits")
    model = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=24,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(features[train_idx], labels[train_idx])
    predictions = model.predict(features[test_idx])
    accuracy = float(accuracy_score(labels[test_idx], predictions))
    report = {
        "accuracy": accuracy,
        "descriptor_version": DESCRIPTOR_VERSION,
        "split_method": split_method,
        "videos_used": len(features),
        "class_distribution": dict(counts),
        "classification_report": classification_report(labels[test_idx], predictions, output_dict=True, zero_division=0),
        "confusion_matrix_labels": sorted(counts),
        "confusion_matrix": confusion_matrix(labels[test_idx], predictions, labels=sorted(counts)).tolist(),
    }
    feature_mean = features[train_idx].mean(axis=0)
    feature_std = features[train_idx].std(axis=0)
    feature_std[feature_std < 1e-5] = 1.0
    standardized = (features[train_idx] - feature_mean) / feature_std
    centroids, radii = {}, {}
    for label in model.classes_:
        class_vectors = standardized[labels[train_idx] == label]
        centroid = class_vectors.mean(axis=0)
        distances = np.linalg.norm(class_vectors - centroid, axis=1)
        centroids[str(label)] = centroid.astype(np.float32)
        radii[str(label)] = float(max(np.percentile(distances, 99), 1.0))
    ood = {
        "mean": feature_mean.astype(np.float32),
        "std": feature_std.astype(np.float32),
        "centroids": centroids,
        "radii": radii,
        "radius_multiplier": 1.25,
        "minimum_probability_margin": 0.12,
    }
    report["unknown_rejection"] = "class-centroid radius plus probability margin"
    with open(args.output, "wb") as handle:
        pickle.dump({"model": model, "descriptor_version": DESCRIPTOR_VERSION, "report": report, "ood": ood}, handle)
    Path(args.metrics).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Temporal validation accuracy: {accuracy * 100:.2f}% ({split_method})")
    print(f"Saved {args.output} and {args.metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--cache-dir", default=".sequence_cache")
    parser.add_argument("--task-file", default="hand_landmarker.task")
    parser.add_argument("--output", default="isl_temporal_model.pkl")
    parser.add_argument("--metrics", default="temporal_model_metrics.json")
    parser.add_argument("--min-videos", type=int, default=8)
    parser.add_argument("--trees", type=int, default=240)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--labels", default="", help="Comma-separated label subset")
    train(parser.parse_args())
