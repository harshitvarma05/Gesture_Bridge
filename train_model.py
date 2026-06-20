import csv
import os
import pickle
import json
from collections import Counter

import numpy as np


DATASET_FILE = "isl_custom_dataset.csv"
MODEL_FILE = "isl_landmark_model.pkl"
METRICS_FILE = "model_metrics.json"
TARGET_FEATURES = 126
SESSION_FILE = "collection_sessions.csv"
MIN_CLASS_SAMPLES = 40
EXPECTED_LABELS = {"Hello", "Help", "Yes", "No", "Thank You", "Water", "Doctor", "Emergency", "Washroom", "Pain", "Chest", "Medicine", "Call", "Caregiver"}


def load_dataset():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"{DATASET_FILE} not found. Run collect_data.py first to collect samples."
        )

    labels = []
    features = []

    with open(DATASET_FILE, "r", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)

        for row in reader:
            if len(row) < 64:
                continue

            label = row[0].strip()
            values = [float(value) for value in row[1:]]
            if len(values) not in (63, TARGET_FEATURES):
                raise ValueError(f"Unsupported feature count {len(values)} for label {label!r}")
            # Legacy one-hand samples remain useful: the absent second hand is zero-filled.
            values.extend([0.0] * (TARGET_FEATURES - len(values)))

            labels.append(label)
            features.append(values)

    if not labels:
        raise ValueError("No valid samples found in the dataset.")

    groups = [f"legacy-{index}" for index in range(len(labels))]
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r", newline="") as file:
            for session in csv.DictReader(file):
                try:
                    start = int(session["start_index"])
                    end = int(session["end_index"])
                    for index in range(max(start, 0), min(end + 1, len(groups))):
                        groups[index] = session["session_id"]
                except (KeyError, TypeError, ValueError):
                    continue

    return np.array(features, dtype=np.float32), np.array(labels), np.array(groups)


def train_model(features, labels, groups):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError as error:
        raise ImportError(
            "scikit-learn is not installed. Run: pip install scikit-learn"
        ) from error

    class_counts = Counter(labels)
    print("\nClass distribution:")
    for label, count in sorted(class_counts.items()):
        print(f"{label}: {count}")

    if len(class_counts) < 2:
        raise ValueError("Collect at least 2 different sign labels before training.")

    underrepresented = {label: count for label, count in class_counts.items() if count < MIN_CLASS_SAMPLES}
    if underrepresented:
        raise ValueError(f"Collect at least {MIN_CLASS_SAMPLES} samples per label. Too small: {underrepresented}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(splitter.split(features, labels, groups))
    x_train, x_test = features[train_indices], features[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    if len(set(y_train)) != len(class_counts) or len(set(y_test)) != len(class_counts):
        raise ValueError("The grouped validation split omitted a class. Collect more separate repetitions per label.")

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nValidation accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
    recorded_groups = {group for group in groups if not str(group).startswith("legacy-")}
    missing_labels = sorted(label for label in EXPECTED_LABELS if label.lower() not in {str(item).lower() for item in labels})
    metrics = {
        "accuracy": float(accuracy),
        "samples": int(len(labels)),
        "features": int(features.shape[1]),
        "class_distribution": {str(label): int(count) for label, count in class_counts.items()},
        "missing_expected_labels": missing_labels,
        "recorded_repetition_groups": len(recorded_groups),
        "validation_reliable": bool(recorded_groups) and not missing_labels,
        "validation_split": "grouped by recorded repetition where session metadata exists",
        "classification_report": report,
    }
    if not metrics["validation_reliable"]:
        print("\nWARNING: Accuracy is provisional. Record separate guided repetitions for every expected label.")
    return model, metrics


def main():
    features, labels, groups = load_dataset()
    feature_lengths = {len(feature) for feature in features}
    if len(feature_lengths) != 1:
        raise ValueError(f"Inconsistent feature lengths found in dataset: {feature_lengths}")
    model, metrics = train_model(features, labels, groups)

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(model, file)

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nSaved trained model to: {MODEL_FILE}")
    print(f"Saved validation metrics to: {METRICS_FILE}")
    print("Your main app.py will automatically load this model on startup.")


if __name__ == "__main__":
    main()
