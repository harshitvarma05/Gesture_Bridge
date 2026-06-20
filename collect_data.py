import csv
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


DATASET_FILE = "isl_custom_dataset.csv"
MODEL_TASK_FILE = "hand_landmarker.task"
SAMPLES_PER_SIGN = 160
REPETITIONS_PER_SIGN = 10
SAMPLES_PER_REPETITION = SAMPLES_PER_SIGN // REPETITIONS_PER_SIGN
SESSION_FILE = "collection_sessions.csv"
MAX_HANDS = 2
SUPPORTED_SIGNS = [
    "Hello",
    "Help",
    "Yes",
    "No",
    "Thank You",
    "Water",
    "Doctor",
    "Emergency",
    "Washroom",
    "Pain",
    "Chest",
    "Medicine",
    "Call",
    "Caregiver",
]


BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_TASK_FILE),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)


def landmarks_to_feature_vector(hand_landmarks_list):
    """
    Converts up to 2 detected hands into a fixed 126-value feature vector.

    Format:
    - first visible hand from left-to-right in the camera frame: 21 landmarks x/y/z
    - second visible hand from left-to-right in the camera frame: 21 landmarks x/y/z

    If only one hand is detected, the second hand is filled with zeros.
    Coordinates are normalized using a shared origin and scale so two-hand spacing
    is preserved for signs such as HELP.
    """
    sorted_hands = sorted(
        hand_landmarks_list[:MAX_HANDS],
        key=lambda hand: sum(landmark.x for landmark in hand) / len(hand)
    )

    all_landmarks = []
    for hand in sorted_hands:
        all_landmarks.extend(hand)

    if not all_landmarks:
        return [0.0] * (MAX_HANDS * 21 * 3)

    origin_x = sum(landmark.x for landmark in all_landmarks) / len(all_landmarks)
    origin_y = sum(landmark.y for landmark in all_landmarks) / len(all_landmarks)
    origin_z = sum(landmark.z for landmark in all_landmarks) / len(all_landmarks)

    values = []
    for hand in sorted_hands:
        for landmark in hand:
            values.extend([
                landmark.x - origin_x,
                landmark.y - origin_y,
                landmark.z - origin_z,
            ])

    missing_hands = MAX_HANDS - len(sorted_hands)
    values.extend([0.0] * (missing_hands * 21 * 3))

    values = np.array(values, dtype=np.float32)
    max_abs_value = np.max(np.abs(values))
    if max_abs_value > 0:
        values = values / max_abs_value

    return values.tolist()


def draw_landmarks(frame, hand_landmarks):
    height, width, _ = frame.shape
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]

    for start_idx, end_idx in connections:
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    for landmark in hand_landmarks:
        center = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, center, 4, (0, 0, 255), -1)


def ensure_dataset_header():
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r", newline="") as file:
            header = next(csv.reader(file), [])
        expected_columns = 1 + MAX_HANDS * 21 * 3
        if len(header) != expected_columns:
            raise ValueError(
                f"{DATASET_FILE} uses {len(header) - 1} features, but collection uses "
                f"{expected_columns - 1}. Run migrate_dataset.py before collecting."
            )
        return

    header = ["label"]
    for hand_index in range(MAX_HANDS):
        for landmark_index in range(21):
            header.extend([
                f"h{hand_index + 1}_x{landmark_index}",
                f"h{hand_index + 1}_y{landmark_index}",
                f"h{hand_index + 1}_z{landmark_index}",
            ])

    with open(DATASET_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)


def append_sample(label, feature_vector):
    with open(DATASET_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([label] + feature_vector)


def dataset_sample_count():
    if not os.path.exists(DATASET_FILE):
        return 0
    with open(DATASET_FILE, "r", newline="") as file:
        return max(sum(1 for _ in file) - 1, 0)


def append_session(label, start_index, end_index, repetition):
    exists = os.path.exists(SESSION_FILE)
    with open(SESSION_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(["session_id", "label", "start_index", "end_index", "recorded_at"])
        session_id = f"{label.lower().replace(' ', '-')}-{datetime.now().strftime('%Y%m%d%H%M%S')}-r{repetition}"
        writer.writerow([session_id, label, start_index, end_index, datetime.now().isoformat(timespec="seconds")])


def print_supported_signs():
    print("\nSupported project labels:")
    for index, sign in enumerate(SUPPORTED_SIGNS, start=1):
        print(f"{index}. {sign}")


def choose_label():
    print_supported_signs()
    label = input("\nEnter the exact label to collect, or type a new label: ").strip()

    if label == "":
        print("No label entered. Exiting.")
        return None

    return label


def main():
    if not os.path.exists(MODEL_TASK_FILE):
        print(f"Missing {MODEL_TASK_FILE}. Keep it in the same folder as collect_data.py")
        return

    label = choose_label()
    if label is None:
        return

    ensure_dataset_header()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("\nDataset collection started.")
    print(f"Record {REPETITIONS_PER_SIGN} separate repetitions; vary angle, distance, and lighting.")
    print("Press C to record each repetition. Collection pauses automatically between repetitions.")
    print("Press Q to quit.")

    collecting = False
    collected_count = 0
    repetition = 1
    repetition_start_index = None
    base_sample_index = dataset_sample_count()
    frame_timestamp_ms = 0
    last_saved_time = 0
    save_interval_seconds = 0.08

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Could not read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms += 33
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            hand_found = False
            detected_hands = []
            if result.hand_landmarks:
                hand_found = True
                detected_hands = result.hand_landmarks[:MAX_HANDS]

                for hand_landmarks in detected_hands:
                    draw_landmarks(frame, hand_landmarks)

                if collecting and collected_count < SAMPLES_PER_SIGN:
                    current_time = time.time()
                    if current_time - last_saved_time >= save_interval_seconds:
                        feature_vector = landmarks_to_feature_vector(detected_hands)
                        if repetition_start_index is None:
                            repetition_start_index = base_sample_index + collected_count
                        append_sample(label, feature_vector)
                        collected_count += 1
                        last_saved_time = current_time

                        if collected_count % SAMPLES_PER_REPETITION == 0:
                            append_session(
                                label,
                                repetition_start_index,
                                base_sample_index + collected_count - 1,
                                repetition,
                            )
                            repetition += 1
                            repetition_start_index = None
                            collecting = False

                if collected_count >= SAMPLES_PER_SIGN:
                    collecting = False

            status = "COLLECTING" if collecting else "PAUSED"
            cv2.rectangle(frame, (20, 20), (920, 145), (0, 0, 0), -1)
            cv2.putText(frame, f"Label: {label}", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            shown_repetition = min(repetition, REPETITIONS_PER_SIGN)
            cv2.putText(frame, f"Status: {status} | Repetition: {shown_repetition}/{REPETITIONS_PER_SIGN} | Samples: {collected_count}/{SAMPLES_PER_SIGN}", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, "C Record next repetition | Change angle between repetitions | Q Quit", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 255), 1)

            if not hand_found:
                cv2.putText(frame, "No hand detected", (35, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("ISL Dataset Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                if collected_count < SAMPLES_PER_SIGN:
                    collecting = not collecting
                else:
                    print("Target sample count already reached for this label.")

            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSaved {collected_count} samples for label: {label}")
    print(f"Dataset file: {DATASET_FILE}")


if __name__ == "__main__":
    main()
