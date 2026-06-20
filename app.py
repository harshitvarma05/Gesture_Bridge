import cv2
import time
import math
import csv
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
import mediapipe as mp
import numpy as np

from gesture_bridge import AlertManager, ContextInterpreter, SafetyAnalyzer, SentenceEngine
from gesture_bridge.temporal import TemporalRecognizer
from gesture_bridge.speech import SpeechService

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Runtime Paths + Text-to-Speech
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
speech_service = SpeechService(rate=160)


def speak(text):
    speech_service.say(text)


# -----------------------------
# Project Configuration
# -----------------------------
PROJECT_TITLE = "GESTURE-BRIDGE"
PROJECT_SUBTITLE = "Distress-Aware Assistive Communication"
SYSTEM_SCOPE_TEXT = "Offline sign recognition, contextual language and covert emergency response"

LOG_FILE_PATH = BASE_DIR / "recognition_log.csv"
MODEL_TASK_FILE = BASE_DIR / "hand_landmarker.task"
PERSONAL_MODEL_FILE = BASE_DIR / "isl_landmark_model.pkl"
PERSONAL_METRICS_FILE = BASE_DIR / "model_metrics.json"

REQUIRED_STABLE_FRAMES = 10
ADD_COOLDOWN_SECONDS = 1.8
RECENT_OUTPUT_LIMIT = 6
MAX_HANDS = 2
WINDOW_TITLE = "Gesture-Bridge | Distress-Aware Silent SOS"
DEFAULT_UI_SCALE_PERCENT = 72
MIN_UI_SCALE_PERCENT = 40
UI_SLIDER_X1 = 1010
UI_SLIDER_X2 = 1245
UI_SLIDER_Y = 35

DATASET_TRAINED_SIGNS = [
    "Hello", "Thank You", "Fever", "Injury", "Drink", "Cry",
    "Come", "Give", "Busy", "Break", "Maybe", "Wrong",
]

SUPPORTED_SIGNS = [
    ("Hello", "Open palm / salute-style greeting"),
    ("Help", "Two-finger help gesture for prototype demo"),
    ("Yes", "Thumb raised"),
    ("No", "Pinky raised"),
    ("Thank You", "Thumb and index close"),
    ("Water", "Three fingers raised"),
    ("Doctor", "Index finger raised"),
    ("Emergency", "Closed fist"),
    ("Washroom", "Four fingers raised"),
    ("Pain", "Personalized pain gesture"),
    ("Chest", "Point toward the chest area"),
    ("Medicine", "Personalized medicine gesture"),
    ("Call", "Phone-call gesture"),
    ("Caregiver", "Personalized caregiver gesture"),
    ("Fever", "Dataset-trained full movement"),
    ("Injury", "Dataset-trained full movement"),
    ("Drink", "Dataset-trained full movement"),
    ("Cry", "Dataset-trained full movement"),
    ("Come", "Dataset-trained full movement"),
    ("Give", "Dataset-trained full movement"),
    ("Busy", "Dataset-trained full movement"),
    ("Break", "Dataset-trained full movement"),
    ("Maybe", "Dataset-trained full movement"),
    ("Wrong", "Dataset-trained full movement"),
]

SIGN_INSTRUCTIONS = {sign_name.lower(): instruction for sign_name, instruction in SUPPORTED_SIGNS}


def load_personal_model():
    if not os.path.exists(PERSONAL_MODEL_FILE):
        return None, "No personalized model"
    allow_provisional = os.getenv("GESTURE_BRIDGE_ALLOW_PROVISIONAL_MODEL", "0") == "1"
    if not allow_provisional:
        try:
            with open(PERSONAL_METRICS_FILE, "r", encoding="utf-8") as metrics_file:
                metrics = json.load(metrics_file)
            if not metrics.get("validation_reliable", False):
                return None, "Provisional model disabled (incomplete gesture coverage)"
            if metrics.get("missing_expected_labels"):
                return None, "Personalized model disabled (missing labels)"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None, "Personalized model disabled (missing validation report)"
    try:
        with open(PERSONAL_MODEL_FILE, "rb") as model_file:
            return pickle.load(model_file), "Personalized model ready"
    except (OSError, pickle.UnpicklingError, AttributeError, EOFError):
        return None, "Personalized model could not be loaded"


PERSONAL_MODEL, PERSONAL_MODEL_STATUS = load_personal_model()


# -----------------------------
# Logging
# -----------------------------
def rotate_log_if_needed(path, max_bytes=5 * 1024 * 1024):
    path = Path(path)
    try:
        if path.stat().st_size >= max_bytes:
            archived = path.with_name(path.name + ".1")
            os.replace(path, archived)
    except FileNotFoundError:
        pass


def append_recognition_log(raw_detection, confirmed_output, response_time):
    rotate_log_if_needed(LOG_FILE_PATH)
    file_exists = False

    try:
        with open(LOG_FILE_PATH, "r", newline=""):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(LOG_FILE_PATH, "a", newline="") as log_file:
        writer = csv.writer(log_file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "raw_detection",
                "confirmed_output",
                "response_time_seconds",
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            raw_detection,
            confirmed_output,
            f"{response_time:.2f}" if response_time is not None else "",
        ])


# -----------------------------
# MediaPipe Tasks Hand Setup
# -----------------------------
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_TASK_FILE)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)


# -----------------------------
# Drawing Helpers
# -----------------------------
def draw_hand_landmarks(frame, hand_landmarks):
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
        cv2.line(frame, start_point, end_point, (255, 196, 46), 2, cv2.LINE_AA)

    for landmark in hand_landmarks:
        center = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, center, 4, (255, 245, 232), -1, cv2.LINE_AA)


UI_BG = (24, 27, 36)
UI_CARD = (34, 38, 49)
UI_CARD_ALT = (41, 46, 59)
UI_BORDER = (69, 77, 96)
UI_TEXT = (243, 245, 249)
UI_MUTED = (163, 170, 186)
UI_ACCENT = (255, 188, 50)
UI_SUCCESS = (102, 211, 137)
UI_WARNING = (60, 184, 255)
UI_DANGER = (83, 83, 244)

TEMPORAL_GUIDE = DATASET_TRAINED_SIGNS
HEURISTIC_GUIDE = ["Help", "Yes", "No", "Water", "Doctor", "Emergency", "Washroom"]
CUSTOM_GUIDE = ["Pain", "Chest", "Medicine", "Call", "Caregiver"]


def draw_rounded_rect(frame, x1, y1, x2, y2, color, radius=14, border=None):
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for center in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius), (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
    if border:
        cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), border, 1, cv2.LINE_AA)
        cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), border, 1, cv2.LINE_AA)
        cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), border, 1, cv2.LINE_AA)
        cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), border, 1, cv2.LINE_AA)


def draw_filled_box(frame, x1, y1, x2, y2):
    draw_rounded_rect(frame, x1, y1, x2, y2, UI_CARD, radius=12, border=UI_BORDER)


def draw_text_line(frame, text, x, y, scale=0.55, thickness=1, color=UI_TEXT):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_status_chip(frame, label, value, x, y, width=230):
    draw_filled_box(frame, x, y, x + width, y + 42)
    draw_text_line(frame, label.upper(), x + 12, y + 16, scale=0.38, thickness=1)
    draw_text_line(frame, value, x + 12, y + 34, scale=0.50, thickness=1)


def draw_section_title(frame, text, x, y):
    draw_text_line(frame, text.upper(), x, y, scale=0.38, thickness=1, color=UI_MUTED)


def clipped(text, limit):
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def draw_progress(frame, x1, y, x2, value, color=UI_ACCENT):
    value = max(0.0, min(float(value), 1.0))
    cv2.line(frame, (x1, y), (x2, y), UI_BORDER, 6, cv2.LINE_AA)
    cv2.line(frame, (x1, y), (int(x1 + (x2 - x1) * value), y), color, 6, cv2.LINE_AA)


def draw_project_interface(
    frame,
    detected_text,
    raw_detected_text,
    stability_value,
    required_stable_frames,
    communication_output,
    recent_output_text,
    response_time_text,
    total_confirmed_outputs,
    auto_mode,
    show_guide,
    show_testing_metrics,
    app_mode,
    text_to_sign_query,
    text_to_sign_result,
    recognition_method,
    prediction_confidence,
    safety_state,
    context_mode,
    alert_status,
    fps,
    sentence_output,
):
    _, frame_width, _ = frame.shape
    left, right = 18, 448
    shown_confirmed = "None yet" if detected_text in ("Unknown", "No hand detected") else detected_text
    shown_raw = "Show a gesture" if raw_detected_text == "No hand detected" else raw_detected_text
    distress_color = UI_SUCCESS if safety_state.level == "CALM" else UI_WARNING if safety_state.level == "ELEVATED" else UI_DANGER

    # Brand bar
    draw_rounded_rect(frame, 18, 16, frame_width - 18, 102, UI_BG, radius=18, border=UI_BORDER)
    cv2.circle(frame, (49, 48), 17, UI_ACCENT, -1, cv2.LINE_AA)
    draw_text_line(frame, "G", 40, 56, scale=0.62, thickness=2, color=UI_BG)
    draw_text_line(frame, PROJECT_TITLE, 78, 48, scale=0.70, thickness=2)
    draw_text_line(frame, PROJECT_SUBTITLE, 78, 73, scale=0.43, color=UI_MUTED)
    draw_rounded_rect(frame, frame_width - 680, 35, frame_width - 507, 72, UI_CARD_ALT, radius=18)
    draw_text_line(frame, clipped(context_mode, 18), frame_width - 662, 59, scale=0.43, color=UI_TEXT)
    draw_rounded_rect(frame, frame_width - 492, 35, frame_width - 280, 72, distress_color, radius=18)
    cv2.circle(frame, (frame_width - 468, 53), 6, UI_BG, -1, cv2.LINE_AA)
    draw_text_line(frame, f"{safety_state.level}  {safety_state.score * 100:.0f}%", frame_width - 453, 59, scale=0.43, thickness=2, color=UI_BG)

    # Recognition card
    draw_rounded_rect(frame, left, 118, right, 258, UI_CARD, radius=16, border=UI_BORDER)
    draw_section_title(frame, "Live recognition", 36, 142)
    draw_text_line(frame, clipped(shown_raw, 38), 36, 178, scale=0.62, thickness=2)
    draw_text_line(frame, f"Confirmed  {clipped(shown_confirmed, 24)}", 36, 207, scale=0.43, color=UI_MUTED)
    draw_text_line(frame, clipped(recognition_method, 34), 36, 234, scale=0.39, color=UI_MUTED)
    draw_progress(frame, 282, 230, 420, prediction_confidence)
    draw_text_line(frame, f"{prediction_confidence * 100:.0f}%", 375, 217, scale=0.36, color=UI_MUTED)

    # Communication card
    draw_rounded_rect(frame, left, 270, right, 388, UI_CARD, radius=16, border=UI_BORDER)
    draw_section_title(frame, "Text to sign" if app_mode == "Text-to-Sign Representation" else "Communication", 36, 294)
    if app_mode == "Text-to-Sign Representation":
        draw_text_line(frame, clipped(text_to_sign_query, 38), 36, 327, scale=0.55, thickness=2)
        draw_text_line(frame, clipped(text_to_sign_result, 48), 36, 358, scale=0.42, color=UI_MUTED)
    else:
        draw_text_line(frame, clipped(communication_output, 39), 36, 327, scale=0.55, thickness=2)
        draw_text_line(frame, clipped(sentence_output, 49), 36, 358, scale=0.41, color=UI_MUTED)

    # Safety card
    draw_rounded_rect(frame, left, 400, right, 532, UI_CARD, radius=16, border=UI_BORDER)
    draw_section_title(frame, "Safety signals", 36, 424)
    draw_text_line(frame, f"{safety_state.level}", 36, 460, scale=0.66, thickness=2, color=distress_color)
    draw_progress(frame, 137, 454, 420, safety_state.score, distress_color)
    if show_testing_metrics:
        metrics = [("SPEED", safety_state.speed), ("TREMOR", safety_state.tremor), ("REPEATS", safety_state.repetition)]
        for index, (label, value) in enumerate(metrics):
            x = 36 + index * 132
            draw_text_line(frame, label, x, 492, scale=0.32, color=UI_MUTED)
            shown = f"{value:.2f}" if isinstance(value, float) else str(value)
            draw_text_line(frame, shown, x, 516, scale=0.48, thickness=2)
    else:
        reason_text = ", ".join(safety_state.reasons) if safety_state.reasons else "No elevated motion signals"
        draw_text_line(frame, clipped(reason_text, 46), 36, 502, scale=0.41, color=UI_MUTED)

    # Runtime card
    draw_rounded_rect(frame, left, 544, right, 626, UI_CARD, radius=16, border=UI_BORDER)
    draw_section_title(frame, "System", 36, 567)
    runtime = f"{fps:.0f} FPS   {response_time_text}   STABLE {stability_value}/{required_stable_frames}"
    draw_text_line(frame, runtime, 36, 593, scale=0.40)
    status_color = UI_SUCCESS if "ready" in alert_status.lower() or "delivered" in alert_status.lower() else UI_WARNING
    cv2.circle(frame, (40, 611), 5, status_color, -1, cv2.LINE_AA)
    draw_text_line(frame, clipped(alert_status, 45), 53, 617, scale=0.35, color=UI_MUTED)

    # Key hints
    draw_rounded_rect(frame, 18, 642, frame_width - 18, 700, UI_BG, radius=15, border=UI_BORDER)
    hints = [("G", "Guide"), ("X", "Context"), ("P", "Text-Sign"), ("S", "Speak"), ("B", "Calibrate"), ("T", "Details"), ("C", "Clear"), ("Q", "Quit")]
    x = 38
    for key, label in hints:
        draw_rounded_rect(frame, x, 655, x + 30, 686, UI_CARD_ALT, radius=7)
        draw_text_line(frame, key, x + 9, 677, scale=0.39, thickness=2, color=UI_ACCENT)
        draw_text_line(frame, label, x + 38, 677, scale=0.36, color=UI_MUTED)
        x += 130 if label != "Text-Sign" else 158

    if show_guide:
        gx1, gx2 = 470, frame_width - 24
        draw_rounded_rect(frame, gx1, 118, gx2, 590, UI_BG, radius=18, border=UI_BORDER)
        draw_text_line(frame, "Gesture capabilities", gx1 + 24, 153, scale=0.62, thickness=2)
        draw_text_line(frame, "What the current build can recognize and how", gx1 + 24, 179, scale=0.40, color=UI_MUTED)
        groups = [
            ("VIDEO + MOTION MODEL", TEMPORAL_GUIDE, UI_ACCENT),
            ("LANDMARK RULES", HEURISTIC_GUIDE, UI_SUCCESS),
            ("PERSONAL ENROLLMENT REQUIRED", CUSTOM_GUIDE, UI_WARNING),
        ]
        y = 215
        for title, items, color in groups:
            cv2.circle(frame, (gx1 + 29, y - 5), 5, color, -1, cv2.LINE_AA)
            draw_text_line(frame, title, gx1 + 43, y, scale=0.36, thickness=2, color=color)
            for row in range(0, len(items), 4):
                draw_text_line(frame, "   |   ".join(items[row:row + 4]), gx1 + 28, y + 29, scale=0.40)
                y += 26
            y += 27
        draw_rounded_rect(frame, gx1 + 22, 520, gx2 - 22, 568, UI_CARD_ALT, radius=12)
        draw_text_line(frame, "Temporal signs require the complete movement - not only the final pose.", gx1 + 40, 550, scale=0.40, color=UI_MUTED)


def draw_scaled_project_interface(frame, ui_scale, **interface_values):
    """Composite a scaled dashboard without shrinking the camera or landmarks."""
    sentinel = np.array([1, 2, 3], dtype=np.uint8)
    dashboard = np.empty_like(frame)
    dashboard[:] = sentinel
    draw_project_interface(dashboard, **interface_values)

    scale = max(MIN_UI_SCALE_PERCENT / 100.0, min(float(ui_scale), 1.0))
    if scale != 1.0:
        dashboard = cv2.resize(dashboard, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    height, width = dashboard.shape[:2]
    height = min(height, frame.shape[0])
    width = min(width, frame.shape[1])
    dashboard = dashboard[:height, :width]
    mask = np.any(dashboard != sentinel, axis=2)
    target = frame[:height, :width]
    target[mask] = dashboard[mask]


def draw_ui_scale_slider(frame, percent):
    """Small cross-platform slider drawn inside the camera view."""
    draw_rounded_rect(frame, 970, 10, 1262, 64, UI_BG, radius=15, border=UI_BORDER)
    draw_text_line(frame, f"Interface {percent}%", 986, 31, scale=0.37, color=UI_MUTED)
    cv2.line(frame, (UI_SLIDER_X1, UI_SLIDER_Y + 10), (UI_SLIDER_X2, UI_SLIDER_Y + 10), UI_BORDER, 4, cv2.LINE_AA)
    ratio = (percent - MIN_UI_SCALE_PERCENT) / (100 - MIN_UI_SCALE_PERCENT)
    knob_x = int(UI_SLIDER_X1 + ratio * (UI_SLIDER_X2 - UI_SLIDER_X1))
    cv2.line(frame, (UI_SLIDER_X1, UI_SLIDER_Y + 10), (knob_x, UI_SLIDER_Y + 10), UI_ACCENT, 4, cv2.LINE_AA)
    cv2.circle(frame, (knob_x, UI_SLIDER_Y + 10), 8, UI_TEXT, -1, cv2.LINE_AA)


def handle_ui_scale_mouse(event, x, y, flags, state):
    inside = UI_SLIDER_X1 - 12 <= x <= UI_SLIDER_X2 + 12 and UI_SLIDER_Y - 8 <= y <= UI_SLIDER_Y + 28
    if event == cv2.EVENT_LBUTTONDOWN and inside:
        state["dragging"] = True
    elif event == cv2.EVENT_LBUTTONUP:
        state["dragging"] = False

    if state["dragging"] and (event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN):
        ratio = (min(max(x, UI_SLIDER_X1), UI_SLIDER_X2) - UI_SLIDER_X1) / (UI_SLIDER_X2 - UI_SLIDER_X1)
        state["percent"] = int(round(MIN_UI_SCALE_PERCENT + ratio * (100 - MIN_UI_SCALE_PERCENT)))


# -----------------------------
# Geometry + Classification Helpers
# -----------------------------
def distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


# Rotation-invariant finger extension helper
def finger_is_extended(hand_landmarks, mcp_index, pip_index, tip_index):
    """
    Orientation-resistant finger detection.
    Instead of checking only whether the fingertip is visually above the joint,
    this checks whether the fingertip is extended away from the finger base and wrist.
    This works better when the palm, knuckles, or side of the hand faces the camera.
    """
    wrist = hand_landmarks[0]
    mcp = hand_landmarks[mcp_index]
    pip = hand_landmarks[pip_index]
    tip = hand_landmarks[tip_index]

    tip_to_mcp = distance(tip, mcp)
    pip_to_mcp = distance(pip, mcp)
    tip_to_wrist = distance(tip, wrist)
    pip_to_wrist = distance(pip, wrist)

    extended_from_base = tip_to_mcp > pip_to_mcp * 1.25
    extended_from_wrist = tip_to_wrist > pip_to_wrist * 1.05

    return extended_from_base and extended_from_wrist


def get_combined_hand_bbox(hand_landmarks_list, frame_width, frame_height):
    x_values = []
    y_values = []

    for hand_landmarks in hand_landmarks_list:
        x_values.extend([landmark.x for landmark in hand_landmarks])
        y_values.extend([landmark.y for landmark in hand_landmarks])

    if not x_values or not y_values:
        return 0, 0, 0, 0

    x_min = int(min(x_values) * frame_width)
    y_min = int(min(y_values) * frame_height)
    x_max = int(max(x_values) * frame_width)
    y_max = int(max(y_values) * frame_height)

    return x_min, y_min, x_max, y_max


def is_thumb_open(hand_landmarks):
    """
    Robust thumb detection for both palm-facing and knuckle-facing views.
    It uses distance from the palm instead of depending on left/right x-position.
    """
    wrist = hand_landmarks[0]
    thumb_mcp = hand_landmarks[2]
    thumb_ip = hand_landmarks[3]
    thumb_tip = hand_landmarks[4]
    index_mcp = hand_landmarks[5]
    pinky_mcp = hand_landmarks[17]

    palm_width = distance(index_mcp, pinky_mcp)
    if palm_width == 0:
        return False

    thumb_tip_to_wrist = distance(thumb_tip, wrist)
    thumb_ip_to_wrist = distance(thumb_ip, wrist)
    thumb_tip_to_index_mcp = distance(thumb_tip, index_mcp)
    thumb_mcp_to_index_mcp = distance(thumb_mcp, index_mcp)
    thumb_tip_to_pinky_mcp = distance(thumb_tip, pinky_mcp)

    extended_from_wrist = thumb_tip_to_wrist > thumb_ip_to_wrist * 1.04
    extended_from_index = thumb_tip_to_index_mcp > thumb_mcp_to_index_mcp * 1.10
    away_from_palm = min(thumb_tip_to_index_mcp, thumb_tip_to_pinky_mcp) > palm_width * 0.36

    return extended_from_wrist and (extended_from_index or away_from_palm)


def get_finger_status(hand_landmarks, handedness_label):
    """
    Returns which fingers are open.
    Output format: [thumb, index, middle, ring, pinky]

    This version avoids relying only on y-coordinates, so it is less sensitive
    to wrist rotation and works better when knuckles face the camera.
    """
    thumb_open = is_thumb_open(hand_landmarks)
    index_open = finger_is_extended(hand_landmarks, 5, 6, 8)
    middle_open = finger_is_extended(hand_landmarks, 9, 10, 12)
    ring_open = finger_is_extended(hand_landmarks, 13, 14, 16)
    pinky_open = finger_is_extended(hand_landmarks, 17, 18, 20)

    return [thumb_open, index_open, middle_open, ring_open, pinky_open]


def classify_gesture(hand_landmarks, handedness_label):
    fingers = get_finger_status(hand_landmarks, handedness_label)
    thumb, index, middle, ring, pinky = fingers

    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    thumb_index_distance = distance(thumb_tip, index_tip)

    if thumb_index_distance < 0.075 and middle and ring and pinky:
        return "Thank You"

    if thumb and index and middle and ring and pinky:
        return "Hello"

    if not thumb and not index and not middle and not ring and not pinky:
        return "Emergency"

    if not thumb and index and middle and not ring and not pinky:
        return "Help"

    if not thumb and index and not middle and not ring and not pinky:
        return "Doctor"

    if thumb and not index and not middle and not ring and not pinky:
        return "Yes"

    if not thumb and not index and not middle and not ring and pinky:
        return "No"

    if not thumb and index and middle and ring and not pinky:
        return "Water"

    if not thumb and index and middle and ring and pinky:
        return "Washroom"

    return "Unknown"


def landmarks_to_feature_vector(hand_landmarks_list, feature_count):
    """Build either the legacy 63-value or current 126-value normalized vector."""
    hand_count = 1 if feature_count == 63 else 2
    hands = sorted(
        hand_landmarks_list[:hand_count],
        key=lambda hand: sum(point.x for point in hand) / len(hand),
    )
    points = [point for hand in hands for point in hand]
    if not points:
        return np.zeros(feature_count, dtype=np.float32)
    origin = np.array([
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
        sum(point.z for point in points) / len(points),
    ])
    values = []
    for hand in hands:
        for point in hand:
            values.extend(np.array([point.x, point.y, point.z]) - origin)
    values.extend([0.0] * (feature_count - len(values)))
    vector = np.asarray(values[:feature_count], dtype=np.float32)
    maximum = np.max(np.abs(vector))
    return vector / maximum if maximum > 0 else vector


def predict_sign(hand_landmarks_list, handedness_label):
    if not hand_landmarks_list:
        return "No hand detected", 0.0, "MediaPipe landmark classifier"

    if PERSONAL_MODEL is not None:
        try:
            feature_count = int(PERSONAL_MODEL.n_features_in_)
            vector = landmarks_to_feature_vector(hand_landmarks_list, feature_count)
            probabilities = PERSONAL_MODEL.predict_proba([vector])[0]
            best_index = int(np.argmax(probabilities))
            confidence = float(probabilities[best_index])
            ordered = np.sort(probabilities)
            margin = confidence - float(ordered[-2]) if len(ordered) > 1 else confidence
            prediction = str(PERSONAL_MODEL.classes_[best_index])
            if confidence >= 0.72 and margin >= 0.18:
                return prediction.title(), confidence, "Personalized model"
        except (AttributeError, ValueError):
            pass

    prediction = classify_gesture(hand_landmarks_list[0], handedness_label)
    confidence = 0.92 if prediction != "Unknown" else 0.0
    return prediction, confidence, "MediaPipe landmark classifier"


# -----------------------------
# Main App
# -----------------------------
def main():
    if not MODEL_TASK_FILE.exists():
        print(f"Startup error: missing MediaPipe model: {MODEL_TASK_FILE}")
        return

    camera_index = int(os.getenv("GESTURE_BRIDGE_CAMERA_INDEX", "0"))
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(
            f"Startup error: camera {camera_index} could not be opened. "
            "Check camera permissions or set GESTURE_BRIDGE_CAMERA_INDEX."
        )
        return

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 720)
    ui_scale_state = {"percent": DEFAULT_UI_SCALE_PERCENT, "dragging": False}
    cv2.setMouseCallback(WINDOW_TITLE, handle_ui_scale_mouse, ui_scale_state)

    stable_gesture = "Unknown"
    previous_detected_text = "Unknown"
    stable_frame_count = 0
    required_stable_frames = REQUIRED_STABLE_FRAMES

    recent_outputs = []
    total_confirmed_outputs = 0
    response_start_time = None
    last_response_time = None

    show_guide = False
    show_testing_metrics = True
    auto_mode = True
    app_mode = "Sign Recognition"
    text_to_sign_query = "None"
    text_to_sign_result = "Press P and enter a supported phrase"
    last_added_gesture = "Unknown"
    last_added_time = 0
    add_cooldown_seconds = ADD_COOLDOWN_SECONDS
    auto_speak_on_add = True

    recognition_method = "MediaPipe landmark classifier"
    prediction_confidence = 0.0
    safety_analyzer = SafetyAnalyzer(profile_path=str(BASE_DIR / "gesture_profile.json"))
    alert_manager = AlertManager(log_path=str(BASE_DIR / "emergency_alerts.jsonl"))
    context_interpreter = ContextInterpreter()
    sentence_engine = SentenceEngine()
    temporal_recognizer = TemporalRecognizer(model_path=str(BASE_DIR / "isl_temporal_model.pkl"))
    safety_state = safety_analyzer.update([])
    last_alert_time = 0.0
    high_distress_since = None
    last_frame_time = time.perf_counter()
    fps = 0.0
    clarification_candidate = None

    print("Gesture-Bridge distress-aware assistive system started.")
    print(f"Recognition model: {PERSONAL_MODEL_STATUS}")
    print(f"Temporal model: {'Ready' if temporal_recognizer.available else 'Not installed'}")
    print("Controls: X = context, Y = confirm suggestion, P = text-to-sign, R = recognition, S = speak, C = clear, G = guide, T = metrics, Q = quit.")
    print(f"Capabilities: {len(TEMPORAL_GUIDE)} temporal, {len(HEURISTIC_GUIDE)} rule-based, {len(CUSTOM_GUIDE)} enrollment-required signs.")

    frame_timestamp_ms = 0

    try:
        landmarker_instance = HandLandmarker.create_from_options(options)
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Startup error: MediaPipe hand tracker failed: {error}")
        cap.release()
        cv2.destroyAllWindows()
        speech_service.close()
        return

    with landmarker_instance as landmarker:
        while True:
            success, frame = cap.read()

            if not success:
                print("Error: Could not read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1280, 720))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms += 33
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            raw_detected_text = "No hand detected"
            display_detected_text = raw_detected_text
            prediction_confidence = 0.0
            detected_hands = []
            clarification_candidate = None

            if result.hand_landmarks and result.handedness:
                detected_hands = result.hand_landmarks[:MAX_HANDS]
                handedness_label = result.handedness[0][0].category_name

                for hand_landmarks in detected_hands:
                    draw_hand_landmarks(frame, hand_landmarks)

                frame_height, frame_width, _ = frame.shape
                x_min, y_min, x_max, y_max = get_combined_hand_bbox(detected_hands, frame_width, frame_height)
                cv2.rectangle(
                    frame,
                    (x_min - 15, y_min - 15),
                    (x_max + 15, y_max + 15),
                    (255, 255, 255),
                    2,
                )

                temporal_prediction = temporal_recognizer.update(detected_hands)
                if temporal_prediction and temporal_prediction[0] == "Still":
                    raw_detected_text, prediction_confidence, recognition_method = predict_sign(
                        detected_hands,
                        handedness_label,
                    )
                elif temporal_prediction and temporal_prediction[1] >= 0.35:
                    raw_detected_text, prediction_confidence = temporal_prediction
                    recognition_method = "Temporal motion model"
                else:
                    raw_detected_text, prediction_confidence, recognition_method = predict_sign(
                        detected_hands,
                        handedness_label,
                    )
                    if raw_detected_text == "Unknown" and temporal_recognizer.last_debug:
                        debug = temporal_recognizer.last_debug
                        display_detected_text = (
                            f"Candidate {debug['candidate']} {debug['confidence']:.2f} - "
                            f"{debug['rejection'] or 'keep moving'}"
                        )
                if recognition_method == "Personalized model" and prediction_confidence < 0.78:
                    clarification_candidate = raw_detected_text
                    display_detected_text = f"Did you mean {clarification_candidate}? Press Y"
                    raw_detected_text = "Unknown"
                else:
                    clarification_candidate = None
                    if not (raw_detected_text == "Unknown" and temporal_recognizer.last_debug):
                        display_detected_text = raw_detected_text

            current_time = time.time()
            if not detected_hands:
                temporal_recognizer.update([])
            safety_state = safety_analyzer.update(detected_hands, time.monotonic())
            frame_now = time.perf_counter()
            instantaneous_fps = 1.0 / max(frame_now - last_frame_time, 0.001)
            fps = instantaneous_fps if fps == 0 else fps * 0.90 + instantaneous_fps * 0.10
            last_frame_time = frame_now

            # Stabilization logic: absence/unknown are observations, never outputs.
            valid_raw_gesture = raw_detected_text not in ["Unknown", "No hand detected"]
            active_required_frames = 5 if (
                valid_raw_gesture and recognition_method == "Temporal motion model"
            ) else required_stable_frames

            if raw_detected_text == previous_detected_text:
                stable_frame_count += 1
            else:
                stable_frame_count = 0
                previous_detected_text = raw_detected_text

                if raw_detected_text not in ["Unknown", "No hand detected"]:
                    response_start_time = time.time()
                else:
                    response_start_time = None

            if valid_raw_gesture and stable_frame_count >= active_required_frames:
                if stable_gesture != raw_detected_text and response_start_time is not None:
                    last_response_time = time.time() - response_start_time
                stable_gesture = raw_detected_text

            detected_text = stable_gesture
            valid_detected_gesture = detected_text not in ["Unknown", "No hand detected"]

            # Automatic text + speech output
            if auto_mode and valid_detected_gesture:
                is_new_gesture = detected_text != last_added_gesture
                cooldown_finished = current_time - last_added_time >= add_cooldown_seconds

                if is_new_gesture and cooldown_finished:
                    recent_outputs.append(detected_text)
                    if len(recent_outputs) > RECENT_OUTPUT_LIMIT:
                        recent_outputs.pop(0)

                    total_confirmed_outputs += 1
                    append_recognition_log(raw_detected_text, detected_text, last_response_time)
                    last_added_gesture = detected_text
                    last_added_time = current_time
                    safety_analyzer.note_confirmed_gesture(detected_text, time.monotonic())
                    sentence_engine.add(detected_text)

                    if auto_speak_on_add:
                        speak(detected_text)

            if raw_detected_text in ["Unknown", "No hand detected"]:
                last_added_gesture = "Unknown"

            # Escalate only after sustained distress plus an emergency-related sign.
            if safety_state.level == "HIGH":
                high_distress_since = high_distress_since or current_time
            else:
                high_distress_since = None

            alert_reason = None
            silent_alert = False
            if safety_state.sos_pattern:
                alert_reason = f"Silent SOS: {safety_state.sos_pattern}"
                silent_alert = True
            elif (
                high_distress_since
                and current_time - high_distress_since >= 1.0
                and detected_text in ("Help", "Emergency", "Doctor")
            ):
                alert_reason = f"Distress + repeated {detected_text} gesture"
            elif detected_text == "Emergency" and stable_frame_count == required_stable_frames:
                alert_reason = "Emergency gesture confirmed"

            if alert_reason and current_time - last_alert_time >= 8.0:
                alert_manager.trigger(
                    alert_reason,
                    "Emergency distress detected. Please check on the Gesture-Bridge user.",
                    silent=silent_alert,
                )
                last_alert_time = current_time
                if not silent_alert:
                    speak("Emergency distress detected. Alerting caregiver.")

            if safety_state.level == "SOS":
                communication_output = "SILENT SOS SENT TO CAREGIVER"
            elif safety_state.level == "HIGH" and valid_detected_gesture:
                communication_output = "Emergency distress detected. Alerting caregiver."
            else:
                communication_output = context_interpreter.interpret(detected_text) if valid_detected_gesture else "Waiting for a stable sign..."
            recent_output_text = " | ".join(recent_outputs) if recent_outputs else "No confirmed outputs yet"
            response_time_text = f"{last_response_time:.2f} sec" if last_response_time is not None else "--"
            stability_value = min(stable_frame_count, active_required_frames)

            ui_scale_percent = ui_scale_state["percent"]
            draw_scaled_project_interface(
                frame,
                ui_scale_percent / 100.0,
                detected_text=detected_text,
                raw_detected_text=display_detected_text,
                stability_value=stability_value,
                required_stable_frames=active_required_frames,
                communication_output=communication_output,
                recent_output_text=recent_output_text,
                response_time_text=response_time_text,
                total_confirmed_outputs=total_confirmed_outputs,
                auto_mode=auto_mode,
                show_guide=show_guide,
                show_testing_metrics=show_testing_metrics,
                app_mode=app_mode,
                text_to_sign_query=text_to_sign_query,
                text_to_sign_result=text_to_sign_result,
                recognition_method=recognition_method,
                prediction_confidence=prediction_confidence,
                safety_state=safety_state,
                context_mode=context_interpreter.context,
                alert_status=alert_manager.last_status,
                fps=fps,
                sentence_output=sentence_engine.compose(),
            )
            draw_ui_scale_slider(frame, ui_scale_percent)

            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("p"):
                app_mode = "Text-to-Sign Representation"
                print("\nSupported phrases:")
                for sign_name, instruction in SUPPORTED_SIGNS:
                    print(f"- {sign_name}: {instruction}")

                phrase = input("Enter phrase for text-to-sign representation: ").strip()
                normalized_phrase = phrase.lower()

                if normalized_phrase in SIGN_INSTRUCTIONS:
                    text_to_sign_query = phrase
                    text_to_sign_result = SIGN_INSTRUCTIONS[normalized_phrase]
                    speak(f"{phrase}. Sign representation: {text_to_sign_result}")
                else:
                    text_to_sign_query = phrase if phrase else "None"
                    text_to_sign_result = "Unsupported phrase. Press G to view supported signs."
                    speak("Unsupported phrase")

            elif key == ord("r"):
                app_mode = "Sign Recognition"
                text_to_sign_query = "None"
                text_to_sign_result = "Press P and enter a supported phrase"
                speak("Sign recognition mode")

            elif key == ord("s"):
                if recent_outputs:
                    full_sentence = " ".join(recent_outputs)
                    speak(full_sentence)
                else:
                    speak("No confirmed outputs to speak")

            elif key == ord("c"):
                recent_outputs.clear()
                sentence_engine.clear()
                safety_analyzer.reset()
                temporal_recognizer.reset()
                total_confirmed_outputs = 0
                last_response_time = None
                text_to_sign_query = "None"
                text_to_sign_result = "Press P and enter a supported phrase"
                app_mode = "Sign Recognition"
                speak("Outputs cleared")

            elif key == ord("x"):
                new_context = context_interpreter.cycle()
                speak(f"{new_context} context")

            elif key == ord("y") and clarification_candidate:
                confirmed = clarification_candidate
                recent_outputs.append(confirmed)
                recent_outputs[:] = recent_outputs[-RECENT_OUTPUT_LIMIT:]
                sentence_engine.add(confirmed)
                total_confirmed_outputs += 1
                append_recognition_log("clarification", confirmed, None)
                speak(context_interpreter.interpret(confirmed))

            elif key == ord("b"):
                safety_analyzer.calibrate_from_window()
                speak(safety_analyzer.calibration_status)

            elif key == ord("a"):
                if alert_manager.acknowledge():
                    speak("Emergency alert acknowledged")

            elif key == ord("k"):
                if alert_manager.cancel():
                    speak("Emergency alert cancelled")

            elif key in (ord("-"), ord("_")):
                ui_scale_state["percent"] = max(MIN_UI_SCALE_PERCENT, ui_scale_state["percent"] - 2)

            elif key in (ord("+"), ord("=")):
                ui_scale_state["percent"] = min(100, ui_scale_state["percent"] + 2)

            elif key == ord("g"):
                show_guide = not show_guide

            elif key == ord("t"):
                show_testing_metrics = not show_testing_metrics

            elif key == ord("m"):
                auto_mode = not auto_mode
                mode_text = "Auto mode" if auto_mode else "Manual mode"
                speak(mode_text)

            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    speech_service.close()


if __name__ == "__main__":
    main()
