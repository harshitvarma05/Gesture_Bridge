import cv2
import time
import math
import csv
import json
import os
import pickle
import platform
from datetime import datetime
from pathlib import Path
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks.python.components import processors

from gesture_bridge import AlertManager, ContextInterpreter, GestureDebouncer, SafetyAnalyzer, SentenceEngine
from gesture_bridge import __version__
from gesture_bridge.speech import SpeechService
from gesture_bridge.emergency import EmergencyController, select_alert_trigger
from gesture_bridge.telemetry import SessionTelemetry
from gesture_bridge.config import env_float, env_int, load_env_file
from gesture_bridge.camera import open_camera

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Runtime Paths + Text-to-Speech
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
load_env_file(BASE_DIR / ".env")
speech_service = SpeechService(rate=178)


def speak(text, urgent=False):
    speech_service.say(text, interrupt=urgent)


# -----------------------------
# Project Configuration
# -----------------------------
PROJECT_TITLE = "GESTURE-BRIDGE"
PROJECT_SUBTITLE = "Distress-Aware Assistive Communication"
SYSTEM_SCOPE_TEXT = "Offline sign recognition, contextual language and covert emergency response"

LOG_FILE_PATH = BASE_DIR / "recognition_log.csv"
MODEL_TASK_FILE = BASE_DIR / "gesture_recognizer.task"
PERSONAL_MODEL_FILE = BASE_DIR / "isl_landmark_model.pkl"
PERSONAL_METRICS_FILE = BASE_DIR / "model_metrics.json"

RECENT_OUTPUT_LIMIT = 6
MAX_HANDS = 2
WINDOW_TITLE = "Gesture-Bridge | Distress-Aware Silent SOS"
DEFAULT_UI_SCALE_PERCENT = 150 if platform.system() in {"Darwin", "Windows"} else 100
MIN_UI_SCALE_PERCENT = 40
MAX_UI_SCALE_PERCENT = 150
LOGICAL_UI_WIDTH = 1280
LOGICAL_UI_HEIGHT = 720
DEFAULT_TEXT_RENDERER = "pillow" if platform.system() in {"Darwin", "Windows"} else "opencv"
TEXT_RENDERER = os.getenv("GESTURE_BRIDGE_TEXT_RENDERER", DEFAULT_TEXT_RENDERER).strip().lower()

SUPPORTED_SIGNS = [
    ("Hello", "Open palm"),
    ("Help", "Victory / V sign"),
    ("Yes", "Thumb up"),
    ("No", "Thumb down"),
    ("Doctor", "Point one finger upward"),
    ("Emergency", "Closed fist"),
    ("Caregiver", "I-love-you hand sign"),
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
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
VisionRunningMode = vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_TASK_FILE)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=0.55,
    min_hand_presence_confidence=0.55,
    min_tracking_confidence=0.55,
    canned_gesture_classifier_options=processors.ClassifierOptions(
        max_results=1,
        score_threshold=0.55,
    ),
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


UI_BG = (32, 22, 15)
UI_CARD = (48, 34, 24)
UI_CARD_ALT = (61, 46, 32)
UI_BORDER = (90, 70, 52)
UI_TEXT = (243, 245, 249)
UI_MUTED = (163, 170, 186)
UI_ACCENT = (255, 188, 50)
UI_SUCCESS = (102, 211, 137)
UI_WARNING = (60, 184, 255)
UI_DANGER = (83, 83, 244)

_TEXT_BATCH = None
_FONT_CACHE = {}
_GLYPH_CACHE = {}
_FONT_CANDIDATES = [
    "/System/Library/Fonts/SFNS.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]


def begin_text_batch():
    global _TEXT_BATCH
    _TEXT_BATCH = []


def end_text_batch():
    global _TEXT_BATCH
    commands = _TEXT_BATCH or []
    _TEXT_BATCH = None
    return commands


def get_ui_font(size):
    size = max(9, int(size))
    if size not in _FONT_CACHE:
        font_path = next((path for path in _FONT_CANDIDATES if Path(path).exists()), None)
        _FONT_CACHE[size] = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
    return _FONT_CACHE[size]


def render_text_commands(frame, commands, coordinate_scale=1.0, offset_x=0, offset_y=0):
    if not commands:
        return
    if TEXT_RENDERER != "pillow":
        for command in commands:
            x = round(command["x"] * coordinate_scale) + offset_x
            y = round(command["y"] * coordinate_scale) + offset_y
            cv2.putText(
                frame, command["text"], (x + 1, y + 1), cv2.FONT_HERSHEY_DUPLEX,
                max(0.25, command["scale"] * coordinate_scale), (8, 12, 18),
                max(2, round((command["thickness"] + 1) * coordinate_scale)), cv2.LINE_AA,
            )
            cv2.putText(
                frame, command["text"], (x, y), cv2.FONT_HERSHEY_DUPLEX,
                max(0.25, command["scale"] * coordinate_scale), command["color"],
                max(1, round(command["thickness"] * coordinate_scale)), cv2.LINE_AA,
            )
        return
    for command in commands:
        size = max(9, round(28 * command["scale"] * coordinate_scale))
        x = round(command["x"] * coordinate_scale) + offset_x
        y = round(command["y"] * coordinate_scale) + offset_y
        b, g, r = command["color"]
        stroke = 1
        cache_key = (command["text"], size, r, g, b, stroke)
        cached = _GLYPH_CACHE.get(cache_key)
        if cached is None:
            font = get_ui_font(size)
            probe = Image.new("RGBA", (1, 1))
            probe_draw = ImageDraw.Draw(probe)
            left, top, right, bottom = probe_draw.textbbox(
                (0, 0), command["text"], font=font, anchor="ls", stroke_width=stroke
            )
            width, height = max(1, right - left), max(1, bottom - top)
            glyph = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            ImageDraw.Draw(glyph).text(
                (-left, -top), command["text"], font=font, fill=(r, g, b, 255),
                anchor="ls", stroke_width=stroke, stroke_fill=(8, 12, 18, 235),
            )
            cached = (np.asarray(glyph), left, top)
            if len(_GLYPH_CACHE) > 512:
                _GLYPH_CACHE.clear()
            _GLYPH_CACHE[cache_key] = cached

        glyph, left, top = cached
        x1, y1 = x + left, y + top
        x2, y2 = x1 + glyph.shape[1], y1 + glyph.shape[0]
        frame_x1, frame_y1 = max(0, x1), max(0, y1)
        frame_x2, frame_y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if frame_x1 >= frame_x2 or frame_y1 >= frame_y2:
            continue
        glyph_x1, glyph_y1 = frame_x1 - x1, frame_y1 - y1
        glyph_x2 = glyph_x1 + (frame_x2 - frame_x1)
        glyph_y2 = glyph_y1 + (frame_y2 - frame_y1)
        source = glyph[glyph_y1:glyph_y2, glyph_x1:glyph_x2]
        alpha = source[..., 3:4].astype(np.float32) / 255.0
        source_bgr = source[..., :3][..., ::-1].astype(np.float32)
        target = frame[frame_y1:frame_y2, frame_x1:frame_x2].astype(np.float32)
        frame[frame_y1:frame_y2, frame_x1:frame_x2] = (source_bgr * alpha + target * (1.0 - alpha)).astype(np.uint8)

POSE_GESTURES = [
    ("Hello", "Open palm", "Says Hello"),
    ("Help", "Victory / V sign", "Says I need help"),
    ("Yes", "Thumb up", "Confirms Yes"),
    ("No", "Thumb down", "Confirms No"),
    ("Doctor", "Point one finger upward", "Requests a doctor"),
    ("Emergency", "Closed fist", "Starts emergency countdown"),
    ("Caregiver", "I-love-you hand sign", "Requests caregiver"),
]
PRETRAINED_GESTURE_MAP = {
    "Open_Palm": "Hello",
    "Victory": "Help",
    "Thumb_Up": "Yes",
    "Thumb_Down": "No",
    "Pointing_Up": "Doctor",
    "Closed_Fist": "Emergency",
    "ILoveYou": "Caregiver",
}
HEURISTIC_ACTIONS = [(name, action) for name, _, action in POSE_GESTURES]
SOS_GESTURES = [
    ("3 taps", "Tap thumb and index 3 times"),
    ("Fist/open", "Close-open fist twice"),
    ("Thumb rub", "Rub thumb repeatedly"),
    ("Palm pulse", "Open palm toward/away from camera"),
]
CUSTOM_ACTIONS = [
    ("Pain", "Personal enrollment"), ("Chest", "Personal enrollment"),
    ("Medicine", "Personal enrollment"), ("Call", "Personal enrollment"),
    ("Caregiver", "Personal enrollment"),
]
TEMPORAL_GUIDE = []
HEURISTIC_GUIDE = [name for name, _ in HEURISTIC_ACTIONS]
CUSTOM_GUIDE = [name for name, _ in CUSTOM_ACTIONS]


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
    command = {
        "text": str(text), "x": x, "y": y, "scale": scale,
        "thickness": thickness, "color": color,
    }
    if _TEXT_BATCH is not None:
        _TEXT_BATCH.append(command)
        return
    render_text_commands(frame, [command])


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
    speech_status,
    active_tab,
    ui_state=None,
    alert_pending_seconds=None,
):
    owns_text_batch = _TEXT_BATCH is None
    if owns_text_batch:
        begin_text_batch()
    _, frame_width, _ = frame.shape
    shown_confirmed = "None yet" if detected_text in ("Unknown", "No hand detected") else detected_text
    shown_raw = "Show a gesture" if raw_detected_text == "No hand detected" else raw_detected_text
    distress_color = UI_SUCCESS if safety_state.level == "CALM" else UI_WARNING if safety_state.level == "ELEVATED" else UI_DANGER

    # A single tabbed detail panel keeps the live camera unobstructed.
    card_text_start = len(_TEXT_BATCH)
    dock_top = 640 if active_tab == "camera" else 486
    draw_rounded_rect(frame, 0, dock_top, frame_width, 720, UI_BG, radius=18)
    tabs = [("recognition", "1", "Recognition"), ("communication", "2", "Voice & sentence"), ("safety", "3", "Safety & alerts")]
    tab_boxes = {}
    tab_x = 14
    tab_y = 650 if active_tab == "camera" else 496
    for tab_id, key_name, label in tabs:
        width = 212
        box = (tab_x, tab_y, tab_x + width, tab_y + 42)
        tab_boxes[tab_id] = box
        selected = active_tab == tab_id
        key_color = UI_ACCENT
        text_color = UI_TEXT
        draw_text_line(frame, key_name, tab_x + 16, tab_y + 27, scale=0.38, thickness=2, color=key_color)
        draw_text_line(frame, label, tab_x + 43, tab_y + 27, scale=0.38, thickness=2, color=text_color)
        if selected:
            cv2.line(frame, (tab_x + 10, tab_y + 38), (tab_x + width - 10, tab_y + 38), UI_ACCENT, 3, cv2.LINE_AA)
        tab_x += width + 10
    hide_box = (tab_x, tab_y, tab_x + 176, tab_y + 42)
    tab_boxes["camera"] = hide_box
    selected = active_tab == "camera"
    draw_text_line(frame, "0", tab_x + 16, tab_y + 27, scale=0.38, thickness=2, color=UI_ACCENT)
    draw_text_line(frame, "Camera only", tab_x + 43, tab_y + 27, scale=0.38, thickness=2, color=UI_TEXT)
    if selected:
        cv2.line(frame, (tab_x + 10, tab_y + 38), (tab_x + 166, tab_y + 38), UI_ACCENT, 3, cv2.LINE_AA)
    if ui_state is not None:
        ui_state["tab_boxes"] = tab_boxes

    panel_y1, panel_y2 = 548, 638
    if active_tab != "camera":
        cv2.line(frame, (18, panel_y1), (1262, panel_y1), UI_BORDER, 1, cv2.LINE_AA)

    if active_tab == "recognition":
        draw_section_title(frame, f"Live recognition  |  {context_mode}  |  {fps:.0f} FPS", 32, 572)
        draw_text_line(frame, clipped(shown_raw, 30), 32, 611, scale=0.66, thickness=2)
        draw_text_line(frame, "CONFIRMED", 350, 573, scale=0.31, color=UI_MUTED)
        draw_text_line(frame, clipped(shown_confirmed, 22), 350, 608, scale=0.52, thickness=2)
        draw_text_line(frame, "HOLD", 610, 573, scale=0.31, color=UI_MUTED)
        draw_progress(frame, 610, 605, 802, stability_value / max(required_stable_frames, 1), UI_ACCENT)
        draw_text_line(frame, f"{stability_value}%", 748, 593, scale=0.32, color=UI_MUTED)
        draw_text_line(frame, "MODEL CONFIDENCE", 835, 573, scale=0.31, color=UI_MUTED)
        draw_progress(frame, 835, 605, 1030, prediction_confidence, UI_SUCCESS)
        draw_text_line(frame, f"{prediction_confidence * 100:.0f}%", 978, 593, scale=0.32, color=UI_MUTED)
        draw_text_line(frame, clipped(recognition_method, 25), 1054, 607, scale=0.32, color=UI_MUTED)

    elif active_tab == "communication":
        section_name = "Text to sign" if app_mode == "Text-to-Sign Representation" else "Voice output"
        draw_section_title(frame, f"{section_name}  |  {context_mode}", 32, 572)
        primary = text_to_sign_query if app_mode == "Text-to-Sign Representation" else communication_output
        secondary = text_to_sign_result if app_mode == "Text-to-Sign Representation" else sentence_output
        draw_text_line(frame, clipped(primary, 65), 32, 610, scale=0.65, thickness=2)
        draw_text_line(frame, "SENTENCE", 590, 572, scale=0.31, color=UI_MUTED)
        draw_text_line(frame, clipped(secondary, 55), 590, 604, scale=0.39, color=UI_TEXT)
        interaction_hint = "Release briefly to repeat" if shown_confirmed != "None yet" else f"Hold steady {stability_value}%"
        draw_text_line(frame, f"{interaction_hint}  |  Voice {speech_status}  |  {response_time_text}", 590, 626, scale=0.31, color=UI_MUTED)

    elif active_tab == "safety":
        draw_section_title(frame, "Safety state", 32, 572)
        draw_text_line(frame, safety_state.level, 32, 612, scale=0.68, thickness=2, color=distress_color)
        draw_progress(frame, 175, 605, 390, safety_state.score, distress_color)
        draw_text_line(frame, f"{safety_state.score * 100:.0f}%", 335, 592, scale=0.32, color=UI_MUTED)
        if show_testing_metrics:
            detail = f"Speed {safety_state.speed:.2f}   Tremor {safety_state.tremor:.2f}   Repeats {safety_state.repetition}"
        else:
            detail = ", ".join(safety_state.reasons) if safety_state.reasons else "No elevated motion signals"
        draw_text_line(frame, "DISTRESS SIGNALS", 430, 572, scale=0.31, color=UI_MUTED)
        draw_text_line(frame, clipped(detail, 52), 430, 608, scale=0.39)
        status_color = UI_SUCCESS if "ready" in alert_status.lower() or "delivered" in alert_status.lower() else UI_WARNING
        draw_text_line(frame, "ALERT DELIVERY", 850, 572, scale=0.31, color=UI_MUTED)
        cv2.circle(frame, (856, 604), 6, status_color, -1, cv2.LINE_AA)
        draw_text_line(frame, clipped(alert_status, 43), 872, 610, scale=0.36)

    footer_text_start = len(_TEXT_BATCH)
    if active_tab != "camera":
        cv2.line(frame, (18, 650), (frame_width - 18, 650), UI_BORDER, 1, cv2.LINE_AA)
        hints = [("G", "Guide"), ("X", "Context"), ("P", "Text-Sign"), ("S", "Speak"), ("B", "Calibrate"), ("T", "Metrics"), ("C", "Clear"), ("K", "Cancel"), ("Q", "Quit")]
        x = 27
        for key, label in hints:
            draw_text_line(frame, key, x + 8, 684, scale=0.36, thickness=2, color=UI_ACCENT)
            draw_text_line(frame, label, x + 34, 683, scale=0.31, color=UI_MUTED)
            x += 137 if label == "Text-Sign" else 124

    if show_guide:
        # Text is batched for speed; remove covered card labels so they cannot be
        # painted over the guide after all shapes have been composed.
        del _TEXT_BATCH[card_text_start:footer_text_start]
        gx1, gx2 = 14, frame_width - 14
        draw_rounded_rect(frame, gx1, 92, gx2, 638, UI_BG, radius=18, border=UI_BORDER)
        draw_text_line(frame, "HOW TO GESTURE", gx1 + 24, 126, scale=0.58, thickness=2)
        draw_text_line(frame, "Prototype pose rules are exact. Motion signs must match the trained video movement.", gx1 + 24, 150, scale=0.37, color=UI_MUTED)

        draw_text_line(frame, "INSTANT POSES", 64, 181, scale=0.38, thickness=2, color=UI_SUCCESS)
        draw_text_line(frame, "GESTURE", 64, 207, scale=0.31, color=UI_MUTED)
        draw_text_line(frame, "WHAT TO DO", 155, 207, scale=0.31, color=UI_MUTED)
        draw_text_line(frame, f"RESULT IN {context_mode.upper()}", 403, 207, scale=0.31, color=UI_MUTED)
        for index, (gesture, instruction, action) in enumerate(POSE_GESTURES):
            y = 235 + index * 31
            draw_text_line(frame, gesture, 64, y, scale=0.38, thickness=2)
            draw_text_line(frame, instruction, 155, y, scale=0.33)
            contextual_action = ContextInterpreter.phrase_for(context_mode, gesture)
            draw_text_line(frame, clipped(contextual_action, 31), 403, y, scale=0.30, color=UI_MUTED)

        draw_text_line(frame, "RELIABLE PRETRAINED MODE", 650, 181, scale=0.38, thickness=2, color=UI_ACCENT)
        draw_text_line(frame, "Google MediaPipe recognizes the seven poses shown at left.", 650, 207, scale=0.33, color=UI_MUTED)
        draw_text_line(frame, "No personal training is required.", 650, 239, scale=0.36, thickness=2)
        draw_text_line(frame, "Hold a pose steadily until the confirmation bar fills.", 650, 270, scale=0.33, color=UI_MUTED)
        draw_text_line(frame, "Dataset motion classes are disabled to prevent false detections.", 650, 301, scale=0.33, color=UI_WARNING)

        draw_text_line(frame, "ALL CONTROLS", 650, 334, scale=0.37, thickness=2, color=UI_ACCENT)
        draw_text_line(frame, "1 Recognition  |  2 Voice  |  3 Safety  |  0 Camera only", 650, 358, scale=0.31)
        draw_text_line(frame, "G Guide | X Context | P Text-Sign | R Recognition | Y Confirm | S Speak", 650, 381, scale=0.28, color=UI_MUTED)
        draw_text_line(frame, "B Calibrate  |  T Metrics  |  C Clear  |  K Cancel  |  A Acknowledge", 650, 404, scale=0.29, color=UI_MUTED)
        draw_text_line(frame, "M Auto/Manual  |  Q Quit  |  Mouse: click tabs", 650, 427, scale=0.29, color=UI_MUTED)

        draw_rounded_rect(frame, 635, 440, 1240, 606, UI_CARD_ALT, radius=13)
        draw_text_line(frame, "OPTIONAL SILENT SOS - disabled by default", 654, 468, scale=0.37, thickness=2, color=UI_DANGER)
        for index, (gesture, instruction) in enumerate(SOS_GESTURES):
            column, row = index // 2, index % 2
            x = 654 + column * 286
            y = 500 + row * 31
            draw_text_line(frame, gesture, x, y, scale=0.35, thickness=2)
            draw_text_line(frame, instruction, x + 88, y, scale=0.30, color=UI_MUTED)
        draw_text_line(frame, "Enable only after calibration; closed-fist Emergency always uses the alert countdown.", 654, 584, scale=0.29, color=UI_WARNING)

    if alert_pending_seconds is not None:
        seconds = max(0, int(math.ceil(alert_pending_seconds)))
        draw_rounded_rect(frame, frame_width - 500, 574, frame_width - 24, 630, UI_DANGER, radius=15)
        draw_text_line(frame, f"EMERGENCY ALERT IN {seconds}s", frame_width - 476, 608, scale=0.52, thickness=2, color=UI_TEXT)
        draw_text_line(frame, "Press K to cancel", frame_width - 190, 608, scale=0.40, thickness=2, color=UI_TEXT)

    if owns_text_batch:
        render_text_commands(frame, end_text_batch())


def draw_scaled_project_interface(frame, ui_scale, **interface_values):
    """Render the HUD at native size or scale it while keeping it bottom-anchored."""
    fit_scale = min(frame.shape[1] / LOGICAL_UI_WIDTH, frame.shape[0] / LOGICAL_UI_HEIGHT)
    interface_scale = max(
        MIN_UI_SCALE_PERCENT / 100.0,
        min(float(ui_scale), MAX_UI_SCALE_PERCENT / 100.0, fit_scale),
    )
    # UI scale is deliberately independent from camera/display resolution. This
    # prevents macOS HiDPI windows from enlarging a 720p raster dashboard.
    scale = interface_scale
    ui_state = interface_values.get("ui_state")
    if ui_state is not None:
        ui_state["hit_scale"] = scale
    if (
        abs(scale - 1.0) < 0.001
        and frame.shape[1] == LOGICAL_UI_WIDTH
        and frame.shape[0] == LOGICAL_UI_HEIGHT
    ):
        if ui_state is not None:
            ui_state["hit_offset"] = (0, 0)
        draw_project_interface(frame, **interface_values)
        return

    sentinel = np.array([1, 2, 3], dtype=np.uint8)
    dashboard = np.empty((LOGICAL_UI_HEIGHT, LOGICAL_UI_WIDTH, 3), dtype=np.uint8)
    dashboard[:] = sentinel
    begin_text_batch()
    draw_project_interface(dashboard, **interface_values)
    text_commands = end_text_batch()

    # The normal HUD only occupies the bottom strip. Crop transparent space before
    # resizing/masking; the larger guide intentionally uses most of the canvas.
    crop_y = (
        88 if interface_values.get("show_guide")
        else 638 if interface_values.get("active_tab") == "camera"
        else 484
    )
    dashboard = dashboard[crop_y:]
    if abs(scale - 1.0) >= 0.001:
        dashboard = cv2.resize(dashboard, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    height, width = dashboard.shape[:2]
    height = min(height, frame.shape[0])
    width = min(width, frame.shape[1])
    dashboard = dashboard[:height, :width]
    mask = np.any(dashboard != sentinel, axis=2)
    full_width = round(LOGICAL_UI_WIDTH * scale)
    full_height = round(LOGICAL_UI_HEIGHT * scale)
    offset_x = max(0, (frame.shape[1] - full_width) // 2)
    full_offset_y = max(0, frame.shape[0] - full_height)
    target_y = max(0, frame.shape[0] - height)
    if ui_state is not None:
        ui_state["hit_offset"] = (offset_x, full_offset_y)
    target = frame[target_y:target_y + height, offset_x:offset_x + width]
    target[mask] = dashboard[mask]
    render_text_commands(
        frame, text_commands, coordinate_scale=scale,
        offset_x=offset_x, offset_y=full_offset_y,
    )


def handle_ui_mouse(event, x, y, flags, state):
    """Handle the clickable tab dock."""
    if event == cv2.EVENT_LBUTTONDOWN:
        hit_scale = max(float(state.get("hit_scale", 1.0)), 0.01)
        offset_x, offset_y = state.get("hit_offset", (0, 0))
        logical_x = (x - offset_x) / hit_scale
        logical_y = (y - offset_y) / hit_scale
        for tab_id, (bx1, by1, bx2, by2) in state.get("tab_boxes", {}).items():
            if bx1 <= logical_x <= bx2 and by1 <= logical_y <= by2:
                state["active_tab"] = "camera" if state.get("active_tab") == tab_id else tab_id
                break


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

    camera_index = env_int("GESTURE_BRIDGE_CAMERA_INDEX", -1, minimum=-1, maximum=10)
    headless = os.getenv("GESTURE_BRIDGE_HEADLESS", "0") == "1"
    enable_silent_sos = os.getenv("GESTURE_BRIDGE_ENABLE_SILENT_SOS", "0") == "1"
    enable_distress_escalation = os.getenv("GESTURE_BRIDGE_ENABLE_DISTRESS_ESCALATION", "0") == "1"
    hidpi_desktop = platform.system() in {"Darwin", "Windows"} and not headless
    display_width = env_int(
        "GESTURE_BRIDGE_DISPLAY_WIDTH", 1920 if hidpi_desktop else 1280,
        minimum=1280, maximum=3840,
    )
    display_height = env_int(
        "GESTURE_BRIDGE_DISPLAY_HEIGHT", 1080 if hidpi_desktop else 720,
        minimum=720, maximum=2160,
    )
    analysis_width = env_int("GESTURE_BRIDGE_ANALYSIS_WIDTH", 640, minimum=320, maximum=1280)
    analysis_height = env_int("GESTURE_BRIDGE_ANALYSIS_HEIGHT", 360, minimum=240, maximum=720)
    desktop_hidpi = platform.system() in {"Darwin", "Windows"}
    default_capture_width = analysis_width if headless else max(display_width, 1920 if desktop_hidpi else 1280)
    default_capture_height = analysis_height if headless else max(display_height, 1080 if desktop_hidpi else 720)
    if headless:
        # An SSH/service deployment never displays the HD frame, so avoid wasting
        # Pi USB bandwidth and memory even if desktop capture values exist in env.
        capture_width, capture_height = analysis_width, analysis_height
    else:
        capture_width = env_int("GESTURE_BRIDGE_CAPTURE_WIDTH", default_capture_width, minimum=320, maximum=3840)
        capture_height = env_int("GESTURE_BRIDGE_CAPTURE_HEIGHT", default_capture_height, minimum=240, maximum=2160)
    camera = open_camera(camera_index, capture_width, capture_height, fps=30)
    if camera is None:
        print(
            f"Startup error: no readable webcam found ({'auto scan' if camera_index < 0 else f'index {camera_index}'}). "
            "Check /dev/video* permissions on Pi or set GESTURE_BRIDGE_CAMERA_INDEX."
        )
        return
    cap = camera.capture
    print(
        f"Camera: index {camera.index}, backend {camera.backend}, "
        f"{camera.width}x{camera.height} at requested {camera.fps:.0f} FPS",
        flush=True,
    )

    ui_scale_state = {
        "active_tab": "communication",
        "tab_boxes": {},
    }
    if not headless:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_TITLE, handle_ui_mouse, ui_scale_state)

    stable_gesture = "Unknown"
    previous_detected_text = "Unknown"
    stability_value = 0
    required_stable_frames = 100

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
    text_entry_active = False
    auto_speak_on_add = True

    recognition_method = "MediaPipe landmark classifier"
    prediction_confidence = 0.0
    safety_analyzer = SafetyAnalyzer(profile_path=str(BASE_DIR / "gesture_profile.json"))
    alert_manager = AlertManager(log_path=str(BASE_DIR / "emergency_alerts.jsonl"))
    context_interpreter = ContextInterpreter()
    sentence_engine = SentenceEngine()
    gesture_debouncer = GestureDebouncer()
    emergency_controller = EmergencyController(
        alert_manager,
        delay_seconds=env_float("GESTURE_BRIDGE_ALERT_COUNTDOWN_SECONDS", 5, minimum=2, maximum=30),
    )
    telemetry = SessionTelemetry(BASE_DIR / "session_reports")
    safety_state = safety_analyzer.update([])
    last_alert_time = 0.0
    high_distress_since = None
    last_frame_time = time.perf_counter()
    fps = 0.0
    clarification_candidate = None

    print("Gesture-Bridge distress-aware assistive system started.", flush=True)
    print(f"Display mode: {'headless (SSH/service)' if headless else 'graphical'}", flush=True)
    print("Recognizer: Google MediaPipe pretrained canned gesture model", flush=True)
    print("Controls: X = context, Y = confirm suggestion, P = text-to-sign, R = recognition, S = speak, C = clear, G = guide, T = metrics, Q = quit.", flush=True)
    print(f"Capabilities: {len(HEURISTIC_GUIDE)} pretrained poses plus distress and silent SOS detection.", flush=True)

    frame_timestamp_ms = 0
    camera_failures = 0

    try:
        recognizer_instance = GestureRecognizer.create_from_options(options)
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Startup error: MediaPipe gesture recognizer failed: {error}")
        cap.release()
        cv2.destroyAllWindows()
        speech_service.close()
        return

    with recognizer_instance as recognizer:
        while True:
            success, frame = cap.read()

            if not success:
                camera_failures += 1
                if camera_failures < 10:
                    time.sleep(0.05)
                    continue
                print("Runtime error: camera stopped returning frames after 10 retries.")
                break
            camera_failures = 0

            camera_frame = cv2.flip(frame, 1)
            analysis_frame = camera_frame
            if analysis_frame.shape[1] != analysis_width or analysis_frame.shape[0] != analysis_height:
                analysis_frame = cv2.resize(
                    analysis_frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA
                )
            rgb_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms += 33
            result = recognizer.recognize_for_video(mp_image, frame_timestamp_ms)

            raw_detected_text = "No hand detected"
            display_detected_text = raw_detected_text
            prediction_confidence = 0.0
            detected_hands = []
            clarification_candidate = None

            if result.hand_landmarks and result.handedness:
                detected_hands = result.hand_landmarks[:MAX_HANDS]
                if result.gestures and result.gestures[0]:
                    category = result.gestures[0][0]
                    prediction_confidence = float(category.score or 0.0)
                    raw_detected_text = PRETRAINED_GESTURE_MAP.get(category.category_name, "Unknown")
                else:
                    raw_detected_text = "Unknown"
                display_detected_text = raw_detected_text

            current_time = time.time()
            monotonic_now = time.monotonic()
            safety_state = safety_analyzer.update(detected_hands, monotonic_now)
            frame_now = time.perf_counter()
            instantaneous_fps = 1.0 / max(frame_now - last_frame_time, 0.001)
            fps = instantaneous_fps if fps == 0 else fps * 0.90 + instantaneous_fps * 0.10
            last_frame_time = frame_now

            # Time-based voting keeps confirmation responsive even at low FPS and
            # requires a brief release before the same pose can speak again.
            valid_raw_gesture = raw_detected_text not in ["Unknown", "No hand detected"]
            if raw_detected_text != previous_detected_text:
                previous_detected_text = raw_detected_text
                response_start_time = current_time if valid_raw_gesture else None

            recognition_state = gesture_debouncer.update(
                raw_detected_text, prediction_confidence, monotonic_now
            )
            stable_gesture = recognition_state.active
            confirmed_event = recognition_state.event
            stability_value = round(recognition_state.progress * 100)
            active_required_frames = 100
            detected_text = stable_gesture
            valid_detected_gesture = detected_text not in ["Unknown", "No hand detected"]

            # Automatic text + speech output
            if auto_mode and confirmed_event:
                if response_start_time is not None:
                    last_response_time = current_time - response_start_time
                recent_outputs.append(confirmed_event)
                recent_outputs[:] = recent_outputs[-RECENT_OUTPUT_LIMIT:]
                total_confirmed_outputs += 1
                append_recognition_log(raw_detected_text, confirmed_event, last_response_time)
                safety_analyzer.note_confirmed_gesture(confirmed_event, monotonic_now)
                sentence_engine.add(confirmed_event)
                telemetry.confirmed(confirmed_event)
                print(
                    f"GESTURE {confirmed_event}: {context_interpreter.interpret(confirmed_event)} "
                    f"(confidence {prediction_confidence:.0%})",
                    flush=True,
                )

                if auto_speak_on_add and confirmed_event != "Emergency":
                    # Preserve normal phrases in order (for example "Hello" then
                    # "I need help"). Emergency speech is combined with its alert
                    # instruction below and may interrupt anything less urgent.
                    speak(context_interpreter.interpret(confirmed_event))

            # Escalate only after sustained distress plus an emergency-related sign.
            if safety_state.level == "HIGH":
                high_distress_since = high_distress_since or current_time
            else:
                high_distress_since = None

            high_distress_seconds = current_time - high_distress_since if high_distress_since else 0.0
            alert_reason, silent_alert = select_alert_trigger(
                confirmed_event,
                detected_text,
                safety_state,
                high_distress_seconds=high_distress_seconds,
                enable_silent_sos=enable_silent_sos,
                enable_distress_escalation=enable_distress_escalation,
            )

            if alert_reason and current_time - last_alert_time >= 8.0:
                message = "Emergency distress detected. Please check on the Gesture-Bridge user."
                if silent_alert:
                    payload = alert_manager.trigger(alert_reason, message, silent=True)
                    telemetry.alert(alert_reason, payload["alert_id"])
                elif emergency_controller.arm(alert_reason, message, now=time.monotonic()):
                    telemetry.alert(alert_reason, state="COUNTDOWN_ARMED")
                    spoken_intent = (
                        context_interpreter.interpret(detected_text)
                        if valid_detected_gesture else "Emergency detected."
                    )
                    speak(f"{spoken_intent} Press K to cancel the alert.", urgent=True)
                last_alert_time = current_time

            delivered_alert = emergency_controller.tick(time.monotonic())
            if delivered_alert:
                telemetry.alert(delivered_alert["reason"], delivered_alert["alert_id"], delivered_alert["state"])
                speak("Emergency alert sent to caregiver.", urgent=True)
                print(
                    f"ALERT {delivered_alert['alert_id']}: {alert_manager.last_status}",
                    flush=True,
                )

            pending_seconds = emergency_controller.remaining(time.monotonic())
            if pending_seconds is not None or safety_state.level in {"HIGH", "SOS"}:
                ui_scale_state["active_tab"] = "safety"

            if pending_seconds is not None:
                communication_output = f"Emergency alert pending - press K to cancel"
            elif safety_state.level == "SOS" and enable_silent_sos:
                communication_output = "SILENT SOS SENT TO CAREGIVER"
            elif safety_state.level == "HIGH" and valid_detected_gesture and enable_distress_escalation:
                communication_output = "Emergency distress detected. Alerting caregiver."
            else:
                communication_output = (
                    context_interpreter.interpret(detected_text)
                    if valid_detected_gesture
                    else context_interpreter.interpret(recent_outputs[-1])
                    if recent_outputs
                    else "Show and briefly hold a gesture..."
                )
            recent_output_text = " | ".join(recent_outputs) if recent_outputs else "No confirmed outputs yet"
            response_time_text = f"{last_response_time:.2f} sec" if last_response_time is not None else "--"
            telemetry.frame(bool(detected_hands), raw_detected_text, recognition_method, fps)

            shown_alert_status = (
                f"Alert pending: {math.ceil(pending_seconds)}s - K cancels"
                if pending_seconds is not None
                else alert_manager.last_status
            )

            key = 255
            if not headless:
                # Keep inference small, then render the camera and UI for the local display.
                frame = camera_frame
                if frame.shape[1] != display_width or frame.shape[0] != display_height:
                    frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
                if detected_hands:
                    for hand_landmarks in detected_hands:
                        draw_hand_landmarks(frame, hand_landmarks)
                    x_min, y_min, x_max, y_max = get_combined_hand_bbox(detected_hands, display_width, display_height)
                    margin = max(15, round(18 * display_width / LOGICAL_UI_WIDTH))
                    cv2.rectangle(
                        frame, (x_min - margin, y_min - margin), (x_max + margin, y_max + margin),
                        UI_TEXT, max(2, round(2 * display_width / LOGICAL_UI_WIDTH)), cv2.LINE_AA,
                    )

                draw_scaled_project_interface(
                    frame, DEFAULT_UI_SCALE_PERCENT / 100.0,
                    detected_text=detected_text, raw_detected_text=display_detected_text,
                    stability_value=stability_value, required_stable_frames=active_required_frames,
                    communication_output=communication_output, recent_output_text=recent_output_text,
                    response_time_text=response_time_text, total_confirmed_outputs=total_confirmed_outputs,
                    auto_mode=auto_mode, show_guide=show_guide, show_testing_metrics=show_testing_metrics,
                    app_mode=app_mode, text_to_sign_query=(text_to_sign_query + "_" if text_entry_active else text_to_sign_query),
                    text_to_sign_result=text_to_sign_result, recognition_method=recognition_method,
                    prediction_confidence=prediction_confidence, safety_state=safety_state,
                    context_mode=context_interpreter.context, alert_status=shown_alert_status, fps=fps,
                    sentence_output=context_interpreter.compose(recent_outputs), speech_status=speech_service.status,
                    active_tab=ui_scale_state["active_tab"], ui_state=ui_scale_state,
                    alert_pending_seconds=pending_seconds,
                )
                cv2.imshow(WINDOW_TITLE, frame)
                try:
                    if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
                key = cv2.waitKey(1) & 0xFF

            if text_entry_active:
                if key in (10, 13):
                    phrase = text_to_sign_query.strip()
                    normalized_phrase = phrase.lower()
                    text_entry_active = False
                    if normalized_phrase in SIGN_INSTRUCTIONS:
                        text_to_sign_result = SIGN_INSTRUCTIONS[normalized_phrase]
                        speak(f"{phrase}. Sign representation: {text_to_sign_result}")
                    else:
                        text_to_sign_query = phrase if phrase else "None"
                        text_to_sign_result = "Unsupported phrase. Press G to view supported signs."
                        speak("Unsupported phrase")
                elif key == 27:
                    text_entry_active = False
                    app_mode = "Sign Recognition"
                elif key in (8, 127):
                    text_to_sign_query = text_to_sign_query[:-1]
                elif 32 <= key <= 126 and len(text_to_sign_query) < 42:
                    text_to_sign_query += chr(key)

            elif key == ord("p"):
                app_mode = "Text-to-Sign Representation"
                ui_scale_state["active_tab"] = "communication"
                text_to_sign_query = ""
                text_to_sign_result = "Type a phrase, then press Enter. Esc cancels."
                text_entry_active = True

            elif key == ord("r"):
                app_mode = "Sign Recognition"
                ui_scale_state["active_tab"] = "recognition"
                text_entry_active = False
                text_to_sign_query = "None"
                text_to_sign_result = "Press P and enter a supported phrase"
                speak("Sign recognition mode")

            elif key == ord("s"):
                ui_scale_state["active_tab"] = "communication"
                if recent_outputs:
                    speak(context_interpreter.compose(recent_outputs))
                else:
                    speak("No confirmed outputs to speak")

            elif key == ord("c"):
                recent_outputs.clear()
                sentence_engine.clear()
                safety_analyzer.reset()
                total_confirmed_outputs = 0
                last_response_time = None
                text_to_sign_query = "None"
                text_to_sign_result = "Press P and enter a supported phrase"
                app_mode = "Sign Recognition"
                text_entry_active = False
                gesture_debouncer.reset()
                stable_gesture = "Unknown"
                speak("Outputs cleared")

            elif key == ord("x"):
                new_context = context_interpreter.cycle()
                recent_outputs.clear()
                sentence_engine.clear()
                stable_gesture = "Unknown"
                gesture_debouncer.reset()
                previous_detected_text = "Unknown"
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
                ui_scale_state["active_tab"] = "safety"
                if emergency_controller.cancel():
                    telemetry.alert("Pending emergency", state="COUNTDOWN_CANCELLED")
                    speak("Emergency alert cancelled")
                elif alert_manager.cancel():
                    speak("Emergency alert cancelled")

            elif key == ord("g"):
                show_guide = not show_guide

            elif key == ord("1"):
                ui_scale_state["active_tab"] = "recognition"

            elif key == ord("2"):
                ui_scale_state["active_tab"] = "communication"

            elif key == ord("3"):
                ui_scale_state["active_tab"] = "safety"

            elif key == ord("0"):
                ui_scale_state["active_tab"] = "camera"

            elif key == ord("t"):
                show_testing_metrics = not show_testing_metrics

            elif key == ord("m"):
                auto_mode = not auto_mode
                mode_text = "Auto mode" if auto_mode else "Manual mode"
                speak(mode_text)

            elif key == ord("q"):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    speech_service.close()
    report_path = telemetry.close()
    if report_path:
        print(f"Session report: {report_path}")


if __name__ == "__main__":
    main()
