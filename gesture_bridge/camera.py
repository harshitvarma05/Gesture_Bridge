"""Cross-platform webcam discovery and low-latency OpenCV setup."""

from dataclasses import dataclass
import platform
import time

@dataclass
class CameraHandle:
    capture: object
    index: int
    backend: str
    width: int
    height: int
    fps: float


def _candidate_indices(preferred_index, scan_limit):
    if preferred_index is not None and int(preferred_index) >= 0:
        return [int(preferred_index)]
    return list(range(max(1, int(scan_limit))))


def open_camera(preferred_index=-1, width=640, height=360, fps=30, scan_limit=6):
    """Return the first camera that produces a frame, not merely one that opens."""
    try:
        import cv2
    except ImportError:
        return None
    linux = platform.system() == "Linux"
    backend = cv2.CAP_V4L2 if linux else cv2.CAP_ANY
    backend_name = "V4L2" if linux else "Auto"

    for index in _candidate_indices(preferred_index, scan_limit):
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened() and backend != cv2.CAP_ANY:
            capture.release()
            capture = cv2.VideoCapture(index, cv2.CAP_ANY)
            backend_name = "Auto"
        if not capture.isOpened():
            capture.release()
            continue

        # MJPG is widely supported by USB webcams and avoids expensive raw USB frames.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        capture.set(cv2.CAP_PROP_FPS, int(fps))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame = None
        deadline = time.monotonic() + 1.2
        while time.monotonic() < deadline:
            success, candidate = capture.read()
            if success and candidate is not None and candidate.size:
                frame = candidate
                break
            time.sleep(0.04)
        if frame is None:
            capture.release()
            continue

        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(fps)
        return CameraHandle(capture, index, backend_name, actual_width, actual_height, actual_fps)
    return None
