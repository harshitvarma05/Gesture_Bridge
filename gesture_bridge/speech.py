"""Non-blocking, failure-tolerant text-to-speech service."""

from queue import Empty, Full, Queue
from pathlib import Path
import platform
import subprocess
import threading


class SpeechService:
    def __init__(self, rate=160):
        self.rate = rate
        self.queue = Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.status = "Starting"
        self.current_process = None
        self.process_lock = threading.Lock()
        self.worker = threading.Thread(target=self._run, name="gesture-bridge-speech", daemon=True)
        self.worker.start()

    def say(self, text, interrupt=False):
        text = str(text).strip()
        if not text:
            return
        if interrupt:
            while True:
                try:
                    self.queue.get_nowait()
                except Empty:
                    break
            with self.process_lock:
                if self.current_process and self.current_process.poll() is None:
                    self.current_process.terminate()
        try:
            self.queue.put_nowait(text)
        except Full:
            # Communication must never freeze recognition; keep the newest message.
            try:
                self.queue.get_nowait()
            except Empty:
                pass
            try:
                self.queue.put_nowait(text)
            except Full:
                pass

    def close(self):
        self.stop_event.set()
        with self.process_lock:
            if self.current_process and self.current_process.poll() is None:
                self.current_process.terminate()

    def _run(self):
        if platform.system() == "Darwin" and Path("/usr/bin/say").exists():
            self.status = "Ready"
            self._run_macos()
            return
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            self.status = "Ready"
        except Exception as error:
            self.status = f"Unavailable: {type(error).__name__}"
            return

        while not self.stop_event.is_set():
            try:
                text = self.queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as error:
                self.status = f"Error: {type(error).__name__}"

    def _run_macos(self):
        """Use macOS' native speech command; it is more reliable than pyttsx3 there."""
        while not self.stop_event.is_set():
            try:
                text = self.queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                with self.process_lock:
                    self.current_process = subprocess.Popen(
                        ["/usr/bin/say", "-r", str(self.rate), text],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                self.current_process.wait()
                with self.process_lock:
                    self.current_process = None
            except (OSError, subprocess.SubprocessError) as error:
                self.status = f"Error: {type(error).__name__}"
