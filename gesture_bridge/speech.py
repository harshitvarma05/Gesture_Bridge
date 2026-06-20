"""Non-blocking, failure-tolerant text-to-speech service."""

from queue import Empty, Full, Queue
import threading


class SpeechService:
    def __init__(self, rate=160):
        self.rate = rate
        self.queue = Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.status = "Starting"
        self.worker = threading.Thread(target=self._run, name="gesture-bridge-speech", daemon=True)
        self.worker.start()

    def say(self, text):
        text = str(text).strip()
        if not text:
            return
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

    def _run(self):
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

