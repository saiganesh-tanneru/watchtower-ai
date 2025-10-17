import os
import sys
import time
import json
import queue
import sqlite3
import threading
import subprocess
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets

# -----------------------------
# Paths & Files
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
LOG_DB = APP_DIR / "activity.db"
FACE_DIR = APP_DIR / "faces"
FACE_DIR.mkdir(parents=True, exist_ok=True)
ENCODINGS_FILE = FACE_DIR / "encodings.json"

# -----------------------------
# Optional libs
# -----------------------------
try:
    from vosk import Model, KaldiRecognizer
    import pyaudio
    VOSK_OK = True
except Exception:
    VOSK_OK = False

try:
    import face_recognition
    FACE_REC_OK = True
except Exception:
    FACE_REC_OK = False

# YOLOv8 object detection (ultralytics)
YOLO_OK = True
try:
    from ultralytics import YOLO
except Exception:
    YOLO_OK = False

# -----------------------------
# Speech Queue (no overlap)
# -----------------------------
_speech_q = queue.Queue()

def _escape_applescript_text(s: str) -> str:
    # escape " characters for AppleScript
    return s.replace('"', '\\"')

def _speech_worker():
    # Speak items sequentially
    while True:
        text = _speech_q.get()
        if not text:
            continue
        try:
            safe = _escape_applescript_text(text)
            osa_script = f'try\nsay "{safe}" using "Samantha"\non error\nsay "{safe}"\nend try'
            subprocess.run(
                ["osascript", "-e", osa_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            # Keep silent on TTS errors
            pass

threading.Thread(target=_speech_worker, daemon=True).start()

def speak_async(text: str):
    _speech_q.put(text)

# -----------------------------
# Utilities
# -----------------------------
def time_greeting() -> str:
    h = datetime.now().hour
    if 5 <= h < 12:
        return "Good morning"
    if 12 <= h < 17:
        return "Good afternoon"
    if 17 <= h < 22:
        return "Good evening"
    return "Hello"

def auto_adjust_lighting(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    b = float(np.mean(gray))
    if b < 80:       # low light
        alpha, beta = 1.5, 25
    elif b > 180:    # too bright / backlit
        alpha, beta = 0.75, -35
    else:
        alpha, beta = 1.0, 0
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# -----------------------------
# DB: activity log
# -----------------------------
def init_db():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            timestamp REAL,
            event_type TEXT,
            detail TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def log_event(event_type: str, detail: str = ""):
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO events (timestamp, event_type, detail) VALUES (?, ?, ?)",
        (time.time(), event_type, detail),
    )
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Encodings DB (if using face_recognition)
# -----------------------------
def load_enc_db() -> dict:
    if not ENCODINGS_FILE.exists():
        return {}
    try:
        with open(ENCODINGS_FILE, "r") as f:
            raw = json.load(f)
        return {k: [np.array(vv, dtype=np.float32) for vv in v] for k, v in raw.items()}
    except Exception:
        return {}

def save_enc_db(db: dict):
    try:
        with open(ENCODINGS_FILE, "w") as f:
            json.dump({k: [vv.tolist() for vv in v] for k, v in db.items()}, f, indent=2)
    except Exception:
        pass

# -----------------------------
# Haar cascade for face boxes (fast & robust)
# -----------------------------
HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -----------------------------
# Video Thread
# -----------------------------
class VideoThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object, object)  # frame, face_boxes
    greet_signal = QtCore.pyqtSignal(str)
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, cam_index: int = 0):
        super().__init__()
        self.cap = cv2.VideoCapture(cam_index)
        self.running = True

        # Greeting rules
        self.enable_greet = True
        self.currently_present = set()
        self.greeted_once = set()  # greet each known person once per session

        # Motion detection params
        self.bg_model = None
        self.MOTION_MIN_AREA = 5000
        self.MOTION_ALPHA = 0.02
        self.motion_detected = False

        # Recognition DB + thresholds
        self.db = load_enc_db() if FACE_REC_OK else {}
        self.strict_thresh = 0.38  # lower = stricter; tweak if needed

        # YOLOv8n object detection
        self.yolo = None
        if YOLO_OK:
            try:
                # Prefer models in models/ directory (moved during cleanup)
                model_path = APP_DIR / "models" / "yolov8n.pt"
                if model_path.exists():
                    model_file = str(model_path)
                else:
                    model_file = "yolov8n.pt"
                # Use CPU for compatibility, suppress logging
                self.yolo = YOLO(model_file)
                self.yolo.fuse()  # fuse model for faster inference (optional)
                # No explicit device selection here; ultralytics will use CPU if no CUDA
            except Exception:
                self.yolo = None

    def _best_match(self, encoding: np.ndarray):
        if not self.db:
            return "Unknown", 1.0
        # Normalize encoding
        nrm = np.linalg.norm(encoding)
        if nrm > 0:
            encoding = encoding / nrm

        best_name = "Unknown"
        best_dist = 1.0
        for name, vecs in self.db.items():
            if not vecs:
                continue
            dists = face_recognition.face_distance(vecs, encoding)
            dmin = float(np.min(dists))
            if dmin < best_dist:
                best_dist = dmin
                best_name = name

        if best_dist < self.strict_thresh:
            return best_name, best_dist
        return "Unknown", best_dist

    def _recognize(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = HAAR.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        boxes: list[tuple[int, int, int, int]] = []
        names: list[str] = []

        for (x, y, w, h) in faces:
            name = "Unknown"
            if FACE_REC_OK:
                # crop with small pad to improve encoding quality
                pad = 10
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    encs = face_recognition.face_encodings(rgb)
                    if encs:
                        name, _ = self._best_match(encs[0])

            boxes.append((x, y, w, h))
            names.append(name)

        return boxes, names

    def _update_motion(self, frame_gray: np.ndarray):
        if self.bg_model is None:
            self.bg_model = frame_gray.astype("float")
            self.motion_detected = False
            return
        cv2.accumulateWeighted(frame_gray, self.bg_model, self.MOTION_ALPHA)
        delta = cv2.absdiff(frame_gray, cv2.convertScaleAbs(self.bg_model))
        th = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        th = cv2.dilate(th, None, iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.motion_detected = any(cv2.contourArea(c) > self.MOTION_MIN_AREA for c in contours)

    def run(self):
        if not self.cap.isOpened():
            self.error_signal.emit("Camera not available")
            return

        # For YOLO class names (if available)
        yolo_names = None
        if YOLO_OK and self.yolo is not None and hasattr(self.yolo, "names"):
            yolo_names = self.yolo.names

        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # Lighting correction
            frame = auto_adjust_lighting(frame)

            # Object detection (YOLOv8n)
            object_detections = []
            if YOLO_OK and self.yolo is not None:
                try:
                    # YOLO expects RGB
                    yolo_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.yolo(yolo_input, verbose=False)
                    # results[0].boxes.xyxy, .cls, .conf
                    for r in results:
                        if hasattr(r, "boxes") and r.boxes is not None:
                            for box, cls_id in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                                class_idx = int(cls_id)
                                class_name = None
                                if yolo_names and class_idx in yolo_names:
                                    class_name = yolo_names[class_idx]
                                elif yolo_names and class_idx < len(yolo_names):
                                    class_name = yolo_names[class_idx]
                                else:
                                    class_name = str(class_idx)
                                # Optionally skip ceiling fan or repetitive objects
                                if class_name and class_name.lower() == "fan":
                                    continue
                                object_detections.append((box, class_name))
                except Exception:
                    object_detections = []

            # Motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._update_motion(gray)

            # Face detection + optional recognition
            face_boxes, names = self._recognize(frame)

            # Draw object detections (blue)
            for box, class_name in object_detections:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = class_name if class_name else "obj"
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 0, 0), 2)

            # Draw face overlays
            for (x, y, w, h), name in zip(face_boxes, names):
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if self.motion_detected:
                cv2.putText(frame, "Motion Detected", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            # Greet known faces once per session
            current_known = set(n for n in names if n != "Unknown")
            for n in current_known:
                if self.enable_greet and n not in self.greeted_once:
                    self.greet_signal.emit(n)
                    self.greeted_once.add(n)
                    log_event("greet", n)
            self.currently_present = current_known

            # Emit frame for display (face_boxes for compatibility)
            self.frame_ready.emit(frame, face_boxes)

        # Cleanup
        try:
            self.cap.release()
        except Exception:
            pass

    def stop(self):
        self.running = False

# -----------------------------
# Voice Command Thread (optional)
# -----------------------------
class VoiceCommandThread(QtCore.QThread):
    command_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.active = False
        self._last_speech_time = None

    def run(self):
        if not VOSK_OK:
            # No voice recognition installed; silently stop thread
            return
        model_path = APP_DIR / "vosk-model-en-us-0.22"
        if not model_path.exists():
            # Model not found; silently stop thread
            return

        try:
            model = Model(str(model_path))
            rec = KaldiRecognizer(model, 16000)
            mic = pyaudio.PyAudio().open(
                format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096
            )
            mic.start_stream()
        except Exception:
            return

        while self.running:
            try:
                data = mic.read(4096, exception_on_overflow=False)
            except Exception:
                break
            if rec.AcceptWaveform(data):
                try:
                    res = json.loads(rec.Result())
                    text = res.get("text", "").strip()
                    if text:
                        text_lc = text.lower()
                        current_time = time.time()
                        # Reset inactivity timer on any speech
                        self._last_speech_time = current_time
                        if "jarvis" in text_lc:
                            self.active = True
                            self.command_signal.emit("wake jarvis")
                        elif self.active:
                            self.command_signal.emit(text)
                    # Check for inactivity timeout
                    if self.active and self._last_speech_time is not None:
                        if time.time() - self._last_speech_time > 10:
                            self.active = False
                except Exception:
                    pass

    def stop(self):
        self.running = False

# -----------------------------
# Application
# -----------------------------
class App(QtWidgets.QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.video = VideoThread(cam_index=0)
        self.voice = VoiceCommandThread()

        self.video.frame_ready.connect(self.on_frame)
        self.video.greet_signal.connect(self.on_greet)
        self.video.error_signal.connect(self.on_error)
        self.voice.command_signal.connect(self.on_command)

        self._startup_done = False
        self._last_frame = None  # for snapshots

        # Start threads
        self.video.start()
        self.voice.start()

        # Delayed startup greeting (skipped if boss already present)
        QtCore.QTimer.singleShot(1500, self._maybe_startup_greet)

    def on_frame(self, frame, _boxes):
        self._last_frame = frame.copy()
        cv2.imshow("Room Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cleanup()

    def on_greet(self, name: str):
        if name.lower() == "sai":
            speak_async("Welcome back, boss")
        else:
            speak_async(f"{time_greeting()} {name}")

    def on_error(self, msg: str):
        speak_async(msg)

    def _maybe_startup_greet(self):
        if self._startup_done:
            return
        # Skip startup greeting if Sai is already present
        if "sai" in {n.lower() for n in getattr(self.video, "currently_present", set())}:
            self._startup_done = True
            return
        speak_async(f"{time_greeting()}, system online. Ready to monitor the room.")
        self._startup_done = True

    # -------------------------
    # Voice commands (if enabled)
    # -------------------------
    def on_command(self, cmd: str):
        c = cmd.lower()
        if cmd == "wake jarvis":
            speak_async("Yes, boss?")
            return
        # Only respond to commands if assistant is active
        if not self.voice.active:
            return
        # Commands to deactivate assistant
        if "go to sleep" in c or "stop listening" in c:
            self.voice.active = False
            speak_async("Going to sleep. Say 'Jarvis' to wake me up.")
            return
        # Basic intents
        if "what time" in c or "time is it" in c:
            speak_async("The time is " + datetime.now().strftime("%I:%M %p"))
        elif "who is in the room" in c or "who's in the room" in c:
            ppl = list(self.video.currently_present)
            if ppl:
                speak_async("I see " + ", ".join(ppl))
            else:
                speak_async("Nobody is currently here.")
        elif "turn off greeting" in c or "disable greeting" in c:
            self.video.enable_greet = False
            speak_async("Greetings disabled.")
        elif "turn on greeting" in c or "enable greeting" in c:
            self.video.enable_greet = True
            speak_async("Greetings enabled.")
        elif "take a snapshot" in c or "snapshot" in c or "take picture" in c:
            if self._last_frame is not None:
                fn = APP_DIR / f"snap_{int(time.time())}.jpg"
                cv2.imwrite(str(fn), self._last_frame)
                speak_async("Snapshot saved.")
            else:
                speak_async("No frame available for snapshot.")
        elif "how is the lighting" in c or "how's the lighting" in c:
            if self._last_frame is not None:
                gray = cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2GRAY)
                b = float(np.mean(gray))
                if b < 80:
                    speak_async("The room looks dim.")
                elif b > 180:
                    speak_async("The room looks very bright.")
                else:
                    speak_async("Lighting looks okay.")
            else:
                speak_async("I can't assess lighting yet.")
        else:
            # Fallback
            speak_async("Sorry, I didn't understand.")

    def cleanup(self):
        try:
            self.video.stop()
            self.voice.stop()
            self.video.wait(500)
            self.voice.wait(500)
        except Exception:
            pass
        cv2.destroyAllWindows()
        self.quit()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    app = App(sys.argv)
    sys.exit(app.exec_())