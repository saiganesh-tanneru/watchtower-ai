import os
import sys
import time
import csv
import math
import signal
import datetime as dt
from pathlib import Path

import cv2
import numpy as np

# ------------------ Config via env ------------------
MONITOR_FPS = int(os.getenv("MONITOR_FPS", "15"))
MOTION_AREA_MIN = int(os.getenv("MOTION_AREA_MIN", "1500"))  # pixels
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.005"))
ROI_ENV = os.getenv("ROI", "").strip()  # "x1,y1;x2,y2;..."

# ------------------ Prep ---------------------------
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def today_csv_path():
    return LOGS_DIR / f"{dt.date.today().isoformat()}_motion.csv"

def ensure_csv_header(path: Path):
    if not path.exists():
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "event", "motion_score", "largest_contour_area"])

def parse_roi(roi_str):
    if not roi_str:
        return None
    pts = []
    for pair in roi_str.split(";"):
        xy = pair.strip().split(",")
        if len(xy) != 2:
            continue
        try:
            pts.append((int(xy[0]), int(xy[1])))
        except ValueError:
            pass
    return np.array(pts, dtype=np.int32) if pts else None

ROI_POLY = parse_roi(ROI_ENV)

# Graceful shutdown
_running = True
_paused = False
def handle_sigint(sig, frame):
    global _running
    _running = False
signal.signal(signal.SIGINT, handle_sigint)

# ------------------ Camera -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera index 0.")
    print("macOS fix: System Settings → Privacy & Security → Camera → allow Terminal / your Python app.")
    print("Also try: restarting the Terminal app after granting permission.")
    sys.exit(1)

# Set a reasonable frame size (let camera decide if unsupported)
cap.set(cv2.CAP_PROP_FPS, MONITOR_FPS)

# Background subtractor (robust to illumination drift)
bs = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# For 'r' reset, we recreate the subtractor
def reset_background():
    global bs
    bs = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# ------------------ Logging ------------------------
ensure_csv_header(today_csv_path())

def log_event(event, motion_score, max_area):
    ensure_csv_header(today_csv_path())
    with open(today_csv_path(), "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), event, int(motion_score), int(max_area)])

last_event = None
last_event_time = 0.0

# ------------------ Main loop ----------------------
print("✅ Room Activity Monitor Ready")
print("Controls: [q]=quit  [r]=reset background  [p]=pause/resume")
print(f"Config: MONITOR_FPS={MONITOR_FPS}, MOTION_AREA_MIN={MOTION_AREA_MIN}, LEARNING_RATE={LEARNING_RATE}")
if ROI_POLY is not None:
    print(f"ROI active with {len(ROI_POLY)} points (only inside polygon is considered).")

delay_ms = max(1, int(1000 / max(1, MONITOR_FPS)))

while _running:
    if _paused:
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('p'):
            _paused = False
        elif key == ord('q'):
            break
        continue

    ok, frame = cap.read()
    if not ok:
        print("WARN: Failed to read frame; retrying...")
        time.sleep(0.05)
        continue

    h, w = frame.shape[:2]

    # Optional ROI mask
    mask = None
    if ROI_POLY is not None and len(ROI_POLY) >= 3:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [ROI_POLY], 255)

    # Apply background subtraction
    fgmask = bs.apply(frame, learningRate=LEARNING_RATE)

    # If ROI is set, limit to ROI
    if mask is not None:
        fgmask = cv2.bitwise_and(fgmask, mask)

    # Threshold + morphology to clean noise
    _, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    th = cv2.dilate(th, np.ones((3,3), np.uint8), iterations=2)

    # Find contours and compute motion score
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas) if areas else 0
    motion_score = int(np.sum(th) // 255)

    # Determine status
    motion = max_area >= MOTION_AREA_MIN
    status = "Motion" if motion else "No motion"

    # Event logging (debounced to avoid spam)
    now = time.time()
    if status != last_event and now - last_event_time > 1.5:
        log_event(status, motion_score, max_area)
        last_event = status
        last_event_time = now

    # Overlay HUD
    hud = frame.copy()
    color = (0, 200, 0) if not motion else (0, 0, 255)
    cv2.putText(hud, f"Status: {status}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(hud, f"Score:{motion_score}  MaxArea:{int(max_area)}  FPS:{MONITOR_FPS}", (16, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Draw ROI for clarity
    if ROI_POLY is not None and len(ROI_POLY) >= 3:
        cv2.polylines(hud, [ROI_POLY], isClosed=True, color=(255, 200, 0), thickness=2)

    cv2.imshow("Room Monitor (local, offline)", hud)

    key = cv2.waitKey(delay_ms) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        reset_background()
    elif key == ord('p'):
        _paused = True

cap.release()
cv2.destroyAllWindows()