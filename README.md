# Room Monitor

This project monitors a room using camera input, face detection/recognition, motion detection, and optional voice commands.

## Models and large assets

To keep the repository clean, large models and extracted model folders are moved into the `models/` directory. The app prefers model files from `models/` if present.

Recommended layout:

- models/
  - yolov8n.pt # YOLOv8 tiny model (object detection)
  - vosk-model-en-us-0.22/ # Vosk offline speech model (if used)
  - ggml-model.bin # Optional local LLaMA/LLM model

If you previously archived models or logs they are in `~/monitor-archives/`.

## Restoring archived files

If you need to restore `activity.db`, `logs/`, or `snapshots/` from the archive run:

```bash
# watchtower-ai (Room Monitor)

A small local monitor that uses camera input for motion detection, face detection/recognition, greeting, and optional voice commands. This README covers setup, model placement, and common troubleshooting steps so contributors can get the project running quickly.

## Quick start (recommended)

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip and install dependencies:

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

3. Download model files and place them under `models/` (see below).

4. Run the monitor:

```bash
./run_monitor.sh
# or
python3 room_monitor.py
```

## Prerequisites

- macOS (tested) or Linux
- Python 3.10+ recommended
- Homebrew (macOS) for optional system deps

Optional system packages (macOS):

```bash
brew install portaudio cmake
# For better GPU support (optional):
brew install libomp
```

If you have trouble building some packages, see the Troubleshooting section.

## Models and where to place them

Create a `models/` directory in the repo root and put large model files there. The code prefers models in `models/`.

Recommended files:

- `models/yolov8n.pt` — YOLOv8 tiny (object detection). Download from Ultralytics releases.
- `models/vosk-model-en-us-0.22/` — Vosk offline speech recognition model folder (optional).
- `models/ggml-model.bin` — Optional GGML-quantized model for local LLM (llama-cpp-python) if you want on-device AI.

Example:

```text
monitor/
  models/
    yolov8n.pt
    vosk-model-en-us-0.22/
    ggml-model.bin
  room_monitor.py
  requirements.txt
```

## Optional: OpenAI fallback

If you want to use OpenAI as a fallback for the assistant features, set:

```bash
export OPENAI_API_KEY="sk-..."
```

The script will attempt to use a local LLM first (if available) and fall back to OpenAI if configured.

## Running the app

Start the app with the provided run script or directly with Python:

```bash
./run_monitor.sh
# or
python3 room_monitor.py
```

The app opens an OpenCV window for the camera feed. Press `q` to quit. If assistant features are enabled, press `a` to type and ask a question.

## Troubleshooting

- `vosk` install errors: pick `vosk==0.3.44` and install PortAudio first (macOS: `brew install portaudio`). If `pyaudio` fails to build, install `sounddevice` and update the script to use it.
- `llama-cpp-python` installation: requires CMake and a compatible GGML model. See the project README for platform-specific notes.
- Camera not available: ensure no other app (Zoom, Photo Booth) is using the camera.
- TTS not working on macOS: ensure `osascript` is available (default) and the project can run `say` via AppleScript.

## Development & contribution notes

- `.gitignore` excludes `models/` and runtime artifacts. Put large or private models in `models/` and do not commit them.
- Use the `models/` directory to keep project root clean.

## Restore archived runtime files

If you previously ran the cleanup and archived logs or the DB, restore them with:

```bash
mv ~/monitor-archives/activity.db ./
mv ~/monitor-archives/logs ./
```

## Next improvements (ideas)

- Helper script `scripts/download_models.sh` to fetch YOLO/Vosk models automatically.
- GitHub Actions to run a lint/test workflow on push.
- Add a small web UI for live monitoring and logs.

If you'd like, I can add any of the above improvements or a helper script to download models automatically.
