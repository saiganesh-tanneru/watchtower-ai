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
A small local monitor that uses camera input for motion detection, face detection/recognition, greeting, and optional voice commands.

Latest release: v0.1

--

## Quick start (recommended)

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip and install Python dependencies:

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

3. Download models (use the helper):

```bash
./scripts/download_models.sh
```

4. Run the monitor:

```bash
./run_monitor.sh
# or
python3 room_monitor.py
```

The app opens a camera window (OpenCV). Press `q` to quit. If assistant features are enabled, press `a` in the window to type a question.

## Prerequisites

- macOS or Linux
- Python 3.10+ recommended
- Homebrew (macOS) for optional system deps (PortAudio/CMake)

Optional system packages (macOS):

```bash
brew install portaudio cmake unzip
```

## Models and placement

Put large model files in `models/` (project root). The code prefers models in `models/`.

Required / recommended models:

- `models/yolov8n.pt` — YOLOv8 tiny (object detection). The helper script downloads it automatically.
- `models/vosk-model-en-us-0.22/` — Vosk offline speech model (optional). The helper can optionally download and extract it.
- `models/ggml-model.bin` — Optional local LLaMA GGML model for on-device assistant features.

Example layout:

```
monitor/
  models/
    yolov8n.pt
    vosk-model-en-us-0.22/
    ggml-model.bin
  room_monitor.py
  requirements.txt
  scripts/download_models.sh
```

## Assistant & OpenAI fallback

If you want assistant features and OpenAI fallback, set the API key in your environment:

```bash
export OPENAI_API_KEY="sk-..."
```

The script tries local LLM first (if configured) and falls back to OpenAI if available.

## Scripts

- `scripts/download_models.sh` — download YOLO and optionally Vosk model into `models/`.

## Troubleshooting

- Vosk installation: use `vosk==0.3.44`. On macOS install PortAudio first (`brew install portaudio`) before `pip install pyaudio`.
- If `pyaudio` fails to build, install `sounddevice` and adapt the script to use it instead.
- LLaMA local models: `llama-cpp-python` requires CMake and being careful about model size; use 7B GGML for moderate machines.
- If the camera is unavailable, close other apps that may use the camera.

## Contributing

Pull requests welcome. Please open issues for feature requests or bug reports. Consider adding small, focused changes and tests where possible.

## License

This project is licensed under the MIT License — see `LICENSE`.

--

If you'd like, I can add GitHub Actions to run `python -m py_compile` on pushes and create a small `scripts/bootstrap.sh` to bootstrap the environment.
