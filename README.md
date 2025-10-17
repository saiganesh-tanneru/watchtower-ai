# watchtower-ai (Room Monitor)

A small local monitor that uses camera input for motion detection, face detection/recognition, greeting, and optional voice commands.

Latest release: v0.1

---

## Quick start (recommended)

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Bootstrap environment (create venv + install deps):

```bash
./scripts/bootstrap.sh
```

3. Download models (helper):

```bash
./scripts/download_models.sh
```

4. Run the monitor:

```bash
./run_monitor.sh
# or
python3 room_monitor.py
```

The app opens an OpenCV camera window. Press `q` to quit. Press `a` (if enabled in the UI) to invoke assistant input.

---

## Prerequisites

- macOS or Linux
- Python 3.10+ recommended
- Homebrew (macOS) for optional system deps

Optional system packages (macOS):

```bash
brew install portaudio cmake unzip
```

---

## Models and placement

Put large model files in `models/` (project root). The code prefers models in `models/`.

Recommended models:

- `models/yolov8n.pt` — YOLOv8 tiny (object detection)
- `models/vosk-model-en-us-0.22/` — Vosk offline speech model (optional)
- `models/ggml-model.bin` — Optional local LLaMA GGML model

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
  scripts/bootstrap.sh
```

---

## Assistant & OpenAI fallback

If you want assistant features and OpenAI fallback, set the API key in your environment:

```bash
export OPENAI_API_KEY="sk-..."
```

The app tries a local LLM first (if configured) and falls back to OpenAI if available.

---

## Scripts

- `scripts/bootstrap.sh` — create `.venv` and install `requirements.txt` into it.
- `scripts/download_models.sh` — download YOLOv8n and optionally the Vosk English model into `models/`.

---

## Troubleshooting

- Vosk install: use `vosk==0.3.44` and install PortAudio first on macOS (`brew install portaudio`).
- If `pyaudio` fails to build, consider using `sounddevice` instead.
- `llama-cpp-python` (local LLM) requires CMake and a compatible GGML model.
- If camera is unavailable, close other apps that may be using it.

---

## Contributing

Pull requests welcome. Please include small focused changes and tests where appropriate. Open an issue first if you're planning a large feature.

## License

This project is licensed under the MIT License — see `LICENSE`.
