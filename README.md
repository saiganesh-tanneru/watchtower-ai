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
# from your shell
mv ~/monitor-archives/activity.db ./
mv ~/monitor-archives/logs ./
mv ~/monitor-archives/snapshots ./
```

## Model installation tips

- YOLOv8: put `yolov8n.pt` in `models/` or the project root. You can download it from Ultralytics releases.
- Vosk: download a Vosk model folder and place it under `models/` (e.g. `models/vosk-model-en-us-0.22/`).
- Local LLM (optional): if using `llama-cpp-python`, place a GGML model at `models/ggml-model.bin` and call `init_local_llama()` in the script or ensure it is automatically detected.

## Notes

- `.gitignore` is configured to ignore `models/`, `activity.db`, `logs/`, and `snapshots/`.
- `.vscode/settings.json` excludes `models/` and archives from file-watcher and explorer.

If you'd like, I can:

- Update `room_monitor.py` to auto-initialize a local LLM when `models/ggml-model.bin` exists.
- Add helper scripts to download common models into `models/` automatically.
