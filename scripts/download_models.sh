#!/usr/bin/env bash
set -euo pipefail

# downloads models into ./models
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"
mkdir -p "$MODELS_DIR"

echo "Downloading YOLOv8n (yolov8n.pt)..."
YOLO_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
curl -L "$YOLO_URL" -o "$MODELS_DIR/yolov8n.pt" || { echo "Failed to download YOLO model"; exit 1; }

echo "YOLO downloaded to $MODELS_DIR/yolov8n.pt"

echo "Optional: download Vosk model (approx 50-200MB)."
read -p "Download Vosk English model? [y/N] " yn
if [[ "$yn" =~ ^[Yy]$ ]]; then
  VOSK_URL="https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip"
  TMP="$MODELS_DIR/vosk-model.zip"
  echo "Downloading Vosk model (may be large)..."
  curl -L "$VOSK_URL" -o "$TMP" || { echo "Failed to download Vosk model"; exit 1; }
  echo "Extracting Vosk model..."
  unzip -o "$TMP" -d "$MODELS_DIR"
  rm -f "$TMP"
  echo "Vosk model extracted into $MODELS_DIR"
fi

echo "Done. Models are in: $MODELS_DIR"
