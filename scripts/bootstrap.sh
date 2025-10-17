#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

echo "Bootstrapping watchtower-ai environment..."

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3.10+ and retry." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "Upgrading pip and installing Python packages from requirements.txt"
python3 -m pip install --upgrade pip setuptools wheel
if [ -f "$ROOT_DIR/requirements.txt" ]; then
  python3 -m pip install -r "$ROOT_DIR/requirements.txt"
else
  echo "requirements.txt not found in $ROOT_DIR" >&2
  exit 1
fi

echo "Bootstrap complete. Activate the venv with: source $VENV_DIR/bin/activate"
echo "Run the monitor: ./run_monitor.sh or python3 room_monitor.py"

exit 0
