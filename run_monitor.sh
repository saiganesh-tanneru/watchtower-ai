#!/bin/bash

# Runner script for the Room Monitor
# Creates virtual environment if missing (with system site packages for offline availability)

set -e

# Create .venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv --system-site-packages
fi

# Activate the virtual environment
source ./.venv/bin/activate

# Ensure system packages like cv2/numpy (installed under /opt/pyvenv) are discoverable
if [ -d "/opt/pyvenv/lib/python3.11/site-packages" ]; then
    export PYTHONPATH="/opt/pyvenv/lib/python3.11/site-packages:${PYTHONPATH:-}"
fi

# Execute the monitor script
python room_monitor.py