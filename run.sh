#!/usr/bin/env bash
# Sets up a virtual environment on first use, then starts Atlas.
# Usage: ./run.sh [main.py options]   (defaults to --source 0, the webcam)
set -e
cd "$(dirname "$0")"

if [ ! -x venv/bin/python ]; then
    echo "First run: creating virtual environment and installing dependencies..."
    python3 -m venv venv
    venv/bin/pip install -q --upgrade pip
    venv/bin/pip install -q -r requirements.txt
fi

if [ $# -eq 0 ]; then
    set -- --source 0
fi
exec venv/bin/python main.py "$@"
