#!/usr/bin/env bash
# Builds dist/Atlas.app with PyInstaller using the active virtual environment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
PYTHON="${PYTHON:-$ROOT/venv/bin/python}"

if ! "$PYTHON" -m PyInstaller --version >/dev/null 2>&1; then
    echo "PyInstaller not found. Run: $PYTHON -m pip install pyinstaller" >&2
    exit 1
fi
if [ ! -f "$ROOT/yolov8s.pt" ]; then
    echo "yolov8s.pt not found in $ROOT. Place the production weights there first." >&2
    exit 1
fi

rm -rf "$ROOT/build/Atlas" "$ROOT/dist/Atlas" "$ROOT/dist/Atlas.app"
"$PYTHON" -m PyInstaller --noconfirm \
    --distpath "$ROOT/dist" --workpath "$ROOT/build" \
    "$ROOT/packaging/macos/atlas.spec"

echo "Built: $ROOT/dist/Atlas.app"
