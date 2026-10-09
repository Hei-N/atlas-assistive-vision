"""Entry point for the packaged macOS app.

Prepares paths and logging, then runs the existing Atlas loop from main.py.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import traceback

from src.runtime_paths import get_app_support_dir, get_log_dir, is_bundled

LOG_FILE_NAME = "atlas.log"
logger = logging.getLogger("atlas")


class _ErrorCollector(logging.Handler):
    """Remembers ERROR records so a failed start can be shown to the user."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _configure_logging() -> _ErrorCollector:
    log_dir = get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / LOG_FILE_NAME, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    collector = _ErrorCollector()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(collector)
    return collector


def _show_error_dialog(message: str) -> None:
    text = f"{message}\\n\\nDetails: {get_log_dir() / LOG_FILE_NAME}".replace('"', "'")
    script = f'display dialog "{text}" with title "Atlas" buttons {{"OK"}} with icon stop'
    try:
        subprocess.run(["/usr/bin/osascript", "-e", script], timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _prepare_argv() -> None:
    # Finder may pass a -psn_* argument; the app is launched without options,
    # so default to the first camera.
    args = [a for a in sys.argv[1:] if not a.startswith("-psn_")]
    if "--source" not in args and not any(a.startswith("--source=") for a in args):
        args += ["--source", "0"]
    sys.argv = [sys.argv[0]] + args


def launch() -> int:
    collector = _configure_logging()
    if is_bundled():
        # Relative runtime paths must not resolve into "/" or the app bundle.
        support_dir = get_app_support_dir()
        support_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(support_dir)
        # Weights are bundled; never reach out to the network.
        os.environ.setdefault("YOLO_OFFLINE", "1")
        os.environ.setdefault("MPLCONFIGDIR", str(support_dir / "matplotlib"))
    _prepare_argv()
    try:
        import main as atlas_main

        atlas_main.run(atlas_main.parse_args())
    except Exception:
        logger.error("Fatal error:\n%s", traceback.format_exc())
        _show_error_dialog("Atlas could not start. See the log for details.")
        return 1
    if collector.messages:
        _show_error_dialog(f"Atlas stopped: {collector.messages[-1]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(launch())
