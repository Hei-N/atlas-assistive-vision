"""Resolution of bundled resources and user-writable runtime locations.

Works both from a source checkout and from a PyInstaller bundle. All
bundle detection lives here.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_NAME = "Atlas"
DEFAULT_CONFIG_RELATIVE_PATH = "config/settings.yaml"


def is_bundled() -> bool:
    """True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def get_resource_root() -> Path:
    """Directory that holds read-only resources (config, model weights)."""
    if is_bundled():
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def get_resource_path(relative_path: str) -> Path:
    return get_resource_root() / relative_path


def get_config_path() -> Path:
    return get_resource_path(DEFAULT_CONFIG_RELATIVE_PATH)


def get_model_path(model_name: str) -> str:
    """Return the model reference to hand to Ultralytics.

    A bundled app must never download weights, so a missing file raises.
    From source, an existing local weights file is used and any other name
    is passed through unchanged (Ultralytics may download it).
    """
    if Path(model_name).is_absolute():
        return model_name
    candidate = get_resource_path(model_name)
    if candidate.is_file():
        return str(candidate)
    if is_bundled():
        raise FileNotFoundError(f"Bundled model weights not found: {model_name!r}")
    return model_name


def get_log_dir() -> Path:
    """User-writable log directory (macOS: ~/Library/Logs/Atlas)."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / APP_NAME
    return Path.home() / ".atlas" / "logs"


def get_app_support_dir() -> Path:
    """User-writable application data directory."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path.home() / ".atlas"
