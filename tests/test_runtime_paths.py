import sys
from pathlib import Path

import pytest

from src import runtime_paths


def test_source_mode_is_not_bundled():
    assert runtime_paths.is_bundled() is False


def test_config_path_points_to_repo_settings():
    path = runtime_paths.get_config_path()
    assert path.is_absolute()
    assert path.name == "settings.yaml"
    assert path.exists()


def test_model_path_passes_unknown_name_through_in_source_mode():
    assert runtime_paths.get_model_path("not_a_real_model.pt") == "not_a_real_model.pt"


def test_model_path_keeps_absolute_path():
    assert runtime_paths.get_model_path("/tmp/model.pt") == "/tmp/model.pt"


def test_bundled_mode_resolves_resources_from_meipass(tmp_path, monkeypatch):
    (tmp_path / "yolov8s.pt").write_bytes(b"x")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert runtime_paths.is_bundled()
    assert runtime_paths.get_resource_root() == Path(tmp_path)
    assert runtime_paths.get_model_path("yolov8s.pt") == str(tmp_path / "yolov8s.pt")


def test_bundled_mode_missing_model_raises_instead_of_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    with pytest.raises(FileNotFoundError):
        runtime_paths.get_model_path("yolov8s.pt")


def test_user_writable_dirs_are_under_home():
    home = Path.home()
    assert home in runtime_paths.get_log_dir().parents
    assert home in runtime_paths.get_app_support_dir().parents
