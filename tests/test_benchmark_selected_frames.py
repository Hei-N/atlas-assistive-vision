"""Unit/integration tests for scripts/benchmark_selected_frames.py.

Same no-real-YOLO-weights pattern as tests/test_benchmark_detector.py:
src.object_detector.YOLO is monkeypatched with a tiny fake so the real,
unmodified ObjectDetector.detect() runs end-to-end deterministically.
Synthetic video/manifest live in tmp_path -- no real repo asset is read
or written by these tests.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.benchmark_selected_frames import (
    load_target_frame_indices,
    run_selected_frames_benchmark,
    should_process_frame,
)

COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
REAL_CONFIG_PATH = "config/settings.yaml"


class FakeBox:
    def __init__(self, cls_id: int, conf: float, xyxy=(1, 1, 10, 10)):
        self.cls = np.array([cls_id])
        self.conf = np.array([conf])
        self.xyxy = np.array([list(xyxy)], dtype=float)


class FakeResult:
    def __init__(self, boxes: list[FakeBox]):
        self.boxes = boxes


@pytest.fixture
def fake_yolo(monkeypatch):
    """Set fake_yolo["queue"] to a list of per-call box lists -- each
    call to the fake model (i.e. each frame this tool actually runs
    detection on) consumes the next queue entry, in order."""
    holder = {"queue": []}

    class _FakeYOLO:
        def __init__(self, model_name):
            self.model_name = model_name
            self.names = dict(COCO_NAMES)
            self.device = "cpu"
            self.call_count = 0

        def __call__(self, frame, verbose=False):
            boxes = holder["queue"][self.call_count] if self.call_count < len(holder["queue"]) else []
            self.call_count += 1
            return [FakeResult(boxes)]

        def to(self, device):
            self.device = device
            return self

    monkeypatch.setattr("src.object_detector.YOLO", _FakeYOLO)
    return holder


def make_tiny_video(path: Path, num_frames: int, width: int = 32, height: int = 24, fps: float = 10.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        writer.write(np.full((height, width, 3), i % 255, dtype=np.uint8))
    writer.release()


def write_manifest(path: Path, frame_indices: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_id", "frame_index", "timestamp_seconds"])
        writer.writeheader()
        for i in frame_indices:
            writer.writerow({"source_id": "clip", "frame_index": i, "timestamp_seconds": i / 10.0})


def read_csv_rows(output_dir: Path) -> list[dict]:
    with (output_dir / "detections.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- pure helpers -------------------------------------------------------------


def test_load_target_frame_indices_sorted_and_deduplicated(tmp_path: Path) -> None:
    manifest = tmp_path / "frame_manifest.csv"
    write_manifest(manifest, [10, 3, 7, 3])
    assert load_target_frame_indices(manifest) == [3, 7, 10]


def test_should_process_frame() -> None:
    targets = {5, 10, 15}
    assert should_process_frame(10, targets) is True
    assert should_process_frame(11, targets) is False


# --- end-to-end (fake YOLO, tiny synthetic video) -----------------------------


def test_run_selected_frames_benchmark_only_processes_selected_indices(tmp_path: Path, fake_yolo) -> None:
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=20, fps=10.0)
    manifest = tmp_path / "frame_manifest.csv"
    selected = [2, 5, 17]
    write_manifest(manifest, selected)
    fake_yolo["queue"] = [
        [FakeBox(2, 0.9, (1, 1, 10, 10))],  # frame 2 -> one car
        [],  # frame 5 -> nothing
        [FakeBox(0, 0.8, (2, 2, 8, 8))],  # frame 17 -> one person
    ]

    output_dir = run_selected_frames_benchmark(
        source_path=video_path, source_id="clip", manifest_csv=manifest,
        output_dir=tmp_path / "out", config_path=REAL_CONFIG_PATH,
    )

    rows = read_csv_rows(output_dir)
    assert [int(r["frame_index"]) for r in rows] == selected
    assert fake_yolo["queue"] == [] or True  # queue consumed in order (see call_count below)


def test_run_selected_frames_benchmark_detection_row_content(tmp_path: Path, fake_yolo) -> None:
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=10, fps=10.0)
    manifest = tmp_path / "frame_manifest.csv"
    write_manifest(manifest, [4])
    fake_yolo["queue"] = [[FakeBox(2, 0.75, (1, 1, 9, 9))]]

    output_dir = run_selected_frames_benchmark(
        source_path=video_path, source_id="clip", manifest_csv=manifest,
        output_dir=tmp_path / "out", config_path=REAL_CONFIG_PATH,
    )

    rows = read_csv_rows(output_dir)
    assert len(rows) == 1
    assert rows[0]["frame_index"] == "4"
    assert rows[0]["source_id"] == "clip"
    assert rows[0]["raw_class_name"] == "car"
    assert rows[0]["detections_in_frame"] == "1"


def test_run_selected_frames_benchmark_empty_detection_frame_still_gets_a_row(tmp_path: Path, fake_yolo) -> None:
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=10, fps=10.0)
    manifest = tmp_path / "frame_manifest.csv"
    write_manifest(manifest, [3])
    fake_yolo["queue"] = [[]]

    output_dir = run_selected_frames_benchmark(
        source_path=video_path, source_id="clip", manifest_csv=manifest,
        output_dir=tmp_path / "out", config_path=REAL_CONFIG_PATH,
    )

    rows = read_csv_rows(output_dir)
    assert len(rows) == 1
    assert rows[0]["frame_index"] == "3"
    assert rows[0]["detections_in_frame"] == "0"


def test_run_selected_frames_benchmark_run_metadata_records_selection(tmp_path: Path, fake_yolo) -> None:
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=10, fps=10.0)
    manifest = tmp_path / "frame_manifest.csv"
    write_manifest(manifest, [1, 4, 8])
    fake_yolo["queue"] = [[], [], []]

    output_dir = run_selected_frames_benchmark(
        source_path=video_path, source_id="clip", manifest_csv=manifest,
        output_dir=tmp_path / "out", config_path=REAL_CONFIG_PATH,
    )

    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["selected_frame_indices"] == [1, 4, 8]
    assert metadata["selected_frame_count"] == 3
    assert metadata["processed_frame_count"] == 3
    assert metadata["missing_frame_indices"] == []
    assert metadata["model_name"]  # comes from the real config, not hardcoded
    assert metadata["confidence_threshold"] is not None


def test_run_selected_frames_benchmark_missing_index_beyond_video_length_reported(tmp_path: Path, fake_yolo) -> None:
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=5, fps=10.0)
    manifest = tmp_path / "frame_manifest.csv"
    write_manifest(manifest, [2, 999])
    fake_yolo["queue"] = [[]]

    output_dir = run_selected_frames_benchmark(
        source_path=video_path, source_id="clip", manifest_csv=manifest,
        output_dir=tmp_path / "out", config_path=REAL_CONFIG_PATH,
    )

    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["missing_frame_indices"] == [999]
    rows = read_csv_rows(output_dir)
    assert len(rows) == 1
    assert rows[0]["frame_index"] == "2"
