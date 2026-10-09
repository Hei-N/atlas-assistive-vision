"""Unit tests for scripts/extract_evaluation_frames.py. Uses a tiny
generated video (cv2.VideoWriter into tmp_path), matching the pattern
already established in tests/test_benchmark_detector.py -- no real
media, no network.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.annotation_schema import validate_annotations
from scripts.extract_evaluation_frames import (
    FrameSelection,
    deterministic_filename,
    extract_frames,
    select_interval_frames,
    write_frame_manifest,
    write_starter_annotations,
)


def make_tiny_video(path: Path, num_frames: int, width: int = 64, height: int = 48, fps: float = 10.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        writer.write(np.full((height, width, 3), i % 255, dtype=np.uint8))
    writer.release()


# --- deterministic_filename --------------------------------------------------


def test_deterministic_filename_is_sortable_and_unique() -> None:
    a = deterministic_filename("test7", 1408, 23.501)
    b = deterministic_filename("test7", 1498, 25.003)
    assert a == "test7_frame_001408_t023.501.jpg"
    assert b == "test7_frame_001498_t025.003.jpg"
    assert a != b
    assert sorted([b, a]) == [a, b]  # frame_index zero-padding keeps numeric sort == string sort


def test_deterministic_filename_pads_frame_index() -> None:
    assert deterministic_filename("s", 5, 0.1) == "s_frame_000005_t000.100.jpg"


# --- select_interval_frames --------------------------------------------------


def test_select_interval_frames_spacing() -> None:
    indices = select_interval_frames(0.0, 3.0, 1.0, fps=10.0)
    assert indices == [0, 10, 20, 30]


def test_select_interval_frames_single_point_when_start_equals_end() -> None:
    indices = select_interval_frames(5.0, 5.0, 1.0, fps=10.0)
    assert indices == [50]


# --- extract_frames ------------------------------------------------------------


def test_extract_frames_native_resolution_and_correct_metadata(tmp_path) -> None:
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=10, width=64, height=48, fps=10.0)
    frames_dir = tmp_path / "frames"

    selections = [FrameSelection(2, "interval_sample"), FrameSelection(7, "multiple_objects")]
    rows, video_info = extract_frames(video_path, "clip", frames_dir, selections)

    assert video_info["width"] == 64
    assert video_info["height"] == 48
    assert len(rows) == 2
    for row in rows:
        img = cv2.imread(str(frames_dir / row["image_filename"]))
        assert img.shape == (48, 64, 3)  # native resolution, never resized


def test_extract_frames_timestamp_matches_frame_index_over_fps(tmp_path) -> None:
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=10, fps=10.0)
    rows, video_info = extract_frames(
        video_path, "clip", tmp_path / "frames", [FrameSelection(5, "interval_sample")]
    )
    assert rows[0]["timestamp_seconds"] == pytest.approx(5 / video_info["fps"], abs=1e-3)


def test_extract_frames_rejects_duplicate_selection(tmp_path) -> None:
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=5)
    with pytest.raises(ValueError, match="Duplicate"):
        extract_frames(
            video_path, "clip", tmp_path / "frames",
            [FrameSelection(1, "interval_sample"), FrameSelection(1, "multiple_objects")],
        )


def test_extract_frames_missing_video_raises_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_frames(tmp_path / "does_not_exist.mp4", "clip", tmp_path / "frames", [])


def test_extract_frames_produces_unique_filenames(tmp_path) -> None:
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=10, fps=10.0)
    selections = [FrameSelection(i, "interval_sample") for i in (1, 3, 5, 7, 9)]
    rows, _info = extract_frames(video_path, "clip", tmp_path / "frames", selections)
    filenames = [r["image_filename"] for r in rows]
    assert len(set(filenames)) == len(filenames)


# --- write_frame_manifest / write_starter_annotations ------------------------


def test_write_frame_manifest_schema(tmp_path) -> None:
    rows = [
        {
            "source_id": "clip", "source_path": "data/input/clip.mp4", "frame_index": 5,
            "timestamp_seconds": 0.5, "image_filename": "clip_frame_000005_t000.500.jpg",
            "width": 64, "height": 48, "selected_reason": "interval_sample",
        }
    ]
    path = tmp_path / "frame_manifest.csv"
    write_frame_manifest(rows, path)
    with path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == [
        "source_id", "source_path", "frame_index", "timestamp_seconds",
        "image_filename", "width", "height", "selected_reason",
    ]


def test_write_starter_annotations_is_empty_and_validates(tmp_path) -> None:
    path = tmp_path / "annotations.json"
    write_starter_annotations(
        source_id="clip", source_path="data/input/clip.mp4",
        video_info={"fps": 30.0, "width": 640, "height": 480, "total_frames": 100},
        dataset_id="ds", scene_tags=["daytime"], lighting="bright",
        camera_motion="stationary", notes="test", path=path,
    )
    document = json.loads(path.read_text())
    assert document["annotations"] == []
    assert "reviewed_frames" not in document  # nothing fabricated as reviewed

    result = validate_annotations(document)
    assert result.is_valid, result.errors
