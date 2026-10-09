"""Unit tests for scripts/review_evaluation_frames.py's pure, testable
logic (manifest loading, navigation math, text formatting, and the
--validate completeness check). The interactive cv2.imshow/waitKey
viewer loop (run_viewer) is intentionally not exercised here -- it
needs a real display, matching the existing precedent in
tests/test_benchmark_detectors.py of only testing pure helpers, not
the interactive/video loop.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.review_evaluation_frames import (
    check_completeness,
    clamp_index,
    format_overlay_text,
    load_manifest,
)

MANIFEST_HEADER = [
    "source_id", "source_path", "frame_index", "timestamp_seconds",
    "image_filename", "width", "height", "selected_reason",
]


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def make_row(source_id: str, frame_index: int, filename: str, reason: str = "interval_sample") -> dict:
    return {
        "source_id": source_id, "source_path": f"data/input/{source_id}.mp4",
        "frame_index": frame_index, "timestamp_seconds": frame_index / 10.0,
        "image_filename": filename, "width": 64, "height": 48, "selected_reason": reason,
    }


# --- load_manifest -----------------------------------------------------------


def test_load_manifest_sorts_by_frame_index(tmp_path: Path) -> None:
    path = tmp_path / "frame_manifest.csv"
    write_manifest(path, [
        make_row("clip", 20, "clip_frame_000020.jpg"),
        make_row("clip", 5, "clip_frame_000005.jpg"),
    ])
    rows = load_manifest(path)
    assert [r["frame_index"] for r in rows] == ["5", "20"]


# --- format_overlay_text ------------------------------------------------------


def test_format_overlay_text_includes_key_fields() -> None:
    row = make_row("clip", 7, "clip_frame_000007.jpg", reason="school_bus")
    lines = format_overlay_text(row, position=2, total=35)
    joined = " ".join(lines)
    assert "clip_frame_000007.jpg" in joined
    assert "frame_index=7" in joined
    assert "school_bus" in joined
    assert "3/35" in joined  # position is zero-based, displayed 1-based


# --- clamp_index ---------------------------------------------------------------


def test_clamp_index_stays_in_bounds() -> None:
    assert clamp_index(-1, total=5) == 0
    assert clamp_index(0, total=5) == 0
    assert clamp_index(4, total=5) == 4
    assert clamp_index(5, total=5) == 4


def test_clamp_index_empty_total() -> None:
    assert clamp_index(0, total=0) == 0


# --- check_completeness --------------------------------------------------------


def make_eval_dir(tmp_path: Path, annotations: dict, num_frames: int = 3) -> Path:
    eval_dir = tmp_path / "pkg"
    frames_dir = eval_dir / "frames"
    frames_dir.mkdir(parents=True)
    rows = [make_row("clip", i, f"clip_frame_{i:06d}.jpg") for i in range(num_frames)]
    write_manifest(eval_dir / "frame_manifest.csv", rows)
    for row in rows:
        (frames_dir / row["image_filename"]).write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG-ish bytes
    (eval_dir / "annotations.json").write_text(json.dumps(annotations))
    return eval_dir


def base_annotations() -> dict:
    return {
        "schema_version": "1.0", "dataset_id": "ds", "created_at": "2026-01-01T00:00:00Z",
        "class_names": ["car"],
        "sources": [{"source_id": "clip", "path": "data/input/clip.mp4", "source_type": "video"}],
        "annotations": [],
    }


def test_check_completeness_all_unreviewed(tmp_path: Path) -> None:
    eval_dir = make_eval_dir(tmp_path, base_annotations())
    result = check_completeness(eval_dir)
    assert result.missing_frame_files == []
    assert result.unreviewed_frame_indices == [0, 1, 2]
    assert not result.is_complete


def test_check_completeness_detects_missing_frame_file(tmp_path: Path) -> None:
    eval_dir = make_eval_dir(tmp_path, base_annotations())
    (eval_dir / "frames" / "clip_frame_000001.jpg").unlink()
    result = check_completeness(eval_dir)
    assert result.missing_frame_files == ["clip_frame_000001.jpg"]
    assert not result.is_complete


def test_check_completeness_reviewed_frames_entry_counts_as_reviewed(tmp_path: Path) -> None:
    annotations = base_annotations()
    annotations["reviewed_frames"] = [
        {"source_id": "clip", "frame_index": 0, "reviewed": True},
        {"source_id": "clip", "frame_index": 1, "reviewed": True},
        {"source_id": "clip", "frame_index": 2, "reviewed": True},
    ]
    eval_dir = make_eval_dir(tmp_path, annotations)
    result = check_completeness(eval_dir)
    assert result.unreviewed_frame_indices == []
    assert result.missing_frame_files == []
    assert result.schema_errors == []
    assert result.is_complete


def test_check_completeness_annotation_entry_counts_as_reviewed(tmp_path: Path) -> None:
    annotations = base_annotations()
    annotations["annotations"] = [
        {
            "annotation_id": "ann_1", "source_id": "clip", "frame_index": 0,
            "class_name": "car", "bounding_box_xyxy": [1, 1, 10, 10],
            "visibility": "clear", "occlusion": "none", "ignore": False, "difficult": False,
        },
    ]
    annotations["reviewed_frames"] = [
        {"source_id": "clip", "frame_index": 1, "reviewed": True},
        {"source_id": "clip", "frame_index": 2, "reviewed": True},
    ]
    eval_dir = make_eval_dir(tmp_path, annotations)
    result = check_completeness(eval_dir)
    assert result.unreviewed_frame_indices == []
    assert result.is_complete


def test_check_completeness_surfaces_schema_errors(tmp_path: Path) -> None:
    annotations = base_annotations()
    annotations["annotations"] = [
        {
            "annotation_id": "ann_1", "source_id": "clip", "frame_index": 0,
            "class_name": "not_a_real_class", "bounding_box_xyxy": [1, 1, 10, 10],
            "visibility": "clear", "occlusion": "none", "ignore": False, "difficult": False,
        },
    ]
    eval_dir = make_eval_dir(tmp_path, annotations)
    result = check_completeness(eval_dir)
    assert result.schema_errors  # unknown class_name is a hard error
    assert not result.is_complete


def test_check_completeness_never_writes_annotations_file(tmp_path: Path) -> None:
    eval_dir = make_eval_dir(tmp_path, base_annotations())
    before = (eval_dir / "annotations.json").read_text()
    check_completeness(eval_dir)
    after = (eval_dir / "annotations.json").read_text()
    assert before == after
