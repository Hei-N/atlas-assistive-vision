"""Unit/integration tests for scripts/benchmark_faster_rcnn_selected_frames.py.

No pretrained weights are downloaded: the real torchvision model factory is
replaced with a fake (via the `model_factory` injection point) that returns
plain torch tensors from a small scripted queue, so the real `_run_inference`
tensor-handling code still runs end-to-end deterministically. Synthetic
frame images live in tmp_path -- no real repo asset is read or written.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from scripts.benchmark_faster_rcnn_selected_frames import (
    COCO91_INDEX_TO_ATLAS_CLASS,
    build_detection_row,
    build_empty_frame_row,
    filter_and_convert_predictions,
    load_target_frames,
    run_faster_rcnn_benchmark,
)

MANIFEST_HEADER = ["source_id", "source_path", "frame_index", "timestamp_seconds",
                   "image_filename", "width", "height", "selected_reason"]


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def make_row(source_id: str, frame_index: int, filename: str, timestamp: float) -> dict:
    return {
        "source_id": source_id, "source_path": f"data/input/{source_id}.mp4",
        "frame_index": frame_index, "timestamp_seconds": timestamp,
        "image_filename": filename, "width": 32, "height": 24, "selected_reason": "interval_sample",
    }


def make_tiny_image(path: Path, width: int = 32, height: int = 24) -> None:
    cv2.imwrite(str(path), np.full((height, width, 3), 100, dtype=np.uint8))


# --- pure helpers -------------------------------------------------------------


def test_load_target_frames_sorted_by_frame_index(tmp_path: Path) -> None:
    manifest = tmp_path / "frame_manifest.csv"
    write_manifest(manifest, [
        make_row("clip", 20, "clip_frame_000020.jpg", 2.0),
        make_row("clip", 5, "clip_frame_000005.jpg", 0.5),
    ])
    rows = load_target_frames(manifest)
    assert [int(r["frame_index"]) for r in rows] == [5, 20]


def test_class_index_mapping_matches_torchvision_coco91() -> None:
    # Exact indices confirmed against FasterRCNN_ResNet50_FPN_V2_Weights
    # .DEFAULT.meta["categories"] during inspection -- pinned here so a
    # future torchvision upgrade that silently changes them is caught.
    assert COCO91_INDEX_TO_ATLAS_CLASS == {
        1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck",
    }


def test_filter_and_convert_predictions_applies_confidence_threshold() -> None:
    boxes = [(0, 0, 10, 10), (0, 0, 20, 20)]
    labels = [3, 3]
    scores = [0.8, 0.3]
    result = filter_and_convert_predictions(boxes, labels, scores, confidence_threshold=0.5)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.8
    assert result[0]["class_name"] == "car"


def test_filter_and_convert_predictions_drops_unmapped_coco_classes() -> None:
    # label 5 = "airplane" in torchvision's COCO-91 list -- not a canonical
    # Atlas class, must be dropped, never invented into car/bus/truck/etc.
    boxes = [(0, 0, 10, 10), (0, 0, 10, 10)]
    labels = [5, 8]
    scores = [0.9, 0.9]
    result = filter_and_convert_predictions(boxes, labels, scores, confidence_threshold=0.5)
    assert len(result) == 1
    assert result[0]["class_name"] == "truck"


def test_filter_and_convert_predictions_empty_input() -> None:
    assert filter_and_convert_predictions([], [], [], confidence_threshold=0.5) == []


def test_build_detection_row_fields() -> None:
    det = {"class_name": "bicycle", "raw_class_id": 2, "confidence": 0.62, "bbox": (10.0, 20.0, 30.0, 60.0)}
    row = build_detection_row(
        source_id="clip", frame_index=7, timestamp=1.5, det=det,
        frame_width=100, frame_height=200, inference_time_ms=50.0,
        total_frame_time_ms=55.0, detections_in_frame=1,
    )
    assert row["source_id"] == "clip"
    assert row["frame_index"] == 7
    assert row["raw_class_name"] == "bicycle"
    assert row["atlas_config_class_name"] == "bicycle"
    assert row["micromobility_candidate"] is True
    assert row["bbox_x1"] == 10.0 and row["bbox_y2"] == 60.0
    assert row["bbox_width"] == 20.0 and row["bbox_height"] == 40.0
    assert row["detections_in_frame"] == 1


def test_build_empty_frame_row_has_zero_detections() -> None:
    row = build_empty_frame_row(
        source_id="clip", frame_index=3, timestamp=0.3,
        frame_width=64, frame_height=48, inference_time_ms=10.0, total_frame_time_ms=11.0,
    )
    assert row["frame_index"] == 3
    assert row["detections_in_frame"] == 0
    assert row["raw_class_name"] == ""


# --- end-to-end with a fake model (no weights downloaded) --------------------


class _FakeModel:
    def __init__(self, per_frame_outputs: list[dict]) -> None:
        self._outputs = per_frame_outputs
        self.call_count = 0

    def __call__(self, inputs):
        if self.call_count < len(self._outputs):
            out = self._outputs[self.call_count]
        else:
            out = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64), "scores": torch.zeros((0,))}
        self.call_count += 1
        return [out]


def _fake_preprocess(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float() / 255.0


def _make_fake_factory(per_frame_outputs: list[dict]):
    def factory(device_name: str):
        model = _FakeModel(per_frame_outputs)
        weights_meta = {
            "weights_enum": "FAKE_FOR_TESTS", "num_params": 123,
            "torch_version": torch.__version__, "torchvision_version": "fake",
        }
        return model, _fake_preprocess, weights_meta
    return factory


def read_csv_rows(output_dir: Path) -> list[dict]:
    with (output_dir / "detections.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_run_faster_rcnn_benchmark_end_to_end_no_weights_downloaded(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    make_tiny_image(frames_dir / "clip_frame_000001.jpg")
    make_tiny_image(frames_dir / "clip_frame_000002.jpg")

    manifest = tmp_path / "frame_manifest.csv"
    write_manifest(manifest, [
        make_row("clip", 1, "clip_frame_000001.jpg", 0.1),
        make_row("clip", 2, "clip_frame_000002.jpg", 0.2),
    ])

    fake_outputs = [
        {"boxes": torch.tensor([[1.0, 1.0, 9.0, 9.0]]), "labels": torch.tensor([3]), "scores": torch.tensor([0.9])},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64), "scores": torch.zeros((0,))},
    ]

    output_dir = run_faster_rcnn_benchmark(
        frames_dir=frames_dir, source_id="clip", manifest_csv=manifest,
        output_dir=tmp_path / "out", confidence_threshold=0.5,
        model_factory=_make_fake_factory(fake_outputs),
    )

    rows = read_csv_rows(output_dir)
    assert len(rows) == 2
    assert rows[0]["frame_index"] == "1"
    assert rows[0]["raw_class_name"] == "car"
    assert rows[1]["frame_index"] == "2"
    assert rows[1]["detections_in_frame"] == "0"

    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["selected_frame_indices"] == [1, 2]
    assert metadata["weights"] == "FAKE_FOR_TESTS"
    assert metadata["model_name"] == "fasterrcnn_resnet50_fpn_v2"

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["total_detections"] == 1
    assert summary["processed_frames"] == 2


def test_run_faster_rcnn_benchmark_respects_confidence_threshold(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    make_tiny_image(frames_dir / "clip_frame_000001.jpg")
    manifest = tmp_path / "frame_manifest.csv"
    write_manifest(manifest, [make_row("clip", 1, "clip_frame_000001.jpg", 0.1)])

    fake_outputs = [
        {"boxes": torch.tensor([[1.0, 1.0, 9.0, 9.0]]), "labels": torch.tensor([3]), "scores": torch.tensor([0.2])},
    ]
    output_dir = run_faster_rcnn_benchmark(
        frames_dir=frames_dir, source_id="clip", manifest_csv=manifest,
        output_dir=tmp_path / "out", confidence_threshold=0.5,
        model_factory=_make_fake_factory(fake_outputs),
    )
    rows = read_csv_rows(output_dir)
    assert len(rows) == 1
    assert rows[0]["detections_in_frame"] == "0"  # below threshold, filtered out


def test_module_does_not_import_production_object_detector() -> None:
    source = Path("scripts/benchmark_faster_rcnn_selected_frames.py").read_text()
    assert "src.object_detector" not in source
    assert "import main" not in source
