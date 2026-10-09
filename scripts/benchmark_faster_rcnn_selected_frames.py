"""Evaluation-only accuracy-reference benchmark: runs torchvision's
pretrained `fasterrcnn_resnet50_fpn_v2` (official COCO weights, unmodified
-- no fine-tuning, no architecture change) on the exact 35 selected Test7
frames, producing a `detections.csv` in the same schema `scripts.
benchmark_detector` writes, so `scripts.evaluate_detector` can consume it
directly without modification.

This is NOT a production detector integration. Faster R-CNN is never
routed through `src/object_detector.py`/`ObjectDetector`, is never called
by `main.py`, and nothing here changes YOLO tooling or evaluator behavior.
It exists purely as an offline accuracy-reference diagnostic -- is a
stronger two-stage detector leaving accuracy on the table that YOLO
misses? -- never as a production candidate.

Reuses (never reimplements) scripts/benchmark_detector.py's
DETECTIONS_CSV_COLUMNS / ATLAS_NORMALIZED_NAME_MAP / round_or_none /
mean_or_none / median_or_none / percentile_or_none / sha256_file /
get_git_commit pure helpers.

Usage:
    python -m scripts.benchmark_faster_rcnn_selected_frames \\
        --frames-dir data/evaluation/test7_v1/frames \\
        --source-id test7 \\
        --manifest data/evaluation/test7_v1/frame_manifest.csv \\
        --output-dir data/evaluation/test7_v1/baselines/fasterrcnn_resnet50_fpn_v2_baseline_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmark_detector import (  # noqa: E402
    ATLAS_NORMALIZED_NAME_MAP,
    DETECTIONS_CSV_COLUMNS,
    SCHEMA_VERSION,
    get_git_commit,
    mean_or_none,
    median_or_none,
    percentile_or_none,
    round_or_none,
    sha256_file,
)

TOOL_VERSION = "1.0.0"

# torchvision's own COCO-91-slot category indices (`__background__` at 0),
# NOT Ultralytics' compact COCO-80 indices used elsewhere in this project --
# see FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]. Only
# Atlas's 6 canonical classes are kept; every other COCO category (and any
# unmapped label id) is discarded -- never invented into car/bus/truck/etc.
COCO91_INDEX_TO_ATLAS_CLASS = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}


def load_target_frames(manifest_csv: Path) -> list[dict]:
    """Rows (sorted by frame_index) from frame_manifest.csv."""
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return sorted(rows, key=lambda r: int(r["frame_index"]))


def filter_and_convert_predictions(
    boxes: list[tuple[float, float, float, float]],
    labels: list[int],
    scores: list[float],
    confidence_threshold: float,
    class_index_map: dict[int, str] | None = None,
) -> list[dict]:
    """Pure conversion/filter step -- no torch/model dependency, fully
    testable with plain lists. Drops any label id not in class_index_map
    (never invented into a supported Atlas class) and any score below
    confidence_threshold."""
    if class_index_map is None:
        class_index_map = COCO91_INDEX_TO_ATLAS_CLASS
    detections = []
    for box, label, score in zip(boxes, labels, scores):
        if score < confidence_threshold:
            continue
        class_name = class_index_map.get(int(label))
        if class_name is None:
            continue
        x1, y1, x2, y2 = box
        detections.append({
            "class_name": class_name,
            "raw_class_id": int(label),
            "confidence": float(score),
            "bbox": (float(x1), float(y1), float(x2), float(y2)),
        })
    return detections


def build_detection_row(
    *, source_id: str, frame_index: int, timestamp: float | None, det: dict,
    frame_width: int, frame_height: int, inference_time_ms: float,
    total_frame_time_ms: float, detections_in_frame: int,
) -> dict:
    x1, y1, x2, y2 = det["bbox"]
    bw, bh = x2 - x1, y2 - y1
    center_x, center_y = x1 + bw / 2, y1 + bh / 2
    return {
        "source_id": source_id,
        "frame_index": frame_index,
        "timestamp_in_source_seconds": round_or_none(timestamp),
        "track_id": "",
        "raw_class_id": det["raw_class_id"],
        "raw_class_name": det["class_name"],
        "atlas_config_class_name": det["class_name"],
        "atlas_normalized_class_name": ATLAS_NORMALIZED_NAME_MAP.get(det["class_name"], det["class_name"]),
        "micromobility_candidate": det["class_name"] in ("bicycle", "motorcycle"),
        "confidence": round_or_none(det["confidence"]),
        "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
        "bbox_width": bw, "bbox_height": bh, "bbox_area": bw * bh,
        "center_x": center_x, "center_y": center_y,
        "normalized_center_x": round_or_none(center_x / frame_width) if frame_width else "",
        "normalized_center_y": round_or_none(center_y / frame_height) if frame_height else "",
        "frame_width": frame_width, "frame_height": frame_height,
        "inference_time_ms": round_or_none(inference_time_ms),
        "preprocess_time_ms": "",
        "postprocess_time_ms": "",
        "total_frame_time_ms": round_or_none(total_frame_time_ms),
        "detections_in_frame": detections_in_frame,
    }


def build_empty_frame_row(
    *, source_id: str, frame_index: int, timestamp: float | None,
    frame_width: int, frame_height: int, inference_time_ms: float,
    total_frame_time_ms: float,
) -> dict:
    row = {column: "" for column in DETECTIONS_CSV_COLUMNS}
    row.update({
        "source_id": source_id,
        "frame_index": frame_index,
        "timestamp_in_source_seconds": round_or_none(timestamp),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "inference_time_ms": round_or_none(inference_time_ms),
        "total_frame_time_ms": round_or_none(total_frame_time_ms),
        "detections_in_frame": 0,
    })
    return row


def run_faster_rcnn_benchmark(
    *, frames_dir: Path, source_id: str, manifest_csv: Path, output_dir: Path,
    confidence_threshold: float = 0.50, device_name: str = "cpu",
    model_factory=None,
) -> Path:
    """model_factory, if given, must return (model, preprocess_transform,
    weights_meta_dict) -- overridable so tests never need to download real
    weights. Defaults to the real torchvision COCO_V1-pretrained model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_target_frames(manifest_csv)
    if not rows:
        raise ValueError(f"No rows found in manifest: {manifest_csv}")

    load_start = time.perf_counter()
    if model_factory is None:
        model, preprocess, weights_meta = _load_real_model(device_name)
    else:
        model, preprocess, weights_meta = model_factory(device_name)
    model_load_time_s = time.perf_counter() - load_start

    inference_times_ms: list[float] = []
    total_frame_times_ms: list[float] = []
    detections_per_frame_counts: list[int] = []
    class_counts: dict[str, int] = {}
    confidences_by_class: dict[str, list[float]] = {}
    frame_width = frame_height = None

    detections_csv_path = output_dir / "detections.csv"
    with detections_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETECTIONS_CSV_COLUMNS)
        writer.writeheader()

        for row in rows:
            frame_index = int(row["frame_index"])
            timestamp = float(row["timestamp_seconds"])
            image_path = frames_dir / row["image_filename"]
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise FileNotFoundError(f"Could not read frame image: {image_path}")
            frame_height, frame_width = frame.shape[:2]

            frame_start = time.perf_counter()
            inference_start = time.perf_counter()
            boxes, labels, scores = _run_inference(model, preprocess, frame, device_name)
            inference_time_ms = (time.perf_counter() - inference_start) * 1000.0

            detections = filter_and_convert_predictions(
                boxes, labels, scores, confidence_threshold
            )
            total_frame_time_ms = (time.perf_counter() - frame_start) * 1000.0

            detections_per_frame_counts.append(len(detections))
            if detections:
                for det in detections:
                    writer.writerow(
                        build_detection_row(
                            source_id=source_id, frame_index=frame_index, timestamp=timestamp,
                            det=det, frame_width=frame_width, frame_height=frame_height,
                            inference_time_ms=inference_time_ms, total_frame_time_ms=total_frame_time_ms,
                            detections_in_frame=len(detections),
                        )
                    )
                    class_counts[det["class_name"]] = class_counts.get(det["class_name"], 0) + 1
                    confidences_by_class.setdefault(det["class_name"], []).append(det["confidence"])
            else:
                writer.writerow(
                    build_empty_frame_row(
                        source_id=source_id, frame_index=frame_index, timestamp=timestamp,
                        frame_width=frame_width, frame_height=frame_height,
                        inference_time_ms=inference_time_ms, total_frame_time_ms=total_frame_time_ms,
                    )
                )

            inference_times_ms.append(inference_time_ms)
            total_frame_times_ms.append(total_frame_time_ms)

    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        peak_rss_bytes *= 1024  # Linux reports KB, macOS reports bytes

    model_hash = None
    try:
        import torch as _torch
        cache_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        if cache_path.exists():
            for f in cache_path.glob("fasterrcnn_resnet50_fpn_v2*"):
                model_hash = sha256_file(f)
                break
    except Exception:
        pass

    commit, commit_status = get_git_commit(_REPO_ROOT)
    mean_total_frame_time_ms = mean_or_none(total_frame_times_ms)
    effective_fps = 1000.0 / mean_total_frame_time_ms if mean_total_frame_time_ms else None

    run_metadata = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_tool_version": TOOL_VERSION,
        "benchmark_tool": "scripts.benchmark_faster_rcnn_selected_frames",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_id": source_id,
        "frames_dir": str(frames_dir),
        "manifest_path": str(manifest_csv),
        "selected_frame_indices": [int(r["frame_index"]) for r in rows],
        "selected_frame_count": len(rows),
        "processed_frame_count": len(rows),
        "model_name": "fasterrcnn_resnet50_fpn_v2",
        "weights": weights_meta.get("weights_enum", "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1"),
        "model_num_params": weights_meta.get("num_params"),
        "model_file_sha256": model_hash,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": None,
        "iou_threshold_note": (
            "torchvision's own internal RPN/ROI NMS IoU thresholds (defaults, unmodified) "
            "-- not exposed/overridden by this script, same convention as scripts/"
            "benchmark_detector.py records for YOLO's own internal NMS."
        ),
        "class_index_mapping": {str(k): v for k, v in COCO91_INDEX_TO_ATLAS_CLASS.items()},
        "input_resolution": (
            {"width": frame_width, "height": frame_height} if frame_width is not None else None
        ),
        "device": device_name,
        "model_load_time_seconds": round_or_none(model_load_time_s),
        "peak_memory_bytes": int(peak_rss_bytes),
        "torch_version": weights_meta.get("torch_version"),
        "torchvision_version": weights_meta.get("torchvision_version"),
        "opencv_version": cv2.__version__,
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "command_line_arguments": sys.argv[1:],
        "git_commit": commit,
        "git_commit_status": commit_status,
    }

    total_detections = sum(class_counts.values())
    summary = {
        "observational_baseline": True,
        "processed_frames": len(rows),
        "total_detections": total_detections,
        "detections_per_class": dict(sorted(class_counts.items())),
        "mean_confidence_by_class": {
            k: round_or_none(mean_or_none(v)) for k, v in confidences_by_class.items()
        },
        "median_confidence_by_class": {
            k: round_or_none(median_or_none(v)) for k, v in confidences_by_class.items()
        },
        "average_detections_per_frame": round_or_none(
            mean_or_none([float(c) for c in detections_per_frame_counts])
        ),
        "mean_inference_time_ms": round_or_none(mean_or_none(inference_times_ms)),
        "median_inference_time_ms": round_or_none(median_or_none(inference_times_ms)),
        "p95_inference_time_ms": round_or_none(percentile_or_none(inference_times_ms, 95)),
        "mean_total_frame_time_ms": round_or_none(mean_total_frame_time_ms),
        "effective_processing_fps": round_or_none(effective_fps),
        "source_fps": 59.91214805213509,  # Test7's known real source fps (not re-derived here)
        "real_time_factor": round_or_none(effective_fps / 59.91214805213509) if effective_fps else None,
    }

    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return output_dir


def _load_real_model(device_name: str):
    import torch
    import torchvision
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.0)
    model.eval()
    model.to(device_name)
    preprocess = weights.transforms()
    weights_meta = {
        "weights_enum": str(weights),
        "num_params": weights.meta.get("num_params"),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }
    return model, preprocess, weights_meta


def _run_inference(model, preprocess, frame_bgr, device_name: str):
    import torch

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)
    input_tensor = preprocess(tensor).to(device_name)
    with torch.no_grad():
        output = model([input_tensor])[0]
    boxes = output["boxes"].cpu().tolist()
    labels = output["labels"].cpu().tolist()
    scores = output["scores"].cpu().tolist()
    return boxes, labels, scores


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run torchvision's pretrained fasterrcnn_resnet50_fpn_v2 (official COCO "
            "weights, unmodified) on an exact set of already-extracted frame images, "
            "as an offline accuracy-reference diagnostic (evaluation tooling only)."
        )
    )
    parser.add_argument("--frames-dir", required=True, help="Directory containing the already-extracted frame JPGs.")
    parser.add_argument("--source-id", required=True, help="source_id to write into detections.csv (must match annotations.json).")
    parser.add_argument("--manifest", required=True, help="Path to a frame_manifest.csv listing exact frames.")
    parser.add_argument("--output-dir", required=True, help="Result directory (created if missing).")
    parser.add_argument("--confidence", type=float, default=0.50, help="Score threshold (default: 0.50, matching production YOLO threshold).")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = run_faster_rcnn_benchmark(
        frames_dir=Path(args.frames_dir), source_id=args.source_id,
        manifest_csv=Path(args.manifest), output_dir=Path(args.output_dir),
        confidence_threshold=args.confidence, device_name=args.device,
    )
    print(f"Faster R-CNN selected-frame benchmark complete. Results written to: {output_dir}")


if __name__ == "__main__":
    main()
