"""Runs Atlas's production detector on ONLY an exact, non-uniform set of
video frame indices (e.g. the 35 hand-selected Test7 evaluation frames in
data/evaluation/test7_v1/frame_manifest.csv), producing a detections.csv
in the same schema scripts/benchmark_detector.py writes -- so scripts/
evaluate_detector.py can consume it directly.

Exists because scripts/benchmark_detector.py only supports --frame-step/
--max-frames (regular-interval sampling starting at frame 0), which
cannot express an arbitrary, non-uniform set of frame indices. This is
evaluation tooling only -- it does not change production detector
behavior, and reuses (never reimplements) benchmark_detector.py's row
-building/summary helpers and main.py's own load_config/
build_object_detector (the exact production detector construction and
confidence threshold). Never opens a live camera.

Usage:
    python -m scripts.benchmark_selected_frames \\
        --source data/input/Test7.mp4 --source-id test7 \\
        --manifest data/evaluation/test7_v1/frame_manifest.csv \\
        --output-dir data/evaluation/test7_v1/baselines/yolov8s_baseline_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import ultralytics

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import main as atlas_main  # noqa: E402
from src.video_source import VideoSource, VideoSourceError  # noqa: E402
from scripts.benchmark_detector import (  # noqa: E402
    DETECTIONS_CSV_COLUMNS,
    SCHEMA_VERSION,
    RunAccumulator,
    TemporalConsistencyProxy,
    build_detection_row,
    build_detector_for_benchmark,
    build_empty_frame_row,
    build_summary,
    get_device_string,
    get_git_commit,
    sha256_file,
)

TOOL_VERSION = "1.0.0"


def load_target_frame_indices(manifest_csv: Path) -> list[int]:
    """Sorted, de-duplicated frame_index values from a frame_manifest.csv
    (as written by scripts/extract_evaluation_frames.py)."""
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return sorted({int(row["frame_index"]) for row in rows})


def should_process_frame(raw_index: int, target_indices: set[int]) -> bool:
    """Whether a sequentially-read raw frame index is one of the exact
    frames we want detections for. A tiny, separately-testable predicate
    so the frame-skipping logic isn't only exercised inside the full
    (untestable-in-a-sandbox) video-reading loop."""
    return raw_index in target_indices


def run_selected_frames_benchmark(
    *,
    source_path: Path,
    source_id: str,
    manifest_csv: Path,
    output_dir: Path,
    config_path: str,
    model_override: str | None = None,
    confidence_override: float | None = None,
) -> Path:
    target_indices = load_target_frame_indices(manifest_csv)
    target_index_set = set(target_indices)
    if not target_indices:
        raise ValueError(f"No frame_index values found in manifest: {manifest_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    config = atlas_main.load_config(config_path)
    region_analyzer = atlas_main.build_region_analyzer(config)

    class _Args:
        model = model_override
        confidence = confidence_override

    detector, resolved_model_name, resolved_confidence = build_detector_for_benchmark(
        config, _Args(), region_analyzer
    )

    accumulator = RunAccumulator()
    accumulator.seed_classes(
        raw_classes=sorted(set(config["classes"].values())),
        normalized_classes=sorted(set({"car": "Vehicle", "truck": "Vehicle", "bus": "Vehicle",
                                        "motorcycle": "Motorcycle", "bicycle": "Bicycle",
                                        "person": "Person"}.values())),
    )
    proxy = TemporalConsistencyProxy()

    video = VideoSource(str(source_path))
    fps = None
    processed_indices: list[int] = []
    raw_index = 0
    try:
        fps = video.get_fps() or None
        detections_csv_path = output_dir / "detections.csv"
        with detections_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DETECTIONS_CSV_COLUMNS)
            writer.writeheader()

            remaining = set(target_index_set)
            while remaining:
                try:
                    frame = video.read_frame()
                except VideoSourceError:
                    break  # normal end-of-stream

                if should_process_frame(raw_index, target_index_set):
                    frame_start = time.perf_counter()
                    inference_start = time.perf_counter()
                    detections = detector.detect(frame)
                    inference_time_ms = (time.perf_counter() - inference_start) * 1000.0
                    total_frame_time_ms = (time.perf_counter() - frame_start) * 1000.0
                    frame_height, frame_width = frame.shape[:2]
                    timestamp = raw_index / fps if fps else None

                    if detections:
                        for obj in detections:
                            writer.writerow(
                                build_detection_row(
                                    source_id=source_id, frame_index=raw_index,
                                    timestamp=timestamp, obj=obj, track_id=None,
                                    frame_width=frame_width, frame_height=frame_height,
                                    inference_time_ms=inference_time_ms,
                                    total_frame_time_ms=total_frame_time_ms,
                                    detections_in_frame=len(detections), detector=detector,
                                )
                            )
                    else:
                        writer.writerow(
                            build_empty_frame_row(
                                source_id=source_id, frame_index=raw_index,
                                timestamp=timestamp, frame_width=frame_width,
                                frame_height=frame_height, inference_time_ms=inference_time_ms,
                                total_frame_time_ms=total_frame_time_ms,
                            )
                        )

                    accumulator.ingest_frame(
                        source_id, raw_index, detections, inference_time_ms,
                        total_frame_time_ms, fps, frame_width, frame_height,
                    )
                    proxy.update(detections)
                    processed_indices.append(raw_index)
                    remaining.discard(raw_index)

                raw_index += 1
    finally:
        video.release()

    accumulator.total_frames = raw_index
    missing_indices = sorted(target_index_set - set(processed_indices))

    model_hash = None
    for candidate in (Path(resolved_model_name), _REPO_ROOT / resolved_model_name):
        if candidate.exists():
            model_hash = sha256_file(candidate)
            break
    commit, commit_status = get_git_commit(_REPO_ROOT)

    run_metadata = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_tool_version": TOOL_VERSION,
        "benchmark_tool": "scripts.benchmark_selected_frames",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "source_id": source_id,
        "selection_method": "exact_frame_indices_from_manifest",
        "manifest_path": str(manifest_csv),
        "selected_frame_indices": target_indices,
        "selected_frame_count": len(target_indices),
        "processed_frame_count": len(processed_indices),
        "missing_frame_indices": missing_indices,
        "model_name": resolved_model_name,
        "model_file_sha256": model_hash,
        "confidence_threshold": resolved_confidence,
        "iou_threshold": None,
        "iou_threshold_note": (
            "Not overridden by Atlas's ObjectDetector (src/object_detector.py calls "
            "self._model(frame, verbose=False) with no iou= argument) -- this run used "
            "Ultralytics' internal default NMS IoU threshold, unconfigured/unexposed, "
            "exactly like scripts/benchmark_detector.py records it."
        ),
        "input_resolution": (
            {"width": accumulator.first_frame_width, "height": accumulator.first_frame_height}
            if accumulator.first_frame_width is not None else None
        ),
        "device": get_device_string(detector),
        "ultralytics_version": ultralytics.__version__,
        "opencv_version": cv2.__version__,
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "source_fps": accumulator.source_fps,
        "command_line_arguments": sys.argv[1:],
        "git_commit": commit,
        "git_commit_status": commit_status,
    }
    summary = build_summary(accumulator, proxy)

    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Atlas's production detector on an exact, non-uniform set of video "
            "frame indices from a frame_manifest.csv (evaluation tooling only)."
        )
    )
    parser.add_argument("--source", required=True, help="Path to the source video file.")
    parser.add_argument("--source-id", required=True, help="source_id to write into detections.csv (must match annotations.json).")
    parser.add_argument("--manifest", required=True, help="Path to a frame_manifest.csv listing exact frame_index values.")
    parser.add_argument("--output-dir", required=True, help="Result directory (created if missing).")
    parser.add_argument("--config", default=atlas_main.DEFAULT_CONFIG_PATH, help="Path to the YAML config file.")
    parser.add_argument("--model", default=None, help="Override config/settings.yaml's model name/path for this run only.")
    parser.add_argument("--confidence", type=float, default=None, help="Override config/settings.yaml's confidence threshold for this run only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = run_selected_frames_benchmark(
        source_path=Path(args.source), source_id=args.source_id,
        manifest_csv=Path(args.manifest), output_dir=Path(args.output_dir),
        config_path=args.config, model_override=args.model, confidence_override=args.confidence,
    )
    print(f"Selected-frame benchmark complete. Results written to: {output_dir}")


if __name__ == "__main__":
    main()
