"""Offline detector-benchmarking harness for Atlas.

Runs the SAME pipeline main.py uses (via main.py's own build_* factory
functions -- never a second, drifting reimplementation) against a saved
video, once per configured detector model, and reports metrics relevant
to this project's specific goals: real-time FPS/latency, detection
counts, tracking stability, bounding-box jitter, and -- the most
important project-specific metric -- how often a vehicle that should be
parked gets classified as moving, both with and without the new
motion-reliability filter (src/motion/motion_state_filter.py), so the
filter's actual effect is visible on the same clip/model.

Headless: no cv2.imshow/waitKey, no display window. Separate from the
real-time main loop entirely -- import this module or run it directly,
it never runs as part of `python main.py`.

IMPORTANT ASSUMPTION (read before trusting the false-positive numbers):
this script assumes the input clip contains NO independently-moving
vehicles -- i.e. a static or panning camera over parked cars only
(scenarios A/B in the project's manual validation plan). Under that
assumption, any vehicle-class track ever classified as moving is a false
positive. There is no ground-truth annotation involved; a clip that
violates this assumption will simply produce a misleading count, which
is why this warning is also printed at startup.

Usage:
    python scripts/benchmark_detectors.py --video path/to/clip.mp4
    python scripts/benchmark_detectors.py --video path/to/clip.mp4 \\
        --models yolov8n.pt,yolov8s.pt,yolov8m.pt \\
        --config config/settings.yaml --output logs/my_benchmark.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

# main.py lives at the repo root, one level up from this scripts/ package.
# Running this file directly (`python scripts/benchmark_detectors.py`)
# does not put the repo root on sys.path by default, so `import main`
# would otherwise fail -- this fixup makes both invocation styles work
# (`python scripts/benchmark_detectors.py` and
# `python -m scripts.benchmark_detectors`).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import main as atlas_main  # noqa: E402 (must follow the sys.path fixup)
from src.motion.motion_resolver import resolve_motions_for_tracks  # noqa: E402
from src.motion.motion_state_filter import MOVING  # noqa: E402
from src.video_source import VideoSource, VideoSourceError  # noqa: E402

DEFAULT_MODELS = ("yolov8n.pt", "yolov8s.pt")
VEHICLE_CLASS_NAMES = frozenset({"bicycle", "car", "motorcycle", "bus", "truck"})

BENCHMARK_CSV_COLUMNS = [
    "model_name",
    "input_width",
    "input_height",
    "avg_inference_time_ms",
    "avg_frame_time_ms",
    "avg_fps",
    "total_detections",
    "avg_detections_per_frame",
    "missed_frames_vehicles",
    "track_creation_count",
    "avg_bbox_jitter_normalized",
    "raw_moving_false_positive_frames",
    "filtered_moving_false_positive_frames",
    "uncertain_result_count",
    "peak_memory_mb",
]


def compute_bbox_jitter(
    position_history: Sequence[tuple[float, float]],
    box_sizes: Sequence[tuple[float, float]],
) -> float:
    """Mean frame-to-frame center displacement, normalized by that
    frame's box diagonal, over a single track's history.

    Returns 0.0 for fewer than 2 samples, or if every box has a
    zero-length diagonal (degenerate input) -- never raises or divides
    by zero.
    """
    if len(position_history) < 2:
        return 0.0

    normalized_displacements = []
    for i in range(1, len(position_history)):
        prev_x, prev_y = position_history[i - 1]
        curr_x, curr_y = position_history[i]
        displacement = math.hypot(curr_x - prev_x, curr_y - prev_y)

        width, height = box_sizes[i]
        diagonal = math.hypot(width, height)
        if diagonal > 0:
            normalized_displacements.append(displacement / diagonal)

    if not normalized_displacements:
        return 0.0
    return sum(normalized_displacements) / len(normalized_displacements)


def compute_missed_frames(frame_indices_seen: Sequence[int]) -> int:
    """Count of gaps in a track's per-frame appearance record.

    `frame_indices_seen` need not be pre-sorted. Returns 0 for fewer than
    2 samples (nothing to compare).
    """
    if len(frame_indices_seen) < 2:
        return 0

    sorted_indices = sorted(frame_indices_seen)
    missed = 0
    for i in range(1, len(sorted_indices)):
        gap = sorted_indices[i] - sorted_indices[i - 1] - 1
        if gap > 0:
            missed += gap
    return missed


def get_peak_memory_mb() -> float | None:
    """Peak resident-set-size memory used by this process so far, in MB.

    Uses the stdlib `resource` module (macOS/Linux only). Returns None
    (never raises) on platforms where it's unavailable, e.g. Windows.
    """
    try:
        import resource
    except ImportError:
        return None

    peak_kb_or_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on Linux, bytes on macOS.
    if sys.platform == "darwin":
        return peak_kb_or_bytes / (1024 * 1024)
    return peak_kb_or_bytes / 1024


class ModelBenchmarkResult:
    """One model's aggregated benchmark metrics -- a plain data holder,
    not a canonical shared model (this is benchmark-tool-specific output,
    not part of the live pipeline's typed data model in src/models.py)."""

    def __init__(
        self,
        model_name: str,
        input_width: int,
        input_height: int,
        avg_inference_time_ms: float,
        avg_frame_time_ms: float,
        total_detections: int,
        frames_processed: int,
        missed_frames_vehicles: int,
        track_creation_count: int,
        avg_bbox_jitter_normalized: float,
        raw_moving_false_positive_frames: int,
        filtered_moving_false_positive_frames: int,
        uncertain_result_count: int,
        peak_memory_mb: float | None,
    ) -> None:
        self.model_name = model_name
        self.input_width = input_width
        self.input_height = input_height
        self.avg_inference_time_ms = avg_inference_time_ms
        self.avg_frame_time_ms = avg_frame_time_ms
        self.avg_fps = 1000.0 / avg_frame_time_ms if avg_frame_time_ms > 0 else 0.0
        self.total_detections = total_detections
        self.avg_detections_per_frame = (
            total_detections / frames_processed if frames_processed > 0 else 0.0
        )
        self.missed_frames_vehicles = missed_frames_vehicles
        self.track_creation_count = track_creation_count
        self.avg_bbox_jitter_normalized = avg_bbox_jitter_normalized
        self.raw_moving_false_positive_frames = raw_moving_false_positive_frames
        self.filtered_moving_false_positive_frames = filtered_moving_false_positive_frames
        self.uncertain_result_count = uncertain_result_count
        self.peak_memory_mb = peak_memory_mb

    def as_csv_row(self) -> list:
        return [
            self.model_name,
            self.input_width,
            self.input_height,
            f"{self.avg_inference_time_ms:.2f}",
            f"{self.avg_frame_time_ms:.2f}",
            f"{self.avg_fps:.2f}",
            self.total_detections,
            f"{self.avg_detections_per_frame:.2f}",
            self.missed_frames_vehicles,
            self.track_creation_count,
            f"{self.avg_bbox_jitter_normalized:.4f}",
            self.raw_moving_false_positive_frames,
            self.filtered_moving_false_positive_frames,
            self.uncertain_result_count,
            f"{self.peak_memory_mb:.1f}" if self.peak_memory_mb is not None else "",
        ]


def run_model_benchmark(
    model_name: str,
    video_path: str,
    config: dict,
    max_frames: int | None,
) -> ModelBenchmarkResult:
    """Run one model through the real pipeline (main.py's own build_*
    factories) against `video_path`, headless, collecting metrics.

    Raw vs. filtered "moving false positive" comparison methodology: a
    vehicle-track-frame counts as a RAW false positive when its resolved
    (motion_resolver.py) speed exceeds the motion filter's own configured
    stationary_enter_speed threshold -- i.e. what a naive one-frame check
    would report with NO temporal confirmation/hysteresis/smoothing.
    It counts as a FILTERED false positive when
    MotionStateFilter reports MOVING. Comparing the two isolates what the
    new filter actually changed, on identical input.
    """
    region_analyzer = atlas_main.build_region_analyzer(config)

    model_config = copy.deepcopy(config)
    model_config["model"]["name"] = model_name
    detector = atlas_main.build_object_detector(model_config, region_analyzer)

    tracker = atlas_main.build_object_tracker(config)
    trajectory_predictor = atlas_main.build_trajectory_predictor(config)
    motion_estimator = atlas_main.build_motion_estimator(config)
    motion_compensator = atlas_main.build_motion_compensator(config)
    motion_state_filter = atlas_main.build_motion_state_filter(config)

    vehicle_class_ids = {
        int(class_id)
        for class_id, class_name in config["classes"].items()
        if class_name in VEHICLE_CLASS_NAMES
    }

    video = VideoSource(video_path)
    input_width, input_height = video.get_frame_size()

    total_inference_time = 0.0
    total_frame_time = 0.0
    total_detections = 0
    frames_processed = 0
    previous_raw_frame = None

    track_creation_ids: set[int] = set()
    vehicle_frame_indices: dict[int, list[int]] = {}
    vehicle_position_history: dict[int, list[tuple[float, float]]] = {}
    vehicle_box_sizes: dict[int, list[tuple[float, float]]] = {}
    raw_moving_false_positive_frames = 0
    filtered_moving_false_positive_frames = 0
    uncertain_result_count = 0

    try:
        while True:
            if max_frames is not None and frames_processed >= max_frames:
                break
            try:
                frame = video.read_frame()
            except VideoSourceError:
                break

            frame_height, frame_width = frame.shape[:2]
            frame_start = time.perf_counter()

            inference_start = time.perf_counter()
            detections = detector.detect(frame)
            total_inference_time += time.perf_counter() - inference_start
            total_detections += len(detections)

            tracked_objects = tracker.update(detections)

            motion_estimate = None
            if motion_estimator is not None and previous_raw_frame is not None:
                motion_estimate = motion_estimator.estimate(
                    previous_raw_frame, frame, None
                )
            previous_raw_frame = frame.copy()

            compensated_motions = {}
            if motion_compensator is not None:
                compensated_motions = motion_compensator.compensate(
                    tracked_objects, motion_estimate
                )

            predictions = {
                obj.track_id: trajectory_predictor.predict(
                    obj.position_history, obj.frames_since_seen,
                    frame_width, frame_height,
                )
                for obj in tracked_objects
            }
            resolved_motions = resolve_motions_for_tracks(
                tracked_objects, compensated_motions, motion_estimate, predictions
            )
            filtered_motions = motion_state_filter.filter_for_tracks(
                tracked_objects, resolved_motions
            )

            for obj in tracked_objects:
                track_creation_ids.add(obj.track_id)
                if obj.class_id not in vehicle_class_ids:
                    continue

                vehicle_frame_indices.setdefault(obj.track_id, []).append(
                    frames_processed
                )
                vehicle_position_history.setdefault(obj.track_id, []).append(
                    obj.center
                )
                vehicle_box_sizes.setdefault(obj.track_id, []).append(
                    (obj.bbox.width, obj.bbox.height)
                )

                resolved = resolved_motions.get(obj.track_id)
                if (
                    resolved is not None
                    and resolved.speed > motion_state_filter.stationary_enter_speed
                ):
                    raw_moving_false_positive_frames += 1

                filtered = filtered_motions.get(obj.track_id)
                if filtered is not None:
                    if filtered.uncertain:
                        uncertain_result_count += 1
                    if filtered.motion_state == MOVING:
                        filtered_moving_false_positive_frames += 1

            total_frame_time += time.perf_counter() - frame_start
            frames_processed += 1
    finally:
        video.release()

    missed_frames_vehicles = sum(
        compute_missed_frames(indices) for indices in vehicle_frame_indices.values()
    )
    jitter_values = [
        compute_bbox_jitter(
            vehicle_position_history[track_id], vehicle_box_sizes[track_id]
        )
        for track_id in vehicle_position_history
    ]
    avg_bbox_jitter_normalized = (
        sum(jitter_values) / len(jitter_values) if jitter_values else 0.0
    )

    avg_inference_time_ms = (
        1000.0 * total_inference_time / frames_processed if frames_processed > 0 else 0.0
    )
    avg_frame_time_ms = (
        1000.0 * total_frame_time / frames_processed if frames_processed > 0 else 0.0
    )

    return ModelBenchmarkResult(
        model_name=model_name,
        input_width=input_width,
        input_height=input_height,
        avg_inference_time_ms=avg_inference_time_ms,
        avg_frame_time_ms=avg_frame_time_ms,
        total_detections=total_detections,
        frames_processed=frames_processed,
        missed_frames_vehicles=missed_frames_vehicles,
        track_creation_count=len(track_creation_ids),
        avg_bbox_jitter_normalized=avg_bbox_jitter_normalized,
        raw_moving_false_positive_frames=raw_moving_false_positive_frames,
        filtered_moving_false_positive_frames=filtered_moving_false_positive_frames,
        uncertain_result_count=uncertain_result_count,
        peak_memory_mb=get_peak_memory_mb(),
    )


def print_console_summary(results: list[ModelBenchmarkResult]) -> None:
    headers = [
        "model", "res", "inf_ms", "frame_ms", "fps", "dets",
        "missed", "tracks", "jitter", "raw_fp", "filt_fp", "uncert", "mem_mb",
    ]
    rows = [
        [
            r.model_name,
            f"{r.input_width}x{r.input_height}",
            f"{r.avg_inference_time_ms:.1f}",
            f"{r.avg_frame_time_ms:.1f}",
            f"{r.avg_fps:.1f}",
            str(r.total_detections),
            str(r.missed_frames_vehicles),
            str(r.track_creation_count),
            f"{r.avg_bbox_jitter_normalized:.3f}",
            str(r.raw_moving_false_positive_frames),
            str(r.filtered_moving_false_positive_frames),
            str(r.uncertain_result_count),
            f"{r.peak_memory_mb:.0f}" if r.peak_memory_mb is not None else "n/a",
        ]
        for r in results
    ]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


def write_csv(results: list[ModelBenchmarkResult], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(BENCHMARK_CSV_COLUMNS)
        for result in results:
            writer.writerow(result.as_csv_row())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark multiple detector models on the same saved video, "
            "using Atlas's own pipeline (main.py's build_* factories). "
            "Separate from the real-time main loop -- headless, metrics only."
        )
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to a saved video clip. Required -- no path is assumed.",
    )
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS),
        help=(
            "Comma-separated Ultralytics model names/paths to compare "
            f"(default: {','.join(DEFAULT_MODELS)} -- both already "
            "downloaded in this repo). A model not already downloaded "
            "will be fetched by Ultralytics on first use."
        ),
    )
    parser.add_argument(
        "--config", default=atlas_main.DEFAULT_CONFIG_PATH,
        help=f"Path to the YAML config file (default: {atlas_main.DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--output", default=None,
        help="CSV output path (default: logs/detector_benchmark_<timestamp>.csv).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Optional cap on frames processed per model (default: entire clip).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = atlas_main.load_config(args.config)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]

    output_path = args.output or str(
        Path("logs") / f"detector_benchmark_{datetime.now():%Y-%m-%d_%H%M%S}.csv"
    )

    print(
        "ASSUMPTION: this clip is assumed to contain NO independently-"
        "moving vehicles (static/panning camera over parked cars only). "
        "Any vehicle-class track ever classified MOVING is counted as a "
        "false positive under that assumption -- see this script's "
        "module docstring."
    )
    print(f"Benchmarking {len(model_names)} model(s) against {args.video!r}...")

    results = []
    for model_name in model_names:
        print(f"  Running {model_name}...")
        result = run_model_benchmark(model_name, args.video, config, args.max_frames)
        results.append(result)

    print()
    print_console_summary(results)
    write_csv(results, output_path)
    print(f"\nCSV written to {output_path}")


if __name__ == "__main__":
    main()
