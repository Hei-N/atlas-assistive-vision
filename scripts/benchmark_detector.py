"""Single-model detector benchmark tool for Atlas (observational baseline
only -- see docs/DETECTOR_BENCHMARK.md).

Separate from scripts/benchmark_detectors.py (existing, plural): that
script compares multiple detector models on one video for this
project's own concerns (vehicle false positives under the motion
filter, jitter, FPS/memory). This tool instead measures ONE model (today: the production YOLOv8s
configuration, reused unmodified) against a fixed image/video/
directory/manifest, producing a structured, comparable, reusable
baseline (run_metadata.json + detections.csv + summary.json) for
future detector-swap decisions.

Reuses main.py's own load_config/build_region_analyzer/
build_object_detector -- the exact production detector construction and
confidence thresholds, never re-implemented. Never opens a live camera.
Never claims precision/recall/F1/mAP/miss-rate/false-positive-rate --
none of those are computable without labeled ground truth, which this
tool does not have or invent.

Usage:
    python -m scripts.benchmark_detector --source path/to/clip.mp4
    python scripts/benchmark_detector.py --source path/to/image.jpg
    python scripts/benchmark_detector.py --source path/to/folder/
    python scripts/benchmark_detector.py --source path/to/manifest.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import cv2
import ultralytics
import yaml

# main.py lives at the repo root, one level up from this scripts/
# package -- same sys.path fixup as the existing plural benchmark
# script, so both `python scripts/benchmark_detector.py` and
# `python -m scripts.benchmark_detector` work.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import main as atlas_main  # noqa: E402 (must follow the sys.path fixup)
from src.object_detector import ObjectDetector  # noqa: E402
from src.object_tracker import ObjectTracker  # noqa: E402
from src.video_source import VideoSource, VideoSourceError  # noqa: E402

SCHEMA_VERSION = "1.0"
BENCHMARK_TOOL_VERSION = "1.0.0"

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".m4v"})
MANIFEST_EXTENSIONS = frozenset({".yaml", ".yml", ".json"})

SOURCE_IMAGE = "IMAGE"
SOURCE_VIDEO = "VIDEO"
SOURCE_DIRECTORY = "DIRECTORY"
SOURCE_MANIFEST = "MANIFEST"

# Protected classes must never regress; priority classes are the ones a
# future detector should improve. Both are RAW/config class names (not
# collapsed into Atlas's normalized "Vehicle" bucket) -- see docs/
# DETECTOR_BENCHMARK.md "Class handling".
PROTECTED_CLASSES = ("car", "bus", "truck")
PRIORITY_CLASSES = ("bicycle", "motorcycle")
ALSO_TRACKED_CLASSES = ("person",)

# Mirrors (does NOT import) src/audio/audio_hazard_resolver.py's private
# _OBJECT_NAME_MAP. Intentionally duplicated, not shared: this is a
# read-only, display-only echo for benchmark output, never a second
# source of truth Threat Assessment depends on.
ATLAS_NORMALIZED_NAME_MAP = {
    "car": "Vehicle",
    "truck": "Vehicle",
    "bus": "Vehicle",
    "motorcycle": "Motorcycle",
    "bicycle": "Bicycle",
    "person": "Person",
}
MICROMOBILITY_NORMALIZED_NAMES = frozenset({"Bicycle", "Motorcycle"})

REQUIRES_GROUND_TRUTH_FOR = (
    "precision", "recall", "f1_score", "mean_average_precision",
    "missed_object_rate", "false_positive_rate", "detection_accuracy",
)

DEFAULT_FRAME_TIME_BUDGET_MS = 1000.0 / 30.0  # assumed 30fps when source fps is unknown

DETECTIONS_CSV_COLUMNS = [
    "source_id", "frame_index", "timestamp_in_source_seconds", "track_id",
    "raw_class_id", "raw_class_name", "atlas_config_class_name",
    "atlas_normalized_class_name", "micromobility_candidate", "confidence",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_width", "bbox_height",
    "bbox_area", "center_x", "center_y", "normalized_center_x", "normalized_center_y",
    "frame_width", "frame_height", "inference_time_ms", "preprocess_time_ms",
    "postprocess_time_ms", "total_frame_time_ms", "detections_in_frame",
]

REQUIRED_MANIFEST_FIELDS = ("id", "path", "source_type")


class BenchmarkSourceError(Exception):
    """Raised when --source does not exist or has an unsupported type."""


class BenchmarkManifestError(Exception):
    """Raised when a manifest file is malformed or an entry is invalid."""


class BenchmarkOutputExistsError(Exception):
    """Raised when the resolved output directory already has content and
    --overwrite was not passed."""


@dataclass(frozen=True)
class SourceEntry:
    """One image or video to run the detector over. A directory or
    manifest source expands into a list of these; a single image/video
    source is a list of exactly one."""

    id: str
    path: Path
    source_type: str  # SOURCE_IMAGE or SOURCE_VIDEO


# --- pure helpers (independently unit-testable) ---------------------------


def round_or_none(value: float | None, ndigits: int = 4) -> float | None:
    return None if value is None else round(value, ndigits)


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def percentile_or_none(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile (matching NumPy's default
    "linear" method) over `values`, pct in [0, 100]. None for an empty
    input; the single value for a length-1 input."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = rank - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Standard intersection-over-union of two (x1, y1, x2, y2) boxes.
    0.0 for non-overlapping or degenerate (zero-area) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def sha256_file(path: Path) -> str | None:
    """Best-effort file hash. None (never raises) if the file can't be
    read -- e.g. a model name Ultralytics resolves from its own cache
    rather than a literal path in this repo."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def get_git_commit(repo_root: Path) -> tuple[str | None, str]:
    """(commit_hash, status_reason). commit_hash is None whenever a
    commit can't be resolved -- explicitly, not silently, including the
    (expected, common) case of no .git directory at all; never raises."""
    if not (repo_root / ".git").exists():
        return None, "no .git repository present"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"git invocation failed: {exc}"
    if result.returncode != 0:
        return None, f"git rev-parse failed: {result.stderr.strip()}"
    return result.stdout.strip(), "resolved via git rev-parse HEAD"


def get_device_string(detector: ObjectDetector) -> str:
    try:
        return str(detector._model.device)
    except AttributeError:
        return "unknown"


def get_raw_class_name(detector: ObjectDetector, class_id: int) -> str:
    """The model's OWN label for this class id (Ultralytics' `model.
    names`), independent of Atlas's config naming -- see docs/
    DETECTOR_BENCHMARK.md "Class handling" for why this is a distinct
    column from atlas_config_class_name."""
    try:
        return str(detector._model.names[class_id])
    except (AttributeError, KeyError, IndexError):
        return "unknown"


def apply_device_override(detector: ObjectDetector, device: str) -> None:
    """Best-effort: moves the already-constructed model to `device`.
    Never raises -- some model/torch combinations don't support `.to()`
    uniformly, and this is documented as best-effort in the CLI help."""
    try:
        detector._model.to(device)
    except Exception:
        pass


# --- source classification and expansion ----------------------------------


def classify_source(path: Path) -> str:
    if not path.exists():
        raise BenchmarkSourceError(f"Source does not exist: {path}")
    if path.is_dir():
        return SOURCE_DIRECTORY
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return SOURCE_IMAGE
    if suffix in VIDEO_EXTENSIONS:
        return SOURCE_VIDEO
    if suffix in MANIFEST_EXTENSIONS:
        return SOURCE_MANIFEST
    raise BenchmarkSourceError(
        f"Unsupported file type {path.suffix!r} ({path}). Supported: "
        f"images {sorted(IMAGE_EXTENSIONS)}, videos {sorted(VIDEO_EXTENSIONS)}, "
        f"manifests {sorted(MANIFEST_EXTENSIONS)}."
    )


def expand_directory(directory: Path) -> list[SourceEntry]:
    entries: list[SourceEntry] = []
    for path in sorted(directory.iterdir()):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            entries.append(SourceEntry(id=path.stem, path=path, source_type=SOURCE_IMAGE))
        elif suffix in VIDEO_EXTENSIONS:
            entries.append(SourceEntry(id=path.stem, path=path, source_type=SOURCE_VIDEO))
    if not entries:
        raise BenchmarkSourceError(
            f"No supported image/video files found in directory: {directory}"
        )
    return entries


def load_manifest(path: Path) -> list[SourceEntry]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f) if path.suffix.lower() == ".json" else yaml.safe_load(f)

    if not isinstance(raw, list):
        raise BenchmarkManifestError(f"Manifest must be a list of entries: {path}")

    entries: list[SourceEntry] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise BenchmarkManifestError(f"Manifest entry {index} is not a mapping: {item!r}")
        missing = [name for name in REQUIRED_MANIFEST_FIELDS if name not in item]
        if missing:
            raise BenchmarkManifestError(
                f"Manifest entry {index} missing required field(s) {missing}: {item!r}"
            )
        entry_id = str(item["id"])
        if entry_id in seen_ids:
            raise BenchmarkManifestError(f"Duplicate manifest id {entry_id!r} in {path}")
        seen_ids.add(entry_id)

        source_type = str(item["source_type"]).strip().lower()
        if source_type not in ("image", "video"):
            raise BenchmarkManifestError(
                f"Manifest entry {entry_id!r} has invalid source_type "
                f"{item['source_type']!r} (must be 'image' or 'video')"
            )
        entry_path = Path(item["path"])
        if not entry_path.exists():
            raise BenchmarkManifestError(
                f"Manifest entry {entry_id!r} path does not exist: {entry_path}"
            )
        entries.append(
            SourceEntry(
                id=entry_id, path=entry_path,
                source_type=SOURCE_IMAGE if source_type == "image" else SOURCE_VIDEO,
            )
        )
    return entries


def resolve_entries(source_path: Path, source_kind: str) -> list[SourceEntry]:
    if source_kind == SOURCE_MANIFEST:
        return load_manifest(source_path)
    if source_kind == SOURCE_DIRECTORY:
        return expand_directory(source_path)
    return [
        SourceEntry(id=source_path.stem or source_path.name, path=source_path, source_type=source_kind)
    ]


# --- frame iteration --------------------------------------------------------


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    frame: "object | None"  # np.ndarray, kept loosely typed to avoid a hard numpy import here
    fps: float | None
    read_ok: bool


def iterate_frames(
    entry: SourceEntry, frame_step: int, max_frames: int | None
) -> Iterator[FrameRecord]:
    """Yields one FrameRecord per processed frame. frame_step/max_frames
    only affect VIDEO entries -- a single IMAGE entry always yields
    exactly one record (there is nothing to "step" over)."""
    if entry.source_type == SOURCE_IMAGE:
        frame = cv2.imread(str(entry.path))
        yield FrameRecord(frame_index=0, frame=frame, fps=None, read_ok=frame is not None)
        return

    video = VideoSource(str(entry.path))
    try:
        fps = video.get_fps() or None
        raw_index = 0
        emitted = 0
        while True:
            if max_frames is not None and emitted >= max_frames:
                break
            try:
                frame = video.read_frame()
            except VideoSourceError:
                # A live-vs-file distinction doesn't apply here -- this
                # tool never opens a camera (see run_benchmark), so any
                # read failure on a video FILE is treated as ordinary
                # end-of-stream, never counted as a "failed frame".
                break
            if raw_index % frame_step == 0:
                yield FrameRecord(frame_index=raw_index, frame=frame, fps=fps, read_ok=True)
                emitted += 1
            raw_index += 1
    finally:
        video.release()


# --- CSV row construction ---------------------------------------------------


def build_detection_row(
    *,
    source_id: str,
    frame_index: int,
    timestamp: float | None,
    obj,
    track_id: int | None,
    frame_width: int,
    frame_height: int,
    inference_time_ms: float,
    total_frame_time_ms: float,
    detections_in_frame: int,
    detector: ObjectDetector,
) -> dict:
    raw_name = get_raw_class_name(detector, obj.class_id)
    normalized = ATLAS_NORMALIZED_NAME_MAP.get(obj.class_name, obj.class_name)
    bbox = obj.bbox
    center_x, center_y = obj.center
    return {
        "source_id": source_id,
        "frame_index": frame_index,
        "timestamp_in_source_seconds": round_or_none(timestamp),
        "track_id": track_id if track_id is not None else "",
        "raw_class_id": obj.class_id,
        "raw_class_name": raw_name,
        "atlas_config_class_name": obj.class_name,
        "atlas_normalized_class_name": normalized,
        "micromobility_candidate": normalized in MICROMOBILITY_NORMALIZED_NAMES,
        "confidence": round_or_none(obj.confidence),
        "bbox_x1": bbox.x1, "bbox_y1": bbox.y1, "bbox_x2": bbox.x2, "bbox_y2": bbox.y2,
        "bbox_width": bbox.width, "bbox_height": bbox.height,
        "bbox_area": bbox.width * bbox.height,
        "center_x": center_x, "center_y": center_y,
        "normalized_center_x": round_or_none(center_x / frame_width) if frame_width else "",
        "normalized_center_y": round_or_none(center_y / frame_height) if frame_height else "",
        "frame_width": frame_width, "frame_height": frame_height,
        "inference_time_ms": round_or_none(inference_time_ms),
        "preprocess_time_ms": "",  # not exposed without forking ObjectDetector -- see docs
        "postprocess_time_ms": "",
        "total_frame_time_ms": round_or_none(total_frame_time_ms),
        "detections_in_frame": detections_in_frame,
    }


def build_empty_frame_row(
    *,
    source_id: str,
    frame_index: int,
    timestamp: float | None,
    frame_width: int,
    frame_height: int,
    inference_time_ms: float,
    total_frame_time_ms: float,
) -> dict:
    """A zero-detection frame still gets exactly one CSV row (frame
    -level columns only) so it isn't silently absent from later
    frame-level analysis. See docs/DETECTOR_BENCHMARK.md."""
    row = {column: "" for column in DETECTIONS_CSV_COLUMNS}
    row.update(
        {
            "source_id": source_id,
            "frame_index": frame_index,
            "timestamp_in_source_seconds": round_or_none(timestamp),
            "frame_width": frame_width,
            "frame_height": frame_height,
            "inference_time_ms": round_or_none(inference_time_ms),
            "total_frame_time_ms": round_or_none(total_frame_time_ms),
            "detections_in_frame": 0,
        }
    )
    return row


# --- temporal consistency proxies (conservative, detector-only) -----------


class TemporalConsistencyProxy:
    """Conservative, LOCAL, single-purpose adjacent-frame association for
    OBSERVATIONAL proxies only -- NOT identity tracking, NOT a
    replacement for src/object_tracker.py's ObjectTracker, and not used
    anywhere outside this benchmark tool. Multi-frame reappearance and
    gap-across-N-frames proxies are out of scope -- see
    docs/DETECTOR_BENCHMARK.md.
    """

    HIGH_CONFIDENCE_THRESHOLD = 0.75
    IOU_MATCH_THRESHOLD = 0.3

    def __init__(self) -> None:
        self._previous: list[dict] = []
        self.class_count_delta_events = 0
        self.high_confidence_disappearance_count = 0
        self.same_region_class_switch_count = 0
        self.confidence_variation_samples: list[float] = []

    def reset(self) -> None:
        """Call between unrelated sources (e.g. each directory/manifest
        entry) -- adjacent-frame proxies are meaningless across cuts."""
        self._previous = []

    def update(self, objects: list) -> None:
        current = [
            {
                "bbox": (obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2),
                "class_name": obj.class_name,
                "confidence": obj.confidence,
                "region": obj.region,
            }
            for obj in objects
        ]

        if len(current) != len(self._previous):
            self.class_count_delta_events += 1

        matched_previous_indices: set[int] = set()
        for cur in current:
            best_iou, best_index = 0.0, None
            for i, prev in enumerate(self._previous):
                if i in matched_previous_indices:
                    continue
                score = iou(cur["bbox"], prev["bbox"])
                if score > best_iou:
                    best_iou, best_index = score, i
            if best_index is not None and best_iou >= self.IOU_MATCH_THRESHOLD:
                matched_previous_indices.add(best_index)
                prev = self._previous[best_index]
                if prev["class_name"] != cur["class_name"] and prev["region"] == cur["region"]:
                    self.same_region_class_switch_count += 1
                self.confidence_variation_samples.append(
                    abs(cur["confidence"] - prev["confidence"])
                )

        for i, prev in enumerate(self._previous):
            if i not in matched_previous_indices and prev["confidence"] >= self.HIGH_CONFIDENCE_THRESHOLD:
                self.high_confidence_disappearance_count += 1

        self._previous = current

    def as_summary_dict(self) -> dict:
        return {
            "note": (
                "Conservative, detector-only proxies from simple adjacent"
                "-frame IoU association local to this benchmark tool -- "
                "NOT identity tracking, NOT ObjectTracker. Multi-frame "
                "reappearance/gap proxies are out of scope for this "
                "tool; see docs/DETECTOR_BENCHMARK.md."
            ),
            "class_count_delta_events_proxy": self.class_count_delta_events,
            "high_confidence_disappearance_count_proxy": self.high_confidence_disappearance_count,
            "same_region_class_switch_count_proxy": self.same_region_class_switch_count,
            "mean_confidence_variation_for_matched_boxes_proxy": round_or_none(
                mean_or_none(self.confidence_variation_samples)
            ),
        }


# --- run-level accumulation --------------------------------------------------


@dataclass
class RunAccumulator:
    total_frames: int = 0
    processed_frames: int = 0
    failed_frames: int = 0
    total_detections: int = 0
    detections_per_raw_class: dict = field(default_factory=dict)
    detections_per_normalized_class: dict = field(default_factory=dict)
    confidences_by_raw_class: dict = field(default_factory=dict)
    frame_keys_containing_class: dict = field(default_factory=dict)  # raw class -> set of (source_id, frame_index)
    inference_times_ms: list = field(default_factory=list)
    total_frame_times_ms: list = field(default_factory=list)
    detections_per_frame_counts: list = field(default_factory=list)
    source_fps: float | None = None
    first_frame_width: int | None = None
    first_frame_height: int | None = None

    def seed_classes(self, raw_classes: list[str], normalized_classes: list[str]) -> None:
        for name in raw_classes:
            self.detections_per_raw_class.setdefault(name, 0)
            self.confidences_by_raw_class.setdefault(name, [])
            self.frame_keys_containing_class.setdefault(name, set())
        for name in normalized_classes:
            self.detections_per_normalized_class.setdefault(name, 0)

    def ingest_frame(
        self, source_id: str, frame_index: int, objects: list,
        inference_time_ms: float, total_frame_time_ms: float,
        fps: float | None, frame_width: int, frame_height: int,
    ) -> None:
        self.processed_frames += 1
        self.inference_times_ms.append(inference_time_ms)
        self.total_frame_times_ms.append(total_frame_time_ms)
        self.detections_per_frame_counts.append(len(objects))
        self.total_detections += len(objects)
        if fps is not None and self.source_fps is None:
            self.source_fps = fps
        if self.first_frame_width is None:
            self.first_frame_width, self.first_frame_height = frame_width, frame_height

        classes_seen_this_frame: set[str] = set()
        for obj in objects:
            raw_name = obj.class_name
            normalized = ATLAS_NORMALIZED_NAME_MAP.get(raw_name, raw_name)
            self.detections_per_raw_class[raw_name] = self.detections_per_raw_class.get(raw_name, 0) + 1
            self.detections_per_normalized_class[normalized] = (
                self.detections_per_normalized_class.get(normalized, 0) + 1
            )
            self.confidences_by_raw_class.setdefault(raw_name, []).append(obj.confidence)
            classes_seen_this_frame.add(raw_name)

        for raw_name in classes_seen_this_frame:
            self.frame_keys_containing_class.setdefault(raw_name, set()).add((source_id, frame_index))


def build_summary(accumulator: RunAccumulator, proxy: TemporalConsistencyProxy) -> dict:
    frame_time_budget_ms = (
        1000.0 / accumulator.source_fps if accumulator.source_fps else DEFAULT_FRAME_TIME_BUDGET_MS
    )
    frames_over_budget = sum(
        1 for t in accumulator.total_frame_times_ms if t > frame_time_budget_ms
    )
    mean_total_frame_time_ms = mean_or_none(accumulator.total_frame_times_ms)
    effective_fps = (
        1000.0 / mean_total_frame_time_ms
        if mean_total_frame_time_ms and mean_total_frame_time_ms > 0
        else None
    )
    real_time_factor = (
        effective_fps / accumulator.source_fps
        if effective_fps is not None and accumulator.source_fps
        else None
    )

    all_summary_classes = sorted(accumulator.detections_per_raw_class)
    percent_frames_containing_class = {}
    for name in all_summary_classes:
        frame_count = len(accumulator.frame_keys_containing_class.get(name, set()))
        percent_frames_containing_class[name] = round_or_none(
            100.0 * frame_count / accumulator.processed_frames
            if accumulator.processed_frames > 0
            else 0.0
        )

    mean_confidence_by_raw_class = {}
    median_confidence_by_raw_class = {}
    confidence_quantiles_by_raw_class = {}
    for name, values in accumulator.confidences_by_raw_class.items():
        mean_confidence_by_raw_class[name] = round_or_none(mean_or_none(values))
        median_confidence_by_raw_class[name] = round_or_none(median_or_none(values))
        confidence_quantiles_by_raw_class[name] = {
            "p25": round_or_none(percentile_or_none(values, 25)),
            "p50": round_or_none(percentile_or_none(values, 50)),
            "p75": round_or_none(percentile_or_none(values, 75)),
            "p90": round_or_none(percentile_or_none(values, 90)),
        }

    return {
        "observational_baseline": True,
        "requires_ground_truth_for": list(REQUIRES_GROUND_TRUTH_FOR),
        "protected_classes": list(PROTECTED_CLASSES),
        "priority_classes": list(PRIORITY_CLASSES),
        "total_frames": accumulator.total_frames,
        "processed_frames": accumulator.processed_frames,
        "failed_frames": accumulator.failed_frames,
        "total_detections": accumulator.total_detections,
        "detections_per_raw_class": dict(sorted(accumulator.detections_per_raw_class.items())),
        "detections_per_normalized_class": dict(
            sorted(accumulator.detections_per_normalized_class.items())
        ),
        "mean_confidence_by_raw_class": mean_confidence_by_raw_class,
        "median_confidence_by_raw_class": median_confidence_by_raw_class,
        "confidence_quantiles_by_raw_class": confidence_quantiles_by_raw_class,
        "average_detections_per_frame": round_or_none(
            mean_or_none([float(c) for c in accumulator.detections_per_frame_counts])
        ),
        "percent_frames_containing_class": percent_frames_containing_class,
        "mean_inference_time_ms": round_or_none(mean_or_none(accumulator.inference_times_ms)),
        "median_inference_time_ms": round_or_none(median_or_none(accumulator.inference_times_ms)),
        "p95_inference_time_ms": round_or_none(
            percentile_or_none(accumulator.inference_times_ms, 95)
        ),
        "mean_total_frame_time_ms": round_or_none(mean_total_frame_time_ms),
        "effective_processing_fps": round_or_none(effective_fps),
        "source_fps": round_or_none(accumulator.source_fps),
        "real_time_factor": round_or_none(real_time_factor),
        "frame_time_budget_ms": round_or_none(frame_time_budget_ms),
        "frames_exceeding_frame_time_budget": frames_over_budget,
        "temporal_consistency_proxies": proxy.as_summary_dict(),
    }


def build_run_metadata(
    *, args: argparse.Namespace, source_path: Path, source_kind: str, entry_count: int,
    resolved_model_name: str, resolved_confidence: float, detector: ObjectDetector,
    accumulator: RunAccumulator,
) -> dict:
    model_hash = None
    for candidate in (Path(resolved_model_name), _REPO_ROOT / resolved_model_name):
        if candidate.exists():
            model_hash = sha256_file(candidate)
            break
    commit, commit_status = get_git_commit(_REPO_ROOT)

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_tool_version": BENCHMARK_TOOL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "source_type": source_kind,
        "source_entry_count": entry_count,
        "model_name": resolved_model_name,
        "model_file_sha256": model_hash,
        "confidence_threshold": resolved_confidence,
        "iou_threshold": None,
        "input_resolution": (
            {"width": accumulator.first_frame_width, "height": accumulator.first_frame_height}
            if accumulator.first_frame_width is not None
            else None
        ),
        "device": get_device_string(detector),
        "ultralytics_version": ultralytics.__version__,
        "opencv_version": cv2.__version__,
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "frame_step": args.frame_step,
        "max_frames": args.max_frames,
        "total_frames_processed": accumulator.processed_frames,
        "command_line_arguments": sys.argv[1:],
        "git_commit": commit,
        "git_commit_status": commit_status,
    }


# --- detector construction / annotation / output-dir resolution -----------


def build_detector_for_benchmark(
    config: dict, args: argparse.Namespace, region_analyzer
) -> tuple[ObjectDetector, str, float]:
    """Reuses main.build_object_detector for production-parity
    construction. --model/--confidence override a DEEP COPY of the
    loaded config, exactly matching scripts/benchmark_detectors.py's
    existing pattern -- the on-disk config file is never touched."""
    model_config = copy.deepcopy(config)
    if args.model is not None:
        model_config["model"]["name"] = args.model
    if args.confidence is not None:
        model_config["model"]["confidence_threshold"] = args.confidence

    detector = atlas_main.build_object_detector(model_config, region_analyzer)
    return detector, model_config["model"]["name"], model_config["model"]["confidence_threshold"]


def write_annotated_frame(
    annotated_dir: Path, source_id: str, frame_index: int, frame, objects: list
) -> None:
    annotated_dir.mkdir(parents=True, exist_ok=True)
    canvas = frame.copy()
    for obj in objects:
        bbox = obj.bbox
        cv2.rectangle(canvas, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 200, 0), 2)
        label = f"{obj.class_name} {obj.confidence:.2f}"
        cv2.putText(
            canvas, label, (bbox.x1, max(bbox.y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1,
        )
    filename = f"{source_id}_frame_{frame_index:06d}.jpg"
    cv2.imwrite(str(annotated_dir / filename), canvas)


def resolve_output_dir(source_path: Path, output_dir_arg: str | None, overwrite: bool) -> Path:
    if output_dir_arg is not None:
        output_dir = Path(output_dir_arg)
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        stem = source_path.stem or source_path.name
        output_dir = Path("logs") / "detector_benchmarks" / f"{stem}_{stamp}"

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise BenchmarkOutputExistsError(
            f"Output directory already has content: {output_dir}. "
            "Pass --overwrite to reuse it, or choose a different --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# --- orchestration -----------------------------------------------------------


def run_benchmark(args: argparse.Namespace) -> Path:
    """Runs the full benchmark and returns the output directory. Raises
    BenchmarkSourceError/BenchmarkManifestError/BenchmarkOutputExistsError
    with a clear message on any invalid input -- never a bare traceback
    for an expected failure mode."""
    if str(args.source).isdigit():
        raise BenchmarkSourceError(
            "Live-camera sources are not supported by the benchmark tool "
            f"(got numeric source {args.source!r}); provide a video file, "
            "image, directory, or manifest path."
        )

    source_path = Path(args.source)
    source_kind = classify_source(source_path)
    entries = resolve_entries(source_path, source_kind)

    config = atlas_main.load_config(args.config)
    region_analyzer = atlas_main.build_region_analyzer(config)
    detector, resolved_model_name, resolved_confidence = build_detector_for_benchmark(
        config, args, region_analyzer
    )
    if args.device is not None:
        apply_device_override(detector, args.device)

    tracker: ObjectTracker | None = atlas_main.build_object_tracker(config) if args.track else None

    output_dir = resolve_output_dir(source_path, args.output_dir, args.overwrite)
    annotated_dir = output_dir / "annotated"

    accumulator = RunAccumulator()
    accumulator.seed_classes(
        raw_classes=sorted(set(config["classes"].values())),
        normalized_classes=sorted(set(ATLAS_NORMALIZED_NAME_MAP.values())),
    )
    proxy = TemporalConsistencyProxy()

    detections_csv_path = output_dir / "detections.csv"
    with detections_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETECTIONS_CSV_COLUMNS)
        writer.writeheader()

        for entry in entries:
            proxy.reset()  # adjacent-frame proxies are meaningless across a source cut
            for record in iterate_frames(entry, args.frame_step, args.max_frames):
                accumulator.total_frames += 1
                if not record.read_ok:
                    accumulator.failed_frames += 1
                    continue

                frame_start = time.perf_counter()
                inference_start = time.perf_counter()
                detections = detector.detect(record.frame)
                inference_time_ms = (time.perf_counter() - inference_start) * 1000.0

                objects_for_row = tracker.update(detections) if tracker is not None else detections
                proxy.update(objects_for_row)

                total_frame_time_ms = (time.perf_counter() - frame_start) * 1000.0
                frame_height, frame_width = record.frame.shape[:2]
                timestamp = record.frame_index / record.fps if record.fps else None

                if objects_for_row:
                    for obj in objects_for_row:
                        track_id = getattr(obj, "track_id", None)
                        writer.writerow(
                            build_detection_row(
                                source_id=entry.id, frame_index=record.frame_index,
                                timestamp=timestamp, obj=obj, track_id=track_id,
                                frame_width=frame_width, frame_height=frame_height,
                                inference_time_ms=inference_time_ms,
                                total_frame_time_ms=total_frame_time_ms,
                                detections_in_frame=len(objects_for_row), detector=detector,
                            )
                        )
                else:
                    writer.writerow(
                        build_empty_frame_row(
                            source_id=entry.id, frame_index=record.frame_index,
                            timestamp=timestamp, frame_width=frame_width,
                            frame_height=frame_height, inference_time_ms=inference_time_ms,
                            total_frame_time_ms=total_frame_time_ms,
                        )
                    )

                accumulator.ingest_frame(
                    entry.id, record.frame_index, objects_for_row, inference_time_ms,
                    total_frame_time_ms, record.fps, frame_width, frame_height,
                )

                if args.annotate:
                    write_annotated_frame(
                        annotated_dir, entry.id, record.frame_index, record.frame, objects_for_row
                    )

    run_metadata = build_run_metadata(
        args=args, source_path=source_path, source_kind=source_kind, entry_count=len(entries),
        resolved_model_name=resolved_model_name, resolved_confidence=resolved_confidence,
        detector=detector, accumulator=accumulator,
    )
    summary = build_summary(accumulator, proxy)

    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return output_dir


# --- CLI ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Observational baseline benchmark for Atlas's detector (today: "
            "YOLOv8s, reused via main.py's own factories/thresholds). "
            "Reports detection frequency, confidence, and runtime behavior "
            "only -- never precision/recall/F1/mAP, which require labeled "
            "ground truth this tool does not have. See "
            "docs/DETECTOR_BENCHMARK.md."
        )
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to an image, video file, directory, or manifest (.yaml/.json).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Result directory (default: logs/detector_benchmarks/<source-stem>_<timestamp>).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override config/settings.yaml's model name/path for this run only.",
    )
    parser.add_argument(
        "--confidence", type=float, default=None,
        help="Override config/settings.yaml's confidence threshold for this run only.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Best-effort device override (e.g. 'cpu', 'cuda:0') passed to the model.",
    )
    parser.add_argument(
        "--frame-step", type=int, default=1,
        help="Process every Nth video frame (default: 1, every frame). Ignored for images.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Cap on processed frames per video entry (default: entire source).",
    )
    parser.add_argument(
        "--annotate", action="store_true",
        help="Write annotated per-frame JPEGs to <output-dir>/annotated/ (off by default).",
    )
    parser.add_argument(
        "--track", action="store_true",
        help="Run the real ObjectTracker alongside detection to populate track_id (off by default).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow reusing a non-empty --output-dir (off by default).",
    )
    parser.add_argument(
        "--config", default=atlas_main.DEFAULT_CONFIG_PATH,
        help=f"Path to the YAML config file (default: {atlas_main.DEFAULT_CONFIG_PATH}).",
    )
    args = parser.parse_args(argv)

    if args.frame_step < 1:
        parser.error("--frame-step must be >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be >= 1")
    if args.confidence is not None and not (0.0 <= args.confidence <= 1.0):
        parser.error("--confidence must be within [0, 1]")

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = run_benchmark(args)
    print(f"Benchmark complete. Results written to: {output_dir}")


if __name__ == "__main__":
    main()
