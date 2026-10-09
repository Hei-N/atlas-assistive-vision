"""Extracts a small, hand-selected set of representative frames from a
source video into an Atlas evaluation-set directory (`frames/` +
`frame_manifest.csv` + a starter `annotations.json`), ready for human
labeling via `scripts/evaluate_detector.py`.

This is deliberately narrow: it does not choose frames automatically
from pixel content (occlusion/distance/etc. are visually judged by a
human before calling this), does not run the detector, and does not
touch any production code. See docs/DETECTOR_EVALUATION_DATASET.md and
docs/DETECTOR_ANNOTATIONS.md for the schema this feeds into.

Frames are saved at the source video's native resolution -- never
resized -- with deterministic filenames that encode both the exact
frame index and the timestamp, so frame identity is always recoverable
without re-reading the video.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: E402

from scripts.annotation_schema import CANONICAL_CLASSES, SCHEMA_VERSION  # noqa: E402

FRAME_MANIFEST_COLUMNS = [
    "source_id", "source_path", "frame_index", "timestamp_seconds",
    "image_filename", "width", "height", "selected_reason",
]


@dataclass(frozen=True)
class FrameSelection:
    frame_index: int
    selected_reason: str


def deterministic_filename(source_id: str, frame_index: int, timestamp_seconds: float) -> str:
    """e.g. "test7_frame_001408_t023.500.jpg" -- sortable, collision
    -free (frame_index is unique per source), and self-describing."""
    return f"{source_id}_frame_{frame_index:06d}_t{timestamp_seconds:07.3f}.jpg"


def select_interval_frames(start_s: float, end_s: float, step_s: float, fps: float) -> list[int]:
    """Evenly spaced frame indices covering [start_s, end_s] at
    roughly `step_s` seconds apart, rounded to real frame indices."""
    indices = []
    t = start_s
    while t <= end_s + 1e-9:
        indices.append(round(t * fps))
        t += step_s
    return indices


def extract_frames(
    video_path: Path, source_id: str, output_frames_dir: Path, selections: list[FrameSelection],
) -> tuple[list[dict], dict]:
    """Extracts every selected frame (native resolution, never resized)
    and returns (manifest_rows, video_info). Raises FileNotFoundError if
    video_path doesn't exist, and ValueError if a requested frame index
    can't be read."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    output_frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        rows = []
        seen_indices: set[int] = set()
        for selection in sorted(selections, key=lambda s: s.frame_index):
            if selection.frame_index in seen_indices:
                raise ValueError(f"Duplicate frame_index in selections: {selection.frame_index}")
            seen_indices.add(selection.frame_index)

            capture.set(cv2.CAP_PROP_POS_FRAMES, selection.frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"Could not read frame_index {selection.frame_index} from {video_path}")

            timestamp_seconds = selection.frame_index / fps
            filename = deterministic_filename(source_id, selection.frame_index, timestamp_seconds)
            cv2.imwrite(str(output_frames_dir / filename), frame)  # native resolution, no resize

            rows.append(
                {
                    "source_id": source_id,
                    "source_path": str(video_path),
                    "frame_index": selection.frame_index,
                    "timestamp_seconds": round(timestamp_seconds, 3),
                    "image_filename": filename,
                    "width": width,
                    "height": height,
                    "selected_reason": selection.selected_reason,
                }
            )
    finally:
        capture.release()

    video_info = {"fps": fps, "width": width, "height": height, "total_frames": total_frames}
    return rows, video_info


def write_frame_manifest(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_starter_annotations(
    *, source_id: str, source_path: str, video_info: dict, dataset_id: str,
    scene_tags: list[str], lighting: str, camera_motion: str, notes: str, path: Path,
) -> None:
    """Empty `annotations` array and no `reviewed_frames` entries -- every
    selected frame is UNREVIEWED until a human actually labels it (see
    docs/DETECTOR_ANNOTATIONS.md section 6). Never fabricates a box, a
    label, or a reviewed-empty marker."""
    document = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "class_names": list(CANONICAL_CLASSES),
        "sources": [
            {
                "source_id": source_id,
                "path": source_path,
                "source_type": "video",
                "width": video_info["width"],
                "height": video_info["height"],
                "total_frames": video_info["total_frames"],
                "fps": video_info["fps"],
                "scene_tags": scene_tags,
                "lighting": lighting,
                "camera_motion": camera_motion,
                "notes": notes,
            }
        ],
        "annotations": [],
    }
    path.write_text(json.dumps(document, indent=2))
