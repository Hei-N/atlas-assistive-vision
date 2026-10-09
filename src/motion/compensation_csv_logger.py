"""Structured CSV logging for the --validate-compensation developer mode
(Atlas Phase 4).

Two loggers live here:
- CompensationCsvLogger: one AGGREGATE row every few frames, matching the
  on-screen COMPENSATION VALIDATION panel (src/motion/
  compensation_validator.py) -- a quick trend overview.
- PerObjectCsvLogger: one row per TRACKED OBJECT per FRAME (unsampled),
  for exporting complete diagnostics from a controlled prerecorded clip
  (--export-per-object-csv) -- fine enough granularity to actually check
  compensation behavior object-by-object, frame-by-frame.

Both are only active when --validate-compensation is passed, and neither
touches TrajectoryPredictor, PathIntersectionAnalyzer, position_history,
or any runtime/hazard decision -- they only read already-computed
MotionEstimate/ValidationStats/PerObjectValidation values and write them
to disk.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import TextIO

from src.models import MotionEstimate, PerObjectValidation, ValidationStats

CSV_COLUMNS = [
    "timestamp",
    "frame_number",
    "camera_dx",
    "camera_dy",
    "camera_motion_magnitude",
    "camera_confidence",
    "feature_count",
    "inlier_count",
    "track_count",
    "raw_motion_average",
    "compensated_motion_average",
    "reduction_percent",
    "stationary_count",
    "moving_count",
    "camera_dominated_count",
    "uncertain_count",
]


def build_csv_filename(now: datetime) -> str:
    """The timestamped filename for a new compensation-validation CSV,
    e.g. "compensation_validation_2026-07-27_205600.csv". Takes `now`
    explicitly (rather than calling datetime.now() internally) so this
    stays a pure, deterministic function for testing."""
    return f"compensation_validation_{now:%Y-%m-%d_%H%M%S}.csv"


class CompensationCsvLogger:
    """Appends one CSV row per log_row() call, flushing after every write
    so data already logged survives an unexpected exit (Ctrl+C, crash).

    Args:
        path: Destination CSV path. Parent directories are created if
            they don't already exist.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_COLUMNS)
        self._file.flush()

    def log_row(
        self,
        timestamp: str,
        frame_number: int,
        camera_motion: MotionEstimate | None,
        stats: ValidationStats,
    ) -> None:
        """Write one row from an already-computed MotionEstimate/
        ValidationStats pair -- the exact same values the on-screen
        COMPENSATION VALIDATION panel is drawn from, so the CSV and the
        panel never disagree.

        Args:
            timestamp: Caller-supplied timestamp string (e.g.
                datetime.now().isoformat()) for this row -- kept as an
                explicit parameter, not computed here, so this method
                stays deterministic/testable.
            frame_number: The run's current frame counter.
            camera_motion: This frame's MotionEstimate, or None if
                unavailable -- feature_count/inlier_count are 0 in that
                case (camera_dx/dy/confidence instead come from `stats`,
                which already encodes the same 0.0 fallback).
            stats: This frame's ValidationStats (aggregate).
        """
        feature_count = (
            camera_motion.tracked_feature_count if camera_motion is not None else 0
        )
        inlier_count = camera_motion.inlier_count if camera_motion is not None else 0
        camera_motion_magnitude = (stats.camera_dx**2 + stats.camera_dy**2) ** 0.5

        self._writer.writerow(
            [
                timestamp,
                frame_number,
                f"{stats.camera_dx:.4f}",
                f"{stats.camera_dy:.4f}",
                f"{camera_motion_magnitude:.4f}",
                f"{stats.camera_confidence:.4f}",
                feature_count,
                inlier_count,
                stats.active_track_count,
                f"{stats.average_raw_speed:.4f}",
                f"{stats.average_compensated_speed:.4f}",
                f"{stats.reduction_percent:.4f}",
                stats.stationary_count,
                stats.moving_count,
                stats.camera_dominated_count,
                stats.uncertain_count,
            ]
        )
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file. Safe to call once at shutdown."""
        self._file.close()


PER_OBJECT_CSV_COLUMNS = [
    "frame_index",
    "timestamp",
    "camera_dx",
    "camera_dy",
    "rotation_deg",
    "scale",
    "feature_count",
    "match_count",
    "inlier_count",
    "inlier_ratio",
    "compensation_confidence",
    "compensation_status",
    "track_id",
    "class_name",
    "raw_vx",
    "raw_vy",
    "raw_speed",
    "compensated_vx",
    "compensated_vy",
    "compensated_speed",
    "motion_state",
    "resolved_vx",
    "resolved_vy",
    "resolved_speed",
    "motion_source",
    "prediction_uncertain",
    "raw_intersects_corridor",
    "resolved_intersects_corridor",
    "filtered_vx",
    "filtered_vy",
    "filtered_speed",
    "motion_filter_state",
    "motion_filter_reason",
    "motion_confirmation_frames",
]


class PerObjectCsvLogger:
    """Exports one row per TRACKED OBJECT per FRAME (unsampled) -- for a
    controlled prerecorded-clip validation run where complete, granular
    data matters more than a quick trend overview (that's what
    CompensationCsvLogger above is for). Flushes after every write, same
    durability guarantee as CompensationCsvLogger.

    Args:
        path: Destination CSV path. Parent directories are created if
            they don't already exist.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(PER_OBJECT_CSV_COLUMNS)
        self._file.flush()

    def log_rows(
        self,
        frame_index: int,
        timestamp: str,
        camera_motion: MotionEstimate | None,
        per_object: list[PerObjectValidation],
    ) -> None:
        """Write one row per entry in `per_object`. Writes nothing when
        `per_object` is empty (no tracked objects this frame) -- never a
        crash, and no misleading camera-only row with blank object
        fields.

        Args:
            frame_index: The run's current frame counter.
            timestamp: Caller-supplied timestamp string (e.g.
                datetime.now().isoformat()) -- kept explicit, not computed
                here, so this method stays deterministic/testable.
            camera_motion: This frame's MotionEstimate, or None if
                unavailable -- every camera-level column falls back to
                0/"UNAVAILABLE" in that case.
            per_object: This frame's PerObjectValidation rows (see
                src/motion/compensation_validator.py).
        """
        if not per_object:
            return

        camera_dx = camera_motion.dx if camera_motion is not None else 0.0
        camera_dy = camera_motion.dy if camera_motion is not None else 0.0
        rotation_deg = camera_motion.rotation_degrees if camera_motion is not None else 0.0
        scale = camera_motion.scale if camera_motion is not None else 1.0
        feature_count = camera_motion.feature_count if camera_motion is not None else 0
        match_count = (
            camera_motion.tracked_feature_count if camera_motion is not None else 0
        )
        inlier_count = camera_motion.inlier_count if camera_motion is not None else 0
        inlier_ratio = camera_motion.inlier_ratio if camera_motion is not None else 0.0
        compensation_confidence = (
            camera_motion.confidence if camera_motion is not None else 0.0
        )
        compensation_status = (
            camera_motion.status if camera_motion is not None else "UNAVAILABLE"
        )

        for row in per_object:
            self._writer.writerow(
                [
                    frame_index,
                    timestamp,
                    f"{camera_dx:.4f}",
                    f"{camera_dy:.4f}",
                    f"{rotation_deg:.4f}",
                    f"{scale:.4f}",
                    feature_count,
                    match_count,
                    inlier_count,
                    f"{inlier_ratio:.4f}",
                    f"{compensation_confidence:.4f}",
                    compensation_status,
                    row.track_id,
                    row.class_name,
                    f"{row.raw_velocity_x:.4f}",
                    f"{row.raw_velocity_y:.4f}",
                    f"{row.raw_speed:.4f}",
                    f"{row.compensated_velocity_x:.4f}",
                    f"{row.compensated_velocity_y:.4f}",
                    f"{row.compensated_speed:.4f}",
                    row.status,
                    f"{row.resolved_velocity_x:.4f}",
                    f"{row.resolved_velocity_y:.4f}",
                    f"{row.resolved_speed:.4f}",
                    row.motion_source,
                    row.prediction_uncertain,
                    row.raw_intersects_corridor,
                    row.resolved_intersects_corridor,
                    f"{row.filtered_velocity_x:.4f}",
                    f"{row.filtered_velocity_y:.4f}",
                    f"{row.filtered_speed:.4f}",
                    row.motion_filter_state,
                    row.motion_filter_reason or "",
                    row.motion_confirmation_frames,
                ]
            )
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file. Safe to call once at shutdown."""
        self._file.close()
