"""Unit tests for src/motion/compensation_csv_logger.py (structured CSV
logging for the --validate-compensation developer mode). No camera,
model, or network required -- everything here is synthetic data written
to a pytest tmp_path.
"""

import csv
from datetime import datetime

from src.models import MotionEstimate, PerObjectValidation, ValidationStats
from src.motion.compensation_csv_logger import (
    CSV_COLUMNS,
    PER_OBJECT_CSV_COLUMNS,
    CompensationCsvLogger,
    PerObjectCsvLogger,
    build_csv_filename,
)


def make_motion_estimate(
    dx: float = 1.5, dy: float = -0.5, confidence: float = 0.8
) -> MotionEstimate:
    tracked_feature_count = 250
    inlier_count = 200
    return MotionEstimate(
        valid=True,
        dx=dx,
        dy=dy,
        rotation_degrees=0.0,
        scale=1.0,
        confidence=confidence,
        feature_count=300,
        tracked_feature_count=tracked_feature_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_count / tracked_feature_count,
        status="VALID" if confidence >= 0.6 else "LOW_CONFIDENCE",
        reason_invalid=None,
        raw_dx=dx,
        raw_dy=dy,
        raw_rotation_degrees=0.0,
        transform_matrix=None,
        debug_prev_points=(),
        debug_current_points=(),
        debug_inlier_flags=(),
    )


def make_validation_stats(
    camera_dx: float = 1.5,
    camera_dy: float = -0.5,
    camera_confidence: float = 0.8,
) -> ValidationStats:
    return ValidationStats(
        active_track_count=2,
        camera_dx=camera_dx,
        camera_dy=camera_dy,
        camera_confidence=camera_confidence,
        camera_status="VALID",
        feature_count=300,
        match_count=250,
        inlier_count=200,
        stationary_count=1,
        moving_count=1,
        camera_dominated_count=0,
        uncertain_count=0,
        average_raw_speed=10.0,
        average_compensated_speed=2.0,
        reduction_percent=80.0,
    )


def test_build_csv_filename_matches_timestamped_format() -> None:
    now = datetime(2026, 7, 27, 20, 56, 0)
    assert build_csv_filename(now) == "compensation_validation_2026-07-27_205600.csv"


def test_csv_file_is_created_with_header(tmp_path) -> None:
    csv_path = tmp_path / "out.csv"
    logger_instance = CompensationCsvLogger(csv_path)
    logger_instance.close()

    assert csv_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == CSV_COLUMNS


def test_header_contains_every_required_column() -> None:
    required = {
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
    }
    assert required.issubset(set(CSV_COLUMNS))


def test_logged_row_matches_validation_statistics(tmp_path) -> None:
    csv_path = tmp_path / "out.csv"
    logger_instance = CompensationCsvLogger(csv_path)

    camera_motion = make_motion_estimate(dx=3.0, dy=4.0, confidence=0.9)
    stats = make_validation_stats(camera_dx=3.0, camera_dy=4.0, camera_confidence=0.9)

    logger_instance.log_row(
        timestamp="2026-07-27T20:56:00",
        frame_number=120,
        camera_motion=camera_motion,
        stats=stats,
    )
    logger_instance.close()

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["timestamp"] == "2026-07-27T20:56:00"
    assert row["frame_number"] == "120"
    assert float(row["camera_dx"]) == 3.0
    assert float(row["camera_dy"]) == 4.0
    # magnitude of (3, 4) is exactly 5.
    assert float(row["camera_motion_magnitude"]) == 5.0
    assert float(row["camera_confidence"]) == 0.9
    assert row["feature_count"] == "250"
    assert row["inlier_count"] == "200"
    assert row["track_count"] == "2"
    assert float(row["raw_motion_average"]) == 10.0
    assert float(row["compensated_motion_average"]) == 2.0
    assert float(row["reduction_percent"]) == 80.0
    assert row["stationary_count"] == "1"
    assert row["moving_count"] == "1"
    assert row["camera_dominated_count"] == "0"
    assert row["uncertain_count"] == "0"


def test_logged_row_handles_missing_camera_motion(tmp_path) -> None:
    csv_path = tmp_path / "out.csv"
    logger_instance = CompensationCsvLogger(csv_path)

    stats = make_validation_stats(camera_dx=0.0, camera_dy=0.0, camera_confidence=0.0)
    logger_instance.log_row(
        timestamp="2026-07-27T20:56:00",
        frame_number=5,
        camera_motion=None,
        stats=stats,
    )
    logger_instance.close()

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["feature_count"] == "0"
    assert rows[0]["inlier_count"] == "0"


def test_rows_are_flushed_immediately_without_closing(tmp_path) -> None:
    csv_path = tmp_path / "out.csv"
    logger_instance = CompensationCsvLogger(csv_path)

    logger_instance.log_row(
        timestamp="2026-07-27T20:56:00",
        frame_number=1,
        camera_motion=None,
        stats=make_validation_stats(),
    )

    # Read the file from a SEPARATE handle while logger_instance's file is
    # still open -- proves log_row() flushed rather than buffering, so
    # data survives an unexpected exit.
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2  # header + one data row

    logger_instance.close()


def test_creates_parent_directories(tmp_path) -> None:
    nested_path = tmp_path / "nested" / "dirs" / "out.csv"
    logger_instance = CompensationCsvLogger(nested_path)
    logger_instance.close()

    assert nested_path.exists()


# --- PerObjectCsvLogger (--export-per-object-csv) ------------------------


def make_per_object_row(
    track_id: int = 1,
    class_name: str = "person",
    raw_velocity_x: float = 12.0,
    raw_velocity_y: float = 0.0,
    compensated_velocity_x: float = 2.0,
    compensated_velocity_y: float = 0.0,
    status: str = "CAMERA-DOMINATED",
    resolved_velocity_x: float = 2.0,
    resolved_velocity_y: float = 0.0,
    motion_source: str = "COMPENSATED",
    prediction_uncertain: bool = False,
    raw_intersects_corridor: bool = False,
    resolved_intersects_corridor: bool = False,
    filtered_velocity_x: float = 2.0,
    filtered_velocity_y: float = 0.0,
    motion_filter_state: str = "MOVING",
    motion_filter_reason: str | None = None,
    motion_confirmation_frames: int = 0,
) -> PerObjectValidation:
    raw_speed = (raw_velocity_x**2 + raw_velocity_y**2) ** 0.5
    compensated_speed = (compensated_velocity_x**2 + compensated_velocity_y**2) ** 0.5
    resolved_speed = (resolved_velocity_x**2 + resolved_velocity_y**2) ** 0.5
    filtered_speed = (filtered_velocity_x**2 + filtered_velocity_y**2) ** 0.5
    return PerObjectValidation(
        track_id=track_id,
        class_name=class_name,
        raw_velocity_x=raw_velocity_x,
        raw_velocity_y=raw_velocity_y,
        compensated_velocity_x=compensated_velocity_x,
        compensated_velocity_y=compensated_velocity_y,
        raw_speed=raw_speed,
        compensated_speed=compensated_speed,
        compensated_direction="right",
        camera_confidence=0.9,
        status=status,
        resolved_velocity_x=resolved_velocity_x,
        resolved_velocity_y=resolved_velocity_y,
        resolved_speed=resolved_speed,
        motion_source=motion_source,
        prediction_uncertain=prediction_uncertain,
        raw_intersects_corridor=raw_intersects_corridor,
        resolved_intersects_corridor=resolved_intersects_corridor,
        filtered_velocity_x=filtered_velocity_x,
        filtered_velocity_y=filtered_velocity_y,
        filtered_speed=filtered_speed,
        motion_filter_state=motion_filter_state,
        motion_filter_reason=motion_filter_reason,
        motion_confirmation_frames=motion_confirmation_frames,
    )


def test_per_object_csv_header_contains_every_required_column() -> None:
    required = {
        "frame_index", "timestamp", "camera_dx", "camera_dy", "rotation_deg",
        "scale", "feature_count", "match_count", "inlier_count", "inlier_ratio",
        "compensation_confidence", "compensation_status", "track_id",
        "class_name", "raw_vx", "raw_vy", "raw_speed", "compensated_vx",
        "compensated_vy", "compensated_speed", "motion_state",
        "resolved_vx", "resolved_vy", "resolved_speed", "motion_source",
        "prediction_uncertain", "raw_intersects_corridor",
        "resolved_intersects_corridor", "filtered_vx", "filtered_vy",
        "filtered_speed", "motion_filter_state", "motion_filter_reason",
        "motion_confirmation_frames",
    }
    assert required.issubset(set(PER_OBJECT_CSV_COLUMNS))


def test_per_object_csv_file_created_with_header(tmp_path) -> None:
    csv_path = tmp_path / "per_object.csv"
    logger_instance = PerObjectCsvLogger(csv_path)
    logger_instance.close()

    assert csv_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == PER_OBJECT_CSV_COLUMNS


def test_per_object_csv_writes_one_row_per_object_per_frame(tmp_path) -> None:
    csv_path = tmp_path / "per_object.csv"
    logger_instance = PerObjectCsvLogger(csv_path)

    camera_motion = make_motion_estimate(dx=2.0, dy=0.0, confidence=0.95)
    rows = [
        make_per_object_row(
            track_id=1, raw_velocity_x=12.0, compensated_velocity_x=2.0,
            resolved_velocity_x=2.0, motion_source="COMPENSATED",
            prediction_uncertain=False, raw_intersects_corridor=True,
            resolved_intersects_corridor=False, filtered_velocity_x=0.0,
            motion_filter_state="STATIONARY", motion_filter_reason=None,
            motion_confirmation_frames=3,
        ),
        make_per_object_row(track_id=2, raw_velocity_x=0.5, compensated_velocity_x=0.5),
    ]

    logger_instance.log_rows(
        frame_index=42,
        timestamp="2026-07-27T20:56:00",
        camera_motion=camera_motion,
        per_object=rows,
    )
    logger_instance.close()

    with csv_path.open(newline="", encoding="utf-8") as f:
        written_rows = list(csv.DictReader(f))

    assert len(written_rows) == 2
    assert written_rows[0]["track_id"] == "1"
    assert written_rows[0]["frame_index"] == "42"
    assert written_rows[0]["timestamp"] == "2026-07-27T20:56:00"
    assert float(written_rows[0]["camera_dx"]) == 2.0
    assert written_rows[0]["compensation_status"] == "VALID"
    assert float(written_rows[0]["raw_vx"]) == 12.0
    assert float(written_rows[0]["compensated_vx"]) == 2.0
    assert written_rows[0]["motion_state"] == "CAMERA-DOMINATED"
    assert float(written_rows[0]["resolved_vx"]) == 2.0
    assert written_rows[0]["motion_source"] == "COMPENSATED"
    assert written_rows[0]["prediction_uncertain"] == "False"
    assert written_rows[0]["raw_intersects_corridor"] == "True"
    assert written_rows[0]["resolved_intersects_corridor"] == "False"
    assert float(written_rows[0]["filtered_vx"]) == 0.0
    assert written_rows[0]["motion_filter_state"] == "STATIONARY"
    assert written_rows[0]["motion_filter_reason"] == ""
    assert written_rows[0]["motion_confirmation_frames"] == "3"
    assert written_rows[1]["track_id"] == "2"


def test_per_object_csv_writes_nothing_with_no_tracked_objects(tmp_path) -> None:
    csv_path = tmp_path / "per_object.csv"
    logger_instance = PerObjectCsvLogger(csv_path)

    logger_instance.log_rows(
        frame_index=1,
        timestamp="2026-07-27T20:56:00",
        camera_motion=make_motion_estimate(),
        per_object=[],
    )
    logger_instance.close()

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1  # header only, no data rows


def test_per_object_csv_handles_missing_camera_motion(tmp_path) -> None:
    csv_path = tmp_path / "per_object.csv"
    logger_instance = PerObjectCsvLogger(csv_path)

    logger_instance.log_rows(
        frame_index=1,
        timestamp="2026-07-27T20:56:00",
        camera_motion=None,
        per_object=[make_per_object_row()],
    )
    logger_instance.close()

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["compensation_status"] == "UNAVAILABLE"
    assert rows[0]["feature_count"] == "0"


def test_per_object_csv_rows_flushed_without_closing(tmp_path) -> None:
    csv_path = tmp_path / "per_object.csv"
    logger_instance = PerObjectCsvLogger(csv_path)

    logger_instance.log_rows(
        frame_index=1,
        timestamp="2026-07-27T20:56:00",
        camera_motion=make_motion_estimate(),
        per_object=[make_per_object_row()],
    )

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2  # header + one data row

    logger_instance.close()
