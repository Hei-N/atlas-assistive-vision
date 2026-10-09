"""Unit tests for src/motion/compensation_validator.py (the
--validate-compensation developer mode). No camera, model, or network
required -- all inputs are synthetic TrackedObject/CompensatedMotion/
MotionEstimate objects.
"""

import pytest

from src.models import (
    BoundingBox,
    CompensatedMotion,
    FilteredMotion,
    MotionEstimate,
    PathIntersectionResult,
    ResolvedMotion,
    TrackedObject,
)
from src.motion.compensation_validator import (
    CAMERA_DOMINATED,
    OBJECT_MOVING,
    STATIONARY,
    UNCERTAIN,
    build_per_object_validation,
    build_validation_stats,
    classify_status,
    should_log_validation_summary,
)
from src.motion.motion_resolver import COMPENSATED, RAW_FALLBACK
from src.motion.motion_state_filter import INSUFFICIENT_HISTORY

STATIONARY_THRESHOLD_PX = 2.0


def make_tracked_object(track_id: int = 1, center: tuple = (50, 50)) -> TrackedObject:
    bbox = BoundingBox(
        x1=center[0] - 10, y1=center[1] - 10, x2=center[0] + 10, y2=center[1] + 10
    )
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=bbox,
        center=center,
        region="center",
        position_history=(center,),
        size_history=((20, 20),),
        direction="right",
        motion_status="stationary",
        frames_since_seen=0,
    )


def make_compensated_motion(
    track_id: int = 1,
    raw_velocity_x: float = 0.0,
    raw_velocity_y: float = 0.0,
    compensated_velocity_x: float = 0.0,
    compensated_velocity_y: float = 0.0,
    camera_motion_applied: bool = True,
) -> CompensatedMotion:
    speed = (compensated_velocity_x**2 + compensated_velocity_y**2) ** 0.5
    return CompensatedMotion(
        track_id=track_id,
        compensated_velocity_x=compensated_velocity_x,
        compensated_velocity_y=compensated_velocity_y,
        compensated_speed=speed,
        compensated_direction="right" if speed > STATIONARY_THRESHOLD_PX else "stationary",
        raw_velocity_x=raw_velocity_x,
        raw_velocity_y=raw_velocity_y,
        camera_motion_applied=camera_motion_applied,
    )


def make_motion_estimate(dx: float = 2.0, dy: float = 0.0, confidence: float = 0.9) -> MotionEstimate:
    tracked_feature_count = 100
    inlier_count = 90
    return MotionEstimate(
        valid=True,
        dx=dx,
        dy=dy,
        rotation_degrees=0.0,
        scale=1.0,
        confidence=confidence,
        feature_count=120,
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


def make_resolved_motion(
    track_id: int = 1,
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
    source: str = COMPENSATED,
    uncertain: bool = False,
) -> ResolvedMotion:
    return ResolvedMotion(
        track_id=track_id,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        speed=(velocity_x**2 + velocity_y**2) ** 0.5,
        source=source,
        uncertain=uncertain,
    )


def make_intersection_result(
    track_id: int = 1, intersects: bool = False, uncertain: bool = False
) -> PathIntersectionResult:
    return PathIntersectionResult(
        valid=True,
        intersects=intersects,
        starts_inside=intersects,
        ends_inside=intersects,
        intersection_point=None,
        track_id=track_id,
        reason="test",
        uncertain=uncertain,
    )


def make_filtered_motion(
    track_id: int = 1,
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
    motion_state: str = "STATIONARY",
    source: str = COMPENSATED,
    uncertain: bool = False,
    reason: str | None = None,
    confirmation_frames: int = 0,
) -> FilteredMotion:
    return FilteredMotion(
        track_id=track_id,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        speed=(velocity_x**2 + velocity_y**2) ** 0.5,
        motion_state=motion_state,
        source=source,
        uncertain=uncertain,
        reason=reason,
        confirmation_frames=confirmation_frames,
    )


# --- classify_status ---------------------------------------------------


def test_classify_status_stationary_when_both_speeds_low() -> None:
    status = classify_status(
        raw_speed=1.0,
        compensated_speed=1.0,
        camera_motion_applied=True,
        stationary_threshold_px=STATIONARY_THRESHOLD_PX,
    )
    assert status == STATIONARY


def test_classify_status_camera_dominated() -> None:
    status = classify_status(
        raw_speed=10.0,
        compensated_speed=0.5,
        camera_motion_applied=True,
        stationary_threshold_px=STATIONARY_THRESHOLD_PX,
    )
    assert status == CAMERA_DOMINATED


def test_classify_status_object_moving() -> None:
    status = classify_status(
        raw_speed=10.0,
        compensated_speed=8.0,
        camera_motion_applied=True,
        stationary_threshold_px=STATIONARY_THRESHOLD_PX,
    )
    assert status == OBJECT_MOVING


def test_classify_status_uncertain_when_camera_not_applied_and_moving() -> None:
    status = classify_status(
        raw_speed=10.0,
        compensated_speed=10.0,
        camera_motion_applied=False,
        stationary_threshold_px=STATIONARY_THRESHOLD_PX,
    )
    assert status == UNCERTAIN


def test_classify_status_stationary_wins_over_uncertain_when_truly_still() -> None:
    # Camera not applied, but the object shows no real motion either way --
    # should be STATIONARY, not UNCERTAIN.
    status = classify_status(
        raw_speed=0.5,
        compensated_speed=0.5,
        camera_motion_applied=False,
        stationary_threshold_px=STATIONARY_THRESHOLD_PX,
    )
    assert status == STATIONARY


# --- build_per_object_validation ----------------------------------------


def test_per_object_validation_with_stationary_object() -> None:
    track = make_tracked_object(track_id=1)
    motion = make_compensated_motion(
        track_id=1,
        raw_velocity_x=1.0,
        raw_velocity_y=0.0,
        compensated_velocity_x=0.2,
        compensated_velocity_y=0.0,
    )
    camera_motion = make_motion_estimate()

    rows = build_per_object_validation(
        [track], {1: motion}, camera_motion, STATIONARY_THRESHOLD_PX, {}, {}, {}, {}
    )

    assert len(rows) == 1
    assert rows[0].status == STATIONARY
    assert rows[0].class_name == "person"
    assert rows[0].camera_confidence == pytest.approx(0.9)


def test_per_object_validation_with_moving_object() -> None:
    track = make_tracked_object(track_id=1)
    motion = make_compensated_motion(
        track_id=1,
        raw_velocity_x=12.0,
        raw_velocity_y=0.0,
        compensated_velocity_x=9.0,
        compensated_velocity_y=0.0,
    )
    camera_motion = make_motion_estimate()

    rows = build_per_object_validation(
        [track], {1: motion}, camera_motion, STATIONARY_THRESHOLD_PX, {}, {}, {}, {}
    )

    assert rows[0].status == OBJECT_MOVING
    assert rows[0].raw_speed == pytest.approx(12.0)
    assert rows[0].compensated_speed == pytest.approx(9.0)


def test_per_object_validation_missing_entry_does_not_crash() -> None:
    track_with_data = make_tracked_object(track_id=1)
    track_without_data = make_tracked_object(track_id=2, center=(150, 150))
    motion = make_compensated_motion(track_id=1)

    rows = build_per_object_validation(
        [track_with_data, track_without_data],
        {1: motion},
        make_motion_estimate(),
        STATIONARY_THRESHOLD_PX,
        {},
        {},
        {},
        {},
    )

    assert len(rows) == 1
    assert rows[0].track_id == 1


def test_per_object_validation_with_no_camera_motion_is_uncertain() -> None:
    track = make_tracked_object(track_id=1)
    motion = make_compensated_motion(
        track_id=1,
        raw_velocity_x=10.0,
        raw_velocity_y=0.0,
        compensated_velocity_x=10.0,
        compensated_velocity_y=0.0,
        camera_motion_applied=False,
    )

    rows = build_per_object_validation(
        [track], {1: motion}, None, STATIONARY_THRESHOLD_PX, {}, {}, {}, {}
    )

    assert rows[0].status == UNCERTAIN
    assert rows[0].camera_confidence == 0.0


def test_per_object_validation_populates_resolved_and_intersection_fields() -> None:
    track = make_tracked_object(track_id=1)
    motion = make_compensated_motion(track_id=1)
    camera_motion = make_motion_estimate()
    resolved = make_resolved_motion(
        track_id=1, velocity_x=3.0, velocity_y=4.0, source=COMPENSATED, uncertain=False
    )
    raw_result = make_intersection_result(track_id=1, intersects=True)
    resolved_result = make_intersection_result(track_id=1, intersects=False)
    filtered = make_filtered_motion(
        track_id=1, velocity_x=0.0, velocity_y=0.0, motion_state="STATIONARY",
        source=COMPENSATED, uncertain=False, confirmation_frames=2,
    )

    rows = build_per_object_validation(
        [track], {1: motion}, camera_motion, STATIONARY_THRESHOLD_PX,
        {1: resolved}, {1: raw_result}, {1: resolved_result}, {1: filtered},
    )

    assert rows[0].resolved_velocity_x == pytest.approx(3.0)
    assert rows[0].resolved_velocity_y == pytest.approx(4.0)
    assert rows[0].resolved_speed == pytest.approx(5.0)
    assert rows[0].motion_source == COMPENSATED
    assert rows[0].prediction_uncertain is False
    assert rows[0].raw_intersects_corridor is True
    assert rows[0].resolved_intersects_corridor is False
    assert rows[0].filtered_velocity_x == pytest.approx(0.0)
    assert rows[0].filtered_velocity_y == pytest.approx(0.0)
    assert rows[0].motion_filter_state == "STATIONARY"
    assert rows[0].motion_filter_reason is None
    assert rows[0].motion_confirmation_frames == 2


def test_per_object_validation_defaults_safely_when_resolved_data_missing() -> None:
    # track_id present in compensated_motions but absent from the 4 new
    # dicts (defensive case -- shouldn't happen in practice).
    track = make_tracked_object(track_id=1)
    motion = make_compensated_motion(track_id=1)

    rows = build_per_object_validation(
        [track], {1: motion}, make_motion_estimate(), STATIONARY_THRESHOLD_PX, {}, {}, {}, {}
    )

    assert len(rows) == 1
    assert rows[0].motion_source == RAW_FALLBACK
    assert rows[0].resolved_velocity_x == 0.0
    assert rows[0].resolved_velocity_y == 0.0
    assert rows[0].prediction_uncertain is True
    assert rows[0].raw_intersects_corridor is False
    assert rows[0].resolved_intersects_corridor is False
    assert rows[0].filtered_velocity_x == 0.0
    assert rows[0].filtered_velocity_y == 0.0
    assert rows[0].motion_filter_state == INSUFFICIENT_HISTORY
    assert rows[0].motion_filter_reason == INSUFFICIENT_HISTORY
    assert rows[0].motion_confirmation_frames == 0


# --- build_validation_stats ----------------------------------------------


def test_validation_stats_with_no_tracks() -> None:
    stats = build_validation_stats([], None)

    assert stats.active_track_count == 0
    assert stats.stationary_count == 0
    assert stats.moving_count == 0
    assert stats.camera_dominated_count == 0
    assert stats.uncertain_count == 0
    assert stats.average_raw_speed == 0.0
    assert stats.average_compensated_speed == 0.0
    assert stats.reduction_percent == 0.0
    assert stats.camera_dx == 0.0
    assert stats.camera_confidence == 0.0


def test_validation_stats_with_stationary_objects() -> None:
    track_a = make_tracked_object(track_id=1)
    track_b = make_tracked_object(track_id=2, center=(150, 150))
    motions = {
        1: make_compensated_motion(1, 0.5, 0.0, 0.3, 0.0),
        2: make_compensated_motion(2, 0.2, 0.1, 0.1, 0.1),
    }
    camera_motion = make_motion_estimate()
    rows = build_per_object_validation(
        [track_a, track_b], motions, camera_motion, STATIONARY_THRESHOLD_PX, {}, {}, {}, {}
    )

    stats = build_validation_stats(rows, camera_motion)

    assert stats.active_track_count == 2
    assert stats.stationary_count == 2
    assert stats.moving_count == 0


def test_validation_stats_with_moving_objects() -> None:
    track = make_tracked_object(track_id=1)
    motion = make_compensated_motion(1, 12.0, 0.0, 9.0, 0.0)
    camera_motion = make_motion_estimate()
    rows = build_per_object_validation(
        [track], {1: motion}, camera_motion, STATIONARY_THRESHOLD_PX, {}, {}, {}, {}
    )

    stats = build_validation_stats(rows, camera_motion)

    assert stats.moving_count == 1
    assert stats.average_raw_speed == pytest.approx(12.0)
    assert stats.average_compensated_speed == pytest.approx(9.0)
    assert stats.reduction_percent == pytest.approx(25.0)


def test_validation_stats_reduction_percent_avoids_division_by_zero() -> None:
    track = make_tracked_object(track_id=1)
    # Both raw and compensated velocity are exactly zero.
    motion = make_compensated_motion(1, 0.0, 0.0, 0.0, 0.0)
    rows = build_per_object_validation(
        [track], {1: motion}, None, STATIONARY_THRESHOLD_PX, {}, {}, {}, {}
    )

    stats = build_validation_stats(rows, None)

    assert stats.average_raw_speed == 0.0
    assert stats.reduction_percent == 0.0  # not NaN/inf, no exception


def test_validation_stats_camera_fields_come_from_motion_estimate() -> None:
    camera_motion = make_motion_estimate(dx=3.5, dy=-1.2, confidence=0.77)
    stats = build_validation_stats([], camera_motion)

    assert stats.camera_dx == pytest.approx(3.5)
    assert stats.camera_dy == pytest.approx(-1.2)
    assert stats.camera_confidence == pytest.approx(0.77)


# --- should_log_validation_summary ---------------------------------------


@pytest.mark.parametrize(
    ("frame_count", "expected"),
    [
        (0, False),
        (1, False),
        (29, False),
        (30, True),
        (59, False),
        (60, True),
        (90, True),
    ],
)
def test_should_log_validation_summary_interval(frame_count: int, expected: bool) -> None:
    assert should_log_validation_summary(frame_count) is expected
