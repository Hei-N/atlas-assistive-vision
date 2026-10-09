"""Unit tests for MotionCompensator. No camera, model, or network
required -- all inputs are synthetic TrackedObject/MotionEstimate objects.
"""

import dataclasses

import pytest

from src.models import BoundingBox, MotionEstimate, TrackedObject
from src.motion.motion_compensator import MotionCompensator


def make_tracked_object(
    track_id: int = 1,
    position_history: tuple = ((0, 0), (10, 0)),
) -> TrackedObject:
    bbox = BoundingBox(x1=0, y1=0, x2=20, y2=20)
    center = position_history[-1] if position_history else (0, 0)
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=bbox,
        center=center,
        region="center",
        position_history=tuple(position_history),
        size_history=((20, 20),) * len(position_history),
        direction="right",
        motion_status="stationary",
        frames_since_seen=0,
    )


def make_motion_estimate(
    valid: bool = True,
    dx: float = 0.0,
    dy: float = 0.0,
    confidence: float = 1.0,
) -> MotionEstimate:
    tracked_feature_count = 100
    inlier_count = 90
    if valid:
        status = "VALID" if confidence >= 0.6 else "LOW_CONFIDENCE"
    else:
        status = "UNAVAILABLE"
    return MotionEstimate(
        valid=valid,
        dx=dx,
        dy=dy,
        rotation_degrees=0.0,
        scale=1.0,
        confidence=confidence,
        feature_count=120,
        tracked_feature_count=tracked_feature_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_count / tracked_feature_count,
        status=status,
        reason_invalid=None if valid else "test: invalid camera motion",
        raw_dx=dx,
        raw_dy=dy,
        raw_rotation_degrees=0.0,
        transform_matrix=None,
        debug_prev_points=(),
        debug_current_points=(),
        debug_inlier_flags=(),
    )


@pytest.fixture
def compensator() -> MotionCompensator:
    return MotionCompensator(min_camera_confidence=0.5, stationary_threshold_px=2.0)


def test_stationary_object_with_camera_translation_cancels_out(
    compensator: MotionCompensator,
) -> None:
    # Raw object motion (+10, 0); camera-induced motion (+10, 0).
    track = make_tracked_object(position_history=((0, 0), (10, 0)))
    camera_motion = make_motion_estimate(dx=10, dy=0)

    result = compensator.compensate([track], camera_motion)[track.track_id]

    assert result.camera_motion_applied
    assert result.compensated_velocity_x == pytest.approx(0.0, abs=0.01)
    assert result.compensated_velocity_y == pytest.approx(0.0, abs=0.01)
    assert result.compensated_direction == "stationary"


def test_moving_object_and_moving_camera_isolates_object_motion(
    compensator: MotionCompensator,
) -> None:
    # Raw object motion (+15, +2); camera-induced motion (+10, 0).
    track = make_tracked_object(position_history=((0, 0), (15, 2)))
    camera_motion = make_motion_estimate(dx=10, dy=0)

    result = compensator.compensate([track], camera_motion)[track.track_id]

    assert result.camera_motion_applied
    assert result.compensated_velocity_x == pytest.approx(5.0, abs=0.01)
    assert result.compensated_velocity_y == pytest.approx(2.0, abs=0.01)
    assert result.compensated_speed == pytest.approx((5.0**2 + 2.0**2) ** 0.5, abs=0.01)


def test_moving_object_with_a_present_but_stationary_camera(
    compensator: MotionCompensator,
) -> None:
    # Distinct from the "no camera estimate at all" case below: here
    # camera_motion IS present and valid, it just measured zero motion
    # (a genuinely still camera) -- subtracting (0, 0) should leave the
    # object's raw velocity untouched.
    track = make_tracked_object(position_history=((0, 0), (15, 2)))
    camera_motion = make_motion_estimate(dx=0.0, dy=0.0, confidence=1.0)

    result = compensator.compensate([track], camera_motion)[track.track_id]

    assert result.camera_motion_applied
    assert result.compensated_velocity_x == pytest.approx(15.0)
    assert result.compensated_velocity_y == pytest.approx(2.0)


def test_no_camera_motion_falls_back_to_raw_velocity(
    compensator: MotionCompensator,
) -> None:
    track = make_tracked_object(position_history=((0, 0), (15, 2)))

    result = compensator.compensate([track], None)[track.track_id]

    assert not result.camera_motion_applied
    assert result.compensated_velocity_x == pytest.approx(15.0)
    assert result.compensated_velocity_y == pytest.approx(2.0)
    assert result.raw_velocity_x == pytest.approx(15.0)
    assert result.raw_velocity_y == pytest.approx(2.0)


def test_invalid_camera_motion_falls_back_to_raw_velocity(
    compensator: MotionCompensator,
) -> None:
    track = make_tracked_object(position_history=((0, 0), (15, 2)))
    camera_motion = make_motion_estimate(valid=False, dx=999, dy=999)

    result = compensator.compensate([track], camera_motion)[track.track_id]

    assert not result.camera_motion_applied
    assert result.compensated_velocity_x == pytest.approx(15.0)
    assert result.compensated_velocity_y == pytest.approx(2.0)


def test_low_confidence_camera_motion_falls_back_to_raw_velocity(
    compensator: MotionCompensator,
) -> None:
    track = make_tracked_object(position_history=((0, 0), (15, 2)))
    # confidence 0.1 < compensator's min_camera_confidence of 0.5.
    camera_motion = make_motion_estimate(dx=999, dy=999, confidence=0.1)

    result = compensator.compensate([track], camera_motion)[track.track_id]

    assert not result.camera_motion_applied
    assert result.compensated_velocity_x == pytest.approx(15.0)
    assert result.compensated_velocity_y == pytest.approx(2.0)


def test_confidence_exactly_at_threshold_is_trusted(
    compensator: MotionCompensator,
) -> None:
    track = make_tracked_object(position_history=((0, 0), (10, 0)))
    camera_motion = make_motion_estimate(dx=10, dy=0, confidence=0.5)

    result = compensator.compensate([track], camera_motion)[track.track_id]

    assert result.camera_motion_applied


def test_track_with_insufficient_history_defaults_safely(
    compensator: MotionCompensator,
) -> None:
    # Only one position ever recorded -- brand-new track.
    track = make_tracked_object(position_history=((5, 5),))
    camera_motion = make_motion_estimate(dx=10, dy=0)

    result = compensator.compensate([track], camera_motion)[track.track_id]

    assert result.raw_velocity_x == 0.0
    assert result.raw_velocity_y == 0.0
    # Camera motion still gets subtracted from the (zero) raw velocity.
    assert result.camera_motion_applied
    assert result.compensated_velocity_x == pytest.approx(-10.0)
    assert result.compensated_velocity_y == pytest.approx(0.0)


def test_track_with_empty_history_defaults_safely(
    compensator: MotionCompensator,
) -> None:
    track = make_tracked_object(position_history=())

    result = compensator.compensate([track], None)[track.track_id]

    assert result.raw_velocity_x == 0.0
    assert result.raw_velocity_y == 0.0
    assert result.compensated_direction == "stationary"


def test_original_track_data_is_preserved(compensator: MotionCompensator) -> None:
    track = make_tracked_object(track_id=7, position_history=((0, 0), (15, 2)))
    snapshot = dataclasses.asdict(track)
    camera_motion = make_motion_estimate(dx=10, dy=0)

    tracks = [track]
    compensator.compensate(tracks, camera_motion)

    # Same object, same list -- never replaced or mutated.
    assert tracks[0] is track
    assert dataclasses.asdict(track) == snapshot


def test_multiple_tracks_are_each_compensated_independently(
    compensator: MotionCompensator,
) -> None:
    track_a = make_tracked_object(track_id=1, position_history=((0, 0), (10, 0)))
    track_b = make_tracked_object(track_id=2, position_history=((0, 0), (0, 10)))
    camera_motion = make_motion_estimate(dx=10, dy=0)

    results = compensator.compensate([track_a, track_b], camera_motion)

    assert set(results.keys()) == {1, 2}
    assert results[1].compensated_velocity_x == pytest.approx(0.0, abs=0.01)
    assert results[2].compensated_velocity_x == pytest.approx(-10.0, abs=0.01)
    assert results[2].compensated_velocity_y == pytest.approx(10.0, abs=0.01)


def test_min_camera_confidence_must_be_in_range() -> None:
    with pytest.raises(ValueError):
        MotionCompensator(min_camera_confidence=1.5, stationary_threshold_px=2.0)
    with pytest.raises(ValueError):
        MotionCompensator(min_camera_confidence=-0.1, stationary_threshold_px=2.0)


def test_stationary_threshold_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        MotionCompensator(min_camera_confidence=0.5, stationary_threshold_px=-1.0)
