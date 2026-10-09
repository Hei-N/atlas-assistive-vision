"""Unit tests for src/motion/motion_resolver.py -- the centralized
VALID/LOW_CONFIDENCE/UNAVAILABLE resolved-motion selection stage. No
camera, model, or network required -- all inputs are synthetic
CompensatedMotion/MotionEstimate/TrajectoryPrediction objects.
"""

import pytest

from src.models import CompensatedMotion, MotionEstimate, TrajectoryPrediction
from src.motion.motion_resolver import (
    COMPENSATED,
    COMPENSATED_LOW_CONFIDENCE,
    RAW_FALLBACK,
    resolve_motion,
    resolve_motions_for_tracks,
)


def make_compensated_motion(
    track_id: int = 1,
    compensated_velocity_x: float = 5.0,
    compensated_velocity_y: float = 0.0,
    raw_velocity_x: float = 12.0,
    raw_velocity_y: float = 0.0,
) -> CompensatedMotion:
    speed = (compensated_velocity_x**2 + compensated_velocity_y**2) ** 0.5
    return CompensatedMotion(
        track_id=track_id,
        compensated_velocity_x=compensated_velocity_x,
        compensated_velocity_y=compensated_velocity_y,
        compensated_speed=speed,
        compensated_direction="right" if speed > 2.0 else "stationary",
        raw_velocity_x=raw_velocity_x,
        raw_velocity_y=raw_velocity_y,
        camera_motion_applied=True,
    )


def make_motion_estimate(status: str = "VALID", confidence: float = 0.9) -> MotionEstimate:
    return MotionEstimate(
        valid=status != "UNAVAILABLE",
        dx=2.0,
        dy=0.0,
        rotation_degrees=0.0,
        scale=1.0,
        confidence=confidence,
        feature_count=120,
        tracked_feature_count=100,
        inlier_count=90,
        inlier_ratio=0.9,
        status=status,
        reason_invalid=None if status != "UNAVAILABLE" else "insufficient features detected",
        raw_dx=2.0,
        raw_dy=0.0,
        raw_rotation_degrees=0.0,
        transform_matrix=None,
        debug_prev_points=(),
        debug_current_points=(),
        debug_inlier_flags=(),
    )


def make_raw_trajectory(
    velocity_x: float = 8.0, velocity_y: float = 0.0, valid: bool = True
) -> TrajectoryPrediction:
    speed = (velocity_x**2 + velocity_y**2) ** 0.5
    return TrajectoryPrediction(
        valid=valid,
        velocity_x=velocity_x if valid else 0.0,
        velocity_y=velocity_y if valid else 0.0,
        speed_px_per_frame=speed if valid else 0.0,
        direction="right" if valid else "unknown",
        current_center=(50, 50),
        predicted_center=(58, 50) if valid else (50, 50),
        observations_used=5,
        prediction_horizon_frames=10,
        uncertain=not valid,
    )


# --- resolve_motion --------------------------------------------------------


def test_valid_status_uses_compensated_velocity_and_is_not_uncertain() -> None:
    compensated = make_compensated_motion(compensated_velocity_x=5.0)
    camera_motion = make_motion_estimate(status="VALID")
    raw_trajectory = make_raw_trajectory(velocity_x=8.0)

    result = resolve_motion(1, compensated, camera_motion, raw_trajectory)

    assert result.track_id == 1
    assert result.velocity_x == pytest.approx(5.0)
    assert result.velocity_y == pytest.approx(0.0)
    assert result.speed == pytest.approx(5.0)
    assert result.source == COMPENSATED
    assert result.uncertain is False


def test_low_confidence_still_uses_compensated_velocity_but_marks_uncertain() -> None:
    compensated = make_compensated_motion(compensated_velocity_x=5.0)
    camera_motion = make_motion_estimate(status="LOW_CONFIDENCE")
    raw_trajectory = make_raw_trajectory(velocity_x=8.0)

    result = resolve_motion(1, compensated, camera_motion, raw_trajectory)

    assert result.velocity_x == pytest.approx(5.0)
    assert result.source == COMPENSATED_LOW_CONFIDENCE
    assert result.uncertain is True


def test_unavailable_falls_back_to_raw_trajectory_velocity() -> None:
    compensated = make_compensated_motion(compensated_velocity_x=5.0)
    camera_motion = make_motion_estimate(status="UNAVAILABLE")
    raw_trajectory = make_raw_trajectory(velocity_x=8.0)

    result = resolve_motion(1, compensated, camera_motion, raw_trajectory)

    assert result.velocity_x == pytest.approx(8.0)
    assert result.source == RAW_FALLBACK
    assert result.uncertain is True


def test_camera_motion_none_is_treated_as_unavailable() -> None:
    compensated = make_compensated_motion(compensated_velocity_x=5.0)
    raw_trajectory = make_raw_trajectory(velocity_x=8.0)

    result = resolve_motion(1, compensated, None, raw_trajectory)

    assert result.velocity_x == pytest.approx(8.0)
    assert result.source == RAW_FALLBACK
    assert result.uncertain is True


def test_no_compensated_motion_falls_back_regardless_of_camera_status() -> None:
    # Compensation disabled or not yet available for this track -- even a
    # VALID camera status can't help without a CompensatedMotion.
    camera_motion = make_motion_estimate(status="VALID")
    raw_trajectory = make_raw_trajectory(velocity_x=8.0)

    result = resolve_motion(1, None, camera_motion, raw_trajectory)

    assert result.velocity_x == pytest.approx(8.0)
    assert result.source == RAW_FALLBACK
    assert result.uncertain is True


def test_fallback_reads_zero_when_raw_trajectory_itself_is_invalid() -> None:
    camera_motion = make_motion_estimate(status="UNAVAILABLE")
    raw_trajectory = make_raw_trajectory(valid=False)

    result = resolve_motion(1, None, camera_motion, raw_trajectory)

    assert result.velocity_x == 0.0
    assert result.velocity_y == 0.0
    assert result.uncertain is True


# --- resolve_motions_for_tracks --------------------------------------------


def test_resolve_motions_for_tracks_builds_one_entry_per_track() -> None:
    from src.models import BoundingBox, TrackedObject

    def make_track(track_id: int) -> TrackedObject:
        return TrackedObject(
            track_id=track_id,
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=BoundingBox(x1=0, y1=0, x2=20, y2=20),
            center=(10, 10),
            region="center",
            position_history=((0, 10), (10, 10)),
            size_history=((20, 20), (20, 20)),
            direction="right",
            motion_status="stationary",
            frames_since_seen=0,
        )

    tracks = [make_track(1), make_track(2)]
    compensated_motions = {
        1: make_compensated_motion(track_id=1, compensated_velocity_x=5.0),
        2: make_compensated_motion(track_id=2, compensated_velocity_x=1.0),
    }
    camera_motion = make_motion_estimate(status="VALID")
    raw_trajectories = {
        1: make_raw_trajectory(velocity_x=8.0),
        2: make_raw_trajectory(velocity_x=1.5),
    }

    resolved = resolve_motions_for_tracks(
        tracks, compensated_motions, camera_motion, raw_trajectories
    )

    assert set(resolved.keys()) == {1, 2}
    assert resolved[1].velocity_x == pytest.approx(5.0)
    assert resolved[2].velocity_x == pytest.approx(1.0)


def test_resolve_motions_for_tracks_skips_track_missing_raw_trajectory() -> None:
    from src.models import BoundingBox, TrackedObject

    track = TrackedObject(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=BoundingBox(x1=0, y1=0, x2=20, y2=20),
        center=(10, 10),
        region="center",
        position_history=((10, 10),),
        size_history=((20, 20),),
        direction="right",
        motion_status="stationary",
        frames_since_seen=0,
    )

    resolved = resolve_motions_for_tracks(
        [track], {1: make_compensated_motion(track_id=1)}, make_motion_estimate(), {}
    )

    assert resolved == {}
