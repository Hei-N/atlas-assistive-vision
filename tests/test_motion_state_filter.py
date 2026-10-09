"""Unit tests for src/motion/motion_state_filter.py -- the stationary-
motion dead-zone/hysteresis/temporal-confirmation filter that sits
between ResolvedMotion and trajectory prediction. No camera, model, or
network required -- all inputs are synthetic ResolvedMotion/
TrajectoryPrediction objects and a real MotionStateFilter/
TrajectoryPredictor/PathIntersectionAnalyzer.
"""

import pytest

from src.models import ResolvedMotion, TrajectoryPrediction
from src.motion.motion_resolver import COMPENSATED, COMPENSATED_LOW_CONFIDENCE, RAW_FALLBACK
from src.motion.motion_state_filter import (
    INSUFFICIENT_HISTORY,
    LOW_COMPENSATION_CONFIDENCE,
    MOVING,
    STATIONARY,
    UNCERTAIN,
    MotionStateFilter,
)
from src.path_intersection_analyzer import PathIntersectionAnalyzer
from src.trajectory_predictor import TrajectoryPredictor

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def make_resolved(
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


@pytest.fixture
def motion_filter() -> MotionStateFilter:
    return MotionStateFilter(
        stationary_enter_speed=2.5,
        moving_enter_speed=4.0,
        moving_confirmation_frames=3,
        stationary_confirmation_frames=3,
        minimum_history_samples=5,
        smoothing_alpha=1.0,  # no smoothing -- deterministic per-call values
        suppress_raw_motion_during_camera_motion=True,
    )


# --- 1. Perfectly stationary, stable camera --------------------------------


def test_perfectly_stationary_stable_camera(motion_filter: MotionStateFilter) -> None:
    resolved = make_resolved(velocity_x=0.0, velocity_y=0.0, source=COMPENSATED)
    result = motion_filter.filter(1, resolved, observations_used=10)

    assert result.motion_state == STATIONARY
    assert result.velocity_x == 0.0
    assert result.velocity_y == 0.0
    assert result.uncertain is False
    assert result.reason is None


# --- 2. Small box jitter ----------------------------------------------------


def test_small_box_jitter_remains_stationary(motion_filter: MotionStateFilter) -> None:
    resolved = make_resolved(velocity_x=1.0, velocity_y=0.5, source=COMPENSATED)
    result = motion_filter.filter(1, resolved, observations_used=10)

    assert result.motion_state == STATIONARY
    assert result.velocity_x == 0.0
    assert result.velocity_y == 0.0


# --- 3. Camera pans, VALID compensation, small residual --------------------


def test_camera_pan_valid_compensation_small_residual_remains_stationary(
    motion_filter: MotionStateFilter,
) -> None:
    resolved = make_resolved(
        velocity_x=0.8, velocity_y=0.0, source=COMPENSATED, uncertain=False
    )
    result = motion_filter.filter(1, resolved, observations_used=10)

    assert result.motion_state == STATIONARY
    assert result.uncertain is False


# --- 4. LOW_CONFIDENCE, small residual --------------------------------------


def test_low_confidence_small_residual_stays_stationary_but_uncertain(
    motion_filter: MotionStateFilter,
) -> None:
    resolved = make_resolved(
        velocity_x=0.5, velocity_y=0.0, source=COMPENSATED_LOW_CONFIDENCE, uncertain=True
    )
    result = motion_filter.filter(1, resolved, observations_used=10)

    assert result.motion_state == STATIONARY
    assert result.uncertain is True
    assert result.reason == LOW_COMPENSATION_CONFIDENCE


def test_low_confidence_can_still_reach_moving_but_always_uncertain(
    motion_filter: MotionStateFilter,
) -> None:
    resolved = make_resolved(
        velocity_x=10.0, velocity_y=0.0, source=COMPENSATED_LOW_CONFIDENCE, uncertain=True
    )
    for _ in range(3):  # moving_confirmation_frames
        result = motion_filter.filter(1, resolved, observations_used=10)

    assert result.motion_state == MOVING
    assert result.uncertain is True
    assert result.reason == LOW_COMPENSATION_CONFIDENCE


# --- 5. RAW_FALLBACK during suspected camera movement -----------------------


def test_raw_fallback_never_asserts_confident_moving_claim(
    motion_filter: MotionStateFilter,
) -> None:
    resolved = make_resolved(
        velocity_x=10.0, velocity_y=0.0, source=RAW_FALLBACK, uncertain=True
    )
    for _ in range(3):  # would confirm MOVING for a COMPENSATED source
        result = motion_filter.filter(1, resolved, observations_used=10)

    assert result.motion_state == UNCERTAIN
    assert result.uncertain is True
    assert result.reason == RAW_FALLBACK
    assert result.velocity_x == 0.0
    assert result.velocity_y == 0.0


def test_raw_fallback_near_zero_still_reported_stationary(
    motion_filter: MotionStateFilter,
) -> None:
    # A genuinely near-zero RAW_FALLBACK reading is not a risky claim --
    # suppression only applies to a would-be MOVING reading.
    resolved = make_resolved(
        velocity_x=0.1, velocity_y=0.0, source=RAW_FALLBACK, uncertain=True
    )
    result = motion_filter.filter(1, resolved, observations_used=10)

    assert result.motion_state == STATIONARY
    assert result.uncertain is True
    assert result.reason == RAW_FALLBACK


# --- 6. Truly moving vehicle, sustained velocity ----------------------------


def test_sustained_moving_velocity_confirms_only_after_confirmation_period(
    motion_filter: MotionStateFilter,
) -> None:
    resolved = make_resolved(velocity_x=10.0, velocity_y=0.0, source=COMPENSATED)

    result = motion_filter.filter(1, resolved, observations_used=10)
    assert result.motion_state == STATIONARY  # frame 1/3
    result = motion_filter.filter(1, resolved, observations_used=10)
    assert result.motion_state == STATIONARY  # frame 2/3
    result = motion_filter.filter(1, resolved, observations_used=10)
    assert result.motion_state == MOVING  # frame 3/3 -- confirmed
    assert result.velocity_x == pytest.approx(10.0)


# --- 7. Single-frame velocity spike -----------------------------------------


def test_single_frame_spike_does_not_become_moving(
    motion_filter: MotionStateFilter,
) -> None:
    spike = make_resolved(velocity_x=10.0, velocity_y=0.0, source=COMPENSATED)
    calm = make_resolved(velocity_x=0.0, velocity_y=0.0, source=COMPENSATED)

    result = motion_filter.filter(1, spike, observations_used=10)
    assert result.motion_state == STATIONARY
    result = motion_filter.filter(1, calm, observations_used=10)
    assert result.motion_state == STATIONARY
    result = motion_filter.filter(1, calm, observations_used=10)
    assert result.motion_state == STATIONARY


# --- 8. Moving object stops --------------------------------------------------


def test_moving_object_stops_only_after_stationary_confirmation(
    motion_filter: MotionStateFilter,
) -> None:
    moving = make_resolved(velocity_x=10.0, velocity_y=0.0, source=COMPENSATED)
    stopped = make_resolved(velocity_x=0.0, velocity_y=0.0, source=COMPENSATED)

    for _ in range(3):
        result = motion_filter.filter(1, moving, observations_used=10)
    assert result.motion_state == MOVING

    result = motion_filter.filter(1, stopped, observations_used=10)
    assert result.motion_state == MOVING  # frame 1/3
    result = motion_filter.filter(1, stopped, observations_used=10)
    assert result.motion_state == MOVING  # frame 2/3
    result = motion_filter.filter(1, stopped, observations_used=10)
    assert result.motion_state == STATIONARY  # frame 3/3 -- confirmed


# --- 9. Insufficient trajectory history --------------------------------------


def test_insufficient_history_reports_insufficient_history(
    motion_filter: MotionStateFilter,
) -> None:
    resolved = make_resolved(velocity_x=10.0, velocity_y=0.0, source=COMPENSATED)
    result = motion_filter.filter(1, resolved, observations_used=3)  # < 5

    assert result.motion_state == INSUFFICIENT_HISTORY
    assert result.uncertain is True
    assert result.reason == INSUFFICIENT_HISTORY
    assert result.velocity_x == 0.0
    assert result.velocity_y == 0.0


# --- 10. Velocity between hysteresis thresholds -----------------------------


def test_speed_in_dead_band_retains_previous_stable_state(
    motion_filter: MotionStateFilter,
) -> None:
    # 3.0 is strictly between stationary_enter_speed=2.5 and
    # moving_enter_speed=4.0.
    ambiguous = make_resolved(velocity_x=3.0, velocity_y=0.0, source=COMPENSATED)

    for _ in range(10):
        result = motion_filter.filter(1, ambiguous, observations_used=10)
        assert result.motion_state == STATIONARY  # never flips


# --- 11. Filtered stationary object -> no false corridor intersection ------


def test_filtered_stationary_object_produces_no_false_intersection(
    motion_filter: MotionStateFilter,
) -> None:
    predictor = TrajectoryPredictor(
        min_observations=5,
        prediction_horizon_frames=10,
        stationary_threshold_px_per_frame=1.5,
        max_missed_frames=5,
        history_window=10,
    )
    analyzer = PathIntersectionAnalyzer(
        corridor_bottom_left_x=0.35,
        corridor_bottom_right_x=0.65,
        corridor_bottom_y=1.0,
        corridor_top_left_x=0.45,
        corridor_top_right_x=0.55,
        corridor_top_y=0.55,
    )
    # A parked car sitting well outside the corridor (far left of frame).
    current_center = (20, 400)

    for jitter_x in (0.5, -0.5, 0.5, -0.5, 0.5):
        resolved = make_resolved(velocity_x=jitter_x, velocity_y=0.0, source=COMPENSATED)
        filtered = motion_filter.filter(1, resolved, observations_used=10)

        prediction = predictor.predict_from_resolved_motion(
            current_center, filtered.velocity_x, filtered.velocity_y,
            observations_used=10, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT,
            uncertain=filtered.uncertain,
        )
        result = analyzer.analyze(1, prediction, FRAME_WIDTH, FRAME_HEIGHT)

        assert prediction.predicted_center == current_center
        assert result.intersects is False


# --- 12. Uncertainty propagation remains intact through PathIntersectionResult


def test_uncertainty_propagates_through_filtered_path(
    motion_filter: MotionStateFilter,
) -> None:
    predictor = TrajectoryPredictor(
        min_observations=5,
        prediction_horizon_frames=10,
        stationary_threshold_px_per_frame=1.5,
        max_missed_frames=5,
        history_window=10,
    )
    analyzer = PathIntersectionAnalyzer(
        corridor_bottom_left_x=0.35,
        corridor_bottom_right_x=0.65,
        corridor_bottom_y=1.0,
        corridor_top_left_x=0.45,
        corridor_top_right_x=0.55,
        corridor_top_y=0.55,
    )
    resolved = make_resolved(
        velocity_x=0.5, velocity_y=0.0, source=COMPENSATED_LOW_CONFIDENCE, uncertain=True
    )
    filtered = motion_filter.filter(1, resolved, observations_used=10)
    assert filtered.uncertain is True

    prediction = predictor.predict_from_resolved_motion(
        (100, 100), filtered.velocity_x, filtered.velocity_y,
        observations_used=10, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT,
        uncertain=filtered.uncertain,
    )
    assert prediction.uncertain is True

    result = analyzer.analyze(1, prediction, FRAME_WIDTH, FRAME_HEIGHT)
    assert result.uncertain is True


# --- filter_for_tracks / pruning --------------------------------------------


def test_filter_for_tracks_prunes_stale_track_state() -> None:
    from src.models import BoundingBox, TrackedObject

    filter_ = MotionStateFilter(
        stationary_enter_speed=2.5, moving_enter_speed=4.0,
        moving_confirmation_frames=3, stationary_confirmation_frames=3,
        minimum_history_samples=5, smoothing_alpha=1.0,
        suppress_raw_motion_during_camera_motion=True,
    )

    def make_track(track_id: int) -> TrackedObject:
        return TrackedObject(
            track_id=track_id, class_id=2, class_name="car", confidence=0.9,
            bbox=BoundingBox(x1=0, y1=0, x2=20, y2=20), center=(10, 10),
            region="center", position_history=((10, 10),) * 6,
            size_history=((20, 20),) * 6, direction="stationary",
            motion_status="stationary", frames_since_seen=0,
        )

    track1 = make_track(1)
    resolved_1 = {1: make_resolved(track_id=1, velocity_x=0.0, velocity_y=0.0)}
    filter_.filter_for_tracks([track1], resolved_1)
    assert 1 in filter_._states

    track2 = make_track(2)
    resolved_2 = {2: make_resolved(track_id=2, velocity_x=0.0, velocity_y=0.0)}
    filter_.filter_for_tracks([track2], resolved_2)
    assert 1 not in filter_._states
    assert 2 in filter_._states


# --- constructor validation --------------------------------------------------


def test_stationary_enter_speed_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        MotionStateFilter(
            stationary_enter_speed=-1.0, moving_enter_speed=4.0,
            moving_confirmation_frames=3, stationary_confirmation_frames=3,
            minimum_history_samples=5, smoothing_alpha=1.0,
            suppress_raw_motion_during_camera_motion=True,
        )


def test_moving_enter_speed_must_be_at_least_stationary_enter_speed() -> None:
    with pytest.raises(ValueError):
        MotionStateFilter(
            stationary_enter_speed=4.0, moving_enter_speed=2.5,
            moving_confirmation_frames=3, stationary_confirmation_frames=3,
            minimum_history_samples=5, smoothing_alpha=1.0,
            suppress_raw_motion_during_camera_motion=True,
        )


def test_moving_confirmation_frames_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        MotionStateFilter(
            stationary_enter_speed=2.5, moving_enter_speed=4.0,
            moving_confirmation_frames=0, stationary_confirmation_frames=3,
            minimum_history_samples=5, smoothing_alpha=1.0,
            suppress_raw_motion_during_camera_motion=True,
        )


def test_stationary_confirmation_frames_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        MotionStateFilter(
            stationary_enter_speed=2.5, moving_enter_speed=4.0,
            moving_confirmation_frames=3, stationary_confirmation_frames=0,
            minimum_history_samples=5, smoothing_alpha=1.0,
            suppress_raw_motion_during_camera_motion=True,
        )


def test_minimum_history_samples_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        MotionStateFilter(
            stationary_enter_speed=2.5, moving_enter_speed=4.0,
            moving_confirmation_frames=3, stationary_confirmation_frames=3,
            minimum_history_samples=0, smoothing_alpha=1.0,
            suppress_raw_motion_during_camera_motion=True,
        )


def test_smoothing_alpha_must_be_within_zero_exclusive_one_inclusive() -> None:
    with pytest.raises(ValueError):
        MotionStateFilter(
            stationary_enter_speed=2.5, moving_enter_speed=4.0,
            moving_confirmation_frames=3, stationary_confirmation_frames=3,
            minimum_history_samples=5, smoothing_alpha=0.0,
            suppress_raw_motion_during_camera_motion=True,
        )
    with pytest.raises(ValueError):
        MotionStateFilter(
            stationary_enter_speed=2.5, moving_enter_speed=4.0,
            moving_confirmation_frames=3, stationary_confirmation_frames=3,
            minimum_history_samples=5, smoothing_alpha=1.1,
            suppress_raw_motion_during_camera_motion=True,
        )


# --- smoothing ---------------------------------------------------------------


def test_smoothing_reduces_impact_of_a_single_noisy_sample() -> None:
    filter_ = MotionStateFilter(
        stationary_enter_speed=2.5, moving_enter_speed=4.0,
        moving_confirmation_frames=3, stationary_confirmation_frames=3,
        minimum_history_samples=5, smoothing_alpha=0.3,
        suppress_raw_motion_during_camera_motion=True,
    )
    calm = make_resolved(velocity_x=0.0, velocity_y=0.0, source=COMPENSATED)
    noisy = make_resolved(velocity_x=20.0, velocity_y=0.0, source=COMPENSATED)

    filter_.filter(1, calm, observations_used=10)
    filter_.filter(1, calm, observations_used=10)
    # One noisy sample: alpha=0.3 means smoothed = 0.3*20 + 0.7*0 = 6.0,
    # which is above moving_enter_speed but this is only frame 1/3 of a
    # pending transition -- must not confirm MOVING immediately.
    result = filter_.filter(1, noisy, observations_used=10)
    assert result.motion_state == STATIONARY
