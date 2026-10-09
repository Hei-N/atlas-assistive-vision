"""Unit tests for TrajectoryPredictor. No camera, model, or network
required -- all inputs are synthetic position histories.
"""

import math

import pytest

from src.trajectory_predictor import TrajectoryPredictor

FRAME_WIDTH = 1000
FRAME_HEIGHT = 1000


@pytest.fixture
def predictor() -> TrajectoryPredictor:
    return TrajectoryPredictor(
        min_observations=5,
        prediction_horizon_frames=10,
        stationary_threshold_px_per_frame=1.5,
        max_missed_frames=5,
        history_window=10,
    )


def test_consistent_rightward_movement(predictor: TrajectoryPredictor) -> None:
    history = [(100 + i * 10, 200) for i in range(5)]
    result = predictor.predict(tuple(history), 0, FRAME_WIDTH, FRAME_HEIGHT)

    assert result.valid
    assert result.velocity_x > 0
    assert abs(result.velocity_y) < 0.01
    assert result.direction == "right"
    assert result.predicted_center[0] > result.current_center[0]


def test_consistent_leftward_movement(predictor: TrajectoryPredictor) -> None:
    history = [(200 - i * 10, 200) for i in range(5)]
    result = predictor.predict(tuple(history), 0, FRAME_WIDTH, FRAME_HEIGHT)

    assert result.valid
    assert result.velocity_x < 0
    assert result.direction == "left"
    assert result.predicted_center[0] < result.current_center[0]


def test_diagonal_movement_upper_right(predictor: TrajectoryPredictor) -> None:
    # x increases (right), y decreases (up on screen) -> "upper-right".
    history = [(100 + i * 10, 200 - i * 10) for i in range(5)]
    result = predictor.predict(tuple(history), 0, FRAME_WIDTH, FRAME_HEIGHT)

    assert result.valid
    assert result.velocity_x > 0
    assert result.velocity_y < 0
    assert result.direction == "upper-right"


def test_diagonal_movement_lower_left(predictor: TrajectoryPredictor) -> None:
    # x decreases (left), y increases (down on screen) -> "lower-left".
    history = [(200 - i * 10, 200 + i * 10) for i in range(5)]
    result = predictor.predict(tuple(history), 0, FRAME_WIDTH, FRAME_HEIGHT)

    assert result.valid
    assert result.direction == "lower-left"


def test_nearly_unchanged_points_are_stationary(
    predictor: TrajectoryPredictor,
) -> None:
    history = [(100, 200), (101, 201), (99, 199), (100, 200), (101, 199)]
    result = predictor.predict(tuple(history), 0, FRAME_WIDTH, FRAME_HEIGHT)

    assert result.valid
    assert result.direction == "stationary"


def test_insufficient_history_is_invalid(predictor: TrajectoryPredictor) -> None:
    history = [(100, 200), (110, 200), (120, 200)]  # fewer than min_observations=5
    result = predictor.predict(tuple(history), 0, FRAME_WIDTH, FRAME_HEIGHT)

    assert not result.valid
    assert result.direction == "unknown"
    assert result.predicted_center == result.current_center


def test_predicted_coordinates_clamped_to_frame_boundaries(
    predictor: TrajectoryPredictor,
) -> None:
    small_frame_width = 50
    small_frame_height = 50
    history = [(10 + i * 5, 10) for i in range(5)]  # moving right toward the edge
    result = predictor.predict(
        tuple(history), 0, small_frame_width, small_frame_height
    )

    assert result.valid
    assert result.predicted_center[0] == small_frame_width - 1
    assert 0 <= result.predicted_center[1] <= small_frame_height - 1


def test_noisy_but_generally_rightward_history_still_predicts_right(
    predictor: TrajectoryPredictor,
) -> None:
    history = [
        (100 + i * 10 + (3 if i % 2 == 0 else -3), 200) for i in range(10)
    ]
    result = predictor.predict(tuple(history), 0, FRAME_WIDTH, FRAME_HEIGHT)

    assert result.valid
    assert result.velocity_x > 0
    assert result.direction == "right"


def test_non_finite_history_handled_safely(predictor: TrajectoryPredictor) -> None:
    history = [
        (100, 200),
        (110, 200),
        (float("nan"), 200),
        (130, 200),
        (140, 200),
    ]
    result = predictor.predict(tuple(history), 0, FRAME_WIDTH, FRAME_HEIGHT)

    assert not result.valid
    assert result.direction == "unknown"


def test_infinite_value_in_history_handled_safely(
    predictor: TrajectoryPredictor,
) -> None:
    history = [
        (100, 200),
        (110, 200),
        (float("inf"), 200),
        (130, 200),
        (140, 200),
    ]
    result = predictor.predict(tuple(history), 0, FRAME_WIDTH, FRAME_HEIGHT)

    assert not result.valid


def test_malformed_point_handled_safely(predictor: TrajectoryPredictor) -> None:
    history = [(100, 200), (110, 200), 42, (130, 200), (140, 200)]
    result = predictor.predict(tuple(history), 0, FRAME_WIDTH, FRAME_HEIGHT)

    assert not result.valid


def test_stale_track_beyond_max_missed_frames_is_invalid(
    predictor: TrajectoryPredictor,
) -> None:
    history = [(100 + i * 10, 200) for i in range(5)]
    result = predictor.predict(tuple(history), 6, FRAME_WIDTH, FRAME_HEIGHT)  # max=5

    assert not result.valid
    assert result.direction == "unknown"


def test_track_within_max_missed_frames_still_predicts(
    predictor: TrajectoryPredictor,
) -> None:
    history = [(100 + i * 10, 200) for i in range(5)]
    result = predictor.predict(tuple(history), 5, FRAME_WIDTH, FRAME_HEIGHT)  # == max

    assert result.valid


def test_history_longer_than_window_uses_only_recent_points(
    predictor: TrajectoryPredictor,
) -> None:
    # history_window=10; give 20 points where the older half moves left and
    # the recent 10 move steadily right -- only the recent trend should win.
    older = [(1000 - i * 10, 200) for i in range(10)]
    recent = [(0 + i * 10, 200) for i in range(10)]
    history = tuple(older + recent)

    result = predictor.predict(history, 0, FRAME_WIDTH, FRAME_HEIGHT)

    assert result.valid
    assert result.observations_used == 10
    assert result.direction == "right"


# --- predict_from_resolved_motion ------------------------------------------


def test_predict_from_resolved_motion_valid_velocity(
    predictor: TrajectoryPredictor,
) -> None:
    result = predictor.predict_from_resolved_motion(
        (100, 200), 10.0, 0.0, 5, FRAME_WIDTH, FRAME_HEIGHT, uncertain=False
    )

    assert result.valid
    assert result.velocity_x == pytest.approx(10.0)
    assert result.velocity_y == pytest.approx(0.0)
    assert result.direction == "right"
    assert result.current_center == (100, 200)
    assert result.predicted_center[0] == 100 + 10.0 * 10  # horizon=10
    assert result.uncertain is False


def test_predict_from_resolved_motion_passes_through_uncertain_true(
    predictor: TrajectoryPredictor,
) -> None:
    result = predictor.predict_from_resolved_motion(
        (100, 200), 10.0, 0.0, 5, FRAME_WIDTH, FRAME_HEIGHT, uncertain=True
    )

    assert result.valid
    assert result.uncertain is True


def test_predict_from_resolved_motion_non_finite_velocity_is_invalid(
    predictor: TrajectoryPredictor,
) -> None:
    result = predictor.predict_from_resolved_motion(
        (100, 200), math.nan, 0.0, 5, FRAME_WIDTH, FRAME_HEIGHT, uncertain=False
    )

    assert not result.valid
    assert result.direction == "unknown"
    assert result.uncertain is True  # always uncertain when invalid, regardless of input


def test_predict_from_resolved_motion_clamps_to_frame_boundaries(
    predictor: TrajectoryPredictor,
) -> None:
    result = predictor.predict_from_resolved_motion(
        (10, 10), 50.0, 0.0, 5, 50, 50, uncertain=False
    )

    assert result.valid
    assert result.predicted_center[0] == 49  # clamped to frame_width - 1


def test_predict_from_resolved_motion_stationary_when_below_threshold(
    predictor: TrajectoryPredictor,
) -> None:
    result = predictor.predict_from_resolved_motion(
        (100, 200), 0.5, 0.0, 5, FRAME_WIDTH, FRAME_HEIGHT, uncertain=False
    )

    assert result.valid
    assert result.direction == "stationary"


# --- resolved-motion integration scenarios (requirement 12.A/B/E) ---------


def test_stationary_object_under_camera_pan_resolves_near_zero(
    predictor: TrajectoryPredictor,
) -> None:
    # Scenario A: a real-world-stationary object's raw position history
    # drifts because the camera is panning (+10 px/frame); resolved
    # velocity (camera motion already subtracted upstream) should be ~0,
    # unlike the raw regression fit which reports the apparent drift.
    raw_history = [(100 + i * 10, 200) for i in range(5)]
    raw = predictor.predict(tuple(raw_history), 0, FRAME_WIDTH, FRAME_HEIGHT)
    assert raw.velocity_x > 5.0  # raw trajectory falsely "moves"

    resolved = predictor.predict_from_resolved_motion(
        raw.current_center, 0.0, 0.0, raw.observations_used,
        FRAME_WIDTH, FRAME_HEIGHT, uncertain=False,
    )

    assert resolved.direction == "stationary"
    assert resolved.predicted_center == resolved.current_center


def test_moving_object_and_moving_camera_preserves_object_motion(
    predictor: TrajectoryPredictor,
) -> None:
    # Scenario B: camera pans (+10 px/frame) while the object also moves
    # independently (+5 px/frame of its own) -- raw history reflects the
    # combined (+15) apparent motion, but the resolved velocity (as
    # already isolated by MotionCompensator/motion_resolver upstream)
    # should preserve just the object's own +5 px/frame.
    resolved = predictor.predict_from_resolved_motion(
        (100, 200), 5.0, 0.0, 5, FRAME_WIDTH, FRAME_HEIGHT, uncertain=False
    )

    assert resolved.valid
    assert resolved.velocity_x == pytest.approx(5.0)
    assert resolved.direction == "right"
    assert resolved.predicted_center[0] > resolved.current_center[0]


def test_zero_camera_motion_resolved_matches_raw(
    predictor: TrajectoryPredictor,
) -> None:
    # Scenario E: with no camera motion at all, the resolved velocity
    # (unchanged from raw) should produce an equivalent trajectory to the
    # raw regression fit.
    history = [(100 + i * 10, 200) for i in range(5)]
    raw = predictor.predict(tuple(history), 0, FRAME_WIDTH, FRAME_HEIGHT)

    resolved = predictor.predict_from_resolved_motion(
        raw.current_center, raw.velocity_x, raw.velocity_y, raw.observations_used,
        FRAME_WIDTH, FRAME_HEIGHT, uncertain=False,
    )

    assert resolved.predicted_center == raw.predicted_center
    assert resolved.direction == raw.direction


def test_min_observations_must_be_at_least_two() -> None:
    with pytest.raises(ValueError):
        TrajectoryPredictor(
            min_observations=1,
            prediction_horizon_frames=10,
            stationary_threshold_px_per_frame=1.5,
            max_missed_frames=5,
            history_window=10,
        )


def test_prediction_horizon_must_be_positive() -> None:
    with pytest.raises(ValueError):
        TrajectoryPredictor(
            min_observations=5,
            prediction_horizon_frames=0,
            stationary_threshold_px_per_frame=1.5,
            max_missed_frames=5,
            history_window=10,
        )


def test_stationary_threshold_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        TrajectoryPredictor(
            min_observations=5,
            prediction_horizon_frames=10,
            stationary_threshold_px_per_frame=-1,
            max_missed_frames=5,
            history_window=10,
        )


def test_max_missed_frames_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        TrajectoryPredictor(
            min_observations=5,
            prediction_horizon_frames=10,
            stationary_threshold_px_per_frame=1.5,
            max_missed_frames=-1,
            history_window=10,
        )


def test_history_window_must_be_at_least_min_observations() -> None:
    with pytest.raises(ValueError):
        TrajectoryPredictor(
            min_observations=5,
            prediction_horizon_frames=10,
            stationary_threshold_px_per_frame=1.5,
            max_missed_frames=5,
            history_window=3,
        )
