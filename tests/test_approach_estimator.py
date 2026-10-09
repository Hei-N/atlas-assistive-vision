"""Unit tests for src/motion/approach_estimator.py -- confirmed-approach
detection, strictly stronger than FilteredMotion.motion_state ==
"MOVING". Pure logic, no camera/model/network involved -- synthetic
TrackedObject/FilteredMotion/TrajectoryPrediction/PathIntersectionResult
data throughout.
"""

import pytest

from src.motion.approach_estimator import (
    APPROACHING,
    APPROACHING_UNCERTAIN,
    GROUND_POINT_CLOSING,
    NOT_APPROACHING,
    SCALE_GROWTH,
    ApproachEstimator,
)
from src.motion.relative_proximity_estimator import FAR, NEAR, RelativeProximityEstimator
from src.models import BoundingBox, FilteredMotion, PathIntersectionResult, TrackedObject, TrajectoryPrediction

FRAME_HEIGHT = 1000
# Centroid of a 4-vertex polygon roughly centered at (100, 300) -- chosen
# so a track moving from y=100 toward y=125 is unambiguously closing on
# it (distance strictly decreasing each sample).
CORRIDOR_POLYGON = ((80, 280), (120, 280), (120, 320), (80, 320))


def make_track(
    track_id: int = 1,
    position_history: tuple = ((100, 100),) * 5,
    size_history: tuple = ((20, 20),) * 5,
    region: str = "left",
) -> TrackedObject:
    center = position_history[-1]
    w, h = size_history[-1]
    bbox = BoundingBox(x1=center[0] - w // 2, y1=center[1] - h // 2, x2=center[0] + w // 2, y2=center[1] + h // 2)
    return TrackedObject(
        track_id=track_id, class_id=2, class_name="car", confidence=0.9,
        bbox=bbox, center=center, region=region, position_history=position_history,
        size_history=size_history, direction="down", motion_status="moving",
        frames_since_seen=0,
    )


def make_filtered(motion_state: str = "MOVING", uncertain: bool = False) -> FilteredMotion:
    return FilteredMotion(
        track_id=1, velocity_x=0.0, velocity_y=5.0, speed=5.0,
        motion_state=motion_state, source="COMPENSATED" if not uncertain else "RAW_FALLBACK",
        uncertain=uncertain, reason=None if not uncertain else "RAW_FALLBACK",
        confirmation_frames=0,
    )


def make_prediction(current_center: tuple, predicted_center: tuple, valid: bool = True) -> TrajectoryPrediction:
    return TrajectoryPrediction(
        valid=valid, velocity_x=0.0, velocity_y=5.0, speed_px_per_frame=5.0,
        direction="down", current_center=current_center, predicted_center=predicted_center,
        observations_used=5, prediction_horizon_frames=10, uncertain=not valid,
    )


def make_intersection(intersects: bool = False, uncertain: bool = False) -> PathIntersectionResult:
    return PathIntersectionResult(
        valid=True, intersects=intersects, starts_inside=intersects, ends_inside=intersects,
        intersection_point=None, track_id=1, reason="test", uncertain=uncertain,
    )


def _closing_track() -> TrackedObject:
    # Bottom-y increasing (100 -> 120), area growing, position converging
    # on CORRIDOR_POLYGON's centroid -- a strong, multi-cue closing signal.
    positions = ((100, 100), (100, 105), (100, 110), (100, 115), (100, 120))
    sizes = ((20, 20), (22, 22), (24, 24), (26, 26), (28, 28))
    return make_track(position_history=positions, size_history=sizes)


@pytest.fixture
def proximity_estimator() -> RelativeProximityEstimator:
    return RelativeProximityEstimator(far_boundary=0.55, near_boundary=0.80)


@pytest.fixture
def estimator(proximity_estimator: RelativeProximityEstimator) -> ApproachEstimator:
    return ApproachEstimator(
        proximity_estimator=proximity_estimator,
        minimum_history_samples=5,
        minimum_approach_cues=2,
        approaching_confirmation_frames=3,
        ground_point_trend_threshold=0.5,
        scale_growth_threshold_fraction=0.02,
        corridor_distance_trend_threshold=0.5,
    )


# --- hard gate: requires confirmed MOVING -----------------------------


def test_stationary_motion_is_never_approaching_regardless_of_cues(estimator: ApproachEstimator) -> None:
    track = _closing_track()
    filtered = make_filtered(motion_state="STATIONARY")
    prediction = make_prediction((100, 120), (100, 170))
    intersection = make_intersection(intersects=True)

    for _ in range(5):
        result = estimator.estimate(
            track, filtered, prediction, intersection, FAR, CORRIDOR_POLYGON, FRAME_HEIGHT
        )

    assert result.state == NOT_APPROACHING
    assert result.uncertain is False
    assert result.evidence_flags == ()


def test_insufficient_history_is_never_approaching(estimator: ApproachEstimator) -> None:
    track = make_track(position_history=((100, 100), (100, 105)), size_history=((20, 20), (22, 22)))
    filtered = make_filtered()
    prediction = make_prediction((100, 105), (100, 155))

    result = estimator.estimate(
        track, filtered, prediction, None, FAR, CORRIDOR_POLYGON, FRAME_HEIGHT
    )

    assert result.state == NOT_APPROACHING
    assert result.confidence == 0.0


# --- confirmation streak: single-frame evidence is never enough ------


def test_single_frame_of_closing_evidence_is_not_approaching(estimator: ApproachEstimator) -> None:
    track = _closing_track()
    filtered = make_filtered()
    prediction = make_prediction((100, 120), (100, 170))
    intersection = make_intersection(intersects=True)

    result = estimator.estimate(
        track, filtered, prediction, intersection, FAR, CORRIDOR_POLYGON, FRAME_HEIGHT
    )

    assert result.state == NOT_APPROACHING  # streak = 1, needs 3


def test_two_consecutive_frames_still_not_enough(estimator: ApproachEstimator) -> None:
    track = _closing_track()
    filtered = make_filtered()
    prediction = make_prediction((100, 120), (100, 170))
    intersection = make_intersection(intersects=True)

    for _ in range(2):
        result = estimator.estimate(
            track, filtered, prediction, intersection, FAR, CORRIDOR_POLYGON, FRAME_HEIGHT
        )

    assert result.state == NOT_APPROACHING


def test_three_consecutive_closing_frames_confirms_approaching(estimator: ApproachEstimator) -> None:
    track = _closing_track()
    filtered = make_filtered()
    prediction = make_prediction((100, 120), (100, 170))
    intersection = make_intersection(intersects=True)

    for _ in range(3):
        result = estimator.estimate(
            track, filtered, prediction, intersection, FAR, CORRIDOR_POLYGON, FRAME_HEIGHT
        )

    assert result.state == APPROACHING
    assert result.uncertain is False
    assert len(result.evidence_flags) >= 2
    assert GROUND_POINT_CLOSING in result.evidence_flags
    assert SCALE_GROWTH in result.evidence_flags


# --- minimum 2 independent cues required -------------------------------


def test_single_cue_alone_never_confirms_approaching(estimator: ApproachEstimator) -> None:
    # Only SCALE_GROWTH true: bottom-y (center_y + height/2) held exactly
    # constant even as height grows, by shrinking center_y in lockstep --
    # isolates growth from the ground-point-trend cue, which otherwise
    # correlates with it (a taller box's bottom edge moves down even at
    # a fixed center). x held far from the corridor centroid throughout,
    # so corridor-distance barely moves -- well under threshold.
    positions = ((500, 500), (500, 499), (500, 498), (500, 497), (500, 496))
    sizes = ((20, 20), (22, 22), (24, 24), (26, 26), (28, 28))
    track = make_track(position_history=positions, size_history=sizes)
    filtered = make_filtered()
    prediction = make_prediction((500, 496), (500, 496))  # not moving toward corridor
    intersection = make_intersection(intersects=False)

    for _ in range(5):
        result = estimator.estimate(
            track, filtered, prediction, intersection, FAR, CORRIDOR_POLYGON, FRAME_HEIGHT
        )

    assert result.evidence_flags == (SCALE_GROWTH,)
    assert result.state == NOT_APPROACHING


# --- moving away never approaches --------------------------------------


def test_moving_away_from_corridor_never_approaches(estimator: ApproachEstimator) -> None:
    # Bottom-y decreasing (120 -> 100), area shrinking, position
    # diverging from the corridor centroid -- the opposite of closing.
    positions = ((100, 120), (100, 115), (100, 110), (100, 105), (100, 100))
    sizes = ((28, 28), (26, 26), (24, 24), (22, 22), (20, 20))
    track = make_track(position_history=positions, size_history=sizes)
    filtered = make_filtered()
    prediction = make_prediction((100, 100), (100, 50))
    intersection = make_intersection(intersects=False)

    for _ in range(5):
        result = estimator.estimate(
            track, filtered, prediction, intersection, FAR, CORRIDOR_POLYGON, FRAME_HEIGHT
        )

    assert result.state == NOT_APPROACHING


# --- uncertain resolved-motion source caps at APPROACHING_UNCERTAIN ---


def test_uncertain_source_confirms_uncertain_not_confident(estimator: ApproachEstimator) -> None:
    track = _closing_track()
    filtered = make_filtered(uncertain=True)
    prediction = make_prediction((100, 120), (100, 170))
    intersection = make_intersection(intersects=True)

    for _ in range(3):
        result = estimator.estimate(
            track, filtered, prediction, intersection, FAR, CORRIDOR_POLYGON, FRAME_HEIGHT
        )

    assert result.state == APPROACHING_UNCERTAIN
    assert result.uncertain is True


# --- de-confirmation ----------------------------------------------------


def test_confirmed_approaching_reverts_after_sustained_non_closing(estimator: ApproachEstimator) -> None:
    track = _closing_track()
    filtered = make_filtered()
    prediction = make_prediction((100, 120), (100, 170))
    intersection = make_intersection(intersects=True)
    for _ in range(3):
        result = estimator.estimate(
            track, filtered, prediction, intersection, FAR, CORRIDOR_POLYGON, FRAME_HEIGHT
        )
    assert result.state == APPROACHING

    # Now feed sustained non-closing evidence for the same track.
    still_track = make_track(position_history=((500, 500),) * 5, size_history=((20, 20),) * 5)
    still_prediction = make_prediction((500, 500), (500, 500))
    still_intersection = make_intersection(intersects=False)
    for _ in range(3):
        result = estimator.estimate(
            still_track, filtered, still_prediction, still_intersection, FAR,
            CORRIDOR_POLYGON, FRAME_HEIGHT,
        )

    assert result.state == NOT_APPROACHING


# --- estimate_for_tracks bulk method + state pruning -----------------


def test_estimate_for_tracks_skips_missing_filtered_or_prediction(estimator: ApproachEstimator) -> None:
    track = _closing_track()
    result = estimator.estimate_for_tracks(
        [track], {}, {}, {}, {1: FAR}, CORRIDOR_POLYGON, FRAME_HEIGHT
    )
    assert result == {}


def test_estimate_for_tracks_prunes_state_for_dropped_tracks(estimator: ApproachEstimator) -> None:
    track = _closing_track()
    filtered = {1: make_filtered()}
    predictions = {1: make_prediction((100, 120), (100, 170))}
    intersections = {1: make_intersection(intersects=True)}

    for _ in range(3):
        estimator.estimate_for_tracks(
            [track], filtered, predictions, intersections, {1: FAR}, CORRIDOR_POLYGON, FRAME_HEIGHT
        )
    assert 1 in estimator._states

    estimator.estimate_for_tracks([], {}, {}, {}, {}, CORRIDOR_POLYGON, FRAME_HEIGHT)
    assert 1 not in estimator._states


# --- constructor validation --------------------------------------------


def test_minimum_approach_cues_out_of_range_raises(proximity_estimator) -> None:
    with pytest.raises(ValueError):
        ApproachEstimator(
            proximity_estimator=proximity_estimator, minimum_history_samples=5,
            minimum_approach_cues=7, approaching_confirmation_frames=3,
            ground_point_trend_threshold=0.5, scale_growth_threshold_fraction=0.02,
            corridor_distance_trend_threshold=0.5,
        )


def test_non_positive_thresholds_raise(proximity_estimator) -> None:
    with pytest.raises(ValueError):
        ApproachEstimator(
            proximity_estimator=proximity_estimator, minimum_history_samples=5,
            minimum_approach_cues=2, approaching_confirmation_frames=3,
            ground_point_trend_threshold=0.0, scale_growth_threshold_fraction=0.02,
            corridor_distance_trend_threshold=0.5,
        )
