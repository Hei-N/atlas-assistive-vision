"""Unit tests for src/motion/relative_proximity_estimator.py -- FAR/MID/
NEAR classification from a bounding box's bottom edge. Pure math, no
camera/model/network involved.
"""

import pytest

from src.motion.relative_proximity_estimator import (
    FAR,
    MID,
    NEAR,
    UNKNOWN,
    RelativeProximityEstimator,
    is_nearer,
)


@pytest.fixture
def estimator() -> RelativeProximityEstimator:
    return RelativeProximityEstimator(far_boundary=0.55, near_boundary=0.80)


# --- zone boundaries -------------------------------------------------


def test_at_or_below_far_boundary_is_far(estimator: RelativeProximityEstimator) -> None:
    assert estimator.estimate(bbox_bottom_y=100, frame_height=1000) == FAR  # 0.10
    assert estimator.estimate(bbox_bottom_y=550, frame_height=1000) == FAR  # exactly 0.55


def test_between_boundaries_is_mid(estimator: RelativeProximityEstimator) -> None:
    assert estimator.estimate(bbox_bottom_y=600, frame_height=1000) == MID  # 0.60
    assert estimator.estimate(bbox_bottom_y=799, frame_height=1000) == MID


def test_at_or_above_near_boundary_is_near(estimator: RelativeProximityEstimator) -> None:
    assert estimator.estimate(bbox_bottom_y=800, frame_height=1000) == NEAR  # exactly 0.80
    assert estimator.estimate(bbox_bottom_y=999, frame_height=1000) == NEAR


# --- UNKNOWN on invalid input -----------------------------------------


def test_zero_frame_height_is_unknown(estimator: RelativeProximityEstimator) -> None:
    assert estimator.estimate(bbox_bottom_y=100, frame_height=0) == UNKNOWN


def test_negative_frame_height_is_unknown(estimator: RelativeProximityEstimator) -> None:
    assert estimator.estimate(bbox_bottom_y=100, frame_height=-10) == UNKNOWN


def test_non_finite_bottom_y_is_unknown(estimator: RelativeProximityEstimator) -> None:
    assert estimator.estimate(bbox_bottom_y=float("nan"), frame_height=1000) == UNKNOWN
    assert estimator.estimate(bbox_bottom_y=float("inf"), frame_height=1000) == UNKNOWN


# --- never uses bounding-box size, only the bottom edge --------------


def test_result_depends_only_on_bottom_y_not_size() -> None:
    # A structural check: estimate() takes only bbox_bottom_y and
    # frame_height -- it cannot consult width/height/area even in
    # principle, matching "do not classify proximity using bounding-box
    # size alone."
    import inspect

    params = set(inspect.signature(RelativeProximityEstimator.estimate).parameters)
    assert not params & {"width", "height", "area", "bbox_width", "bbox_height"}


# --- is_nearer ordinal comparisons -----------------------------------


def test_is_nearer_orders_zones_correctly() -> None:
    assert is_nearer(NEAR, FAR) is True
    assert is_nearer(NEAR, MID) is True
    assert is_nearer(MID, FAR) is True
    assert is_nearer(FAR, NEAR) is False
    assert is_nearer(FAR, FAR) is False


def test_is_nearer_never_true_with_unknown() -> None:
    assert is_nearer(NEAR, UNKNOWN) is False
    assert is_nearer(UNKNOWN, FAR) is False
    assert is_nearer(UNKNOWN, UNKNOWN) is False


# --- estimate_for_object / estimate_for_tracks -------------------------


def test_estimate_for_tracks_returns_one_zone_per_object() -> None:
    from src.models import BoundingBox, TrackedObject

    def make_object(track_id: int, y2: int) -> TrackedObject:
        bbox = BoundingBox(x1=0, y1=0, x2=20, y2=y2)
        return TrackedObject(
            track_id=track_id, class_id=2, class_name="car", confidence=0.9,
            bbox=bbox, center=(10, y2 - 10), region="left", position_history=((10, y2 - 10),),
            size_history=((20, 20),), direction="right", motion_status="moving",
            frames_since_seen=0,
        )

    estimator = RelativeProximityEstimator(far_boundary=0.55, near_boundary=0.80)
    objects = [make_object(1, y2=100), make_object(2, y2=900)]

    result = estimator.estimate_for_tracks(objects, frame_height=1000)

    assert result == {1: FAR, 2: NEAR}


# --- constructor validation --------------------------------------------


def test_far_boundary_must_be_less_than_near_boundary() -> None:
    with pytest.raises(ValueError):
        RelativeProximityEstimator(far_boundary=0.80, near_boundary=0.55)


def test_equal_boundaries_raise() -> None:
    with pytest.raises(ValueError):
        RelativeProximityEstimator(far_boundary=0.5, near_boundary=0.5)


def test_boundaries_out_of_range_raise() -> None:
    with pytest.raises(ValueError):
        RelativeProximityEstimator(far_boundary=-0.1, near_boundary=0.8)
    with pytest.raises(ValueError):
        RelativeProximityEstimator(far_boundary=0.5, near_boundary=1.1)
