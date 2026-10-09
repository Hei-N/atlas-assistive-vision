"""Unit tests for ObjectTracker. No camera, model, or network required."""

import pytest

from src.models import BoundingBox, Detection
from src.object_tracker import ObjectTracker


def make_detection(
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.9,
    x1: int = 0,
    y1: int = 0,
    x2: int = 20,
    y2: int = 40,
    region: str = "center",
) -> Detection:
    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        center=bbox.center,
        region=region,
    )


@pytest.fixture
def tracker() -> ObjectTracker:
    return ObjectTracker(
        max_match_distance_px=50,
        max_disappeared_frames=2,
        min_match_iou=0.3,
        history_length=5,
        stationary_threshold_px=4,
        size_change_threshold_fraction=0.05,
    )


def test_new_detection_gets_sequential_track_id(tracker: ObjectTracker) -> None:
    result = tracker.update([make_detection()])
    assert result[0].track_id == 1

    result = tracker.update([make_detection(x1=200, y1=200, x2=220, y2=240)])
    assert result[0].track_id == 2


def test_same_object_small_movement_keeps_id(tracker: ObjectTracker) -> None:
    first = tracker.update([make_detection(x1=0, y1=0, x2=20, y2=40)])
    second = tracker.update([make_detection(x1=2, y1=2, x2=22, y2=42)])
    assert first[0].track_id == second[0].track_id


def test_two_same_class_objects_far_apart_get_distinct_ids(
    tracker: ObjectTracker,
) -> None:
    det_a = make_detection(x1=0, y1=0, x2=20, y2=40)
    det_b = make_detection(x1=500, y1=500, x2=520, y2=540)
    first = tracker.update([det_a, det_b])
    assert first[0].track_id != first[1].track_id

    det_a2 = make_detection(x1=3, y1=3, x2=23, y2=43)
    det_b2 = make_detection(x1=503, y1=503, x2=523, y2=543)
    second = tracker.update([det_a2, det_b2])
    assert second[0].track_id == first[0].track_id
    assert second[1].track_id == first[1].track_id


def test_different_class_objects_never_matched(tracker: ObjectTracker) -> None:
    person = make_detection(class_id=0, class_name="person", x1=0, y1=0, x2=20, y2=40)
    first = tracker.update([person])
    person_id = first[0].track_id

    car_same_spot = make_detection(
        class_id=2, class_name="car", x1=0, y1=0, x2=20, y2=40
    )
    second = tracker.update([car_same_spot])
    assert second[0].track_id != person_id


def test_track_dropped_after_max_disappeared_frames(tracker: ObjectTracker) -> None:
    det = make_detection()
    first_id = tracker.update([det])[0].track_id

    tracker.update([])  # frames_since_seen = 1
    tracker.update([])  # frames_since_seen = 2, still within limit
    tracker.update([])  # frames_since_seen = 3, exceeds max_disappeared_frames=2

    reappeared_id = tracker.update([det])[0].track_id
    assert reappeared_id != first_id


def test_track_survives_gap_within_limit(tracker: ObjectTracker) -> None:
    det = make_detection()
    first_id = tracker.update([det])[0].track_id

    tracker.update([])  # frames_since_seen = 1, still within limit of 2

    reappeared_id = tracker.update([det])[0].track_id
    assert reappeared_id == first_id


def test_motion_status_approaching_when_box_grows(tracker: ObjectTracker) -> None:
    tracker.update([make_detection(x1=0, y1=0, x2=20, y2=40)])
    result = tracker.update([make_detection(x1=0, y1=0, x2=25, y2=50)])
    assert result[0].motion_status == "approaching"


def test_motion_status_moving_away_when_box_shrinks(tracker: ObjectTracker) -> None:
    tracker.update([make_detection(x1=0, y1=0, x2=25, y2=50)])
    result = tracker.update([make_detection(x1=0, y1=0, x2=20, y2=40)])
    assert result[0].motion_status == "moving away"


def test_motion_status_stationary_when_unchanged(tracker: ObjectTracker) -> None:
    tracker.update([make_detection(x1=0, y1=0, x2=20, y2=40)])
    result = tracker.update([make_detection(x1=0, y1=0, x2=20, y2=40)])
    assert result[0].motion_status == "stationary"


@pytest.mark.parametrize(
    ("second_box", "expected_direction"),
    [
        ((30, 0, 50, 40), "right"),
        ((-30, 0, -10, 40), "left"),
        ((0, -30, 20, 10), "up"),
        ((0, 30, 20, 70), "down"),
        ((30, 30, 50, 70), "down-right"),
        ((1, 1, 21, 41), "stationary"),
    ],
)
def test_direction_reflects_movement(
    tracker: ObjectTracker, second_box: tuple, expected_direction: str
) -> None:
    tracker.update([make_detection(x1=0, y1=0, x2=20, y2=40)])
    x1, y1, x2, y2 = second_box
    result = tracker.update([make_detection(x1=x1, y1=y1, x2=x2, y2=y2)])
    assert result[0].direction == expected_direction


def test_position_and_size_history_capped_and_grow() -> None:
    small_history_tracker = ObjectTracker(
        max_match_distance_px=1000,
        max_disappeared_frames=2,
        min_match_iou=0.3,
        history_length=3,
        stationary_threshold_px=1,
        size_change_threshold_fraction=0.05,
    )

    centers = []
    for i in range(4):
        det = make_detection(x1=i * 10, y1=0, x2=i * 10 + 20, y2=40)
        result = small_history_tracker.update([det])
        centers.append(det.center)

    assert len(result[0].position_history) == 3
    assert len(result[0].size_history) == 3
    # Oldest sample (from the very first frame) should have been evicted.
    assert result[0].position_history[0] == centers[1]
    assert result[0].position_history[-1] == centers[3]


def test_iou_allows_match_despite_distance_exceeding_threshold(
    tracker: ObjectTracker,
) -> None:
    # A large box that shifts by more than max_match_distance_px (50) but
    # still overlaps its previous position heavily (high IoU) should keep
    # its ID -- this is the exact scenario that causes ID churn on
    # high-resolution frames, where a small on-screen movement covers many
    # more raw pixels than on a low-resolution frame.
    first = tracker.update([make_detection(x1=0, y1=0, x2=200, y2=200)])
    second = tracker.update([make_detection(x1=60, y1=0, x2=260, y2=200)])
    assert second[0].track_id == first[0].track_id


def test_no_match_when_both_distance_and_iou_fail(tracker: ObjectTracker) -> None:
    first = tracker.update([make_detection(x1=0, y1=0, x2=200, y2=200)])
    second = tracker.update([make_detection(x1=400, y1=400, x2=600, y2=600)])
    assert second[0].track_id != first[0].track_id


def test_max_match_distance_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ObjectTracker(
            max_match_distance_px=0,
            max_disappeared_frames=1,
            min_match_iou=0.3,
            history_length=2,
            stationary_threshold_px=1,
            size_change_threshold_fraction=0.05,
        )


@pytest.mark.parametrize("invalid_iou", [0.0, -0.1, 1.1])
def test_min_match_iou_must_be_within_valid_range(invalid_iou: float) -> None:
    with pytest.raises(ValueError):
        ObjectTracker(
            max_match_distance_px=50,
            max_disappeared_frames=1,
            min_match_iou=invalid_iou,
            history_length=2,
            stationary_threshold_px=1,
            size_change_threshold_fraction=0.05,
        )


def test_history_length_must_be_at_least_two() -> None:
    with pytest.raises(ValueError):
        ObjectTracker(
            max_match_distance_px=50,
            max_disappeared_frames=1,
            min_match_iou=0.3,
            history_length=1,
            stationary_threshold_px=1,
            size_change_threshold_fraction=0.05,
        )
