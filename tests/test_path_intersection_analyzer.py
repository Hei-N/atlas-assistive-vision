"""Unit tests for PathIntersectionAnalyzer. No camera, model, or network
required -- all inputs are synthetic TrajectoryPrediction objects.
"""

import pytest

from src.models import TrajectoryPrediction
from src.path_intersection_analyzer import PathIntersectionAnalyzer

FRAME_WIDTH = 100
FRAME_HEIGHT = 100


def make_prediction(
    valid: bool = True,
    current_center: tuple = (0, 0),
    predicted_center: tuple = (0, 0),
    uncertain: bool = False,
) -> TrajectoryPrediction:
    return TrajectoryPrediction(
        valid=valid,
        velocity_x=0.0,
        velocity_y=0.0,
        speed_px_per_frame=0.0,
        direction="right",
        current_center=current_center,
        predicted_center=predicted_center,
        observations_used=5,
        prediction_horizon_frames=10,
        uncertain=uncertain,
    )


@pytest.fixture
def analyzer() -> PathIntersectionAnalyzer:
    # Bottom edge x:[20,80] at y=100; top edge x:[40,60] at y=50, on a
    # 100x100 frame -- a symmetric trapezoid, easy to hand-verify.
    return PathIntersectionAnalyzer(
        corridor_bottom_left_x=0.2,
        corridor_bottom_right_x=0.8,
        corridor_bottom_y=1.0,
        corridor_top_left_x=0.4,
        corridor_top_right_x=0.6,
        corridor_top_y=0.5,
    )


def test_build_corridor_polygon_vertices(analyzer: PathIntersectionAnalyzer) -> None:
    polygon = analyzer.build_corridor_polygon(FRAME_WIDTH, FRAME_HEIGHT)
    assert polygon == ((20, 100), (80, 100), (60, 50), (40, 50))


def test_left_to_right_trajectory_entering_corridor(
    analyzer: PathIntersectionAnalyzer,
) -> None:
    # At y=70, the corridor spans x:[32,68].
    prediction = make_prediction(current_center=(0, 70), predicted_center=(50, 70))
    result = analyzer.analyze(1, prediction, FRAME_WIDTH, FRAME_HEIGHT)

    assert result.valid
    assert result.intersects
    assert not result.starts_inside
    assert result.ends_inside
    assert result.intersection_point == (32, 70)


def test_right_to_left_trajectory_entering_corridor(
    analyzer: PathIntersectionAnalyzer,
) -> None:
    prediction = make_prediction(current_center=(100, 70), predicted_center=(50, 70))
    result = analyzer.analyze(1, prediction, FRAME_WIDTH, FRAME_HEIGHT)

    assert result.valid
    assert result.intersects
    assert not result.starts_inside
    assert result.ends_inside
    assert result.intersection_point == (68, 70)


def test_trajectory_fully_outside_corridor(analyzer: PathIntersectionAnalyzer) -> None:
    # Above the corridor's top (y=50); a horizontal line at y=10 never
    # enters the corridor's vertical range.
    prediction = make_prediction(current_center=(0, 10), predicted_center=(100, 10))
    result = analyzer.analyze(1, prediction, FRAME_WIDTH, FRAME_HEIGHT)

    assert result.valid
    assert not result.intersects
    assert not result.starts_inside
    assert not result.ends_inside
    assert result.intersection_point is None


def test_trajectory_starting_inside_is_detected(
    analyzer: PathIntersectionAnalyzer,
) -> None:
    # At y=90, the corridor spans x:[24,76]; (50,90) is inside.
    prediction = make_prediction(current_center=(50, 90), predicted_center=(200, 90))
    result = analyzer.analyze(1, prediction, FRAME_WIDTH, FRAME_HEIGHT)

    assert result.valid
    assert result.starts_inside
    assert result.intersects
    assert "starts inside" in result.reason


def test_trajectory_ending_inside_is_detected(
    analyzer: PathIntersectionAnalyzer,
) -> None:
    prediction = make_prediction(current_center=(200, 90), predicted_center=(50, 90))
    result = analyzer.analyze(1, prediction, FRAME_WIDTH, FRAME_HEIGHT)

    assert result.valid
    assert result.ends_inside
    assert not result.starts_inside
    assert result.intersects
    assert "enters corridor" in result.reason


def test_trajectory_crossing_only_the_boundary_is_detected(
    analyzer: PathIntersectionAnalyzer,
) -> None:
    # Both endpoints are outside the corridor (checked below), but the
    # straight line between them passes through the interior (midpoint
    # (50, 75) is inside: at y=75 the corridor spans x:[30,70]).
    prediction = make_prediction(current_center=(10, 100), predicted_center=(90, 50))
    result = analyzer.analyze(1, prediction, FRAME_WIDTH, FRAME_HEIGHT)

    assert result.valid
    assert not result.starts_inside
    assert not result.ends_inside
    assert result.intersects
    assert result.intersection_point is not None
    assert result.reason == "segment crosses corridor boundary"


def test_invalid_trajectory_does_not_produce_false_safe_result(
    analyzer: PathIntersectionAnalyzer,
) -> None:
    prediction = make_prediction(valid=False)
    result = analyzer.analyze(1, prediction, FRAME_WIDTH, FRAME_HEIGHT)

    assert not result.valid
    assert not result.intersects
    assert not result.starts_inside
    assert not result.ends_inside
    assert result.intersection_point is None
    assert "invalid" in result.reason
    assert "not evaluated" in result.reason
    assert result.uncertain


def test_uncertain_is_always_true_for_invalid_prediction_even_if_flagged_false(
    analyzer: PathIntersectionAnalyzer,
) -> None:
    # An invalid prediction is never a confident result, regardless of
    # what its own (unused, since invalid) uncertain flag says.
    prediction = make_prediction(valid=False, uncertain=False)
    result = analyzer.analyze(1, prediction, FRAME_WIDTH, FRAME_HEIGHT)
    assert result.uncertain


def test_uncertain_propagates_false_from_a_confident_valid_prediction(
    analyzer: PathIntersectionAnalyzer,
) -> None:
    prediction = make_prediction(
        current_center=(0, 10), predicted_center=(100, 10), uncertain=False
    )
    result = analyzer.analyze(1, prediction, FRAME_WIDTH, FRAME_HEIGHT)
    assert result.valid
    assert not result.uncertain


def test_uncertain_propagates_true_from_a_low_confidence_prediction(
    analyzer: PathIntersectionAnalyzer,
) -> None:
    # Same geometry as the fully-outside-corridor test above (intersects
    # False), but built from an uncertain prediction -- this is exactly
    # the "no confident SAFE result" case: intersects=False here must NOT
    # be read as a confident safe outcome.
    prediction = make_prediction(
        current_center=(0, 10), predicted_center=(100, 10), uncertain=True
    )
    result = analyzer.analyze(1, prediction, FRAME_WIDTH, FRAME_HEIGHT)
    assert result.valid
    assert not result.intersects
    assert result.uncertain


def test_corridor_scales_correctly_across_resolutions(
    analyzer: PathIntersectionAnalyzer,
) -> None:
    small = analyzer.build_corridor_polygon(100, 100)
    large = analyzer.build_corridor_polygon(1000, 1000)

    assert small == ((20, 100), (80, 100), (60, 50), (40, 50))
    assert large == ((200, 1000), (800, 1000), (600, 500), (400, 500))
    for (sx, sy), (lx, ly) in zip(small, large):
        assert lx == sx * 10
        assert ly == sy * 10


def test_endpoint_exactly_on_boundary_handled_consistently(
    analyzer: PathIntersectionAnalyzer,
) -> None:
    # The predicted_center lands exactly on the corridor's left boundary
    # edge (the same point computed as the intersection in the
    # left-to-right test above) rather than clearly past it.
    prediction = make_prediction(current_center=(0, 70), predicted_center=(32, 70))
    result = analyzer.analyze(1, prediction, FRAME_WIDTH, FRAME_HEIGHT)

    assert result.valid
    assert not result.starts_inside
    assert result.ends_inside
    assert result.intersects
    assert result.intersection_point == (32, 70)


def test_track_id_is_propagated(analyzer: PathIntersectionAnalyzer) -> None:
    prediction = make_prediction(current_center=(0, 70), predicted_center=(50, 70))
    result = analyzer.analyze(42, prediction, FRAME_WIDTH, FRAME_HEIGHT)
    assert result.track_id == 42


def test_bottom_left_must_be_less_than_bottom_right() -> None:
    with pytest.raises(ValueError):
        PathIntersectionAnalyzer(
            corridor_bottom_left_x=0.8,
            corridor_bottom_right_x=0.2,
            corridor_bottom_y=1.0,
            corridor_top_left_x=0.4,
            corridor_top_right_x=0.6,
            corridor_top_y=0.5,
        )


def test_top_left_must_be_less_than_top_right() -> None:
    with pytest.raises(ValueError):
        PathIntersectionAnalyzer(
            corridor_bottom_left_x=0.2,
            corridor_bottom_right_x=0.8,
            corridor_bottom_y=1.0,
            corridor_top_left_x=0.6,
            corridor_top_right_x=0.4,
            corridor_top_y=0.5,
        )


def test_top_y_must_be_less_than_bottom_y() -> None:
    with pytest.raises(ValueError):
        PathIntersectionAnalyzer(
            corridor_bottom_left_x=0.2,
            corridor_bottom_right_x=0.8,
            corridor_bottom_y=0.5,
            corridor_top_left_x=0.4,
            corridor_top_right_x=0.6,
            corridor_top_y=0.5,
        )


def test_fraction_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        PathIntersectionAnalyzer(
            corridor_bottom_left_x=0.2,
            corridor_bottom_right_x=1.5,
            corridor_bottom_y=1.0,
            corridor_top_left_x=0.4,
            corridor_top_right_x=0.6,
            corridor_top_y=0.5,
        )
