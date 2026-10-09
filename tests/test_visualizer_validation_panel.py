"""Unit tests for Visualizer.draw_validation_panel / the resolved-motion
additions to the --validate-compensation per-object debug block. No
camera, model, or network required -- uses a blank numpy frame and
synthetic TrackedObject/PerObjectValidation/TrajectoryPrediction/
ValidationStats objects.
"""

import numpy as np
import pytest

from src.models import (
    BoundingBox,
    PerObjectValidation,
    TrackedObject,
    TrajectoryPrediction,
    ValidationStats,
)
from src.visualizer import Visualizer

FRAME_SIZE = 300


def make_tracked_object(track_id: int = 1, center: tuple = (150, 150)) -> TrackedObject:
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


def make_per_object_row(
    track_id: int = 1,
    status: str = "OBJECT-MOVING",
    resolved_velocity_x: float = 5.0,
    resolved_velocity_y: float = 0.0,
    motion_source: str = "COMPENSATED",
    prediction_uncertain: bool = False,
    motion_filter_state: str = "MOVING",
    motion_filter_reason: str | None = None,
) -> PerObjectValidation:
    return PerObjectValidation(
        track_id=track_id,
        class_name="person",
        raw_velocity_x=12.0,
        raw_velocity_y=0.0,
        compensated_velocity_x=5.0,
        compensated_velocity_y=0.0,
        raw_speed=12.0,
        compensated_speed=5.0,
        compensated_direction="right",
        camera_confidence=0.9,
        status=status,
        resolved_velocity_x=resolved_velocity_x,
        resolved_velocity_y=resolved_velocity_y,
        resolved_speed=(resolved_velocity_x**2 + resolved_velocity_y**2) ** 0.5,
        motion_source=motion_source,
        prediction_uncertain=prediction_uncertain,
        raw_intersects_corridor=True,
        resolved_intersects_corridor=False,
        filtered_velocity_x=resolved_velocity_x,
        filtered_velocity_y=resolved_velocity_y,
        filtered_speed=(resolved_velocity_x**2 + resolved_velocity_y**2) ** 0.5,
        motion_filter_state=motion_filter_state,
        motion_filter_reason=motion_filter_reason,
        motion_confirmation_frames=0,
    )


def make_resolved_prediction(
    current_center: tuple = (150, 150),
    predicted_center: tuple = (200, 150),
    direction: str = "right",
    valid: bool = True,
) -> TrajectoryPrediction:
    return TrajectoryPrediction(
        valid=valid,
        velocity_x=5.0,
        velocity_y=0.0,
        speed_px_per_frame=5.0,
        direction=direction,
        current_center=current_center,
        predicted_center=predicted_center,
        observations_used=5,
        prediction_horizon_frames=10,
        uncertain=False,
    )


def make_validation_stats() -> ValidationStats:
    return ValidationStats(
        active_track_count=1,
        camera_dx=2.0,
        camera_dy=0.0,
        camera_confidence=0.9,
        camera_status="VALID",
        feature_count=120,
        match_count=100,
        inlier_count=90,
        stationary_count=0,
        moving_count=1,
        camera_dominated_count=0,
        uncertain_count=0,
        average_raw_speed=12.0,
        average_compensated_speed=5.0,
        reduction_percent=58.3,
    )


@pytest.fixture
def visualizer() -> Visualizer:
    return Visualizer("test-window")


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)


def test_draw_validation_panel_renders_without_crashing(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    obj = make_tracked_object()
    visualizer.draw_validation_panel(
        blank_frame,
        [obj],
        make_validation_stats(),
        [make_per_object_row()],
        {1: make_resolved_prediction()},
    )
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_draw_validation_panel_handles_missing_resolved_prediction(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    obj = make_tracked_object()
    # No entry for track_id 1 in resolved_predictions -- must not crash.
    visualizer.draw_validation_panel(
        blank_frame, [obj], make_validation_stats(), [make_per_object_row()], {}
    )
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_draw_validation_panel_skips_resolved_arrow_when_stationary(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    obj = make_tracked_object()
    stationary_prediction = make_resolved_prediction(direction="stationary")
    # Must not raise even though the resolved arrow is skipped.
    visualizer.draw_validation_panel(
        blank_frame, [obj], make_validation_stats(), [make_per_object_row()],
        {1: stationary_prediction},
    )
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_draw_validation_panel_skips_resolved_arrow_when_invalid(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    obj = make_tracked_object()
    invalid_prediction = make_resolved_prediction(valid=False)
    visualizer.draw_validation_panel(
        blank_frame, [obj], make_validation_stats(), [make_per_object_row()],
        {1: invalid_prediction},
    )
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_draw_validation_panel_with_no_tracked_objects_still_draws_summary(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    visualizer.draw_validation_panel(blank_frame, [], make_validation_stats(), [], {})
    # Summary panel alone still draws something.
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))
