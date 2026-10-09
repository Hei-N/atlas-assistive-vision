"""Unit tests for the camera-motion-diagnostics visualization added to
Visualizer (Atlas Phase 4 validation hardening): draw_motion_estimate's
now-bounded arrow and non-finite safety, and draw_motion_legend. No
camera, model, or network required -- uses a blank numpy frame and
synthetic MotionEstimate objects.
"""

import math

import numpy as np
import pytest

from src.models import MotionEstimate
from src.visualizer import Visualizer

FRAME_SIZE = 200


def make_motion_estimate(
    valid: bool = True,
    dx: float = 5.0,
    dy: float = 0.0,
    confidence: float = 0.9,
    status: str = "VALID",
    reason_invalid: str | None = None,
) -> MotionEstimate:
    return MotionEstimate(
        valid=valid,
        dx=dx,
        dy=dy,
        rotation_degrees=0.0,
        scale=1.0,
        confidence=confidence,
        feature_count=200,
        tracked_feature_count=150,
        inlier_count=140,
        inlier_ratio=140 / 150,
        status=status,
        reason_invalid=reason_invalid,
        raw_dx=dx,
        raw_dy=dy,
        raw_rotation_degrees=0.0,
        transform_matrix=None,
        debug_prev_points=((10, 10),),
        debug_current_points=((15, 10),),
        debug_inlier_flags=(True,),
    )


@pytest.fixture
def visualizer() -> Visualizer:
    return Visualizer("test-window")


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)


def test_draw_motion_estimate_with_valid_estimate_draws_something(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    visualizer.draw_motion_estimate(blank_frame, make_motion_estimate())
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_draw_motion_estimate_with_none_draws_nothing(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    visualizer.draw_motion_estimate(blank_frame, None)
    assert np.array_equal(blank_frame, np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8))


def test_draw_motion_estimate_with_invalid_estimate_does_not_crash(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    invalid = make_motion_estimate(
        valid=False, status="UNAVAILABLE", reason_invalid="insufficient features detected"
    )
    visualizer.draw_motion_estimate(blank_frame, invalid)
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_draw_motion_estimate_with_extreme_translation_does_not_crash(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    huge = make_motion_estimate(dx=100_000.0, dy=-100_000.0)
    visualizer.draw_motion_estimate(blank_frame, huge)
    # No exception, and the text block still renders.
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_draw_motion_estimate_with_non_finite_dx_skips_arrow_safely(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    non_finite = make_motion_estimate(dx=math.nan, dy=math.inf)
    # Must not raise, and the rest of the method (text block) still runs.
    visualizer.draw_motion_estimate(blank_frame, non_finite)
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_draw_motion_legend_renders_without_crashing(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    visualizer.draw_motion_legend(blank_frame)
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_bounded_arrow_end_clamps_to_frame_when_center_near_edge() -> None:
    # Center near the right/bottom edge: even after the max-length scale,
    # the naive (unclamped) endpoint would fall outside the frame --
    # confirms the frame-boundary clamp (not just the max-length one).
    center = (FRAME_SIZE - 2, FRAME_SIZE - 2)
    end = Visualizer._bounded_arrow_end(
        center, dx=50.0, dy=50.0, frame_width=FRAME_SIZE, frame_height=FRAME_SIZE
    )
    assert 0 <= end[0] < FRAME_SIZE
    assert 0 <= end[1] < FRAME_SIZE
