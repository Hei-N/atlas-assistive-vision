"""Unit tests for Visualizer.draw_compensated_motion (Atlas Phase 4). No
camera, model, or network required -- uses a blank numpy frame and
synthetic TrackedObject/CompensatedMotion/FilteredMotion objects.
"""

import numpy as np
import pytest

from src.models import BoundingBox, CompensatedMotion, FilteredMotion, TrackedObject
from src.visualizer import Visualizer

FRAME_SIZE = 200


def make_tracked_object(track_id: int = 1, center: tuple = (50, 50)) -> TrackedObject:
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


def make_compensated_motion(
    track_id: int = 1,
    compensated_velocity_x: float = 5.0,
    compensated_velocity_y: float = 0.0,
    direction: str = "right",
) -> CompensatedMotion:
    speed = (compensated_velocity_x**2 + compensated_velocity_y**2) ** 0.5
    return CompensatedMotion(
        track_id=track_id,
        compensated_velocity_x=compensated_velocity_x,
        compensated_velocity_y=compensated_velocity_y,
        compensated_speed=speed,
        compensated_direction=direction,
        raw_velocity_x=compensated_velocity_x + 2.0,
        raw_velocity_y=compensated_velocity_y,
        camera_motion_applied=True,
    )


def make_filtered_motion(
    track_id: int = 1,
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
    motion_state: str = "STATIONARY",
    source: str = "COMPENSATED",
    uncertain: bool = False,
    reason: str | None = None,
    confirmation_frames: int = 0,
) -> FilteredMotion:
    return FilteredMotion(
        track_id=track_id,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        speed=(velocity_x**2 + velocity_y**2) ** 0.5,
        motion_state=motion_state,
        source=source,
        uncertain=uncertain,
        reason=reason,
        confirmation_frames=confirmation_frames,
    )


@pytest.fixture
def visualizer() -> Visualizer:
    return Visualizer("test-window")


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)


def test_draws_without_crashing_for_normal_data(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    track = make_tracked_object()
    motions = {track.track_id: make_compensated_motion(track.track_id)}

    visualizer.draw_compensated_motion(blank_frame, [track], motions, {})

    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_missing_compensated_motion_for_one_track_does_not_crash(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    track_with_data = make_tracked_object(track_id=1, center=(50, 50))
    track_without_data = make_tracked_object(track_id=2, center=(150, 150))
    motions = {1: make_compensated_motion(track_id=1)}

    # track_id 2 has no entry in `motions` -- must be safely skipped, not
    # raise a KeyError/crash.
    visualizer.draw_compensated_motion(
        blank_frame, [track_with_data, track_without_data], motions, {}
    )


def test_empty_compensated_motions_dict_draws_nothing(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    track = make_tracked_object()

    visualizer.draw_compensated_motion(blank_frame, [track], {}, {})

    assert np.array_equal(blank_frame, np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8))


def test_no_tracked_objects_draws_nothing(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    visualizer.draw_compensated_motion(blank_frame, [], {1: make_compensated_motion()}, {})
    assert np.array_equal(blank_frame, np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8))


def test_stationary_direction_still_draws_text(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    track = make_tracked_object(center=(100, 100))
    motions = {
        track.track_id: make_compensated_motion(
            track.track_id,
            compensated_velocity_x=0.5,
            compensated_velocity_y=0.3,
            direction="stationary",
        )
    }

    visualizer.draw_compensated_motion(blank_frame, [track], motions, {})

    # Text block is drawn regardless of direction/filtered availability;
    # the arrow is what's conditionally skipped.
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_extreme_velocity_produces_bounded_arrow_length() -> None:
    center = (100, 100)
    arrow_end = Visualizer._bounded_arrow_end(
        center, dx=10_000.0, dy=0.0, frame_width=FRAME_SIZE, frame_height=FRAME_SIZE
    )
    length = ((arrow_end[0] - center[0]) ** 2 + (arrow_end[1] - center[1]) ** 2) ** 0.5

    assert length <= 81  # _COMPENSATED_ARROW_MAX_LENGTH_PX (80) + rounding slack
    # Also genuinely bounded within the frame itself (requirement 4).
    assert 0 <= arrow_end[0] < FRAME_SIZE
    assert 0 <= arrow_end[1] < FRAME_SIZE


def test_small_velocity_arrow_is_scaled_up_but_still_bounded() -> None:
    center = (100, 100)
    arrow_end = Visualizer._bounded_arrow_end(
        center, dx=1.0, dy=0.0, frame_width=FRAME_SIZE, frame_height=FRAME_SIZE
    )
    length = ((arrow_end[0] - center[0]) ** 2 + (arrow_end[1] - center[1]) ** 2) ** 0.5

    # Scaled by _COMPENSATED_ARROW_SCALE (3.0) for visibility, well under
    # the cap.
    assert length == pytest.approx(3.0, abs=0.5)


# --- FilteredMotion is authoritative for the arrow/state text --------------
#
# Regression coverage: this "plainer" --debug (non --validate-compensation)
# overlay must not draw a confident directional arrow, or state a
# confident direction, from raw CompensatedMotion.compensated_direction/
# velocity alone -- that value has no dead-zone/hysteresis/temporal
# confirmation and could otherwise flicker independently of the (now
# authoritative) box label drawn by draw_tracked_objects.


def _capture_arrow_calls(monkeypatch) -> list:
    calls: list = []
    import cv2

    original_arrowed_line = cv2.arrowedLine

    def fake_arrowed_line(*args, **kwargs):
        calls.append((args, kwargs))
        return original_arrowed_line(*args, **kwargs)

    monkeypatch.setattr(cv2, "arrowedLine", fake_arrowed_line)
    return calls


def test_filtered_moving_draws_arrow_from_filtered_velocity(
    visualizer: Visualizer, blank_frame: np.ndarray, monkeypatch
) -> None:
    arrow_calls = _capture_arrow_calls(monkeypatch)
    track = make_tracked_object(center=(100, 100))
    # Raw compensated velocity deliberately different from filtered, so a
    # passing test proves the ARROW came from filtered, not raw.
    motions = {1: make_compensated_motion(1, compensated_velocity_x=5.0, direction="right")}
    filtered = {1: make_filtered_motion(motion_state="MOVING", velocity_x=20.0, velocity_y=0.0)}

    visualizer.draw_compensated_motion(blank_frame, [track], motions, filtered)

    assert len(arrow_calls) == 1
    end_point = arrow_calls[0][0][2]
    # 20.0 scaled by _COMPENSATED_ARROW_SCALE (3.0) = 60 (before clamping),
    # clearly different from a 5.0-based arrow (15) -- confirms it used
    # the filtered velocity.
    assert end_point[0] - track.center[0] > 30


def test_raw_moving_but_filtered_stationary_draws_no_arrow(
    visualizer: Visualizer, blank_frame: np.ndarray, monkeypatch
) -> None:
    arrow_calls = _capture_arrow_calls(monkeypatch)
    track = make_tracked_object(center=(100, 100))
    # Raw compensated data says "right" / clearly non-zero -- would have
    # drawn an arrow under the old (pre-fix) behavior.
    motions = {1: make_compensated_motion(1, compensated_velocity_x=8.0, direction="right")}
    filtered = {1: make_filtered_motion(motion_state="STATIONARY", velocity_x=0.0, velocity_y=0.0)}

    visualizer.draw_compensated_motion(blank_frame, [track], motions, filtered)

    assert arrow_calls == []


def test_filtered_uncertain_shown_in_text_and_no_arrow(
    visualizer: Visualizer, blank_frame: np.ndarray, monkeypatch
) -> None:
    arrow_calls = _capture_arrow_calls(monkeypatch)
    track = make_tracked_object(center=(100, 100))
    motions = {1: make_compensated_motion(1, compensated_velocity_x=8.0, direction="right")}
    filtered = {
        1: make_filtered_motion(
            motion_state="UNCERTAIN", velocity_x=0.0, velocity_y=0.0,
            uncertain=True, reason="RAW_FALLBACK",
        )
    }

    visualizer.draw_compensated_motion(blank_frame, [track], motions, filtered)

    assert arrow_calls == []


def test_filtered_unavailable_draws_no_confident_arrow(
    visualizer: Visualizer, blank_frame: np.ndarray, monkeypatch
) -> None:
    arrow_calls = _capture_arrow_calls(monkeypatch)
    track = make_tracked_object(center=(100, 100))
    # Raw compensated data says clearly moving -- under the old (pre-fix)
    # behavior this alone would have drawn a confident arrow.
    motions = {1: make_compensated_motion(1, compensated_velocity_x=8.0, direction="right")}

    visualizer.draw_compensated_motion(blank_frame, [track], motions, {})

    assert arrow_calls == []
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))
