"""Unit tests for the real-world ("field") display redesign of
Visualizer.draw_tracked_objects / draw_region_lines / draw_attention_zone
-- the three calls main.py makes unconditionally (no --debug or
--validate-compensation flag needed), so this is what a real user sees by
default. No camera, model, or network required -- uses blank numpy
frames and synthetic TrackedObject/AttentionZone objects.
"""

import numpy as np
import pytest
from PIL import ImageFont

from src.models import BoundingBox, FilteredMotion, TrackedObject
from src.region_analyzer import AttentionZone
from src.visualizer import Visualizer, _load_field_font

FRAME_SIZE = 200


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


def make_tracked_object(
    track_id: int = 1,
    bbox: BoundingBox = BoundingBox(x1=50, y1=50, x2=100, y2=120),
    class_name: str = "person",
) -> TrackedObject:
    center = (
        bbox.x1 + (bbox.x2 - bbox.x1) // 2,
        bbox.y1 + (bbox.y2 - bbox.y1) // 2,
    )
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        bbox=bbox,
        center=center,
        region="center",
        position_history=(center,),
        size_history=((bbox.x2 - bbox.x1, bbox.y2 - bbox.y1),),
        direction="right",
        motion_status="moving",
        frames_since_seen=0,
    )


@pytest.fixture
def visualizer() -> Visualizer:
    return Visualizer("test-window")


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)


# --- draw_tracked_objects -------------------------------------------------


def test_draw_tracked_objects_draws_something(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    visualizer.draw_tracked_objects(blank_frame, [make_tracked_object()], {})
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_draw_tracked_objects_with_empty_list_draws_nothing(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    visualizer.draw_tracked_objects(blank_frame, [], {})
    assert np.array_equal(blank_frame, np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8))


def test_draw_tracked_objects_does_not_crash_at_top_left_edge(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    box = BoundingBox(x1=0, y1=0, x2=30, y2=30)
    visualizer.draw_tracked_objects(blank_frame, [make_tracked_object(bbox=box)], {})
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_draw_tracked_objects_does_not_crash_at_bottom_right_edge(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    box = BoundingBox(
        x1=FRAME_SIZE - 30, y1=FRAME_SIZE - 30, x2=FRAME_SIZE, y2=FRAME_SIZE
    )
    visualizer.draw_tracked_objects(blank_frame, [make_tracked_object(bbox=box)], {})
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_draw_tracked_objects_does_not_crash_with_tiny_box(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    box = BoundingBox(x1=90, y1=90, x2=94, y2=94)  # 4x4px
    visualizer.draw_tracked_objects(blank_frame, [make_tracked_object(bbox=box)], {})
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_draw_tracked_objects_with_multiple_objects_does_not_crash(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    objects = [
        make_tracked_object(track_id=1, bbox=BoundingBox(x1=10, y1=10, x2=50, y2=60)),
        make_tracked_object(track_id=2, bbox=BoundingBox(x1=100, y1=100, x2=150, y2=160)),
    ]
    visualizer.draw_tracked_objects(blank_frame, objects, {})
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


# --- FilteredMotion is authoritative over raw tracker motion ---------------
#
# Regression coverage for the raw-vs-filtered consistency bug: the default
# (always-on) overlay must never show a confident MOVING claim built from
# raw, uncompensated TrackedObject.motion_status/direction when the
# resolved+filtered pipeline (available whenever --debug ran) says
# otherwise. _draw_pil_text is monkeypatched to capture exactly what text
# would have been rendered, since asserting on rendered pixels can't
# distinguish "STATIONARY" from "MOVING" the way asserting on the actual
# label string can.


def _capture_pil_entries(monkeypatch) -> dict:
    captured: dict = {}

    def fake_draw_pil_text(frame, entries):
        captured["entries"] = entries

    monkeypatch.setattr(Visualizer, "_draw_pil_text", staticmethod(fake_draw_pil_text))
    return captured


def test_raw_moving_but_filtered_stationary_shows_stationary(
    visualizer: Visualizer, blank_frame: np.ndarray, monkeypatch
) -> None:
    captured = _capture_pil_entries(monkeypatch)
    # make_tracked_object()'s default motion_status/direction already read
    # "moving"/"right" -- the raw signal here disagrees with filtered.
    obj = make_tracked_object()
    filtered = make_filtered_motion(motion_state="STATIONARY", velocity_x=0.0, velocity_y=0.0)

    visualizer.draw_tracked_objects(blank_frame, [obj], {1: filtered})

    label_text = captured["entries"][0][1]
    assert "STATIONARY" in label_text
    assert "MOVING" not in label_text


def test_raw_moving_but_filtered_uncertain_shows_uncertain_not_moving(
    visualizer: Visualizer, blank_frame: np.ndarray, monkeypatch
) -> None:
    captured = _capture_pil_entries(monkeypatch)
    obj = make_tracked_object()
    filtered = make_filtered_motion(
        motion_state="UNCERTAIN", velocity_x=0.0, velocity_y=0.0,
        uncertain=True, reason="RAW_FALLBACK",
    )

    visualizer.draw_tracked_objects(blank_frame, [obj], {1: filtered})

    label_text = captured["entries"][0][1]
    assert "UNCERTAIN" in label_text
    assert "MOVING" not in label_text


def test_filtered_moving_shows_moving_state_and_velocity(
    visualizer: Visualizer, blank_frame: np.ndarray, monkeypatch
) -> None:
    captured = _capture_pil_entries(monkeypatch)
    obj = make_tracked_object()
    filtered = make_filtered_motion(motion_state="MOVING", velocity_x=8.0, velocity_y=1.0)

    visualizer.draw_tracked_objects(blank_frame, [obj], {1: filtered})

    label_text = captured["entries"][0][1]
    velocity_text = captured["entries"][1][1]
    assert "MOVING" in label_text
    assert "8.0" in velocity_text


def test_filtered_motion_unavailable_labels_raw_clearly(
    visualizer: Visualizer, blank_frame: np.ndarray, monkeypatch
) -> None:
    captured = _capture_pil_entries(monkeypatch)
    obj = make_tracked_object()  # raw motion_status="moving"

    visualizer.draw_tracked_objects(blank_frame, [obj], {})  # no entry for track_id 1

    label_text = captured["entries"][0][1]
    assert "RAW:" in label_text
    assert "unconfirmed" in captured["entries"][1][1]


def test_stationary_filtered_object_shows_zero_velocity(
    visualizer: Visualizer, blank_frame: np.ndarray, monkeypatch
) -> None:
    captured = _capture_pil_entries(monkeypatch)
    obj = make_tracked_object()
    filtered = make_filtered_motion(motion_state="STATIONARY", velocity_x=0.0, velocity_y=0.0)

    visualizer.draw_tracked_objects(blank_frame, [obj], {1: filtered})

    velocity_text = captured["entries"][1][1]
    assert "0.0,0.0" in velocity_text.replace(" ", "")


def test_class_confidence_track_id_and_box_still_displayed_with_filtered_motion(
    visualizer: Visualizer, blank_frame: np.ndarray, monkeypatch
) -> None:
    captured = _capture_pil_entries(monkeypatch)
    obj = make_tracked_object(track_id=7, class_name="bicycle")
    filtered = make_filtered_motion(motion_state="STATIONARY")

    visualizer.draw_tracked_objects(blank_frame, [obj], {7: filtered})

    label_text = captured["entries"][0][1]
    assert "bicycle" in label_text
    assert "ID 7" in label_text
    # The box itself (corner brackets) is drawn directly to pixels, not
    # via the captured PIL text entries -- confirm it still rendered.
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


# --- draw_region_lines / draw_attention_zone ------------------------------


def test_draw_region_lines_still_visibly_draws(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    visualizer.draw_region_lines(blank_frame, left_px=66, right_px=133)
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


def test_draw_attention_zone_still_visibly_draws(
    visualizer: Visualizer, blank_frame: np.ndarray
) -> None:
    zone = AttentionZone(x_min=50, y_min=40, x_max=150, y_max=160)
    visualizer.draw_attention_zone(blank_frame, zone)
    assert not np.array_equal(blank_frame, np.zeros_like(blank_frame))


# --- _load_field_font ------------------------------------------------------


def test_load_field_font_returns_a_usable_font() -> None:
    font = _load_field_font(15)
    # Must support getbbox (used to size label chips) without raising.
    left, top, right, bottom = font.getbbox("person | ID 1")
    assert right > left
    assert bottom > top


def test_load_field_font_falls_back_when_no_system_font_found(monkeypatch) -> None:
    monkeypatch.setattr("src.visualizer._FONT_PATH_CANDIDATES", ())
    _load_field_font.cache_clear()
    try:
        font = _load_field_font(15)
        assert isinstance(font, ImageFont.ImageFont | ImageFont.FreeTypeFont)
    finally:
        _load_field_font.cache_clear()
