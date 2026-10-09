"""Unit tests for src/audio/scene_summarizer.py -- combines related
per-frame candidate hazard AudioEvents into fewer, concise spoken
summaries. Rebuckets by REGION ONLY; within a region, only the highest
hazard level present survives, and only levels 1-2 (INFORMATIONAL) are
ever merged into a named list -- levels 3-5 (WARNING) always speak a
single event, never a blended sentence. No camera, model, network, or
subprocess involved -- pure logic over synthetic AudioEvent/TrackedObject
data.
"""

import pytest

from src.audio.scene_summarizer import SCENE_SUMMARY, SceneSummarizer
from src.models import AudioEvent, BoundingBox, TrackedObject


def make_tracked_object(
    track_id: int, class_name: str = "car", region: str = "left"
) -> TrackedObject:
    bbox = BoundingBox(x1=0, y1=0, x2=20, y2=20)
    return TrackedObject(
        track_id=track_id,
        class_id=2,
        class_name=class_name,
        confidence=0.9,
        bbox=bbox,
        center=(10, 10),
        region=region,
        position_history=((10, 10),),
        size_history=((20, 20),),
        direction="right",
        motion_status="moving",
        frames_since_seen=0,
    )


def make_event(
    track_id: int,
    level: str = "LEVEL_1_MOVING_FAR",
    priority: str = "INFORMATIONAL",
    region: str = "left",
    message: str | None = None,
    created_at: float = 0.0,
    delivery_profile: str = "level_1",
) -> AudioEvent:
    if message is None:
        message = f"placeholder {level} {track_id}"
    return AudioEvent(
        key=f"{level}:{track_id}:{region}",
        event_type=level,
        priority=priority,
        message=message,
        track_id=track_id,
        created_at=created_at,
        delivery_profile=delivery_profile,
    )


def make_summarizer(
    enabled: bool = True,
    minimum_events: int = 2,
    max_objects_named: int = 3,
    prefer_single_summary: bool = True,
    fallback_message: str = "Multiple hazards ahead. Please wait.",
) -> SceneSummarizer:
    return SceneSummarizer(
        enabled=enabled,
        minimum_events=minimum_events,
        max_objects_named=max_objects_named,
        same_frame_only=True,
        prefer_single_summary=prefer_single_summary,
        fallback_message=fallback_message,
    )


# --- A: Level-1/2 named-list merging ---------------------------------


def test_two_level1_different_classes_merge() -> None:
    summarizer = make_summarizer()
    objects = [
        make_tracked_object(1, class_name="car", region="left"),
        make_tracked_object(2, class_name="bicycle", region="left"),
    ]
    events = [make_event(1, region="left"), make_event(2, region="left")]

    result = summarizer.summarize(events, objects, timestamp=0.0)

    assert len(result) == 1
    assert result[0].message == "Vehicle and bicycle moving from the left."
    assert result[0].event_type == SCENE_SUMMARY


def test_two_level1_same_class_uses_count_wording() -> None:
    summarizer = make_summarizer()
    objects = [
        make_tracked_object(1, class_name="car", region="left"),
        make_tracked_object(2, class_name="car", region="left"),
    ]
    events = [make_event(1, region="left"), make_event(2, region="left")]

    result = summarizer.summarize(events, objects, timestamp=0.0)

    assert result[0].message == "2 vehicles moving from the left."


def test_two_level2_merge_with_on_direction_wording() -> None:
    summarizer = make_summarizer()
    objects = [
        make_tracked_object(1, class_name="car", region="left"),
        make_tracked_object(2, class_name="bicycle", region="left"),
    ]
    events = [
        make_event(1, level="LEVEL_2_MOVING_NEARBY", region="left", delivery_profile="level_2"),
        make_event(2, level="LEVEL_2_MOVING_NEARBY", region="left", delivery_profile="level_2"),
    ]

    result = summarizer.summarize(events, objects, timestamp=0.0)

    assert result[0].message == "Vehicle and bicycle moving nearby on the left."


def test_overflow_uses_multiple_wording() -> None:
    summarizer = make_summarizer(max_objects_named=3)
    objects = [make_tracked_object(t, class_name="car", region="left") for t in range(1, 5)]
    events = [make_event(t, region="left") for t in range(1, 5)]

    result = summarizer.summarize(events, objects, timestamp=0.0)

    assert len(result) == 1
    assert result[0].message == "Multiple vehicles moving from the left."


# --- B: max-level-wins, lower levels suppressed ---------------------------


def test_mixed_levels_only_highest_survives() -> None:
    summarizer = make_summarizer()
    objects = [
        make_tracked_object(1, class_name="car", region="left"),
        make_tracked_object(2, class_name="bicycle", region="left"),
    ]
    events = [
        make_event(1, level="LEVEL_1_MOVING_FAR", region="left"),
        make_event(
            2, level="LEVEL_3_APPROACHING", priority="WARNING", region="left",
            message="Bicycle approaching from the left.", delivery_profile="level_3",
        ),
    ]

    result = summarizer.summarize(events, objects, timestamp=0.0)

    assert len(result) == 1
    assert result[0].message == "Bicycle approaching from the left."
    assert result[0].track_id == 2


# --- C: WARNING-tier ties never merge, never flood -------------------------


def test_two_level3_ties_never_merge_only_earliest_kept() -> None:
    summarizer = make_summarizer()
    objects = [
        make_tracked_object(1, class_name="car", region="left"),
        make_tracked_object(2, class_name="bicycle", region="left"),
    ]
    events = [
        make_event(
            1, level="LEVEL_3_APPROACHING", priority="WARNING", region="left",
            message="Vehicle approaching from the left.", created_at=1.0, delivery_profile="level_3",
        ),
        make_event(
            2, level="LEVEL_3_APPROACHING", priority="WARNING", region="left",
            message="Bicycle approaching from the left.", created_at=0.5, delivery_profile="level_3",
        ),
    ]

    result = summarizer.summarize(events, objects, timestamp=0.0)

    assert len(result) == 1
    assert result[0].track_id == 2  # earliest created_at


def test_two_level5_ties_never_diluted() -> None:
    summarizer = make_summarizer()
    objects = [
        make_tracked_object(1, class_name="car", region="left"),
        make_tracked_object(2, class_name="bicycle", region="left"),
    ]
    events = [
        make_event(
            1, level="LEVEL_5_HIGH_DANGER", priority="WARNING", region="left",
            message="Warning! Vehicle approaching from the left. Please wait.",
            created_at=0.0, delivery_profile="level_5",
        ),
        make_event(
            2, level="LEVEL_5_HIGH_DANGER", priority="WARNING", region="left",
            message="Warning! Bicycle approaching from the left. Please wait.",
            created_at=1.0, delivery_profile="level_5",
        ),
    ]

    result = summarizer.summarize(events, objects, timestamp=0.0)

    assert len(result) == 1
    assert "Multiple" not in result[0].message
    assert result[0].message.startswith("Warning!")


# --- D: single event always passes through unchanged -----------------


def test_single_event_any_level_passes_through_unchanged() -> None:
    summarizer = make_summarizer()
    objects = [make_tracked_object(1, class_name="car", region="left")]
    original = make_event(
        1, level="LEVEL_5_HIGH_DANGER", priority="WARNING", region="left",
        message="Warning! Vehicle approaching from the left. Please wait.",
        delivery_profile="level_5",
    )

    result = summarizer.summarize([original], objects, timestamp=0.0)

    assert result == [original]


# --- E: uncertain members never blended into a merged sentence -------


def test_uncertain_member_prevents_merge() -> None:
    summarizer = make_summarizer()
    objects = [
        make_tracked_object(1, class_name="car", region="left"),
        make_tracked_object(2, class_name="bicycle", region="left"),
    ]
    events = [
        make_event(1, region="left", message="Vehicle moving from the left."),
        make_event(
            2, region="left",
            message="Possible bicycle movement from the left. Please wait.",
        ),
    ]

    result = summarizer.summarize(events, objects, timestamp=0.0)

    assert len(result) == 2  # spoken individually, never blended


# --- F: minimum_events threshold ------------------------------------------


def test_below_minimum_events_speaks_individually() -> None:
    summarizer = make_summarizer(minimum_events=3)
    objects = [
        make_tracked_object(1, class_name="car", region="left"),
        make_tracked_object(2, class_name="bicycle", region="left"),
    ]
    events = [make_event(1, region="left"), make_event(2, region="left")]

    result = summarizer.summarize(events, objects, timestamp=0.0)

    assert len(result) == 2


# --- G: enabled / prefer_single_summary toggles ---------------------------


def test_disabled_summarizer_passes_through_everything() -> None:
    summarizer = make_summarizer(enabled=False)
    objects = [
        make_tracked_object(1, class_name="car", region="left"),
        make_tracked_object(2, class_name="bicycle", region="left"),
    ]
    events = [make_event(1, region="left"), make_event(2, region="left")]

    result = summarizer.summarize(events, objects, timestamp=0.0)

    assert result == events


def test_prefer_single_summary_false_returns_informational_tier_individually() -> None:
    summarizer = make_summarizer(prefer_single_summary=False)
    objects = [
        make_tracked_object(1, class_name="car", region="left"),
        make_tracked_object(2, class_name="bicycle", region="left"),
    ]
    events = [make_event(1, region="left"), make_event(2, region="left")]

    result = summarizer.summarize(events, objects, timestamp=0.0)

    assert len(result) == 2
    assert {e.track_id for e in result} == {1, 2}


def test_prefer_single_summary_false_still_suppresses_lower_levels() -> None:
    summarizer = make_summarizer(prefer_single_summary=False)
    objects = [
        make_tracked_object(1, class_name="car", region="left"),
        make_tracked_object(2, class_name="bicycle", region="left"),
    ]
    events = [
        make_event(1, level="LEVEL_1_MOVING_FAR", region="left"),
        make_event(
            2, level="LEVEL_3_APPROACHING", priority="WARNING", region="left",
            message="Bicycle approaching from the left.", delivery_profile="level_3",
        ),
    ]

    result = summarizer.summarize(events, objects, timestamp=0.0)

    assert len(result) == 1
    assert result[0].track_id == 2


# --- H: defensive fallback -------------------------------------------------


def test_event_with_unresolvable_track_passes_through_unchanged() -> None:
    summarizer = make_summarizer()
    objects = [make_tracked_object(1, class_name="car", region="left")]
    events = [make_event(1, region="left"), make_event(2, region="left")]

    result = summarizer.summarize(events, objects, timestamp=0.0)

    assert len(result) == 2  # never silently dropped, never crashes


def test_unresolvable_class_uses_fallback_message() -> None:
    summarizer = make_summarizer(fallback_message="Multiple hazards ahead. Please wait.")
    objects = [
        make_tracked_object(1, class_name="traffic_cone", region="left"),
        make_tracked_object(2, class_name="traffic_cone", region="left"),
    ]
    events = [make_event(1, region="left"), make_event(2, region="left")]

    result = summarizer.summarize(events, objects, timestamp=0.0)

    assert len(result) == 1
    assert result[0].message == "Multiple hazards ahead. Please wait."
    assert result[0].event_type == SCENE_SUMMARY


def test_empty_events_list_returns_empty() -> None:
    summarizer = make_summarizer()
    assert summarizer.summarize([], [], timestamp=0.0) == []


# --- constructor validation ------------------------------------------------


def test_max_objects_named_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        make_summarizer(max_objects_named=0)


def test_same_frame_only_false_raises() -> None:
    with pytest.raises(ValueError):
        SceneSummarizer(
            enabled=True, minimum_events=2, max_objects_named=3,
            same_frame_only=False, prefer_single_summary=True,
            fallback_message="Multiple hazards ahead. Please wait.",
        )


def test_empty_fallback_message_raises() -> None:
    with pytest.raises(ValueError):
        make_summarizer(fallback_message="")


def test_minimum_events_below_two_raises() -> None:
    with pytest.raises(ValueError):
        make_summarizer(minimum_events=1)
