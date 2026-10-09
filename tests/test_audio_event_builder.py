"""Unit tests for src/audio/audio_event_builder.py -- the thin adapter
that wraps resolved AudioHazardResults into AudioEvents. No camera,
model, network, or subprocess involved -- pure logic over synthetic
AudioHazardResult data. Wording/priority/hazard-level decision tests
live in tests/test_audio_hazard_resolver.py, since this module no
longer makes any of those decisions itself.
"""

from src.audio.audio_event_builder import AudioEventBuilder
from src.models import AudioHazardResult


def make_hazard(
    track_id: int = 1,
    level: str = "LEVEL_1_MOVING_FAR",
    priority: str = "INFORMATIONAL",
    region: str = "left",
    message: str = "Vehicle moving from the left.",
    delivery_profile: str = "level_1",
) -> AudioHazardResult:
    return AudioHazardResult(
        track_id=track_id,
        level=level,
        priority=priority,
        class_name="car",
        region=region,
        proximity_zone="FAR",
        approach_state="NOT_APPROACHING",
        intersects_corridor=False,
        uncertain=False,
        reason_codes=("FAR",),
        recommended_message=message,
        delivery_profile=delivery_profile,
    )


def test_one_event_per_hazard_result() -> None:
    builder = AudioEventBuilder()
    hazards = {1: make_hazard(track_id=1), 2: make_hazard(track_id=2, region="right")}

    events = builder.build_events(hazards, timestamp=0.0)

    assert len(events) == 2
    assert {e.track_id for e in events} == {1, 2}


def test_message_priority_and_delivery_profile_copied_verbatim() -> None:
    builder = AudioEventBuilder()
    hazard = make_hazard(
        message="Warning! Vehicle approaching from the left. Please wait.",
        priority="WARNING",
        delivery_profile="level_5",
    )

    events = builder.build_events({1: hazard}, timestamp=0.0)

    assert events[0].message == hazard.recommended_message
    assert events[0].priority == "WARNING"
    assert events[0].delivery_profile == "level_5"
    assert events[0].event_type == hazard.level


def test_key_format_includes_level_track_and_region() -> None:
    builder = AudioEventBuilder()
    hazard = make_hazard(track_id=42, level="LEVEL_3_APPROACHING", region="right")

    events = builder.build_events({42: hazard}, timestamp=0.0)

    assert events[0].key == "LEVEL_3_APPROACHING:42:right"


def test_different_levels_for_same_track_produce_different_keys() -> None:
    # Structural proof of the escalation mechanism: a different level
    # for the same track/region is a different dedup key, so it bypasses
    # whatever cooldown applied to the previous level.
    builder = AudioEventBuilder()
    low = make_hazard(track_id=1, level="LEVEL_1_MOVING_FAR", region="left")
    high = make_hazard(track_id=1, level="LEVEL_5_HIGH_DANGER", region="left")

    low_key = builder.build_events({1: low}, timestamp=0.0)[0].key
    high_key = builder.build_events({1: high}, timestamp=0.0)[0].key

    assert low_key != high_key


def test_created_at_uses_the_given_timestamp() -> None:
    builder = AudioEventBuilder()
    events = builder.build_events({1: make_hazard()}, timestamp=123.5)
    assert events[0].created_at == 123.5


def test_empty_hazard_results_produces_no_events() -> None:
    builder = AudioEventBuilder()
    assert builder.build_events({}, timestamp=0.0) == []
