"""Unit tests for src/audio/event_policy.py -- the cooldown/
deduplication gate between AudioEventBuilder and SpeechQueue. Pure
logic, no subprocess involved.

Cooldown is selected by PRIORITY only (not by the specific hazard-level
event_type) -- an escalation/de-escalation to a different hazard level
for the same object is already a different dedup key (see
AudioEventBuilder), so no per-level cooldown table is needed here.
"""

import pytest

from src.audio.event_policy import EventPolicy
from src.models import AudioEvent


def make_event(
    key: str = "LEVEL_1_MOVING_FAR:1:left",
    event_type: str = "LEVEL_1_MOVING_FAR",
    priority: str = "INFORMATIONAL",
    message: str = "Vehicle moving from the left.",
    track_id: int = 1,
    created_at: float = 0.0,
) -> AudioEvent:
    return AudioEvent(
        key=key, event_type=event_type, priority=priority,
        message=message, track_id=track_id, created_at=created_at,
        delivery_profile="level_1",
    )


@pytest.fixture
def policy() -> EventPolicy:
    return EventPolicy(
        cooldown_same_event_seconds=4.0,
        cooldown_informational_seconds=8.0,
    )


# --- duplicate suppression / cooldown ---------------------------------


def test_first_occurrence_is_always_admitted(policy: EventPolicy) -> None:
    event = make_event()
    admitted = policy.filter([event], now=0.0)
    assert admitted == [event]


def test_same_key_suppressed_within_cooldown(policy: EventPolicy) -> None:
    event = make_event(key="LEVEL_1_MOVING_FAR:1:left")
    policy.mark_spoken(event.key, now=0.0)

    admitted = policy.filter([event], now=1.0)  # informational cooldown is 8s

    assert admitted == []


def test_same_key_admitted_again_after_cooldown_expires(policy: EventPolicy) -> None:
    event = make_event(key="LEVEL_1_MOVING_FAR:1:left")
    policy.mark_spoken(event.key, now=0.0)

    admitted = policy.filter([event], now=8.1)  # just past the 8s informational cooldown

    assert admitted == [event]


def test_different_keys_never_suppress_each_other(policy: EventPolicy) -> None:
    event_a = make_event(key="LEVEL_1_MOVING_FAR:1:left")
    event_b = make_event(key="LEVEL_1_MOVING_FAR:2:right", track_id=2)
    policy.mark_spoken(event_a.key, now=0.0)

    admitted = policy.filter([event_b], now=0.1)

    assert admitted == [event_b]


def test_admission_alone_does_not_start_cooldown(policy: EventPolicy) -> None:
    # filter() must NOT itself call mark_spoken() -- only the worker
    # does, at actual speech time. So filtering the same event twice in
    # a row (without an intervening mark_spoken) admits it both times.
    event = make_event()
    first = policy.filter([event], now=0.0)
    second = policy.filter([event], now=0.01)

    assert first == [event]
    assert second == [event]


def test_escalation_to_a_different_level_bypasses_the_old_keys_cooldown(
    policy: EventPolicy,
) -> None:
    # A different hazard level for the same track/region is a different
    # key -- it must never be suppressed by the previous level's cooldown.
    low = make_event(key="LEVEL_1_MOVING_FAR:1:left", event_type="LEVEL_1_MOVING_FAR")
    high = make_event(
        key="LEVEL_5_HIGH_DANGER:1:left", event_type="LEVEL_5_HIGH_DANGER", priority="WARNING"
    )
    policy.mark_spoken(low.key, now=0.0)

    admitted = policy.filter([high], now=0.1)

    assert admitted == [high]


# --- cooldown by priority --------------------------------------------------


def test_warning_priority_uses_same_event_cooldown(policy: EventPolicy) -> None:
    event = make_event(
        key="LEVEL_5_HIGH_DANGER:1:left", event_type="LEVEL_5_HIGH_DANGER", priority="WARNING"
    )
    policy.mark_spoken(event.key, now=0.0)

    assert policy.filter([event], now=3.9) == []  # still within 4s
    assert policy.filter([event], now=4.1) == [event]


def test_informational_priority_uses_informational_cooldown(policy: EventPolicy) -> None:
    event = make_event(
        key="LEVEL_1_MOVING_FAR:1:left", event_type="LEVEL_1_MOVING_FAR", priority="INFORMATIONAL"
    )
    policy.mark_spoken(event.key, now=0.0)

    assert policy.filter([event], now=7.9) == []  # still within 8s
    assert policy.filter([event], now=8.1) == [event]


# --- priority-aware fallback for scene_summary events ----------------------


def test_scene_summary_warning_uses_same_event_cooldown(policy: EventPolicy) -> None:
    event = make_event(
        key="scene_summary:LEVEL_5_HIGH_DANGER:left", event_type="scene_summary", priority="WARNING"
    )
    policy.mark_spoken(event.key, now=0.0)

    assert policy.filter([event], now=3.9) == []
    assert policy.filter([event], now=4.1) == [event]


def test_scene_summary_informational_uses_informational_cooldown(policy: EventPolicy) -> None:
    event = make_event(
        key="scene_summary:LEVEL_1_MOVING_FAR:left",
        event_type="scene_summary",
        priority="INFORMATIONAL",
    )
    policy.mark_spoken(event.key, now=0.0)

    assert policy.filter([event], now=7.9) == []
    assert policy.filter([event], now=8.1) == [event]


# --- constructor validation --------------------------------------------


def test_negative_cooldowns_raise() -> None:
    with pytest.raises(ValueError):
        EventPolicy(cooldown_same_event_seconds=-1.0, cooldown_informational_seconds=8.0)
    with pytest.raises(ValueError):
        EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=-1.0)
