"""Unit tests for src/audio/speech_queue.py -- the bounded, deduplicated,
priority-ordered speech queue. Pure data-structure logic, no subprocess
involved.
"""

import pytest

from src.audio.speech_queue import SpeechQueue
from src.models import AudioEvent


def make_event(
    key: str = "LEVEL_1_MOVING_FAR:1:left",
    priority: str = "INFORMATIONAL",
    message: str = "Vehicle moving from the left.",
    track_id: int = 1,
    created_at: float = 0.0,
) -> AudioEvent:
    return AudioEvent(
        key=key, event_type="LEVEL_1_MOVING_FAR", priority=priority,
        message=message, track_id=track_id, created_at=created_at,
        delivery_profile="level_1",
    )


@pytest.fixture
def queue() -> SpeechQueue:
    return SpeechQueue(max_size=3, max_age_seconds=5.0)


# --- basic enqueue/pop ---------------------------------------------------


def test_pop_next_on_empty_queue_returns_none(queue: SpeechQueue) -> None:
    assert queue.pop_next(now=0.0) is None


def test_enqueue_then_pop_returns_the_event(queue: SpeechQueue) -> None:
    event = make_event()
    queue.enqueue(event)
    assert queue.pop_next(now=0.0) == event
    assert len(queue) == 0


# --- structural deduplication -------------------------------------------


def test_enqueueing_same_key_replaces_pending_entry(queue: SpeechQueue) -> None:
    old_event = make_event(key="approaching:1:left", message="old", created_at=0.0)
    new_event = make_event(key="approaching:1:left", message="new", created_at=1.0)

    queue.enqueue(old_event)
    queue.enqueue(new_event)

    assert len(queue) == 1
    assert queue.pop_next(now=1.0).message == "new"


# --- priority ordering ---------------------------------------------------


def test_warning_pops_before_informational(queue: SpeechQueue) -> None:
    informational = make_event(key="approaching:1:left", priority="INFORMATIONAL")
    warning = make_event(key="uncertain:2:right", priority="WARNING", track_id=2)

    queue.enqueue(informational)
    queue.enqueue(warning)

    assert queue.pop_next(now=0.0) == warning
    assert queue.pop_next(now=0.0) == informational


def test_fifo_among_equal_priority(queue: SpeechQueue) -> None:
    first = make_event(key="approaching:1:left", created_at=0.0)
    second = make_event(key="approaching:2:right", created_at=1.0, track_id=2)

    queue.enqueue(second)  # enqueued out of chronological order
    queue.enqueue(first)

    assert queue.pop_next(now=2.0) == first
    assert queue.pop_next(now=2.0) == second


# --- bounded / overflow ---------------------------------------------------


def test_queue_overflow_evicts_lowest_priority(queue: SpeechQueue) -> None:
    # max_size=3 -- enqueue 2 informational + 1 warning, then a 4th
    # (informational) should trigger eviction of a low-priority entry,
    # never the warning.
    queue.enqueue(make_event(key="k1", priority="INFORMATIONAL", created_at=0.0))
    queue.enqueue(make_event(key="k2", priority="INFORMATIONAL", created_at=1.0))
    queue.enqueue(make_event(key="k3", priority="WARNING", created_at=2.0))
    assert len(queue) == 3

    queue.enqueue(make_event(key="k4", priority="INFORMATIONAL", created_at=3.0))

    assert len(queue) == 3  # bounded -- never grows past max_size
    remaining_keys = {queue.pop_next(now=3.0).key for _ in range(3)}
    assert "k3" in remaining_keys  # the WARNING must survive eviction


def test_overflow_evicts_oldest_among_equal_priority(queue: SpeechQueue) -> None:
    queue.enqueue(make_event(key="k1", priority="INFORMATIONAL", created_at=0.0))
    queue.enqueue(make_event(key="k2", priority="INFORMATIONAL", created_at=1.0))
    queue.enqueue(make_event(key="k3", priority="INFORMATIONAL", created_at=2.0))
    queue.enqueue(make_event(key="k4", priority="INFORMATIONAL", created_at=3.0))

    remaining_keys = {queue.pop_next(now=3.0).key for _ in range(3)}
    assert "k1" not in remaining_keys  # oldest was evicted
    assert remaining_keys == {"k2", "k3", "k4"}


def test_never_grows_unbounded(queue: SpeechQueue) -> None:
    for i in range(50):
        queue.enqueue(make_event(key=f"k{i}", created_at=float(i)))
    assert len(queue) <= 3


# --- staleness / expiration -----------------------------------------------


def test_stale_event_is_dropped_before_being_returned(queue: SpeechQueue) -> None:
    stale = make_event(key="approaching:1:left", created_at=0.0)
    queue.enqueue(stale)

    result = queue.pop_next(now=5.1)  # max_age_seconds=5.0

    assert result is None
    assert len(queue) == 0


def test_fresh_event_survives_staleness_check(queue: SpeechQueue) -> None:
    fresh = make_event(key="approaching:1:left", created_at=0.0)
    queue.enqueue(fresh)

    assert queue.pop_next(now=4.9) == fresh


def test_stale_and_fresh_mixed_only_fresh_returned(queue: SpeechQueue) -> None:
    stale = make_event(key="k1", created_at=0.0)
    fresh = make_event(key="k2", created_at=5.0, track_id=2)
    queue.enqueue(stale)
    queue.enqueue(fresh)

    assert queue.pop_next(now=5.1) == fresh
    assert queue.pop_next(now=5.1) is None


# --- evict_track (escalation/de-escalation replaces stale entries) --------


def test_evict_track_removes_all_pending_entries_for_that_track(queue: SpeechQueue) -> None:
    queue.enqueue(make_event(key="LEVEL_1_MOVING_FAR:1:left", track_id=1))
    queue.enqueue(make_event(key="LEVEL_2_MOVING_NEARBY:1:left", track_id=1))
    queue.enqueue(make_event(key="LEVEL_1_MOVING_FAR:2:right", track_id=2))

    queue.evict_track(1)

    assert len(queue) == 1
    assert queue.pop_next(now=0.0).track_id == 2


def test_evict_track_is_a_noop_when_nothing_pending_for_that_track(queue: SpeechQueue) -> None:
    queue.enqueue(make_event(key="LEVEL_1_MOVING_FAR:2:right", track_id=2))

    queue.evict_track(999)

    assert len(queue) == 1


# --- clear (system-health messages replace ALL pending operational audio) -


def test_clear_removes_every_pending_entry_regardless_of_track(queue: SpeechQueue) -> None:
    queue.enqueue(make_event(key="LEVEL_1_MOVING_FAR:1:left", track_id=1))
    queue.enqueue(make_event(key="LEVEL_2_MOVING_NEARBY:2:right", track_id=2))

    removed = queue.clear()

    assert removed == 2
    assert len(queue) == 0
    assert queue.pop_next(now=0.0) is None


def test_clear_on_empty_queue_returns_zero(queue: SpeechQueue) -> None:
    assert queue.clear() == 0
    assert len(queue) == 0


# --- constructor validation --------------------------------------------


def test_max_size_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        SpeechQueue(max_size=0, max_age_seconds=5.0)


def test_max_age_seconds_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        SpeechQueue(max_size=3, max_age_seconds=-1.0)
