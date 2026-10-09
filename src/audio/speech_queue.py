"""Bounded, deduplicated, priority-ordered speech queue for Atlas's
audio-warning pipeline -- the third stage: AudioEventBuilder ->
EventPolicy -> SpeechQueue -> AudioWorker -> macOS `say`.

Pure data-structure logic, no subprocess/network calls -- fully unit
testable.
"""

from __future__ import annotations

from src.models import AudioEvent

# Lower rank = spoken sooner. WARNING (hazard levels 3-5) always pops
# before INFORMATIONAL (hazard levels 1-2). There is no CRITICAL tier.
_PRIORITY_RANK = {"WARNING": 0, "INFORMATIONAL": 1}
_UNKNOWN_PRIORITY_RANK = 99


class SpeechQueue:
    """A bounded queue of pending AudioEvents, keyed by event.key.

    Being dict-backed gives structural deduplication for free:
    enqueueing an event whose key already has a pending entry replaces
    it with the fresher one (same object, updated info) rather than
    creating a second entry -- "duplicate events removed" with no
    separate dedup pass.

    Args:
        max_size: Maximum number of pending events held at once. When a
            new event would exceed this, the single worst pending entry
            (lowest priority, then oldest) is evicted first -- bounded
            memory, never unlimited growth. Must be >= 1.
        max_age_seconds: An event still unspoken after this long is
            dropped rather than spoken late -- a "vehicle approaching"
            notice about a now-stale frame is worse than silence. Must
            be >= 0.
    """

    def __init__(self, max_size: int, max_age_seconds: float) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if max_age_seconds < 0:
            raise ValueError(f"max_age_seconds must be >= 0, got {max_age_seconds}")

        self._max_size = max_size
        self._max_age_seconds = max_age_seconds
        self._items: dict[str, AudioEvent] = {}

    def __len__(self) -> int:
        return len(self._items)

    def enqueue(self, event: AudioEvent) -> None:
        """Add (or replace the pending entry for) an event. Evicts the
        single worst pending entry if this would exceed max_size."""
        self._items[event.key] = event
        if len(self._items) > self._max_size:
            self._evict_worst()

    def clear(self) -> int:
        """Remove every pending entry, regardless of track or key.
        Returns the number of entries removed. Used when a critical
        system-health message must replace ALL pending operational
        audio (informational events, scene summaries, uncertain object
        messages alike) -- unlike evict_track(), this is not scoped to
        one object."""
        removed = len(self._items)
        self._items.clear()
        return removed

    def evict_track(self, track_id: int) -> None:
        """Remove every pending entry for this track_id, regardless of
        key. Used when a fresh hazard assessment for an object arrives
        (an escalation or de-escalation to a different hazard level,
        which is structurally a different key -- see AudioEventBuilder)
        so a stale pending message about the SAME object is never left
        behind to be spoken after a newer assessment already superseded
        it. A no-op if nothing is pending for this track."""
        stale_keys = [key for key, event in self._items.items() if event.track_id == track_id]
        for key in stale_keys:
            del self._items[key]

    def pop_next(self, now: float) -> AudioEvent | None:
        """Drop stale entries, then return and remove the highest-
        priority, oldest-among-equal-priority survivor. None if empty
        (or everything was stale)."""
        self._drop_stale(now)
        if not self._items:
            return None
        best_key = min(self._items, key=lambda key: self._sort_key(self._items[key]))
        return self._items.pop(best_key)

    def _drop_stale(self, now: float) -> None:
        stale_keys = [
            key
            for key, event in self._items.items()
            if now - event.created_at > self._max_age_seconds
        ]
        for key in stale_keys:
            del self._items[key]

    def _evict_worst(self) -> None:
        # "Worst" for eviction = lowest priority, and among equal
        # priority, the OLDEST (newer perception data about the same
        # situation supersedes older data, so prefer keeping it).
        worst_key = max(self._items, key=lambda key: self._eviction_key(self._items[key]))
        del self._items[worst_key]

    @staticmethod
    def _sort_key(event: AudioEvent) -> tuple[int, float]:
        # Used by pop_next(): lower is "better" (spoken sooner) --
        # highest priority first, then oldest first among equal
        # priority (FIFO within a tier).
        return (_PRIORITY_RANK.get(event.priority, _UNKNOWN_PRIORITY_RANK), event.created_at)

    @staticmethod
    def _eviction_key(event: AudioEvent) -> tuple[int, float]:
        # Used by _evict_worst(): higher is "worse" (evicted first) --
        # lowest priority first, then oldest first among equal priority
        # (note the negated created_at, opposite of _sort_key's ordering).
        return (_PRIORITY_RANK.get(event.priority, _UNKNOWN_PRIORITY_RANK), -event.created_at)
