"""Cooldown/deduplication policy for Atlas's audio-warning pipeline --
the second stage: AudioEventBuilder -> EventPolicy -> SpeechQueue ->
AudioWorker -> macOS `say`.

EventPolicy answers exactly one question: "given what's actually been
spoken recently, should this candidate event be allowed through at
all?" It does NOT own the speech queue, does NOT decide priority
ordering, and does NOT invoke subprocess -- those are SpeechQueue's and
AudioWorker's jobs respectively. Kept deliberately narrow and stateless
beyond its own cooldown bookkeeping.

Pure logic, no subprocess/network calls -- fully unit testable.
"""

from __future__ import annotations

from src.models import AudioEvent


class EventPolicy:
    """Per-key cooldown, selected by event priority.

    Cooldown is measured from the last time an event with this exact
    key was ACTUALLY SPOKEN (via mark_spoken(), called by AudioWorker at
    the moment it fires the speech subprocess) -- not from when it was
    merely admitted/queued. This matters: a bounded queue can evict a
    pending event before it's ever spoken, and if the cooldown clock had
    already started at admission time, a genuinely new occurrence of the
    same event could be wrongly suppressed for something that was never
    actually said.

    Deliberately keyed by PRIORITY, not by the specific hazard-level
    event_type: an escalation/de-escalation to a different hazard level
    for the same object is already a different dedup KEY (see
    AudioEventBuilder), which alone bypasses whatever cooldown applied
    to the previous level -- no per-level cooldown table is needed here.

    Args:
        cooldown_same_event_seconds: Cooldown for any WARNING-priority
            event (hazard levels 3-5, and WARNING-priority
            "scene_summary" events) -- the more urgent tier, so the
            shorter cooldown (allowed to re-alert soonest if the
            situation persists).
        cooldown_informational_seconds: Cooldown for any INFORMATIONAL-
            priority event (hazard levels 1-2, and INFORMATIONAL-
            priority "scene_summary" events) -- the least urgent tier,
            so the longer cooldown.
    """

    def __init__(
        self,
        cooldown_same_event_seconds: float,
        cooldown_informational_seconds: float,
    ) -> None:
        if cooldown_same_event_seconds < 0:
            raise ValueError(
                "cooldown_same_event_seconds must be >= 0, got "
                f"{cooldown_same_event_seconds}"
            )
        if cooldown_informational_seconds < 0:
            raise ValueError(
                "cooldown_informational_seconds must be >= 0, got "
                f"{cooldown_informational_seconds}"
            )

        self._cooldown_by_priority = {
            "WARNING": cooldown_same_event_seconds,
            "INFORMATIONAL": cooldown_informational_seconds,
        }
        self._last_spoken: dict[str, float] = {}

    def filter(self, events: list[AudioEvent], now: float) -> list[AudioEvent]:
        """Return only the events not currently on cooldown."""
        admitted = []
        for event in events:
            last = self._last_spoken.get(event.key)
            cooldown = self._cooldown_for(event)
            if last is not None and now - last < cooldown:
                continue
            admitted.append(event)
        return admitted

    def _cooldown_for(self, event: AudioEvent) -> float:
        return self._cooldown_by_priority.get(
            event.priority, self._cooldown_by_priority["WARNING"]
        )

    def mark_spoken(self, key: str, now: float) -> None:
        """Record that an event with this key was actually spoken now --
        called by AudioWorker at the moment it fires the speech
        subprocess, never earlier."""
        self._last_spoken[key] = now
