"""Converts resolved AudioHazardResults into candidate spoken AudioEvents
-- a thin adapter stage in Atlas's real-time audio-warning pipeline:

    ... -> AudioHazardResolver -> AudioEventBuilder -> SceneSummarizer ->
    EventPolicy -> SpeechQueue -> AudioWorker -> macOS `say`

This module does NOT decide hazard levels, wording, priority, or
delivery profile -- src/audio/audio_hazard_resolver.py's
AudioHazardResolver already decided all of that (combining confirmed
motion, proximity, confirmed approach, and resolved corridor conflict).
AudioEventBuilder's only remaining job is wrapping each AudioHazardResult
into an AudioEvent: building the dedup key and copying level/priority/
message/delivery_profile straight through, never re-deriving them.

Pure logic, no OpenCV/subprocess/network calls -- fully unit testable
with synthetic AudioHazardResult data.
"""

from __future__ import annotations

from src.models import AudioEvent, AudioHazardResult


class AudioEventBuilder:
    """Wraps this frame's resolved AudioHazardResults into AudioEvents.
    Stateless -- no cooldown/dedup/queueing/hazard-decision logic lives
    here.
    """

    def build_events(
        self,
        hazard_results: dict[int, AudioHazardResult],
        timestamp: float,
    ) -> list[AudioEvent]:
        """One AudioEvent per hazard result this frame.

        The dedup key includes the hazard level
        (`"{level}:{track_id}:{region}"`), so an escalation or
        de-escalation to a different level for the same object is
        structurally a different key -- it bypasses whatever cooldown
        applied to the previous level rather than being suppressed by
        it (stale entries for the same track are separately evicted from
        SpeechQueue by AudioWorker.enqueue_events(), since a changed key
        alone doesn't remove an already-pending different-key entry).
        """
        events = []
        for track_id, hazard in hazard_results.items():
            events.append(
                AudioEvent(
                    key=f"{hazard.level}:{track_id}:{hazard.region}",
                    event_type=hazard.level,
                    priority=hazard.priority,
                    message=hazard.recommended_message,
                    track_id=track_id,
                    created_at=timestamp,
                    delivery_profile=hazard.delivery_profile,
                )
            )
        return events
