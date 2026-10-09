"""Combines related per-frame candidate AudioEvents into fewer, concise
spoken summaries -- the stage between AudioEventBuilder and EventPolicy:

    AudioEventBuilder.build_events() -> SceneSummarizer.summarize() ->
    EventPolicy.filter() -> SpeechQueue.enqueue() -> AudioWorker.tick()

Rebuckets by REGION ONLY (hazard level is now the ordering/priority
signal, not a separate bucketing axis -- see src/audio/audio_hazard_
resolver.py for the 5-level hierarchy this reuses without duplicating
any of its decision logic). Within a region:

- Only the HIGHEST hazard level present survives -- lower-level events
  for other objects in the same region are dropped this frame ("Level 5
  must never be diluted," "suppress unrelated Level 1 information,"
  applied uniformly to every level so the rule is one principle, not a
  4/5-specific special case).
- WARNING-tier levels (3/4/5) are NEVER merged into a named list, even
  when several objects share the same top level in the same region --
  "prefer one immediate hazard over a long inventory." Only the single
  earliest such event is spoken; the rest are dropped this frame (still
  re-evaluated fresh next frame if the situation persists).
- INFORMATIONAL-tier levels (1/2) MAY be merged into a named list,
  using class counting and pluralization, but only when every candidate
  message is confidently worded (never starts with "Possible").
  AudioEvent has no structured uncertain flag, so uncertainty is read
  from the message prefix.

Deterministic, config-driven grouping only -- no threat scoring, no
LLM/cloud/generative text, no SAFE/UNSAFE guidance. Reuses
AudioHazardResolver's own object-name/direction-phrase maps and level
metadata so summary wording matches individual-event wording exactly.
Pure logic, no OpenCV/subprocess/network calls -- fully unit testable
with synthetic AudioEvent/TrackedObject data.
"""

from __future__ import annotations

from src.audio.audio_hazard_resolver import (
    LEVEL_1_MOVING_FAR,
    LEVEL_2_MOVING_NEARBY,
    _DELIVERY_PROFILE_BY_LEVEL,
    _DIRECTION_PHRASE_MAP,
    _LEVEL_RANK,
    _ON_DIRECTION_PHRASE_MAP,
    _OBJECT_NAME_MAP,
    _PRIORITY_BY_LEVEL,
)
from src.models import AudioEvent, TrackedObject

SCENE_SUMMARY = "scene_summary"

# Fixed, deterministic naming order -- never input/insertion order, so
# wording is identical regardless of the order objects appear in a frame.
_CANONICAL_CLASS_ORDER = ("Vehicle", "Motorcycle", "Bicycle", "Person")

_PLURAL_MAP = {
    "Vehicle": "vehicles",
    "Motorcycle": "motorcycles",
    "Bicycle": "bicycles",
    "Person": "people",
}

_VALID_REGIONS = ("left", "center", "right")

# Only Level 1/2 (INFORMATIONAL) buckets are ever merged into a named
# list -- Level 3/4/5 (WARNING) always speaks a single event, never a
# blended inventory sentence.
_MERGEABLE_RANKS = {_LEVEL_RANK[LEVEL_1_MOVING_FAR], _LEVEL_RANK[LEVEL_2_MOVING_NEARBY]}
_LEVEL_BY_RANK = {rank: level for level, rank in _LEVEL_RANK.items()}


def _capitalize(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


class SceneSummarizer:
    """Buckets this frame's candidate hazard AudioEvents by region and,
    within each region, keeps only the highest hazard level present --
    merging same-level INFORMATIONAL events into one named-list summary
    where possible, never merging WARNING-tier events. Stateless -- no
    cooldown/dedup/queueing logic lives here (that's EventPolicy's and
    SpeechQueue's job respectively); this stage only decides what gets
    said, never when or how often.

    Args:
        enabled: Master switch. False makes summarize() a pure pass-
            through, no bucketing performed at all.
        minimum_events: A same-level bucket smaller than this is never
            merged -- passed through as its original individual event(s)
            unchanged.
        max_objects_named: Above this many total objects in a mergeable
            bucket, wording switches from naming each class to a
            generic count phrase (e.g. "Multiple vehicles" / "Multiple
            hazards") to keep the sentence short.
        same_frame_only: Documents that this component only ever
            aggregates within a single frame's candidate batch -- no
            cross-frame accumulation. Only True is currently supported.
        prefer_single_summary: When False, region bucketing/level-
            filtering still happens (lower levels are still suppressed
            in favor of the highest present) but same-level merging into
            a named list never happens -- summarize() returns the
            highest-level event(s) one-for-one instead of merged.
        fallback_message: Spoken as a last-resort WARNING-priority
            summary when a mergeable bucket qualifies but can't be
            safely worded (e.g. an object's class can't be resolved) --
            never silently dropped.
    """

    def __init__(
        self,
        enabled: bool,
        minimum_events: int,
        max_objects_named: int,
        same_frame_only: bool,
        prefer_single_summary: bool,
        fallback_message: str,
    ) -> None:
        if minimum_events < 2:
            raise ValueError(f"minimum_events must be >= 2, got {minimum_events}")
        if max_objects_named < 1:
            raise ValueError(f"max_objects_named must be >= 1, got {max_objects_named}")
        if not same_frame_only:
            raise ValueError(
                "same_frame_only=False is not supported -- SceneSummarizer "
                "only ever aggregates within a single frame's candidate batch"
            )
        if not fallback_message:
            raise ValueError("fallback_message must be non-empty")

        self._enabled = enabled
        self._minimum_events = minimum_events
        self._max_objects_named = max_objects_named
        self._prefer_single_summary = prefer_single_summary
        self._fallback_message = fallback_message

    def summarize(
        self,
        events: list[AudioEvent],
        tracked_objects: list[TrackedObject],
        timestamp: float,
    ) -> list[AudioEvent]:
        """Returns a new list. Never mutates `events`."""
        if not self._enabled or not events:
            return list(events)

        class_name_by_track = {obj.track_id: obj.class_name for obj in tracked_objects}
        region_by_track = {obj.track_id: obj.region for obj in tracked_objects}

        passthrough: list[AudioEvent] = []
        region_buckets: dict[str, list[AudioEvent]] = {}
        for event in events:
            region = region_by_track.get(event.track_id) if event.track_id is not None else None
            if region not in _VALID_REGIONS:
                # Can't safely determine which bucket this belongs to --
                # never guess, just pass it through unchanged.
                passthrough.append(event)
                continue
            region_buckets.setdefault(region, []).append(event)

        result: list[AudioEvent] = list(passthrough)
        for region, region_events in region_buckets.items():
            result.extend(
                self._resolve_region(region, region_events, class_name_by_track, timestamp)
            )
        return result

    def _resolve_region(
        self,
        region: str,
        region_events: list[AudioEvent],
        class_name_by_track: dict[int, str],
        timestamp: float,
    ) -> list[AudioEvent]:
        max_rank = max(_LEVEL_RANK.get(e.event_type, 0) for e in region_events)
        top_events = [e for e in region_events if _LEVEL_RANK.get(e.event_type, 0) == max_rank]
        # Events NOT at the max rank are intentionally dropped here --
        # "suppress unrelated lower-level information," applied uniformly
        # to every level rather than only 4/5.

        if len(top_events) == 1:
            return top_events

        if max_rank not in _MERGEABLE_RANKS:
            # WARNING-tier (levels 3-5) ties are never merged into one
            # blended sentence and never all spoken individually --
            # "prefer one immediate hazard over a long inventory" is a
            # safety rule here, not a preference toggle. Speak only the
            # single earliest one; the rest are dropped this frame, not
            # lost forever -- re-evaluated fresh next frame if the
            # situation persists.
            return [min(top_events, key=lambda e: e.created_at)]

        if not self._prefer_single_summary:
            # Mergeable (informational) tier, but merging disabled --
            # every top-level event is spoken individually (one-for-one);
            # lower levels in this region are still dropped above.
            return top_events

        if len(top_events) < self._minimum_events or any(
            e.message.startswith("Possible") for e in top_events
        ):
            # Too few to bother merging, or at least one member is
            # itself uncertain -- never blend confidence levels into one
            # sentence; speak them individually instead.
            return top_events

        summary = self._build_named_summary(
            region, max_rank, top_events, class_name_by_track, timestamp
        )
        return [summary] if summary is not None else top_events

    def _build_named_summary(
        self,
        region: str,
        rank: int,
        events: list[AudioEvent],
        class_name_by_track: dict[int, str],
        timestamp: float,
    ) -> AudioEvent | None:
        direction = _DIRECTION_PHRASE_MAP.get(region)
        on_direction = _ON_DIRECTION_PHRASE_MAP.get(region)
        if direction is None or on_direction is None:
            return None

        counts = self._class_counts(events, class_name_by_track)
        if counts is None:
            return self._fallback_summary(region, rank, events, timestamp)

        total = sum(counts.values())
        if total > self._max_objects_named:
            if len(counts) == 1:
                only_name = next(iter(counts))
                subject = f"multiple {_PLURAL_MAP[only_name]}"
            else:
                subject = "multiple hazards"
        else:
            subject = self._named_list(counts)

        level = _LEVEL_BY_RANK[rank]
        if level == LEVEL_1_MOVING_FAR:
            message = f"{_capitalize(subject)} moving {direction}."
        else:
            message = f"{_capitalize(subject)} moving nearby {on_direction}."

        return AudioEvent(
            key=f"{SCENE_SUMMARY}:{level}:{region}",
            event_type=SCENE_SUMMARY,
            priority=_PRIORITY_BY_LEVEL[level],
            message=message,
            track_id=None,
            created_at=timestamp,
            delivery_profile=_DELIVERY_PROFILE_BY_LEVEL[level],
        )

    def _fallback_summary(
        self, region: str, rank: int, events: list[AudioEvent], timestamp: float
    ) -> AudioEvent:
        level = _LEVEL_BY_RANK[rank]
        return AudioEvent(
            key=f"{SCENE_SUMMARY}:{level}:{region}:fallback",
            event_type=SCENE_SUMMARY,
            priority=_PRIORITY_BY_LEVEL[level],
            message=self._fallback_message,
            track_id=None,
            created_at=timestamp,
            delivery_profile=_DELIVERY_PROFILE_BY_LEVEL[level],
        )

    def _class_counts(
        self, events: list[AudioEvent], class_name_by_track: dict[int, str]
    ) -> dict[str, int] | None:
        """Distinct-class counts, or None if any member's class can't be
        resolved -- the caller must treat None as "can't safely word
        this," never guess."""
        counts: dict[str, int] = {}
        for event in events:
            raw_class = class_name_by_track.get(event.track_id) if event.track_id is not None else None
            name = _OBJECT_NAME_MAP.get(raw_class) if raw_class is not None else None
            if name is None:
                return None
            counts[name] = counts.get(name, 0) + 1
        return counts

    def _named_list(self, counts: dict[str, int]) -> str:
        """Lowercase, canonically-ordered named list, e.g. "vehicle and
        bicycle" or "2 vehicles and a person" -- capitalization of the
        first character is the caller's responsibility."""
        parts = []
        for name in _CANONICAL_CLASS_ORDER:
            if name not in counts:
                continue
            count = counts[name]
            parts.append(f"{count} {_PLURAL_MAP[name]}" if count > 1 else name.lower())

        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        return ", ".join(parts[:-1]) + f", and {parts[-1]}"
