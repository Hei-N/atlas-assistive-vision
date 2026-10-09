"""User-selectable operating modes for Atlas's audio pipeline -- MINIMAL
(traffic-only), BALANCED (traffic + relevant people, the default), and
DETAILED (structured environmental summaries).

    AudioHazardResolver.resolve_for_tracks()
            |  dict[track_id, AudioHazardResult]
            v
    OperatingModePolicy.filter_events()   -- stage 1: class/person ELIGIBILITY
            v
    AudioEventBuilder.build_events()  ->  SceneSummarizer.summarize()
            |  list[AudioEvent]
            v
    OperatingModePolicy.finalize_events() -- stage 2: Level-5 message form +
            |  danger-episode repeat policy + Detailed summary caps
            v
    EventPolicy -> SpeechQueue -> AudioWorker

Two methods, two non-overlapping responsibilities -- never the same job
invoked twice. This module does NOT decide hazard levels (that's
AudioHazardResolver's job, unchanged and untouched), does NOT build
wording for anything except the Level-5 danger phrase, and does NOT
duplicate any perception. All three modes see identical upstream
perception and hazard-level decisions -- only what gets spoken, how
often, and in how much detail differs.

Pure logic, no OpenCV/subprocess/network calls -- fully unit testable
with synthetic AudioHazardResult/AudioEvent data.
"""

from __future__ import annotations

import dataclasses
import logging

from src.audio.audio_hazard_resolver import (
    LEVEL_2_MOVING_NEARBY,
    LEVEL_5_HIGH_DANGER,
    PRIORITY_WARNING,
    _LEVEL_RANK,
)
from src.audio.scene_summarizer import SCENE_SUMMARY
from src.models import AudioEvent, AudioHazardResult, OperatingModeProfile, OperatingModeTrace

logger = logging.getLogger("atlas")

MINIMAL = "MINIMAL"
BALANCED = "BALANCED"
DETAILED = "DETAILED"
_VALID_MODES = (MINIMAL, BALANCED, DETAILED)

# Person-announcement eligibility, one per mode -- named to avoid any
# confusion with Python's None.
PERSON_INELIGIBLE = "NONE"
PERSON_RELEVANT_ONLY = "RELEVANT_ONLY"
PERSON_BROAD = "BROAD"

TRAFFIC_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
PERSON_CLASS = "person"

# Stable mode-filter/repeat/summary reason codes -- logged in
# OperatingModeTrace, referenced by tests, never renamed casually.
ALLOWED_TRAFFIC_EVENT = "ALLOWED_TRAFFIC_EVENT"
SUPPRESSED_TRAFFIC_DISABLED = "SUPPRESSED_TRAFFIC_DISABLED"
SUPPRESSED_PERSON_IN_MINIMAL = "SUPPRESSED_PERSON_IN_MINIMAL"
SUPPRESSED_IRRELEVANT_PERSON_IN_BALANCED = "SUPPRESSED_IRRELEVANT_PERSON_IN_BALANCED"
ALLOWED_RELEVANT_PERSON = "ALLOWED_RELEVANT_PERSON"
ALLOWED_DETAILED_CONTEXT = "ALLOWED_DETAILED_CONTEXT"
ALLOWED_NEW_DANGER_EPISODE = "ALLOWED_NEW_DANGER_EPISODE"
ALLOWED_DANGER_REPEAT = "ALLOWED_DANGER_REPEAT"
SUPPRESSED_DETAILED_REPEAT = "SUPPRESSED_DETAILED_REPEAT"
SUPPRESSED_REPEAT_LIMIT = "SUPPRESSED_REPEAT_LIMIT"
SUPPRESSED_UNCHANGED_SUMMARY = "SUPPRESSED_UNCHANGED_SUMMARY"
SUPPRESSED_SUMMARY_GROUP_LIMIT = "SUPPRESSED_SUMMARY_GROUP_LIMIT"
# Realized through SpeechQueue.evict_track rather than emitted directly
# by this policy.
REPLACED_BY_HIGHER_DANGER = "REPLACED_BY_HIGHER_DANGER"

_NOT_RELEVANT_RANK = _LEVEL_RANK[LEVEL_2_MOVING_NEARBY]

# The only hazard family this policy currently tracks episodes for --
# _apply_danger_policy() is only ever invoked for LEVEL_5_HIGH_DANGER
# events (gated in finalize_events()). Kept as an explicit named
# constant, not hardcoded inline, so a future family (e.g. a
# LEVEL_4_PATH_CONFLICT episode concept) could be added without
# colliding with this one, even for the same class/region.
HAZARD_FAMILY_IMMEDIATE_DANGER = "IMMEDIATE_DANGER"

# Defensive fallback when a summary message exceeds a mode's configured
# max_summary_words -- short, fixed, and deliberately non-urgent (never
# competes with an actual WARNING-tier message for attention).
_SUMMARY_WORD_LIMIT_FALLBACK = "Multiple objects ahead."


def parse_operating_mode(value: str) -> str:
    """Centralized mode validation -- the ONLY place a raw string becomes
    a validated mode, used identically by the CLI and config loading.

    Raises:
        ValueError: If `value` (case-insensitively) isn't "minimal",
            "balanced", or "detailed".
    """
    normalized = value.strip().upper()
    if normalized not in _VALID_MODES:
        raise ValueError(
            f"Invalid operating mode {value!r} -- must be one of "
            f"{', '.join(m.lower() for m in _VALID_MODES)}"
        )
    return normalized


class _EpisodeState:
    """Mutable per-(normalized_class, region, hazard_family) Level-5
    danger-episode state, carried across frames. Not a frozen model --
    internal bookkeeping.

    Deliberately NOT keyed by track_id or operating mode:
      - track_id alone is insufficient (a brief occlusion reassigns a
        new track ID to the same physical hazard -- this codebase has
        no visual re-identification, and this correction does not add
        one), and requiring it would break exactly the continuity this
        state is meant to preserve.
      - operating mode is redundant: OperatingModePolicy.set_mode()
        already clears ALL episode state on any mode switch, so a
        mode-scoped key would never actually distinguish anything a
        fresh dict wouldn't already.
    A different CLASS or REGION is a different key by construction, so
    a vehicle danger episode on the left and a bicycle danger episode on
    the left are fully independent -- neither's cycle count or grace
    window can be consumed or corrupted by the other. Known, disclosed
    limitation: two DIFFERENT physical objects of the SAME class in the
    SAME region, both dangerous within the same grace window, still
    share one episode (indistinguishable without visual re-ID) -- see
    docs/OPERATING_MODES.md."""

    __slots__ = ("cycle_count", "last_seen_time", "last_admitted_time")

    def __init__(self, now: float) -> None:
        self.cycle_count = 0
        self.last_seen_time = now
        self.last_admitted_time: float | None = None


class OperatingModePolicy:
    """Filters and reshapes already-resolved hazard events by operating
    mode. See module docstring for the two-stage pipeline placement.

    Args:
        mode: Initial mode ("MINIMAL"/"BALANCED"/"DETAILED" -- already
            validated by the caller via parse_operating_mode()).
        mode_source: "CLI", "CONFIG", or "DEFAULT" -- how `mode` was
            selected, carried into every trace for observability.
        profiles: mode -> OperatingModeProfile, one entry per valid mode.
    """

    def __init__(
        self, mode: str, mode_source: str, profiles: dict[str, OperatingModeProfile]
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        missing = set(_VALID_MODES) - set(profiles)
        if missing:
            raise ValueError(f"profiles missing entries for: {sorted(missing)}")

        self._mode = mode
        self._mode_source = mode_source
        self._profiles = profiles
        self._episodes: dict[str, _EpisodeState] = {}
        self._last_summary_by_region: dict[str, str] = {}

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def mode_source(self) -> str:
        return self._mode_source

    def set_mode(self, mode: str, source: str = "RUNTIME") -> None:
        """The future-app-facing entry point: validates centrally, then
        clears this policy's own danger-episode ledger and summary-dedup
        state (no old-mode pending state bleeds into the new mode).
        Never touches SpeechQueue/EventPolicy/perception state, never
        replays the startup announcement, never restarts detection --
        none of those are reachable from here."""
        self._mode = parse_operating_mode(mode)
        self._mode_source = source
        self._episodes.clear()
        self._last_summary_by_region.clear()

    def _profile(self) -> OperatingModeProfile:
        return self._profiles[self._mode]

    # --- stage 1: eligibility -------------------------------------------

    def filter_events(
        self, hazard_results: dict[int, AudioHazardResult], now: float
    ) -> tuple[dict[int, AudioHazardResult], dict[int, OperatingModeTrace]]:
        """Drops ineligible tracks entirely, before any wording exists.
        Uses AudioHazardResult.class_name directly -- no tracked_objects
        needed."""
        profile = self._profile()
        allowed: dict[int, AudioHazardResult] = {}
        traces: dict[int, OperatingModeTrace] = {}

        for track_id, hazard in hazard_results.items():
            if hazard.class_name == PERSON_CLASS:
                allowed_flag, reason = self._person_eligibility(hazard, profile)
            else:
                allowed_flag = profile.announce_traffic
                reason = ALLOWED_TRAFFIC_EVENT if allowed_flag else SUPPRESSED_TRAFFIC_DISABLED

            if allowed_flag:
                allowed[track_id] = hazard

            traces[track_id] = OperatingModeTrace(
                track_id=track_id,
                operating_mode=self._mode,
                mode_source=self._mode_source,
                event_allowed_by_mode=allowed_flag,
                mode_filter_reason=reason,
                mode_warning_repeat_enabled=profile.highest_danger_repeat_enabled,
                hazard_episode_key=None,
                hazard_episode_cycle_count=0,
                repeat_due_timestamp=None,
                repeat_admitted=False,
                repeat_suppression_reason=None,
                detailed_summary_group_count=0,
                detailed_summary_word_count=0,
                final_mode_adjusted_message=None,
            )
            logger.debug("Operating-mode filter: %s", traces[track_id])

        return allowed, traces

    @staticmethod
    def _person_eligibility(
        hazard: AudioHazardResult, profile: OperatingModeProfile
    ) -> tuple[bool, str]:
        if profile.person_eligibility == PERSON_INELIGIBLE:
            return False, SUPPRESSED_PERSON_IN_MINIMAL
        if profile.person_eligibility == PERSON_BROAD:
            return True, ALLOWED_DETAILED_CONTEXT
        # RELEVANT_ONLY -- close/approaching/path-crossing all map onto
        # the existing hazard-level ladder; "ordinary far movement"
        # (Level 1) is exactly "not relevant."
        relevant = _LEVEL_RANK.get(hazard.level, 0) >= _NOT_RELEVANT_RANK
        return (
            (True, ALLOWED_RELEVANT_PERSON)
            if relevant
            else (False, SUPPRESSED_IRRELEVANT_PERSON_IN_BALANCED)
        )

    # --- stage 2: danger message form + repeat policy + summary caps ----

    def finalize_events(
        self, events: list[AudioEvent], now: float
    ) -> tuple[list[AudioEvent], list[OperatingModeTrace]]:
        """Applied to SceneSummarizer's output. Only Level-5 events and
        non-WARNING scene_summary events are ever touched -- Levels 1-4
        and Minimal/Balanced summaries pass through unchanged."""
        profile = self._profile()
        result: list[AudioEvent] = []
        traces: list[OperatingModeTrace] = []
        summary_group_count = 0

        for event in events:
            if event.event_type == LEVEL_5_HIGH_DANGER:
                adjusted, trace = self._apply_danger_policy(event, profile, now)
                traces.append(trace)
                if adjusted is not None:
                    result.append(adjusted)
                continue

            if event.event_type == SCENE_SUMMARY and event.priority != PRIORITY_WARNING:
                adjusted, trace, summary_group_count = self._apply_summary_policy(
                    event, profile, summary_group_count
                )
                traces.append(trace)
                if adjusted is not None:
                    result.append(adjusted)
                continue

            result.append(event)

        for trace in traces:
            logger.debug("Operating-mode finalize: %s", trace)
        return result, traces

    def _apply_danger_policy(self, event: AudioEvent, profile, now: float):
        region = self._region_from_key(event.key)
        # AudioHazardResolver's Level-5 template always starts with
        # "Warning! {Name} ..." -- the same class name AudioEventBuilder
        # already rendered is reused here (lowercased) as the episode
        # identity's class component, so a vehicle and a bicycle in the
        # same region never share an episode.
        name = self._extract_name(event.message)
        episode_key = f"{name.lower()}:{region}:{HAZARD_FAMILY_IMMEDIATE_DANGER}"
        state = self._episodes.get(episode_key)
        # Reuses the same repeat-delay value as the semantic grace window
        # for track-ID-churn continuity (see _EpisodeState's docstring) --
        # one documented value serving both purposes, rather than a
        # second unused config key.
        grace = profile.highest_danger_repeat_delay_seconds

        active = state is not None and (now - state.last_seen_time) < grace
        if not active:
            state = _EpisodeState(now)
            self._episodes[episode_key] = state

        state.last_seen_time = now

        if not active:
            state.cycle_count = 1
            state.last_admitted_time = now
            admitted, repeat_admitted, reason = True, False, ALLOWED_NEW_DANGER_EPISODE
        elif not profile.highest_danger_repeat_enabled:
            admitted, repeat_admitted, reason = False, False, SUPPRESSED_DETAILED_REPEAT
        elif state.cycle_count >= profile.highest_danger_max_cycles:
            admitted, repeat_admitted, reason = False, False, SUPPRESSED_REPEAT_LIMIT
        elif (now - (state.last_admitted_time or 0.0)) < grace:
            admitted, repeat_admitted, reason = False, False, SUPPRESSED_REPEAT_LIMIT
        else:
            state.cycle_count += 1
            state.last_admitted_time = now
            admitted, repeat_admitted, reason = True, True, ALLOWED_DANGER_REPEAT

        final_message = None
        adjusted_event = None
        if admitted:
            final_message = self._level5_message(name, region, self._mode)
            adjusted_event = dataclasses.replace(event, message=final_message)

        repeat_due = (
            (state.last_admitted_time or now) + grace if state.last_admitted_time else None
        )
        trace = OperatingModeTrace(
            track_id=event.track_id,
            operating_mode=self._mode,
            mode_source=self._mode_source,
            event_allowed_by_mode=admitted,
            mode_filter_reason=reason,
            mode_warning_repeat_enabled=profile.highest_danger_repeat_enabled,
            hazard_episode_key=episode_key,
            hazard_episode_cycle_count=state.cycle_count,
            repeat_due_timestamp=repeat_due,
            repeat_admitted=repeat_admitted,
            repeat_suppression_reason=None if admitted else reason,
            detailed_summary_group_count=0,
            detailed_summary_word_count=0,
            final_mode_adjusted_message=final_message,
        )
        return adjusted_event, trace

    def _apply_summary_policy(self, event: AudioEvent, profile, summary_group_count: int):
        if not profile.structured_scene_summaries:
            return event, self._passthrough_trace(event, profile), summary_group_count

        region = self._region_from_key(event.key)

        if self._last_summary_by_region.get(region) == event.message:
            trace = self._suppressed_summary_trace(
                event, profile, SUPPRESSED_UNCHANGED_SUMMARY, summary_group_count
            )
            return None, trace, summary_group_count

        if summary_group_count >= profile.max_summary_object_groups:
            trace = self._suppressed_summary_trace(
                event, profile, SUPPRESSED_SUMMARY_GROUP_LIMIT, summary_group_count
            )
            return None, trace, summary_group_count

        original_message = event.message
        word_count = len(original_message.split())
        if word_count > profile.max_summary_words:
            # Defensive -- the deterministic SceneSummarizer templates
            # rarely approach this ceiling, but it's structurally
            # enforced, not just assumed. A short, fixed, non-urgent
            # fallback rather than truncating mid-sentence.
            event = dataclasses.replace(event, message=_SUMMARY_WORD_LIMIT_FALLBACK)
            word_count = len(_SUMMARY_WORD_LIMIT_FALLBACK.split())

        summary_group_count += 1
        self._last_summary_by_region[region] = original_message
        trace = OperatingModeTrace(
            track_id=None,
            operating_mode=self._mode,
            mode_source=self._mode_source,
            event_allowed_by_mode=True,
            mode_filter_reason=ALLOWED_DETAILED_CONTEXT,
            mode_warning_repeat_enabled=profile.highest_danger_repeat_enabled,
            hazard_episode_key=None,
            hazard_episode_cycle_count=0,
            repeat_due_timestamp=None,
            repeat_admitted=False,
            repeat_suppression_reason=None,
            detailed_summary_group_count=summary_group_count,
            detailed_summary_word_count=word_count,
            final_mode_adjusted_message=event.message,
        )
        return event, trace, summary_group_count

    def _passthrough_trace(self, event: AudioEvent, profile) -> OperatingModeTrace:
        return OperatingModeTrace(
            track_id=event.track_id, operating_mode=self._mode, mode_source=self._mode_source,
            event_allowed_by_mode=True, mode_filter_reason=ALLOWED_TRAFFIC_EVENT,
            mode_warning_repeat_enabled=profile.highest_danger_repeat_enabled,
            hazard_episode_key=None, hazard_episode_cycle_count=0, repeat_due_timestamp=None,
            repeat_admitted=False, repeat_suppression_reason=None,
            detailed_summary_group_count=0, detailed_summary_word_count=0,
            final_mode_adjusted_message=event.message,
        )

    def _suppressed_summary_trace(self, event, profile, reason, group_count) -> OperatingModeTrace:
        return OperatingModeTrace(
            track_id=event.track_id, operating_mode=self._mode, mode_source=self._mode_source,
            event_allowed_by_mode=False, mode_filter_reason=reason,
            mode_warning_repeat_enabled=profile.highest_danger_repeat_enabled,
            hazard_episode_key=None, hazard_episode_cycle_count=0, repeat_due_timestamp=None,
            repeat_admitted=False, repeat_suppression_reason=reason,
            detailed_summary_group_count=group_count,
            detailed_summary_word_count=len(event.message.split()),
            final_mode_adjusted_message=None,
        )

    @staticmethod
    def _region_from_key(key: str) -> str:
        parts = key.split(":")
        return parts[2] if len(parts) >= 3 else "center"

    @staticmethod
    def _extract_name(message: str) -> str:
        # AudioHazardResolver's Level-5 template always starts with
        # "Warning! {Name} ..." -- extracting from the rendered text
        # mirrors the same cross-stage-parsing convention SceneSummarizer
        # already uses (checking a message's "Possible" prefix) rather
        # than adding a new field just for this one rewrite.
        after_bang = message.split("!", 1)[1].strip()
        return after_bang.split(" ", 1)[0]

    @staticmethod
    def _level5_message(name: str, region: str, mode: str) -> str:
        if region == "center":
            phrase = f"{name} directly ahead"
        else:
            direction = {"left": "on the left", "right": "on the right"}.get(region, "ahead")
            phrase = f"{name} {direction}"
        if mode == DETAILED:
            return f"Warning! {phrase}!"
        return f"Warning! {phrase}! {phrase}!"
