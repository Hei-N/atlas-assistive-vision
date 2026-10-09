"""Five-level audio hazard decision.

Maps a ThreatAssessment (src/threat_assessment.py) onto this module's
spoken-hazard vocabulary and message templates. The level itself, including
escalation, de-escalation and persistence timing, is decided by
ThreatAssessmentEngine; this module owns wording only and keeps no
per-track level state.

    LEVEL_1_MOVING_FAR    -- confirmed movement, FAR, no approach/conflict
    LEVEL_2_MOVING_NEARBY -- confirmed movement, NEAR, no approach/conflict
    LEVEL_3_APPROACHING   -- MID/NEAR + confirmed closing motion
    LEVEL_4_PATH_CONFLICT -- resolved trajectory currently intersects the corridor
    LEVEL_5_HIGH_DANGER   -- NEAR + confirmed approaching + resolved intersection,
                             all three confident (never from one uncertain cue)

These names are used by HazardTrace, AudioEvent.event_type, delivery-profile
lookups and OperatingModePolicy episode keys. They are translated from the
canonical ThreatLevel names (NORMAL_MOVEMENT, NEARBY_PRESENCE,
CONFIRMED_APPROACH, PATH_CONFLICT, IMMEDIATE_DANGER).

Whether and how often an event is spoken is decided downstream by
src/audio/event_policy.py (cooldown) and src/audio/operating_mode_policy.py
(Level-5 episode and repeat policy).

See docs/AUDIO_ALERT_LEVELS.md for thresholds and message templates.

Pure logic with no OpenCV, subprocess or network calls.
"""

from __future__ import annotations

import logging

from src.models import AudioHazardResult, DeliveryProfile, HazardTrace, ThreatAssessment
from src.threat_assessment import CONFIRMED_APPROACH as _THREAT_CONFIRMED_APPROACH
from src.threat_assessment import IMMEDIATE_DANGER as _THREAT_IMMEDIATE_DANGER
from src.threat_assessment import NEARBY_PRESENCE as _THREAT_NEARBY_PRESENCE
from src.threat_assessment import NORMAL_MOVEMENT as _THREAT_NORMAL_MOVEMENT
from src.threat_assessment import PATH_CONFLICT as _THREAT_PATH_CONFLICT

logger = logging.getLogger("atlas")

LEVEL_1_MOVING_FAR = "LEVEL_1_MOVING_FAR"
LEVEL_2_MOVING_NEARBY = "LEVEL_2_MOVING_NEARBY"
LEVEL_3_APPROACHING = "LEVEL_3_APPROACHING"
LEVEL_4_PATH_CONFLICT = "LEVEL_4_PATH_CONFLICT"
LEVEL_5_HIGH_DANGER = "LEVEL_5_HIGH_DANGER"

PRIORITY_WARNING = "WARNING"
PRIORITY_INFORMATIONAL = "INFORMATIONAL"

_LEVEL_RANK = {
    LEVEL_1_MOVING_FAR: 1,
    LEVEL_2_MOVING_NEARBY: 2,
    LEVEL_3_APPROACHING: 3,
    LEVEL_4_PATH_CONFLICT: 4,
    LEVEL_5_HIGH_DANGER: 5,
}

_PRIORITY_BY_LEVEL = {
    LEVEL_1_MOVING_FAR: PRIORITY_INFORMATIONAL,
    LEVEL_2_MOVING_NEARBY: PRIORITY_INFORMATIONAL,
    LEVEL_3_APPROACHING: PRIORITY_WARNING,
    LEVEL_4_PATH_CONFLICT: PRIORITY_WARNING,
    LEVEL_5_HIGH_DANGER: PRIORITY_WARNING,
}

_DELIVERY_PROFILE_BY_LEVEL = {
    LEVEL_1_MOVING_FAR: "level_1",
    LEVEL_2_MOVING_NEARBY: "level_2",
    LEVEL_3_APPROACHING: "level_3",
    LEVEL_4_PATH_CONFLICT: "level_4",
    LEVEL_5_HIGH_DANGER: "level_5",
}

# Translates src/threat_assessment.py's canonical ThreatLevel vocabulary to
# this module's own, pre-existing level names -- the ONLY place that
# mapping happens; nothing in this module re-derives which level an
# object is at.
_AUDIO_LEVEL_BY_THREAT_LEVEL = {
    _THREAT_NORMAL_MOVEMENT: LEVEL_1_MOVING_FAR,
    _THREAT_NEARBY_PRESENCE: LEVEL_2_MOVING_NEARBY,
    _THREAT_CONFIRMED_APPROACH: LEVEL_3_APPROACHING,
    _THREAT_PATH_CONFLICT: LEVEL_4_PATH_CONFLICT,
    _THREAT_IMMEDIATE_DANGER: LEVEL_5_HIGH_DANGER,
}

# Exactly as specified -- never speak a raw class name, a track ID, a
# class ID, a confidence value, or a pixel speed.
_OBJECT_NAME_MAP = {
    "car": "Vehicle",
    "truck": "Vehicle",
    "bus": "Vehicle",
    "motorcycle": "Motorcycle",
    "bicycle": "Bicycle",
    "person": "Person",
}

_DIRECTION_PHRASE_MAP = {"left": "from the left", "right": "from the right", "center": "ahead"}
_ON_DIRECTION_PHRASE_MAP = {"left": "on the left", "right": "on the right", "center": "ahead"}


def _level_1_message(name: str, direction: str, uncertain: bool) -> str:
    if uncertain:
        return f"Possible {name.lower()} movement {direction}."
    return f"{name} moving {direction}."


def _level_2_message(name: str, direction: str, on_direction: str, uncertain: bool) -> str:
    if uncertain:
        # Reuses the exact Level-1 uncertain template (not "on the
        # left") -- deliberately, so there's only one uncertain-movement
        # phrasing in the whole system rather than two near-duplicates.
        return f"Possible {name.lower()} movement {direction}."
    return f"{name} moving nearby {on_direction}."


def _level_3_message(name: str, region: str, direction: str, uncertain: bool) -> str:
    if uncertain:
        # Forward uniquely uses "directly ahead" here -- the approved
        # APPROACH-family uncertain template -- distinct from every
        # other uncertain/confident template's plain "ahead" (the
        # shared _DIRECTION_PHRASE_MAP is intentionally NOT changed;
        # this is a one-off branch just for this template).
        forward_or_direction = "directly ahead" if region == "center" else direction
        return f"Possible {name.lower()} approach {forward_or_direction}."
    return f"{name} approaching {direction}."


def _level_4_message(name: str, direction: str, uncertain: bool) -> str:
    if uncertain:
        return f"Possible {name.lower()} crossing {direction}."
    return f"Caution. {name} crossing your path {direction}. Please wait."


def _level_5_message(name: str, region: str, direction: str) -> str:
    # The spec's own examples use inconsistent verbs across directions
    # ("approaching" for left, "entering your path" for right) -- a
    # deterministic system needs one template per condition, not
    # per-direction variation with no functional difference, so this
    # uses "approaching" uniformly for left/right and "ahead" alone for
    # center (matching the spec's own center example). Never uncertain:
    # Level 5 is structurally unreachable when any contributing signal
    # is uncertain (see ThreatAssessmentEngine/classify_instantaneous_
    # threat).
    if region == "center":
        return f"Warning! {name} ahead. Please wait."
    return f"Warning! {name} approaching {direction}. Please wait."


_MESSAGE_BUILDERS = {
    LEVEL_1_MOVING_FAR: lambda name, region, direction, on_direction, uncertain: (
        _level_1_message(name, direction, uncertain)
    ),
    LEVEL_2_MOVING_NEARBY: lambda name, region, direction, on_direction, uncertain: (
        _level_2_message(name, direction, on_direction, uncertain)
    ),
    LEVEL_3_APPROACHING: lambda name, region, direction, on_direction, uncertain: (
        _level_3_message(name, region, direction, uncertain)
    ),
    LEVEL_4_PATH_CONFLICT: lambda name, region, direction, on_direction, uncertain: (
        _level_4_message(name, direction, uncertain)
    ),
    LEVEL_5_HIGH_DANGER: lambda name, region, direction, on_direction, uncertain: (
        _level_5_message(name, region, direction)
    ),
}


class AudioHazardResolver:
    """Translates canonical ThreatAssessments into AudioHazardResults --
    wording and audio-only eligibility (announce_level_1/2) only, no
    threat-level decision or temporal state of its own.

    Args:
        announce_level_1: When False, LEVEL_1_MOVING_FAR is fully
            suppressed (no result, no trace hazard_level) -- far-away
            movement can otherwise become noisy in a busy scene. This is
            an audio-eligibility toggle, not a reclassification: the
            object's canonical level is still NORMAL_MOVEMENT regardless.
        announce_level_2: Same, for LEVEL_2_MOVING_NEARBY.
        delivery_profiles: Level-name -> DeliveryProfile, used ONLY to
            populate HazardTrace's delivery_rate_wpm/delivery_cue_enabled
            fields for observability -- AudioWorker independently looks
            up the same config to actually apply it; this resolver never
            decides playback behavior itself.
    """

    def __init__(
        self,
        announce_level_1: bool,
        announce_level_2: bool,
        delivery_profiles: dict[str, DeliveryProfile],
    ) -> None:
        self._announce_level_1 = announce_level_1
        self._announce_level_2 = announce_level_2
        self._delivery_profiles = delivery_profiles

    def resolve_for_tracks(
        self,
        threat_assessments: dict[int, ThreatAssessment],
        now: float,
    ) -> tuple[dict[int, AudioHazardResult], dict[int, HazardTrace]]:
        """Translate this frame's canonical ThreatAssessments in one call.

        Returns (hazard_results, hazard_traces). hazard_results only
        contains tracks eligible for an AudioHazardResult this frame
        (post class/region-mapping safety check, post announce-toggle).
        hazard_traces contains one entry for every assessed track, so a
        suppressed/silent decision is still observable. `now` is accepted
        for interface symmetry with the previous stateful version and for
        any future audio-only (not threat-classification) use -- this
        resolver does not currently use it for anything itself.
        """
        del now  # unused -- see docstring; kept for interface stability

        results: dict[int, AudioHazardResult] = {}
        traces: dict[int, HazardTrace] = {}

        for track_id, assessment in threat_assessments.items():
            result, trace = self._resolve_assessment(assessment)
            if result is not None:
                results[track_id] = result
            traces[track_id] = trace
            logger.debug("Hazard trace: %s", trace)

        return results, traces

    def _resolve_assessment(
        self, assessment: ThreatAssessment
    ) -> tuple[AudioHazardResult | None, HazardTrace]:
        level = _AUDIO_LEVEL_BY_THREAT_LEVEL[assessment.level]
        name = _OBJECT_NAME_MAP.get(assessment.object_class)
        direction = _DIRECTION_PHRASE_MAP.get(assessment.region)
        on_direction = _ON_DIRECTION_PHRASE_MAP.get(assessment.region)

        eligible = name is not None and direction is not None
        if eligible and level == LEVEL_1_MOVING_FAR and not self._announce_level_1:
            eligible = False
        elif eligible and level == LEVEL_2_MOVING_NEARBY and not self._announce_level_2:
            eligible = False

        result: AudioHazardResult | None = None
        message: str | None = None
        if eligible:
            message = _MESSAGE_BUILDERS[level](
                name, assessment.region, direction, on_direction, assessment.uncertain
            )
            result = AudioHazardResult(
                track_id=assessment.track_id,
                level=level,
                priority=_PRIORITY_BY_LEVEL[level],
                class_name=assessment.object_class,
                region=assessment.region,
                proximity_zone=assessment.proximity_zone,
                approach_state=assessment.approach_state,
                intersects_corridor=assessment.intersects_corridor,
                uncertain=assessment.uncertain,
                reason_codes=assessment.reason_codes,
                recommended_message=message,
                delivery_profile=_DELIVERY_PROFILE_BY_LEVEL[level],
            )

        profile = self._delivery_profiles.get(_DELIVERY_PROFILE_BY_LEVEL.get(level, ""))
        trace = HazardTrace(
            track_id=assessment.track_id,
            proximity_zone=assessment.proximity_zone,
            approach_state=assessment.approach_state,
            approach_confidence=assessment.confidence,
            # Frames the current level has been held (ThreatAssessment.persistence_frames).
            approach_confirmation_frames=assessment.persistence_frames,
            approach_evidence_flags=assessment.approach_evidence_flags,
            hazard_level=level if eligible else None,
            hazard_reason_codes=assessment.reason_codes if eligible else (),
            delivery_rate_wpm=profile.rate_wpm if profile else None,
            delivery_cue_enabled=profile.cue_enabled if profile else None,
            # Transitions are carried by ThreatAssessment.escalated.
            previous_hazard_level=None,
            escalation=assessment.escalated,
            deescalation_suppressed=assessment.deescalation_pending,
            final_spoken_message=message,
        )
        return result, trace
