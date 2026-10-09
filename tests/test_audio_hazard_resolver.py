"""Unit tests for src/audio/audio_hazard_resolver.py -- translation of a
canonical ThreatAssessment (src/threat_assessment.py) into wording. This
module owns NO threat-level decision or temporal state of its own
anymore (see src/threat_assessment.py's ThreatAssessmentEngine, tested in
tests/test_threat_assessment.py) -- these tests build ThreatAssessment
objects directly, matching the resolver's actual (now stateless) input
shape, and never re-derive the decision tree here.
"""

import pytest

from src.audio.audio_hazard_resolver import (
    LEVEL_1_MOVING_FAR,
    LEVEL_2_MOVING_NEARBY,
    LEVEL_3_APPROACHING,
    LEVEL_4_PATH_CONFLICT,
    LEVEL_5_HIGH_DANGER,
    PRIORITY_INFORMATIONAL,
    PRIORITY_WARNING,
    AudioHazardResolver,
)
from src.models import DeliveryProfile, ThreatAssessment
from src.motion.approach_estimator import APPROACHING, APPROACHING_UNCERTAIN, NOT_APPROACHING
from src.motion.relative_proximity_estimator import FAR, NEAR, UNKNOWN
from src.threat_assessment import (
    CONFIRMED_APPROACH,
    IMMEDIATE_DANGER,
    NEARBY_PRESENCE,
    NORMAL_MOVEMENT,
    PATH_CONFLICT,
    ThreatAssessmentEngine,
)


def make_assessment(
    track_id: int = 1,
    object_class: str = "car",
    region: str = "left",
    level: str = NORMAL_MOVEMENT,
    uncertain: bool = False,
    proximity_zone: str = FAR,
    approach_state: str = NOT_APPROACHING,
    intersects_corridor: bool = False,
    reason_codes: tuple = (),
    escalated: bool = False,
    deescalation_pending: bool = False,
    persistence_frames: int = 1,
) -> ThreatAssessment:
    return ThreatAssessment(
        track_id=track_id, object_class=object_class, region=region, level=level,
        confidence=1.0, uncertain=uncertain, reason_codes=reason_codes,
        proximity_zone=proximity_zone, approach_state=approach_state,
        approach_evidence_flags=(), intersects_corridor=intersects_corridor,
        corridor_intersection_uncertain=False, time_to_conflict_seconds=None,
        persistence_frames=persistence_frames, escalated=escalated,
        deescalation_pending=deescalation_pending, frame_index=None, timestamp=0.0,
    )


def make_profiles() -> dict[str, DeliveryProfile]:
    return {
        f"level_{i}": DeliveryProfile(rate_wpm=170 + i * 5, cue_enabled=i >= 4, cue_path=None, cue_gain=0.0)
        for i in range(1, 6)
    }


def make_resolver(announce_level_1: bool = True, announce_level_2: bool = True) -> AudioHazardResolver:
    return AudioHazardResolver(
        announce_level_1=announce_level_1,
        announce_level_2=announce_level_2,
        delivery_profiles=make_profiles(),
    )


def resolve_one(resolver, assessment, now=0.0):
    results, traces = resolver.resolve_for_tracks({assessment.track_id: assessment}, now)
    return results.get(assessment.track_id), traces[assessment.track_id]


# --- 1: far lateral vehicle -> Level 1, not approaching --------------


def test_far_lateral_vehicle_is_level_1_not_approaching() -> None:
    resolver = make_resolver()
    assessment = make_assessment(region="left", level=NORMAL_MOVEMENT, proximity_zone=FAR)
    result, _ = resolve_one(resolver, assessment)

    assert result.level == LEVEL_1_MOVING_FAR
    assert result.priority == PRIORITY_INFORMATIONAL
    assert result.recommended_message == "Vehicle moving from the left."


# --- 2: nearby lateral vehicle -> Level 2, not approaching ------------


def test_nearby_lateral_vehicle_is_level_2_not_approaching() -> None:
    resolver = make_resolver()
    assessment = make_assessment(region="left", level=NEARBY_PRESENCE, proximity_zone=NEAR)
    result, _ = resolve_one(resolver, assessment)

    assert result.level == LEVEL_2_MOVING_NEARBY
    assert result.recommended_message == "Vehicle moving nearby on the left."


# --- 3: nearby parked vehicle -> no movement warning ------------------
# (STATIONARY objects never get a ThreatAssessment at all -- see
# src/threat_assessment.py's classify_instantaneous_threat -- so there is
# nothing for AudioHazardResolver to translate; simulated here as an
# empty assessments dict, matching what ThreatAssessmentEngine.
# assess_for_tracks() actually returns for such a track.)


def test_nearby_parked_vehicle_produces_no_event() -> None:
    resolver = make_resolver()
    results, traces = resolver.resolve_for_tracks({}, now=0.0)
    assert results == {}
    assert traces == {}


# --- 8: resolved corridor intersection -> Level 4 ---------------------


def test_resolved_intersection_is_level_4() -> None:
    resolver = make_resolver()
    assessment = make_assessment(
        region="left", level=PATH_CONFLICT, intersects_corridor=True,
        reason_codes=("CORRIDOR_INTERSECTS", "PATH_CONFLICT_PERSISTENT"),
    )
    result, _ = resolve_one(resolver, assessment)

    assert result.level == LEVEL_4_PATH_CONFLICT
    assert result.priority == PRIORITY_WARNING
    assert result.recommended_message == "Caution. Vehicle crossing your path from the left. Please wait."


# --- 9: NEAR + APPROACHING + intersection -> Level 5 -------------------


def test_near_approaching_and_intersecting_is_level_5() -> None:
    resolver = make_resolver()
    assessment = make_assessment(
        region="left", level=IMMEDIATE_DANGER, proximity_zone=NEAR,
        approach_state=APPROACHING, intersects_corridor=True,
    )
    result, _ = resolve_one(resolver, assessment)

    assert result.level == LEVEL_5_HIGH_DANGER
    assert result.uncertain is False
    assert result.recommended_message == "Warning! Vehicle approaching from the left. Please wait."


# --- 10/11: nearness alone / movement alone never reach Level 5 -------
# (These are classify_instantaneous_threat's job, thoroughly covered in
# tests/test_threat_assessment.py -- here we only confirm the resolver
# doesn't second-guess a canonical non-Level-5 assessment into Level 5.)


def test_nearness_alone_never_reaches_level_5() -> None:
    resolver = make_resolver()
    assessment = make_assessment(region="left", level=NEARBY_PRESENCE, proximity_zone=NEAR)
    result, _ = resolve_one(resolver, assessment)
    assert result.level != LEVEL_5_HIGH_DANGER
    assert result.level == LEVEL_2_MOVING_NEARBY


def test_movement_alone_never_reaches_level_5() -> None:
    resolver = make_resolver()
    assessment = make_assessment(region="left", level=NORMAL_MOVEMENT, proximity_zone=FAR)
    result, _ = resolve_one(resolver, assessment)
    assert result.level != LEVEL_5_HIGH_DANGER


# --- 12: uncertain evidence never reaches Level 5 (capped at 3 or 4) --


def test_uncertain_intersection_caps_at_level_4() -> None:
    resolver = make_resolver()
    assessment = make_assessment(
        region="left", level=PATH_CONFLICT, uncertain=True,
        approach_state=APPROACHING, proximity_zone=NEAR, intersects_corridor=True,
    )
    result, _ = resolve_one(resolver, assessment)

    assert result.level == LEVEL_4_PATH_CONFLICT
    assert result.uncertain is True
    assert result.recommended_message == "Possible vehicle crossing from the left."


def test_uncertain_approach_caps_at_level_3() -> None:
    resolver = make_resolver()
    assessment = make_assessment(
        region="left", level=CONFIRMED_APPROACH, uncertain=True,
        approach_state=APPROACHING_UNCERTAIN, proximity_zone=NEAR,
    )
    result, _ = resolve_one(resolver, assessment)

    assert result.level == LEVEL_3_APPROACHING
    assert result.uncertain is True
    assert result.recommended_message == "Possible vehicle approach from the left."


# --- object-level uncertainty: exact approved templates, every direction -


def test_uncertain_movement_level_1_right() -> None:
    resolver = make_resolver()
    assessment = make_assessment(region="right", level=NORMAL_MOVEMENT, uncertain=True, proximity_zone=FAR)
    result, _ = resolve_one(resolver, assessment)

    assert result.level == LEVEL_1_MOVING_FAR
    assert result.uncertain is True
    assert result.recommended_message == "Possible vehicle movement from the right."


def test_uncertain_movement_level_1_forward() -> None:
    resolver = make_resolver()
    assessment = make_assessment(region="center", level=NORMAL_MOVEMENT, uncertain=True, proximity_zone=FAR)
    result, _ = resolve_one(resolver, assessment)

    assert result.level == LEVEL_1_MOVING_FAR
    assert result.uncertain is True
    assert result.recommended_message == "Possible vehicle movement ahead."


def test_uncertain_movement_level_2_forward() -> None:
    resolver = make_resolver()
    assessment = make_assessment(region="center", level=NEARBY_PRESENCE, uncertain=True, proximity_zone=NEAR)
    result, _ = resolve_one(resolver, assessment)

    assert result.level == LEVEL_2_MOVING_NEARBY
    assert result.uncertain is True
    assert result.recommended_message == "Possible vehicle movement ahead."


def test_uncertain_approach_level_3_right() -> None:
    resolver = make_resolver()
    assessment = make_assessment(
        region="right", level=CONFIRMED_APPROACH, uncertain=True,
        approach_state=APPROACHING_UNCERTAIN, proximity_zone=NEAR,
    )
    result, _ = resolve_one(resolver, assessment)

    assert result.level == LEVEL_3_APPROACHING
    assert result.uncertain is True
    assert result.recommended_message == "Possible vehicle approach from the right."


def test_uncertain_approach_level_3_forward_uses_directly_ahead() -> None:
    resolver = make_resolver()
    assessment = make_assessment(
        region="center", level=CONFIRMED_APPROACH, uncertain=True,
        approach_state=APPROACHING_UNCERTAIN, proximity_zone=NEAR,
    )
    result, _ = resolve_one(resolver, assessment)

    assert result.level == LEVEL_3_APPROACHING
    assert result.uncertain is True
    assert result.recommended_message == "Possible vehicle approach directly ahead."


def test_uncertain_crossing_level_4_right() -> None:
    resolver = make_resolver()
    assessment = make_assessment(
        region="right", level=PATH_CONFLICT, uncertain=True,
        approach_state=APPROACHING, proximity_zone=NEAR, intersects_corridor=True,
    )
    result, _ = resolve_one(resolver, assessment)

    assert result.level == LEVEL_4_PATH_CONFLICT
    assert result.uncertain is True
    assert result.recommended_message == "Possible vehicle crossing from the right."


def test_uncertain_crossing_level_4_forward() -> None:
    resolver = make_resolver()
    assessment = make_assessment(
        region="center", level=PATH_CONFLICT, uncertain=True,
        approach_state=APPROACHING, proximity_zone=NEAR, intersects_corridor=True,
    )
    result, _ = resolve_one(resolver, assessment)

    assert result.level == LEVEL_4_PATH_CONFLICT
    assert result.uncertain is True
    assert result.recommended_message == "Possible vehicle crossing ahead."


def test_uncertain_message_never_says_probably_or_maybe_safe() -> None:
    resolver = make_resolver()
    assessment = make_assessment(region="left", level=NORMAL_MOVEMENT, uncertain=True, proximity_zone=FAR)
    result, _ = resolve_one(resolver, assessment)

    lowered = result.recommended_message.lower()
    for forbidden in ("probably", "maybe safe", "likely safe", "uncertainty detected"):
        assert forbidden not in lowered


# --- 13/14/15: exact message prefixes -----------------------------------


def test_level_5_message_begins_with_warning() -> None:
    resolver = make_resolver()
    assessment = make_assessment(
        region="right", level=IMMEDIATE_DANGER, approach_state=APPROACHING,
        proximity_zone=NEAR, intersects_corridor=True,
    )
    result, _ = resolve_one(resolver, assessment)
    assert result.recommended_message.startswith("Warning!")


def test_level_4_message_begins_with_caution() -> None:
    resolver = make_resolver()
    assessment = make_assessment(region="right", level=PATH_CONFLICT, intersects_corridor=True)
    result, _ = resolve_one(resolver, assessment)
    assert result.recommended_message.startswith("Caution.")


def test_level_1_uses_neutral_wording() -> None:
    resolver = make_resolver()
    assessment = make_assessment(region="center", level=NORMAL_MOVEMENT, proximity_zone=FAR)
    result, _ = resolve_one(resolver, assessment)
    assert result.recommended_message == "Vehicle moving ahead."
    for word in ("Warning", "Caution", "Possible"):
        assert word not in result.recommended_message


# --- announce_level_1/2 toggles -------------------------------------------


def test_announce_level_1_false_suppresses_level_1() -> None:
    resolver = make_resolver(announce_level_1=False)
    assessment = make_assessment(region="left", level=NORMAL_MOVEMENT, proximity_zone=FAR)
    result, trace = resolve_one(resolver, assessment)
    assert result is None
    assert trace.hazard_level is None


def test_announce_level_2_false_suppresses_level_2() -> None:
    resolver = make_resolver(announce_level_2=False)
    assessment = make_assessment(region="left", level=NEARBY_PRESENCE, proximity_zone=NEAR)
    result, _ = resolve_one(resolver, assessment)
    assert result is None


def test_level_3_never_suppressed_by_announce_toggles() -> None:
    resolver = make_resolver(announce_level_1=False, announce_level_2=False)
    assessment = make_assessment(
        region="left", level=CONFIRMED_APPROACH, approach_state=APPROACHING, proximity_zone=NEAR,
    )
    result, _ = resolve_one(resolver, assessment)
    assert result is not None
    assert result.level == LEVEL_3_APPROACHING


# --- defensive: unmapped class/region ------------------------------------


def test_unmapped_class_produces_no_event() -> None:
    resolver = make_resolver()
    assessment = make_assessment(object_class="traffic_cone", region="left", level=NORMAL_MOVEMENT)
    result, trace = resolve_one(resolver, assessment)
    assert result is None
    assert trace.hazard_level is None


def test_unknown_proximity_folds_into_level_1() -> None:
    resolver = make_resolver()
    assessment = make_assessment(region="left", level=NORMAL_MOVEMENT, proximity_zone=UNKNOWN)
    result, _ = resolve_one(resolver, assessment)
    assert result.level == LEVEL_1_MOVING_FAR  # UNKNOWN folds into the neutral default


# --- delivery_profile field ------------------------------------------------


def test_delivery_profile_matches_level() -> None:
    resolver = make_resolver()
    assessment = make_assessment(
        region="left", level=IMMEDIATE_DANGER, approach_state=APPROACHING,
        proximity_zone=NEAR, intersects_corridor=True,
    )
    result, _ = resolve_one(resolver, assessment)
    assert result.delivery_profile == "level_5"


# --- SINGLE SOURCE OF TRUTH: no independent state, no override ---------------


def test_resolver_is_stateless_no_deescalation_hold_constructor_param() -> None:
    # The old 3.0s threat-classification hold no longer exists on this
    # resolver at all -- constructing it never accepts (or needs) a
    # deescalation_hold_seconds argument anymore.
    import inspect

    params = set(inspect.signature(AudioHazardResolver.__init__).parameters)
    assert "deescalation_hold_seconds" not in params


def test_resolver_has_no_recent_peak_or_hold_state() -> None:
    resolver = make_resolver()
    # No internal per-track de-escalation bookkeeping remains.
    assert not hasattr(resolver, "_recent_peak")
    assert not hasattr(resolver, "_deescalation_hold_seconds")


def test_resolver_mirrors_canonical_level_across_consecutive_frames_no_independent_hold() -> None:
    """Feeding a HIGH level then a LOW level for the same track, with no
    ThreatAssessmentEngine hold reflected in the second assessment
    (deescalation_pending=False, i.e. the canonical engine already
    decided the drop was warranted), must produce the LOW level
    immediately -- proving AudioHazardResolver applies no de-escalation
    hold of its own on top of whatever the canonical assessment says."""
    resolver = make_resolver()
    high = make_assessment(
        track_id=1, region="left", level=IMMEDIATE_DANGER, approach_state=APPROACHING,
        proximity_zone=NEAR, intersects_corridor=True,
    )
    result_high, _ = resolve_one(resolver, high, now=0.0)
    assert result_high.level == LEVEL_5_HIGH_DANGER

    # A fresh call, moments later, where the CANONICAL assessment itself
    # already reports a low level (as it would immediately after
    # ThreatAssessmentEngine's own hold expires) -- the old resolver-
    # owned hold would have suppressed this for up to 3s; the new
    # stateless resolver must not.
    low = make_assessment(track_id=1, region="left", level=NORMAL_MOVEMENT, proximity_zone=FAR)
    result_low, trace_low = resolve_one(resolver, low, now=0.05)
    assert result_low is not None
    assert result_low.level == LEVEL_1_MOVING_FAR
    assert trace_low.deescalation_suppressed is False


def test_escalation_and_deescalation_pending_are_copied_from_assessment() -> None:
    resolver = make_resolver()
    assessment = make_assessment(
        region="left", level=IMMEDIATE_DANGER, approach_state=APPROACHING,
        proximity_zone=NEAR, intersects_corridor=True, escalated=True, deescalation_pending=True,
    )
    _, trace = resolve_one(resolver, assessment)
    assert trace.escalation is True
    assert trace.deescalation_suppressed is True


def test_changing_canonical_level_changes_audio_mapping() -> None:
    """Same track/region/class, only ThreatAssessment.level differs --
    the mapped audio level must differ identically, one-to-one."""
    resolver = make_resolver()
    for threat_level, expected in (
        (NORMAL_MOVEMENT, LEVEL_1_MOVING_FAR),
        (NEARBY_PRESENCE, LEVEL_2_MOVING_NEARBY),
        (CONFIRMED_APPROACH, LEVEL_3_APPROACHING),
        (PATH_CONFLICT, LEVEL_4_PATH_CONFLICT),
        (IMMEDIATE_DANGER, LEVEL_5_HIGH_DANGER),
    ):
        assessment = make_assessment(
            level=threat_level, proximity_zone=NEAR, approach_state=APPROACHING,
            intersects_corridor=True,
        )
        result, _ = resolve_one(resolver, assessment)
        assert result.level == expected


# --- DE-ESCALATION: canonical timing controls audio-visible level ------------
# (Full escalation/de-escalation-timing coverage lives in tests/
# test_threat_assessment.py; these confirm the audio layer faithfully
# reflects whatever ThreatAssessmentEngine decides, end to end.)


def test_canonical_075s_deescalation_controls_audio_visible_level() -> None:
    engine = ThreatAssessmentEngine(
        path_conflict_confirmation_frames=1, deescalation_seconds=0.75,
        immediate_danger_deescalation_seconds=1.5, disappearance_grace_seconds=1.0,
    )
    resolver = make_resolver()
    obj_id = 1
    filtered = {obj_id: _filtered("MOVING")}
    approach_high = {obj_id: _approach(APPROACHING, NEAR)}
    approach_low = {obj_id: _approach(NOT_APPROACHING, FAR)}

    assessments = engine.assess_for_tracks(
        [_object(obj_id)], filtered, approach_high, {}, {obj_id: NEAR}, now=0.0,
    )
    results, _ = resolver.resolve_for_tracks(assessments, now=0.0)
    assert results[obj_id].level == LEVEL_3_APPROACHING

    # 0.5s later: still within the 0.75s canonical hold.
    assessments = engine.assess_for_tracks(
        [_object(obj_id)], filtered, approach_low, {}, {obj_id: FAR}, now=0.5,
    )
    results, _ = resolver.resolve_for_tracks(assessments, now=0.5)
    assert results[obj_id].level == LEVEL_3_APPROACHING

    # 0.8s later: hold expired -- audio-visible level drops too.
    assessments = engine.assess_for_tracks(
        [_object(obj_id)], filtered, approach_low, {}, {obj_id: FAR}, now=0.8,
    )
    results, _ = resolver.resolve_for_tracks(assessments, now=0.8)
    assert results[obj_id].level == LEVEL_1_MOVING_FAR


def test_canonical_15s_level5_deescalation_controls_audio_visible_level() -> None:
    engine = ThreatAssessmentEngine(
        path_conflict_confirmation_frames=1, deescalation_seconds=0.75,
        immediate_danger_deescalation_seconds=1.5, disappearance_grace_seconds=1.0,
    )
    resolver = make_resolver()
    obj_id = 1
    filtered = {obj_id: _filtered("MOVING")}
    approach_high = {obj_id: _approach(APPROACHING, NEAR)}
    approach_low = {obj_id: _approach(NOT_APPROACHING, FAR)}
    intersection_high = {obj_id: _intersection(intersects=True)}

    assessments = engine.assess_for_tracks(
        [_object(obj_id)], filtered, approach_high, intersection_high, {obj_id: NEAR}, now=0.0,
    )
    results, _ = resolver.resolve_for_tracks(assessments, now=0.0)
    assert results[obj_id].level == LEVEL_5_HIGH_DANGER

    # 1.0s later -- still within the 1.5s Level-5 hold.
    assessments = engine.assess_for_tracks(
        [_object(obj_id)], filtered, approach_low, {}, {obj_id: FAR}, now=1.0,
    )
    results, _ = resolver.resolve_for_tracks(assessments, now=1.0)
    assert results[obj_id].level == LEVEL_5_HIGH_DANGER

    # 1.6s after the peak -- hold expired.
    assessments = engine.assess_for_tracks(
        [_object(obj_id)], filtered, approach_low, {}, {obj_id: FAR}, now=1.6,
    )
    results, _ = resolver.resolve_for_tracks(assessments, now=1.6)
    assert results[obj_id].level == LEVEL_1_MOVING_FAR


def _object(track_id: int, region: str = "left", class_name: str = "car"):
    from src.models import BoundingBox, TrackedObject

    bbox = BoundingBox(x1=0, y1=0, x2=20, y2=20)
    return TrackedObject(
        track_id=track_id, class_id=2, class_name=class_name, confidence=0.9,
        bbox=bbox, center=(10, 10), region=region, position_history=((10, 10),) * 5,
        size_history=((20, 20),) * 5, direction="right", motion_status="moving",
        frames_since_seen=0,
    )


def _filtered(motion_state: str = "MOVING", uncertain: bool = False):
    from src.models import FilteredMotion

    return FilteredMotion(
        track_id=1, velocity_x=5.0, velocity_y=0.0, speed=5.0,
        motion_state=motion_state, source="COMPENSATED" if not uncertain else "RAW_FALLBACK",
        uncertain=uncertain, reason=None, confirmation_frames=0,
    )


def _approach(state: str = NOT_APPROACHING, proximity_zone: str = FAR):
    from src.models import ApproachResult

    return ApproachResult(
        track_id=1, state=state, confidence=1.0 if state != NOT_APPROACHING else 0.0,
        proximity_zone=proximity_zone, closing_score=0.0, corridor_distance_trend=0.0,
        scale_growth_trend=0.0, ground_point_trend=0.0, confirmation_frames=3,
        uncertain=(state == APPROACHING_UNCERTAIN), evidence_flags=(),
    )


def _intersection(intersects: bool = False, uncertain: bool = False):
    from src.models import PathIntersectionResult

    return PathIntersectionResult(
        valid=True, intersects=intersects, starts_inside=intersects, ends_inside=intersects,
        intersection_point=None, track_id=1, reason="test", uncertain=uncertain,
    )
