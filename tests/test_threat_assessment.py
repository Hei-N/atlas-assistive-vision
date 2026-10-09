"""Unit tests for src/threat_assessment.py -- the canonical Level 1-5
threat classification (classify_instantaneous_threat, ThreatAssessmentEngine,
rank_active_threats). Pure logic, no camera/model/network/subprocess
involved -- synthetic TrackedObject/FilteredMotion/ApproachResult/
PathIntersectionResult data throughout, matching tests/
test_audio_hazard_resolver.py's own conventions.
"""

from __future__ import annotations

import pytest

from src.conflict_imminence import CONFLICT_DISTANT, CONFLICT_IMMINENT, CONFLICT_SOON
from src.models import BoundingBox, ApproachResult, ConflictImminence, FilteredMotion, PathIntersectionResult, TrackedObject
from src.motion.approach_estimator import APPROACHING, APPROACHING_UNCERTAIN, NOT_APPROACHING
from src.motion.relative_proximity_estimator import FAR, MID, NEAR
from src.threat_assessment import (
    CONFIRMED_APPROACH,
    IMMEDIATE_DANGER,
    NEARBY_PRESENCE,
    NORMAL_MOVEMENT,
    PATH_CONFLICT,
    THREAT_LEVEL_RANK,
    ThreatAssessmentEngine,
    classify_instantaneous_threat,
    rank_active_threats,
)
from src.models import ThreatAssessment


def make_conflict_imminence(
    track_id: int = 1, status: str = CONFLICT_IMMINENT, uncertain: bool = False,
    first_conflict_step: float | None = 1.0,
) -> ConflictImminence:
    return ConflictImminence(
        track_id=track_id, status=status, first_conflict_step=first_conflict_step,
        estimated_seconds_to_conflict=None, confidence=0.0 if uncertain else 1.0,
        uncertain=uncertain, reason_codes=(), frame_index=None, timestamp=0.0,
    )


def make_object(track_id: int = 1, class_name: str = "car", region: str = "left") -> TrackedObject:
    bbox = BoundingBox(x1=0, y1=0, x2=20, y2=20)
    return TrackedObject(
        track_id=track_id, class_id=2, class_name=class_name, confidence=0.9,
        bbox=bbox, center=(10, 10), region=region, position_history=((10, 10),) * 5,
        size_history=((20, 20),) * 5, direction="right", motion_status="moving",
        frames_since_seen=0,
    )


def make_filtered(motion_state: str = "MOVING", uncertain: bool = False) -> FilteredMotion:
    return FilteredMotion(
        track_id=1, velocity_x=5.0, velocity_y=0.0, speed=5.0,
        motion_state=motion_state, source="COMPENSATED" if not uncertain else "RAW_FALLBACK",
        uncertain=uncertain, reason=None, confirmation_frames=0,
    )


def make_approach(state: str = NOT_APPROACHING, proximity_zone: str = FAR) -> ApproachResult:
    return ApproachResult(
        track_id=1, state=state, confidence=1.0 if state != NOT_APPROACHING else 0.0,
        proximity_zone=proximity_zone, closing_score=0.0, corridor_distance_trend=0.0,
        scale_growth_trend=0.0, ground_point_trend=0.0, confirmation_frames=3,
        uncertain=(state == APPROACHING_UNCERTAIN), evidence_flags=(),
    )


def make_intersection(intersects: bool = False, uncertain: bool = False) -> PathIntersectionResult:
    return PathIntersectionResult(
        valid=True, intersects=intersects, starts_inside=intersects, ends_inside=intersects,
        intersection_point=None, track_id=1, reason="test", uncertain=uncertain,
    )


def make_engine(
    path_conflict_confirmation_frames: int = 3,
    deescalation_seconds: float = 0.75,
    immediate_danger_deescalation_seconds: float = 1.5,
    disappearance_grace_seconds: float = 1.0,
) -> ThreatAssessmentEngine:
    return ThreatAssessmentEngine(
        path_conflict_confirmation_frames=path_conflict_confirmation_frames,
        deescalation_seconds=deescalation_seconds,
        immediate_danger_deescalation_seconds=immediate_danger_deescalation_seconds,
        disappearance_grace_seconds=disappearance_grace_seconds,
    )


def assess_once(
    engine: ThreatAssessmentEngine, obj, filtered, approach, intersection, proximity_zone, now,
    frame_index=None, conflict_imminence=None,
):
    result = engine.assess_for_tracks(
        [obj], {obj.track_id: filtered}, {obj.track_id: approach},
        {obj.track_id: intersection} if intersection is not None else {},
        {obj.track_id: proximity_zone}, now, frame_index,
        {obj.track_id: conflict_imminence} if conflict_imminence is not None else None,
    )
    return result.get(obj.track_id)


# --- classify_instantaneous_threat (pure decision tree) -----------------------


class TestLevel1NormalMovement:
    def test_compensated_confirmed_movement_far(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(NOT_APPROACHING, FAR), None, FAR
        )
        assert result[0] == NORMAL_MOVEMENT
        assert result[1] is False

    def test_no_corridor_conflict_mid_without_approach(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(NOT_APPROACHING, MID), None, MID
        )
        assert result[0] == NORMAL_MOVEMENT

    def test_stationary_bbox_jitter_does_not_qualify(self):
        result = classify_instantaneous_threat(
            make_filtered("STATIONARY"), make_approach(NOT_APPROACHING, FAR), None, FAR
        )
        assert result is None


class TestLevel2NearbyPresence:
    def test_nearby_stationary_car(self):
        # NEAR proximity alone (motion irrelevant to NEAR classification,
        # but Level 2 still requires confirmed MOVING per the existing
        # silent-object convention -- a genuinely STATIONARY near object
        # produces no assessment, matching AudioHazardResolver).
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(NOT_APPROACHING, NEAR), None, NEAR
        )
        assert result[0] == NEARBY_PRESENCE

    def test_nearby_stationary_bicycle_no_assessment(self):
        result = classify_instantaneous_threat(
            make_filtered("STATIONARY"), make_approach(NOT_APPROACHING, NEAR), None, NEAR
        )
        assert result is None

    def test_nearby_person_moving(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(NOT_APPROACHING, NEAR), None, NEAR
        )
        assert result[0] == NEARBY_PRESENCE

    def test_nearby_object_does_not_become_approaching(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(NOT_APPROACHING, NEAR), None, NEAR
        )
        assert result[0] != CONFIRMED_APPROACH


class TestLevel3ConfirmedApproach:
    def test_confirmed_multiframe_closing_motion(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(APPROACHING, MID), None, MID
        )
        assert result[0] == CONFIRMED_APPROACH
        assert result[1] is False

    def test_one_frame_expansion_rejected(self):
        # ApproachEstimator itself never confirms off one frame -- from
        # this layer's perspective that means approach.state is simply
        # NOT_APPROACHING; the classifier must not promote it anyway.
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(NOT_APPROACHING, MID), None, MID
        )
        assert result[0] != CONFIRMED_APPROACH

    def test_uncertain_raw_fallback_does_not_falsely_confirm(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING", uncertain=True),
            make_approach(APPROACHING_UNCERTAIN, NEAR), None, NEAR,
        )
        assert result[0] == CONFIRMED_APPROACH
        assert result[1] is True  # uncertain -- never a confident claim


class TestLevel4PathConflict:
    def test_confident_intersection(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(NOT_APPROACHING, FAR),
            make_intersection(intersects=True, uncertain=False), FAR,
        )
        assert result[0] == PATH_CONFLICT
        assert result[1] is False

    def test_uncertain_intersection_handled_conservatively(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(NOT_APPROACHING, FAR),
            make_intersection(intersects=True, uncertain=True), FAR,
        )
        assert result[0] == PATH_CONFLICT
        assert result[1] is True

    def test_no_intersection_never_triggers(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(APPROACHING, NEAR),
            make_intersection(intersects=False), NEAR,
        )
        assert result[0] != PATH_CONFLICT


class TestLevel5ImmediateDanger:
    def test_close_approach_confident_conflict(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(APPROACHING, NEAR),
            make_intersection(intersects=True, uncertain=False), NEAR,
        )
        assert result[0] == IMMEDIATE_DANGER
        assert result[1] is False

    def test_exactly_two_strong_signals_not_enough(self):
        # NEAR + approaching, but no corridor conflict -- only 2 signals.
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(APPROACHING, NEAR),
            make_intersection(intersects=False), NEAR,
        )
        assert result[0] != IMMEDIATE_DANGER

    def test_close_stationary_alone_not_enough(self):
        result = classify_instantaneous_threat(
            make_filtered("STATIONARY"), make_approach(NOT_APPROACHING, NEAR),
            None, NEAR,
        )
        assert result is None

    def test_fast_but_far_not_enough(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(APPROACHING, FAR),
            make_intersection(intersects=False), FAR,
        )
        assert result[0] not in (IMMEDIATE_DANGER, CONFIRMED_APPROACH)
        assert result[0] == NORMAL_MOVEMENT

    def test_approach_outside_corridor_not_enough(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(APPROACHING, NEAR),
            make_intersection(intersects=False), NEAR,
        )
        assert result[0] == CONFIRMED_APPROACH
        assert result[0] != IMMEDIATE_DANGER

    def test_uncertain_motion_alone_not_enough(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING", uncertain=True), make_approach(NOT_APPROACHING, FAR),
            None, FAR,
        )
        assert result[0] != IMMEDIATE_DANGER
        assert result[1] is True


class TestLevel5ImminentPathwayB:
    """Conflict Imminence V1's second Level 5 pathway: confirmed approach
    + confident IMMINENT + persisted corridor conflict, even without NEAR
    proximity."""

    def test_imminent_pathway_qualifies_without_near_proximity(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(APPROACHING, MID),
            make_intersection(intersects=True, uncertain=False), MID,
            make_conflict_imminence(status=CONFLICT_IMMINENT, uncertain=False),
        )
        assert result[0] == IMMEDIATE_DANGER
        assert result[1] is False

    def test_imminent_alone_not_enough(self):
        # IMMINENT conflict imminence, but no approach and no corridor
        # conflict -- only 1 signal.
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(NOT_APPROACHING, FAR),
            make_intersection(intersects=False), FAR,
            make_conflict_imminence(status=CONFLICT_IMMINENT, uncertain=False),
        )
        assert result[0] != IMMEDIATE_DANGER

    def test_exactly_two_of_three_pathway_b_signals_not_enough(self):
        # Approach confirmed + IMMINENT, but NO persisted corridor
        # conflict -- only 2 of the 3 pathway-B signals.
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(APPROACHING, MID),
            make_intersection(intersects=False), MID,
            make_conflict_imminence(status=CONFLICT_IMMINENT, uncertain=False),
        )
        assert result[0] != IMMEDIATE_DANGER

    def test_uncertain_imminent_does_not_create_level_5(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(APPROACHING, MID),
            make_intersection(intersects=True, uncertain=False), MID,
            make_conflict_imminence(status=CONFLICT_IMMINENT, uncertain=True),
        )
        assert result[0] != IMMEDIATE_DANGER
        assert result[0] == PATH_CONFLICT

    def test_soon_status_does_not_qualify(self):
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(APPROACHING, MID),
            make_intersection(intersects=True, uncertain=False), MID,
            make_conflict_imminence(status=CONFLICT_SOON, uncertain=False),
        )
        assert result[0] != IMMEDIATE_DANGER

    def test_no_conflict_imminence_preserves_old_behavior(self):
        # conflict_imminence defaults to None -- byte-for-byte identical
        # to pre-Conflict-Imminence-V1 behavior for this input.
        with_none = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(APPROACHING, MID),
            make_intersection(intersects=True, uncertain=False), MID,
        )
        explicit_none = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(APPROACHING, MID),
            make_intersection(intersects=True, uncertain=False), MID, None,
        )
        assert with_none == explicit_none
        assert with_none[0] == PATH_CONFLICT

    def test_lateral_path_conflict_remains_level_4_without_confirmed_approach(self):
        # Corridor conflict + IMMINENT, but approach is NOT confirmed
        # (e.g. a sideways-crossing object) -- stays Level 4, never
        # promoted to Level 5 by conflict imminence alone.
        result = classify_instantaneous_threat(
            make_filtered("MOVING"), make_approach(NOT_APPROACHING, MID),
            make_intersection(intersects=True, uncertain=False), MID,
            make_conflict_imminence(status=CONFLICT_IMMINENT, uncertain=False),
        )
        assert result[0] == PATH_CONFLICT


# --- ThreatAssessmentEngine: path-conflict persistence (Level 4/5 gate) -------


class TestPathConflictPersistence:
    def test_one_frame_edge_contact_rejected(self):
        engine = make_engine(path_conflict_confirmation_frames=3)
        obj = make_object()
        result = assess_once(
            engine, obj, make_filtered("MOVING"), make_approach(NOT_APPROACHING, FAR),
            make_intersection(intersects=True), FAR, now=0.0,
        )
        assert result.level != PATH_CONFLICT

    def test_persistent_intersection_confirms_after_n_frames(self):
        engine = make_engine(path_conflict_confirmation_frames=3)
        obj = make_object()
        result = None
        for i in range(3):
            result = assess_once(
                engine, obj, make_filtered("MOVING"), make_approach(NOT_APPROACHING, FAR),
                make_intersection(intersects=True), FAR, now=float(i),
            )
        assert result.level == PATH_CONFLICT
        assert "PATH_CONFLICT_PERSISTENT" in result.reason_codes

    def test_distant_non_intersecting_path_never_qualifies(self):
        engine = make_engine(path_conflict_confirmation_frames=1)
        obj = make_object()
        result = assess_once(
            engine, obj, make_filtered("MOVING"), make_approach(APPROACHING, NEAR),
            make_intersection(intersects=False), NEAR, now=0.0,
        )
        assert result.level != PATH_CONFLICT


# --- ThreatAssessmentEngine: conflict-imminence pathway B, end-to-end --------


class TestEngineConflictImminencePathway:
    def test_imminent_pathway_reaches_level_5_through_engine(self):
        engine = make_engine(path_conflict_confirmation_frames=1)
        obj = make_object()
        result = assess_once(
            engine, obj, make_filtered("MOVING"), make_approach(APPROACHING, MID),
            make_intersection(intersects=True, uncertain=False), MID, now=0.0,
            conflict_imminence=make_conflict_imminence(status=CONFLICT_IMMINENT, uncertain=False),
        )
        assert result.level == IMMEDIATE_DANGER
        assert result.conflict_imminence_status == CONFLICT_IMMINENT

    def test_no_conflict_imminence_passed_behaves_identically(self):
        engine = make_engine(path_conflict_confirmation_frames=1)
        obj = make_object()
        result = assess_once(
            engine, obj, make_filtered("MOVING"), make_approach(APPROACHING, MID),
            make_intersection(intersects=True, uncertain=False), MID, now=0.0,
        )
        assert result.level == PATH_CONFLICT
        assert result.conflict_imminence_status is None
        assert result.first_conflict_step is None


# --- TEMPORAL: escalation / de-escalation / persistence -----------------------


class TestTemporal:
    def test_direct_escalation_skips_intermediate_levels(self):
        engine = make_engine(path_conflict_confirmation_frames=3)
        obj = make_object()
        # Build corridor-conflict persistence while proximity is FAR and
        # not approaching -- stays at Level 1 the whole time.
        for i in range(2):
            result = assess_once(
                engine, obj, make_filtered("MOVING"), make_approach(NOT_APPROACHING, FAR),
                make_intersection(intersects=True), FAR, now=float(i),
            )
            assert result.level == NORMAL_MOVEMENT
        # Third frame: confirmation streak completes AND strong evidence
        # (NEAR + approaching) appears simultaneously -- should jump
        # straight to IMMEDIATE_DANGER, never stopping at 2/3/4 first.
        result = assess_once(
            engine, obj, make_filtered("MOVING"), make_approach(APPROACHING, NEAR),
            make_intersection(intersects=True, uncertain=False), NEAR, now=2.0,
        )
        assert result.level == IMMEDIATE_DANGER
        assert result.escalated is True

    def test_level_5_deescalation_is_slower(self):
        engine = make_engine(
            path_conflict_confirmation_frames=1, deescalation_seconds=0.75,
            immediate_danger_deescalation_seconds=1.5,
        )
        obj = make_object()
        result = assess_once(
            engine, obj, make_filtered("MOVING"), make_approach(APPROACHING, NEAR),
            make_intersection(intersects=True, uncertain=False), NEAR, now=0.0,
        )
        assert result.level == IMMEDIATE_DANGER

        # 1.0s later, evidence has dropped to plain movement -- still
        # within the 1.5s Level-5 hold, so Level 5 keeps being reported.
        result = assess_once(
            engine, obj, make_filtered("MOVING"), make_approach(NOT_APPROACHING, FAR),
            make_intersection(intersects=False), FAR, now=1.0,
        )
        assert result.level == IMMEDIATE_DANGER
        assert result.deescalation_pending is True

        # 1.6s after the peak -- hold has expired, drops to the natural level.
        result = assess_once(
            engine, obj, make_filtered("MOVING"), make_approach(NOT_APPROACHING, FAR),
            make_intersection(intersects=False), FAR, now=1.6,
        )
        assert result.level == NORMAL_MOVEMENT

    def test_levels_1_to_4_deescalate_in_075_seconds(self):
        engine = make_engine(
            path_conflict_confirmation_frames=1, deescalation_seconds=0.75,
            immediate_danger_deescalation_seconds=1.5,
        )
        obj = make_object()
        result = assess_once(
            engine, obj, make_filtered("MOVING"), make_approach(APPROACHING, MID),
            None, MID, now=0.0,
        )
        assert result.level == CONFIRMED_APPROACH

        # 0.5s later -- still within the 0.75s hold.
        result = assess_once(
            engine, obj, make_filtered("MOVING"), make_approach(NOT_APPROACHING, FAR),
            None, FAR, now=0.5,
        )
        assert result.level == CONFIRMED_APPROACH
        assert result.deescalation_pending is True

        # 0.8s later -- hold expired.
        result = assess_once(
            engine, obj, make_filtered("MOVING"), make_approach(NOT_APPROACHING, FAR),
            None, FAR, now=0.8,
        )
        assert result.level == NORMAL_MOVEMENT
        assert result.deescalation_pending is False

    def test_disappearance_grace_preserves_state_on_reappearance(self):
        engine = make_engine(
            path_conflict_confirmation_frames=1, immediate_danger_deescalation_seconds=1.5,
            disappearance_grace_seconds=1.0,
        )
        obj = make_object()
        result = assess_once(
            engine, obj, make_filtered("MOVING"), make_approach(APPROACHING, NEAR),
            make_intersection(intersects=True, uncertain=False), NEAR, now=0.0,
        )
        assert result.level == IMMEDIATE_DANGER

        # Track vanishes for 0.5s (within the 1.0s grace) -- call with an
        # EMPTY tracked_objects list, as main.py would when the tracker
        # doesn't emit this track_id for a frame.
        empty_result = engine.assess_for_tracks([], {}, {}, {}, {}, now=0.5)
        assert empty_result == {}

        # Reappears at 1.0s (still within the Level-5 1.5s hold measured
        # from the original peak) with weak evidence -- Level 5 should
        # still be held, proving the episode survived the gap.
        result = assess_once(
            engine, obj, make_filtered("MOVING"), make_approach(NOT_APPROACHING, FAR),
            make_intersection(intersects=False), FAR, now=1.0,
        )
        assert result.level == IMMEDIATE_DANGER
        assert result.deescalation_pending is True

    def test_stale_track_removed_after_grace_expires(self):
        engine = make_engine(
            path_conflict_confirmation_frames=1, immediate_danger_deescalation_seconds=1.5,
            disappearance_grace_seconds=1.0,
        )
        obj = make_object()
        result = assess_once(
            engine, obj, make_filtered("MOVING"), make_approach(APPROACHING, NEAR),
            make_intersection(intersects=True, uncertain=False), NEAR, now=0.0,
        )
        assert result.level == IMMEDIATE_DANGER

        # Absent for longer than disappearance_grace_seconds -- state is
        # pruned entirely.
        engine.assess_for_tracks([], {}, {}, {}, {}, now=2.0)

        # Reappears with weak evidence -- since state was actually
        # dropped, this is a fresh episode: reports the natural (low)
        # level immediately, not a held Level 5, and persistence resets.
        result = assess_once(
            engine, obj, make_filtered("MOVING"), make_approach(NOT_APPROACHING, FAR),
            make_intersection(intersects=False), FAR, now=2.1,
        )
        assert result.level == NORMAL_MOVEMENT
        assert result.persistence_frames == 1


# --- RANKING --------------------------------------------------------------------


def make_assessment(
    track_id=1, level=NORMAL_MOVEMENT, region="left", proximity_zone=FAR,
    time_to_conflict_seconds=None, persistence_frames=1, conflict_imminence_status=None,
) -> ThreatAssessment:
    return ThreatAssessment(
        track_id=track_id, object_class="car", region=region, level=level,
        confidence=0.5, uncertain=False, reason_codes=(), proximity_zone=proximity_zone,
        approach_state=NOT_APPROACHING, approach_evidence_flags=(), intersects_corridor=False,
        corridor_intersection_uncertain=False, time_to_conflict_seconds=time_to_conflict_seconds,
        persistence_frames=persistence_frames, escalated=False, deescalation_pending=False,
        frame_index=None, timestamp=0.0, conflict_imminence_status=conflict_imminence_status,
    )


class TestRanking:
    def test_level_5_beats_level_4(self):
        low = make_assessment(track_id=1, level=PATH_CONFLICT)
        high = make_assessment(track_id=2, level=IMMEDIATE_DANGER)
        ranked = rank_active_threats([low, high])
        assert ranked[0].track_id == 2

    def test_shorter_ttc_wins_when_available(self):
        far_ttc = make_assessment(track_id=1, level=IMMEDIATE_DANGER, time_to_conflict_seconds=5.0)
        near_ttc = make_assessment(track_id=2, level=IMMEDIATE_DANGER, time_to_conflict_seconds=1.0)
        ranked = rank_active_threats([far_ttc, near_ttc])
        assert ranked[0].track_id == 2

    def test_directly_ahead_tiebreak(self):
        left = make_assessment(track_id=1, level=CONFIRMED_APPROACH, region="left")
        center = make_assessment(track_id=2, level=CONFIRMED_APPROACH, region="center")
        ranked = rank_active_threats([left, center])
        assert ranked[0].track_id == 2

    def test_ranking_is_stable_and_deterministic(self):
        a = make_assessment(track_id=1, level=NORMAL_MOVEMENT, region="left")
        b = make_assessment(track_id=2, level=NORMAL_MOVEMENT, region="left")
        ranked_once = [t.track_id for t in rank_active_threats([a, b])]
        ranked_again = [t.track_id for t in rank_active_threats([a, b])]
        assert ranked_once == ranked_again == [1, 2]

    def test_imminent_beats_soon_at_same_level(self):
        soon = make_assessment(track_id=1, level=PATH_CONFLICT, conflict_imminence_status=CONFLICT_SOON)
        imminent = make_assessment(track_id=2, level=PATH_CONFLICT, conflict_imminence_status=CONFLICT_IMMINENT)
        ranked = rank_active_threats([soon, imminent])
        assert ranked[0].track_id == 2

    def test_soon_beats_distant_at_same_level(self):
        distant = make_assessment(track_id=1, level=PATH_CONFLICT, conflict_imminence_status=CONFLICT_DISTANT)
        soon = make_assessment(track_id=2, level=PATH_CONFLICT, conflict_imminence_status=CONFLICT_SOON)
        ranked = rank_active_threats([distant, soon])
        assert ranked[0].track_id == 2

    def test_unknown_does_not_outrank_confident_soon(self):
        unknown = make_assessment(track_id=1, level=PATH_CONFLICT, conflict_imminence_status=None)
        soon = make_assessment(track_id=2, level=PATH_CONFLICT, conflict_imminence_status=CONFLICT_SOON)
        ranked = rank_active_threats([unknown, soon])
        assert ranked[0].track_id == 2

    def test_threat_level_still_dominates_imminence(self):
        # A lower threat level with IMMINENT conflict imminence must
        # never outrank a higher threat level with no imminence data.
        low_level_imminent = make_assessment(
            track_id=1, level=CONFIRMED_APPROACH, conflict_imminence_status=CONFLICT_IMMINENT,
        )
        high_level = make_assessment(track_id=2, level=IMMEDIATE_DANGER, conflict_imminence_status=None)
        ranked = rank_active_threats([low_level_imminent, high_level])
        assert ranked[0].track_id == 2


# --- MODE INDEPENDENCE ----------------------------------------------------------


class TestModeIndependence:
    def test_identical_input_produces_identical_assessment(self):
        # ThreatAssessmentEngine takes no operating-mode parameter at all
        # -- two independently constructed engines fed identical
        # perception input must produce identical results, proving there
        # is no hidden mode-dependent behavior to vary in the first place.
        engine_a = make_engine()
        engine_b = make_engine()
        obj = make_object()
        filtered = make_filtered("MOVING")
        approach = make_approach(APPROACHING, NEAR)
        intersection = make_intersection(intersects=True, uncertain=False)

        result_a = assess_once(engine_a, obj, filtered, approach, intersection, NEAR, now=0.0)
        result_b = assess_once(engine_b, obj, filtered, approach, intersection, NEAR, now=0.0)
        assert result_a.level == result_b.level
        assert result_a.uncertain == result_b.uncertain
        assert result_a.reason_codes == result_b.reason_codes


# --- constructor validation -----------------------------------------------------


class TestConstructorValidation:
    def test_rejects_zero_path_conflict_confirmation_frames(self):
        with pytest.raises(ValueError):
            make_engine(path_conflict_confirmation_frames=0)

    def test_rejects_negative_deescalation_seconds(self):
        with pytest.raises(ValueError):
            make_engine(deescalation_seconds=-1.0)

    def test_rejects_negative_immediate_danger_deescalation_seconds(self):
        with pytest.raises(ValueError):
            make_engine(immediate_danger_deescalation_seconds=-1.0)

    def test_rejects_negative_disappearance_grace_seconds(self):
        with pytest.raises(ValueError):
            make_engine(disappearance_grace_seconds=-1.0)


def test_threat_level_rank_ordering():
    assert (
        THREAT_LEVEL_RANK[NORMAL_MOVEMENT]
        < THREAT_LEVEL_RANK[NEARBY_PRESENCE]
        < THREAT_LEVEL_RANK[CONFIRMED_APPROACH]
        < THREAT_LEVEL_RANK[PATH_CONFLICT]
        < THREAT_LEVEL_RANK[IMMEDIATE_DANGER]
    )
