"""Threat Assessment Levels 1-5 for tracked objects.

Classifies each tracked object from perception data computed upstream:
FilteredMotion, ApproachResult, PathIntersectionResult, the proximity zone
and, optionally, ConflictImminence. No distance or time-to-conflict is
invented; ThreatAssessment.time_to_conflict_seconds is reported as
unavailable.

    LEVEL 1  NORMAL_MOVEMENT     -- confirmed movement, not nearby,
                                    approaching or in conflict
    LEVEL 2  NEARBY_PRESENCE     -- meaningfully close, no confirmed
                                    approach or conflict
    LEVEL 3  CONFIRMED_APPROACH  -- MID/NEAR proximity with multi-frame
                                    confirmed closing motion
    LEVEL 4  PATH_CONFLICT       -- predicted path intersects the pedestrian
                                    corridor on several consecutive frames
    LEVEL 5  IMMEDIATE_DANGER    -- NEAR + confirmed approach + persistent
                                    corridor conflict (never from a single
                                    uncertain cue)

This is classification only. It does not decide what is spoken (that is
AudioHazardResolver) and is independent of the operating mode.

classify_instantaneous_threat() is the pure single-frame decision tree
shared with AudioHazardResolver, which maps its result onto its own
LEVEL_*_ string constants.

ThreatAssessmentEngine adds per-track temporal state: multi-frame
persistence of path-conflict evidence, a level-dependent de-escalation hold
(0.75 s for Levels 1-4, 1.5 s for Level 5, independent of the resolver's
speech-repetition hold), and a short disappearance grace window so a
one-frame tracking gap does not reset the threat episode.

Conflict imminence is optional. When a confident IMMINENT reading, a
confirmed approach and a persisted corridor conflict coincide, Level 5 can
be reached before proximity is NEAR. Without it, behavior is unchanged.

The module has no OpenCV, subprocess or network dependencies and is unit
testable with synthetic inputs.
"""

from __future__ import annotations

import logging

from src.conflict_imminence import CONFLICT_IMMINENT, IMMINENCE_STATUS_RANK
from src.models import (
    ApproachResult,
    ConflictImminence,
    FilteredMotion,
    PathIntersectionResult,
    ThreatAssessment,
    TrackedObject,
)
from src.motion.approach_estimator import APPROACHING, APPROACHING_UNCERTAIN
from src.motion.relative_proximity_estimator import MID, NEAR, UNKNOWN

logger = logging.getLogger("atlas")

NORMAL_MOVEMENT = "NORMAL_MOVEMENT"
NEARBY_PRESENCE = "NEARBY_PRESENCE"
CONFIRMED_APPROACH = "CONFIRMED_APPROACH"
PATH_CONFLICT = "PATH_CONFLICT"
IMMEDIATE_DANGER = "IMMEDIATE_DANGER"

THREAT_LEVEL_RANK = {
    NORMAL_MOVEMENT: 1,
    NEARBY_PRESENCE: 2,
    CONFIRMED_APPROACH: 3,
    PATH_CONFLICT: 4,
    IMMEDIATE_DANGER: 5,
}

_CONFLICT_CANDIDATE = "CONFLICT_CANDIDATE"
_NO_CONFLICT_CANDIDATE = "NO_CONFLICT_CANDIDATE"


def classify_instantaneous_threat(
    filtered: FilteredMotion,
    approach: ApproachResult,
    intersection: PathIntersectionResult | None,
    proximity_zone: str,
    conflict_imminence: ConflictImminence | None = None,
) -> tuple[str, bool, tuple[str, ...]] | None:
    """The single-frame decision tree, shared by ThreatAssessmentEngine
    and AudioHazardResolver. Returns (level, uncertain, reason_codes), or
    None when there is no meaningful threat to report at all (motion not
    confirmed and not meaningfully near/approaching/in-conflict -- e.g. a
    stationary, far-away, irrelevant object never gets a level, matching
    AudioHazardResolver's pre-existing silent-object convention exactly).

    Callers that need multi-frame corridor-conflict persistence (Level 4/5
    require it -- see module docstring) must pass an `intersection` that
    has ALREADY been gated by that persistence requirement; this function
    itself is stateless and trusts whatever PathIntersectionResult it is
    given for this one frame.

    `conflict_imminence` is optional (default None, matching the pre-
    Conflict-Imminence-V1 signature exactly for any caller that doesn't
    pass it) -- see the LEVEL 5 pathway B below.
    """
    intersects_confident = (
        intersection is not None and intersection.intersects and not intersection.uncertain
    )
    intersects_uncertain = (
        intersection is not None and intersection.intersects and intersection.uncertain
    )
    approach_confirmed = approach.state == APPROACHING
    approach_uncertain_closing = approach.state == APPROACHING_UNCERTAIN
    motion_confirmed = filtered.motion_state == "MOVING"
    imminent_confident = (
        conflict_imminence is not None
        and conflict_imminence.status == CONFLICT_IMMINENT
        and not conflict_imminence.uncertain
    )

    # LEVEL 5, pathway A -- three confident conditions at once, never
    # from one uncertain cue, never from nearness or movement alone.
    if proximity_zone == NEAR and approach_confirmed and intersects_confident:
        return (
            IMMEDIATE_DANGER,
            False,
            ("NEAR", "APPROACHING_CONFIRMED", "CORRIDOR_INTERSECTS"),
        )

    # LEVEL 5, pathway B (Conflict Imminence V1) -- confirmed approach +
    # a confident IMMINENT predicted-conflict reading + persisted
    # corridor conflict, even when proximity has not yet reached NEAR.
    # Still exactly three confident conditions ANDed together, mirroring
    # pathway A's shape -- IMMINENT alone, or any two of these three,
    # is never sufficient.
    if approach_confirmed and intersects_confident and imminent_confident:
        return (
            IMMEDIATE_DANGER,
            False,
            ("APPROACHING_CONFIRMED", "CORRIDOR_INTERSECTS", "CONFLICT_IMMINENCE_IMMINENT"),
        )

    # LEVEL 4 -- resolved (and, from the caller, already persistence-
    # gated) corridor conflict, motion-state-agnostic: an object sitting
    # in the corridor is worth flagging regardless of its own MOVING/
    # STATIONARY classification.
    if intersects_confident:
        return (PATH_CONFLICT, False, ("CORRIDOR_INTERSECTS",))
    if intersects_uncertain:
        return (PATH_CONFLICT, True, ("CORRIDOR_INTERSECTS_UNCERTAIN",))

    # LEVEL 3 -- confirmed (or uncertain-but-closing) approach, MID/NEAR
    # proximity. The uncertain-approach branch is this level's documented
    # uncertainty cap -- an uncertain approach never reaches Level 4/5.
    if proximity_zone in (MID, NEAR):
        if approach_confirmed:
            return (CONFIRMED_APPROACH, False, (proximity_zone, "APPROACHING_CONFIRMED"))
        if approach_uncertain_closing:
            return (CONFIRMED_APPROACH, True, (proximity_zone, "APPROACHING_UNCERTAIN"))

    # LEVEL 2 / LEVEL 1 -- require confirmed MOVING; otherwise no
    # meaningful threat classification at all (matches the existing
    # STATIONARY/INSUFFICIENT_HISTORY -> silent convention).
    if not motion_confirmed:
        return None

    if proximity_zone == NEAR:
        return (NEARBY_PRESENCE, filtered.uncertain, (NEAR,))
    # FAR, MID-without-confirmed-approach, or UNKNOWN proximity all fold
    # into the neutral Level 1 -- never escalate on ambiguous data.
    return (NORMAL_MOVEMENT, filtered.uncertain, (proximity_zone,))


class _TrackThreatState:
    """Mutable per-track state carried across frames. Not a frozen model
    -- internal bookkeeping, never exposed outside this module."""

    __slots__ = (
        "last_seen",
        "conflict_confirmed",
        "conflict_candidate_state",
        "conflict_candidate_streak",
        "peak_level",
        "peak_uncertain",
        "peak_reason_codes",
        "peak_rank",
        "peak_time",
        "reported_rank",
        "persistence_frames",
    )

    def __init__(self, now: float) -> None:
        self.last_seen = now
        self.conflict_confirmed = False
        self.conflict_candidate_state: str | None = None
        self.conflict_candidate_streak = 0
        self.peak_level: str | None = None
        self.peak_uncertain = False
        self.peak_reason_codes: tuple[str, ...] = ()
        self.peak_rank = 0
        self.peak_time = now
        self.reported_rank = 0
        self.persistence_frames = 0


class ThreatAssessmentEngine:
    """Stateful per-track Level 1-5 threat classifier.

    Args:
        path_conflict_confirmation_frames: Consecutive frames a resolved
            corridor intersection (confident or uncertain) must be
            observed before Level 4/5 may be reported at all -- a single
            -frame edge contact is never sufficient. Must be >= 1.
        deescalation_seconds: After a higher level is reported for a
            track, a lower natural reading is withheld (the higher level
            keeps being reported) until this many continuous seconds of
            lower-risk evidence have passed -- applies when the currently
            -held level is anything other than IMMEDIATE_DANGER. Must be
            >= 0.
        immediate_danger_deescalation_seconds: Same, specifically for
            when the currently-held level is IMMEDIATE_DANGER --
            deliberately held longer than the other levels. Must be >= 0.
        disappearance_grace_seconds: How long a track's internal state is
            retained after it stops appearing in tracked_objects (e.g. a
            brief occlusion) before being dropped as stale. While within
            this grace window, no assessment is produced for the track
            (there is no fresh perception data to build one from), but
            its episode state (peak level, persistence, corridor-conflict
            streak) is preserved so a reappearance within the window
            continues the same episode rather than restarting cold. Must
            be >= 0.
    """

    def __init__(
        self,
        path_conflict_confirmation_frames: int,
        deescalation_seconds: float,
        immediate_danger_deescalation_seconds: float,
        disappearance_grace_seconds: float,
    ) -> None:
        if path_conflict_confirmation_frames < 1:
            raise ValueError(
                "path_conflict_confirmation_frames must be >= 1, got "
                f"{path_conflict_confirmation_frames}"
            )
        if deescalation_seconds < 0:
            raise ValueError(f"deescalation_seconds must be >= 0, got {deescalation_seconds}")
        if immediate_danger_deescalation_seconds < 0:
            raise ValueError(
                "immediate_danger_deescalation_seconds must be >= 0, got "
                f"{immediate_danger_deescalation_seconds}"
            )
        if disappearance_grace_seconds < 0:
            raise ValueError(
                f"disappearance_grace_seconds must be >= 0, got {disappearance_grace_seconds}"
            )

        self._path_conflict_confirmation_frames = path_conflict_confirmation_frames
        self._deescalation_seconds = deescalation_seconds
        self._immediate_danger_deescalation_seconds = immediate_danger_deescalation_seconds
        self._disappearance_grace_seconds = disappearance_grace_seconds
        self._states: dict[int, _TrackThreatState] = {}

    def assess_for_tracks(
        self,
        tracked_objects: list[TrackedObject],
        filtered_motions: dict[int, FilteredMotion],
        approach_results: dict[int, ApproachResult],
        resolved_intersection_results: dict[int, PathIntersectionResult],
        proximity_by_track: dict[int, str],
        now: float,
        frame_index: int | None = None,
        conflict_imminence_by_track: dict[int, ConflictImminence] | None = None,
    ) -> dict[int, ThreatAssessment]:
        """Assess every currently-tracked object in one call.

        Only tracks with resolved filtered-motion AND approach data
        produce an assessment (mirrors AudioHazardResolver.
        resolve_for_tracks()'s own "skip incomplete data" convention).
        Tracks absent from `tracked_objects` this frame are neither
        assessed nor immediately forgotten -- see
        disappearance_grace_seconds.

        `conflict_imminence_by_track` is optional (default None, treated
        as empty) -- omitting it (or a missing track_id within it)
        reproduces this method's exact pre-Conflict-Imminence-V1
        behavior for that track; see classify_instantaneous_threat's
        LEVEL 5 pathway B.
        """
        active_ids = {obj.track_id for obj in tracked_objects}
        for stale_id in [tid for tid in self._states if tid not in active_ids]:
            state = self._states[stale_id]
            if (now - state.last_seen) >= self._disappearance_grace_seconds:
                del self._states[stale_id]

        conflict_imminence_by_track = conflict_imminence_by_track or {}
        results: dict[int, ThreatAssessment] = {}
        for obj in tracked_objects:
            filtered = filtered_motions.get(obj.track_id)
            approach = approach_results.get(obj.track_id)
            if filtered is None or approach is None:
                continue
            intersection = resolved_intersection_results.get(obj.track_id)
            proximity_zone = proximity_by_track.get(obj.track_id, UNKNOWN)
            conflict_imminence = conflict_imminence_by_track.get(obj.track_id)

            assessment = self._assess_track(
                obj, filtered, approach, intersection, proximity_zone, now, frame_index,
                conflict_imminence,
            )
            if assessment is not None:
                results[obj.track_id] = assessment
                logger.debug("Threat assessment: %s", assessment)

        return results

    def _assess_track(
        self,
        obj: TrackedObject,
        filtered: FilteredMotion,
        approach: ApproachResult,
        intersection: PathIntersectionResult | None,
        proximity_zone: str,
        now: float,
        frame_index: int | None,
        conflict_imminence: ConflictImminence | None = None,
    ) -> ThreatAssessment | None:
        state = self._states.setdefault(obj.track_id, _TrackThreatState(now))
        state.last_seen = now

        raw_conflict_candidate = intersection is not None and intersection.intersects
        self._update_conflict_confirmation(state, raw_conflict_candidate)

        # Levels 4/5 may only be reached once corridor-conflict evidence
        # has persisted for path_conflict_confirmation_frames -- until
        # then, suppress the intersection evidence entirely so
        # classify_instantaneous_threat falls through to whatever
        # Level 1-3 the remaining (approach/proximity/motion) evidence
        # supports, never a fabricated "not in conflict" claim.
        effective_intersection = intersection if state.conflict_confirmed else None

        natural = classify_instantaneous_threat(
            filtered, approach, effective_intersection, proximity_zone, conflict_imminence
        )
        if natural is None:
            level, uncertain, reason_codes, rank = None, False, (), 0
        else:
            level, uncertain, reason_codes = natural
            rank = THREAT_LEVEL_RANK[level]
            if level == PATH_CONFLICT or level == IMMEDIATE_DANGER:
                reason_codes = reason_codes + ("PATH_CONFLICT_PERSISTENT",)

        if state.peak_level is None:
            admit = True
        else:
            hold_seconds = (
                self._immediate_danger_deescalation_seconds
                if state.peak_rank == THREAT_LEVEL_RANK[IMMEDIATE_DANGER]
                else self._deescalation_seconds
            )
            held = rank < state.peak_rank and (now - state.peak_time) < hold_seconds
            admit = not held

        if admit:
            state.peak_level, state.peak_uncertain, state.peak_reason_codes = level, uncertain, reason_codes
            state.peak_rank, state.peak_time = rank, now

        reported_level = state.peak_level
        if reported_level is None:
            state.reported_rank = 0
            state.persistence_frames = 0
            return None

        reported_rank = state.peak_rank
        escalated = reported_rank > state.reported_rank
        state.persistence_frames = 1 if escalated or state.persistence_frames == 0 else state.persistence_frames + 1
        state.reported_rank = reported_rank

        deescalation_pending = rank < reported_rank

        return ThreatAssessment(
            track_id=obj.track_id,
            object_class=obj.class_name,
            region=obj.region,
            level=reported_level,
            confidence=approach.confidence,
            uncertain=state.peak_uncertain,
            reason_codes=state.peak_reason_codes,
            proximity_zone=proximity_zone,
            approach_state=approach.state,
            approach_evidence_flags=approach.evidence_flags,
            intersects_corridor=reported_level in (PATH_CONFLICT, IMMEDIATE_DANGER),
            corridor_intersection_uncertain=intersection.uncertain if intersection is not None else False,
            time_to_conflict_seconds=None,
            persistence_frames=state.persistence_frames,
            escalated=escalated,
            deescalation_pending=deescalation_pending,
            frame_index=frame_index,
            timestamp=now,
            conflict_imminence_status=(
                conflict_imminence.status if conflict_imminence is not None else None
            ),
            first_conflict_step=(
                conflict_imminence.first_conflict_step if conflict_imminence is not None else None
            ),
        )

    def _update_conflict_confirmation(self, state: _TrackThreatState, candidate_present: bool) -> None:
        """Confirmation-streak state machine for corridor-conflict
        persistence -- mirrors MotionStateFilter/ApproachEstimator's own
        _update_confirmed_state pattern exactly, for consistency."""
        candidate = _CONFLICT_CANDIDATE if candidate_present else _NO_CONFLICT_CANDIDATE
        currently = _CONFLICT_CANDIDATE if state.conflict_confirmed else _NO_CONFLICT_CANDIDATE
        if candidate == currently:
            state.conflict_candidate_state = None
            state.conflict_candidate_streak = 0
            return

        if state.conflict_candidate_state == candidate:
            state.conflict_candidate_streak += 1
        else:
            state.conflict_candidate_state = candidate
            state.conflict_candidate_streak = 1

        if state.conflict_candidate_streak >= self._path_conflict_confirmation_frames:
            state.conflict_confirmed = candidate == _CONFLICT_CANDIDATE
            state.conflict_candidate_state = None
            state.conflict_candidate_streak = 0


def rank_active_threats(assessments: list[ThreatAssessment]) -> list[ThreatAssessment]:
    """Order active threats for downstream audio policy to prioritize.

    Sort key, in order:
        1. Highest threat level first.
        2. Most imminent conflict_imminence_status first (IMMINENT > SOON
           > DISTANT > NONE/UNKNOWN -- see
           src.conflict_imminence.IMMINENCE_STATUS_RANK). A missing/
           UNKNOWN reading ranks with NONE, never ahead of a confidently
           SOON/IMMINENT hazard merely because its own value is missing.
        3. Shortest time_to_conflict_seconds first when available (never
           populated today -- see ThreatAssessment.time_to_conflict_seconds
           -- so this is currently a no-op, kept for forward compatibility
           rather than removed and re-added later).
        4. region == "center" (directly ahead) before "left"/"right".
        5. NEAR proximity_zone before MID/FAR/UNKNOWN, as the closest
           available proxy for "nearer to the corridor" (Atlas has no
           per-side corridor-distance value on ThreatAssessment itself).
        6. Higher persistence_frames first, as a final, deterministic
           tie-breaker.

    Python's sort is stable, so any remaining tie preserves the input
    list's relative order -- this function never reorders equally-ranked
    entries arbitrarily.
    """
    proximity_rank = {NEAR: 0, MID: 1, "FAR": 2}

    def sort_key(assessment: ThreatAssessment) -> tuple:
        rank = THREAT_LEVEL_RANK.get(assessment.level, 0)
        imminence_rank = IMMINENCE_STATUS_RANK.get(assessment.conflict_imminence_status, 0)
        ttc = (
            assessment.time_to_conflict_seconds
            if assessment.time_to_conflict_seconds is not None
            else float("inf")
        )
        center_first = 0 if assessment.region == "center" else 1
        zone_rank = proximity_rank.get(assessment.proximity_zone, 3)
        return (-rank, -imminence_rank, ttc, center_first, zone_rank, -assessment.persistence_frames)

    return sorted(assessments, key=sort_key)
