"""Conservative image-space conflict-imminence estimation for Atlas
(Conflict Imminence V1).

Answers: "if this object continues on its current resolved trajectory,
how soon does its predicted path affect the pedestrian corridor, in
image-space terms?" This is NOT physical time-to-collision -- see
ConflictImminence's docstring (src/models.py) for exactly why
estimated_seconds_to_conflict stays None: the trajectory prediction
horizon (TrajectoryPrediction.prediction_horizon_frames) is defined in
frame-count units, and Atlas has no fixed, reliable frame-to-real-
seconds conversion (measured per-frame processing time fluctuates with
system load -- there is no assumed/target FPS anywhere in this
codebase). All timing here is expressed in predicted trajectory FRAME
-STEPS instead.

Reuses PathIntersectionAnalyzer's already-computed corridor-intersection
geometry (intersection_point, starts_inside, ends_inside) rather than
implementing a second intersection algorithm: the earliest predicted
conflict step is derived from where the intersection point falls along
the existing current_center -> predicted_center segment, expressed as a
fraction of the segment multiplied by prediction_horizon_frames. No
change to PathIntersectionAnalyzer or PathIntersectionResult is needed
or made.

Deliberately does NOT depend on ApproachResult: a sideways-crossing
object (e.g. a bicycle) can have meaningful conflict imminence without
ever satisfying ApproachEstimator's "closing on the user" cues, so this
signal is evaluated purely from FilteredMotion + TrajectoryPrediction +
PathIntersectionResult, independent of confirmed-approach evidence. The
two remain separate, composable signals -- see
src/threat_assessment.py's Level 5 pathway B for how they combine.

ConflictImminenceEstimator mirrors ThreatAssessmentEngine's per-track
temporal-state shape (not a new design): a confirmation-streak gate
before ESCALATING to a more urgent status (mirrors
ThreatAssessmentEngine._update_conflict_confirmation's pattern, applied
here to escalation instead of a single boolean) and a real-elapsed-
seconds hold before DE-escalating (mirrors ThreatAssessmentEngine's own
peak/hold de-escalation). Escalating is deliberately NOT instant here
(unlike ThreatAssessmentEngine's own level escalation) because a single
noisy trajectory-velocity fit can shift the geometric first-conflict
-step estimate more easily than a whole Level 1-5 change (which already
requires multiple independently-confirmed upstream signals at once) --
so this signal gets its own short confirmation gate on the way up, not
just on the way down. disappearance_grace_seconds mirrors
ThreatAssessmentEngine's default (1.0s).

Pure logic (aside from the estimator's internal per-track state), no
OpenCV/subprocess/network calls -- fully unit testable with synthetic
FilteredMotion/TrajectoryPrediction/PathIntersectionResult data.
"""

from __future__ import annotations

import logging
import math

from src.models import ConflictImminence, FilteredMotion, PathIntersectionResult, TrackedObject, TrajectoryPrediction

logger = logging.getLogger("atlas")

CONFLICT_NONE = "NONE"
CONFLICT_DISTANT = "DISTANT"
CONFLICT_SOON = "SOON"
CONFLICT_IMMINENT = "IMMINENT"
CONFLICT_UNKNOWN = "UNKNOWN"

# Ordering for escalation/de-escalation comparisons and for the ranking
# tiebreaker in src/threat_assessment.py's rank_active_threats. UNKNOWN
# is deliberately ranked with NONE (0) -- an UNKNOWN reading must never
# outrank a confidently-computed SOON/IMMINENT hazard merely because its
# own numeric evidence is missing.
IMMINENCE_STATUS_RANK = {
    CONFLICT_UNKNOWN: 0,
    CONFLICT_NONE: 0,
    CONFLICT_DISTANT: 1,
    CONFLICT_SOON: 2,
    CONFLICT_IMMINENT: 3,
}

_MOVING = "MOVING"
_STATIONARY = "STATIONARY"


class _TrackImminenceState:
    """Mutable per-track state carried across frames. Not a frozen model
    -- internal bookkeeping, never exposed outside this module."""

    __slots__ = (
        "last_seen",
        "candidate_status",
        "candidate_streak",
        "confirmed_status",
        "peak_status",
        "peak_rank",
        "peak_uncertain",
        "peak_first_conflict_step",
        "peak_reason_codes",
        "peak_time",
    )

    def __init__(self, now: float) -> None:
        self.last_seen = now
        self.candidate_status: str | None = None
        self.candidate_streak = 0
        self.confirmed_status = CONFLICT_NONE
        self.peak_status = CONFLICT_NONE
        self.peak_rank = 0
        self.peak_uncertain = False
        self.peak_first_conflict_step: float | None = None
        self.peak_reason_codes: tuple[str, ...] = ()
        self.peak_time = now


class ConflictImminenceEstimator:
    """Stateful per-track conservative conflict-imminence classifier.

    Args:
        imminent_step_frames: A predicted first-conflict-step at or below
            this many predicted frames ahead is categorized IMMINENT.
            Must be > 0. Chosen relative to trajectory.
            trajectory_prediction_horizon_frames (default horizon 10;
            default imminent_step_frames 3 -- roughly the nearest third
            of the horizon). Requires real-world tuning -- see module
            docstring on why this is frame-count-based, not seconds.
        soon_step_frames: A predicted first-conflict-step at or below
            this many predicted frames ahead (and above
            imminent_step_frames) is categorized SOON; anything further
            out (but still within the prediction horizon) is DISTANT.
            Must be >= imminent_step_frames.
        confirmation_frames: Consecutive frames a candidate status must
            be the SAME new value before it is confirmed at all --
            applies uniformly to any status change (mirrors
            ThreatAssessmentEngine._update_conflict_confirmation's
            streak pattern). Must be >= 1.
        deescalation_seconds: Once a higher-ranked status has been
            confirmed and reported for a track, a lower-ranked confirmed
            reading is withheld (the higher status keeps being reported)
            until this many continuous real-elapsed seconds of lower
            -ranked confirmed evidence have passed. This is a real
            wall-clock hold (time.time()-based, exactly like
            ThreatAssessmentEngine.deescalation_seconds) -- a different
            kind of "seconds" than estimated_seconds_to_conflict (which
            stays None): this one measures real elapsed time between
            calls, which IS reliably measurable; the unavailable one
            would require converting predicted trajectory frame-steps
            into real seconds, which is not defensible here. Must be >= 0.
        disappearance_grace_seconds: How long a track's internal
            imminence state is retained after it stops appearing in
            tracked_objects, mirroring ThreatAssessmentEngine's own
            disappearance_grace_seconds default (1.0s) for consistency.
            Must be >= 0.
    """

    def __init__(
        self,
        imminent_step_frames: float,
        soon_step_frames: float,
        confirmation_frames: int,
        deescalation_seconds: float,
        disappearance_grace_seconds: float,
    ) -> None:
        if imminent_step_frames <= 0:
            raise ValueError(
                f"imminent_step_frames must be > 0, got {imminent_step_frames}"
            )
        if soon_step_frames < imminent_step_frames:
            raise ValueError(
                "soon_step_frames must be >= imminent_step_frames, got "
                f"soon_step_frames={soon_step_frames}, "
                f"imminent_step_frames={imminent_step_frames}"
            )
        if confirmation_frames < 1:
            raise ValueError(
                f"confirmation_frames must be >= 1, got {confirmation_frames}"
            )
        if deescalation_seconds < 0:
            raise ValueError(f"deescalation_seconds must be >= 0, got {deescalation_seconds}")
        if disappearance_grace_seconds < 0:
            raise ValueError(
                f"disappearance_grace_seconds must be >= 0, got {disappearance_grace_seconds}"
            )

        self._imminent_step_frames = imminent_step_frames
        self._soon_step_frames = soon_step_frames
        self._confirmation_frames = confirmation_frames
        self._deescalation_seconds = deescalation_seconds
        self._disappearance_grace_seconds = disappearance_grace_seconds
        self._states: dict[int, _TrackImminenceState] = {}

    def estimate_for_tracks(
        self,
        tracked_objects: list[TrackedObject],
        filtered_motions: dict[int, FilteredMotion],
        resolved_predictions: dict[int, TrajectoryPrediction],
        resolved_intersection_results: dict[int, PathIntersectionResult],
        now: float,
        frame_index: int | None = None,
    ) -> dict[int, ConflictImminence]:
        """Estimate conflict imminence for every currently-tracked object.

        Only tracks with resolved filtered-motion AND a resolved
        trajectory prediction produce an estimate (mirrors
        ThreatAssessmentEngine.assess_for_tracks's own "skip incomplete
        data" convention). Tracks absent from `tracked_objects` this
        frame are neither estimated nor immediately forgotten -- see
        disappearance_grace_seconds.
        """
        active_ids = {obj.track_id for obj in tracked_objects}
        for stale_id in [tid for tid in self._states if tid not in active_ids]:
            state = self._states[stale_id]
            if (now - state.last_seen) >= self._disappearance_grace_seconds:
                del self._states[stale_id]

        results: dict[int, ConflictImminence] = {}
        for obj in tracked_objects:
            filtered = filtered_motions.get(obj.track_id)
            prediction = resolved_predictions.get(obj.track_id)
            if filtered is None or prediction is None:
                continue
            intersection = resolved_intersection_results.get(obj.track_id)

            estimate = self._estimate_track(
                obj, filtered, prediction, intersection, now, frame_index
            )
            results[obj.track_id] = estimate
            logger.debug("Conflict imminence: %s", estimate)

        return results

    def _estimate_track(
        self,
        obj: TrackedObject,
        filtered: FilteredMotion,
        prediction: TrajectoryPrediction,
        intersection: PathIntersectionResult | None,
        now: float,
        frame_index: int | None,
    ) -> ConflictImminence:
        state = self._states.setdefault(obj.track_id, _TrackImminenceState(now))
        state.last_seen = now

        raw_status, raw_uncertain, raw_step, raw_reason_codes = self._raw_estimate(
            filtered, prediction, intersection
        )

        self._update_confirmed_status(state, raw_status)
        confirmed_rank = IMMINENCE_STATUS_RANK[state.confirmed_status]

        held = confirmed_rank < state.peak_rank and (now - state.peak_time) < self._deescalation_seconds
        admit = not held
        if admit:
            state.peak_status = state.confirmed_status
            state.peak_rank = confirmed_rank
            # The uncertain/step/reason fields track this frame's raw
            # reading whenever the confirmed status itself is admitted --
            # frozen (held) only while a de-escalation is being withheld,
            # exactly like ThreatAssessmentEngine's peak_uncertain/
            # peak_reason_codes handling.
            state.peak_uncertain = raw_uncertain
            state.peak_first_conflict_step = raw_step
            state.peak_reason_codes = raw_reason_codes
            state.peak_time = now

        return ConflictImminence(
            track_id=obj.track_id,
            status=state.peak_status,
            first_conflict_step=state.peak_first_conflict_step,
            estimated_seconds_to_conflict=None,
            confidence=0.0 if (state.peak_uncertain or state.peak_status == CONFLICT_UNKNOWN) else 1.0,
            uncertain=state.peak_uncertain,
            reason_codes=state.peak_reason_codes,
            frame_index=frame_index,
            timestamp=now,
        )

    def _raw_estimate(
        self,
        filtered: FilteredMotion,
        prediction: TrajectoryPrediction,
        intersection: PathIntersectionResult | None,
    ) -> tuple[str, bool, float | None, tuple[str, ...]]:
        """This frame's instantaneous (unconfirmed) status -- returns
        (status, uncertain, first_conflict_step, reason_codes)."""
        if filtered.motion_state == _STATIONARY:
            # A stationary object never gets a *predicted dynamic*
            # conflict estimate -- an existing corridor obstruction is a
            # proximity/path-conflict concern handled directly by
            # ThreatAssessmentEngine's own Level 4 pathway, not invented
            # future motion here.
            return CONFLICT_NONE, False, None, ("STATIONARY_NO_DYNAMIC_CONFLICT",)

        if filtered.motion_state != _MOVING:
            # UNCERTAIN / INSUFFICIENT_HISTORY -- not enough reliable
            # motion evidence to say anything at all.
            return CONFLICT_UNKNOWN, True, None, ("INSUFFICIENT_MOTION_EVIDENCE",)

        if intersection is None or not intersection.valid:
            return CONFLICT_UNKNOWN, True, None, ("TRAJECTORY_UNAVAILABLE",)

        if not intersection.intersects:
            return CONFLICT_NONE, intersection.uncertain, None, ("NO_PREDICTED_CONFLICT",)

        step = self._first_conflict_step(prediction, intersection)

        if step <= self._imminent_step_frames:
            status = CONFLICT_IMMINENT
        elif step <= self._soon_step_frames:
            status = CONFLICT_SOON
        else:
            status = CONFLICT_DISTANT

        uncertain = intersection.uncertain
        if uncertain and status == CONFLICT_IMMINENT:
            # Uncertain evidence (low-confidence camera-motion
            # compensation, or RAW_FALLBACK) can never produce a
            # confident IMMINENT report -- capped at SOON instead of a
            # false-confident claim from a small step number alone.
            status = CONFLICT_SOON
            reason = ("PREDICTED_CONFLICT_UNCERTAIN_CAPPED",)
        elif status == CONFLICT_IMMINENT:
            reason = ("PREDICTED_CONFLICT_IMMINENT",)
        elif status == CONFLICT_SOON:
            reason = ("PREDICTED_CONFLICT_SOON",)
        else:
            reason = ("PREDICTED_CONFLICT_DISTANT",)

        return status, uncertain, step, reason

    @staticmethod
    def _first_conflict_step(
        prediction: TrajectoryPrediction, intersection: PathIntersectionResult
    ) -> float:
        """Earliest predicted frame-step at which the segment meaningfully
        enters the corridor, derived from PathIntersectionAnalyzer's
        already-computed geometry -- never a second intersection
        algorithm. Assumes intersection.intersects is True."""
        horizon = prediction.prediction_horizon_frames
        start = prediction.current_center
        end = prediction.predicted_center

        if intersection.starts_inside:
            return 0.0

        if intersection.intersection_point is not None:
            segment_length = math.dist(start, end)
            if segment_length <= 0:
                return 0.0
            traveled = math.dist(start, intersection.intersection_point)
            fraction = max(0.0, min(1.0, traveled / segment_length))
            return fraction * horizon

        if intersection.ends_inside:
            # Geometric edge case (e.g. an exact-boundary floating-point
            # miss in the edge-crossing search) -- conservatively assume
            # the conflict is no sooner than the full predicted horizon.
            return float(horizon)

        # Not expected when intersection.intersects is True (see
        # PathIntersectionAnalyzer.analyze); fall back to the full horizon.
        return float(horizon)

    def _update_confirmed_status(self, state: _TrackImminenceState, raw_status: str) -> None:
        """confirmation_frames consecutive identical raw readings before
        the confirmed candidate status changes -- mirrors
        ThreatAssessmentEngine._update_conflict_confirmation's streak
        pattern, generalized from a boolean to this module's five-way
        status. Applies uniformly to escalation AND de-escalation of the
        *confirmed* value; the real-seconds hold in _estimate_track is a
        second, independent layer applied on top of this one (mirroring
        ThreatAssessmentEngine's own two-layer confirm-then-hold shape)."""
        if raw_status == state.confirmed_status:
            state.candidate_status = None
            state.candidate_streak = 0
            return

        if state.candidate_status == raw_status:
            state.candidate_streak += 1
        else:
            state.candidate_status = raw_status
            state.candidate_streak = 1

        if state.candidate_streak >= self._confirmation_frames:
            state.confirmed_status = raw_status
            state.candidate_status = None
            state.candidate_streak = 0
