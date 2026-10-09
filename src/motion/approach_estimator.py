"""Confirmed-approach detection for Atlas's audio hazard system.

Answers a strictly stronger question than FilteredMotion.motion_state ==
"MOVING": is this object actually getting closer, not just moving
somehow? Does NOT redefine MOVING -- MOVING keeps its existing meaning
(src/motion/motion_state_filter.py) and is used here as a hard
prerequisite gate, not evidence in its own right.

Mirrors MotionStateFilter's own architecture: a dead-zone-free but
otherwise identical confirmation-streak state machine (pending
transitions must persist for several consecutive frames before the
confirmed state flips), so a single noisy frame can never claim
"approaching" on its own.

Requires >= minimum_approach_cues (default 2) of 6 independent cues to
be true THIS frame, sustained for >= approaching_confirmation_frames
(default 3) consecutive frames, before confirming APPROACHING. A
confirmed-closing streak built on an uncertain resolved-motion source
(FilteredMotion.uncertain) is reported APPROACHING_UNCERTAIN, never a
confident APPROACHING claim.

Pure logic, no OpenCV/drawing calls, no dependency on a live camera or
YOLO model -- fully unit testable with synthetic TrackedObject/
FilteredMotion/TrajectoryPrediction/PathIntersectionResult data.
"""

from __future__ import annotations

import math

from src.models import (
    ApproachResult,
    FilteredMotion,
    PathIntersectionResult,
    TrackedObject,
    TrajectoryPrediction,
)
from src.motion.relative_proximity_estimator import UNKNOWN, RelativeProximityEstimator, is_nearer

NOT_APPROACHING = "NOT_APPROACHING"
APPROACHING_UNCERTAIN = "APPROACHING_UNCERTAIN"
APPROACHING = "APPROACHING"

GROUND_POINT_CLOSING = "GROUND_POINT_CLOSING"
PROXIMITY_INCREASING = "PROXIMITY_INCREASING"
CORRIDOR_DISTANCE_DECREASING = "CORRIDOR_DISTANCE_DECREASING"
SCALE_GROWTH = "SCALE_GROWTH"
TRAJECTORY_TOWARD_CORRIDOR = "TRAJECTORY_TOWARD_CORRIDOR"
PATH_INTERSECTS = "PATH_INTERSECTS"

_ALL_CUES = (
    GROUND_POINT_CLOSING,
    PROXIMITY_INCREASING,
    CORRIDOR_DISTANCE_DECREASING,
    SCALE_GROWTH,
    TRAJECTORY_TOWARD_CORRIDOR,
    PATH_INTERSECTS,
)

_CLOSING = "CLOSING"
_NOT_CLOSING = "NOT_CLOSING"


class _TrackApproachState:
    """Mutable per-track state carried across frames. Not a frozen
    model -- internal bookkeeping, never exposed outside this module."""

    __slots__ = ("confirmed_closing", "candidate_state", "candidate_streak")

    def __init__(self) -> None:
        self.confirmed_closing = False
        self.candidate_state: str | None = None
        self.candidate_streak = 0


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _least_squares_slope(values: list[float]) -> float:
    """Least-squares slope over index t = 0..n-1 -- same regression
    convention as TrajectoryPredictor._fit_velocity, generalized to a
    single 1D series."""
    n = len(values)
    if n < 2:
        return 0.0
    t_mean = (n - 1) / 2
    denominator = sum((t - t_mean) ** 2 for t in range(n))
    if denominator == 0:
        return 0.0
    value_mean = sum(values) / n
    numerator = sum((t - t_mean) * (v - value_mean) for t, v in enumerate(values))
    return numerator / denominator


def _polygon_centroid(polygon: tuple[tuple[int, int], ...]) -> tuple[float, float]:
    """Simple vertex-average centroid -- an approximation, not an exact
    area-weighted geometric centroid, sufficient for a small, roughly
    symmetric corridor quadrilateral used only as a trend reference
    point."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


class ApproachEstimator:
    """Stateful dead-zone-free, confirmation-streak approach detector.

    Args:
        proximity_estimator: Reused (not duplicated) to classify the
            oldest history sample's proximity zone for the PROXIMITY_
            INCREASING cue, using the exact same far/near boundaries as
            the caller's current-frame proximity classification.
        minimum_history_samples: Minimum position-history samples
            required before approach is evaluated at all; below this,
            NOT_APPROACHING (confident negative, not evaluated). Must be
            >= 2.
        minimum_approach_cues: Minimum number of the 6 cues (out of 6)
            that must be true in the same frame for it to count as a
            CLOSING candidate. Must be within [1, 6].
        approaching_confirmation_frames: Consecutive CLOSING-candidate
            frames required before the confirmed state actually flips to
            closing (and the same count of NOT-CLOSING frames to flip
            back) -- a single noisy frame can never flip the label.
            Must be >= 1.
        ground_point_trend_threshold: Pixels/frame -- the GROUND_POINT_
            CLOSING cue requires the bottom-center y trend to exceed
            this. Must be > 0.
        scale_growth_threshold_fraction: Fractional bbox-area growth per
            frame -- the SCALE_GROWTH cue requires the area trend
            (normalized by mean area) to exceed this. Must be > 0.
        corridor_distance_trend_threshold: Pixels/frame -- the CORRIDOR_
            DISTANCE_DECREASING cue requires the distance-to-corridor-
            centroid trend to be more negative than this. Must be > 0.
    """

    def __init__(
        self,
        proximity_estimator: RelativeProximityEstimator,
        minimum_history_samples: int,
        minimum_approach_cues: int,
        approaching_confirmation_frames: int,
        ground_point_trend_threshold: float,
        scale_growth_threshold_fraction: float,
        corridor_distance_trend_threshold: float,
    ) -> None:
        if minimum_history_samples < 2:
            raise ValueError(
                f"minimum_history_samples must be >= 2, got {minimum_history_samples}"
            )
        if not (1 <= minimum_approach_cues <= len(_ALL_CUES)):
            raise ValueError(
                f"minimum_approach_cues must be within [1, {len(_ALL_CUES)}], "
                f"got {minimum_approach_cues}"
            )
        if approaching_confirmation_frames < 1:
            raise ValueError(
                "approaching_confirmation_frames must be >= 1, got "
                f"{approaching_confirmation_frames}"
            )
        if ground_point_trend_threshold <= 0:
            raise ValueError(
                "ground_point_trend_threshold must be > 0, got "
                f"{ground_point_trend_threshold}"
            )
        if scale_growth_threshold_fraction <= 0:
            raise ValueError(
                "scale_growth_threshold_fraction must be > 0, got "
                f"{scale_growth_threshold_fraction}"
            )
        if corridor_distance_trend_threshold <= 0:
            raise ValueError(
                "corridor_distance_trend_threshold must be > 0, got "
                f"{corridor_distance_trend_threshold}"
            )

        self._proximity_estimator = proximity_estimator
        self._minimum_history_samples = minimum_history_samples
        self._minimum_approach_cues = minimum_approach_cues
        self._approaching_confirmation_frames = approaching_confirmation_frames
        self._ground_point_trend_threshold = ground_point_trend_threshold
        self._scale_growth_threshold_fraction = scale_growth_threshold_fraction
        self._corridor_distance_trend_threshold = corridor_distance_trend_threshold
        self._states: dict[int, _TrackApproachState] = {}

    def estimate(
        self,
        track: TrackedObject,
        filtered: FilteredMotion,
        resolved_prediction: TrajectoryPrediction,
        resolved_intersection: PathIntersectionResult | None,
        proximity_zone: str,
        corridor_polygon: tuple[tuple[int, int], ...],
        frame_height: int,
    ) -> ApproachResult:
        """Estimate one track's approach state for the current frame."""
        state = self._states.setdefault(track.track_id, _TrackApproachState())

        gate_passed = (
            filtered.motion_state == "MOVING"
            and len(track.position_history) >= self._minimum_history_samples
            and len(track.position_history) == len(track.size_history)
        )
        if not gate_passed:
            state.confirmed_closing = False
            state.candidate_state = None
            state.candidate_streak = 0
            return ApproachResult(
                track_id=track.track_id,
                state=NOT_APPROACHING,
                confidence=0.0,
                proximity_zone=proximity_zone,
                closing_score=0.0,
                corridor_distance_trend=0.0,
                scale_growth_trend=0.0,
                ground_point_trend=0.0,
                confirmation_frames=0,
                uncertain=False,
                evidence_flags=(),
            )

        positions = track.position_history
        sizes = track.size_history
        n = len(positions)

        bottom_y_history = [pos[1] + size[1] / 2 for pos, size in zip(positions, sizes)]
        ground_point_trend = _least_squares_slope(bottom_y_history)

        areas = [w * h for w, h in sizes]
        mean_area = sum(areas) / n
        scale_growth_trend = (
            _least_squares_slope(areas) / mean_area if mean_area > 0 else 0.0
        )

        corridor_centroid = _polygon_centroid(corridor_polygon)
        distances = [math.dist(p, corridor_centroid) for p in positions]
        corridor_distance_trend = _least_squares_slope(distances)

        oldest_zone = self._proximity_estimator.estimate(bottom_y_history[0], frame_height)

        trajectory_toward_corridor = False
        if resolved_prediction.valid:
            current_dist = math.dist(resolved_prediction.current_center, corridor_centroid)
            predicted_dist = math.dist(resolved_prediction.predicted_center, corridor_centroid)
            trajectory_toward_corridor = predicted_dist < current_dist

        path_intersects = resolved_intersection is not None and resolved_intersection.intersects

        cue_values = {
            GROUND_POINT_CLOSING: ground_point_trend > self._ground_point_trend_threshold,
            PROXIMITY_INCREASING: is_nearer(proximity_zone, oldest_zone),
            CORRIDOR_DISTANCE_DECREASING: (
                corridor_distance_trend < -self._corridor_distance_trend_threshold
            ),
            SCALE_GROWTH: scale_growth_trend > self._scale_growth_threshold_fraction,
            TRAJECTORY_TOWARD_CORRIDOR: trajectory_toward_corridor,
            PATH_INTERSECTS: path_intersects,
        }
        evidence_flags = tuple(name for name in _ALL_CUES if cue_values[name])
        confidence = len(evidence_flags) / len(_ALL_CUES)

        closing_score = (
            _clip(ground_point_trend / self._ground_point_trend_threshold, -1.0, 1.0)
            + _clip(
                -corridor_distance_trend / self._corridor_distance_trend_threshold, -1.0, 1.0
            )
            + _clip(scale_growth_trend / self._scale_growth_threshold_fraction, -1.0, 1.0)
        ) / 3.0

        candidate = _CLOSING if len(evidence_flags) >= self._minimum_approach_cues else _NOT_CLOSING
        self._update_confirmed_state(state, candidate)

        if not state.confirmed_closing:
            final_state = NOT_APPROACHING
            uncertain = False
        else:
            final_state = APPROACHING_UNCERTAIN if filtered.uncertain else APPROACHING
            uncertain = filtered.uncertain

        return ApproachResult(
            track_id=track.track_id,
            state=final_state,
            confidence=confidence,
            proximity_zone=proximity_zone,
            closing_score=closing_score,
            corridor_distance_trend=corridor_distance_trend,
            scale_growth_trend=scale_growth_trend,
            ground_point_trend=ground_point_trend,
            confirmation_frames=state.candidate_streak,
            uncertain=uncertain,
            evidence_flags=evidence_flags,
        )

    def estimate_for_tracks(
        self,
        tracked_objects: list[TrackedObject],
        filtered_motions: dict[int, FilteredMotion],
        resolved_predictions: dict[int, TrajectoryPrediction],
        resolved_intersection_results: dict[int, PathIntersectionResult],
        proximity_by_track: dict[int, str],
        corridor_polygon: tuple[tuple[int, int], ...],
        frame_height: int,
    ) -> dict[int, ApproachResult]:
        """Estimate every tracked object in one call.

        Prunes internal per-track state for track_ids no longer present,
        so a dropped track's state doesn't linger forever. A track
        missing from filtered_motions/resolved_predictions is skipped
        rather than raising, matching this codebase's established
        "never crash on a missing dict entry" convention.
        """
        active_ids = {obj.track_id for obj in tracked_objects}
        for stale_id in set(self._states) - active_ids:
            del self._states[stale_id]

        results: dict[int, ApproachResult] = {}
        for obj in tracked_objects:
            filtered = filtered_motions.get(obj.track_id)
            prediction = resolved_predictions.get(obj.track_id)
            if filtered is None or prediction is None:
                continue
            intersection = resolved_intersection_results.get(obj.track_id)
            proximity_zone = proximity_by_track.get(obj.track_id, UNKNOWN)
            results[obj.track_id] = self.estimate(
                obj, filtered, prediction, intersection, proximity_zone,
                corridor_polygon, frame_height,
            )
        return results

    def _update_confirmed_state(self, state: _TrackApproachState, candidate: str) -> None:
        """Advance (or reset) the pending-transition streak, flipping
        confirmed_closing only once the streak reaches the configured
        confirmation-frame count -- mirrors MotionStateFilter's own
        _update_confirmed_state exactly."""
        currently = _CLOSING if state.confirmed_closing else _NOT_CLOSING
        if candidate == currently:
            state.candidate_state = None
            state.candidate_streak = 0
            return

        if state.candidate_state == candidate:
            state.candidate_streak += 1
        else:
            state.candidate_state = candidate
            state.candidate_streak = 1

        if state.candidate_streak >= self._approaching_confirmation_frames:
            state.confirmed_closing = candidate == _CLOSING
            state.candidate_state = None
            state.candidate_streak = 0
