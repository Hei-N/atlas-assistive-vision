"""Unit tests for src/conflict_imminence.py -- ConflictImminenceEstimator
and its raw geometric first-conflict-step derivation. Pure logic, no
camera/model/network/subprocess involved -- synthetic FilteredMotion/
TrajectoryPrediction/PathIntersectionResult data throughout, matching
tests/test_threat_assessment.py's own conventions.
"""

from __future__ import annotations

import inspect

import pytest

from src.conflict_imminence import (
    CONFLICT_DISTANT,
    CONFLICT_IMMINENT,
    CONFLICT_NONE,
    CONFLICT_SOON,
    CONFLICT_UNKNOWN,
    IMMINENCE_STATUS_RANK,
    ConflictImminenceEstimator,
)
from src.models import BoundingBox, FilteredMotion, PathIntersectionResult, TrackedObject, TrajectoryPrediction


def make_object(track_id: int = 1, region: str = "left") -> TrackedObject:
    bbox = BoundingBox(x1=0, y1=0, x2=20, y2=20)
    return TrackedObject(
        track_id=track_id, class_id=2, class_name="car", confidence=0.9,
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


def make_prediction(
    current_center: tuple[int, int] = (0, 0),
    predicted_center: tuple[int, int] = (100, 0),
    horizon: int = 10,
    uncertain: bool = False,
    valid: bool = True,
) -> TrajectoryPrediction:
    return TrajectoryPrediction(
        valid=valid, velocity_x=10.0, velocity_y=0.0, speed_px_per_frame=10.0,
        direction="right", current_center=current_center, predicted_center=predicted_center,
        observations_used=5, prediction_horizon_frames=horizon, uncertain=uncertain,
    )


def make_intersection(
    intersects: bool = False,
    starts_inside: bool = False,
    ends_inside: bool = False,
    intersection_point: tuple[int, int] | None = None,
    uncertain: bool = False,
    valid: bool = True,
) -> PathIntersectionResult:
    return PathIntersectionResult(
        valid=valid, intersects=intersects, starts_inside=starts_inside, ends_inside=ends_inside,
        intersection_point=intersection_point, track_id=1, reason="test", uncertain=uncertain,
    )


def make_estimator(
    imminent_step_frames: float = 3,
    soon_step_frames: float = 7,
    confirmation_frames: int = 1,
    deescalation_seconds: float = 0.0,
    disappearance_grace_seconds: float = 1.0,
) -> ConflictImminenceEstimator:
    return ConflictImminenceEstimator(
        imminent_step_frames=imminent_step_frames,
        soon_step_frames=soon_step_frames,
        confirmation_frames=confirmation_frames,
        deescalation_seconds=deescalation_seconds,
        disappearance_grace_seconds=disappearance_grace_seconds,
    )


def estimate_once(estimator, obj, filtered, prediction, intersection, now, frame_index=None):
    result = estimator.estimate_for_tracks(
        [obj], {obj.track_id: filtered}, {obj.track_id: prediction},
        {obj.track_id: intersection} if intersection is not None else {},
        now, frame_index,
    )
    return result.get(obj.track_id)


# --- GEOMETRY ----------------------------------------------------------------


class TestGeometry:
    def test_no_intersection_is_none(self):
        estimator = make_estimator()
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("MOVING"),
            make_prediction(), make_intersection(intersects=False), now=0.0,
        )
        assert result.status == CONFLICT_NONE
        assert result.first_conflict_step is None

    def test_early_intersection_is_imminent_with_correct_step(self):
        # current=(0,0) -> predicted=(100,0), horizon=10, intersection at
        # x=30 (30% along) -> first_conflict_step == 3.0 <= imminent(3).
        estimator = make_estimator()
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("MOVING"),
            make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
            make_intersection(intersects=True, ends_inside=True, intersection_point=(30, 0)),
            now=0.0,
        )
        assert result.status == CONFLICT_IMMINENT
        assert result.first_conflict_step == pytest.approx(3.0)

    def test_mid_intersection_is_soon(self):
        estimator = make_estimator()
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("MOVING"),
            make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
            make_intersection(intersects=True, ends_inside=True, intersection_point=(50, 0)),
            now=0.0,
        )
        assert result.status == CONFLICT_SOON
        assert result.first_conflict_step == pytest.approx(5.0)

    def test_later_intersection_is_distant(self):
        estimator = make_estimator()
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("MOVING"),
            make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
            make_intersection(intersects=True, ends_inside=True, intersection_point=(80, 0)),
            now=0.0,
        )
        assert result.status == CONFLICT_DISTANT
        assert result.first_conflict_step == pytest.approx(8.0)

    def test_starts_inside_is_step_zero(self):
        estimator = make_estimator()
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("MOVING"),
            make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
            make_intersection(intersects=True, starts_inside=True, ends_inside=True),
            now=0.0,
        )
        assert result.status == CONFLICT_IMMINENT
        assert result.first_conflict_step == pytest.approx(0.0)

    def test_one_edge_noise_contact_rejected_without_confirmation(self):
        # confirmation_frames=3 -- a single IMMINENT-worthy frame must not
        # be reported as IMMINENT yet.
        estimator = make_estimator(confirmation_frames=3)
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("MOVING"),
            make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
            make_intersection(intersects=True, ends_inside=True, intersection_point=(10, 0)),
            now=0.0,
        )
        assert result.status == CONFLICT_NONE


# --- TIMING --------------------------------------------------------------------


class TestTiming:
    def test_estimated_seconds_to_conflict_always_none(self):
        estimator = make_estimator()
        obj = make_object()
        for point, expected_status in (
            ((10, 0), CONFLICT_IMMINENT), ((50, 0), CONFLICT_SOON), ((80, 0), CONFLICT_DISTANT),
        ):
            result = estimate_once(
                estimator, obj, make_filtered("MOVING"),
                make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
                make_intersection(intersects=True, ends_inside=True, intersection_point=point),
                now=0.0,
            )
            assert result.status == expected_status
            assert result.estimated_seconds_to_conflict is None

    def test_no_fabricated_seconds_when_timing_unavailable(self):
        estimator = make_estimator()
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("UNCERTAIN"), make_prediction(),
            make_intersection(intersects=False), now=0.0,
        )
        assert result.status == CONFLICT_UNKNOWN
        assert result.estimated_seconds_to_conflict is None


# --- STATIONARY ------------------------------------------------------------------


class TestStationary:
    def test_stationary_object_never_gets_dynamic_conflict(self):
        estimator = make_estimator()
        obj = make_object()
        # Even with geometry that would otherwise be IMMINENT.
        result = estimate_once(
            estimator, obj, make_filtered("STATIONARY"),
            make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
            make_intersection(intersects=True, starts_inside=True, ends_inside=True),
            now=0.0,
        )
        assert result.status == CONFLICT_NONE
        assert result.first_conflict_step is None
        assert result.uncertain is False


# --- LATERAL / DECOUPLING FROM APPROACH ------------------------------------------


class TestLateralDecoupling:
    def test_estimate_for_tracks_has_no_approach_parameter(self):
        params = set(inspect.signature(ConflictImminenceEstimator.estimate_for_tracks).parameters)
        assert not any("approach" in p.lower() for p in params)

    def test_sideways_crossing_produces_imminence_without_approach_evidence(self):
        # A perpendicular (sideways) predicted path crossing the corridor
        # produces a real status purely from motion + intersection
        # geometry -- ApproachResult is never consulted at all.
        estimator = make_estimator()
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("MOVING"),
            make_prediction(current_center=(-50, 300), predicted_center=(50, 300), horizon=10),
            make_intersection(intersects=True, ends_inside=True, intersection_point=(-20, 300)),
            now=0.0,
        )
        assert result.status in (CONFLICT_IMMINENT, CONFLICT_SOON, CONFLICT_DISTANT)


# --- UNCERTAINTY -----------------------------------------------------------------


class TestUncertainty:
    def test_raw_fallback_uncertain_intersection_capped_below_imminent(self):
        estimator = make_estimator()
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("MOVING", uncertain=True),
            make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10, uncertain=True),
            make_intersection(intersects=True, ends_inside=True, intersection_point=(10, 0), uncertain=True),
            now=0.0,
        )
        assert result.status != CONFLICT_IMMINENT
        assert result.status == CONFLICT_SOON
        assert result.uncertain is True
        assert result.confidence == 0.0

    def test_uncertain_motion_state_is_unknown(self):
        estimator = make_estimator()
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("UNCERTAIN"), make_prediction(),
            make_intersection(intersects=True), now=0.0,
        )
        assert result.status == CONFLICT_UNKNOWN
        assert result.uncertain is True

    def test_insufficient_history_is_unknown(self):
        estimator = make_estimator()
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("INSUFFICIENT_HISTORY"), make_prediction(),
            make_intersection(intersects=True), now=0.0,
        )
        assert result.status == CONFLICT_UNKNOWN

    def test_invalid_intersection_is_unknown(self):
        estimator = make_estimator()
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("MOVING"), make_prediction(),
            make_intersection(intersects=False, valid=False), now=0.0,
        )
        assert result.status == CONFLICT_UNKNOWN

    def test_missing_intersection_result_is_unknown(self):
        estimator = make_estimator()
        obj = make_object()
        result = estimator.estimate_for_tracks(
            [obj], {obj.track_id: make_filtered("MOVING")},
            {obj.track_id: make_prediction()}, {}, now=0.0,
        )[obj.track_id]
        assert result.status == CONFLICT_UNKNOWN


# --- TEMPORAL STABILITY ------------------------------------------------------------


class TestTemporalStability:
    def test_one_frame_imminent_spike_rejected(self):
        estimator = make_estimator(confirmation_frames=3)
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("MOVING"),
            make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
            make_intersection(intersects=True, ends_inside=True, intersection_point=(10, 0)),
            now=0.0,
        )
        assert result.status == CONFLICT_NONE

    def test_persistent_imminent_accepted_after_confirmation_streak(self):
        estimator = make_estimator(confirmation_frames=3)
        obj = make_object()
        result = None
        for i in range(3):
            result = estimate_once(
                estimator, obj, make_filtered("MOVING"),
                make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
                make_intersection(intersects=True, ends_inside=True, intersection_point=(10, 0)),
                now=float(i),
            )
        assert result.status == CONFLICT_IMMINENT

    def test_deescalation_does_not_flicker(self):
        estimator = make_estimator(confirmation_frames=1, deescalation_seconds=0.75)
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("MOVING"),
            make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
            make_intersection(intersects=True, ends_inside=True, intersection_point=(10, 0)),
            now=0.0,
        )
        assert result.status == CONFLICT_IMMINENT

        # 0.1s later evidence disappears entirely -- still within the
        # 0.75s hold, so IMMINENT keeps being reported (no flicker).
        result = estimate_once(
            estimator, obj, make_filtered("MOVING"),
            make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
            make_intersection(intersects=False), now=0.1,
        )
        assert result.status == CONFLICT_IMMINENT

        # 0.8s after the peak -- hold has expired.
        result = estimate_once(
            estimator, obj, make_filtered("MOVING"),
            make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
            make_intersection(intersects=False), now=0.8,
        )
        assert result.status == CONFLICT_NONE

    def test_disappearance_grace_preserves_state_on_reappearance(self):
        estimator = make_estimator(confirmation_frames=1, deescalation_seconds=0.75, disappearance_grace_seconds=1.0)
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("MOVING"),
            make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
            make_intersection(intersects=True, ends_inside=True, intersection_point=(10, 0)),
            now=0.0,
        )
        assert result.status == CONFLICT_IMMINENT

        # Vanishes for 0.5s (within grace) -- no assessment for an absent
        # track, but state is preserved.
        empty_result = estimator.estimate_for_tracks([], {}, {}, {}, now=0.5)
        assert empty_result == {}

        # Reappears at 0.7s (within the 0.75s hold measured from peak)
        # with weak evidence -- still held.
        result = estimate_once(
            estimator, obj, make_filtered("MOVING"),
            make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
            make_intersection(intersects=False), now=0.7,
        )
        assert result.status == CONFLICT_IMMINENT

    def test_stale_track_removed_after_grace_expires(self):
        estimator = make_estimator(confirmation_frames=1, deescalation_seconds=5.0, disappearance_grace_seconds=1.0)
        obj = make_object()
        result = estimate_once(
            estimator, obj, make_filtered("MOVING"),
            make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
            make_intersection(intersects=True, ends_inside=True, intersection_point=(10, 0)),
            now=0.0,
        )
        assert result.status == CONFLICT_IMMINENT

        # Absent for longer than disappearance_grace_seconds -- state is
        # pruned entirely, even though deescalation_seconds is huge.
        estimator.estimate_for_tracks([], {}, {}, {}, now=2.0)

        result = estimate_once(
            estimator, obj, make_filtered("MOVING"),
            make_prediction(current_center=(0, 0), predicted_center=(100, 0), horizon=10),
            make_intersection(intersects=False), now=2.1,
        )
        assert result.status == CONFLICT_NONE


# --- RANK ORDERING -------------------------------------------------------------------


def test_imminence_status_rank_ordering():
    assert (
        IMMINENCE_STATUS_RANK[CONFLICT_NONE]
        == IMMINENCE_STATUS_RANK[CONFLICT_UNKNOWN]
        < IMMINENCE_STATUS_RANK[CONFLICT_DISTANT]
        < IMMINENCE_STATUS_RANK[CONFLICT_SOON]
        < IMMINENCE_STATUS_RANK[CONFLICT_IMMINENT]
    )


# --- CONSTRUCTOR VALIDATION ------------------------------------------------------------


class TestConstructorValidation:
    def test_rejects_zero_imminent_step_frames(self):
        with pytest.raises(ValueError):
            make_estimator(imminent_step_frames=0)

    def test_rejects_soon_below_imminent(self):
        with pytest.raises(ValueError):
            make_estimator(imminent_step_frames=5, soon_step_frames=3)

    def test_rejects_zero_confirmation_frames(self):
        with pytest.raises(ValueError):
            make_estimator(confirmation_frames=0)

    def test_rejects_negative_deescalation_seconds(self):
        with pytest.raises(ValueError):
            make_estimator(deescalation_seconds=-1.0)

    def test_rejects_negative_disappearance_grace_seconds(self):
        with pytest.raises(ValueError):
            make_estimator(disappearance_grace_seconds=-1.0)
