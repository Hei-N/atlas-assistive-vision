"""Developer-only camera-motion-compensation validation (Atlas Phase 4,
--validate-compensation CLI mode).

Pure functions -- no cv2/drawing calls, no dependency on a live camera or
YOLO model, so they can be fully unit tested with synthetic TrackedObject/
CompensatedMotion/MotionEstimate/PathIntersectionResult data. This module
helps a developer visually confirm compensation is working on real
footage, and also builds the per-object CSV/debug-panel row that includes
the resolved motion (src/motion/motion_resolver.py), the filtered motion
(src/motion/motion_state_filter.py), and both's corridor-intersection
outcome -- see PerObjectValidation's resolved_velocity_x/y, motion_source,
prediction_uncertain, raw_intersects_corridor,
resolved_intersects_corridor, filtered_velocity_x/y, motion_filter_state,
motion_filter_reason, motion_confirmation_frames fields.
"""

from __future__ import annotations

import math

from src.models import (
    CompensatedMotion,
    FilteredMotion,
    MotionEstimate,
    PathIntersectionResult,
    PerObjectValidation,
    ResolvedMotion,
    TrackedObject,
    ValidationStats,
)
from src.motion.motion_resolver import RAW_FALLBACK
from src.motion.motion_state_filter import INSUFFICIENT_HISTORY

STATIONARY = "STATIONARY"
CAMERA_DOMINATED = "CAMERA-DOMINATED"
OBJECT_MOVING = "OBJECT-MOVING"
UNCERTAIN = "UNCERTAIN"

_REDUCTION_EPSILON = 1e-6


def classify_status(
    raw_speed: float,
    compensated_speed: float,
    camera_motion_applied: bool,
    stationary_threshold_px: float,
) -> str:
    """Deterministic per-object validation status from existing fields.

    Precedence (most specific case checked first):
      1. Camera motion was NOT trusted/applied (missing, invalid, or below
         the compensator's confidence threshold -- see
         CompensatedMotion.camera_motion_applied) and the object still
         shows non-trivial raw motion: UNCERTAIN. Without a trustworthy
         camera estimate, apparent motion can't be confidently attributed
         to the camera or the object.
      2. Camera motion WAS applied, raw speed was clearly above the
         stationary threshold, but the compensated speed fell to/below
         it: CAMERA-DOMINATED. The apparent motion is fully explained by
         estimated camera movement.
      3. Compensated speed at/below the stationary threshold: STATIONARY.
         Covers both "never appeared to move" and the untrusted-but-
         genuinely-still case from rule 1's fallthrough.
      4. Otherwise (compensated speed remains above threshold): OBJECT-
         MOVING -- genuine motion remains after compensation.

    Args:
        raw_speed: Magnitude of the object's uncompensated velocity
            (pixels/frame).
        compensated_speed: Magnitude of the object's compensated velocity
            (pixels/frame), from CompensatedMotion.compensated_speed.
        camera_motion_applied: From CompensatedMotion.camera_motion_applied.
        stationary_threshold_px: Same threshold MotionCompensator itself
            uses (its stationary_threshold_px) -- reused here rather than
            inventing a second one, so "stationary" means the same thing
            throughout the pipeline.
    """
    if not camera_motion_applied and raw_speed > stationary_threshold_px:
        return UNCERTAIN
    if (
        camera_motion_applied
        and raw_speed > stationary_threshold_px
        and compensated_speed <= stationary_threshold_px
    ):
        return CAMERA_DOMINATED
    if compensated_speed <= stationary_threshold_px:
        return STATIONARY
    return OBJECT_MOVING


def build_per_object_validation(
    tracked_objects: list[TrackedObject],
    compensated_motions: dict[int, CompensatedMotion],
    camera_motion: MotionEstimate | None,
    stationary_threshold_px: float,
    resolved_motions: dict[int, ResolvedMotion],
    raw_intersections: dict[int, PathIntersectionResult],
    resolved_intersections: dict[int, PathIntersectionResult],
    filtered_motions: dict[int, FilteredMotion],
) -> list[PerObjectValidation]:
    """One PerObjectValidation per tracked object with compensated data.

    A track_id missing from compensated_motions (compensation disabled,
    or not yet available) is simply skipped -- never raises. A track
    present in compensated_motions but missing from resolved_motions/
    raw_intersections/resolved_intersections/filtered_motions (defensive
    -- shouldn't happen, all four are built from the same tracked_objects
    list) degrades to a safe RAW_FALLBACK/zero-velocity/uncertain=True/
    intersects=False/INSUFFICIENT_HISTORY row rather than crashing or
    being dropped.
    """
    camera_confidence = camera_motion.confidence if camera_motion is not None else 0.0

    rows: list[PerObjectValidation] = []
    for obj in tracked_objects:
        motion = compensated_motions.get(obj.track_id)
        if motion is None:
            continue

        raw_speed = math.hypot(motion.raw_velocity_x, motion.raw_velocity_y)
        status = classify_status(
            raw_speed,
            motion.compensated_speed,
            motion.camera_motion_applied,
            stationary_threshold_px,
        )

        resolved = resolved_motions.get(obj.track_id)
        if resolved is not None:
            resolved_velocity_x = resolved.velocity_x
            resolved_velocity_y = resolved.velocity_y
            resolved_speed = resolved.speed
            motion_source = resolved.source
            prediction_uncertain = resolved.uncertain
        else:
            resolved_velocity_x = 0.0
            resolved_velocity_y = 0.0
            resolved_speed = 0.0
            motion_source = RAW_FALLBACK
            prediction_uncertain = True

        raw_intersection = raw_intersections.get(obj.track_id)
        resolved_intersection = resolved_intersections.get(obj.track_id)
        raw_intersects_corridor = (
            raw_intersection.intersects if raw_intersection is not None else False
        )
        resolved_intersects_corridor = (
            resolved_intersection.intersects
            if resolved_intersection is not None
            else False
        )

        filtered = filtered_motions.get(obj.track_id)
        if filtered is not None:
            filtered_velocity_x = filtered.velocity_x
            filtered_velocity_y = filtered.velocity_y
            filtered_speed = filtered.speed
            motion_filter_state = filtered.motion_state
            motion_filter_reason = filtered.reason
            motion_confirmation_frames = filtered.confirmation_frames
        else:
            filtered_velocity_x = 0.0
            filtered_velocity_y = 0.0
            filtered_speed = 0.0
            motion_filter_state = INSUFFICIENT_HISTORY
            motion_filter_reason = INSUFFICIENT_HISTORY
            motion_confirmation_frames = 0

        rows.append(
            PerObjectValidation(
                track_id=obj.track_id,
                class_name=obj.class_name,
                raw_velocity_x=motion.raw_velocity_x,
                raw_velocity_y=motion.raw_velocity_y,
                compensated_velocity_x=motion.compensated_velocity_x,
                compensated_velocity_y=motion.compensated_velocity_y,
                raw_speed=raw_speed,
                compensated_speed=motion.compensated_speed,
                compensated_direction=motion.compensated_direction,
                camera_confidence=camera_confidence,
                status=status,
                resolved_velocity_x=resolved_velocity_x,
                resolved_velocity_y=resolved_velocity_y,
                resolved_speed=resolved_speed,
                motion_source=motion_source,
                prediction_uncertain=prediction_uncertain,
                raw_intersects_corridor=raw_intersects_corridor,
                resolved_intersects_corridor=resolved_intersects_corridor,
                filtered_velocity_x=filtered_velocity_x,
                filtered_velocity_y=filtered_velocity_y,
                filtered_speed=filtered_speed,
                motion_filter_state=motion_filter_state,
                motion_filter_reason=motion_filter_reason,
                motion_confirmation_frames=motion_confirmation_frames,
            )
        )
    return rows


def build_validation_stats(
    per_object: list[PerObjectValidation],
    camera_motion: MotionEstimate | None,
) -> ValidationStats:
    """Aggregate ValidationStats from a list of PerObjectValidation rows.

    average_raw_speed/average_compensated_speed are 0.0 with no active
    tracks (nothing to average). reduction_percent is 0.0 whenever
    average_raw_speed is ~0 -- deliberately never divides by (near) zero.
    """
    active_track_count = len(per_object)

    stationary_count = sum(1 for row in per_object if row.status == STATIONARY)
    moving_count = sum(1 for row in per_object if row.status == OBJECT_MOVING)
    camera_dominated_count = sum(
        1 for row in per_object if row.status == CAMERA_DOMINATED
    )
    uncertain_count = sum(1 for row in per_object if row.status == UNCERTAIN)

    if active_track_count > 0:
        average_raw_speed = (
            sum(row.raw_speed for row in per_object) / active_track_count
        )
        average_compensated_speed = (
            sum(row.compensated_speed for row in per_object) / active_track_count
        )
    else:
        average_raw_speed = 0.0
        average_compensated_speed = 0.0

    if average_raw_speed > _REDUCTION_EPSILON:
        reduction_percent = (
            100.0
            * (average_raw_speed - average_compensated_speed)
            / average_raw_speed
        )
    else:
        reduction_percent = 0.0

    return ValidationStats(
        active_track_count=active_track_count,
        camera_dx=camera_motion.dx if camera_motion is not None else 0.0,
        camera_dy=camera_motion.dy if camera_motion is not None else 0.0,
        camera_confidence=camera_motion.confidence if camera_motion is not None else 0.0,
        camera_status=camera_motion.status if camera_motion is not None else "UNAVAILABLE",
        feature_count=camera_motion.feature_count if camera_motion is not None else 0,
        match_count=camera_motion.tracked_feature_count if camera_motion is not None else 0,
        inlier_count=camera_motion.inlier_count if camera_motion is not None else 0,
        stationary_count=stationary_count,
        moving_count=moving_count,
        camera_dominated_count=camera_dominated_count,
        uncertain_count=uncertain_count,
        average_raw_speed=average_raw_speed,
        average_compensated_speed=average_compensated_speed,
        reduction_percent=reduction_percent,
    )


def should_log_validation_summary(frame_count: int, interval: int = 30) -> bool:
    """Whether a periodic --validate-compensation summary should be logged
    this frame -- every `interval` frames, never on frame 0, so a summary
    doesn't fire before any real data has accumulated."""
    return frame_count > 0 and frame_count % interval == 0
