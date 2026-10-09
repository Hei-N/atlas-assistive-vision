"""Centralized resolved-motion selection stage for Atlas.

Chooses the single motion vector downstream trajectory prediction and
corridor analysis should use, based on the current frame's camera-motion
compensation status. This is the ONLY place that branches on
MotionEstimate.status for this purpose -- trajectory prediction and
corridor analysis stay source-agnostic (they just receive a velocity and
an `uncertain` flag), so status checks are not scattered across modules.

Decision:
    VALID           -> compensated velocity, uncertain=False
    LOW_CONFIDENCE   -> compensated velocity (still useful directionally),
                        uncertain=True
    UNAVAILABLE, or no CompensatedMotion available for this track at all
                     -> the raw trajectory's own regression-fit velocity
                        (already computed every debug frame), uncertain=True

Pure logic, no OpenCV/drawing calls, no dependency on a live camera or
YOLO model -- fully unit testable with synthetic CompensatedMotion/
MotionEstimate/TrajectoryPrediction objects.
"""

from __future__ import annotations

import math

from src.models import (
    CompensatedMotion,
    MotionEstimate,
    ResolvedMotion,
    TrackedObject,
    TrajectoryPrediction,
)

COMPENSATED = "COMPENSATED"
COMPENSATED_LOW_CONFIDENCE = "COMPENSATED_LOW_CONFIDENCE"
RAW_FALLBACK = "RAW_FALLBACK"


def resolve_motion(
    track_id: int,
    compensated: CompensatedMotion | None,
    camera_motion: MotionEstimate | None,
    raw_trajectory: TrajectoryPrediction,
) -> ResolvedMotion:
    """Select the motion vector to use for this track's trajectory prediction.

    Args:
        track_id: The tracked object this decision is for.
        compensated: This track's CompensatedMotion for the current frame,
            or None if compensation is disabled or not yet available for
            this track.
        camera_motion: The current frame's MotionEstimate, or None if
            camera-motion estimation is disabled or unavailable this
            frame. Treated the same as an UNAVAILABLE status.
        raw_trajectory: This track's raw TrajectoryPrediction (from
            TrajectoryPredictor.predict() over raw position history),
            used as the fallback velocity source when compensation isn't
            trustworthy. Safely reads (0.0, 0.0) when raw_trajectory
            itself is invalid.

    Returns:
        A ResolvedMotion. See module docstring for the exact decision.
    """
    status = camera_motion.status if camera_motion is not None else "UNAVAILABLE"

    if compensated is not None and status == "VALID":
        velocity_x = compensated.compensated_velocity_x
        velocity_y = compensated.compensated_velocity_y
        return ResolvedMotion(
            track_id=track_id,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            speed=math.hypot(velocity_x, velocity_y),
            source=COMPENSATED,
            uncertain=False,
        )

    if compensated is not None and status == "LOW_CONFIDENCE":
        velocity_x = compensated.compensated_velocity_x
        velocity_y = compensated.compensated_velocity_y
        return ResolvedMotion(
            track_id=track_id,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            speed=math.hypot(velocity_x, velocity_y),
            source=COMPENSATED_LOW_CONFIDENCE,
            uncertain=True,
        )

    # UNAVAILABLE, or compensation not available for this track at all.
    velocity_x = raw_trajectory.velocity_x
    velocity_y = raw_trajectory.velocity_y
    return ResolvedMotion(
        track_id=track_id,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        speed=math.hypot(velocity_x, velocity_y),
        source=RAW_FALLBACK,
        uncertain=True,
    )


def resolve_motions_for_tracks(
    tracked_objects: list[TrackedObject],
    compensated_motions: dict[int, CompensatedMotion],
    camera_motion: MotionEstimate | None,
    raw_trajectories: dict[int, TrajectoryPrediction],
) -> dict[int, ResolvedMotion]:
    """Resolve motion for every tracked object in one call.

    A track missing from raw_trajectories (should not happen -- both dicts
    are built from the same tracked_objects list) is skipped rather than
    raising, matching this codebase's established "never crash on a
    missing dict entry" convention.
    """
    resolved: dict[int, ResolvedMotion] = {}
    for obj in tracked_objects:
        raw_trajectory = raw_trajectories.get(obj.track_id)
        if raw_trajectory is None:
            continue
        resolved[obj.track_id] = resolve_motion(
            obj.track_id,
            compensated_motions.get(obj.track_id),
            camera_motion,
            raw_trajectory,
        )
    return resolved
