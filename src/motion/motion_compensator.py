"""Camera-motion compensation for tracked-object velocity (Atlas Phase 4).

Separates a tracked object's *apparent* image-space velocity into two
parts: motion caused by the camera/wearer's head turning, and motion
caused by the object actually moving in the world. It does this by
subtracting the camera's estimated background motion (MotionEstimate, from
a MotionEstimator such as VisualMotionEstimator) from each object's own
raw frame-to-frame pixel velocity (derived here from TrackedObject's
existing position_history -- no new per-object velocity field exists
upstream).

This module is standalone: it does not import ObjectTracker, cv2, or
main.py, and nothing in the live pipeline calls it yet.

Sign convention (verified against the current project, not assumed):
    - Positive x is rightward, positive y is downward, throughout this
      codebase (see BoundingBox and MotionEstimate in src/models.py, and
      ObjectTracker._compute_direction's dx/dy sign checks).
    - VisualMotionEstimator's dx/dy report the apparent 2D image-space
      displacement of the BACKGROUND/scene between two frames (its module
      docstring: "the 2D image-plane motion of the BACKGROUND... how much
      the camera/wearer's head appears to have moved, based on how
      stationary scenery... shifted in the image") -- not a literal
      real-world "camera displacement" and not an inverse/reciprocal
      quantity. A real-world-stationary object is, by definition, part of
      that background: its own raw pixel velocity (computed the same way,
      from its tracked center's frame-to-frame pixel shift) is therefore
      measured in the exact same convention and should numerically match
      the camera's dx/dy for a stationary object.
    - Consequently NO sign inversion is required: compensated_dx =
      raw_object_dx - camera_dx, exactly as given, with both terms already
      in the same coordinate frame. This directly matches the required
      behavior: a stationary object with raw motion (+10, 0) alongside
      camera motion (+10, 0) must compensate to (~0, ~0), which only holds
      if both quantities carry the same sign for the same apparent
      direction of pixel travel -- confirmed by VisualMotionEstimator's
      own tests (tests/test_visual_motion_estimator.py), which show a
      background shifted +10px right is reported as dx=+10, not -10.
"""

from __future__ import annotations

import math

from src.models import CompensatedMotion, MotionEstimate, TrackedObject

STATIONARY = "stationary"


class MotionCompensator:
    """Subtracts estimated camera motion from each tracked object's raw
    frame-to-frame pixel velocity.

    Args:
        min_camera_confidence: Minimum MotionEstimate.confidence required
            to trust and apply camera-motion compensation, in [0.0, 1.0].
            Camera motion below this (or None, or invalid) triggers a safe
            fallback to each object's raw (uncompensated) velocity.
        stationary_threshold_px: Maximum compensated per-axis velocity
            (pixels/frame) still classified "stationary" in
            compensated_direction -- mirrors ObjectTracker's own
            stationary_threshold_px convention.
    """

    def __init__(
        self,
        min_camera_confidence: float,
        stationary_threshold_px: float,
    ) -> None:
        if not (0.0 <= min_camera_confidence <= 1.0):
            raise ValueError(
                "min_camera_confidence must be within [0, 1], got "
                f"{min_camera_confidence}"
            )
        if stationary_threshold_px < 0:
            raise ValueError(
                "stationary_threshold_px must be >= 0, got "
                f"{stationary_threshold_px}"
            )

        self._min_camera_confidence = min_camera_confidence
        self._stationary_threshold_px = stationary_threshold_px

    def compensate(
        self,
        tracks: list[TrackedObject],
        camera_motion: MotionEstimate | None,
    ) -> dict[int, CompensatedMotion]:
        """Compute camera-motion-compensated velocity for each track.

        Args:
            tracks: Tracked objects, as produced by ObjectTracker.update().
                Not mutated -- TrackedObject is a frozen dataclass, and
                this method only reads position_history from it.
            camera_motion: The current MotionEstimate (e.g. from
                VisualMotionEstimator.estimate()), or None if motion
                estimation is disabled/unavailable this frame. When None,
                invalid, or below min_camera_confidence, every track
                safely falls back to its raw (uncompensated) velocity
                rather than raising.

        Returns:
            One CompensatedMotion per input track, keyed by track_id
            (matching the dict[track_id, ...] pattern main.py already uses
            for TrajectoryPrediction/PathIntersectionResult).
        """
        camera_dx, camera_dy, camera_motion_applied = self._resolve_camera_motion(
            camera_motion
        )

        results: dict[int, CompensatedMotion] = {}
        for track in tracks:
            raw_dx, raw_dy = self._raw_velocity(track)

            if camera_motion_applied:
                compensated_dx = raw_dx - camera_dx
                compensated_dy = raw_dy - camera_dy
            else:
                compensated_dx = raw_dx
                compensated_dy = raw_dy

            results[track.track_id] = CompensatedMotion(
                track_id=track.track_id,
                compensated_velocity_x=compensated_dx,
                compensated_velocity_y=compensated_dy,
                compensated_speed=math.hypot(compensated_dx, compensated_dy),
                compensated_direction=self._classify_direction(
                    compensated_dx, compensated_dy
                ),
                raw_velocity_x=raw_dx,
                raw_velocity_y=raw_dy,
                camera_motion_applied=camera_motion_applied,
            )

        return results

    def _resolve_camera_motion(
        self, camera_motion: MotionEstimate | None
    ) -> tuple[float, float, bool]:
        """Decide whether camera motion is trustworthy enough to apply.

        Returns (camera_dx, camera_dy, applied). applied is False -- with
        camera_dx/camera_dy irrelevant zeros -- when camera_motion is
        None, invalid, or below min_camera_confidence: the three "missing
        or unreliable" cases this module must fall back safely for.
        """
        if camera_motion is None:
            return 0.0, 0.0, False
        if not camera_motion.valid:
            return 0.0, 0.0, False
        if camera_motion.confidence < self._min_camera_confidence:
            return 0.0, 0.0, False
        return camera_motion.dx, camera_motion.dy, True

    @staticmethod
    def _raw_velocity(track: TrackedObject) -> tuple[float, float]:
        """The object's own most recent frame-to-frame pixel velocity.

        TrackedObject carries no precomputed per-frame velocity field, so
        this is derived from the last two position_history samples (the
        object's raw pixel movement, matching camera_motion's own
        single-step dx/dy convention). Fewer than two samples (a
        brand-new track) safely defaults to (0.0, 0.0) -- "no motion
        detected yet" -- mirroring ObjectTracker._compute_direction's own
        "insufficient history" default of "stationary".
        """
        history = track.position_history
        if len(history) < 2:
            return 0.0, 0.0

        prev_x, prev_y = history[-2]
        curr_x, curr_y = history[-1]
        return float(curr_x - prev_x), float(curr_y - prev_y)

    def _classify_direction(self, dx: float, dy: float) -> str:
        """Sign-based direction classification from compensated velocity.

        Same threshold-per-axis rule and vocabulary as
        ObjectTracker._compute_direction (left/right/up/down/diagonal/
        stationary), applied here to a compensated (dx, dy) pair instead
        of a raw position-history deque.
        """
        horizontal = ""
        if dx > self._stationary_threshold_px:
            horizontal = "right"
        elif dx < -self._stationary_threshold_px:
            horizontal = "left"

        vertical = ""
        if dy > self._stationary_threshold_px:
            vertical = "down"
        elif dy < -self._stationary_threshold_px:
            vertical = "up"

        if not horizontal and not vertical:
            return STATIONARY
        if horizontal and vertical:
            return f"{vertical}-{horizontal}"
        return horizontal or vertical
