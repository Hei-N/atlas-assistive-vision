"""Stationary-motion filtering layer for Atlas.

Sits between ResolvedMotion (src/motion/motion_resolver.py) and
TrajectoryPredictor.predict_from_resolved_motion. Root cause this exists
to fix: the resolved-motion chain used a raw, single-frame velocity with
zero smoothing/dead-zone/temporal-confirmation anywhere in it, so
bounding-box jitter or residual camera-compensation error could flip a
genuinely parked vehicle's reported motion state on a single noisy frame.

Pipeline, per tracked object, per frame:
    1. Minimum-history gate -- too few samples yet -> INSUFFICIENT_HISTORY.
    2. Exponential smoothing of the resolved velocity (same
       alpha*raw + (1-alpha)*previous convention as
       VisualMotionEstimator._apply_smoothing).
    3. Dead-zone + hysteresis + temporal confirmation: a speed must stay
       on one side of its threshold for several consecutive frames before
       the CONFIRMED state (STATIONARY/MOVING) actually flips. A single
       frame, or a value strictly between the two thresholds, never
       flips anything.
    4. Filtered velocity is (0.0, 0.0) whenever the confirmed state is
       not MOVING -- so a stationary object generates no meaningful
       future trajectory.
    5. Source-aware handling: COMPENSATED_LOW_CONFIDENCE and RAW_FALLBACK
       always keep uncertain=True (unchanged from ResolvedMotion's own
       contract); RAW_FALLBACK additionally never asserts a confident
       MOVING claim when suppress_raw_motion_during_camera_motion is on.

Pure logic, no OpenCV/drawing calls, no dependency on a live camera or
YOLO model -- fully unit testable with synthetic ResolvedMotion objects.
"""

from __future__ import annotations

import math

from src.models import FilteredMotion, ResolvedMotion, TrackedObject
from src.motion.motion_resolver import COMPENSATED_LOW_CONFIDENCE, RAW_FALLBACK

STATIONARY = "STATIONARY"
MOVING = "MOVING"
UNCERTAIN = "UNCERTAIN"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"

LOW_COMPENSATION_CONFIDENCE = "LOW_COMPENSATION_CONFIDENCE"


class _TrackFilterState:
    """Mutable per-track state carried across frames. Not a frozen model
    -- this is internal bookkeeping, never exposed outside this module."""

    __slots__ = (
        "confirmed_state",
        "candidate_state",
        "candidate_streak",
        "smoothed_velocity_x",
        "smoothed_velocity_y",
    )

    def __init__(self) -> None:
        self.confirmed_state = STATIONARY
        self.candidate_state: str | None = None
        self.candidate_streak = 0
        self.smoothed_velocity_x: float | None = None
        self.smoothed_velocity_y: float | None = None


class MotionStateFilter:
    """Stateful dead-zone/hysteresis/temporal-confirmation motion filter.

    Args:
        stationary_enter_speed: At/below this smoothed speed
            (pixels/frame), a frame counts as a STATIONARY candidate.
            Also acts as the "dead zone" -- filtered velocity is zeroed
            whenever the confirmed state is STATIONARY. Must be >= 0.
        moving_enter_speed: At/above this smoothed speed (pixels/frame),
            a frame counts as a MOVING candidate. Must be >=
            stationary_enter_speed (the hysteresis gap between the two
            prevents rapid alternation; a speed strictly between them
            counts as neither and never disturbs the confirmed state).
        moving_confirmation_frames: Consecutive MOVING-candidate frames
            required before the confirmed state flips from STATIONARY to
            MOVING. Must be >= 1.
        stationary_confirmation_frames: Consecutive STATIONARY-candidate
            frames required before the confirmed state flips from MOVING
            back to STATIONARY. Must be >= 1.
        minimum_history_samples: Minimum position-history samples
            required before motion is considered reliable at all; below
            this, INSUFFICIENT_HISTORY is reported instead of a
            confident classification. Must be >= 1.
        smoothing_alpha: Exponential-smoothing weight on the newest
            resolved-velocity sample, applied before dead-zone/hysteresis
            -- 1.0 means no smoothing (raw passthrough), closer to 0.0
            means heavier smoothing. Must be within (0, 1].
        suppress_raw_motion_during_camera_motion: When True, a MOVING
            reading whose source is RAW_FALLBACK is downgraded to
            UNCERTAIN (velocity zeroed) rather than shown as a confident
            moving claim -- raw fallback velocity can't be trusted to
            distinguish object motion from unmeasured camera motion. A
            genuinely near-zero RAW_FALLBACK reading is still reported
            STATIONARY (not a risky claim).
    """

    def __init__(
        self,
        stationary_enter_speed: float,
        moving_enter_speed: float,
        moving_confirmation_frames: int,
        stationary_confirmation_frames: int,
        minimum_history_samples: int,
        smoothing_alpha: float,
        suppress_raw_motion_during_camera_motion: bool,
    ) -> None:
        if stationary_enter_speed < 0:
            raise ValueError(
                "stationary_enter_speed must be >= 0, got "
                f"{stationary_enter_speed}"
            )
        if moving_enter_speed < stationary_enter_speed:
            raise ValueError(
                "moving_enter_speed must be >= stationary_enter_speed, got "
                f"moving_enter_speed={moving_enter_speed}, "
                f"stationary_enter_speed={stationary_enter_speed}"
            )
        if moving_confirmation_frames < 1:
            raise ValueError(
                "moving_confirmation_frames must be >= 1, got "
                f"{moving_confirmation_frames}"
            )
        if stationary_confirmation_frames < 1:
            raise ValueError(
                "stationary_confirmation_frames must be >= 1, got "
                f"{stationary_confirmation_frames}"
            )
        if minimum_history_samples < 1:
            raise ValueError(
                "minimum_history_samples must be >= 1, got "
                f"{minimum_history_samples}"
            )
        if not (0.0 < smoothing_alpha <= 1.0):
            raise ValueError(
                f"smoothing_alpha must be within (0, 1], got {smoothing_alpha}"
            )

        self._stationary_enter_speed = stationary_enter_speed
        self._moving_enter_speed = moving_enter_speed
        self._moving_confirmation_frames = moving_confirmation_frames
        self._stationary_confirmation_frames = stationary_confirmation_frames
        self._minimum_history_samples = minimum_history_samples
        self._smoothing_alpha = smoothing_alpha
        self._suppress_raw_motion_during_camera_motion = (
            suppress_raw_motion_during_camera_motion
        )
        self._states: dict[int, _TrackFilterState] = {}

    @property
    def stationary_enter_speed(self) -> float:
        """The configured dead-zone/stationary-candidate threshold
        (pixels/frame) -- exposed read-only for callers (e.g. the
        detector benchmark script) that want to compare behavior with
        and without this filter using the same threshold."""
        return self._stationary_enter_speed

    def filter(
        self,
        track_id: int,
        resolved: ResolvedMotion,
        observations_used: int,
    ) -> FilteredMotion:
        """Filter one track's ResolvedMotion for this frame.

        Args:
            track_id: The tracked object this decision is for.
            resolved: This track's ResolvedMotion for the current frame.
            observations_used: Number of position-history samples
                available for this track (e.g. len(position_history)).

        Returns:
            A FilteredMotion. See module docstring for the exact pipeline.
        """
        state = self._states.setdefault(track_id, _TrackFilterState())

        if observations_used < self._minimum_history_samples:
            return FilteredMotion(
                track_id=track_id,
                velocity_x=0.0,
                velocity_y=0.0,
                speed=0.0,
                motion_state=INSUFFICIENT_HISTORY,
                source=resolved.source,
                uncertain=True,
                reason=INSUFFICIENT_HISTORY,
                confirmation_frames=0,
            )

        smoothed_vx, smoothed_vy = self._smooth(state, resolved)
        smoothed_speed = math.hypot(smoothed_vx, smoothed_vy)

        instantaneous = self._classify_instantaneous(smoothed_speed)
        self._update_confirmed_state(state, instantaneous)

        confirmed_state = state.confirmed_state
        confirmation_frames = state.candidate_streak

        if confirmed_state == STATIONARY:
            filtered_vx, filtered_vy, filtered_speed = 0.0, 0.0, 0.0
        else:
            filtered_vx, filtered_vy, filtered_speed = (
                smoothed_vx, smoothed_vy, smoothed_speed
            )

        motion_state = confirmed_state
        uncertain = resolved.uncertain
        reason = None

        if resolved.source == RAW_FALLBACK:
            reason = RAW_FALLBACK
            if (
                self._suppress_raw_motion_during_camera_motion
                and confirmed_state == MOVING
            ):
                motion_state = UNCERTAIN
                filtered_vx = filtered_vy = filtered_speed = 0.0
        elif resolved.source == COMPENSATED_LOW_CONFIDENCE:
            reason = LOW_COMPENSATION_CONFIDENCE

        return FilteredMotion(
            track_id=track_id,
            velocity_x=filtered_vx,
            velocity_y=filtered_vy,
            speed=filtered_speed,
            motion_state=motion_state,
            source=resolved.source,
            uncertain=uncertain,
            reason=reason if uncertain else None,
            confirmation_frames=confirmation_frames,
        )

    def filter_for_tracks(
        self,
        tracked_objects: list[TrackedObject],
        resolved_motions: dict[int, ResolvedMotion],
    ) -> dict[int, FilteredMotion]:
        """Filter every tracked object in one call.

        Prunes internal per-track state for track_ids no longer present
        in tracked_objects, so a dropped track's state doesn't linger
        forever. A track missing from resolved_motions (should not
        happen -- both are built from the same tracked_objects list) is
        skipped rather than raising, matching this codebase's established
        "never crash on a missing dict entry" convention.
        """
        active_ids = {obj.track_id for obj in tracked_objects}
        for stale_id in set(self._states) - active_ids:
            del self._states[stale_id]

        filtered: dict[int, FilteredMotion] = {}
        for obj in tracked_objects:
            resolved = resolved_motions.get(obj.track_id)
            if resolved is None:
                continue
            filtered[obj.track_id] = self.filter(
                obj.track_id, resolved, len(obj.position_history)
            )
        return filtered

    def _smooth(
        self, state: _TrackFilterState, resolved: ResolvedMotion
    ) -> tuple[float, float]:
        """Exponential smoothing: smoothed = alpha*raw + (1-alpha)*previous.

        The first estimate for a track passes raw values through
        unchanged (matching VisualMotionEstimator._apply_smoothing's own
        convention).
        """
        if state.smoothed_velocity_x is None:
            smoothed_vx, smoothed_vy = resolved.velocity_x, resolved.velocity_y
        else:
            alpha = self._smoothing_alpha
            smoothed_vx = (
                alpha * resolved.velocity_x + (1 - alpha) * state.smoothed_velocity_x
            )
            smoothed_vy = (
                alpha * resolved.velocity_y + (1 - alpha) * state.smoothed_velocity_y
            )

        state.smoothed_velocity_x = smoothed_vx
        state.smoothed_velocity_y = smoothed_vy
        return smoothed_vx, smoothed_vy

    def _classify_instantaneous(self, smoothed_speed: float) -> str | None:
        """This frame's raw candidate reading, or None if it falls in the
        hysteresis dead band (strictly between the two thresholds) --
        ambiguous readings never nudge the confirmed state."""
        if smoothed_speed >= self._moving_enter_speed:
            return MOVING
        if smoothed_speed <= self._stationary_enter_speed:
            return STATIONARY
        return None

    def _update_confirmed_state(
        self, state: _TrackFilterState, instantaneous: str | None
    ) -> None:
        """Advance (or reset) the pending-transition streak, flipping the
        confirmed state only once the streak reaches the relevant
        confirmation-frame count."""
        if instantaneous is None or instantaneous == state.confirmed_state:
            state.candidate_state = None
            state.candidate_streak = 0
            return

        if state.candidate_state == instantaneous:
            state.candidate_streak += 1
        else:
            state.candidate_state = instantaneous
            state.candidate_streak = 1

        required = (
            self._moving_confirmation_frames
            if instantaneous == MOVING
            else self._stationary_confirmation_frames
        )
        if state.candidate_streak >= required:
            state.confirmed_state = instantaneous
            state.candidate_state = None
            state.candidate_streak = 0
