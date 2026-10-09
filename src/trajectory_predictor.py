"""2D image-space trajectory estimation.

Pure math, no OpenCV/drawing calls and no dependency on a live camera or
YOLO model, so it can be fully unit tested with synthetic position
histories. Estimates are in image pixels only -- they carry no real-world
distance, speed, or collision meaning (there is no depth or GPS input).

Smoothing method: ordinary least-squares linear regression of x(t) and
y(t) over the most recent observations (t = index within the window),
rather than averaging consecutive frame-to-frame deltas. Averaging
consecutive deltas telescopes to (newest - oldest) / (n - 1), which is
mathematically identical to a plain two-point slope and uses none of the
in-between observations. Least-squares regression weights every point in
the window, so per-frame jitter is genuinely reduced, while remaining a
short, standard, easily-understood closed-form calculation.
"""

from __future__ import annotations

import math

from src.models import TrajectoryPrediction

UNKNOWN = "unknown"
STATIONARY = "stationary"


class TrajectoryPredictor:
    """Estimates smoothed 2D motion and predicts a future image position.

    Args:
        min_observations: Minimum number of position samples required
            before a trajectory is computed at all. Must be >= 2.
        prediction_horizon_frames: How many frames ahead to project the
            constant-velocity prediction. Must be > 0.
        stationary_threshold_px_per_frame: Speed (pixels/frame) below
            which the object is classified "stationary" and no long
            prediction arrow should be drawn.
        max_missed_frames: A track whose frames_since_seen exceeds this
            value yields an invalid ("unknown") prediction rather than
            extrapolating from stale data.
        history_window: Maximum number of most-recent position samples
            used for the regression fit, independent of how much history
            the caller passes in. Must be >= min_observations.
    """

    def __init__(
        self,
        min_observations: int,
        prediction_horizon_frames: int,
        stationary_threshold_px_per_frame: float,
        max_missed_frames: int,
        history_window: int,
    ) -> None:
        if min_observations < 2:
            raise ValueError(
                f"min_observations must be >= 2, got {min_observations}"
            )
        if prediction_horizon_frames <= 0:
            raise ValueError(
                "prediction_horizon_frames must be positive, got "
                f"{prediction_horizon_frames}"
            )
        if stationary_threshold_px_per_frame < 0:
            raise ValueError(
                "stationary_threshold_px_per_frame must be >= 0, got "
                f"{stationary_threshold_px_per_frame}"
            )
        if max_missed_frames < 0:
            raise ValueError(
                f"max_missed_frames must be >= 0, got {max_missed_frames}"
            )
        if history_window < min_observations:
            raise ValueError(
                "history_window must be >= min_observations, got "
                f"history_window={history_window}, "
                f"min_observations={min_observations}"
            )

        self._min_observations = min_observations
        self._prediction_horizon_frames = prediction_horizon_frames
        self._stationary_threshold_px_per_frame = stationary_threshold_px_per_frame
        self._max_missed_frames = max_missed_frames
        self._history_window = history_window

    def predict(
        self,
        position_history: tuple[tuple[float, float], ...],
        frames_since_seen: int,
        frame_width: int,
        frame_height: int,
    ) -> TrajectoryPrediction:
        """Estimate motion and predict a future image-space center.

        Args:
            position_history: Oldest-to-newest (x, y) centers for a track,
                as maintained by ObjectTracker.
            frames_since_seen: Consecutive frames since this track was
                last matched to a detection.
            frame_width: Current frame width, in pixels (for clamping).
            frame_height: Current frame height, in pixels (for clamping).

        Returns:
            A TrajectoryPrediction. If the data is insufficient, stale, or
            malformed, or the computed result isn't finite, `valid` is
            False, direction is "unknown", velocity/speed are 0.0, and
            predicted_center equals current_center.
        """
        window = tuple(position_history)[-self._history_window :]
        current_center = self._safe_last_point(window)

        if frames_since_seen > self._max_missed_frames:
            return self._invalid_result(current_center, len(window))
        if len(window) < self._min_observations:
            return self._invalid_result(current_center, len(window))
        if not all(self._is_finite_point(point) for point in window):
            return self._invalid_result(current_center, len(window))

        velocity_x, velocity_y = self._fit_velocity(window)
        if not (math.isfinite(velocity_x) and math.isfinite(velocity_y)):
            return self._invalid_result(current_center, len(window))

        speed = math.hypot(velocity_x, velocity_y)

        current_x, current_y = window[-1]
        predicted_x = current_x + velocity_x * self._prediction_horizon_frames
        predicted_y = current_y + velocity_y * self._prediction_horizon_frames
        if not (math.isfinite(predicted_x) and math.isfinite(predicted_y)):
            return self._invalid_result(current_center, len(window))

        predicted_center = (
            self._clamp(round(predicted_x), frame_width),
            self._clamp(round(predicted_y), frame_height),
        )

        direction = self._classify_direction(velocity_x, velocity_y, speed)

        return TrajectoryPrediction(
            valid=True,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            speed_px_per_frame=speed,
            direction=direction,
            current_center=(int(current_x), int(current_y)),
            predicted_center=predicted_center,
            observations_used=len(window),
            prediction_horizon_frames=self._prediction_horizon_frames,
            uncertain=False,
        )

    def predict_from_resolved_motion(
        self,
        current_center: tuple[int, int],
        velocity_x: float,
        velocity_y: float,
        observations_used: int,
        frame_width: int,
        frame_height: int,
        uncertain: bool,
    ) -> TrajectoryPrediction:
        """Build a TrajectoryPrediction from an explicitly-given velocity,
        instead of fitting one from raw position history.

        Used for the camera-motion-compensation-aware ("resolved")
        trajectory: the caller (see src/motion/motion_resolver.py) has
        already decided which velocity to trust this frame, so this
        method receives it explicitly rather than silently pulling a
        velocity field from a track itself -- this keeps the resolution
        decision centralized in one place instead of scattered here.

        Args:
            current_center: The object's actual current (x, y) center.
            velocity_x: The resolved horizontal velocity (pixels/frame).
            velocity_y: The resolved vertical velocity.
            observations_used: Informational only (e.g. position history
                length) -- carried through to the result, not used in the
                computation below.
            frame_width: Current frame width, in pixels (for clamping).
            frame_height: Current frame height, in pixels (for clamping).
            uncertain: Whether this resolved motion should be treated as
                not confident (see ResolvedMotion.uncertain) -- carried
                straight through to the result, except when the
                computation itself is invalid (non-finite), which is
                always uncertain regardless of what was passed in.

        Returns:
            A TrajectoryPrediction. If velocity_x/velocity_y (or the
            resulting predicted position) aren't finite, `valid` is False
            and `uncertain` is True, matching the raw-history invalid path.
        """
        if not (math.isfinite(velocity_x) and math.isfinite(velocity_y)):
            return self._invalid_result(current_center, observations_used)

        speed = math.hypot(velocity_x, velocity_y)

        predicted_x = current_center[0] + velocity_x * self._prediction_horizon_frames
        predicted_y = current_center[1] + velocity_y * self._prediction_horizon_frames
        if not (math.isfinite(predicted_x) and math.isfinite(predicted_y)):
            return self._invalid_result(current_center, observations_used)

        predicted_center = (
            self._clamp(round(predicted_x), frame_width),
            self._clamp(round(predicted_y), frame_height),
        )

        direction = self._classify_direction(velocity_x, velocity_y, speed)

        return TrajectoryPrediction(
            valid=True,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            speed_px_per_frame=speed,
            direction=direction,
            current_center=current_center,
            predicted_center=predicted_center,
            observations_used=observations_used,
            prediction_horizon_frames=self._prediction_horizon_frames,
            uncertain=uncertain,
        )

    def _fit_velocity(
        self, window: tuple[tuple[float, float], ...]
    ) -> tuple[float, float]:
        """Least-squares slope of x(t) and y(t) over the window's indices."""
        n = len(window)
        t_mean = (n - 1) / 2
        denominator = sum((t - t_mean) ** 2 for t in range(n))

        x_mean = sum(point[0] for point in window) / n
        y_mean = sum(point[1] for point in window) / n

        numerator_x = sum(
            (t - t_mean) * (point[0] - x_mean) for t, point in enumerate(window)
        )
        numerator_y = sum(
            (t - t_mean) * (point[1] - y_mean) for t, point in enumerate(window)
        )

        return numerator_x / denominator, numerator_y / denominator

    def _classify_direction(
        self, velocity_x: float, velocity_y: float, speed: float
    ) -> str:
        """Classify 2D image-space direction from a velocity vector.

        Below stationary_threshold_px_per_frame the object is classified
        "stationary" regardless of its (noisy) velocity sign. Otherwise,
        the label is a sign-based 8-way compass classification (screen
        y increases downward).
        """
        if speed < self._stationary_threshold_px_per_frame:
            return STATIONARY

        horizontal = ""
        if velocity_x > 0:
            horizontal = "right"
        elif velocity_x < 0:
            horizontal = "left"

        vertical = ""
        if velocity_y > 0:
            vertical = "down"
        elif velocity_y < 0:
            vertical = "up"

        if vertical and horizontal:
            vertical_word = "upper" if vertical == "up" else "lower"
            return f"{vertical_word}-{horizontal}"
        if vertical:
            return "upward" if vertical == "up" else "downward"
        if horizontal:
            return horizontal
        return STATIONARY

    @staticmethod
    def _is_finite_point(point) -> bool:
        try:
            x, y = point
        except (TypeError, ValueError):
            return False
        return (
            isinstance(x, (int, float))
            and isinstance(y, (int, float))
            and math.isfinite(x)
            and math.isfinite(y)
            and not isinstance(x, bool)
            and not isinstance(y, bool)
        )

    @staticmethod
    def _safe_last_point(
        window: tuple[tuple[float, float], ...]
    ) -> tuple[int, int]:
        if window and TrajectoryPredictor._is_finite_point(window[-1]):
            x, y = window[-1]
            return (int(x), int(y))
        return (0, 0)

    @staticmethod
    def _clamp(value: int, dimension: int) -> int:
        return max(0, min(value, dimension - 1))

    def _invalid_result(
        self, current_center: tuple[int, int], observations_used: int
    ) -> TrajectoryPrediction:
        return TrajectoryPrediction(
            valid=False,
            velocity_x=0.0,
            velocity_y=0.0,
            speed_px_per_frame=0.0,
            direction=UNKNOWN,
            current_center=current_center,
            predicted_center=current_center,
            observations_used=observations_used,
            prediction_horizon_frames=self._prediction_horizon_frames,
            uncertain=True,
        )
