"""Generic camera-motion estimator interface (Atlas Phase 4).

Downstream code (main.py, future risk/warning logic) should depend only on
this abstraction and the MotionEstimate result type, not on any specific
estimator implementation -- so a visual (optical-flow) estimator, an
IMU-based estimator, or a visual+IMU fusion estimator can be swapped in
without changing any caller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.models import BoundingBox, MotionEstimate


class MotionEstimator(ABC):
    """Estimates 2D image-plane camera motion between consecutive frames."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal temporal state.

        Implementations must ensure that after reset() returns, the next
        call to estimate() behaves as if this estimator were newly
        constructed (e.g. no leftover temporal-smoothing history, no
        leftover invalid-frame counters). Callers must invoke this when
        the video source changes or restarts, so state from one source
        never leaks into estimates for another.
        """
        ...

    @abstractmethod
    def estimate(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        excluded_regions: list[BoundingBox] | None = None,
    ) -> MotionEstimate:
        """Estimate global camera/background motion between two frames.

        Args:
            previous_frame: The earlier frame (BGR or grayscale), as
                returned by VideoSource.read_frame(). Callers own frame
                lifetime/caching; this method does not retain a reference
                to it or assume it stays valid afterward.
            current_frame: The frame immediately following
                previous_frame, same dimensions.
            excluded_regions: Optional foreground bounding boxes (e.g. from
                object tracking) to exclude from background feature
                detection, so a moving person/vehicle doesn't dominate the
                camera-motion estimate. None or an empty list disables
                exclusion for this call.

        Returns:
            A MotionEstimate. Implementations must never raise on
            malformed/insufficient input -- they must return a
            MotionEstimate with valid=False and a populated
            reason_invalid instead.
        """
        ...
