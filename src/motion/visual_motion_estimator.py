"""Sparse-optical-flow camera-motion estimator (Atlas Phase 4).

Estimates the 2D image-plane motion of the BACKGROUND between two
consecutive frames -- i.e. how much the camera/wearer's head appears to
have moved, based on how stationary scenery (corners, edges, textured
surfaces) shifted in the image. This is the "camera motion" half of
separating camera motion from independent object motion; it does not
apply that separation itself (no trajectory correction happens here).

Pipeline: detect strong corner features in the previous frame (optionally
avoiding foreground object boxes) -> track them into the current frame
with pyramidal Lucas-Kanade optical flow -> fit a single global 2D
similarity transform (translation + rotation + uniform scale) to those
correspondences with RANSAC, which separates the majority "background"
motion from outlier tracks (moving people/vehicles, mistracked points) --
-> decode dx/dy/rotation/scale from that transform -> smooth over time.

Why RANSAC instead of averaging every flow vector: a plain average is
corrupted by any large or numerous outlier motion (a walking person, a
passing car, a mistracked point) pulling the "average" away from the
background's actual motion. RANSAC repeatedly fits a transform from a
small random subset of correspondences and keeps whichever fit the most
points agree with (the "inliers") -- so a minority of independently-moving
points don't affect the result at all, as long as the background still
forms the majority.

Coordinate/sign conventions (see MotionEstimate in src/models.py for the
full field reference):
    - dx > 0: background moved right. dy > 0: background moved down.
      These are read directly off the fitted transform's translation with
      no sign flip -- empirically verified: shifting a synthetic frame's
      content by (+10px right, +5px down) is recovered as dx=+9.97,
      dy=+4.98 (see tests/test_visual_motion_estimator.py).
    - rotation_degrees: empirically verified to be the OPPOSITE sign of
      cv2.getRotationMatrix2D's own `angle` parameter (which OpenCV
      documents as counter-clockwise-positive). Rotating a synthetic
      frame's content by cv2.getRotationMatrix2D(center, +5.0, 1.0)
      (counter-clockwise by OpenCV's convention) is recovered by this
      estimator as rotation_degrees ~= -5.0. Equivalently: a POSITIVE
      rotation_degrees here means the background content rotated
      CLOCKWISE on screen; NEGATIVE means counter-clockwise.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.models import BoundingBox, MotionEstimate
from src.motion.motion_estimator import MotionEstimator

STATUS_VALID = "VALID"
STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE"
STATUS_UNAVAILABLE = "UNAVAILABLE"


class VisualMotionEstimator(MotionEstimator):
    """Estimates camera motion from consecutive frames via sparse optical
    flow and a RANSAC-fit global 2D similarity transform.

    Args:
        max_features: Maximum number of corner features to detect per
            frame (cv2.goodFeaturesToTrack maxCorners).
        quality_level: Minimum accepted corner quality, relative to the
            best corner found, in (0, 1) (cv2.goodFeaturesToTrack
            qualityLevel). Lower values accept weaker corners.
        min_distance_px: Minimum pixel distance enforced between detected
            features (cv2.goodFeaturesToTrack minDistance).
        block_size: Neighborhood size used for corner-quality computation
            (cv2.goodFeaturesToTrack blockSize).
        lk_window_size: Search window side length (pixels) for pyramidal
            Lucas-Kanade optical flow (cv2.calcOpticalFlowPyrLK winSize).
        lk_max_level: Maximum pyramid level for optical flow (0 = no
            pyramid, single-scale).
        ransac_threshold_px: Maximum reprojection error (pixels) for a
            correspondence to be considered a RANSAC inlier
            (cv2.estimateAffinePartial2D ransacReprojThreshold).
        min_tracked_features: Minimum successfully-tracked features
            required before attempting a transform fit; below this, the
            estimate is invalid.
        min_inliers: Minimum RANSAC inlier count required to accept the
            fitted transform; below this, the estimate is invalid. Must be
            <= min_tracked_features.
        exclude_foreground: Whether to exclude excluded_regions from
            feature detection when estimate() is called with them.
        exclusion_padding_px: Extra padding (pixels) added around each
            excluded region before masking it out.
        smoothing_enabled: Whether to exponentially smooth dx/dy/rotation
            over time.
        smoothing_alpha: Exponential-smoothing weight on the newest raw
            estimate, in (0, 1]. `smoothed = alpha*raw + (1-alpha)*previous_smoothed`.
            Higher alpha tracks raw values more closely (less smoothing);
            lower alpha smooths more heavily (slower to react).
        max_consecutive_invalid_frames: After this many consecutive
            invalid estimates, smoothing state is cleared so a stale
            smoothed value doesn't linger indefinitely once valid
            estimates resume.
        low_confidence_threshold: Below this confidence (but still
            `valid`), the returned estimate's `status` is
            "LOW_CONFIDENCE" rather than "VALID" -- a labeling threshold
            only; it does not reject the estimate (see
            MotionCompensator.min_camera_confidence for the separate,
            independently-configured threshold that decides whether
            compensation actually *trusts* an estimate enough to apply
            it).
        max_translation_px: Reject the estimate ("implausibly large
            frame-to-frame transformation") if the raw (pre-smoothing)
            translation magnitude, hypot(raw_dx, raw_dy), exceeds this.
            A heuristic upper bound on how far the background could
            plausibly shift between two consecutive frames -- not a
            scientifically derived limit.
        max_rotation_degrees: Reject the estimate (same reason) if
            abs(raw_rotation_degrees) exceeds this. Also a heuristic
            bound, not scientifically derived.
    """

    def __init__(
        self,
        max_features: int,
        quality_level: float,
        min_distance_px: float,
        block_size: int,
        lk_window_size: int,
        lk_max_level: int,
        ransac_threshold_px: float,
        min_tracked_features: int,
        min_inliers: int,
        exclude_foreground: bool,
        exclusion_padding_px: int,
        smoothing_enabled: bool,
        smoothing_alpha: float,
        max_consecutive_invalid_frames: int,
        low_confidence_threshold: float,
        max_translation_px: float,
        max_rotation_degrees: float,
    ) -> None:
        if max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {max_features}")
        if not (0.0 < quality_level < 1.0):
            raise ValueError(
                f"quality_level must be within (0, 1), got {quality_level}"
            )
        if min_distance_px <= 0:
            raise ValueError(
                f"min_distance_px must be positive, got {min_distance_px}"
            )
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if lk_window_size < 3:
            raise ValueError(f"lk_window_size must be >= 3, got {lk_window_size}")
        if lk_max_level < 0:
            raise ValueError(f"lk_max_level must be >= 0, got {lk_max_level}")
        if ransac_threshold_px <= 0:
            raise ValueError(
                f"ransac_threshold_px must be positive, got {ransac_threshold_px}"
            )
        if min_tracked_features < 1:
            raise ValueError(
                f"min_tracked_features must be >= 1, got {min_tracked_features}"
            )
        if min_inliers < 1:
            raise ValueError(f"min_inliers must be >= 1, got {min_inliers}")
        if min_inliers > min_tracked_features:
            raise ValueError(
                "min_inliers must be <= min_tracked_features, got "
                f"min_inliers={min_inliers}, "
                f"min_tracked_features={min_tracked_features}"
            )
        if exclusion_padding_px < 0:
            raise ValueError(
                f"exclusion_padding_px must be >= 0, got {exclusion_padding_px}"
            )
        if not (0.0 < smoothing_alpha <= 1.0):
            raise ValueError(
                f"smoothing_alpha must be within (0, 1], got {smoothing_alpha}"
            )
        if max_consecutive_invalid_frames < 1:
            raise ValueError(
                "max_consecutive_invalid_frames must be >= 1, got "
                f"{max_consecutive_invalid_frames}"
            )
        if not (0.0 <= low_confidence_threshold <= 1.0):
            raise ValueError(
                "low_confidence_threshold must be within [0, 1], got "
                f"{low_confidence_threshold}"
            )
        if max_translation_px <= 0:
            raise ValueError(
                f"max_translation_px must be positive, got {max_translation_px}"
            )
        if max_rotation_degrees <= 0:
            raise ValueError(
                "max_rotation_degrees must be positive, got "
                f"{max_rotation_degrees}"
            )

        self._max_features = max_features
        self._quality_level = quality_level
        self._min_distance_px = min_distance_px
        self._block_size = block_size
        self._lk_window_size = lk_window_size
        self._lk_max_level = lk_max_level
        self._ransac_threshold_px = ransac_threshold_px
        self._min_tracked_features = min_tracked_features
        self._min_inliers = min_inliers
        self._exclude_foreground = exclude_foreground
        self._exclusion_padding_px = exclusion_padding_px
        self._smoothing_enabled = smoothing_enabled
        self._smoothing_alpha = smoothing_alpha
        self._max_consecutive_invalid_frames = max_consecutive_invalid_frames
        self._low_confidence_threshold = low_confidence_threshold
        self._max_translation_px = max_translation_px
        self._max_rotation_degrees = max_rotation_degrees

        self._smoothed_dx: float | None = None
        self._smoothed_dy: float | None = None
        self._smoothed_rotation: float | None = None
        self._consecutive_invalid_count = 0
        self._last_frame_shape: tuple[int, int] | None = None

    def reset(self) -> None:
        """Clear all internal temporal state (smoothing history, the
        consecutive-invalid-frame counter, and the last-seen frame shape).
        """
        self._clear_smoothing_state()
        self._consecutive_invalid_count = 0
        self._last_frame_shape = None

    def estimate(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        excluded_regions: list[BoundingBox] | None = None,
    ) -> MotionEstimate:
        """Estimate global camera/background motion between two frames.

        See MotionEstimator.estimate for the general contract. Never
        raises on malformed/insufficient input -- returns valid=False with
        reason_invalid instead.
        """
        if previous_frame is None or current_frame is None:
            return self._invalid("previous or current frame is None")
        if previous_frame.size == 0 or current_frame.size == 0:
            return self._invalid("previous or current frame is empty")
        if previous_frame.shape[:2] != current_frame.shape[:2]:
            return self._invalid(
                "previous and current frame have different dimensions"
            )

        current_shape = current_frame.shape[:2]
        if (
            self._last_frame_shape is not None
            and self._last_frame_shape != current_shape
        ):
            self._clear_smoothing_state()
        self._last_frame_shape = current_shape

        prev_gray = self._to_grayscale(previous_frame)
        curr_gray = self._to_grayscale(current_frame)
        mask = self._build_feature_mask(prev_gray.shape, excluded_regions)

        prev_points = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self._max_features,
            qualityLevel=self._quality_level,
            minDistance=self._min_distance_px,
            blockSize=self._block_size,
            mask=mask,
        )
        feature_count = 0 if prev_points is None else len(prev_points)
        if prev_points is None or feature_count < self._min_tracked_features:
            return self._invalid(
                "insufficient features detected", feature_count=feature_count
            )

        next_points, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_points,
            None,
            winSize=(self._lk_window_size, self._lk_window_size),
            maxLevel=self._lk_max_level,
        )
        if next_points is None or status is None:
            return self._invalid("optical flow tracking failed")

        status_mask = status.ravel() == 1
        good_prev = prev_points.reshape(-1, 2)[status_mask]
        good_next = next_points.reshape(-1, 2)[status_mask]
        tracked_feature_count = len(good_prev)
        debug_prev_points = self._to_point_tuples(good_prev)
        debug_current_points = self._to_point_tuples(good_next)

        if tracked_feature_count < self._min_tracked_features:
            return self._invalid(
                "insufficient successfully tracked features",
                feature_count=feature_count,
                tracked_feature_count=tracked_feature_count,
                prev_points=debug_prev_points,
                curr_points=debug_current_points,
            )

        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            good_prev,
            good_next,
            method=cv2.RANSAC,
            ransacReprojThreshold=self._ransac_threshold_px,
        )
        if matrix is None:
            return self._invalid(
                "transformation estimation failed",
                feature_count=feature_count,
                tracked_feature_count=tracked_feature_count,
                prev_points=debug_prev_points,
                curr_points=debug_current_points,
            )

        inlier_flags = tuple(bool(v) for v in inlier_mask.ravel())
        inlier_count = sum(inlier_flags)

        if inlier_count < self._min_inliers:
            return self._invalid(
                "insufficient RANSAC inliers",
                feature_count=feature_count,
                tracked_feature_count=tracked_feature_count,
                inlier_count=inlier_count,
                prev_points=debug_prev_points,
                curr_points=debug_current_points,
                inlier_flags=inlier_flags,
            )

        a, b, tx = float(matrix[0][0]), float(matrix[0][1]), float(matrix[0][2])
        c, d, ty = float(matrix[1][0]), float(matrix[1][1]), float(matrix[1][2])
        scale = math.hypot(a, c)
        raw_dx, raw_dy = tx, ty
        raw_rotation_degrees = math.degrees(math.atan2(c, a))
        inlier_ratio = inlier_count / tracked_feature_count

        common_diagnostics = dict(
            feature_count=feature_count,
            tracked_feature_count=tracked_feature_count,
            inlier_count=inlier_count,
            prev_points=debug_prev_points,
            curr_points=debug_current_points,
            inlier_flags=inlier_flags,
        )

        if not (
            math.isfinite(raw_dx)
            and math.isfinite(raw_dy)
            and math.isfinite(raw_rotation_degrees)
            and math.isfinite(scale)
        ):
            return self._invalid("non-finite transformation", **common_diagnostics)

        if (
            math.hypot(raw_dx, raw_dy) > self._max_translation_px
            or abs(raw_rotation_degrees) > self._max_rotation_degrees
        ):
            return self._invalid(
                "implausibly large frame-to-frame transformation",
                **common_diagnostics,
            )

        confidence = self._compute_confidence(tracked_feature_count, inlier_count)
        dx, dy, rotation_degrees = self._apply_smoothing(
            raw_dx, raw_dy, raw_rotation_degrees
        )

        self._consecutive_invalid_count = 0

        return MotionEstimate(
            valid=True,
            dx=dx,
            dy=dy,
            rotation_degrees=rotation_degrees,
            scale=scale,
            confidence=confidence,
            feature_count=feature_count,
            tracked_feature_count=tracked_feature_count,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            status=self._compute_status(valid=True, confidence=confidence),
            reason_invalid=None,
            raw_dx=raw_dx,
            raw_dy=raw_dy,
            raw_rotation_degrees=raw_rotation_degrees,
            transform_matrix=((a, b, tx), (c, d, ty)),
            debug_prev_points=debug_prev_points,
            debug_current_points=debug_current_points,
            debug_inlier_flags=inlier_flags,
        )

    def _compute_status(self, valid: bool, confidence: float) -> str:
        """VALID / LOW_CONFIDENCE / UNAVAILABLE from existing valid+confidence.

        UNAVAILABLE when not valid (see reason_invalid for why); otherwise
        LOW_CONFIDENCE below low_confidence_threshold, else VALID. This is
        a labeling threshold only -- it never rejects an estimate (that's
        what min_tracked_features/min_inliers/max_translation_px/
        max_rotation_degrees are for); MotionCompensator applies its own,
        separately-configured confidence gate before actually trusting an
        estimate enough to use it.
        """
        if not valid:
            return STATUS_UNAVAILABLE
        if confidence < self._low_confidence_threshold:
            return STATUS_LOW_CONFIDENCE
        return STATUS_VALID

    def _compute_confidence(
        self, tracked_feature_count: int, inlier_count: int
    ) -> float:
        """Heuristic confidence in [0.0, 1.0] -- not a calibrated
        probability. Mostly the RANSAC inlier ratio (how much of what we
        tracked agrees with the fitted global motion), plus a smaller
        bonus for having comfortably more than the bare-minimum feature
        count (more evidence backing the fit).
        """
        if tracked_feature_count == 0:
            return 0.0
        inlier_ratio = inlier_count / tracked_feature_count
        feature_sufficiency = min(
            1.0, tracked_feature_count / (2 * self._min_tracked_features)
        )
        confidence = 0.7 * inlier_ratio + 0.3 * feature_sufficiency
        return max(0.0, min(1.0, confidence))

    def _apply_smoothing(
        self, raw_dx: float, raw_dy: float, raw_rotation_degrees: float
    ) -> tuple[float, float, float]:
        """Exponential smoothing: smoothed = alpha*raw + (1-alpha)*previous.

        The first estimate since construction/reset (or whenever smoothing
        is disabled) passes raw values through unchanged.
        """
        if not self._smoothing_enabled or self._smoothed_dx is None:
            dx, dy, rotation = raw_dx, raw_dy, raw_rotation_degrees
        else:
            alpha = self._smoothing_alpha
            dx = alpha * raw_dx + (1 - alpha) * self._smoothed_dx
            dy = alpha * raw_dy + (1 - alpha) * self._smoothed_dy
            rotation = (
                alpha * raw_rotation_degrees + (1 - alpha) * self._smoothed_rotation
            )

        if self._smoothing_enabled:
            self._smoothed_dx = dx
            self._smoothed_dy = dy
            self._smoothed_rotation = rotation

        return dx, dy, rotation

    def _build_feature_mask(
        self,
        gray_shape: tuple[int, ...],
        excluded_regions: list[BoundingBox] | None,
    ) -> np.ndarray:
        """White-everywhere mask, except black (excluded) padded/clipped
        rectangles over each excluded_regions box, when foreground
        exclusion is enabled and regions were provided. A no-op (all
        white) otherwise -- safe for excluded_regions being None, empty,
        or containing a partially/fully out-of-frame box.
        """
        height, width = gray_shape[:2]
        mask = np.full((height, width), 255, dtype=np.uint8)
        if not self._exclude_foreground or not excluded_regions:
            return mask

        for box in excluded_regions:
            x1 = max(0, box.x1 - self._exclusion_padding_px)
            y1 = max(0, box.y1 - self._exclusion_padding_px)
            x2 = min(width, box.x2 + self._exclusion_padding_px)
            y2 = min(height, box.y2 + self._exclusion_padding_px)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, thickness=-1)
        return mask

    def _clear_smoothing_state(self) -> None:
        self._smoothed_dx = None
        self._smoothed_dy = None
        self._smoothed_rotation = None

    def _invalid(
        self,
        reason: str,
        feature_count: int = 0,
        tracked_feature_count: int = 0,
        inlier_count: int = 0,
        prev_points: tuple[tuple[int, int], ...] = (),
        curr_points: tuple[tuple[int, int], ...] = (),
        inlier_flags: tuple[bool, ...] = (),
    ) -> MotionEstimate:
        """Build an invalid MotionEstimate, tracking the consecutive-
        invalid-frame counter and clearing smoothing state once that
        counter reaches max_consecutive_invalid_frames.
        """
        self._consecutive_invalid_count += 1
        if self._consecutive_invalid_count >= self._max_consecutive_invalid_frames:
            self._clear_smoothing_state()

        inlier_ratio = inlier_count / tracked_feature_count if tracked_feature_count else 0.0

        return MotionEstimate(
            valid=False,
            dx=0.0,
            dy=0.0,
            rotation_degrees=0.0,
            scale=1.0,
            confidence=0.0,
            feature_count=feature_count,
            tracked_feature_count=tracked_feature_count,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            status=self._compute_status(valid=False, confidence=0.0),
            reason_invalid=reason,
            raw_dx=0.0,
            raw_dy=0.0,
            raw_rotation_degrees=0.0,
            transform_matrix=None,
            debug_prev_points=prev_points,
            debug_current_points=curr_points,
            debug_inlier_flags=inlier_flags,
        )

    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _to_point_tuples(points: np.ndarray) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(round(float(x))), int(round(float(y)))) for x, y in points
        )
