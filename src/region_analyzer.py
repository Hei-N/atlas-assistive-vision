"""Region classification and attention-zone geometry for Atlas Phase 1.

This module contains pure geometric logic with no OpenCV drawing calls and
no dependency on a live camera or model, so it can be fully unit tested.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models import BoundingBox

LEFT = "left"
CENTER = "center"
RIGHT = "right"


@dataclass(frozen=True)
class AttentionZone:
    """Pixel-space rectangle representing the central attention zone.

    Attributes:
        x_min: Left edge x-coordinate (pixels).
        y_min: Top edge y-coordinate (pixels).
        x_max: Right edge x-coordinate (pixels).
        y_max: Bottom edge y-coordinate (pixels).
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int


class RegionAnalyzer:
    """Classifies points into left/center/right regions and manages the
    attention zone.

    Args:
        left_boundary: Fraction of frame width (0.0-1.0) below which a
            point is classified as "left".
        right_boundary: Fraction of frame width (0.0-1.0) above which a
            point is classified as "right".
        attention_zone_fractions: Tuple of
            (x_min_fraction, x_max_fraction, y_min_fraction, y_max_fraction)
            defining the attention zone as fractions of frame dimensions.
    """

    def __init__(
        self,
        left_boundary: float,
        right_boundary: float,
        attention_zone_fractions: tuple[float, float, float, float],
    ) -> None:
        if not (0.0 <= left_boundary < right_boundary <= 1.0):
            raise ValueError(
                "left_boundary must be < right_boundary, both within [0, 1]: "
                f"got left={left_boundary}, right={right_boundary}"
            )
        self._left_boundary = left_boundary
        self._right_boundary = right_boundary
        (
            self._x_min_frac,
            self._x_max_frac,
            self._y_min_frac,
            self._y_max_frac,
        ) = attention_zone_fractions

    def classify_point(self, x: int, frame_width: int) -> str:
        """Classify an x-coordinate into left/center/right.

        Args:
            x: The x-coordinate of the point, in pixels.
            frame_width: The total width of the frame, in pixels.

        Returns:
            One of "left", "center", "right".

        Raises:
            ValueError: If frame_width is not positive.
        """
        if frame_width <= 0:
            raise ValueError(f"frame_width must be positive, got {frame_width}")

        fraction = x / frame_width
        if fraction < self._left_boundary:
            return LEFT
        if fraction > self._right_boundary:
            return RIGHT
        return CENTER

    def get_region_boundaries_px(self, frame_width: int) -> tuple[int, int]:
        """Return the (left_boundary_px, right_boundary_px) divider lines.

        Args:
            frame_width: The total width of the frame, in pixels.

        Returns:
            A tuple of pixel x-coordinates for the left and right dividers.
        """
        left_px = int(self._left_boundary * frame_width)
        right_px = int(self._right_boundary * frame_width)
        return left_px, right_px

    def compute_attention_zone(
        self, frame_width: int, frame_height: int
    ) -> AttentionZone:
        """Compute the attention zone rectangle in pixel coordinates.

        Args:
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            An AttentionZone with pixel-space coordinates.

        Raises:
            ValueError: If frame_width or frame_height is not positive.
        """
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(
                "frame_width and frame_height must be positive, got "
                f"({frame_width}, {frame_height})"
            )
        return AttentionZone(
            x_min=int(self._x_min_frac * frame_width),
            y_min=int(self._y_min_frac * frame_height),
            x_max=int(self._x_max_frac * frame_width),
            y_max=int(self._y_max_frac * frame_height),
        )

    @staticmethod
    def point_in_zone(x: int, y: int, zone: AttentionZone) -> bool:
        """Return True if the point (x, y) falls within the attention zone."""
        return zone.x_min <= x <= zone.x_max and zone.y_min <= y <= zone.y_max

    @staticmethod
    def box_overlaps_zone(box: BoundingBox, zone: AttentionZone) -> bool:
        """Return True if the bounding box overlaps the attention zone.

        Uses standard axis-aligned rectangle overlap test.
        """
        no_overlap = (
            box.x2 < zone.x_min
            or box.x1 > zone.x_max
            or box.y2 < zone.y_min
            or box.y1 > zone.y_max
        )
        return not no_overlap
