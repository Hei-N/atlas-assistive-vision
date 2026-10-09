"""Image-space pedestrian-corridor / path-intersection analysis.

Pure geometry, no OpenCV/drawing calls and no dependency on a live camera
or YOLO model, so it can be fully unit tested with synthetic
TrajectoryPrediction objects. This is an image-space warning precursor
only -- it carries no real-world distance, depth, or collision-certainty
meaning.
"""

from __future__ import annotations

import math

from src.models import PathIntersectionResult, TrajectoryPrediction

INVALID_REASON = "trajectory prediction invalid; intersection not evaluated"
NO_INTERSECTION_REASON = "segment does not intersect corridor"


class PathIntersectionAnalyzer:
    """Builds a pedestrian-corridor polygon and tests predicted paths
    against it.

    Args:
        corridor_bottom_left_x: Left edge x, as a fraction of frame width,
            at the corridor's bottom (nearest the camera wearer).
        corridor_bottom_right_x: Right edge x, as a fraction of frame
            width, at the corridor's bottom.
        corridor_bottom_y: The corridor's bottom edge y, as a fraction of
            frame height.
        corridor_top_left_x: Left edge x, as a fraction of frame width, at
            the corridor's top (toward the horizon).
        corridor_top_right_x: Right edge x, as a fraction of frame width,
            at the corridor's top.
        corridor_top_y: The corridor's top edge y, as a fraction of frame
            height. Must be less than corridor_bottom_y (the top edge is
            higher on screen than the bottom edge).
    """

    def __init__(
        self,
        corridor_bottom_left_x: float,
        corridor_bottom_right_x: float,
        corridor_bottom_y: float,
        corridor_top_left_x: float,
        corridor_top_right_x: float,
        corridor_top_y: float,
    ) -> None:
        for name, value in (
            ("corridor_bottom_left_x", corridor_bottom_left_x),
            ("corridor_bottom_right_x", corridor_bottom_right_x),
            ("corridor_bottom_y", corridor_bottom_y),
            ("corridor_top_left_x", corridor_top_left_x),
            ("corridor_top_right_x", corridor_top_right_x),
            ("corridor_top_y", corridor_top_y),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be within [0, 1], got {value}")

        if corridor_bottom_left_x >= corridor_bottom_right_x:
            raise ValueError(
                "corridor_bottom_left_x must be < corridor_bottom_right_x, "
                f"got left={corridor_bottom_left_x}, "
                f"right={corridor_bottom_right_x}"
            )
        if corridor_top_left_x >= corridor_top_right_x:
            raise ValueError(
                "corridor_top_left_x must be < corridor_top_right_x, got "
                f"left={corridor_top_left_x}, right={corridor_top_right_x}"
            )
        if corridor_top_y >= corridor_bottom_y:
            raise ValueError(
                "corridor_top_y must be < corridor_bottom_y (the top edge "
                f"is higher on screen), got top={corridor_top_y}, "
                f"bottom={corridor_bottom_y}"
            )

        self._bottom_left_x = corridor_bottom_left_x
        self._bottom_right_x = corridor_bottom_right_x
        self._bottom_y = corridor_bottom_y
        self._top_left_x = corridor_top_left_x
        self._top_right_x = corridor_top_right_x
        self._top_y = corridor_top_y

    def build_corridor_polygon(
        self, frame_width: int, frame_height: int
    ) -> tuple[tuple[int, int], ...]:
        """Return the corridor's 4 pixel-space vertices for this frame.

        Args:
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            (bottom_left, bottom_right, top_right, top_left) in pixel
            coordinates, scaled to the given frame dimensions.
        """
        bottom_left = (
            round(self._bottom_left_x * frame_width),
            round(self._bottom_y * frame_height),
        )
        bottom_right = (
            round(self._bottom_right_x * frame_width),
            round(self._bottom_y * frame_height),
        )
        top_right = (
            round(self._top_right_x * frame_width),
            round(self._top_y * frame_height),
        )
        top_left = (
            round(self._top_left_x * frame_width),
            round(self._top_y * frame_height),
        )
        return (bottom_left, bottom_right, top_right, top_left)

    def analyze(
        self,
        track_id: int,
        prediction: TrajectoryPrediction,
        frame_width: int,
        frame_height: int,
    ) -> PathIntersectionResult:
        """Test a track's predicted path segment against the corridor.

        Args:
            track_id: The tracked object this analysis is for.
            prediction: The object's TrajectoryPrediction.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            A PathIntersectionResult. If `prediction.valid` is False, the
            result is `valid=False` with a reason explaining that
            intersection could not be evaluated -- never a false "outside
            the corridor" (safe-looking) result.
        """
        if not prediction.valid:
            return PathIntersectionResult(
                valid=False,
                intersects=False,
                starts_inside=False,
                ends_inside=False,
                intersection_point=None,
                track_id=track_id,
                reason=INVALID_REASON,
                uncertain=True,
            )

        polygon = self.build_corridor_polygon(frame_width, frame_height)
        start = prediction.current_center
        end = prediction.predicted_center

        starts_inside = self._point_in_polygon(start, polygon)
        ends_inside = self._point_in_polygon(end, polygon)

        intersection_point = None
        closest_distance = None
        for i in range(len(polygon)):
            edge_start = polygon[i]
            edge_end = polygon[(i + 1) % len(polygon)]
            point = self._segment_intersection(start, end, edge_start, edge_end)
            if point is None:
                continue
            distance = math.dist(start, point)
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance
                intersection_point = (round(point[0]), round(point[1]))

        crosses_edge = intersection_point is not None
        intersects = starts_inside or ends_inside or crosses_edge

        if not intersects:
            reason = NO_INTERSECTION_REASON
        elif starts_inside and ends_inside:
            reason = "segment stays within corridor"
        elif starts_inside:
            reason = "segment starts inside corridor and exits"
        elif ends_inside:
            reason = "segment enters corridor from outside"
        else:
            reason = "segment crosses corridor boundary"

        return PathIntersectionResult(
            valid=True,
            intersects=intersects,
            starts_inside=starts_inside,
            ends_inside=ends_inside,
            intersection_point=intersection_point,
            track_id=track_id,
            reason=reason,
            uncertain=prediction.uncertain,
        )

    @staticmethod
    def _point_in_polygon(
        point: tuple[float, float], polygon: tuple[tuple[int, int], ...]
    ) -> bool:
        """Even-odd ray-casting point-in-polygon test."""
        x, y = point
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_intersect:
                    inside = not inside
        return inside

    @staticmethod
    def _segment_intersection(
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        p4: tuple[float, float],
    ) -> tuple[float, float] | None:
        """Intersection point of segments p1-p2 and p3-p4, or None.

        Standard parametric line-segment intersection: solves for t and u
        such that p1 + t*(p2-p1) == p3 + u*(p4-p3). A real crossing exists
        only when both t and u fall within [0, 1].
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None  # Parallel or collinear -- no distinct crossing point.

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denominator

        if 0 <= t <= 1 and 0 <= u <= 1:
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return None
