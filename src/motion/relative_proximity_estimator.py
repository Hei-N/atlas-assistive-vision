"""Perspective-aware relative proximity zones for Atlas's audio hazard
system.

Classifies how far down the frame a tracked object's bounding-box
bottom edge falls, as a heuristic proxy for "how close to the camera
wearer" -- objects lower in frame are closer, assuming a roughly flat
ground plane and a forward-facing camera (the same assumption the
pedestrian corridor itself already makes). This is a relative,
image-space classification only -- FAR/MID/NEAR are NOT physical
distances (no depth, stereo, LIDAR, or GPS input exists anywhere in
this pipeline) and must never be presented as meters or any other
real-world unit.

Deliberately does NOT use bounding-box size/area (a "do not classify
proximity using bounding-box size alone" requirement) -- scale is a
separate, less reliable signal (a genuinely larger vehicle class, or
partial occlusion, changes apparent size without changing position)
reserved for src/motion/approach_estimator.py's SCALE_GROWTH cue.

Pure math, no OpenCV/drawing calls, no dependency on a live camera or
YOLO model -- fully unit testable with synthetic bounding boxes.
"""

from __future__ import annotations

import math

from src.models import TrackedObject

FAR = "FAR"
MID = "MID"
NEAR = "NEAR"
UNKNOWN = "UNKNOWN"

# Ordinal ranking for "is this zone nearer than that one" comparisons
# (e.g. ApproachEstimator's PROXIMITY_INCREASING cue). UNKNOWN has no
# defined ordinal position -- never treated as nearer or farther than
# anything, matching this codebase's "never guess from missing data"
# convention.
_ZONE_RANK = {FAR: 0, MID: 1, NEAR: 2}


def is_nearer(zone_a: str, zone_b: str) -> bool:
    """True iff zone_a is strictly ordinally nearer than zone_b (e.g.
    is_nearer(NEAR, FAR) is True). False if either zone is UNKNOWN."""
    if zone_a not in _ZONE_RANK or zone_b not in _ZONE_RANK:
        return False
    return _ZONE_RANK[zone_a] > _ZONE_RANK[zone_b]


class RelativeProximityEstimator:
    """Classifies FAR/MID/NEAR from a bounding box's bottom edge.

    Args:
        far_boundary: y-fraction of frame height (0.0-1.0) at/below
            which an object is classified FAR. Defaults to 0.55 to match
            corridor.corridor_top_y's own default, keeping proximity
            perspective-consistent with the existing pedestrian
            corridor. Must be < near_boundary.
        near_boundary: y-fraction of frame height (0.0-1.0) at/above
            which an object is classified NEAR. Defaults to 0.80.
            Between far_boundary and near_boundary is MID.
    """

    def __init__(self, far_boundary: float, near_boundary: float) -> None:
        if not (0.0 <= far_boundary < near_boundary <= 1.0):
            raise ValueError(
                "far_boundary must be < near_boundary, both within [0, 1]: "
                f"got far={far_boundary}, near={near_boundary}"
            )
        self._far_boundary = far_boundary
        self._near_boundary = near_boundary

    def estimate(self, bbox_bottom_y: float, frame_height: int) -> str:
        """Classify a single bounding-box bottom edge.

        Args:
            bbox_bottom_y: The object's bounding-box bottom edge
                y-coordinate, in pixels (e.g. TrackedObject.bbox.y2).
            frame_height: The frame's height, in pixels.

        Returns:
            One of FAR, MID, NEAR, or UNKNOWN (frame_height <= 0, or
            bbox_bottom_y is non-finite -- never a crash, never a guess).
        """
        if frame_height <= 0:
            return UNKNOWN
        if not (isinstance(bbox_bottom_y, (int, float)) and math.isfinite(bbox_bottom_y)):
            return UNKNOWN

        fraction = bbox_bottom_y / frame_height
        if fraction <= self._far_boundary:
            return FAR
        if fraction >= self._near_boundary:
            return NEAR
        return MID

    def estimate_for_object(self, obj: TrackedObject, frame_height: int) -> str:
        """Convenience wrapper: classify a TrackedObject directly from
        its current frame's bbox.y2."""
        return self.estimate(obj.bbox.y2, frame_height)

    def estimate_for_tracks(
        self, tracked_objects: list[TrackedObject], frame_height: int
    ) -> dict[int, str]:
        """Classify every tracked object in one call."""
        return {
            obj.track_id: self.estimate_for_object(obj, frame_height)
            for obj in tracked_objects
        }
