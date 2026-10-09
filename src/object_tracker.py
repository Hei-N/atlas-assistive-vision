"""Cross-frame object tracking and motion analysis for Atlas Phase 2.

Pure geometric/bookkeeping logic with no OpenCV drawing calls and no
dependency on a live camera or YOLO model, so it can be fully unit tested
with synthetic Detection objects.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from src.models import BoundingBox, Detection, TrackedObject

STATIONARY = "stationary"
APPROACHING = "approaching"
MOVING_AWAY = "moving away"


@dataclass
class _Track:
    """Internal mutable per-object tracking state."""

    track_id: int
    class_id: int
    positions: deque
    sizes: deque
    last_bbox: BoundingBox
    frames_since_seen: int = 0


class ObjectTracker:
    """Assigns stable IDs to detections across frames and derives motion.

    A detection is matched to an existing track of the same class if
    *either* its centroid is within max_match_distance_px of the track's
    last known position, *or* its box overlaps the track's last known box
    by at least min_match_iou (IoU). Requiring only one of the two signals
    keeps tracking robust to camera resolution: centroid distance alone is
    an absolute-pixel threshold that gets too tight on high-resolution
    frames, while IoU is resolution-independent and reconnects a
    slow-moving or momentarily-jittery box even when its centroid drifts
    past the distance threshold.

    Args:
        max_match_distance_px: Maximum centroid distance (pixels) allowed
            when matching a detection to an existing track of the same
            class.
        max_disappeared_frames: Number of consecutive frames a track may go
            unmatched before it is dropped. A detection that reappears
            after this many frames is treated as a new object. Kept
            generous enough to bridge brief misses (occlusion, a
            confidence dip near the detector's threshold) without
            reassigning a new ID.
        min_match_iou: Minimum bounding-box IoU (intersection over union,
            0.0-1.0 exclusive of 0) allowed when matching a detection to an
            existing track of the same class.
        history_length: Maximum number of (position, size) samples retained
            per track.
        stationary_threshold_px: Maximum oldest-to-newest centroid movement
            (pixels, per axis) still considered "stationary".
        size_change_threshold_fraction: Minimum fractional change in
            bounding-box area (oldest vs newest) required to report
            "approaching"/"moving away" instead of "stationary".
    """

    def __init__(
        self,
        max_match_distance_px: float,
        max_disappeared_frames: int,
        min_match_iou: float,
        history_length: int,
        stationary_threshold_px: float,
        size_change_threshold_fraction: float,
    ) -> None:
        if max_match_distance_px <= 0:
            raise ValueError(
                "max_match_distance_px must be positive, got "
                f"{max_match_distance_px}"
            )
        if max_disappeared_frames < 0:
            raise ValueError(
                "max_disappeared_frames must be >= 0, got "
                f"{max_disappeared_frames}"
            )
        if not (0.0 < min_match_iou <= 1.0):
            raise ValueError(
                f"min_match_iou must be within (0, 1], got {min_match_iou}"
            )
        if history_length < 2:
            raise ValueError(f"history_length must be >= 2, got {history_length}")

        self._max_match_distance_px = max_match_distance_px
        self._max_disappeared_frames = max_disappeared_frames
        self._min_match_iou = min_match_iou
        self._history_length = history_length
        self._stationary_threshold_px = stationary_threshold_px
        self._size_change_threshold_fraction = size_change_threshold_fraction
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """Match detections to existing tracks and return TrackedObjects.

        Args:
            detections: This frame's Detections, as produced by
                ObjectDetector.detect().

        Returns:
            One TrackedObject per input detection (same order), each
            carrying a stable track_id and derived motion state. Tracks
            that go unmatched this frame are aged and, once past
            max_disappeared_frames, dropped entirely (not returned).
        """
        assignment = self._match(detections)

        matched_track_ids = set(assignment.values())
        for track_id, track in self._tracks.items():
            if track_id not in matched_track_ids:
                track.frames_since_seen += 1
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.frames_since_seen <= self._max_disappeared_frames
        }

        tracked_objects: list[TrackedObject] = []
        for index, detection in enumerate(detections):
            track_id = assignment.get(index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(
                    track_id=track_id,
                    class_id=detection.class_id,
                    positions=deque(maxlen=self._history_length),
                    sizes=deque(maxlen=self._history_length),
                    last_bbox=detection.bbox,
                )

            track = self._tracks[track_id]
            track.frames_since_seen = 0
            track.positions.append(detection.center)
            track.sizes.append((detection.bbox.width, detection.bbox.height))
            track.last_bbox = detection.bbox

            tracked_objects.append(
                TrackedObject(
                    track_id=track_id,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    center=detection.center,
                    region=detection.region,
                    position_history=tuple(track.positions),
                    size_history=tuple(track.sizes),
                    direction=self._compute_direction(track.positions),
                    motion_status=self._compute_motion_status(track.sizes),
                    frames_since_seen=track.frames_since_seen,
                )
            )

        return tracked_objects

    def _match(self, detections: list[Detection]) -> dict[int, int]:
        """Greedily match detection indices to existing track IDs.

        Only detection/track pairs sharing the same class_id, and passing
        either the centroid-distance or the IoU test, are eligible. Pairs
        are accepted highest-IoU-first (distance as a tiebreaker), and each
        detection and each track is used in at most one pair.
        """
        candidates = []
        for det_index, detection in enumerate(detections):
            for track_id, track in self._tracks.items():
                if track.class_id != detection.class_id:
                    continue
                distance = math.dist(detection.center, track.positions[-1])
                iou = self._iou(detection.bbox, track.last_bbox)
                if distance > self._max_match_distance_px and iou < self._min_match_iou:
                    continue
                candidates.append((iou, distance, det_index, track_id))

        candidates.sort(key=lambda item: (-item[0], item[1]))

        assignment: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _iou, _distance, det_index, track_id in candidates:
            if det_index in assignment or track_id in used_tracks:
                continue
            assignment[det_index] = track_id
            used_tracks.add(track_id)

        return assignment

    @staticmethod
    def _iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
        """Intersection-over-union of two axis-aligned bounding boxes."""
        inter_x1 = max(box_a.x1, box_b.x1)
        inter_y1 = max(box_a.y1, box_b.y1)
        inter_x2 = min(box_a.x2, box_b.x2)
        inter_y2 = min(box_a.y2, box_b.y2)
        intersection = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        if intersection <= 0:
            return 0.0

        union = (
            box_a.width * box_a.height
            + box_b.width * box_b.height
            - intersection
        )
        if union <= 0:
            return 0.0
        return intersection / union

    def _compute_direction(self, positions: deque) -> str:
        """Derive movement direction from oldest-vs-newest centroid."""
        if len(positions) < 2:
            return STATIONARY

        oldest_x, oldest_y = positions[0]
        newest_x, newest_y = positions[-1]
        dx = newest_x - oldest_x
        dy = newest_y - oldest_y

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

    def _compute_motion_status(self, sizes: deque) -> str:
        """Derive approaching/moving-away/stationary from bbox area change."""
        if len(sizes) < 2:
            return STATIONARY

        oldest_w, oldest_h = sizes[0]
        newest_w, newest_h = sizes[-1]
        oldest_area = oldest_w * oldest_h
        if oldest_area <= 0:
            return STATIONARY

        newest_area = newest_w * newest_h
        change_fraction = (newest_area - oldest_area) / oldest_area
        if change_fraction > self._size_change_threshold_fraction:
            return APPROACHING
        if change_fraction < -self._size_change_threshold_fraction:
            return MOVING_AWAY
        return STATIONARY
