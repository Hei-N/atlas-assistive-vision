"""Drawing and on-screen display utilities for Atlas Phase 1.

The Visualizer is responsible only for rendering: boxes, labels, region
divider lines, the attention zone, and debug overlays. It contains no
detection or region-classification logic.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Callable

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from src.models import (
    BoundingBox,
    CompensatedMotion,
    FilteredMotion,
    MotionEstimate,
    PathIntersectionResult,
    PerObjectValidation,
    TrackedObject,
    TrajectoryPrediction,
    ValidationStats,
)
from src.region_analyzer import AttentionZone

_BOX_COLOR = (0, 255, 0)
_LABEL_COLOR = (0, 255, 0)
_REGION_LINE_COLOR = (255, 255, 0)
_ATTENTION_ZONE_COLOR = (0, 0, 255)
_DEBUG_TEXT_COLOR = (255, 255, 255)
_TRAJECTORY_COLOR = (0, 165, 255)
_TRAJECTORY_HISTORY_COLOR = (255, 0, 255)
_CORRIDOR_COLOR = (255, 128, 0)
_INTERSECTS_COLOR = (0, 0, 255)
_NO_INTERSECT_COLOR = (0, 200, 0)
_UNKNOWN_COLOR = (200, 200, 200)
_MOTION_INLIER_COLOR = (0, 255, 0)
_MOTION_OUTLIER_COLOR = (0, 0, 255)
_MOTION_ARROW_COLOR = (255, 0, 0)
_MOTION_TEXT_COLOR = (255, 255, 255)
_MOTION_INVALID_COLOR = (0, 0, 255)
_COMPENSATED_MOTION_COLOR = (0, 255, 255)
_RAW_MOTION_COLOR = (255, 191, 0)
_RESOLVED_TRAJECTORY_COLOR = (180, 0, 255)
_COMPENSATED_ARROW_SCALE = 3.0
_COMPENSATED_ARROW_MAX_LENGTH_PX = 80
_VALIDATION_PANEL_COLOR = (255, 255, 255)
_VALIDATION_PANEL_WIDTH_PX = 260
_LEGEND_ENTRIES = (
    ("raw motion", _RAW_MOTION_COLOR),
    ("compensated motion", _COMPENSATED_MOTION_COLOR),
    ("predicted trajectory", _TRAJECTORY_COLOR),
)
_VALIDATION_STATUS_COLORS = {
    "STATIONARY": (180, 180, 180),
    "CAMERA-DOMINATED": (255, 255, 0),
    "OBJECT-MOVING": (0, 0, 255),
    "UNCERTAIN": (128, 128, 128),
}
# Colors for the temporally-confirmed motion_filter_state. Distinct from
# _VALIDATION_STATUS_COLORS above (classify_status); this state drives
# filtered_velocity_x/y.
_MOTION_FILTER_STATE_COLORS = {
    "STATIONARY": (180, 180, 180),
    "MOVING": (0, 0, 255),
    "UNCERTAIN": (0, 255, 255),
    "INSUFFICIENT_HISTORY": (128, 128, 128),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Real-world ("field") display styling ---------------------------------
# Applies only to the three calls main.py makes unconditionally (no
# --debug/--validate-compensation needed): draw_tracked_objects,
# draw_region_lines, draw_attention_zone. Every other draw_* method above
# is developer/validation tooling and keeps its existing Hershey styling
# untouched, on purpose -- see the module docstring.
_FIELD_ACCENT_COLOR = (191, 212, 45)  # teal, BGR
_FIELD_CHIP_BG_COLOR = (39, 24, 17)  # near-black translucent, BGR
_FIELD_CHIP_ALPHA = 0.55
_FIELD_TEXT_COLOR = (249, 245, 241)  # off-white, BGR
_FIELD_FONT_SIZE_PRIMARY = 15
_FIELD_FONT_SIZE_SECONDARY = 12
_FIELD_CHIP_PADDING_PX = 4

_REGION_LINE_ALPHA = 0.45
_ATTENTION_ZONE_ALPHA = 0.35
_ATTENTION_ZONE_THICKNESS_PX = 1

_BRACKET_LENGTH_PX = 18
_BRACKET_MIN_LENGTH_PX = 4
_BRACKET_THICKNESS_PX = 2

# Ordered fallback list for anti-aliased field-display text (cv2.putText
# only supports blocky Hershey fonts). macOS system fonts first, then a
# couple of common Linux paths for portability (e.g. CI). Never assumed
# to exist -- _load_field_font() checks each with Path.exists() and falls
# back to PIL's built-in default font if none are present, so this can
# never raise regardless of platform.
_FONT_PATH_CANDIDATES = (
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


@functools.lru_cache(maxsize=None)
def _load_field_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a real (anti-aliased) font for the field display at `size`.

    Tries each candidate system font path in order; falls back to PIL's
    built-in default font if none exist or load successfully, so this
    never raises even on a platform with none of the candidate fonts.
    """
    for path in _FONT_PATH_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


class Visualizer:
    """Draws detections, region boundaries, and debug info onto frames."""

    def __init__(self, window_name: str) -> None:
        self._window_name = window_name

    def draw_tracked_objects(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        filtered_motions: dict[int, FilteredMotion],
    ) -> None:
        """Draw a real-world-facing box + label for each tracked object, in
        place -- this is what a user sees by default, with no --debug or
        --validate-compensation flag needed, so legibility against an
        arbitrary real-world background matters more than raw cv2.putText
        speed.

        Style: four corner brackets ("AR-style") instead of a full
        rectangle, plus a translucent chip behind each label line rendered
        in a real anti-aliased font (see _load_field_font) rather than
        cv2's blocky Hershey font.

        Motion-state label: a track present in `filtered_motions` (i.e.
        --debug is on, so the resolved+filtered pipeline actually ran --
        see main.py) shows the AUTHORITATIVE, temporally-confirmed,
        camera-motion-aware FilteredMotion.motion_state and velocity,
        colored via _MOTION_FILTER_STATE_COLORS -- never
        TrackedObject.motion_status/direction (raw, uncompensated pixel
        displacement) in that case, so a parked car during a camera pan
        can never be shown here as a confident MOVING claim while the
        --validate-compensation panel simultaneously shows STATIONARY.
        A track absent from filtered_motions (plain no-flag run, where
        this pipeline never executes) falls back to the raw tracker
        fields, but clearly prefixed "RAW"/"raw dir" and colored
        _RAW_MOTION_COLOR -- never presented as if it were an
        authoritative claim. TrackedObject.motion_status/direction
        themselves are untouched (still computed, still available
        internally) -- only what this method chooses to DISPLAY changes.
        """
        frame_height, frame_width = frame.shape[:2]
        pil_entries: list[tuple[tuple[int, int], str, tuple[int, int, int], int]] = []

        for obj in tracked_objects:
            box = obj.bbox
            self._draw_corner_brackets(
                frame, box, _FIELD_ACCENT_COLOR, _BRACKET_THICKNESS_PX
            )

            filtered = filtered_motions.get(obj.track_id)
            if filtered is not None:
                state_color = _MOTION_FILTER_STATE_COLORS.get(
                    filtered.motion_state, _FIELD_TEXT_COLOR
                )
                label = (
                    f"{obj.class_name} | ID {obj.track_id} | "
                    f"{filtered.motion_state} | {obj.region}"
                )
                direction_label = (
                    f"vel: ({filtered.velocity_x:.1f},{filtered.velocity_y:.1f})"
                )
            else:
                state_color = _RAW_MOTION_COLOR
                label = (
                    f"{obj.class_name} | ID {obj.track_id} | "
                    f"RAW:{obj.motion_status} | {obj.region}"
                )
                direction_label = f"raw dir: {obj.direction} (unconfirmed)"

            label_origin = (box.x1, max(box.y1 - 24, 4))
            direction_origin = (box.x1, min(box.y2 + 6, frame_height - 16))

            self._draw_text_chip(
                frame, label, label_origin, _FIELD_FONT_SIZE_PRIMARY,
                frame_width, frame_height,
            )
            self._draw_text_chip(
                frame, direction_label, direction_origin, _FIELD_FONT_SIZE_SECONDARY,
                frame_width, frame_height,
            )

            pil_entries.append(
                (label_origin, label, state_color, _FIELD_FONT_SIZE_PRIMARY)
            )
            pil_entries.append(
                (direction_origin, direction_label, state_color,
                 _FIELD_FONT_SIZE_SECONDARY)
            )

        # One batched PIL round trip per frame regardless of object count
        # -- see _draw_pil_text.
        self._draw_pil_text(frame, pil_entries)

    @staticmethod
    def _draw_corner_brackets(
        frame: np.ndarray,
        box: BoundingBox,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draw a four-corner-bracket box outline instead of a full
        rectangle -- a lighter, more product-like visual language for the
        real-world display than a solid rectangle. Arm length is clamped
        to at most 40% of the box's shorter side so small boxes never get
        overlapping/crossed brackets."""
        shorter_side = max(1, min(box.width, box.height))
        arm = max(_BRACKET_MIN_LENGTH_PX, min(_BRACKET_LENGTH_PX, int(0.4 * shorter_side)))

        corners = (
            ((box.x1, box.y1), (1, 1)),
            ((box.x2, box.y1), (-1, 1)),
            ((box.x1, box.y2), (1, -1)),
            ((box.x2, box.y2), (-1, -1)),
        )
        for (cx, cy), (sign_x, sign_y) in corners:
            cv2.line(frame, (cx, cy), (cx + sign_x * arm, cy), color, thickness)
            cv2.line(frame, (cx, cy), (cx, cy + sign_y * arm), color, thickness)

    @staticmethod
    def _draw_text_chip(
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        font_size: int,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Alpha-blend a translucent background rectangle sized to `text`
        behind `origin`, so field-display labels stay legible over any
        real-world background. ROI-local blend (not full-frame) since this
        runs per label, potentially several times per frame."""
        font = _load_field_font(font_size)
        left, top, right, bottom = font.getbbox(text)
        text_width = right - left
        text_height = bottom - top

        x1 = max(0, origin[0] - _FIELD_CHIP_PADDING_PX)
        y1 = max(0, origin[1] - _FIELD_CHIP_PADDING_PX)
        x2 = min(frame_width, origin[0] + text_width + _FIELD_CHIP_PADDING_PX)
        y2 = min(frame_height, origin[1] + text_height + _FIELD_CHIP_PADDING_PX)
        if x2 <= x1 or y2 <= y1:
            return

        roi = frame[y1:y2, x1:x2]
        overlay = np.full_like(roi, _FIELD_CHIP_BG_COLOR, dtype=np.uint8)
        cv2.addWeighted(overlay, _FIELD_CHIP_ALPHA, roi, 1 - _FIELD_CHIP_ALPHA, 0, dst=roi)

    @staticmethod
    def _draw_pil_text(
        frame: np.ndarray,
        entries: list[tuple[tuple[int, int], str, tuple[int, int, int], int]],
    ) -> None:
        """Render every (origin, text, color_bgr, font_size) entry onto
        `frame` in ONE BGR<->RGB/PIL round trip, regardless of how many
        entries there are -- batching per frame keeps the color-space
        conversion cost O(1) per frame rather than O(objects). No-op for
        an empty entry list (never touches `frame`)."""
        if not entries:
            return

        rgb_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(rgb_image)
        for origin, text, color_bgr, font_size in entries:
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            draw.text(origin, text, font=_load_field_font(font_size), fill=color_rgb)

        frame[:] = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _blend_overlay(
        frame: np.ndarray,
        draw_fn: Callable[[np.ndarray], None],
        alpha: float,
    ) -> None:
        """Draw via `draw_fn` onto a copy of `frame`, then alpha-blend that
        copy back over the original -- lets an existing solid-color draw
        function (e.g. a full-strength cv2.line/cv2.rectangle call) render
        at reduced opacity without duplicating its drawing logic."""
        overlay = frame.copy()
        draw_fn(overlay)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

    def draw_trajectories(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        predictions: dict[int, TrajectoryPrediction],
        draw_history: bool,
        history_points_to_draw: int,
    ) -> None:
        """Draw trajectory info for each tracked object.

        Debug-only visualization: an optional trail of recent center
        points, a "path: <direction>" label, and -- only when the
        prediction is valid and not stationary -- an arrow from the
        object's current center to its predicted future center plus a
        marker at that point. Drawn below the existing detection label
        and "dir:" line, never covering them.
        """
        for obj in tracked_objects:
            prediction = predictions.get(obj.track_id)
            if prediction is None:
                continue

            if draw_history and history_points_to_draw > 0:
                trail = obj.position_history[-history_points_to_draw:]
                for point in trail:
                    cv2.circle(frame, point, 2, _TRAJECTORY_HISTORY_COLOR, -1)

            path_label = f"path: {prediction.direction}"
            path_origin = (obj.bbox.x1, obj.bbox.y2 + 36)
            cv2.putText(
                frame, path_label, path_origin, _FONT, 0.5,
                _TRAJECTORY_COLOR, 1,
            )

            if not prediction.valid or prediction.direction == "stationary":
                continue

            cv2.arrowedLine(
                frame,
                obj.center,
                prediction.predicted_center,
                _TRAJECTORY_COLOR,
                2,
                tipLength=0.3,
            )
            cv2.circle(frame, prediction.predicted_center, 5, _TRAJECTORY_COLOR, -1)

    def draw_corridor(
        self,
        frame: np.ndarray,
        corridor_polygon: list[tuple[int, int]],
        draw_enabled: bool,
    ) -> None:
        """Draw the pedestrian-corridor outline, if enabled."""
        if not draw_enabled:
            return

        points = np.array(corridor_polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=True, color=_CORRIDOR_COLOR, thickness=2)

        top_left = corridor_polygon[3]
        label_origin = (top_left[0], max(top_left[1] - 10, 15))
        cv2.putText(
            frame, "pedestrian corridor", label_origin, _FONT, 0.5,
            _CORRIDOR_COLOR, 1,
        )

    def draw_path_intersections(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        intersection_results: dict[int, PathIntersectionResult],
        marker_radius: int,
    ) -> None:
        """Draw intersection status for each tracked object.

        Debug-only: marks the intersection point (if any) and a short
        status label -- "path intersects corridor", "path outside
        corridor", or "insufficient trajectory data". Does not redraw the
        trajectory arrow; that's handled by draw_trajectories.
        """
        for obj in tracked_objects:
            result = intersection_results.get(obj.track_id)
            if result is None:
                continue

            if not result.valid:
                status_label = "insufficient trajectory data"
                color = _UNKNOWN_COLOR
            elif result.intersects:
                status_label = "path intersects corridor"
                color = _INTERSECTS_COLOR
            else:
                status_label = "path outside corridor"
                color = _NO_INTERSECT_COLOR

            status_origin = (obj.bbox.x1, obj.bbox.y2 + 54)
            cv2.putText(
                frame, status_label, status_origin, _FONT, 0.5, color, 1
            )

            if result.intersection_point is not None:
                cv2.circle(
                    frame, result.intersection_point, marker_radius, color, 2
                )

    def draw_motion_estimate(
        self, frame: np.ndarray, motion_estimate: MotionEstimate | None
    ) -> None:
        """Draw Phase 4 camera-motion debug visualization, if available.

        Debug-only: accepted (inlier) feature points and their flow
        trails, rejected (outlier) points, an arrow showing the raw
        (dx, dy) motion vector at true pixel scale from the frame center
        (no exaggeration -- this visualizes the estimate, does not alter
        the video), and a text block with dx/dy/rotation/confidence/
        tracked-feature-count/inlier-count. When motion_estimate is None
        (motion estimation disabled, or no previous frame yet) or invalid,
        draws a single "CAMERA MOTION: INVALID -- <reason>" line instead.
        """
        if motion_estimate is None:
            return

        text_origin_y = 113  # continues straight below draw_debug_info's block

        if not motion_estimate.valid:
            cv2.putText(
                frame,
                f"CAMERA MOTION: INVALID -- {motion_estimate.reason_invalid}",
                (10, text_origin_y),
                _FONT, 0.55, _MOTION_INVALID_COLOR, 1,
            )
            return

        for prev_point, curr_point, is_inlier in zip(
            motion_estimate.debug_prev_points,
            motion_estimate.debug_current_points,
            motion_estimate.debug_inlier_flags,
        ):
            color = _MOTION_INLIER_COLOR if is_inlier else _MOTION_OUTLIER_COLOR
            if is_inlier:
                cv2.line(frame, prev_point, curr_point, color, 1)
            cv2.circle(frame, curr_point, 2, color, -1)

        frame_height, frame_width = frame.shape[:2]
        center = (frame_width // 2, frame_height // 2)
        if self._is_finite_xy(motion_estimate.dx, motion_estimate.dy):
            arrow_end = (
                max(0, min(round(center[0] + motion_estimate.dx), frame_width - 1)),
                max(0, min(round(center[1] + motion_estimate.dy), frame_height - 1)),
            )
            cv2.arrowedLine(frame, center, arrow_end, _MOTION_ARROW_COLOR, 2, tipLength=0.3)

        lines = [
            f"Camera motion [{motion_estimate.status}]: dx={motion_estimate.dx:.1f} "
            f"dy={motion_estimate.dy:.1f} rot={motion_estimate.rotation_degrees:.2f}deg "
            f"scale={motion_estimate.scale:.3f}",
            f"confidence={motion_estimate.confidence:.2f} "
            f"features={motion_estimate.feature_count} "
            f"matches={motion_estimate.tracked_feature_count} "
            f"inliers={motion_estimate.inlier_count} "
            f"inlier_ratio={motion_estimate.inlier_ratio:.2f}",
        ]
        for i, line in enumerate(lines):
            y = text_origin_y + i * 22
            cv2.putText(frame, line, (10, y), _FONT, 0.5, _MOTION_TEXT_COLOR, 1)

    def draw_motion_legend(self, frame: np.ndarray) -> None:
        """Small fixed-position legend distinguishing the three
        motion-vector styles drawn elsewhere (raw / compensated /
        predicted trajectory) so they're never ambiguous on screen.
        Debug-only; drawn once per frame in the bottom-left corner, the
        one screen area not already used by another overlay.
        """
        frame_height = frame.shape[0]
        origin_y = frame_height - 20 * len(_LEGEND_ENTRIES) - 10
        for i, (label, color) in enumerate(_LEGEND_ENTRIES):
            y = origin_y + i * 20
            cv2.line(frame, (10, y - 4), (30, y - 4), color, 2)
            cv2.putText(frame, label, (36, y), _FONT, 0.45, color, 1)

    def draw_compensated_motion(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        compensated_motions: dict[int, CompensatedMotion],
        filtered_motions: dict[int, FilteredMotion],
    ) -> None:
        """Draw Phase 4 camera-motion-compensated velocity, debug-only
        (shown when --debug is on but --validate-compensation is not).

        For every tracked object with a corresponding CompensatedMotion: a
        short text block (raw vs. compensated velocity) plus a raw-motion
        line (never a confident arrow -- see _draw_raw_motion_line). The
        text's final line and the directional arrow are driven by
        FilteredMotion when available (dead-zone/hysteresis/temporal-
        confirmation-filtered -- see src/motion/motion_state_filter.py),
        never by CompensatedMotion.compensated_direction/velocity directly
        -- that raw, single-frame value has no smoothing or confirmation
        and could otherwise flicker an arrow onto a genuinely parked
        object even while draw_tracked_objects's box label (also
        FilteredMotion-driven) correctly and stably reads STATIONARY. A
        track missing from compensated_motions (compensation disabled, or
        not yet available) is simply skipped, never a crash. A track
        missing from filtered_motions (defensive -- --debug being on
        should mean it's already computed for every current track) falls
        back to a clearly "(unconfirmed)"-labeled raw reading, and never
        draws a confident directional arrow from it. This remains a
        parallel debug data stream only -- it does not affect trajectory
        prediction or corridor-intersection decisions.
        """
        frame_height, frame_width = frame.shape[:2]
        for obj in tracked_objects:
            motion = compensated_motions.get(obj.track_id)
            if motion is None:
                continue

            filtered = filtered_motions.get(obj.track_id)

            origin_y = obj.bbox.y2 + 72
            lines = [
                f"comp ID {obj.track_id}",
                f"raw: ({motion.raw_velocity_x:.1f}, {motion.raw_velocity_y:.1f})",
                f"comp: ({motion.compensated_velocity_x:.1f}, "
                f"{motion.compensated_velocity_y:.1f})",
            ]
            if filtered is not None:
                uncertain_suffix = " UNCERTAIN" if filtered.uncertain else ""
                lines.append(
                    f"[{filtered.motion_state}] filtered speed: "
                    f"{filtered.speed:.1f}{uncertain_suffix}"
                )
            else:
                lines.append(
                    f"raw comp speed: {motion.compensated_speed:.1f} "
                    f"{motion.compensated_direction} (unconfirmed)"
                )
            for i, line in enumerate(lines):
                cv2.putText(
                    frame, line, (obj.bbox.x1, origin_y + i * 16),
                    _FONT, 0.42, _COMPENSATED_MOTION_COLOR, 1,
                )

            self._draw_raw_motion_line(
                frame, obj.center, motion.raw_velocity_x, motion.raw_velocity_y,
                frame_width, frame_height,
            )

            if filtered is None:
                # No temporally-confirmed motion available -- the raw
                # motion line above already shows the unconfirmed signal;
                # never assert a confident directional arrow without it.
                continue
            if filtered.motion_state != "MOVING":
                continue
            if not self._is_finite_xy(filtered.velocity_x, filtered.velocity_y):
                continue

            arrow_end = self._bounded_arrow_end(
                obj.center, filtered.velocity_x, filtered.velocity_y,
                frame_width, frame_height,
            )
            cv2.arrowedLine(
                frame, obj.center, arrow_end, _COMPENSATED_MOTION_COLOR, 2, tipLength=0.3
            )

    @classmethod
    def _draw_raw_motion_line(
        cls,
        frame: np.ndarray,
        center: tuple[int, int],
        dx: float,
        dy: float,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Draw the object's raw (uncompensated) velocity as a plain line
        with an open-circle tip -- deliberately NOT an arrowhead, so it
        can never be visually confused with the arrow-headed compensated/
        trajectory vectors (requirement: unambiguous styling). Skipped
        silently if the vector isn't finite or is ~zero."""
        if not cls._is_finite_xy(dx, dy):
            return
        if math.hypot(dx, dy) < 0.01:
            return

        end = cls._bounded_arrow_end(center, dx, dy, frame_width, frame_height)
        cv2.line(frame, center, end, _RAW_MOTION_COLOR, 1)
        cv2.circle(frame, end, 4, _RAW_MOTION_COLOR, 1)

    @staticmethod
    def _is_finite_xy(x: float, y: float) -> bool:
        """Whether both components of a 2D vector/point are finite
        (not NaN/inf) -- guards every arrow-drawing call so a degenerate
        upstream value is skipped safely instead of passed to cv2."""
        return math.isfinite(x) and math.isfinite(y)

    @staticmethod
    def _bounded_arrow_end(
        center: tuple[int, int],
        dx: float,
        dy: float,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int]:
        """Scale (dx, dy) for visibility, clamped to a max on-screen
        length so an extreme value can't draw an arrow that covers the
        frame, then clamp the final point into the frame's own
        boundaries -- so an arrow starting near an edge can never end up
        drawn (or attempted) outside the visible image."""
        scaled_dx = dx * _COMPENSATED_ARROW_SCALE
        scaled_dy = dy * _COMPENSATED_ARROW_SCALE
        length = math.hypot(scaled_dx, scaled_dy)
        if length > _COMPENSATED_ARROW_MAX_LENGTH_PX:
            factor = _COMPENSATED_ARROW_MAX_LENGTH_PX / length
            scaled_dx *= factor
            scaled_dy *= factor

        x = round(center[0] + scaled_dx)
        y = round(center[1] + scaled_dy)
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        return (x, y)

    def draw_validation_panel(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        stats: ValidationStats,
        per_object: list[PerObjectValidation],
        resolved_predictions: dict[int, TrajectoryPrediction],
    ) -> None:
        """Draw the developer-only --validate-compensation overlay.

        An aggregate summary panel (top-right corner, so it doesn't
        collide with the existing top-left FPS/camera-motion debug text)
        plus a per-object validation block -- status, raw vs. compensated
        vs. resolved vs. FILTERED velocity/speed, motion source, the
        temporally-confirmed motion_filter_state (and its reason, if
        uncertain), camera confidence, and raw-vs-resolved corridor-
        intersection outcome -- drawn in place of the plainer
        draw_compensated_motion block while this mode is
        active. A track missing from per_object (or from
        resolved_predictions, defensively) is simply skipped, never a
        crash. resolved_predictions is used only to draw the resolved-
        trajectory arrow; the actual corridor decision it's built from
        already happened upstream (see PerObjectValidation.
        resolved_intersects_corridor) -- this call is purely visual/debug.
        """
        self._draw_validation_summary_panel(frame, stats)

        frame_height, frame_width = frame.shape[:2]
        per_object_by_id = {row.track_id: row for row in per_object}
        for obj in tracked_objects:
            row = per_object_by_id.get(obj.track_id)
            if row is None:
                continue
            resolved_prediction = resolved_predictions.get(obj.track_id)
            self._draw_validation_object_block(
                frame, obj, row, resolved_prediction, frame_width, frame_height
            )

    def _draw_validation_summary_panel(
        self, frame: np.ndarray, stats: ValidationStats
    ) -> None:
        frame_width = frame.shape[1]
        x = max(0, frame_width - _VALIDATION_PANEL_WIDTH_PX - 10)
        lines = [
            "COMPENSATION VALIDATION",
            f"tracks={stats.active_track_count}",
            f"camera=({stats.camera_dx:.1f},{stats.camera_dy:.1f}) "
            f"conf={stats.camera_confidence:.2f} [{stats.camera_status}]",
            f"features={stats.feature_count} matches={stats.match_count} "
            f"inliers={stats.inlier_count}",
            f"raw_avg={stats.average_raw_speed:.1f} "
            f"comp_avg={stats.average_compensated_speed:.1f}",
            f"reduction={stats.reduction_percent:.1f}%",
            f"stationary={stats.stationary_count} moving={stats.moving_count}",
            f"cam_dominated={stats.camera_dominated_count} "
            f"uncertain={stats.uncertain_count}",
        ]
        for i, line in enumerate(lines):
            y = 25 + i * 20
            cv2.putText(frame, line, (x, y), _FONT, 0.45, _VALIDATION_PANEL_COLOR, 1)

    @classmethod
    def _draw_validation_object_block(
        cls,
        frame: np.ndarray,
        obj: TrackedObject,
        row: PerObjectValidation,
        resolved_prediction: TrajectoryPrediction | None,
        frame_width: int,
        frame_height: int,
    ) -> None:
        color = _VALIDATION_STATUS_COLORS.get(row.status, _COMPENSATED_MOTION_COLOR)
        filter_color = _MOTION_FILTER_STATE_COLORS.get(
            row.motion_filter_state, _COMPENSATED_MOTION_COLOR
        )
        origin_y = obj.bbox.y2 + 72
        uncertain_suffix = " UNCERTAIN" if row.prediction_uncertain else ""
        lines: list[tuple[str, tuple[int, int, int]]] = [
            (f"[{row.status}] ID {row.track_id} {row.class_name}", color),
            (
                f"raw: ({row.raw_velocity_x:.1f},{row.raw_velocity_y:.1f}) "
                f"spd={row.raw_speed:.1f}",
                color,
            ),
            (
                f"comp: ({row.compensated_velocity_x:.1f},"
                f"{row.compensated_velocity_y:.1f}) spd={row.compensated_speed:.1f}",
                color,
            ),
            (f"{row.compensated_direction} conf={row.camera_confidence:.2f}", color),
            (
                f"resolved [{row.motion_source}]: ({row.resolved_velocity_x:.1f},"
                f"{row.resolved_velocity_y:.1f}) spd={row.resolved_speed:.1f}"
                f"{uncertain_suffix}",
                color,
            ),
            (
                f"[{row.motion_filter_state}] filtered: "
                f"({row.filtered_velocity_x:.1f},{row.filtered_velocity_y:.1f}) "
                f"spd={row.filtered_speed:.1f} src={row.motion_source}"
                f"{uncertain_suffix} conf={row.motion_confirmation_frames}",
                filter_color,
            ),
            (
                f"corridor: raw={row.raw_intersects_corridor} "
                f"resolved={row.resolved_intersects_corridor}",
                color,
            ),
        ]
        if row.motion_filter_reason is not None:
            lines.append((f"reason={row.motion_filter_reason}", filter_color))

        for i, (line, line_color) in enumerate(lines):
            cv2.putText(
                frame, line, (obj.bbox.x1, origin_y + i * 16),
                _FONT, 0.42, line_color, 1,
            )

        cls._draw_raw_motion_line(
            frame, obj.center, row.raw_velocity_x, row.raw_velocity_y,
            frame_width, frame_height,
        )

        if (
            resolved_prediction is not None
            and resolved_prediction.valid
            and resolved_prediction.direction != "stationary"
        ):
            cv2.arrowedLine(
                frame, obj.center, resolved_prediction.predicted_center,
                _RESOLVED_TRAJECTORY_COLOR, 2, tipLength=0.3,
            )

        if row.compensated_direction == "stationary":
            return
        if not cls._is_finite_xy(row.compensated_velocity_x, row.compensated_velocity_y):
            return

        arrow_end = cls._bounded_arrow_end(
            obj.center, row.compensated_velocity_x, row.compensated_velocity_y,
            frame_width, frame_height,
        )
        cv2.arrowedLine(frame, obj.center, arrow_end, color, 2, tipLength=0.3)

    def draw_region_lines(self, frame: np.ndarray, left_px: int, right_px: int) -> None:
        """Draw vertical lines marking the left/center/right region
        boundaries, at reduced opacity -- a subtle spatial reference
        rather than a bold divider, for the real-world display."""
        height = frame.shape[0]

        def _draw(target: np.ndarray) -> None:
            cv2.line(target, (left_px, 0), (left_px, height), _REGION_LINE_COLOR, 1)
            cv2.line(target, (right_px, 0), (right_px, height), _REGION_LINE_COLOR, 1)

        self._blend_overlay(frame, _draw, _REGION_LINE_ALPHA)

    def draw_attention_zone(self, frame: np.ndarray, zone: AttentionZone) -> None:
        """Draw the central attention-zone rectangle, at reduced opacity
        and thinner stroke -- a subtle reference rather than a bold alert
        box, for the real-world display."""

        def _draw(target: np.ndarray) -> None:
            cv2.rectangle(
                target,
                (zone.x_min, zone.y_min),
                (zone.x_max, zone.y_max),
                _ATTENTION_ZONE_COLOR,
                _ATTENTION_ZONE_THICKNESS_PX,
            )

        self._blend_overlay(frame, _draw, _ATTENTION_ZONE_ALPHA)

    def draw_debug_info(
        self,
        frame: np.ndarray,
        fps: float,
        frame_width: int,
        frame_height: int,
        left_px: int,
        right_px: int,
        zone: AttentionZone,
    ) -> None:
        """Overlay FPS, frame size, region boundaries, and zone bounds."""
        lines = [
            f"FPS: {fps:.1f}",
            f"Frame: {frame_width}x{frame_height}",
            f"Regions: left<{left_px}px  right>{right_px}px",
            f"Zone: ({zone.x_min},{zone.y_min}) - ({zone.x_max},{zone.y_max})",
        ]
        for i, line in enumerate(lines):
            y = 25 + i * 22
            cv2.putText(frame, line, (10, y), _FONT, 0.55, _DEBUG_TEXT_COLOR, 1)

    def show(self, frame: np.ndarray) -> None:
        """Display the frame in the configured window."""
        cv2.imshow(self._window_name, frame)

    def close(self) -> None:
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()
