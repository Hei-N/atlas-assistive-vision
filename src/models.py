"""Typed data models shared across the Atlas Phase 1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates.

    Attributes:
        x1: Left edge x-coordinate (pixels).
        y1: Top edge y-coordinate (pixels).
        x2: Right edge x-coordinate (pixels).
        y2: Bottom edge y-coordinate (pixels).
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Width of the box in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height of the box in pixels."""
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        """Center point (x, y) of the box in pixels."""
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)


@dataclass(frozen=True)
class Detection:
    """A single detected object with region information.

    Attributes:
        class_id: Integer COCO class ID.
        class_name: Human-readable class name (e.g. "bicycle").
        confidence: Model confidence score in [0.0, 1.0].
        bbox: The detection's BoundingBox.
        center: (x, y) center point of the bounding box, in pixels.
        region: Horizontal region of the frame the detection falls in
            ("left", "center", or "right").
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    center: tuple[int, int]
    region: str


@dataclass(frozen=True)
class TrackedObject:
    """A Detection enriched with cross-frame tracking and motion state.

    Attributes:
        track_id: Stable identifier assigned by the ObjectTracker, retained
            across frames for as long as the object keeps being matched.
        class_id: Integer COCO class ID.
        class_name: Human-readable class name (e.g. "bicycle").
        confidence: Model confidence score in [0.0, 1.0] for this frame.
        bbox: The detection's BoundingBox for this frame.
        center: (x, y) center point of the bounding box, in pixels.
        region: Horizontal region of the frame the detection falls in
            ("left", "center", or "right").
        position_history: Oldest-to-newest (x, y) centers observed for this
            track, capped to the tracker's configured history length.
        size_history: Oldest-to-newest (width, height) pairs observed for
            this track, capped to the tracker's configured history length.
        direction: Movement direction derived from position_history, one of
            "left", "right", "up", "down", a diagonal combination (e.g.
            "up-left"), or "stationary".
        motion_status: One of "approaching", "moving away", or "stationary",
            derived from the change in bounding-box size over size_history.
        frames_since_seen: Consecutive frames since this track was last
            matched to a detection. Always 0 for a TrackedObject returned
            by ObjectTracker.update() today, since unmatched tracks are
            aged out internally rather than re-emitted; kept as an
            explicit field so downstream consumers (e.g.
            TrajectoryPredictor) can reason about staleness without
            depending on tracker internals.
    """

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    center: tuple[int, int]
    region: str
    position_history: tuple[tuple[int, int], ...]
    size_history: tuple[tuple[int, int], ...]
    direction: str
    motion_status: str
    frames_since_seen: int


@dataclass(frozen=True)
class TrajectoryPrediction:
    """A 2D image-space motion estimate and future-position prediction.

    Produced by TrajectoryPredictor from a track's position history. This
    is a smoothed, image-pixel-space estimate for visualization and
    technical validation only -- it is not a real-world distance, speed,
    or collision determination (no depth or GPS information is involved).

    Attributes:
        valid: Whether the prediction below is trustworthy. False when
            history is insufficient, stale, malformed, or the computed
            result is non-finite; in that case velocity/speed are 0.0,
            direction is "unknown", and predicted_center == current_center.
        velocity_x: Estimated horizontal motion in pixels/frame (smoothed).
        velocity_y: Estimated vertical motion in pixels/frame (smoothed).
        speed_px_per_frame: Magnitude of (velocity_x, velocity_y).
        direction: One of "left", "right", "upward", "downward",
            "upper-left", "upper-right", "lower-left", "lower-right",
            "stationary", or "unknown" (when not valid).
        current_center: The object's actual last-observed (x, y) center
            (unsmoothed), i.e. where a drawn prediction arrow should start.
        predicted_center: The estimated (x, y) center after
            prediction_horizon_frames, clamped to the frame boundaries.
        observations_used: Number of position samples used for this
            estimate.
        prediction_horizon_frames: How many frames ahead predicted_center
            was projected.
        uncertain: Whether this prediction should NOT be treated as a
            confident result. False for a normal, valid raw (regression-fit)
            prediction -- unchanged meaning from before this field existed.
            Always True when valid is False (an invalid prediction is never
            confident). When built from a resolved (camera-motion-aware)
            velocity via TrajectoryPredictor.predict_from_resolved_motion,
            True whenever the underlying ResolvedMotion was itself
            uncertain (LOW_CONFIDENCE or RAW_FALLBACK camera-motion status)
            -- see src/motion/motion_resolver.py. Downstream consumers (e.g.
            PathIntersectionResult) must not read a "no intersection" result
            as a confident SAFE signal when this is True.
    """

    valid: bool
    velocity_x: float
    velocity_y: float
    speed_px_per_frame: float
    direction: str
    current_center: tuple[int, int]
    predicted_center: tuple[int, int]
    observations_used: int
    prediction_horizon_frames: int
    uncertain: bool


@dataclass(frozen=True)
class PathIntersectionResult:
    """Whether a tracked object's predicted path enters the pedestrian
    corridor, in image space.

    Produced by PathIntersectionAnalyzer from a TrajectoryPrediction and
    the corridor polygon. This is an image-space geometric signal only --
    it is not a real-world collision determination (no depth, no GPS, no
    certainty claim).

    Attributes:
        valid: Whether this analysis could be evaluated at all. False when
            the underlying TrajectoryPrediction was itself invalid (e.g.
            insufficient history) -- in that case intersects/starts_inside/
            ends_inside are all False and intersection_point is None, but
            this must NOT be read as "path is safe/outside": see `reason`.
        intersects: True if the predicted path segment (current_center to
            predicted_center) starts inside the corridor, ends inside the
            corridor, or crosses a corridor boundary edge. Only meaningful
            when `valid` is True.
        starts_inside: True if current_center falls within the corridor.
        ends_inside: True if predicted_center falls within the corridor.
        intersection_point: The point where the segment crosses the
            corridor boundary, closest to current_center, if any such
            crossing exists. None if there is no boundary crossing (e.g.
            the segment stays entirely inside, stays entirely outside, or
            the analysis is invalid).
        track_id: The tracked object this result belongs to.
        reason: Short human-readable explanation of the outcome, e.g.
            "trajectory prediction invalid; intersection not evaluated",
            "segment enters corridor from outside", "segment does not
            intersect corridor".
        uncertain: Copied straight from the source TrajectoryPrediction's
            `uncertain` field. When True, `intersects=False` must NOT be
            read as a confident "path is safe" result -- it means the
            underlying prediction was built from a low-confidence or
            unavailable camera-motion-compensation estimate (or was itself
            invalid), not that a real analysis confidently found no
            intersection.
    """

    valid: bool
    intersects: bool
    starts_inside: bool
    ends_inside: bool
    intersection_point: tuple[int, int] | None
    track_id: int
    reason: str
    uncertain: bool


@dataclass(frozen=True)
class MotionEstimate:
    """Estimated 2D image-plane motion of the camera/background between
    two consecutive frames (Atlas Phase 4).

    Produced by a MotionEstimator (e.g. VisualMotionEstimator) from sparse
    optical flow on background feature points, robustly fit with RANSAC.
    This is a camera-motion estimate ONLY -- it does not correct or
    subtract anything from object trajectories, and it does not represent:
    meters, physical walking speed, GPS movement, real-world vehicle
    speed, or a full 3D camera pose. It is image-plane pixels and degrees.

    Coordinate conventions:
        - dx > 0 means the estimated background motion is toward the
          RIGHT; dx < 0 means toward the LEFT.
        - dy > 0 means the estimated background motion is DOWNWARD;
          dy < 0 means UPWARD. (Image y increases downward.)
        - rotation_degrees follows OpenCV's estimateAffinePartial2D
          decomposition (atan2 of the matrix's rotation/scale block, in
          degrees) in image coordinates (y-down). See
          src/motion/visual_motion_estimator.py's module docstring for the
          empirically-verified sign relative to cv2.getRotationMatrix2D.

    Attributes:
        valid: Whether this estimate is trustworthy. False when the input
            frames are malformed/mismatched, too few features could be
            detected or tracked, or the transform/RANSAC fit failed or had
            too few inliers -- see reason_invalid. When False, dx/dy/
            rotation_degrees/scale are 0.0/0.0/0.0/1.0 (a no-motion,
            no-scale default) and must not be treated as a real estimate.
        dx: Smoothed (if smoothing is enabled) horizontal background
            motion, in pixels.
        dy: Smoothed vertical background motion, in pixels.
        rotation_degrees: Smoothed in-plane rotation, in degrees.
        scale: Uniform scale component of the fitted similarity transform.
            Expected to be close to 1.0 for a non-zooming camera; not used
            for correction, only reported from the same transform fit.
        confidence: Heuristic score in [0.0, 1.0] combining RANSAC inlier
            ratio and tracked-feature sufficiency (see
            VisualMotionEstimator's module docstring for the exact
            formula). This is an explainable heuristic, not a statistically
            calibrated probability.
        feature_count: Number of raw corner features detected in the
            previous frame (cv2.goodFeaturesToTrack), before optical-flow
            tracking -- distinct from tracked_feature_count below (the
            "number of valid matches"), which is always <= this.
        tracked_feature_count: Number of feature points successfully
            tracked (matched) by optical flow from the previous frame into
            the current frame.
        inlier_count: Number of those tracked points RANSAC accepted as
            consistent with the fitted global transform.
        inlier_ratio: inlier_count / tracked_feature_count (0.0 when
            tracked_feature_count is 0). How much of what was tracked
            agrees with the fitted global motion.
        status: One of "VALID" (trustworthy), "LOW_CONFIDENCE" (valid but
            confidence below the estimator's configured threshold -- use
            with caution), or "UNAVAILABLE" (valid is False; see
            reason_invalid for why).
        reason_invalid: Short human-readable explanation of why `valid`
            is False (e.g. "insufficient features detected"). None when
            valid is True.
        raw_dx: Horizontal motion before temporal smoothing was applied.
        raw_dy: Vertical motion before temporal smoothing was applied.
        raw_rotation_degrees: Rotation before temporal smoothing.
        transform_matrix: The raw fitted 2x3 similarity transform, as
            ((a, b, tx), (c, d, ty)), or None if no transform was fit.
            Not interpreted further here.
        debug_prev_points: Previous-frame (x, y) pixel positions of every
            successfully-tracked feature point (both inliers and
            outliers). Empty when unavailable (e.g. valid is False before
            optical flow ran).
        debug_current_points: The same points' tracked (x, y) positions in
            the current frame -- parallel to debug_prev_points.
        debug_inlier_flags: Parallel to debug_prev_points/
            debug_current_points -- True where RANSAC accepted that point
            as an inlier, False where it was rejected as an outlier.
    """

    valid: bool
    dx: float
    dy: float
    rotation_degrees: float
    scale: float
    confidence: float
    feature_count: int
    tracked_feature_count: int
    inlier_count: int
    inlier_ratio: float
    status: str
    reason_invalid: str | None
    raw_dx: float
    raw_dy: float
    raw_rotation_degrees: float
    transform_matrix: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    debug_prev_points: tuple[tuple[int, int], ...]
    debug_current_points: tuple[tuple[int, int], ...]
    debug_inlier_flags: tuple[bool, ...]


@dataclass(frozen=True)
class CompensatedMotion:
    """A tracked object's velocity with estimated camera motion subtracted
    out (Atlas Phase 4), isolating apparent motion caused by the object's
    own movement from motion caused by the camera/wearer's head turning.

    Produced by MotionCompensator from a TrackedObject's own position
    history and a MotionEstimate (see src/motion/motion_compensator.py).
    Not used for trajectory or corridor decisions in the live pipeline.

    Both the object's own raw velocity and the camera's dx/dy are measured
    in the same image-space convention (pixels/frame, positive x = right,
    positive y = down -- see BoundingBox/MotionEstimate above), so
    compensation is a direct subtraction with no sign inversion: a
    real-world-stationary object's pixels move in lockstep with the rest
    of the background when the camera pans, so its raw apparent velocity
    approximately equals the camera's estimated background motion, and the
    two cancel out.

    Attributes:
        track_id: The tracked object this result belongs to.
        compensated_velocity_x: The object's horizontal velocity
            (pixels/frame) with estimated camera motion subtracted out.
        compensated_velocity_y: The object's vertical velocity
            (pixels/frame) with estimated camera motion subtracted out.
        compensated_speed: Magnitude of (compensated_velocity_x,
            compensated_velocity_y).
        compensated_direction: One of "left", "right", "up", "down", a
            diagonal combination (e.g. "up-left"), or "stationary" -- the
            same vocabulary as TrackedObject.direction, computed from the
            compensated velocity instead of raw position history.
        raw_velocity_x: The object's own uncompensated horizontal velocity
            (pixels/frame), from its two most recent position_history
            samples, before camera-motion subtraction.
        raw_velocity_y: The object's own uncompensated vertical velocity
            (pixels/frame).
        camera_motion_applied: Whether camera-motion compensation was
            actually applied. False (compensated_* equals raw_*) when the
            supplied camera motion was None, invalid, or below the
            compensator's configured confidence threshold -- a safe
            fallback, not an error.
    """

    track_id: int
    compensated_velocity_x: float
    compensated_velocity_y: float
    compensated_speed: float
    compensated_direction: str
    raw_velocity_x: float
    raw_velocity_y: float
    camera_motion_applied: bool


@dataclass(frozen=True)
class PerObjectValidation:
    """One tracked object's camera-motion-compensation validation row
    (Atlas Phase 4, developer-only --validate-compensation mode).

    Produced by src/motion/compensation_validator.py from a
    CompensatedMotion plus the frame's MotionEstimate. Purely a debug/
    validation view -- never consumed by trajectory prediction or
    corridor-intersection logic.

    Attributes:
        track_id: The tracked object this row belongs to.
        class_name: The object's detected class (e.g. "person"), for
            display alongside the track ID.
        raw_velocity_x: Object's uncompensated horizontal velocity
            (pixels/frame), from CompensatedMotion.raw_velocity_x.
        raw_velocity_y: Object's uncompensated vertical velocity.
        compensated_velocity_x: Horizontal velocity with camera motion
            subtracted out, from CompensatedMotion.compensated_velocity_x.
        compensated_velocity_y: Vertical velocity with camera motion
            subtracted out.
        raw_speed: Magnitude of (raw_velocity_x, raw_velocity_y).
        compensated_speed: Magnitude of (compensated_velocity_x,
            compensated_velocity_y), from CompensatedMotion.
        compensated_direction: From CompensatedMotion.compensated_direction.
        camera_confidence: This frame's MotionEstimate.confidence, or 0.0
            if no camera motion estimate was available.
        status: One of "STATIONARY", "CAMERA-DOMINATED", "OBJECT-MOVING",
            "UNCERTAIN" -- see classify_status in
            src/motion/compensation_validator.py for the exact,
            deterministic rule and its precedence.
        resolved_velocity_x: The object's resolved horizontal velocity
            (pixels/frame) actually used for trajectory prediction -- see
            ResolvedMotion / src/motion/motion_resolver.py.
        resolved_velocity_y: The object's resolved vertical velocity.
        resolved_speed: Magnitude of (resolved_velocity_x,
            resolved_velocity_y).
        motion_source: One of "COMPENSATED", "COMPENSATED_LOW_CONFIDENCE",
            "RAW_FALLBACK" -- which motion this row's resolved velocity
            came from, from ResolvedMotion.source.
        prediction_uncertain: The resolved TrajectoryPrediction's
            `uncertain` field -- True whenever the resolved motion should
            not be treated as a confident result (see TrajectoryPrediction).
        raw_intersects_corridor: Whether the RAW (uncompensated)
            trajectory's predicted path intersects the pedestrian
            corridor -- kept only for debug comparison, never the actual
            decision.
        resolved_intersects_corridor: Whether the RESOLVED (camera-motion-
            aware) trajectory's predicted path intersects the pedestrian
            corridor computed from the FILTERED velocity below -- this is
            the actual corridor decision.
        filtered_velocity_x: The object's horizontal velocity after
            src/motion/motion_state_filter.py's dead-zone/hysteresis/
            temporal-confirmation filtering -- this, not
            resolved_velocity_x, is what actually drives
            resolved_intersects_corridor. See FilteredMotion.
        filtered_velocity_y: The object's filtered vertical velocity.
        filtered_speed: Magnitude of (filtered_velocity_x,
            filtered_velocity_y).
        motion_filter_state: One of "STATIONARY", "MOVING", "UNCERTAIN",
            "INSUFFICIENT_HISTORY" -- from FilteredMotion.motion_state.
            Distinct from `status` above (the older classify_status
            STATIONARY/CAMERA-DOMINATED/OBJECT-MOVING/UNCERTAIN concept);
            this is the newer, temporally-confirmed motion state that
            actually determines filtered_velocity_x/y.
        motion_filter_reason: Why motion_filter_state is UNCERTAIN or
            INSUFFICIENT_HISTORY (e.g. "LOW_COMPENSATION_CONFIDENCE",
            "RAW_FALLBACK", "INSUFFICIENT_HISTORY"), or None when the
            state is a confident STATIONARY/MOVING. See FilteredMotion.
        motion_confirmation_frames: How many consecutive frames have
            supported a pending (not-yet-confirmed) state transition, from
            FilteredMotion.confirmation_frames -- 0 when there is no
            pending transition.
    """

    track_id: int
    class_name: str
    raw_velocity_x: float
    raw_velocity_y: float
    compensated_velocity_x: float
    compensated_velocity_y: float
    raw_speed: float
    compensated_speed: float
    compensated_direction: str
    camera_confidence: float
    status: str
    resolved_velocity_x: float
    resolved_velocity_y: float
    resolved_speed: float
    motion_source: str
    prediction_uncertain: bool
    raw_intersects_corridor: bool
    resolved_intersects_corridor: bool
    filtered_velocity_x: float
    filtered_velocity_y: float
    filtered_speed: float
    motion_filter_state: str
    motion_filter_reason: str | None
    motion_confirmation_frames: int


@dataclass(frozen=True)
class ValidationStats:
    """Aggregate camera-motion-compensation validation statistics for one
    frame (Atlas Phase 4, developer-only --validate-compensation mode).

    Produced by src/motion/compensation_validator.py from a list of
    PerObjectValidation rows plus the frame's MotionEstimate.

    Attributes:
        active_track_count: Number of tracked objects with compensated
            data this frame (i.e. len of the PerObjectValidation list this
            was built from).
        camera_dx: This frame's MotionEstimate.dx, or 0.0 if unavailable.
        camera_dy: This frame's MotionEstimate.dy, or 0.0 if unavailable.
        camera_confidence: This frame's MotionEstimate.confidence, or 0.0
            if unavailable.
        camera_status: This frame's MotionEstimate.status ("VALID" /
            "LOW_CONFIDENCE" / "UNAVAILABLE"), or "UNAVAILABLE" if there's
            no MotionEstimate at all this frame.
        feature_count: This frame's MotionEstimate.feature_count (raw
            detected corners), or 0 if unavailable.
        match_count: This frame's MotionEstimate.tracked_feature_count
            (successfully tracked matches), or 0 if unavailable.
        inlier_count: This frame's MotionEstimate.inlier_count, or 0 if
            unavailable.
        stationary_count: Number of objects classified STATIONARY.
        moving_count: Number of objects classified OBJECT-MOVING.
        camera_dominated_count: Number classified CAMERA-DOMINATED.
        uncertain_count: Number classified UNCERTAIN.
        average_raw_speed: Mean raw_speed across active tracks. 0.0 when
            there are no active tracks.
        average_compensated_speed: Mean compensated_speed across active
            tracks. 0.0 when there are no active tracks.
        reduction_percent: 100 * (average_raw_speed -
            average_compensated_speed) / average_raw_speed -- how much of
            the average apparent speed compensation removed. 0.0 (never a
            division error) whenever average_raw_speed is ~0, since
            there's nothing to reduce.
    """

    active_track_count: int
    camera_dx: float
    camera_dy: float
    camera_confidence: float
    camera_status: str
    feature_count: int
    match_count: int
    inlier_count: int
    stationary_count: int
    moving_count: int
    camera_dominated_count: int
    uncertain_count: int
    average_raw_speed: float
    average_compensated_speed: float
    reduction_percent: float


@dataclass(frozen=True)
class ResolvedMotion:
    """The single motion vector selected for downstream trajectory
    prediction, centralizing the VALID/LOW_CONFIDENCE/UNAVAILABLE decision
    in one place (src/motion/motion_resolver.py) instead of scattering
    camera-motion-status checks across trajectory prediction and corridor
    analysis.

    Produced by resolve_motion() from a CompensatedMotion, a
    MotionEstimate, and the raw TrajectoryPrediction (used as the fallback
    velocity source). Does not replace or overwrite either raw or
    compensated velocity -- both remain available on CompensatedMotion;
    this is a third, separate representation.

    Attributes:
        track_id: The tracked object this result belongs to.
        velocity_x: The selected horizontal velocity (pixels/frame) to use
            for trajectory prediction.
        velocity_y: The selected vertical velocity.
        speed: Magnitude of (velocity_x, velocity_y).
        source: One of "COMPENSATED" (camera-motion status was VALID),
            "COMPENSATED_LOW_CONFIDENCE" (status was LOW_CONFIDENCE --
            compensated velocity is still used for directional
            information, but the result is marked uncertain), or
            "RAW_FALLBACK" (status was UNAVAILABLE, or no CompensatedMotion
            was available for this track at all -- falls back to the raw
            trajectory's own regression-fit velocity).
        uncertain: Whether this resolved motion should not be treated as a
            confident result. False only when source is "COMPENSATED".
            Threaded through to TrajectoryPrediction.uncertain and then
            PathIntersectionResult.uncertain by the caller.
    """

    track_id: int
    velocity_x: float
    velocity_y: float
    speed: float
    source: str
    uncertain: bool


@dataclass(frozen=True)
class FilteredMotion:
    """The stationary-motion-filtered velocity actually fed into
    trajectory prediction, sitting between ResolvedMotion and
    TrajectoryPredictor.predict_from_resolved_motion.

    Produced by src/motion/motion_state_filter.py's MotionStateFilter from
    a ResolvedMotion plus stateful per-track dead-zone/hysteresis/
    temporal-confirmation filtering -- this absorbs bounding-box jitter,
    residual camera-compensation error, and single-frame velocity spikes
    so a genuinely stationary object doesn't flicker between reported
    states. Does not replace or overwrite ResolvedMotion (raw-selected
    velocity remains available there); this is a fourth, separate
    representation alongside raw/compensated/resolved.

    Attributes:
        track_id: The tracked object this result belongs to.
        velocity_x: The filtered horizontal velocity (pixels/frame) --
            (0.0, 0.0) whenever motion_state is STATIONARY,
            UNCERTAIN, or INSUFFICIENT_HISTORY (a non-confident-moving
            object must not generate a meaningful future trajectory).
        velocity_y: The filtered vertical velocity.
        speed: Magnitude of (velocity_x, velocity_y).
        motion_state: One of "STATIONARY", "MOVING", "UNCERTAIN",
            "INSUFFICIENT_HISTORY" -- the temporally-confirmed state (see
            MotionStateFilter for the exact hysteresis/confirmation
            rule). Orthogonal to `uncertain` below by design: a
            LOW_CONFIDENCE-sourced MOVING reading is still reported
            MOVING (still directionally useful) but always with
            uncertain=True, never as an unqualified confident claim.
        source: Passthrough of ResolvedMotion.source ("COMPENSATED",
            "COMPENSATED_LOW_CONFIDENCE", or "RAW_FALLBACK").
        uncertain: Whether this filtered motion should not be treated as
            a confident result. True whenever source is not "COMPENSATED",
            whenever motion_state is UNCERTAIN or INSUFFICIENT_HISTORY, or
            whenever a RAW_FALLBACK-sourced MOVING reading was suppressed
            (see MotionStateFilter's suppress_raw_motion_during_camera_
            motion). Threaded through to TrajectoryPrediction.uncertain
            and then PathIntersectionResult.uncertain by the caller,
            exactly like ResolvedMotion.uncertain was before this filter
            existed.
        reason: Why `uncertain` is True (e.g. "LOW_COMPENSATION_CONFIDENCE",
            "RAW_FALLBACK", "INSUFFICIENT_HISTORY"), or None when
            uncertain is False.
        confirmation_frames: Consecutive frames supporting a pending (not
            yet confirmed) state transition -- 0 when there is no pending
            transition (the confirmed state matches the latest reading,
            or the reading fell in the hysteresis dead band).
    """

    track_id: int
    velocity_x: float
    velocity_y: float
    speed: float
    motion_state: str
    source: str
    uncertain: bool
    reason: str | None
    confirmation_frames: int


@dataclass(frozen=True)
class AudioEvent:
    """One candidate spoken warning, built from resolved perception
    results only -- never raw trajectory/intersection data, never a
    debug value (track ID, class ID, confidence, pixel speed).

    Produced by src/audio/audio_event_builder.py's AudioEventBuilder,
    filtered by src/audio/event_policy.py's EventPolicy (cooldown),
    queued by src/audio/speech_queue.py's SpeechQueue (bounded,
    deduplicated, priority-ordered), and spoken by src/audio/
    audio_worker.py's AudioWorker. main.py never builds or sees the
    spoken text directly -- perception, event generation, and speech
    playback are kept strictly separate.

    Attributes:
        key: Deduplication/cooldown identity, "event_type:track_id:
            region" (or a bounded scene-level key when no stable track
            exists). Two events with the same key are the "same event"
            for cooldown and queue-deduplication purposes.
        event_type: One of "approaching", "corridor_crossing",
            "uncertain" -- see AudioEventBuilder for exactly when each
            is produced.
        priority: "WARNING" (corridor_crossing, uncertain) or
            "INFORMATIONAL" (approaching) -- determines SpeechQueue
            pop order.
        message: The exact, fully-rendered text to speak. Already
            contains no track IDs, class IDs, confidence values, pixel
            speeds, or debug values, and never asserts the road is safe.
        track_id: The tracked object this event is about, or None for a
            bounded scene-level fallback event.
        created_at: time.time() when this event was built -- used by
            SpeechQueue to drop stale (too-old-to-still-be-relevant)
            entries before they're ever spoken.
        delivery_profile: "level_1" through "level_5" for a hazard-level
            event (see AudioHazardResult), or "scene_summary" for a
            SceneSummarizer-produced summary -- the key AudioWorker
            looks up in its configured delivery-profile map (rate_wpm/
            cue) at speak time. AudioWorker never decides urgency
            itself, only looks this up; an unrecognized value falls
            back to AudioWorker's general configured rate.
    """

    key: str
    event_type: str
    priority: str
    message: str
    track_id: int | None
    created_at: float
    delivery_profile: str


@dataclass(frozen=True)
class ApproachResult:
    """Whether a tracked object is confirmed to be closing on the user or
    pedestrian corridor -- stronger evidence than FilteredMotion.MOVING
    alone (see src/motion/approach_estimator.py's module docstring for
    the full gate + 6-cue + confirmation-streak rule).

    Produced by ApproachEstimator.estimate() from a track's position/
    size history, its FilteredMotion, its resolved TrajectoryPrediction,
    its resolved PathIntersectionResult, and its current proximity zone.
    Does NOT redefine or replace FilteredMotion.motion_state -- MOVING
    keeps its existing meaning; this is an additional, stricter
    classification layered on top.

    Attributes:
        track_id: The tracked object this result belongs to.
        state: One of "NOT_APPROACHING" (confident negative -- gate
            failed, or not enough cues, or de-confirmed), "APPROACHING_
            UNCERTAIN" (cues + streak confirm closing, but the underlying
            FilteredMotion's source isn't COMPENSATED, so it's not a
            confident claim), "APPROACHING" (confident: gate passed,
            >= minimum_approach_cues cues true for
            >= approaching_confirmation_frames consecutive frames, and
            the underlying motion is COMPENSATED-sourced).
        confidence: cues_true_count / 6 this frame -- a heuristic
            fraction, not a calibrated probability (0.0 whenever the
            hard gate fails, since no cues are evaluated then).
        proximity_zone: Passthrough of the RelativeProximityEstimator
            result used for this estimate ("FAR"/"MID"/"NEAR"/"UNKNOWN").
        closing_score: Mean of the 3 continuous-trend cues (ground_point,
            corridor_distance, scale_growth), each clipped to [-1, 1]
            after dividing by its own configured threshold (corridor_
            distance_trend is negated first, since a decreasing distance
            is what indicates closing) -- positive means net closing
            evidence, negative means net opening evidence. 0.0 whenever
            the hard gate fails.
        corridor_distance_trend: Pixels/frame least-squares slope of
            distance-to-corridor-centroid over the history window;
            negative means decreasing distance (closing). 0.0 whenever
            the hard gate fails.
        scale_growth_trend: Least-squares slope of bounding-box area
            over the window, expressed as a fraction of the window's
            mean area per frame; positive means growing. 0.0 whenever
            the hard gate fails.
        ground_point_trend: Pixels/frame least-squares slope of the
            derived bottom-center y-coordinate over the window; positive
            means moving down-frame (closer, in image space). 0.0
            whenever the hard gate fails.
        confirmation_frames: Consecutive frames supporting a pending
            (not yet confirmed) state transition -- 0 when there is no
            pending transition.
        uncertain: True iff state == "APPROACHING_UNCERTAIN". A
            confident "NOT_APPROACHING" is NOT uncertain -- it's a
            confident negative, not an ambiguous result.
        evidence_flags: Which of the 6 cue names were true this frame,
            from ("GROUND_POINT_CLOSING", "PROXIMITY_INCREASING",
            "CORRIDOR_DISTANCE_DECREASING", "SCALE_GROWTH",
            "TRAJECTORY_TOWARD_CORRIDOR", "PATH_INTERSECTS"). Empty
            whenever the hard gate fails (no cues evaluated).
    """

    track_id: int
    state: str
    confidence: float
    proximity_zone: str
    closing_score: float
    corridor_distance_trend: float
    scale_growth_trend: float
    ground_point_trend: float
    confirmation_frames: int
    uncertain: bool
    evidence_flags: tuple[str, ...]


@dataclass(frozen=True)
class AudioHazardResult:
    """One tracked object's resolved audio hazard level -- the single
    decision point combining confirmed motion (FilteredMotion),
    proximity (RelativeProximityEstimator), confirmed approach
    (ApproachEstimator), and resolved corridor conflict
    (PathIntersectionResult) into exactly one of Atlas's 5 spoken hazard
    levels. See src/audio/audio_hazard_resolver.py for the full decision
    tree and every message template.

    Produced by AudioHazardResolver.resolve_for_tracks(). AudioEventBuilder
    consumes this directly and does not re-derive wording, priority, or
    urgency -- this is the single source of truth for "what should be
    said and how urgently."

    Attributes:
        track_id: The tracked object this result belongs to.
        level: One of "LEVEL_1_MOVING_FAR", "LEVEL_2_MOVING_NEARBY",
            "LEVEL_3_APPROACHING", "LEVEL_4_PATH_CONFLICT",
            "LEVEL_5_HIGH_DANGER" -- see AudioHazardResolver for the
            exact per-level conditions.
        priority: "INFORMATIONAL" (levels 1-2) or "WARNING" (levels 3-5).
        class_name: The object's detected class (e.g. "car"), for
            AudioEventBuilder's key construction -- never spoken
            directly (recommended_message already has the human-facing
            name substituted in).
        region: The object's current region ("left"/"center"/"right").
        proximity_zone: Passthrough of the RelativeProximityEstimator
            result used for this decision.
        approach_state: Passthrough of ApproachResult.state used for
            this decision.
        intersects_corridor: Whether the resolved PathIntersectionResult
            for this track intersects the corridor this frame.
        uncertain: Whether this level should be spoken with its
            cautionary ("Possible ...") template instead of its
            confident one -- see AudioHazardResolver's per-level
            uncertainty-cap rules. Always False for LEVEL_5 (Level 5 is
            structurally unreachable when any contributing signal is
            uncertain).
        reason_codes: Short strings explaining exactly why this level
            was selected (e.g. ("NEAR", "APPROACHING_CONFIRMED",
            "CORRIDOR_INTERSECTS") for Level 5), for the hazard trace.
        recommended_message: The exact, fully-rendered text to speak --
            AudioEventBuilder copies this verbatim into AudioEvent.message.
            Already contains no track IDs, class IDs, confidence values,
            or pixel speeds, and never asserts the road is safe.
        delivery_profile: "level_1" through "level_5" -- the key
            AudioWorker looks up in its configured delivery-profile map
            (rate_wpm/cue) at speak time. AudioWorker never decides
            urgency itself, only looks this up.
    """

    track_id: int
    level: str
    priority: str
    class_name: str
    region: str
    proximity_zone: str
    approach_state: str
    intersects_corridor: bool
    uncertain: bool
    reason_codes: tuple[str, ...]
    recommended_message: str
    delivery_profile: str


@dataclass(frozen=True)
class DeliveryProfile:
    """One hazard level's spoken-delivery configuration -- how, not
    what, to say (wording itself lives in AudioHazardResolver).

    Attributes:
        rate_wpm: Words per minute for `say -r` at this level.
        cue_enabled: Whether a short local alert-cue sound should play
            (via `afplay`, never a `say` volume/pitch trick -- macOS
            `say` has no clean per-utterance volume control, see
            AudioWorker's module docstring) immediately before the
            spoken message at this level.
        cue_path: Path to a user-provided local audio file, or None.
            Atlas never bundles or downloads a sound. Ignored (silently,
            never a crash) if cue_enabled is True but this is None or
            the file doesn't exist -- speech still plays normally.
        cue_gain: `afplay -v` volume for the cue file only, 0.0-1.0.
            Never changes the system's global output volume.
    """

    rate_wpm: float
    cue_enabled: bool
    cue_path: str | None
    cue_gain: float


@dataclass(frozen=True)
class HazardTrace:
    """One tracked object's hazard-decision snapshot for one frame --
    runtime-observable proof of exactly why a given hazard level was (or
    wasn't) selected and spoken, mirroring AudioSequencingTrace's own
    "verify at runtime, not just by listening" purpose for the separate
    startup-sequencing concern. Logged at DEBUG level by
    AudioHazardResolver (one line per active track would be too frequent
    for the INFO level used by startup-transition logging).

    Attributes:
        track_id: The tracked object this snapshot belongs to.
        proximity_zone: This frame's RelativeProximityEstimator result.
        approach_state: This frame's ApproachResult.state.
        approach_confidence: This frame's ApproachResult.confidence.
        approach_confirmation_frames: This frame's ApproachResult.
            confirmation_frames.
        approach_evidence_flags: This frame's ApproachResult.
            evidence_flags.
        hazard_level: The level actually selected this frame (before
            any de-escalation-hold suppression), or None if the object
            was silent (STATIONARY/INSUFFICIENT_HISTORY, or an
            announce_level_1/2 toggle suppressed it).
        hazard_reason_codes: This frame's AudioHazardResult.reason_codes,
            or () if hazard_level is None.
        delivery_rate_wpm: The resolved DeliveryProfile.rate_wpm for
            hazard_level, or None if hazard_level is None.
        delivery_cue_enabled: The resolved DeliveryProfile.cue_enabled
            for hazard_level, or None if hazard_level is None.
        previous_hazard_level: The level recorded for this track last
            frame, or None if there wasn't one (new track, or was
            previously silent).
        escalation: True iff hazard_level's rank is strictly higher than
            previous_hazard_level's rank this frame.
        deescalation_suppressed: True iff a natural (lower) level was
            computed this frame but withheld entirely by the
            de-escalation hold -- in that case hazard_level is None even
            though a level was internally computed.
        final_spoken_message: The message actually handed to
            AudioEventBuilder this frame, or None if nothing was
            produced (silent, or de-escalation-suppressed).
    """

    track_id: int
    proximity_zone: str
    approach_state: str
    approach_confidence: float
    approach_confirmation_frames: int
    approach_evidence_flags: tuple[str, ...]
    hazard_level: str | None
    hazard_reason_codes: tuple[str, ...]
    delivery_rate_wpm: float | None
    delivery_cue_enabled: bool | None
    previous_hazard_level: str | None
    escalation: bool
    deescalation_suppressed: bool
    final_spoken_message: str | None


@dataclass(frozen=True)
class ThreatAssessment:
    """One tracked object's canonical Level 1-5 threat classification --
    the single, audio-independent source of truth for "how urgent is
    this object right now," produced by src/threat_assessment.py's
    ThreatAssessmentEngine from the SAME resolved perception
    (FilteredMotion, ApproachResult, PathIntersectionResult, proximity
    zone) AudioHazardResolver already uses. This does not replace
    AudioHazardResult -- that remains the audio system's own decision
    (wording, delivery profile, its own de-escalation-hold timing for
    speech repetition). ThreatAssessment is upstream of and reusable by
    any future consumer (audio or otherwise) that needs "what level is
    this object at" without needing spoken-message concerns.

    Attributes:
        track_id: The tracked object this assessment belongs to.
        object_class: The object's detected class (e.g. "car", "person")
            -- never spoken directly, matches AudioHazardResult.class_name.
        region: The object's current region ("left"/"center"/"right").
        level: One of "NORMAL_MOVEMENT", "NEARBY_PRESENCE",
            "CONFIRMED_APPROACH", "PATH_CONFLICT", "IMMEDIATE_DANGER" --
            see src/threat_assessment.py for the exact per-level
            conditions and ThreatLevel's rank ordering.
        confidence: Reused directly from the contributing ApproachResult.
            confidence (cues_true_count / 6) -- the only graded,
            already-computed confidence heuristic anywhere in this
            pipeline. Not a calibrated probability, and not
            re-interpreted or recomputed for this new purpose.
        uncertain: Whether `level` should NOT be treated as a fully
            confident claim -- e.g. because the underlying motion source
            was COMPENSATED_LOW_CONFIDENCE/RAW_FALLBACK, or the approach/
            corridor evidence was itself uncertain. Always False for
            IMMEDIATE_DANGER (structurally unreachable from any uncertain
            contributing signal -- see classify_instantaneous_threat).
        reason_codes: Short strings explaining exactly why `level` was
            selected (e.g. ("NEAR", "APPROACHING_CONFIRMED",
            "CORRIDOR_INTERSECTS") for IMMEDIATE_DANGER) -- mirrors
            AudioHazardResult.reason_codes' auditability purpose.
        proximity_zone: The distance/relevance signal actually available
            to Atlas -- RelativeProximityEstimator's FAR/MID/NEAR/UNKNOWN
            image-space zone (bbox-bottom-edge-position based). Never a
            physical distance: no depth/stereo/GPS input exists anywhere
            in this pipeline.
        approach_state: Passthrough of ApproachResult.state
            ("NOT_APPROACHING"/"APPROACHING_UNCERTAIN"/"APPROACHING") --
            the approach evidence this assessment was built from.
        approach_evidence_flags: Passthrough of ApproachResult.
            evidence_flags -- which of the 6 independent closing cues
            were true, for auditability.
        intersects_corridor: Whether the resolved PathIntersectionResult
            for this track intersects the pedestrian corridor this
            frame, AFTER this engine's own multi-frame persistence gate
            (see path_conflict_confirmation_frames) -- a single-frame
            edge contact alone never sets this True.
        corridor_intersection_uncertain: Whether the contributing
            PathIntersectionResult.uncertain was True (low-confidence
            camera-motion/trajectory source) when intersects_corridor
            was computed.
        time_to_conflict_seconds: Always None today. Atlas has no
            calibrated depth, real-world speed, or metric distance
            anywhere in this pipeline, so a genuine time-to-conflict
            cannot be computed. Callers (e.g. rank_active_threats) can treat
            "unavailable" uniformly instead of a sentinel number.
            Conflict imminence does not populate it either (see
            ConflictImminence.estimated_seconds_to_conflict); use
            conflict_imminence_status for the categorical signal.
        persistence_frames: Consecutive frames (across calls to
            ThreatAssessmentEngine.assess_for_tracks) that `level` has
            been continuously reported for this track, including this
            one -- 1 on the frame `level` first takes this value.
        escalated: True iff `level`'s rank is strictly higher than the
            level reported for this same track on the previous frame it
            was assessed.
        deescalation_pending: True iff this frame's raw (instantaneous)
            evidence would justify a LOWER level than `level`, but the
            engine's de-escalation hold is still withholding that drop
            -- `level` still reflects the last confirmed (higher) reading
            while this is True.
        frame_index: The frame counter value this assessment was
            produced on, or None if the caller doesn't track one.
        timestamp: time.time() when this assessment was produced.
        conflict_imminence_status: Passthrough of the contributing
            ConflictImminence.status ("NONE"/"DISTANT"/"SOON"/"IMMINENT"/
            "UNKNOWN"), or None if no ConflictImminence was supplied to
            ThreatAssessmentEngine.assess_for_tracks for this track (e.g.
            conflict imminence is disabled, or this assessment predates
            Conflict Imminence V1's integration). See
            src/conflict_imminence.py -- an image-space, frame-step-based
            signal only, never a real-world time claim.
        first_conflict_step: Passthrough of the contributing
            ConflictImminence.first_conflict_step (predicted frames ahead
            at which the object's path first meaningfully enters the
            pedestrian corridor), or None when unavailable/not applicable.
    """

    track_id: int
    object_class: str
    region: str
    level: str
    confidence: float
    uncertain: bool
    reason_codes: tuple[str, ...]
    proximity_zone: str
    approach_state: str
    approach_evidence_flags: tuple[str, ...]
    intersects_corridor: bool
    corridor_intersection_uncertain: bool
    time_to_conflict_seconds: float | None
    persistence_frames: int
    escalated: bool
    deescalation_pending: bool
    frame_index: int | None
    timestamp: float
    conflict_imminence_status: str | None = None
    first_conflict_step: float | None = None


@dataclass(frozen=True)
class ConflictImminence:
    """A conservative, image-space signal answering: "if this object
    continues on its current resolved trajectory, how soon does its
    predicted path affect the pedestrian corridor?" Produced by
    src/conflict_imminence.py's ConflictImminenceEstimator from the same
    resolved FilteredMotion/TrajectoryPrediction/PathIntersectionResult
    ThreatAssessmentEngine already consumes.

    This is explicitly NOT physical time-to-collision: Atlas has no
    calibrated depth, monocular-depth model, real-world speed, or metric
    distance anywhere in this pipeline. "How soon" is expressed in
    predicted TRAJECTORY FRAME-STEPS (see first_conflict_step), not real
    seconds, because the trajectory prediction horizon is itself defined
    in frame-count units and Atlas has no reliable, fixed frame-to-
    real-time-seconds conversion (measured per-frame processing time
    fluctuates with system load) -- see estimated_seconds_to_conflict.

    Attributes:
        track_id: The tracked object this estimate belongs to.
        status: One of "NONE" (no predicted meaningful corridor conflict),
            "DISTANT" (a predicted conflict exists but late in the
            prediction horizon), "SOON" (conflict within a meaningfully
            near part of the horizon), "IMMINENT" (conflict very early in
            the horizon, backed by confident, persisted evidence), or
            "UNKNOWN" (insufficient or unreliable motion/trajectory
            evidence to say anything -- e.g. UNCERTAIN/INSUFFICIENT_
            HISTORY motion state, or an invalid trajectory/intersection
            result). A stationary object (FilteredMotion.motion_state ==
            "STATIONARY") always reports "NONE" here -- an existing
            corridor obstruction is a proximity/path-conflict concern
            (Level 4, via ThreatAssessmentEngine directly), never a
            predicted dynamic conflict for an object that is not moving.
        first_conflict_step: The earliest predicted future trajectory
            step (in predicted FRAMES ahead, fractional) at which the
            object's predicted path first meaningfully enters the
            pedestrian corridor -- derived from the same corridor-
            intersection geometry PathIntersectionAnalyzer already
            computes (the intersection point's fractional distance along
            the current_center -> predicted_center segment, multiplied by
            TrajectoryPrediction.prediction_horizon_frames). None when
            status is "NONE" or "UNKNOWN" (no meaningful predicted
            conflict, or not evaluable).
        estimated_seconds_to_conflict: Always None today, deliberately.
            Populating this would require converting predicted frame
            steps to elapsed seconds, and Atlas has no fixed, reliable
            frame interval (see module docstring). It must never be read
            as "seconds until collision with the user."
        confidence: 1.0 when `status` was computed from confident (non-
            uncertain) evidence, 0.0 when `uncertain` is True or `status`
            is "UNKNOWN". A simple confident/not-confident indicator, not
            a graded, calibrated probability -- there is no independent
            graded evidence quantity (unlike ApproachResult.confidence's
            6-cue count) to compute a finer value from here.
        uncertain: Whether `status` should NOT be treated as a fully
            confident claim -- True whenever the contributing
            TrajectoryPrediction/PathIntersectionResult was itself
            uncertain (low-confidence camera-motion compensation or
            RAW_FALLBACK source), or motion evidence was insufficient. An
            uncertain reading is capped at "SOON" at most -- evidence
            quality insufficient to report a confident "IMMINENT" is
            reported "SOON" instead (see ConflictImminenceEstimator).
        reason_codes: Short strings explaining why `status` was selected
            (e.g. ("STATIONARY",), ("PREDICTED_CONFLICT_IMMINENT",)) --
            mirrors ThreatAssessment.reason_codes' auditability purpose.
        frame_index: The frame counter value this estimate was produced
            on, or None if the caller doesn't track one.
        timestamp: time.time() when this estimate was produced.
    """

    track_id: int
    status: str
    first_conflict_step: float | None
    estimated_seconds_to_conflict: float | None
    confidence: float
    uncertain: bool
    reason_codes: tuple[str, ...]
    frame_index: int | None
    timestamp: float


@dataclass(frozen=True)
class OperatingModeProfile:
    """One operating mode's fully-resolved behavior settings -- MINIMAL,
    BALANCED, or DETAILED (src/audio/operating_mode_policy.py). All three
    modes share the exact same upstream perception and hazard-level
    decisions (AudioHazardResolver); this profile only controls what
    OperatingModePolicy filters/reshapes downstream of that.

    Attributes:
        mode: "MINIMAL", "BALANCED", or "DETAILED".
        announce_traffic: Whether traffic-class (car/truck/bus/
            motorcycle/bicycle) events are eligible at all. True in every
            shipped profile -- kept explicit for forward compatibility
            and symmetry with person_eligibility, not because any mode
            currently disables it.
        person_eligibility: "NONE" (Minimal -- person events never
            eligible), "RELEVANT_ONLY" (Balanced -- eligible only when
            the person's own hazard level is NEAR/approaching/path-
            conflict, i.e. not the ordinary-far-movement level), or
            "BROAD" (Detailed -- always eligible, including ordinary
            movement, feeding broader scene summaries).
        highest_danger_repeat_enabled: Whether an unchanged Level-5
            episode may be spoken a second time (see hazard_episode_*
            trace fields) rather than exactly once.
        highest_danger_repeat_delay_seconds: Minimum seconds after the
            first admission of a Level-5 episode before a repeat may be
            admitted.
        highest_danger_max_cycles: Maximum number of times the same
            unchanged Level-5 episode may be spoken in total (including
            the first).
        structured_scene_summaries: Whether Detailed-mode's group/word
            caps (below) apply at all.
        max_summary_object_groups: Maximum number of non-WARNING summary
            events admitted in the same frame (Detailed only).
        max_summary_words: Maximum word count for a single summary
            message before it falls back to SceneSummarizer's configured
            fallback_message instead of being spoken verbatim (Detailed
            only).
    """

    mode: str
    announce_traffic: bool
    person_eligibility: str
    highest_danger_repeat_enabled: bool
    highest_danger_repeat_delay_seconds: float
    highest_danger_max_cycles: int
    structured_scene_summaries: bool
    max_summary_object_groups: int
    max_summary_words: int


@dataclass(frozen=True)
class OperatingModeTrace:
    """One track's (or one frame-level summary/episode decision's)
    operating-mode filtering snapshot -- runtime-observable proof of
    exactly why an event was allowed, suppressed, or reworded by mode,
    mirroring AudioSequencingTrace/HazardTrace's own "verify at runtime"
    purpose for their respective concerns. Logged at DEBUG level by
    OperatingModePolicy.

    Attributes:
        track_id: The tracked object this snapshot belongs to, or None
            for a frame-level (summary/episode) decision not tied to a
            single track.
        operating_mode: The mode active when this decision was made.
        mode_source: "CLI", "CONFIG", or "DEFAULT" -- how the active
            mode was originally selected.
        event_allowed_by_mode: Whether filter_events() kept this track's
            hazard result eligible.
        mode_filter_reason: One of the stable reason codes documented in
            operating_mode_policy.py (e.g. "ALLOWED_TRAFFIC_EVENT",
            "SUPPRESSED_PERSON_IN_MINIMAL").
        mode_warning_repeat_enabled: This mode's
            OperatingModeProfile.highest_danger_repeat_enabled value.
        hazard_episode_key: The Level-5 danger-episode identity string
            (region-based -- see module docstring for why track_id is
            deliberately excluded), or None if this event isn't a
            Level-5 danger event.
        hazard_episode_cycle_count: How many times this episode has been
            admitted so far (including this decision, if admitted).
        repeat_due_timestamp: The earliest time a repeat of this episode
            may next be admitted, or None if not applicable.
        repeat_admitted: Whether this specific call admitted a repeat
            (cycle_count > 1) rather than a fresh episode or a
            suppression.
        repeat_suppression_reason: Why a repeat/summary was suppressed
            (e.g. "SUPPRESSED_REPEAT_LIMIT"), or None if not suppressed.
        detailed_summary_group_count: How many non-WARNING summary
            events were admitted this frame (Detailed mode's group cap).
        detailed_summary_word_count: Word count of this summary message,
            or 0 if not a summary.
        final_mode_adjusted_message: The message text after
            finalize_events()'s mode-specific rewriting, or None if this
            event was suppressed or needed no rewriting.
    """

    track_id: int | None
    operating_mode: str
    mode_source: str
    event_allowed_by_mode: bool
    mode_filter_reason: str
    mode_warning_repeat_enabled: bool
    hazard_episode_key: str | None
    hazard_episode_cycle_count: int
    repeat_due_timestamp: float | None
    repeat_admitted: bool
    repeat_suppression_reason: str | None
    detailed_summary_group_count: int
    detailed_summary_word_count: int
    final_mode_adjusted_message: str | None


@dataclass(frozen=True)
class SystemHealthTrace:
    """One frame's critical-system-health snapshot -- runtime-observable
    proof of exactly why a system-failure/repeat/recovery message was
    (or wasn't) spoken, mirroring AudioSequencingTrace/HazardTrace/
    OperatingModeTrace's own "verify at runtime" purpose for their
    respective concerns. Produced by src/system/system_health_monitor.py's
    SystemHealthMonitor once per frame.

    Attributes:
        system_health_state: The current PRIMARY state this frame --
            one of "HEALTHY", "CAMERA_BLOCKED", "CAMERA_FEED_LOST",
            "DETECTION_UNAVAILABLE". Only one state is ever primary even
            if multiple conditions are simultaneously true -- see
            SystemHealthMonitor's module docstring for the exact
            priority order and why.
        previous_system_health_state: The state recorded on the prior
            call, for detecting a transition.
        health_state_started_at: time.time() when the CURRENT state was
            first observed (not necessarily confirmed/spoken yet).
        health_confirmation_elapsed: Seconds the current state has been
            continuously observed -- compared against the relevant
            confirmation threshold to decide whether it's confirmed.
        system_failure_episode_key: The stable key
            ("system:camera_blocked" etc.) for the currently-active
            failure episode, or None if healthy.
        system_failure_cycle_count: How many times the current failure
            episode has been spoken so far (including this frame, if
            spoken now) -- capped at failure_max_cycles.
        system_failure_repeat_due_at: The earliest time a repeat of the
            current failure may next be spoken, or None if not
            applicable.
        operational_audio_suppressed: Whether process_audio_events() was
            skipped entirely this frame -- True for both the
            FAILURE_ACTIVE phase (system_health_state != "HEALTHY") AND
            the RECOVERING phase (system_health_state == "HEALTHY" but
            recovery_candidate is True and recovery_announced is False
            -- continuous health hasn't yet been confirmed for
            recovery_confirmation_seconds). Only False once truly
            HEALTHY_OPERATIONAL: never failed, or recovery_announced is
            True this call or a prior one. See
            src/system/system_health_monitor.py's module docstring for
            the full three-phase breakdown.
        operational_queue_items_cleared: How many pending SpeechQueue
            entries were removed this frame by SpeechQueue.clear() when
            a new failure/recovery message was admitted -- 0 on a frame
            where nothing was cleared.
        recovery_candidate: Whether the system is currently HEALTHY and
            a failure was previously spoken (i.e. recovery is being
            timed, even if not yet confirmed/announced).
        recovery_confirmation_elapsed: Seconds HEALTHY has been
            continuously observed since the failure cleared -- compared
            against recovery_confirmation_seconds.
        recovery_announced: Whether "Detection restored." was spoken
            this frame.
        system_message: The exact system message spoken this frame, or
            None if nothing was spoken.
        health_reason_codes: Short strings explaining this frame's
            decision (e.g. why a repeat was or wasn't admitted).
    """

    system_health_state: str
    previous_system_health_state: str
    health_state_started_at: float
    health_confirmation_elapsed: float
    system_failure_episode_key: str | None
    system_failure_cycle_count: int
    system_failure_repeat_due_at: float | None
    operational_audio_suppressed: bool
    operational_queue_items_cleared: int
    recovery_candidate: bool
    recovery_confirmation_elapsed: float
    recovery_announced: bool
    system_message: str | None
    health_reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class AudioSequencingTrace:
    """One frame's snapshot of the startup/warning audio-sequencing
    state -- runtime-observable proof that the startup announcement and
    object-warning speech never overlap, for offline verification
    (rather than only judging correctness by listening to a recording).

    Produced by src/audio/audio_sequencing_tracer.py's
    AudioSequencingTracer, called once per frame from main.py's
    process_audio_events(). Purely observational: nothing here makes any
    admission/speaking decision -- it only records decisions already
    made by StartupAudioGate/AudioEventBuilder/EventPolicy/SpeechQueue/
    AudioWorker.

    Attributes:
        startup_state: StartupAudioGate.state this frame -- one of
            "INITIALIZING", "STARTUP_SPEAKING", "READY".
        startup_process_pid: StartupAnnouncer's speech subprocess PID,
            or None before it starts / after it's no longer tracked.
        startup_speech_start_time: time.time() of the first frame the
            startup announcement was observed actually speaking
            (StartupAnnouncer.is_speaking first True), or None if that
            hasn't happened yet this run.
        startup_speech_finish_time: time.time() of the first frame after
            that where the startup announcement was observed to have
            actually finished (has_announced True and is_speaking False
            -- i.e. its subprocess's poll() indicated completion, not
            merely that speaking was triggered). None until then.
        warning_pipeline_enabled: True iff the warning pipeline ran this
            frame (equivalent to `not StartupAudioGate.warnings_blocked`).
        candidate_created_during_startup: Sticky (once True, stays True)
            safety-net flag -- True only if AudioEventBuilder.
            build_events() was ever actually invoked while the gate was
            blocking warnings. Always False in normal operation, since
            process_audio_events() never calls build_events() at all
            while blocked; this field exists to make that guarantee
            independently verifiable rather than merely asserted.
        startup_gate_decision: "SUPPRESSED_STARTUP_EXCLUSIVE" while
            warnings are blocked, "ADMITTED" once the gate is READY.
        startup_gate_reason: Human-readable reason for the decision
            above, f"startup_state={startup_state}".
        queue_size_during_startup: len(SpeechQueue) observed while
            blocked (always expected to be 0 -- see invariant A), or 0
            once READY (not meaningful once warnings are flowing).
        first_warning_candidate_time: time.time() of the first frame,
            after READY, that AudioEventBuilder.build_events() was
            actually called. None until then.
        first_warning_speech_start_time: time.time() of the first frame
            AudioWorker was observed actually speaking a warning
            (AudioWorker.is_speaking first True). None until then. Must
            always be >= startup_speech_finish_time (invariant C).
        active_speech_process_count: StartupAnnouncer.is_speaking +
            AudioWorker.is_speaking this frame, as an int in {0, 1, 2}.
            Must never be 2 -- see invariant A/the "only one speech
            subprocess at a time" requirement.
    """

    startup_state: str
    startup_process_pid: int | None
    startup_speech_start_time: float | None
    startup_speech_finish_time: float | None
    warning_pipeline_enabled: bool
    candidate_created_during_startup: bool
    startup_gate_decision: str
    startup_gate_reason: str
    queue_size_during_startup: int
    first_warning_candidate_time: float | None
    first_warning_speech_start_time: float | None
    active_speech_process_count: int
