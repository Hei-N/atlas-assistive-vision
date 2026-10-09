"""Critical system-health failure detection and recovery for Atlas.

Detects exactly three critical failure conditions -- the camera lens
physically blocked, the camera/video feed lost, and the object detector
unavailable -- and one recovery condition, and decides when (and at
most how often) each of the four approved spoken messages should be
emitted. This is a pure decision component: it does not touch
SpeechQueue/EventPolicy/AudioWorker itself (see process_system_health()
in main.py for that orchestration), and it does not build AudioEvents
for anything except these four conditions.

Deliberately narrow by design (see docs/SYSTEM_HEALTH_AUDIO.md):
    - Never fires from a single bad frame/failure -- every condition
      requires either a time-based confirmation window (camera-blocked,
      recovery) or a consecutive-failure count (feed-lost, detector-
      unavailable), mirroring the confirmation-streak pattern already
      used by MotionStateFilter and ApproachEstimator.
    - Only one state is ever "primary" at a time, by priority:
      CAMERA_FEED_LOST > DETECTION_UNAVAILABLE > CAMERA_BLOCKED. Feed
      loss is both the most fundamental failure (no frame exists at
      all -- nothing else can even be evaluated) and the most certain
      (a hard read failure, not a heuristic). Detector unavailability
      is next: frames exist but can't be processed, also a hard,
      exception-confirmed failure. Camera-blocked is last because it is
      the least certain of the three -- a pixel-statistics heuristic
      that can never fully distinguish physical obstruction from a
      legitimately dark, flat, or blank real scene -- so a harder,
      more certain failure is always reported in preference to it when
      more than one condition is simultaneously true.
    - Each failure is spoken once on confirmation, at most one repeat
      after a delay, never a third repeat for the same uninterrupted
      episode. A different failure type starts a fresh episode
      immediately, bypassing the old one's cooldown -- the same
      "different key bypasses old cooldown" principle already used for
      Level-5 danger episodes in OperatingModePolicy.
    - "Detection restored." is spoken at most once per recovery, only
      after a confirmation window of continuous health, and only if a
      failure was actually spoken previously (a failure that was never
      announced has nothing to "restore" from).

Three conceptual phases (not a fourth `system_health_state` value --
derived from `operational_audio_suppressed` + `recovery_candidate` +
`recovery_announced`, so the existing 4-value state vocabulary
(HEALTHY/CAMERA_BLOCKED/CAMERA_FEED_LOST/DETECTION_UNAVAILABLE) doesn't
need to grow just to express this):
    - FAILURE_ACTIVE: system_health_state != HEALTHY.
      operational_audio_suppressed is always True. Ordinary object/
      summary/uncertainty audio never runs; only the system-failure
      message (initial + at most one repeat) is spoken.
    - RECOVERING: system_health_state == HEALTHY, a failure was
      previously spoken, but continuous health hasn't yet been observed
      for recovery_confirmation_seconds (recovery_candidate is True,
      recovery_announced is False). operational_audio_suppressed
      remains True throughout -- perception keeps running (frame reads,
      detector calls, tracking, motion) and keeps updating this
      monitor's own confirmation counters, but no ordinary object/
      summary/uncertain-object audio may be generated yet, since the
      subsystem's reliability over that window is exactly what's still
      being confirmed. If the failure returns before the window
      elapses, the confirmation timer is discarded (see evaluate()) and
      a fresh failure episode begins immediately.
    - HEALTHY_OPERATIONAL: system_health_state == HEALTHY and either no
      failure was ever spoken, or recovery_announced just became True
      THIS call (recovery_candidate True, recovery_announced True), or
      recovery was already announced on an earlier call.
      operational_audio_suppressed is False -- ordinary audio may run
      again, generated only from this frame's fresh perception (nothing
      accumulated during FAILURE_ACTIVE/RECOVERING is ever queued
      retroactively, since process_audio_events() was never even called
      during those phases -- there is nothing cached to discard).

Camera-blocked heuristic limitation (disclosed, not overclaimed): the
Laplacian-variance / Canny-edge-density check below is a cheap,
conservative pixel-statistics signal. It cannot exhaustively distinguish
every obstruction case from an unusually dark-and-flat real scene (e.g.
a close-up blank wall in dim light). It is tuned to require BOTH very
low variance AND very low edge density, sustained for a confirmation
window, specifically to avoid false positives on ordinary nighttime
streets (which still have edges from streetlights, silhouettes, and
structure) -- but it is not a robust, general obstruction detector.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from src.models import AudioEvent, SystemHealthTrace

logger = logging.getLogger("atlas")

HEALTHY = "HEALTHY"
CAMERA_BLOCKED = "CAMERA_BLOCKED"
CAMERA_FEED_LOST = "CAMERA_FEED_LOST"
DETECTION_UNAVAILABLE = "DETECTION_UNAVAILABLE"

SYSTEM_CAMERA_BLOCKED = "SYSTEM_CAMERA_BLOCKED"
SYSTEM_CAMERA_FEED_LOST = "SYSTEM_CAMERA_FEED_LOST"
SYSTEM_DETECTION_UNAVAILABLE = "SYSTEM_DETECTION_UNAVAILABLE"
SYSTEM_DETECTION_RESTORED = "SYSTEM_DETECTION_RESTORED"

KEY_CAMERA_BLOCKED = "system:camera_blocked"
KEY_CAMERA_FEED_LOST = "system:camera_feed_lost"
KEY_DETECTION_UNAVAILABLE = "system:detection_unavailable"
KEY_DETECTION_RESTORED = "system:detection_restored"

# The four -- and only four -- approved spoken messages. No wording
# beyond this without explicit approval; no mode-specific variants.
_MESSAGE_BY_STATE = {
    CAMERA_BLOCKED: "Warning! Camera blocked.",
    CAMERA_FEED_LOST: "Warning! Camera feed lost.",
    DETECTION_UNAVAILABLE: "Warning! Detection unavailable.",
}
_RECOVERY_MESSAGE = "Detection restored."

_EVENT_TYPE_BY_STATE = {
    CAMERA_BLOCKED: SYSTEM_CAMERA_BLOCKED,
    CAMERA_FEED_LOST: SYSTEM_CAMERA_FEED_LOST,
    DETECTION_UNAVAILABLE: SYSTEM_DETECTION_UNAVAILABLE,
}
_KEY_BY_STATE = {
    CAMERA_BLOCKED: KEY_CAMERA_BLOCKED,
    CAMERA_FEED_LOST: KEY_CAMERA_FEED_LOST,
    DETECTION_UNAVAILABLE: KEY_DETECTION_UNAVAILABLE,
}

REASON_DISABLED = "DISABLED"
REASON_HEALTHY = "HEALTHY"
REASON_STILL_CONFIRMING = "STILL_CONFIRMING"
REASON_CONFIRMED_NEW_FAILURE = "CONFIRMED_NEW_FAILURE"
REASON_FAILURE_REPEAT_ADMITTED = "FAILURE_REPEAT_ADMITTED"
REASON_REPEAT_NOT_DUE = "REPEAT_NOT_DUE"
REASON_REPEAT_LIMIT_REACHED = "REPEAT_LIMIT_REACHED"
REASON_RECOVERY_CONFIRMING = "RECOVERY_CONFIRMING"
REASON_RECOVERY_ANNOUNCED = "RECOVERY_ANNOUNCED"
REASON_NO_PRIOR_FAILURE = "NO_PRIOR_FAILURE"


class _FailureEpisode:
    """One critical-failure episode's speak/repeat bookkeeping -- a
    single record, since only one primary system state is ever active
    at a time (unlike OperatingModePolicy's per-class/region episode
    ledger, which tracks many simultaneous danger episodes)."""

    __slots__ = ("cycle_count", "last_admitted_time")

    def __init__(self) -> None:
        self.cycle_count = 0
        self.last_admitted_time: float | None = None


class SystemHealthMonitor:
    """Tracks camera/detector health across frames and decides when a
    critical-failure or recovery message should be spoken.

    Callers feed per-frame outcomes via record_frame_read_result(),
    record_frame_for_blocked_check(), and record_detector_result(), then
    call evaluate(now) once per frame to get the resulting state, an
    optional AudioEvent to speak, and a SystemHealthTrace for logging.

    Args:
        enabled: If False, evaluate() always reports HEALTHY and never
            produces an event -- an escape hatch, not expected to be
            used in normal operation.
        camera_blocked_confirmation_seconds: Consecutive seconds a frame
            must look blocked (see _looks_blocked) before CAMERA_BLOCKED
            is confirmed. Must be > 0.
        variance_threshold: Laplacian-variance threshold below which a
            frame is considered too flat/uniform. Must be > 0.
        edge_density_threshold: Canny edge-density threshold (fraction
            of edge pixels) below which a frame is considered too
            featureless. Must be > 0.
        camera_feed_lost_consecutive_failed_reads: Consecutive failed
            live-source frame reads before CAMERA_FEED_LOST is
            confirmed. Must be >= 1.
        detection_unavailable_consecutive_failures: Consecutive detector
            failures before DETECTION_UNAVAILABLE is confirmed. Must be
            >= 1.
        failure_repeat_seconds: Minimum seconds after a failure message
            was spoken before one repeat may be spoken, if the same
            failure is still active. Must be >= 0.
        failure_max_cycles: Maximum number of times the same
            uninterrupted failure episode may be spoken in total
            (including the first). Must be >= 1.
        recovery_confirmation_seconds: Consecutive seconds the system
            must be HEALTHY after a spoken failure before "Detection
            restored." is spoken. Must be > 0.
    """

    def __init__(
        self,
        enabled: bool,
        camera_blocked_confirmation_seconds: float,
        variance_threshold: float,
        edge_density_threshold: float,
        camera_feed_lost_consecutive_failed_reads: int,
        detection_unavailable_consecutive_failures: int,
        failure_repeat_seconds: float,
        failure_max_cycles: int,
        recovery_confirmation_seconds: float,
    ) -> None:
        if camera_blocked_confirmation_seconds <= 0:
            raise ValueError(
                "camera_blocked_confirmation_seconds must be > 0, got "
                f"{camera_blocked_confirmation_seconds}"
            )
        if variance_threshold <= 0:
            raise ValueError(f"variance_threshold must be > 0, got {variance_threshold}")
        if edge_density_threshold <= 0:
            raise ValueError(
                f"edge_density_threshold must be > 0, got {edge_density_threshold}"
            )
        if camera_feed_lost_consecutive_failed_reads < 1:
            raise ValueError(
                "camera_feed_lost_consecutive_failed_reads must be >= 1, got "
                f"{camera_feed_lost_consecutive_failed_reads}"
            )
        if detection_unavailable_consecutive_failures < 1:
            raise ValueError(
                "detection_unavailable_consecutive_failures must be >= 1, got "
                f"{detection_unavailable_consecutive_failures}"
            )
        if failure_repeat_seconds < 0:
            raise ValueError(
                f"failure_repeat_seconds must be >= 0, got {failure_repeat_seconds}"
            )
        if failure_max_cycles < 1:
            raise ValueError(f"failure_max_cycles must be >= 1, got {failure_max_cycles}")
        if recovery_confirmation_seconds <= 0:
            raise ValueError(
                "recovery_confirmation_seconds must be > 0, got "
                f"{recovery_confirmation_seconds}"
            )

        self._enabled = enabled
        self._camera_blocked_confirmation_seconds = camera_blocked_confirmation_seconds
        self._variance_threshold = variance_threshold
        self._edge_density_threshold = edge_density_threshold
        self._feed_lost_threshold = camera_feed_lost_consecutive_failed_reads
        self._detection_unavailable_threshold = detection_unavailable_consecutive_failures
        self._failure_repeat_seconds = failure_repeat_seconds
        self._failure_max_cycles = failure_max_cycles
        self._recovery_confirmation_seconds = recovery_confirmation_seconds

        self._consecutive_failed_reads = 0
        self._consecutive_detector_failures = 0
        self._blocked_since: float | None = None

        self._confirmed_state = HEALTHY
        self._state_started_at = 0.0
        self._episode: _FailureEpisode | None = None
        self._failure_spoken = False
        self._pending_recovery_since: float | None = None

    def record_frame_read_result(self, success: bool) -> None:
        """Record whether the most recent live-source frame read
        succeeded. Never call this for a video-file source reaching
        normal EOF -- that is not a failure (see
        VideoSource.is_live_source())."""
        if success:
            self._consecutive_failed_reads = 0
        else:
            self._consecutive_failed_reads += 1

    def record_frame_for_blocked_check(self, frame: np.ndarray | None, now: float) -> None:
        """Feed the current frame (or None, if no frame was read this
        iteration) to the camera-blocked heuristic. Resets the blocked
        streak immediately on any frame that doesn't look blocked, or
        when no frame is available (nothing to evaluate)."""
        if frame is None or not self._looks_blocked(frame):
            self._blocked_since = None
            return
        if self._blocked_since is None:
            self._blocked_since = now

    def record_detector_result(self, success: bool) -> None:
        """Record whether the most recent detector call succeeded. A
        frame with zero detections is a success -- only an exception or
        other unusable-result failure counts here."""
        if success:
            self._consecutive_detector_failures = 0
        else:
            self._consecutive_detector_failures += 1

    def evaluate(self, now: float | None = None) -> tuple[AudioEvent | None, SystemHealthTrace]:
        """Combine the latest recorded signals into this frame's primary
        state, decide whether a message is due, and return it (or None)
        alongside a full SystemHealthTrace for logging."""
        if now is None:
            now = time.time()

        if not self._enabled:
            trace = SystemHealthTrace(
                system_health_state=HEALTHY,
                previous_system_health_state=HEALTHY,
                health_state_started_at=now,
                health_confirmation_elapsed=0.0,
                system_failure_episode_key=None,
                system_failure_cycle_count=0,
                system_failure_repeat_due_at=None,
                operational_audio_suppressed=False,
                operational_queue_items_cleared=0,
                recovery_candidate=False,
                recovery_confirmation_elapsed=0.0,
                recovery_announced=False,
                system_message=None,
                health_reason_codes=(REASON_DISABLED,),
            )
            return None, trace

        previous_state = self._confirmed_state
        new_state = self._resolve_candidate_state(now)

        if new_state != previous_state:
            self._confirmed_state = new_state
            self._state_started_at = now
            if new_state != HEALTHY:
                self._episode = _FailureEpisode()
            else:
                self._pending_recovery_since = now
            logger.info("System health state changed: %s -> %s", previous_state, new_state)

        event: AudioEvent | None = None
        reason_codes: list[str] = []
        recovery_candidate = False
        recovery_elapsed = 0.0
        recovery_announced = False

        if new_state != HEALTHY:
            assert self._episode is not None
            if self._episode.cycle_count == 0:
                event = self._build_failure_event(new_state, now)
                self._episode.cycle_count = 1
                self._episode.last_admitted_time = now
                self._failure_spoken = True
                reason_codes.append(REASON_CONFIRMED_NEW_FAILURE)
            elif self._episode.cycle_count >= self._failure_max_cycles:
                reason_codes.append(REASON_REPEAT_LIMIT_REACHED)
            elif (now - self._episode.last_admitted_time) >= self._failure_repeat_seconds:
                event = self._build_failure_event(new_state, now)
                self._episode.cycle_count += 1
                self._episode.last_admitted_time = now
                reason_codes.append(REASON_FAILURE_REPEAT_ADMITTED)
            else:
                reason_codes.append(REASON_REPEAT_NOT_DUE)
        else:
            if self._failure_spoken:
                recovery_candidate = True
                assert self._pending_recovery_since is not None
                recovery_elapsed = now - self._pending_recovery_since
                if recovery_elapsed >= self._recovery_confirmation_seconds:
                    event = self._build_recovery_event(now)
                    recovery_announced = True
                    self._failure_spoken = False
                    self._episode = None
                    reason_codes.append(REASON_RECOVERY_ANNOUNCED)
                else:
                    reason_codes.append(REASON_RECOVERY_CONFIRMING)
            else:
                reason_codes.append(REASON_NO_PRIOR_FAILURE)

        episode_key = _KEY_BY_STATE.get(new_state) if new_state != HEALTHY else None
        episode_cycle_count = self._episode.cycle_count if self._episode is not None else 0
        repeat_due_at = (
            self._episode.last_admitted_time + self._failure_repeat_seconds
            if self._episode is not None and self._episode.last_admitted_time is not None
            else None
        )

        # Suppression covers two of the three conceptual phases, not just
        # one: FAILURE_ACTIVE (new_state != HEALTHY) AND RECOVERING
        # (new_state == HEALTHY, a failure was spoken, but continuous
        # health hasn't yet been confirmed for recovery_confirmation_
        # seconds). Only HEALTHY_OPERATIONAL -- never failed, or just
        # confirmed/announced recovery THIS call -- lifts suppression.
        # This is what stops ordinary object/summary/uncertainty audio
        # from resuming during an unconfirmed, potentially-still-unstable
        # recovery window.
        operational_audio_suppressed = (new_state != HEALTHY) or (
            recovery_candidate and not recovery_announced
        )

        trace = SystemHealthTrace(
            system_health_state=new_state,
            previous_system_health_state=previous_state,
            health_state_started_at=self._state_started_at,
            health_confirmation_elapsed=now - self._state_started_at,
            system_failure_episode_key=episode_key,
            system_failure_cycle_count=episode_cycle_count,
            system_failure_repeat_due_at=repeat_due_at,
            operational_audio_suppressed=operational_audio_suppressed,
            operational_queue_items_cleared=0,
            recovery_candidate=recovery_candidate,
            recovery_confirmation_elapsed=recovery_elapsed,
            recovery_announced=recovery_announced,
            system_message=event.message if event is not None else None,
            health_reason_codes=tuple(reason_codes),
        )
        logger.debug("System health: %s", trace)
        return event, trace

    def _resolve_candidate_state(self, now: float) -> str:
        if self._consecutive_failed_reads >= self._feed_lost_threshold:
            return CAMERA_FEED_LOST
        if self._consecutive_detector_failures >= self._detection_unavailable_threshold:
            return DETECTION_UNAVAILABLE
        if (
            self._blocked_since is not None
            and (now - self._blocked_since) >= self._camera_blocked_confirmation_seconds
        ):
            return CAMERA_BLOCKED
        return HEALTHY

    def _looks_blocked(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        variance = float(cv2.Laplacian(small, cv2.CV_64F).var())
        edges = cv2.Canny(small, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size
        return variance < self._variance_threshold and edge_density < self._edge_density_threshold

    @staticmethod
    def _build_failure_event(state: str, now: float) -> AudioEvent:
        return AudioEvent(
            key=_KEY_BY_STATE[state],
            event_type=_EVENT_TYPE_BY_STATE[state],
            priority="WARNING",
            message=_MESSAGE_BY_STATE[state],
            track_id=None,
            created_at=now,
            delivery_profile="level_5",
        )

    @staticmethod
    def _build_recovery_event(now: float) -> AudioEvent:
        return AudioEvent(
            key=KEY_DETECTION_RESTORED,
            event_type=SYSTEM_DETECTION_RESTORED,
            priority="INFORMATIONAL",
            message=_RECOVERY_MESSAGE,
            track_id=None,
            created_at=now,
            delivery_profile="level_1",
        )
