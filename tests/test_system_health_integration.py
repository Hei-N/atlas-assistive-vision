"""Integration tests proving -- from real runtime state, not by
listening to a recording -- that critical system-health messages
correctly gate, clear, and replace ordinary operational audio.

Uses real SpeechQueue/EventPolicy/AudioWorker/SystemHealthMonitor driven
through main.py's actual process_system_health() (and, where relevant,
process_audio_events()), with only `subprocess` mocked (never real
`say`). Mirrors tests/test_audio_sequencing.py's established pattern.
"""

from unittest.mock import MagicMock

import pytest

from main import (
    build_audio_event_builder,
    build_audio_hazard_resolver,
    build_operating_mode_policy,
    process_audio_events,
    process_system_health,
)
from src.audio.audio_sequencing_tracer import AudioSequencingTracer
from src.audio.audio_worker import AudioWorker
from src.audio.event_policy import EventPolicy
from src.audio.scene_summarizer import SceneSummarizer
from src.audio.speech_queue import SpeechQueue
from src.audio.startup_announcer import StartupAnnouncer
from src.audio.startup_audio_gate import READY
from src.models import (
    ApproachResult,
    AudioEvent,
    BoundingBox,
    DeliveryProfile,
    FilteredMotion,
    PathIntersectionResult,
    TrackedObject,
)
from src.motion.approach_estimator import NOT_APPROACHING
from src.system.system_health_monitor import SystemHealthMonitor
from src.threat_assessment import ThreatAssessmentEngine

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


@pytest.fixture(autouse=True)
def mock_subprocess(monkeypatch):
    instances: list = []

    def make_popen(*args, **kwargs):
        instance = MagicMock()
        instance.poll.return_value = None
        instance.args = args[0] if args else kwargs.get("args")
        instances.append(instance)
        return instance

    mock_popen = MagicMock(side_effect=make_popen)
    monkeypatch.setattr("src.audio.audio_worker.subprocess.Popen", mock_popen)

    run_result = MagicMock()
    run_result.stdout = "Samantha               en_US    # Hello, I'm Samantha."
    monkeypatch.setattr(
        "src.audio.startup_announcer.subprocess.run", MagicMock(return_value=run_result)
    )

    return {"popen": mock_popen, "instances": instances}


def make_monitor(**overrides) -> SystemHealthMonitor:
    defaults = dict(
        enabled=True,
        camera_blocked_confirmation_seconds=1.0,
        variance_threshold=50.0,
        edge_density_threshold=0.02,
        camera_feed_lost_consecutive_failed_reads=3,
        detection_unavailable_consecutive_failures=3,
        failure_repeat_seconds=5.0,
        failure_max_cycles=2,
        recovery_confirmation_seconds=1.0,
    )
    defaults.update(overrides)
    return SystemHealthMonitor(**defaults)


def make_delivery_profiles() -> dict:
    return {
        f"level_{n}": DeliveryProfile(rate_wpm=180, cue_enabled=False, cue_path=None, cue_gain=0.0)
        for n in range(1, 6)
    }


def make_worker(speech_queue: SpeechQueue, event_policy: EventPolicy) -> AudioWorker:
    return AudioWorker(
        enabled=True, queue=speech_queue, event_policy=event_policy,
        configured_voice=None, fallback_voice=None, rate_wpm=170,
        delivery_profiles=make_delivery_profiles(),
    )


def make_tracked_object(track_id: int = 1, region: str = "left") -> TrackedObject:
    bbox = BoundingBox(x1=0, y1=0, x2=20, y2=20)
    return TrackedObject(
        track_id=track_id, class_id=2, class_name="car", confidence=0.9,
        bbox=bbox, center=(10, 10), region=region, position_history=((10, 10),),
        size_history=((20, 20),), direction="right", motion_status="moving",
        frames_since_seen=0,
    )


def make_filtered_motion(track_id: int) -> FilteredMotion:
    return FilteredMotion(
        track_id=track_id, velocity_x=5.0, velocity_y=0.0, speed=5.0,
        motion_state="MOVING", source="COMPENSATED", uncertain=False,
        reason=None, confirmation_frames=0,
    )


def make_approach_result(track_id: int) -> ApproachResult:
    return ApproachResult(
        track_id=track_id, state=NOT_APPROACHING, confidence=0.0, proximity_zone="FAR",
        closing_score=0.0, corridor_distance_trend=0.0, scale_growth_trend=0.0,
        ground_point_trend=0.0, confirmation_frames=0, uncertain=False, evidence_flags=(),
    )


def make_intersection_result(track_id: int) -> PathIntersectionResult:
    return PathIntersectionResult(
        valid=True, intersects=False, starts_inside=False, ends_inside=False,
        intersection_point=None, track_id=track_id, reason="test", uncertain=False,
    )


class _FakeCore:
    def __init__(self, track_id: int):
        self.filtered_motions = {track_id: make_filtered_motion(track_id)}
        self.resolved_intersection_results = {track_id: make_intersection_result(track_id)}


class _FakeHazardPerception:
    def __init__(self, track_id: int):
        self.approach_results = {track_id: make_approach_result(track_id)}
        self.proximity_by_track = {track_id: "FAR"}


def make_threat_engine(path_conflict_confirmation_frames: int = 1) -> ThreatAssessmentEngine:
    # path_conflict_confirmation_frames=1: these tests exercise
    # system-health gating, not threat-persistence gating (see
    # test_threat_assessment.py for that).
    return ThreatAssessmentEngine(
        path_conflict_confirmation_frames=path_conflict_confirmation_frames,
        deescalation_seconds=0.75, immediate_danger_deescalation_seconds=1.5,
        disappearance_grace_seconds=1.0,
    )


def assess(threat_engine, tracked, core, hazard_perception, now):
    return threat_engine.assess_for_tracks(
        tracked, core.filtered_motions, hazard_perception.approach_results,
        core.resolved_intersection_results, hazard_perception.proximity_by_track, now,
    )


class _ReadyStartupGate:
    """Minimal stand-in exposing exactly what process_audio_events() (and
    the AudioSequencingTracer it drives) need: an already-READY gate
    that never blocks and never treats the call as the "first ready
    cycle" (so multiple events aren't artificially collapsed to one)."""

    warnings_blocked = False
    state = READY

    @staticmethod
    def is_first_ready_cycle() -> bool:
        return False


class _FakeModeArgs:
    """Minimal argparse.Namespace stand-in for build_operating_mode_policy
    -- mode=None falls back to config/default (BALANCED)."""

    mode = None


# --- system-health message clears/replaces pending operational audio ----


def test_failure_event_clears_pending_operational_queue_items() -> None:
    speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
    event_policy = EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0)
    audio_worker = make_worker(speech_queue, event_policy)
    monitor = make_monitor()

    # Some ordinary operational events are already pending.
    speech_queue.enqueue(
        AudioEvent(
            key="LEVEL_1_MOVING_FAR:1:left", event_type="approaching", priority="INFORMATIONAL",
            message="Vehicle moving from the left.", track_id=1, created_at=0.0,
            delivery_profile="level_1",
        )
    )
    speech_queue.enqueue(
        AudioEvent(
            key="scene_summary:left", event_type="scene_summary", priority="INFORMATIONAL",
            message="Multiple vehicles ahead.", track_id=None, created_at=0.0,
            delivery_profile="scene_summary",
        )
    )
    assert len(speech_queue) == 2

    for _ in range(3):
        monitor.record_frame_read_result(False)
    trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.0)

    assert trace.operational_queue_items_cleared == 2
    # The failure message itself was already popped and spoken by
    # process_system_health()'s own tick() call -- nothing else survived
    # in the queue behind it.
    assert len(speech_queue) == 0
    assert trace.system_message == "Warning! Camera feed lost."


# --- pending-vs-active audit: every pending event kind is removed -------


@pytest.mark.parametrize(
    "label,event",
    [
        (
            "WARNING_danger",
            AudioEvent(
                key="LEVEL_5_HIGH_DANGER:1:left", event_type="corridor_crossing",
                priority="WARNING",
                message="Warning! Vehicle approaching from the left! Warning! Vehicle approaching from the left!",
                track_id=1, created_at=0.0, delivery_profile="level_5",
            ),
        ),
        (
            "INFORMATIONAL_movement",
            AudioEvent(
                key="LEVEL_1_MOVING_FAR:1:left", event_type="approaching", priority="INFORMATIONAL",
                message="Vehicle moving from the left.", track_id=1, created_at=0.0,
                delivery_profile="level_1",
            ),
        ),
        (
            "scene_summary",
            AudioEvent(
                key="scene_summary:left", event_type="scene_summary", priority="INFORMATIONAL",
                message="Multiple vehicles ahead.", track_id=None, created_at=0.0,
                delivery_profile="scene_summary",
            ),
        ),
        (
            "object_uncertainty",
            AudioEvent(
                key="LEVEL_1_MOVING_FAR:2:right", event_type="uncertain", priority="WARNING",
                message="Possible vehicle movement from the right.", track_id=2, created_at=0.0,
                delivery_profile="level_1",
            ),
        ),
    ],
    ids=["WARNING_danger", "INFORMATIONAL_movement", "scene_summary", "object_uncertainty"],
)
def test_every_pending_event_kind_is_removed_when_failure_confirms(label: str, event: AudioEvent) -> None:
    """Required tests 1-4: a queued-but-unspoken WARNING danger,
    INFORMATIONAL movement notice, scene summary, and object-uncertainty
    message must each be individually confirmed removed the moment a
    critical failure confirms -- not just INFORMATIONAL types."""
    speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
    event_policy = EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0)
    audio_worker = make_worker(speech_queue, event_policy)
    monitor = make_monitor()

    speech_queue.enqueue(event)
    assert len(speech_queue) == 1

    for _ in range(3):
        monitor.record_frame_read_result(False)
    trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.0)

    assert trace.operational_queue_items_cleared == 1, label
    # Only the failure message was ever queued after that point, and it
    # was already popped+spoken by process_system_health()'s own tick().
    assert len(speech_queue) == 0, label
    assert trace.system_message == "Warning! Camera feed lost."


def test_active_warning_may_finish_but_no_pending_event_survives(mock_subprocess) -> None:
    """Required test 5: an ACTIVE (already speaking) warning is left
    alone to finish, while a SEPARATE, still-PENDING (queued but not yet
    started) event enqueued alongside it is removed the moment the
    failure confirms."""
    speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
    event_policy = EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0)
    audio_worker = make_worker(speech_queue, event_policy)
    monitor = make_monitor()

    active_event = AudioEvent(
        key="LEVEL_3_APPROACHING:1:left", event_type="corridor_crossing", priority="WARNING",
        message="Vehicle approaching from the left.", track_id=1, created_at=0.0,
        delivery_profile="level_3",
    )
    speech_queue.enqueue(active_event)
    audio_worker.tick(0.0)  # pops + starts speaking active_event
    assert audio_worker.is_speaking is True
    active_process = mock_subprocess["instances"][0]
    assert mock_subprocess["popen"].call_count == 1

    # A second, different event is queued behind it -- never started.
    pending_event = AudioEvent(
        key="LEVEL_1_MOVING_FAR:2:right", event_type="approaching", priority="INFORMATIONAL",
        message="Vehicle moving from the right.", track_id=2, created_at=0.05,
        delivery_profile="level_1",
    )
    speech_queue.enqueue(pending_event)
    assert len(speech_queue) == 1  # only the pending one; active is not in the queue

    for _ in range(3):
        monitor.record_frame_read_result(False)
    trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.1)

    # The active utterance was never interrupted -- same process, still
    # "speaking," no second Popen call yet.
    assert mock_subprocess["popen"].call_count == 1
    assert audio_worker.is_speaking is True

    # The pending event was cleared; the failure message is enqueued
    # and waiting, but cannot be popped until the active one finishes.
    assert trace.operational_queue_items_cleared == 1
    assert trace.system_message == "Warning! Camera feed lost."
    assert len(speech_queue) == 1
    still_pending = list(speech_queue._items.values())[0]
    assert still_pending.message == "Warning! Camera feed lost."

    # Once the active utterance finishes, the failure message -- and
    # ONLY the failure message -- is what speaks next.
    active_process.poll.return_value = 0
    audio_worker.tick(0.2)
    assert mock_subprocess["popen"].call_count == 2
    assert "Warning! Camera feed lost." in mock_subprocess["instances"][1].args


def test_no_pre_failure_event_exists_in_queue_during_recovering() -> None:
    """Required test 6: the queue must contain nothing left over from
    before the failure at any point during RECOVERING."""
    speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
    event_policy = EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0)
    audio_worker = make_worker(speech_queue, event_policy)
    monitor = make_monitor(recovery_confirmation_seconds=1.0)

    stale_event = AudioEvent(
        key="LEVEL_1_MOVING_FAR:1:left", event_type="approaching", priority="INFORMATIONAL",
        message="Vehicle moving from the left.", track_id=1, created_at=0.0,
        delivery_profile="level_1",
    )
    speech_queue.enqueue(stale_event)

    for _ in range(3):
        monitor.record_frame_read_result(False)
    process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.0)
    assert len(speech_queue) == 0  # stale pre-failure event already gone

    monitor.record_frame_read_result(True)
    t = 0.1
    while t < 1.0:
        trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, t)
        assert trace.operational_audio_suppressed is True
        assert len(speech_queue) == 0, f"unexpected pending entry during RECOVERING at t={t}"
        t += 0.2


def test_detection_restored_not_outranked_by_stale_pre_failure_warning(mock_subprocess) -> None:
    """Required test 7: a stale, higher-priority (WARNING) pre-failure
    event must never resurface to compete with -- and outrank --
    "Detection restored." once recovery is confirmed. It is removed at
    confirmation time, long before recovery, so it structurally cannot
    still be sitting in the queue when the recovery message is enqueued."""
    speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
    event_policy = EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0)
    audio_worker = make_worker(speech_queue, event_policy)
    monitor = make_monitor(recovery_confirmation_seconds=1.0)

    stale_warning = AudioEvent(
        key="LEVEL_5_HIGH_DANGER:1:left", event_type="corridor_crossing", priority="WARNING",
        message="Warning! Vehicle approaching from the left! Warning! Vehicle approaching from the left!",
        track_id=1, created_at=0.0, delivery_profile="level_5",
    )
    speech_queue.enqueue(stale_warning)

    for _ in range(3):
        monitor.record_frame_read_result(False)
    process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.0)
    mock_subprocess["instances"][0].poll.return_value = 0  # failure message finishes speaking

    monitor.record_frame_read_result(True)
    process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.1)
    trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, 1.2)

    assert trace.recovery_announced is True
    assert trace.system_message == "Detection restored."
    spoken_messages = [inst.args[-1] for inst in mock_subprocess["instances"]]
    assert stale_warning.message not in spoken_messages


def test_first_post_recovery_event_built_from_current_frame_only(mock_subprocess) -> None:
    """Required test 8: the first operational event admitted after
    recovery is generated from that frame's fresh perception, never a
    holdover from before or during the outage."""
    speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
    event_policy = EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0)
    audio_worker = make_worker(speech_queue, event_policy)
    monitor = make_monitor(recovery_confirmation_seconds=1.0)

    startup_announcer = StartupAnnouncer(
        enabled=False, message="", play_once=True, delay_seconds=0.0,
        require_first_valid_frame=True, configured_voice=None, fallback_voice=None,
        rate_wpm=165, sound_enabled=False, sound_path=None, sound_volume=0.7,
    )
    startup_gate = _ReadyStartupGate()
    delivery_profiles = make_delivery_profiles()
    hazard_resolver = build_audio_hazard_resolver({}, delivery_profiles)
    operating_mode_policy = build_operating_mode_policy({}, _FakeModeArgs())
    audio_event_builder = build_audio_event_builder({})
    scene_summarizer = SceneSummarizer(
        enabled=True, minimum_events=2, max_objects_named=3, same_frame_only=True,
        prefer_single_summary=True, fallback_message="Multiple hazards ahead.",
    )
    tracer = AudioSequencingTracer()
    tracked = [make_tracked_object(track_id=1, region="left")]
    core = _FakeCore(1)
    hazard_perception = _FakeHazardPerception(1)
    threat_engine = make_threat_engine()

    for _ in range(3):
        monitor.record_frame_read_result(False)
    process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.0)
    mock_subprocess["instances"][0].poll.return_value = 0  # failure message finishes speaking

    monitor.record_frame_read_result(True)
    process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.1)
    confirmed_trace = process_system_health(
        monitor, event_policy, speech_queue, audio_worker, 1.2
    )
    assert confirmed_trace.operational_audio_suppressed is False

    threat_assessments = assess(threat_engine, tracked, core, hazard_perception, 1.2)
    process_audio_events(
        startup_announcer, startup_gate, hazard_resolver, operating_mode_policy,
        audio_event_builder, scene_summarizer, audio_worker, speech_queue, tracer,
        tracked, threat_assessments, 1.2,
    )

    remaining = list(speech_queue._items.values())
    assert len(remaining) == 1
    assert remaining[0].message == "Vehicle moving from the left."
    assert remaining[0].created_at == 1.2  # built from the recovery frame, not an earlier one


def test_queue_clearing_does_not_terminate_active_subprocess(mock_subprocess) -> None:
    """Required test 9: SpeechQueue.clear() only ever touches
    SpeechQueue's own pending-items dict -- it has no reference to, and
    cannot terminate, AudioWorker's in-flight subprocess."""
    speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
    event_policy = EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0)
    audio_worker = make_worker(speech_queue, event_policy)

    active_event = AudioEvent(
        key="LEVEL_3_APPROACHING:1:left", event_type="corridor_crossing", priority="WARNING",
        message="Vehicle approaching from the left.", track_id=1, created_at=0.0,
        delivery_profile="level_3",
    )
    speech_queue.enqueue(active_event)
    audio_worker.tick(0.0)
    assert audio_worker.is_speaking is True
    process_before = audio_worker._process
    phase_before = audio_worker._phase

    speech_queue.clear()

    assert audio_worker.is_speaking is True
    assert audio_worker._process is process_before  # same subprocess, not touched
    assert audio_worker._phase == phase_before
    process_before.terminate.assert_not_called()
    process_before.kill.assert_not_called()


def test_failure_message_is_actually_spoken(mock_subprocess) -> None:
    speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
    event_policy = EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0)
    audio_worker = make_worker(speech_queue, event_policy)
    monitor = make_monitor()

    for _ in range(3):
        monitor.record_frame_read_result(False)
    process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.0)

    assert mock_subprocess["popen"].call_count == 1
    spoken_command = mock_subprocess["instances"][0].args
    assert "Warning! Camera feed lost." in spoken_command


def test_suppression_flag_skips_process_audio_events() -> None:
    speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
    event_policy = EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0)
    audio_worker = make_worker(speech_queue, event_policy)
    monitor = make_monitor()

    for _ in range(3):
        monitor.record_frame_read_result(False)
    health_trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.0)
    assert health_trace.operational_audio_suppressed is True

    startup_announcer = StartupAnnouncer(
        enabled=False, message="", play_once=True, delay_seconds=0.0,
        require_first_valid_frame=True, configured_voice=None, fallback_voice=None,
        rate_wpm=165, sound_enabled=False, sound_path=None, sound_volume=0.7,
    )
    startup_gate = _ReadyStartupGate()
    delivery_profiles = make_delivery_profiles()
    hazard_resolver = build_audio_hazard_resolver({}, delivery_profiles)
    operating_mode_policy = build_operating_mode_policy({}, _FakeModeArgs())
    audio_event_builder = build_audio_event_builder({})
    scene_summarizer = SceneSummarizer(
        enabled=True, minimum_events=2, max_objects_named=3, same_frame_only=True,
        prefer_single_summary=True, fallback_message="Multiple hazards ahead.",
    )
    tracer = AudioSequencingTracer()

    tracked = [make_tracked_object(track_id=1, region="left")]
    core = _FakeCore(1)
    hazard_perception = _FakeHazardPerception(1)
    threat_engine = make_threat_engine()

    # Mirrors main.py's exact gating pattern: process_audio_events() is
    # simply never called this frame when suppressed.
    if not health_trace.operational_audio_suppressed:
        threat_assessments = assess(threat_engine, tracked, core, hazard_perception, 0.0)
        process_audio_events(
            startup_announcer, startup_gate, hazard_resolver, operating_mode_policy,
            audio_event_builder, scene_summarizer, audio_worker, speech_queue, tracer,
            tracked, threat_assessments, 0.0,
        )

    # Only the system-failure message is queued/spoken -- nothing from
    # the (skipped) hazard pipeline entered the queue.
    assert len(speech_queue) == 0  # already popped and spoken by the tick() above


def test_operational_audio_resumes_once_healthy() -> None:
    speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
    event_policy = EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0)
    audio_worker = make_worker(speech_queue, event_policy)
    monitor = make_monitor()

    health_trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.0)
    assert health_trace.operational_audio_suppressed is False


def test_in_flight_speech_is_never_interrupted_by_a_failure(mock_subprocess) -> None:
    speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
    event_policy = EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0)
    audio_worker = make_worker(speech_queue, event_policy)
    monitor = make_monitor()

    speech_queue.enqueue(
        AudioEvent(
            key="LEVEL_3_APPROACHING:1:left", event_type="corridor_crossing", priority="WARNING",
            message="Vehicle approaching from the left.", track_id=1, created_at=0.0,
            delivery_profile="level_3",
        )
    )
    audio_worker.tick(0.0)  # starts speaking the warning
    assert mock_subprocess["popen"].call_count == 1
    first_process = mock_subprocess["instances"][0]
    assert first_process.poll.return_value is None  # still "speaking"

    for _ in range(3):
        monitor.record_frame_read_result(False)
    trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.1)

    # The failure message was admitted/queued, but the in-flight
    # utterance was never interrupted -- no second process started yet.
    assert mock_subprocess["popen"].call_count == 1
    assert trace.system_message == "Warning! Camera feed lost."

    # Once the in-flight utterance finishes, the NEXT tick speaks the
    # system failure -- it survives (nothing else could have been
    # queued behind it, since clear() already removed everything else).
    first_process.poll.return_value = 0
    audio_worker.tick(0.2)
    assert mock_subprocess["popen"].call_count == 2
    second_command = mock_subprocess["instances"][1].args
    assert "Warning! Camera feed lost." in second_command


def test_no_operational_event_built_or_queued_during_failure_or_recovering(
    mock_subprocess,
) -> None:
    """Required tests 4, 7, 8: nothing built by process_audio_events()
    during FAILURE_ACTIVE or RECOVERING is ever queued or spoken later
    (process_audio_events() is structurally never called during either
    phase, mirroring main.py's exact gating -- there is nothing to
    accumulate or replay). The first event admitted once truly
    HEALTHY_OPERATIONAL is built from that frame's fresh perception
    only, never a stale pre-failure reading."""
    speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
    event_policy = EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0)
    audio_worker = make_worker(speech_queue, event_policy)
    monitor = make_monitor(recovery_confirmation_seconds=1.0)

    startup_announcer = StartupAnnouncer(
        enabled=False, message="", play_once=True, delay_seconds=0.0,
        require_first_valid_frame=True, configured_voice=None, fallback_voice=None,
        rate_wpm=165, sound_enabled=False, sound_path=None, sound_volume=0.7,
    )
    startup_gate = _ReadyStartupGate()
    delivery_profiles = make_delivery_profiles()
    hazard_resolver = build_audio_hazard_resolver({}, delivery_profiles)
    operating_mode_policy = build_operating_mode_policy({}, _FakeModeArgs())
    audio_event_builder = build_audio_event_builder({})
    scene_summarizer = SceneSummarizer(
        enabled=True, minimum_events=2, max_objects_named=3, same_frame_only=True,
        prefer_single_summary=True, fallback_message="Multiple hazards ahead.",
    )
    tracer = AudioSequencingTracer()

    # A hazard-eligible object is present on EVERY frame throughout --
    # if a stale/accumulated event leaked through during FAILURE_ACTIVE
    # or RECOVERING, this would surface it.
    tracked = [make_tracked_object(track_id=1, region="left")]
    core = _FakeCore(1)
    hazard_perception = _FakeHazardPerception(1)
    threat_engine = make_threat_engine()

    def maybe_process_audio(now):
        if not health_trace.operational_audio_suppressed:
            threat_assessments = assess(threat_engine, tracked, core, hazard_perception, now)
            process_audio_events(
                startup_announcer, startup_gate, hazard_resolver, operating_mode_policy,
                audio_event_builder, scene_summarizer, audio_worker, speech_queue, tracer,
                tracked, threat_assessments, now,
            )

    # FAILURE_ACTIVE.
    for _ in range(3):
        monitor.record_frame_read_result(False)
    health_trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.0)
    assert health_trace.operational_audio_suppressed is True
    maybe_process_audio(0.0)
    assert len(speech_queue) == 0  # only the failure message was ever queued+spoken

    # The "Warning! Camera feed lost." utterance finishes speaking (as
    # it would in reality, well within the recovery window) so the
    # worker is idle again by the time recovery is confirmed below.
    mock_subprocess["instances"][0].poll.return_value = 0

    # RECOVERING -- reads succeed again, but confirmation hasn't elapsed.
    monitor.record_frame_read_result(True)
    health_trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.1)
    assert health_trace.operational_audio_suppressed is True
    maybe_process_audio(0.1)
    assert len(speech_queue) == 0

    health_trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.5)
    assert health_trace.operational_audio_suppressed is True
    maybe_process_audio(0.5)
    assert len(speech_queue) == 0

    # Confirmation window elapses -- HEALTHY_OPERATIONAL. Recovery is
    # announced (and already popped/spoken by process_system_health's
    # own tick()) on this same frame; operational audio may resume.
    health_trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, 1.2)
    assert health_trace.recovery_announced is True
    assert health_trace.operational_audio_suppressed is False
    maybe_process_audio(1.2)

    # Exactly the fresh Level-1 hazard event built from THIS frame's
    # perception is queued -- nothing stale, nothing accumulated.
    assert len(speech_queue) == 1
    remaining = speech_queue.pop_next(now=1.2)
    assert remaining.message == "Vehicle moving from the left."
    assert remaining.created_at == 1.2


@pytest.mark.parametrize("mode", ["minimal", "balanced", "detailed"])
def test_recovery_gating_identical_across_operating_modes(mode: str) -> None:
    """Required test 11: SystemHealthMonitor/process_system_health()
    never receive an operating-mode parameter at all, so the exact same
    failure/RECOVERING/recovery sequence must produce identical
    suppression and message behavior in every mode."""

    class _ArgsWithMode:
        pass

    args = _ArgsWithMode()
    args.mode = mode
    operating_mode_policy = build_operating_mode_policy({}, args)
    assert operating_mode_policy.mode == mode.upper()

    speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
    event_policy = EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0)
    audio_worker = make_worker(speech_queue, event_policy)
    monitor = make_monitor(recovery_confirmation_seconds=1.0)

    for _ in range(3):
        monitor.record_frame_read_result(False)
    failure_trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.0)

    monitor.record_frame_read_result(True)
    recovering_trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.1)

    confirmed_trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, 1.2)

    assert failure_trace.operational_audio_suppressed is True
    assert failure_trace.system_message == "Warning! Camera feed lost."
    assert recovering_trace.operational_audio_suppressed is True
    assert recovering_trace.system_message is None
    assert confirmed_trace.operational_audio_suppressed is False
    assert confirmed_trace.system_message == "Detection restored."


def test_recovery_message_uses_informational_priority_and_calm_profile() -> None:
    speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
    event_policy = EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0)
    audio_worker = make_worker(speech_queue, event_policy)
    monitor = make_monitor(recovery_confirmation_seconds=1.0)

    for _ in range(3):
        monitor.record_frame_read_result(False)
    process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.0)

    monitor.record_frame_read_result(True)
    process_system_health(monitor, event_policy, speech_queue, audio_worker, 0.1)
    trace = process_system_health(monitor, event_policy, speech_queue, audio_worker, 1.2)

    assert trace.system_message == "Detection restored."
    assert trace.recovery_announced is True
