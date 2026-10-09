"""Test5 acceptance suite: proves -- from real runtime state and
timestamps, not by listening to a recording -- that the cinematic
startup announcement and object-warning speech never overlap.

Uses real StartupAnnouncer/StartupAudioGate/SpeechQueue/EventPolicy/
AudioWorker/AudioSequencingTracer/AudioEventBuilder/SceneSummarizer
driven through main.py's actual process_audio_events(), with only
`subprocess` mocked (never real `say`). subprocess.Popen is a single
shared module-level attribute used by BOTH src/audio/startup_announcer.py
and src/audio/audio_worker.py (they import the same `subprocess` module
object), so it is patched exactly ONCE per test via the autouse fixture
below, with a side_effect that hands back a fresh, independently
controllable MagicMock instance per call -- patching it twice with two
different mocks would silently make the second patch win for both
call sites and produce false negatives.
"""

from unittest.mock import MagicMock

import pytest

from main import (
    CoreMotionResult,
    build_approach_estimator,
    build_audio_hazard_resolver,
    build_delivery_profiles,
    build_operating_mode_policy,
    build_path_intersection_analyzer,
    build_relative_proximity_estimator,
    build_threat_assessment_engine,
    compute_hazard_perception,
    process_audio_events,
)
from src.audio.audio_event_builder import AudioEventBuilder
from src.audio.audio_sequencing_tracer import ADMITTED, SUPPRESSED_STARTUP_EXCLUSIVE, AudioSequencingTracer
from src.audio.audio_worker import AudioWorker
from src.audio.event_policy import EventPolicy
from src.audio.scene_summarizer import SceneSummarizer
from src.audio.speech_queue import SpeechQueue
from src.audio.startup_announcer import StartupAnnouncer
from src.audio.startup_audio_gate import INITIALIZING, READY, STARTUP_SPEAKING, StartupAudioGate
from src.models import (
    BoundingBox,
    FilteredMotion,
    PathIntersectionResult,
    TrackedObject,
    TrajectoryPrediction,
)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# build_path_intersection_analyzer requires an explicit corridor section
# (unlike every other build_* factory here, which defaults gracefully) --
# matches config/settings.yaml's own default corridor geometry.
_CONFIG = {
    "corridor": {
        "corridor_bottom_left_x": 0.35,
        "corridor_bottom_right_x": 0.65,
        "corridor_bottom_y": 1.0,
        "corridor_top_left_x": 0.45,
        "corridor_top_right_x": 0.55,
        "corridor_top_y": 0.55,
    }
}


class _FakeModeArgs:
    """Minimal argparse.Namespace stand-in for build_operating_mode_policy
    -- mode=None means "not set on the CLI," falling back to config/
    default (BALANCED), which allows this file's "car"-class test
    objects regardless (traffic is always eligible in every mode)."""

    mode = None


def _voices_stdout(names) -> str:
    return "\n".join(f"{name}               en_US    # Hello, I'm {name}." for name in names)


@pytest.fixture(autouse=True)
def mock_subprocess(monkeypatch):
    run_result = MagicMock()
    run_result.stdout = _voices_stdout(("Samantha", "Karen"))
    mock_run = MagicMock(return_value=run_result)

    instances: list = []

    def make_popen(*args, **kwargs):
        instance = MagicMock()
        instance.poll.return_value = None  # "still running" by default
        instances.append(instance)
        return instance

    mock_popen = MagicMock(side_effect=make_popen)

    monkeypatch.setattr("src.audio.startup_announcer.subprocess.run", mock_run)
    monkeypatch.setattr("src.audio.startup_announcer.subprocess.Popen", mock_popen)
    monkeypatch.setattr("src.audio.audio_worker.subprocess.Popen", mock_popen)

    return {"run": mock_run, "popen": mock_popen, "instances": instances}


def make_announcer(delay_seconds: float = 0.0) -> StartupAnnouncer:
    return StartupAnnouncer(
        enabled=True,
        message="Atlas online. Detection system activated.",
        play_once=True,
        delay_seconds=delay_seconds,
        require_first_valid_frame=True,
        configured_voice=None,
        fallback_voice=None,
        rate_wpm=165,
        sound_enabled=False,
        sound_path=None,
        sound_volume=0.7,
    )


def advance_ready(announcer: StartupAnnouncer, monkeypatch) -> None:
    announcer.mark_camera_ready()
    announcer.notify_frame_read()
    future = announcer._ready_since + announcer._delay_seconds + 0.01
    monkeypatch.setattr("src.audio.startup_announcer.time.time", lambda: future)


def make_worker(speech_queue: SpeechQueue, delivery_profiles: dict) -> AudioWorker:
    policy = EventPolicy(
        cooldown_same_event_seconds=4.0,
        cooldown_informational_seconds=8.0,
    )
    return AudioWorker(
        enabled=True, queue=speech_queue, event_policy=policy,
        configured_voice=None, fallback_voice=None, rate_wpm=170,
        delivery_profiles=delivery_profiles,
    )


def make_summarizer() -> SceneSummarizer:
    return SceneSummarizer(
        enabled=True, minimum_events=2, max_objects_named=3, same_frame_only=True,
        prefer_single_summary=True,
        fallback_message="Multiple hazards ahead. Please wait.",
    )


def make_tracked_object(track_id: int = 1, region: str = "left") -> TrackedObject:
    bbox = BoundingBox(x1=0, y1=0, x2=20, y2=20)
    return TrackedObject(
        track_id=track_id, class_id=2, class_name="car", confidence=0.9,
        bbox=bbox, center=(10, 10), region=region, position_history=((10, 10),),
        size_history=((20, 20),), direction="right", motion_status="moving",
        frames_since_seen=0,
    )


def make_trajectory_prediction(track_id: int, center: tuple[int, int] = (10, 10)) -> TrajectoryPrediction:
    return TrajectoryPrediction(
        valid=True, velocity_x=5.0, velocity_y=0.0, speed_px_per_frame=5.0,
        direction="right", current_center=center, predicted_center=(center[0] + 50, center[1]),
        observations_used=1, prediction_horizon_frames=10, uncertain=False,
    )


def make_intersection_result(track_id: int) -> PathIntersectionResult:
    return PathIntersectionResult(
        valid=True, intersects=False, starts_inside=False, ends_inside=False,
        intersection_point=None, track_id=track_id, reason="test", uncertain=False,
    )


def make_core_with_moving_object(track_id: int = 1) -> CoreMotionResult:
    filtered = {
        track_id: FilteredMotion(
            track_id=track_id, velocity_x=5.0, velocity_y=0.0, speed=5.0,
            motion_state="MOVING", source="COMPENSATED", uncertain=False,
            reason=None, confirmation_frames=0,
        )
    }
    return CoreMotionResult(
        predictions={}, resolved_motions={}, filtered_motions=filtered,
        resolved_predictions={track_id: make_trajectory_prediction(track_id)},
        resolved_intersection_results={track_id: make_intersection_result(track_id)},
    )


EMPTY_CORE = CoreMotionResult(
    predictions={}, resolved_motions={}, filtered_motions={},
    resolved_predictions={}, resolved_intersection_results={},
)


class _Harness:
    """Bundles one full, real (subprocess-mocked) audio-sequencing stack
    for a test -- avoids repeating the same construction in every test
    below. Uses main.py's own build_* factories (empty config -> every
    documented default) for the hazard pipeline, so this is a genuine
    integration test of the real wiring, not a hand-rolled approximation."""

    def __init__(self, delay_seconds: float = 0.0) -> None:
        config: dict = _CONFIG
        self.announcer = make_announcer(delay_seconds)
        self.gate = StartupAudioGate(self.announcer)
        self.speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
        self.delivery_profiles = build_delivery_profiles(config)
        self.worker = make_worker(self.speech_queue, self.delivery_profiles)
        self.tracer = AudioSequencingTracer()
        self.builder = AudioEventBuilder()
        self.summarizer = make_summarizer()
        self.proximity_estimator = build_relative_proximity_estimator(config)
        self.approach_estimator = build_approach_estimator(config, self.proximity_estimator)
        self.hazard_resolver = build_audio_hazard_resolver(config, self.delivery_profiles)
        self.path_intersection_analyzer = build_path_intersection_analyzer(config)
        self.mode_policy = build_operating_mode_policy(config, _FakeModeArgs())
        # path_conflict_confirmation_frames=1: this suite exercises audio
        # sequencing/timing, not threat-persistence gating (that's
        # test_threat_assessment.py's job) -- same pattern used in
        # test_hazard_levels_integration.py / test_audio_hazard_resolver.py.
        self.threat_engine = build_threat_assessment_engine(
            {"threat_assessment": {"path_conflict_confirmation_frames": 1}}
        )

    def process(self, tracked_objects, core, timestamp):
        hazard_perception = compute_hazard_perception(
            tracked_objects, core, self.proximity_estimator, self.approach_estimator,
            self.path_intersection_analyzer, FRAME_WIDTH, FRAME_HEIGHT,
        )
        threat_assessments = self.threat_engine.assess_for_tracks(
            tracked_objects, core.filtered_motions, hazard_perception.approach_results,
            core.resolved_intersection_results, hazard_perception.proximity_by_track,
            timestamp,
        )
        return process_audio_events(
            self.announcer, self.gate, self.hazard_resolver, self.mode_policy, self.builder,
            self.summarizer, self.worker, self.speech_queue, self.tracer,
            tracked_objects, threat_assessments, timestamp,
        )


# --- 1/8: startup and warning speech never overlap; exactly one subprocess ---


def test_startup_and_warning_speech_never_overlap(monkeypatch, mock_subprocess) -> None:
    h = _Harness()
    obj = make_tracked_object(1, region="left")
    core = make_core_with_moving_object(1)

    advance_ready(h.announcer, monkeypatch)
    h.announcer.tick()  # starts startup speech -- instances[0], poll=None

    trace = h.process([obj], core, 0.0)
    assert trace.startup_state == STARTUP_SPEAKING
    assert trace.active_speech_process_count <= 1
    assert trace.warning_pipeline_enabled is False
    mock_subprocess["popen"].assert_called_once()  # only startup fired so far

    mock_subprocess["instances"][0].poll.return_value = 0  # startup finishes

    trace2 = h.process([obj], core, 1.0)
    assert trace2.startup_state == READY
    assert trace2.active_speech_process_count <= 1
    assert mock_subprocess["popen"].call_count == 2  # the warning has now fired
    # At the boundary frame: startup's own process is finished, only the
    # warning's process may be running -- never both simultaneously.
    assert mock_subprocess["instances"][0].poll() is not None
    assert mock_subprocess["instances"][1].poll() is None


# --- 2/7: no warnings enter (or ever reach) the queue while startup active ---


def test_no_warnings_enter_queue_while_startup_active(monkeypatch, mock_subprocess) -> None:
    h = _Harness()
    obj = make_tracked_object(1, region="left")
    core = make_core_with_moving_object(1)  # would produce a real warning once ready

    advance_ready(h.announcer, monkeypatch)
    h.announcer.tick()

    for i in range(5):
        trace = h.process([obj], core, float(i))
        assert trace.queue_size_during_startup == 0
        assert len(h.speech_queue) == 0
        assert h.worker.is_speaking is False

    mock_subprocess["popen"].assert_called_once()  # only the startup line


# --- 3/10: perception continues; mode-independence (structural) -----------


def test_process_audio_events_cannot_gate_perception_or_depend_on_mode() -> None:
    import inspect

    params = set(inspect.signature(process_audio_events).parameters)
    assert not params & {"detector", "tracker", "motion_estimator", "video", "args", "debug", "mode"}

    gate_params = set(inspect.signature(StartupAudioGate.__init__).parameters)
    assert not gate_params & {"args", "debug", "mode"}


# --- 4: events observed during startup are discarded permanently ----------


def test_pre_ready_events_are_never_created_or_replayed(monkeypatch, mock_subprocess) -> None:
    h = _Harness()
    build_calls = []
    original_build_events = h.builder.build_events

    def spying_build_events(*args, **kwargs):
        build_calls.append(args[-1])  # timestamp
        return original_build_events(*args, **kwargs)

    h.builder.build_events = spying_build_events
    obj = make_tracked_object(1, region="left")
    core = make_core_with_moving_object(1)

    advance_ready(h.announcer, monkeypatch)
    h.announcer.tick()

    for i in range(5):
        trace = h.process([obj], core, float(i))
        assert trace.candidate_created_during_startup is False

    assert build_calls == []  # never invoked while blocked -- nothing to replay

    mock_subprocess["instances"][0].poll.return_value = 0
    trace = h.process([obj], core, 10.0)
    assert build_calls == [10.0]  # first call, current frame only


# --- 5: READY occurs only after actual subprocess completion --------------


def test_ready_only_after_actual_subprocess_completion(monkeypatch, mock_subprocess) -> None:
    h = _Harness()
    advance_ready(h.announcer, monkeypatch)
    h.announcer.tick()

    trace = h.process([], EMPTY_CORE, 0.0)
    assert trace.startup_state == STARTUP_SPEAKING
    assert trace.startup_speech_finish_time is None

    mock_subprocess["instances"][0].poll.return_value = 0  # actual completion

    trace2 = h.process([], EMPTY_CORE, 5.0)
    assert trace2.startup_state == READY
    assert trace2.startup_speech_finish_time == 5.0


# --- 6: a fresh warning after READY can speak ------------------------------


def test_fresh_warning_after_ready_can_speak(monkeypatch, mock_subprocess) -> None:
    h = _Harness()
    obj = make_tracked_object(1, region="left")
    core = make_core_with_moving_object(1)

    advance_ready(h.announcer, monkeypatch)
    h.announcer.tick()
    mock_subprocess["instances"][0].poll.return_value = 0  # startup already done

    trace = h.process([obj], core, 0.0)

    assert trace.startup_state == READY
    assert trace.first_warning_speech_start_time == 0.0
    assert mock_subprocess["popen"].call_count == 2  # startup + this warning


# --- 9: quitting during startup terminates cleanly -------------------------


def test_shutdown_during_startup_terminates_cleanly(monkeypatch, mock_subprocess) -> None:
    h = _Harness()
    advance_ready(h.announcer, monkeypatch)
    h.announcer.tick()  # startup speaking

    h.announcer.shutdown()
    h.worker.shutdown()  # never spoke -- must be a safe no-op

    mock_subprocess["instances"][0].terminate.assert_called_once()


# --- 11: muted audit mode -- deterministic, never invokes real `say` ------


def simulate_startup_completion(announcer: StartupAnnouncer, gate: StartupAudioGate, monkeypatch) -> str:
    """Muted-audit helper: deterministically drives the gate to READY
    using only the mocked subprocess/clock -- no real `say` process, no
    real sleeping, fully reproducible."""
    advance_ready(announcer, monkeypatch)
    announcer.tick()
    if announcer._process is not None:
        announcer._process.poll.return_value = 0
    return gate.state


def test_muted_audit_mode_simulates_startup_completion_deterministically(
    monkeypatch, mock_subprocess
) -> None:
    announcer = make_announcer()
    gate = StartupAudioGate(announcer)

    state = simulate_startup_completion(announcer, gate, monkeypatch)

    assert state == READY
    for instance in mock_subprocess["instances"]:
        assert isinstance(instance, MagicMock)  # never a real subprocess.Popen result
    mock_subprocess["run"].assert_called()  # only the mocked voice query


# --- gate decision/reason vocabulary (supports invariant D/the trace fields) --


def test_gate_decision_and_reason_fields() -> None:
    h = _Harness()

    trace_blocked = h.process([], EMPTY_CORE, 0.0)
    assert trace_blocked.startup_state == INITIALIZING
    assert trace_blocked.startup_gate_decision == SUPPRESSED_STARTUP_EXCLUSIVE
    assert trace_blocked.startup_gate_reason == "startup_state=INITIALIZING"

    disabled_announcer = StartupAnnouncer(
        enabled=False, message="x", play_once=True, delay_seconds=0.0,
        require_first_valid_frame=True, configured_voice=None, fallback_voice=None,
        rate_wpm=None, sound_enabled=False, sound_path=None, sound_volume=0.7,
    )
    disabled_gate = StartupAudioGate(disabled_announcer)
    config: dict = _CONFIG
    speech_queue = SpeechQueue(5, 10.0)
    delivery_profiles = build_delivery_profiles(config)
    worker = make_worker(speech_queue, delivery_profiles)
    tracer = AudioSequencingTracer()
    hazard_resolver = build_audio_hazard_resolver(config, delivery_profiles)
    mode_policy = build_operating_mode_policy(config, _FakeModeArgs())
    trace_ready = process_audio_events(
        disabled_announcer, disabled_gate, hazard_resolver, mode_policy, AudioEventBuilder(),
        make_summarizer(), worker, speech_queue, tracer, [], {}, 0.0,
    )
    assert trace_ready.startup_gate_decision == ADMITTED
    assert trace_ready.startup_gate_reason == f"startup_state={READY}"


# --- E: first-ready-cycle single-event admission limit ---------------------


def test_first_ready_cycle_admits_at_most_one_highest_priority_event(
    monkeypatch, mock_subprocess
) -> None:
    h = _Harness()
    objects = [make_tracked_object(1, region="left"), make_tracked_object(2, region="right")]
    filtered = {
        1: FilteredMotion(track_id=1, velocity_x=5.0, velocity_y=0.0, speed=5.0,
                           motion_state="MOVING", source="COMPENSATED", uncertain=False,
                           reason=None, confirmation_frames=0),
        2: FilteredMotion(track_id=2, velocity_x=5.0, velocity_y=0.0, speed=5.0,
                           motion_state="MOVING", source="COMPENSATED", uncertain=False,
                           reason=None, confirmation_frames=0),
    }
    core = CoreMotionResult(
        predictions={}, resolved_motions={}, filtered_motions=filtered,
        resolved_predictions={
            1: make_trajectory_prediction(1), 2: make_trajectory_prediction(2)
        },
        resolved_intersection_results={
            1: make_intersection_result(1), 2: make_intersection_result(2)
        },
    )

    advance_ready(h.announcer, monkeypatch)
    h.announcer.tick()
    mock_subprocess["instances"][0].poll.return_value = 0

    h.process(objects, core, 0.0)

    # Two candidate events existed (both "moving", different objects/
    # regions -- SceneSummarizer's minimum_events=2 bucketing requires
    # the SAME region to merge, so these stay two distinct events);
    # the first-ready-cycle limit must have kept only one queued/spoken.
    assert mock_subprocess["popen"].call_count == 2  # startup + exactly one warning
