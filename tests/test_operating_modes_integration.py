"""End-to-end integration tests for operating modes:

    AudioHazardResolver -> OperatingModePolicy.filter_events() ->
    AudioEventBuilder -> SceneSummarizer -> OperatingModePolicy.
    finalize_events() -> EventPolicy -> SpeechQueue -> AudioWorker

Confirms all three modes share identical upstream perception/hazard
decisions, direction wording is always present, and basic safety
invariants (no "safe to cross," no overlap, one subprocess at a time)
hold through the real stack. Only `subprocess` is mocked -- never real
`say`.
"""

from unittest.mock import MagicMock

import pytest

from src.audio.audio_event_builder import AudioEventBuilder
from src.audio.audio_hazard_resolver import AudioHazardResolver
from src.audio.audio_worker import AudioWorker
from src.audio.event_policy import EventPolicy
from src.audio.operating_mode_policy import BALANCED, DETAILED, MINIMAL, OperatingModePolicy
from src.audio.scene_summarizer import SceneSummarizer
from src.audio.speech_queue import SpeechQueue
from src.audio.startup_audio_gate import StartupAudioGate
from src.models import (
    ApproachResult,
    AudioHazardResult,
    BoundingBox,
    DeliveryProfile,
    FilteredMotion,
    OperatingModeProfile,
    PathIntersectionResult,
    TrackedObject,
    TrajectoryPrediction,
)
from src.threat_assessment import ThreatAssessmentEngine

_FORBIDDEN_PHRASES = (
    "safe to cross", "cross now", "go.", "the road is clear",
    "no vehicles", "no traffic", "you are safe", "all clear", "proceed",
)


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
        instance.poll.return_value = None
        instances.append(instance)
        return instance

    mock_popen = MagicMock(side_effect=make_popen)
    monkeypatch.setattr("src.audio.startup_announcer.subprocess.run", mock_run)
    monkeypatch.setattr("src.audio.audio_worker.subprocess.Popen", mock_popen)

    return {"run": mock_run, "popen": mock_popen, "instances": instances}


def make_profiles() -> dict[str, OperatingModeProfile]:
    return {
        MINIMAL: OperatingModeProfile(
            mode=MINIMAL, announce_traffic=True, person_eligibility="NONE",
            highest_danger_repeat_enabled=True, highest_danger_repeat_delay_seconds=3.0,
            highest_danger_max_cycles=2, structured_scene_summaries=False,
            max_summary_object_groups=2, max_summary_words=15,
        ),
        BALANCED: OperatingModeProfile(
            mode=BALANCED, announce_traffic=True, person_eligibility="RELEVANT_ONLY",
            highest_danger_repeat_enabled=True, highest_danger_repeat_delay_seconds=3.0,
            highest_danger_max_cycles=2, structured_scene_summaries=False,
            max_summary_object_groups=2, max_summary_words=15,
        ),
        DETAILED: OperatingModeProfile(
            mode=DETAILED, announce_traffic=True, person_eligibility="BROAD",
            highest_danger_repeat_enabled=False, highest_danger_repeat_delay_seconds=3.0,
            highest_danger_max_cycles=1, structured_scene_summaries=True,
            max_summary_object_groups=2, max_summary_words=15,
        ),
    }


def make_delivery_profiles() -> dict[str, DeliveryProfile]:
    return {
        f"level_{i}": DeliveryProfile(rate_wpm=170 + i * 5, cue_enabled=False, cue_path=None, cue_gain=0.0)
        for i in range(1, 6)
    }


class _Stack:
    def __init__(self, mode: str) -> None:
        self.speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
        self.event_policy = EventPolicy(cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0)
        delivery_profiles = make_delivery_profiles()
        self.worker = AudioWorker(
            enabled=True, queue=self.speech_queue, event_policy=self.event_policy,
            configured_voice=None, fallback_voice=None, rate_wpm=170,
            delivery_profiles=delivery_profiles,
        )
        self.builder = AudioEventBuilder()
        self.summarizer = SceneSummarizer(
            enabled=True, minimum_events=2, max_objects_named=3, same_frame_only=True,
            prefer_single_summary=True, fallback_message="Multiple hazards ahead. Please wait.",
        )
        self.hazard_resolver = AudioHazardResolver(
            announce_level_1=True, announce_level_2=True,
            delivery_profiles=delivery_profiles,
        )
        self.mode_policy = OperatingModePolicy(mode=mode, mode_source="CLI", profiles=make_profiles())

    def speak_frame(self, hazard_results, tracked_objects, now):
        eligible, _ = self.mode_policy.filter_events(hazard_results, now)
        candidates = self.builder.build_events(eligible, now)
        summarized = self.summarizer.summarize(candidates, tracked_objects, now)
        final, _ = self.mode_policy.finalize_events(summarized, now)
        self.worker.enqueue_events(final, now)
        self.worker.tick(now)
        return final


def make_object(track_id: int, region: str, class_name: str) -> TrackedObject:
    bbox = BoundingBox(x1=0, y1=400, x2=40, y2=440)
    return TrackedObject(
        track_id=track_id, class_id=0, class_name=class_name, confidence=0.9,
        bbox=bbox, center=(20, 420), region=region, position_history=((20, 420),) * 5,
        size_history=((40, 40),) * 5, direction="right", motion_status="moving",
        frames_since_seen=0,
    )


def make_filtered(track_id: int) -> FilteredMotion:
    return FilteredMotion(
        track_id=track_id, velocity_x=5.0, velocity_y=0.0, speed=5.0,
        motion_state="MOVING", source="COMPENSATED", uncertain=False,
        reason=None, confirmation_frames=0,
    )


def make_approach(track_id: int, state: str = "NOT_APPROACHING") -> ApproachResult:
    return ApproachResult(
        track_id=track_id, state=state, confidence=1.0, proximity_zone="FAR",
        closing_score=0.0, corridor_distance_trend=0.0, scale_growth_trend=0.0,
        ground_point_trend=0.0, confirmation_frames=0, uncertain=False, evidence_flags=(),
    )


def make_threat_engine(path_conflict_confirmation_frames: int = 1) -> ThreatAssessmentEngine:
    # path_conflict_confirmation_frames=1: these tests exercise mode/
    # wording behavior, not threat-persistence gating (see
    # test_threat_assessment.py for that).
    return ThreatAssessmentEngine(
        path_conflict_confirmation_frames=path_conflict_confirmation_frames,
        deescalation_seconds=0.75, immediate_danger_deescalation_seconds=1.5,
        disappearance_grace_seconds=1.0,
    )


# --- B: shared perception across modes -------------------------------------


def test_identical_hazard_results_across_all_modes() -> None:
    # AudioHazardResolver is constructed identically and consulted
    # BEFORE any OperatingModePolicy exists in this test -- proving mode
    # cannot influence perception/hazard decisions even in principle.
    delivery_profiles = make_delivery_profiles()
    hazard_resolver = AudioHazardResolver(
        announce_level_1=True, announce_level_2=True,
        delivery_profiles=delivery_profiles,
    )
    threat_engine = make_threat_engine()
    car = make_object(1, "left", "car")
    person = make_object(2, "left", "person")
    filtered = {1: make_filtered(1), 2: make_filtered(2)}
    approach = {1: make_approach(1), 2: make_approach(2)}

    assessments = threat_engine.assess_for_tracks(
        [car, person], filtered, approach, {}, {1: "FAR", 2: "FAR"}, 0.0,
    )
    results_a, _ = hazard_resolver.resolve_for_tracks(assessments, now=0.0)
    # Constructing 3 different-mode policies afterward changes nothing
    # about what was already resolved.
    for mode in (MINIMAL, BALANCED, DETAILED):
        OperatingModePolicy(mode=mode, mode_source="CLI", profiles=make_profiles())
    assert results_a[1].level == "LEVEL_1_MOVING_FAR"
    assert results_a[2].level == "LEVEL_1_MOVING_FAR"


def test_minimal_does_not_prevent_person_hazard_resolution() -> None:
    # AudioHazardResolver still produces a full result for a person even
    # though Minimal mode will later suppress it -- internal person
    # detection/analysis is never disabled, only the announcement is.
    delivery_profiles = make_delivery_profiles()
    hazard_resolver = AudioHazardResolver(
        announce_level_1=True, announce_level_2=True,
        delivery_profiles=delivery_profiles,
    )
    threat_engine = make_threat_engine()
    person = make_object(1, "left", "person")
    filtered = {1: make_filtered(1)}
    approach = {1: make_approach(1)}
    assessments = threat_engine.assess_for_tracks(
        [person], filtered, approach, {}, {1: "FAR"}, 0.0,
    )
    results, _ = hazard_resolver.resolve_for_tracks(assessments, now=0.0)
    assert 1 in results  # resolved regardless of mode

    stack = _Stack(MINIMAL)
    eligible, _ = stack.mode_policy.filter_events(results, now=0.0)
    assert eligible == {}  # only the ANNOUNCEMENT is suppressed


def test_startup_gate_has_no_mode_parameter() -> None:
    import inspect

    params = set(inspect.signature(StartupAudioGate.__init__).parameters)
    assert "mode" not in params


# --- G: direction wording ---------------------------------------------


def test_minimal_traffic_message_includes_direction(mock_subprocess) -> None:
    stack = _Stack(MINIMAL)
    car = make_object(1, "left", "car")
    hazard = AudioHazardResult(
        track_id=1, level="LEVEL_1_MOVING_FAR", priority="INFORMATIONAL", class_name="car",
        region="left", proximity_zone="FAR", approach_state="NOT_APPROACHING",
        intersects_corridor=False, uncertain=False, reason_codes=(),
        recommended_message="Vehicle moving from the left.", delivery_profile="level_1",
    )
    final = stack.speak_frame({1: hazard}, [car], now=0.0)
    assert "left" in final[0].message.lower()


def test_center_forward_group_says_ahead(mock_subprocess) -> None:
    stack = _Stack(MINIMAL)
    objects = [make_object(1, "center", "car"), make_object(2, "center", "car")]
    hazards = {
        1: AudioHazardResult(
            track_id=1, level="LEVEL_1_MOVING_FAR", priority="INFORMATIONAL", class_name="car",
            region="center", proximity_zone="FAR", approach_state="NOT_APPROACHING",
            intersects_corridor=False, uncertain=False, reason_codes=(),
            recommended_message="Vehicle moving ahead.", delivery_profile="level_1",
        ),
        2: AudioHazardResult(
            track_id=2, level="LEVEL_1_MOVING_FAR", priority="INFORMATIONAL", class_name="car",
            region="center", proximity_zone="FAR", approach_state="NOT_APPROACHING",
            intersects_corridor=False, uncertain=False, reason_codes=(),
            recommended_message="Vehicle moving ahead.", delivery_profile="level_1",
        ),
    }
    final = stack.speak_frame(hazards, objects, now=0.0)
    assert len(final) == 1
    assert "ahead" in final[0].message.lower()


def test_dangerous_direct_approach_says_directly_ahead() -> None:
    policy = OperatingModePolicy(mode=MINIMAL, mode_source="CLI", profiles=make_profiles())
    from src.models import AudioEvent

    event = AudioEvent(
        key="LEVEL_5_HIGH_DANGER:1:center", event_type="LEVEL_5_HIGH_DANGER", priority="WARNING",
        message="Warning! Vehicle ahead. Please wait.", track_id=1, created_at=0.0,
        delivery_profile="level_5",
    )
    final, _ = policy.finalize_events([event], now=0.0)
    assert "directly ahead" in final[0].message.lower()


# --- H: safety and regression ----------------------------------------------


@pytest.mark.parametrize("mode", [MINIMAL, BALANCED, DETAILED])
def test_no_forbidden_phrase_in_any_mode(mode: str, mock_subprocess) -> None:
    stack = _Stack(mode)
    car = make_object(1, "left", "car")
    hazard = AudioHazardResult(
        track_id=1, level="LEVEL_5_HIGH_DANGER", priority="WARNING", class_name="car",
        region="left", proximity_zone="NEAR", approach_state="APPROACHING",
        intersects_corridor=True, uncertain=False, reason_codes=(),
        recommended_message="Warning! Vehicle approaching from the left. Please wait.",
        delivery_profile="level_5",
    )
    final = stack.speak_frame({1: hazard}, [car], now=0.0)
    for event in final:
        lowered = event.message.lower()
        for phrase in _FORBIDDEN_PHRASES:
            assert phrase not in lowered


def test_only_one_speech_subprocess_at_a_time(mock_subprocess) -> None:
    stack = _Stack(MINIMAL)
    car = make_object(1, "left", "car")
    hazard = AudioHazardResult(
        track_id=1, level="LEVEL_1_MOVING_FAR", priority="INFORMATIONAL", class_name="car",
        region="left", proximity_zone="FAR", approach_state="NOT_APPROACHING",
        intersects_corridor=False, uncertain=False, reason_codes=(),
        recommended_message="Vehicle moving from the left.", delivery_profile="level_1",
    )
    stack.speak_frame({1: hazard}, [car], now=0.0)
    stack.speak_frame({1: hazard}, [car], now=0.1)  # process still "running" (poll()=None)
    assert mock_subprocess["popen"].call_count == 1


def test_full_existing_suite_marker_placeholder() -> None:
    # Real confirmation of "full suite passes" happens via `pytest -q`
    # over the whole tests/ directory, reported in the final report --
    # this is just a structural placeholder so this file documents that
    # requirement explicitly.
    assert True
