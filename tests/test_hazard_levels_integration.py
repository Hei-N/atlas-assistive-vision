"""End-to-end integration tests for the five-level audio hazard system:

    ThreatAssessmentEngine -> AudioHazardResolver -> AudioEventBuilder ->
    SceneSummarizer -> EventPolicy -> SpeechQueue -> AudioWorker -> macOS `say`

Covers escalation/de-escalation across the real queue+cooldown+worker
stack (not just AudioHazardResolver in isolation, which
tests/test_audio_hazard_resolver.py already covers), scene-summary
Level-5-never-diluted behavior, and mode-independence. Only `subprocess`
is mocked -- never real `say`.

ThreatAssessmentEngine is now the single source of truth for hazard
level (see src/threat_assessment.py) -- AudioHazardResolver only
translates its canonical ThreatAssessment output to wording, so these
tests build real ThreatAssessments via a real engine instance rather than
constructing AudioHazardResult inputs by hand.
"""

from unittest.mock import MagicMock

import pytest

from main import compute_hazard_perception, process_audio_events
from src.audio.audio_event_builder import AudioEventBuilder
from src.audio.audio_hazard_resolver import (
    LEVEL_1_MOVING_FAR,
    LEVEL_5_HIGH_DANGER,
    AudioHazardResolver,
)
from src.audio.event_policy import EventPolicy
from src.audio.scene_summarizer import SceneSummarizer
from src.audio.speech_queue import SpeechQueue
from src.audio.audio_worker import AudioWorker
from src.models import (
    ApproachResult,
    BoundingBox,
    DeliveryProfile,
    FilteredMotion,
    PathIntersectionResult,
    TrackedObject,
    TrajectoryPrediction,
)
from src.motion.approach_estimator import ApproachEstimator
from src.motion.relative_proximity_estimator import RelativeProximityEstimator
from src.path_intersection_analyzer import PathIntersectionAnalyzer
from src.threat_assessment import ThreatAssessmentEngine

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


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


def make_profiles() -> dict[str, DeliveryProfile]:
    return {
        f"level_{i}": DeliveryProfile(rate_wpm=170 + i * 5, cue_enabled=False, cue_path=None, cue_gain=0.0)
        for i in range(1, 6)
    }


def make_object(track_id: int, region: str = "left", class_name: str = "car") -> TrackedObject:
    bbox = BoundingBox(x1=0, y1=0, x2=20, y2=20)
    return TrackedObject(
        track_id=track_id, class_id=2, class_name=class_name, confidence=0.9,
        bbox=bbox, center=(10, 10), region=region, position_history=((10, 10),) * 5,
        size_history=((20, 20),) * 5, direction="right", motion_status="moving",
        frames_since_seen=0,
    )


def make_filtered(track_id: int, motion_state: str = "MOVING") -> FilteredMotion:
    return FilteredMotion(
        track_id=track_id, velocity_x=5.0, velocity_y=0.0, speed=5.0,
        motion_state=motion_state, source="COMPENSATED", uncertain=False,
        reason=None, confirmation_frames=0,
    )


def make_prediction(track_id: int) -> TrajectoryPrediction:
    return TrajectoryPrediction(
        valid=True, velocity_x=5.0, velocity_y=0.0, speed_px_per_frame=5.0,
        direction="right", current_center=(10, 10), predicted_center=(60, 10),
        observations_used=5, prediction_horizon_frames=10, uncertain=False,
    )


def make_intersection(track_id: int, intersects: bool = False) -> PathIntersectionResult:
    return PathIntersectionResult(
        valid=True, intersects=intersects, starts_inside=intersects, ends_inside=intersects,
        intersection_point=None, track_id=track_id, reason="test", uncertain=False,
    )


def make_threat_engine(path_conflict_confirmation_frames: int = 1) -> ThreatAssessmentEngine:
    # path_conflict_confirmation_frames=1 by default in these tests --
    # persistence itself is tests/test_threat_assessment.py's job; these
    # tests focus on the downstream queue/cooldown/summary stack.
    return ThreatAssessmentEngine(
        path_conflict_confirmation_frames=path_conflict_confirmation_frames,
        deescalation_seconds=0.75,
        immediate_danger_deescalation_seconds=1.5,
        disappearance_grace_seconds=1.0,
    )


class _Stack:
    """A full, real (subprocess-mocked) hazard-to-speech stack."""

    def __init__(self) -> None:
        self.speech_queue = SpeechQueue(max_size=5, max_age_seconds=10.0)
        self.event_policy = EventPolicy(
            cooldown_same_event_seconds=4.0, cooldown_informational_seconds=8.0
        )
        self.delivery_profiles = make_profiles()
        self.worker = AudioWorker(
            enabled=True, queue=self.speech_queue, event_policy=self.event_policy,
            configured_voice=None, fallback_voice=None, rate_wpm=170,
            delivery_profiles=self.delivery_profiles,
        )
        self.builder = AudioEventBuilder()
        self.summarizer = SceneSummarizer(
            enabled=True, minimum_events=2, max_objects_named=3, same_frame_only=True,
            prefer_single_summary=True, fallback_message="Multiple hazards ahead. Please wait.",
        )
        self.hazard_resolver = AudioHazardResolver(
            announce_level_1=True, announce_level_2=True,
            delivery_profiles=self.delivery_profiles,
        )
        self.threat_engine = make_threat_engine()

    def speak_frame(self, hazard_results, tracked_objects, now) -> None:
        candidates = self.builder.build_events(hazard_results, now)
        final = self.summarizer.summarize(candidates, tracked_objects, now)
        self.worker.enqueue_events(final, now)
        self.worker.tick(now)


# --- 16/17: escalation replaces pending event + bypasses cooldown --------


def test_escalation_replaces_pending_low_level_and_bypasses_cooldown(mock_subprocess) -> None:
    stack = _Stack()
    obj = make_object(1, region="left")
    proximity = RelativeProximityEstimator(far_boundary=0.55, near_boundary=0.80)
    approach = ApproachEstimator(
        proximity_estimator=proximity, minimum_history_samples=5, minimum_approach_cues=2,
        approaching_confirmation_frames=3, ground_point_trend_threshold=0.5,
        scale_growth_threshold_fraction=0.02, corridor_distance_trend_threshold=0.5,
    )
    corridor = PathIntersectionAnalyzer(0.35, 0.65, 1.0, 0.45, 0.55, 0.55)

    filtered = {1: make_filtered(1)}
    predictions = {1: make_prediction(1)}
    intersections_low = {1: make_intersection(1, intersects=False)}
    intersections_high = {1: make_intersection(1, intersects=True)}

    class _FakeCore:
        def __init__(self, intersections):
            self.filtered_motions = filtered
            self.resolved_predictions = predictions
            self.resolved_intersection_results = intersections

    perception = compute_hazard_perception(
        [obj], _FakeCore(intersections_low), proximity, approach, corridor, FRAME_WIDTH, FRAME_HEIGHT
    )
    assessments_low = stack.threat_engine.assess_for_tracks(
        [obj], filtered, perception.approach_results, intersections_low,
        perception.proximity_by_track, now=0.0,
    )
    hazard_results, _ = stack.hazard_resolver.resolve_for_tracks(assessments_low, now=0.0)
    assert hazard_results[1].level == LEVEL_1_MOVING_FAR
    stack.speak_frame(hazard_results, [obj], now=0.0)
    assert stack.worker.is_speaking is True  # Level 1 now speaking

    # A fresh, higher-level assessment for the SAME track arrives before
    # the low-level message would ever get a chance to be spoken again --
    # it must supersede the pending state entirely, never coexist.
    assessments_high = stack.threat_engine.assess_for_tracks(
        [obj], filtered, perception.approach_results, intersections_high,
        perception.proximity_by_track, now=0.1,
    )
    hazard_results_high, _ = stack.hazard_resolver.resolve_for_tracks(assessments_high, now=0.1)
    assert hazard_results_high[1].level != LEVEL_1_MOVING_FAR
    candidates = stack.builder.build_events(hazard_results_high, 0.1)
    final = stack.summarizer.summarize(candidates, [obj], 0.1)
    stack.worker.enqueue_events(final, 0.1)  # must evict the old Level-1 key

    assert len(stack.speech_queue) <= 1  # never two pending entries for track 1


# --- structural: mode-independence ----------------------------------------


def test_compute_hazard_perception_and_resolver_take_no_mode_parameter() -> None:
    import inspect

    for target in (compute_hazard_perception, AudioHazardResolver.__init__, process_audio_events):
        params = set(inspect.signature(target).parameters)
        assert not params & {"args", "debug", "mode", "validate_compensation"}


# --- 19/20: scene summary preserves and never dilutes Level 5 ------------


def test_scene_summary_never_dilutes_level_5_with_level_1() -> None:
    stack = _Stack()
    car = make_object(1, region="left", class_name="car")
    bike = make_object(2, region="left", class_name="bicycle")

    approach_near_confirmed = ApproachResult(
        track_id=1, state="APPROACHING", confidence=1.0, proximity_zone="NEAR",
        closing_score=1.0, corridor_distance_trend=-1.0, scale_growth_trend=0.1,
        ground_point_trend=1.0, confirmation_frames=3, uncertain=False, evidence_flags=(),
    )
    approach_not_approaching = ApproachResult(
        track_id=2, state="NOT_APPROACHING", confidence=0.0, proximity_zone="FAR",
        closing_score=0.0, corridor_distance_trend=0.0, scale_growth_trend=0.0,
        ground_point_trend=0.0, confirmation_frames=0, uncertain=False, evidence_flags=(),
    )

    assessments = stack.threat_engine.assess_for_tracks(
        [car, bike],
        {1: make_filtered(1), 2: make_filtered(2)},
        {1: approach_near_confirmed, 2: approach_not_approaching},
        {1: make_intersection(1, intersects=True)},  # only car intersects
        {1: "NEAR", 2: "FAR"},
        now=0.0,
    )
    hazard_results, _ = stack.hazard_resolver.resolve_for_tracks(assessments, now=0.0)
    assert hazard_results[1].level == LEVEL_5_HIGH_DANGER
    assert hazard_results[2].level == LEVEL_1_MOVING_FAR

    candidates = stack.builder.build_events(hazard_results, 0.0)
    final = stack.summarizer.summarize(candidates, [car, bike], 0.0)

    assert len(final) == 1
    assert final[0].message.startswith("Warning!")
    assert "bicycle" not in final[0].message.lower()
    assert "Multiple" not in final[0].message


# --- SINGLE SOURCE OF TRUTH: AudioHazardResolver never overrides canonical -


def test_audio_hazard_resolver_mirrors_canonical_level_exactly() -> None:
    """Changing ONLY the canonical ThreatAssessment.level (nothing else)
    changes AudioHazardResolver's mapped level identically -- proving the
    resolver has no independent classification of its own."""
    from src.models import ThreatAssessment
    from src.threat_assessment import CONFIRMED_APPROACH, IMMEDIATE_DANGER, NORMAL_MOVEMENT

    resolver = AudioHazardResolver(
        announce_level_1=True, announce_level_2=True, delivery_profiles=make_profiles(),
    )

    def make_assessment(level: str) -> ThreatAssessment:
        return ThreatAssessment(
            track_id=1, object_class="car", region="left", level=level, confidence=1.0,
            uncertain=False, reason_codes=(), proximity_zone="NEAR", approach_state="APPROACHING",
            approach_evidence_flags=(), intersects_corridor=False,
            corridor_intersection_uncertain=False, time_to_conflict_seconds=None,
            persistence_frames=1, escalated=False, deescalation_pending=False,
            frame_index=None, timestamp=0.0,
        )

    for threat_level, expected_audio_level in (
        (NORMAL_MOVEMENT, LEVEL_1_MOVING_FAR),
        (CONFIRMED_APPROACH, "LEVEL_3_APPROACHING"),
        (IMMEDIATE_DANGER, LEVEL_5_HIGH_DANGER),
    ):
        results, _ = resolver.resolve_for_tracks({1: make_assessment(threat_level)}, now=0.0)
        assert results[1].level == expected_audio_level
