"""Unit tests for main.py's build_* wiring helpers relevant to Phase 4
camera-motion compensation and the --validate-compensation developer
mode, plus compute_core_motion() (the motion-reliability restructuring
milestone: core perception must run in every mode, not just --debug). No
camera, model, or network required -- build_motion_compensator is pure
config -> MotionCompensator construction, build_validation_writer/
build_compensation_csv_logger only need a writable tmp path (no webcam),
and compute_core_motion is pure logic over synthetic TrackedObject/
CompensatedMotion/MotionEstimate data plus real (but camera-free)
TrajectoryPredictor/MotionStateFilter/PathIntersectionAnalyzer instances.
"""

from datetime import datetime
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from main import (
    CoreMotionResult,
    HazardPerceptionResult,
    build_compensation_csv_logger,
    build_motion_compensator,
    build_validation_writer,
    compute_core_motion,
    compute_hazard_perception,
    parse_args,
    process_audio_events,
)
from src.models import BoundingBox, CompensatedMotion, MotionEstimate, TrackedObject
from src.motion.compensation_csv_logger import CompensationCsvLogger
from src.motion.compensation_validator import (
    build_per_object_validation,
    build_validation_stats,
)
from src.motion.motion_compensator import MotionCompensator
from src.motion.motion_state_filter import MotionStateFilter
from src.path_intersection_analyzer import PathIntersectionAnalyzer
from src.trajectory_predictor import TrajectoryPredictor
from src.visualizer import Visualizer


def test_builds_from_valid_configuration() -> None:
    config = {
        "motion_compensation": {
            "enabled": True,
            "min_camera_confidence": 0.7,
            "stationary_threshold_px": 3.0,
        }
    }

    compensator = build_motion_compensator(config)

    assert isinstance(compensator, MotionCompensator)


def test_missing_section_uses_safe_defaults_and_still_starts() -> None:
    config = {}  # motion_compensation section entirely absent

    compensator = build_motion_compensator(config)

    # Missing section is NOT the same as disabled here (unlike
    # motion_estimation) -- Atlas must still start, using documented
    # defaults, per the explicit requirement for this milestone.
    assert isinstance(compensator, MotionCompensator)


def test_explicit_enabled_false_returns_none() -> None:
    config = {"motion_compensation": {"enabled": False}}

    compensator = build_motion_compensator(config)

    assert compensator is None


def test_invalid_configuration_raises_value_error() -> None:
    config = {
        "motion_compensation": {
            "enabled": True,
            "min_camera_confidence": 1.5,  # out of [0, 1]
            "stationary_threshold_px": 2.0,
        }
    }

    with pytest.raises(ValueError):
        build_motion_compensator(config)


def test_partial_section_fills_missing_fields_with_defaults() -> None:
    # Section present (so enabled defaults to True) but individual fields
    # omitted -- must still construct successfully via per-field defaults.
    config = {"motion_compensation": {}}

    compensator = build_motion_compensator(config)

    assert isinstance(compensator, MotionCompensator)


# --- --validate-compensation / --validation-output CLI flags ------------


def test_validate_compensation_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", "--source", "0"])

    args = parse_args()

    assert args.validate_compensation is False
    assert args.validation_output is None


def test_validate_compensation_flag_can_be_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--source", "0", "--validate-compensation", "--validation-output", "out.mp4"],
    )

    args = parse_args()

    assert args.validate_compensation is True
    assert args.validation_output == "out.mp4"


# --- build_validation_writer ---------------------------------------------


def test_validation_writer_not_created_without_output_path() -> None:
    writer = build_validation_writer(None, 100, 100, 30.0)
    assert writer is None


def test_validation_writer_opens_for_a_valid_path(tmp_path) -> None:
    output_path = str(tmp_path / "validation_output.mp4")

    writer = build_validation_writer(output_path, 64, 48, 30.0)

    assert writer is not None
    assert writer.isOpened()
    writer.release()


def test_validation_writer_falls_back_to_default_fps_when_unusable(tmp_path) -> None:
    output_path = str(tmp_path / "validation_output_no_fps.mp4")

    writer = build_validation_writer(output_path, 64, 48, fps=0.0)

    assert writer is not None
    assert writer.isOpened()
    writer.release()


def test_validation_writer_returns_none_for_unwritable_path() -> None:
    # A directory that doesn't exist -- OpenCV cannot open a writer there.
    bad_path = "/this/directory/does/not/exist/output.mp4"

    writer = build_validation_writer(bad_path, 64, 48, 30.0)

    assert writer is None


# --- build_compensation_csv_logger ---------------------------------------


def test_no_csv_logger_and_no_file_created_without_validate_compensation(
    tmp_path,
) -> None:
    csv_logger = build_compensation_csv_logger(
        validate_compensation=False, logs_dir=tmp_path, now=datetime(2026, 7, 27, 20, 56, 0)
    )

    assert csv_logger is None
    # No CSV file of any kind should have been created in the logs dir.
    assert list(tmp_path.iterdir()) == []


def test_csv_logger_created_in_validation_mode(tmp_path) -> None:
    now = datetime(2026, 7, 27, 20, 56, 0)

    csv_logger = build_compensation_csv_logger(
        validate_compensation=True, logs_dir=tmp_path, now=now
    )

    try:
        assert isinstance(csv_logger, CompensationCsvLogger)
        expected_path = tmp_path / "compensation_validation_2026-07-27_205600.csv"
        assert expected_path.exists()
    finally:
        if csv_logger is not None:
            csv_logger.close()


# --- compute_core_motion (core perception must run in every mode) --------

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_SIZE = 200


def make_trajectory_predictor() -> TrajectoryPredictor:
    return TrajectoryPredictor(
        min_observations=5,
        prediction_horizon_frames=10,
        stationary_threshold_px_per_frame=1.5,
        max_missed_frames=5,
        history_window=10,
    )


def make_motion_state_filter() -> MotionStateFilter:
    return MotionStateFilter(
        stationary_enter_speed=2.5,
        moving_enter_speed=4.0,
        moving_confirmation_frames=3,
        stationary_confirmation_frames=3,
        minimum_history_samples=5,
        smoothing_alpha=1.0,
        suppress_raw_motion_during_camera_motion=True,
    )


def make_path_intersection_analyzer() -> PathIntersectionAnalyzer:
    return PathIntersectionAnalyzer(
        corridor_bottom_left_x=0.35,
        corridor_bottom_right_x=0.65,
        corridor_bottom_y=1.0,
        corridor_top_left_x=0.45,
        corridor_top_right_x=0.55,
        corridor_top_y=0.55,
    )


def make_tracked_object(
    track_id: int = 1,
    position_history: tuple = ((100, 100),) * 6,
    direction: str = "stationary",
    motion_status: str = "stationary",
) -> TrackedObject:
    center = position_history[-1]
    bbox = BoundingBox(
        x1=center[0] - 10, y1=center[1] - 10, x2=center[0] + 10, y2=center[1] + 10
    )
    return TrackedObject(
        track_id=track_id,
        class_id=2,
        class_name="car",
        confidence=0.9,
        bbox=bbox,
        center=center,
        region="center",
        position_history=position_history,
        size_history=((20, 20),) * len(position_history),
        direction=direction,
        motion_status=motion_status,
        frames_since_seen=0,
    )


def make_compensated_motion(
    track_id: int = 1,
    compensated_velocity_x: float = 0.0,
    compensated_velocity_y: float = 0.0,
    raw_velocity_x: float = 0.0,
    raw_velocity_y: float = 0.0,
) -> CompensatedMotion:
    speed = (compensated_velocity_x**2 + compensated_velocity_y**2) ** 0.5
    return CompensatedMotion(
        track_id=track_id,
        compensated_velocity_x=compensated_velocity_x,
        compensated_velocity_y=compensated_velocity_y,
        compensated_speed=speed,
        compensated_direction="right" if speed > 2.0 else "stationary",
        raw_velocity_x=raw_velocity_x,
        raw_velocity_y=raw_velocity_y,
        camera_motion_applied=True,
    )


def make_motion_estimate(status: str = "VALID") -> MotionEstimate:
    return MotionEstimate(
        valid=status != "UNAVAILABLE",
        dx=0.0,
        dy=0.0,
        rotation_degrees=0.0,
        scale=1.0,
        confidence=0.9 if status == "VALID" else 0.3,
        feature_count=120,
        tracked_feature_count=100,
        inlier_count=90,
        inlier_ratio=0.9,
        status=status,
        reason_invalid=None if status != "UNAVAILABLE" else "insufficient features detected",
        raw_dx=0.0,
        raw_dy=0.0,
        raw_rotation_degrees=0.0,
        transform_matrix=None,
        debug_prev_points=(),
        debug_current_points=(),
        debug_inlier_flags=(),
    )


# --- A: normal mode executes motion resolution and filtering --------------


def test_compute_core_motion_runs_full_pipeline_with_no_mode_concept() -> None:
    # compute_core_motion's signature has no args/debug/mode parameter at
    # all -- this alone proves it cannot behave differently by mode. Here
    # we also confirm it actually populates every stage's output.
    track = make_tracked_object(track_id=1)
    compensated = {1: make_compensated_motion(track_id=1)}

    core = compute_core_motion(
        [track], compensated, make_motion_estimate("VALID"),
        make_trajectory_predictor(), make_motion_state_filter(),
        make_path_intersection_analyzer(), FRAME_WIDTH, FRAME_HEIGHT,
    )

    assert isinstance(core, CoreMotionResult)
    assert 1 in core.predictions
    assert 1 in core.resolved_motions
    assert 1 in core.filtered_motions
    assert 1 in core.resolved_predictions
    assert 1 in core.resolved_intersection_results


# --- B: normal mode display uses filtered motion, not raw ------------------


def test_normal_mode_display_uses_filtered_motion_end_to_end(monkeypatch) -> None:
    # Raw position history drifts steadily right (ObjectTracker's own raw
    # direction/motion_status would read "moving"), but the compensated
    # velocity is ~0 (camera motion fully explains the drift) -- the
    # filtered pipeline must say STATIONARY, and draw_tracked_objects must
    # show that, end to end, with zero debug-flag concept anywhere in the
    # call chain (compute_core_motion has none; draw_tracked_objects just
    # receives whatever dict it's given).
    history = tuple((100 + i * 10, 100) for i in range(6))
    track = make_tracked_object(
        track_id=1, position_history=history, direction="right", motion_status="moving"
    )
    compensated = {1: make_compensated_motion(track_id=1, compensated_velocity_x=0.0)}

    core = compute_core_motion(
        [track], compensated, make_motion_estimate("VALID"),
        make_trajectory_predictor(), make_motion_state_filter(),
        make_path_intersection_analyzer(), FRAME_WIDTH, FRAME_HEIGHT,
    )

    captured: dict = {}

    def fake_draw_pil_text(frame, entries):
        captured["entries"] = entries

    monkeypatch.setattr(Visualizer, "_draw_pil_text", staticmethod(fake_draw_pil_text))
    visualizer = Visualizer("test-window")
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)

    visualizer.draw_tracked_objects(frame, [track], core.filtered_motions)

    label_text = captured["entries"][0][1]
    assert "STATIONARY" in label_text
    assert "MOVING" not in label_text


# --- C: debug off vs. debug on -> identical core results -------------------


def test_core_motion_identical_regardless_of_debug_mode() -> None:
    # Two fully independent component sets (stateful MotionStateFilter
    # included), simulating "a normal-mode run" and "a debug-mode run"
    # fed the IDENTICAL sequence of inputs. compute_core_motion itself
    # takes no mode parameter, so results must match every frame.
    normal = (
        make_trajectory_predictor(), make_motion_state_filter(), make_path_intersection_analyzer()
    )
    debug = (
        make_trajectory_predictor(), make_motion_state_filter(), make_path_intersection_analyzer()
    )

    for i in range(6):
        history = tuple((100 + j * 3, 100) for j in range(i + 1))
        track = make_tracked_object(track_id=1, position_history=history)
        compensated = {1: make_compensated_motion(track_id=1, compensated_velocity_x=3.0)}
        motion_estimate = make_motion_estimate("VALID")

        normal_core = compute_core_motion(
            [track], compensated, motion_estimate, *normal, FRAME_WIDTH, FRAME_HEIGHT
        )
        debug_core = compute_core_motion(
            [track], compensated, motion_estimate, *debug, FRAME_WIDTH, FRAME_HEIGHT
        )

        assert normal_core.predictions == debug_core.predictions
        assert normal_core.resolved_motions == debug_core.resolved_motions
        assert normal_core.filtered_motions == debug_core.filtered_motions
        assert normal_core.resolved_predictions == debug_core.resolved_predictions
        assert (
            normal_core.resolved_intersection_results
            == debug_core.resolved_intersection_results
        )


# --- D: debug overlays add rendering without changing core decisions ------


def test_debug_only_extras_do_not_mutate_core_motion() -> None:
    track = make_tracked_object(track_id=1)
    compensated = {1: make_compensated_motion(track_id=1)}
    analyzer = make_path_intersection_analyzer()

    core = compute_core_motion(
        [track], compensated, make_motion_estimate("VALID"),
        make_trajectory_predictor(), make_motion_state_filter(),
        analyzer, FRAME_WIDTH, FRAME_HEIGHT,
    )
    before = {
        "resolved_motions": dict(core.resolved_motions),
        "filtered_motions": dict(core.filtered_motions),
        "resolved_predictions": dict(core.resolved_predictions),
        "resolved_intersection_results": dict(core.resolved_intersection_results),
    }

    # Debug-only extras: the raw corridor comparison check, and drawing.
    analyzer.analyze(1, core.predictions[1], FRAME_WIDTH, FRAME_HEIGHT)
    visualizer = Visualizer("test-window")
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    visualizer.draw_trajectories(frame, [track], core.predictions, True, 5)

    assert core.resolved_motions == before["resolved_motions"]
    assert core.filtered_motions == before["filtered_motions"]
    assert core.resolved_predictions == before["resolved_predictions"]
    assert core.resolved_intersection_results == before["resolved_intersection_results"]


# --- E: validation instrumentation adds data without changing core --------


def test_validation_only_extras_do_not_mutate_core_motion() -> None:
    track = make_tracked_object(track_id=1)
    compensated = {1: make_compensated_motion(track_id=1)}
    motion_estimate = make_motion_estimate("VALID")
    analyzer = make_path_intersection_analyzer()

    core = compute_core_motion(
        [track], compensated, motion_estimate,
        make_trajectory_predictor(), make_motion_state_filter(),
        analyzer, FRAME_WIDTH, FRAME_HEIGHT,
    )
    before_resolved = dict(core.resolved_motions)
    before_filtered = dict(core.filtered_motions)

    intersection_results = {
        1: analyzer.analyze(1, core.predictions[1], FRAME_WIDTH, FRAME_HEIGHT)
    }
    per_object_validation = build_per_object_validation(
        [track], compensated, motion_estimate, 2.0,
        core.resolved_motions, intersection_results,
        core.resolved_intersection_results, core.filtered_motions,
    )
    build_validation_stats(per_object_validation, motion_estimate)

    assert core.resolved_motions == before_resolved
    assert core.filtered_motions == before_filtered


# --- F: zero-detection and missing-compensation paths remain safe ---------


def test_compute_core_motion_handles_zero_detections() -> None:
    core = compute_core_motion(
        [], {}, None, make_trajectory_predictor(), make_motion_state_filter(),
        make_path_intersection_analyzer(), FRAME_WIDTH, FRAME_HEIGHT,
    )

    assert core.predictions == {}
    assert core.resolved_motions == {}
    assert core.filtered_motions == {}
    assert core.resolved_predictions == {}
    assert core.resolved_intersection_results == {}


def test_compute_core_motion_handles_missing_compensation_via_raw_fallback() -> None:
    history = tuple((100 + i * 3, 100) for i in range(6))
    track = make_tracked_object(track_id=1, position_history=history)

    # compensated_motions={} and motion_estimate=None -- motion
    # compensation not configured at all.
    core = compute_core_motion(
        [track], {}, None, make_trajectory_predictor(), make_motion_state_filter(),
        make_path_intersection_analyzer(), FRAME_WIDTH, FRAME_HEIGHT,
    )

    assert core.filtered_motions[1].source == "RAW_FALLBACK"
    assert core.filtered_motions[1].uncertain is True


# --- process_audio_events (exclusive startup-audio sequencing) ------------
# No camera/subprocess involved -- everything but the fake gate is a
# MagicMock, since these tests target process_audio_events' own
# branching logic in isolation. tests/test_audio_sequencing.py covers
# the same behavior end to end with real StartupAnnouncer/
# StartupAudioGate/SpeechQueue/AudioWorker/AudioSequencingTracer.


class _FakeGate:
    def __init__(self, warnings_blocked: bool, first_ready_cycle: bool = False) -> None:
        self.warnings_blocked = warnings_blocked
        self._first_ready_cycle = first_ready_cycle

    def is_first_ready_cycle(self) -> bool:
        return self._first_ready_cycle


EMPTY_CORE = CoreMotionResult(
    predictions={}, resolved_motions={}, filtered_motions={},
    resolved_predictions={}, resolved_intersection_results={},
)

EMPTY_HAZARD_PERCEPTION = HazardPerceptionResult(proximity_by_track={}, approach_results={})

# process_audio_events no longer takes core/hazard_perception -- it takes
# the already-canonical threat_assessments dict (ThreatAssessmentEngine's
# output) directly. EMPTY_CORE/EMPTY_HAZARD_PERCEPTION above are kept only
# because other tests in this file (compute_core_motion/
# compute_hazard_perception tests) still use them.
EMPTY_THREAT_ASSESSMENTS: dict = {}


def _call_process_audio_events(
    gate, builder, summarizer, worker, tracked_objects, timestamp,
    hazard_resolver=None, threat_assessments=None,
    mode_policy=None,
):
    if threat_assessments is None:
        threat_assessments = EMPTY_THREAT_ASSESSMENTS
    announcer = MagicMock()
    speech_queue = MagicMock()
    tracer = MagicMock()
    if hazard_resolver is None:
        hazard_resolver = MagicMock()
        hazard_resolver.resolve_for_tracks.return_value = ({}, {})
    if mode_policy is None:
        mode_policy = MagicMock()
        # Identity pass-through by default -- existing tests assert on
        # builder/summarizer/worker call args, so operating-mode
        # filtering must not alter them unless a test opts in.
        mode_policy.filter_events.side_effect = lambda hazard_results, now: (hazard_results, {})
        mode_policy.finalize_events.side_effect = lambda events, now: (events, [])
    process_audio_events(
        announcer, gate, hazard_resolver, mode_policy, builder, summarizer, worker,
        speech_queue, tracer, tracked_objects, threat_assessments, timestamp,
    )
    return tracer


def test_process_audio_events_is_a_complete_noop_while_blocked() -> None:
    gate = _FakeGate(warnings_blocked=True)
    builder = MagicMock()
    summarizer = MagicMock()
    worker = MagicMock()
    hazard_resolver = MagicMock()

    tracer = _call_process_audio_events(
        gate, builder, summarizer, worker, [], 0.0, hazard_resolver=hazard_resolver
    )

    hazard_resolver.resolve_for_tracks.assert_not_called()
    builder.build_events.assert_not_called()
    summarizer.summarize.assert_not_called()
    worker.enqueue_events.assert_not_called()
    worker.tick.assert_not_called()
    # candidates_built=False must have been reported to the tracer.
    assert tracer.record_frame.call_args[0][-1] is False


def test_process_audio_events_runs_full_pipeline_when_ready() -> None:
    gate = _FakeGate(warnings_blocked=False)
    builder = MagicMock()
    summarizer = MagicMock()
    worker = MagicMock()
    hazard_resolver = MagicMock()
    hazard_results = {1: "hazard-result"}
    hazard_resolver.resolve_for_tracks.return_value = (hazard_results, {})
    candidates = ["candidate-event"]
    finals = ["final-event"]
    builder.build_events.return_value = candidates
    summarizer.summarize.return_value = finals

    tracer = _call_process_audio_events(
        gate, builder, summarizer, worker, [], 42.0, hazard_resolver=hazard_resolver
    )

    hazard_resolver.resolve_for_tracks.assert_called_once()
    builder.build_events.assert_called_once_with(hazard_results, 42.0)
    summarizer.summarize.assert_called_once_with(candidates, [], 42.0)
    worker.enqueue_events.assert_called_once_with(finals, 42.0)
    worker.tick.assert_called_once_with(42.0)
    assert tracer.record_frame.call_args[0][-1] is True


def test_process_audio_events_never_replays_events_built_while_blocked() -> None:
    # Simulates a frame during startup speech (blocked -- nothing built)
    # followed by the frame right after it finishes (ready). The second
    # call must build a completely fresh batch, never anything from the
    # blocked call (there was nothing to replay in the first place).
    builder = MagicMock()
    summarizer = MagicMock()
    worker = MagicMock()
    summarizer.summarize.return_value = []

    _call_process_audio_events(_FakeGate(True), builder, summarizer, worker, [], 1.0)
    _call_process_audio_events(_FakeGate(False), builder, summarizer, worker, [], 2.0)

    assert builder.build_events.call_count == 1  # only the unblocked call
    _, called_timestamp = builder.build_events.call_args[0]
    assert called_timestamp == 2.0  # the current frame, not the blocked one


def test_process_audio_events_limits_first_ready_cycle_to_one_event() -> None:
    # On the very first frame the gate reports READY, even if several
    # objects are already in frame, at most the single highest-priority
    # event may be admitted -- READY must not itself sound like a burst.
    gate = _FakeGate(warnings_blocked=False, first_ready_cycle=True)
    builder = MagicMock()
    summarizer = MagicMock()
    worker = MagicMock()
    informational = MagicMock(priority="INFORMATIONAL", created_at=1.0)
    warning = MagicMock(priority="WARNING", created_at=2.0)
    summarizer.summarize.return_value = [informational, warning]

    _call_process_audio_events(gate, builder, summarizer, worker, [], 0.0)

    admitted = worker.enqueue_events.call_args[0][0]
    assert admitted == [warning]  # WARNING beats INFORMATIONAL


def test_process_audio_events_does_not_limit_after_first_ready_cycle() -> None:
    gate = _FakeGate(warnings_blocked=False, first_ready_cycle=False)
    builder = MagicMock()
    summarizer = MagicMock()
    worker = MagicMock()
    events = [MagicMock(priority="INFORMATIONAL", created_at=1.0), MagicMock(priority="WARNING", created_at=2.0)]
    summarizer.summarize.return_value = events

    _call_process_audio_events(gate, builder, summarizer, worker, [], 0.0)

    admitted = worker.enqueue_events.call_args[0][0]
    assert admitted == events  # both survive -- not the first cycle


def test_process_audio_events_has_no_perception_component_parameters() -> None:
    # Structural proof that this function cannot gate perception (or
    # threat classification) even in principle: it has no detector/
    # tracker/motion-estimator parameter at all -- `threat_assessments`
    # (ThreatAssessmentEngine's canonical output) is always computed by
    # the caller beforehand, unconditionally, every frame.
    import inspect

    params = set(inspect.signature(process_audio_events).parameters)
    assert "threat_assessments" in params
    assert "core" not in params
    assert "hazard_perception" not in params
    assert not params & {"detector", "tracker", "motion_estimator", "video"}


def test_compute_hazard_perception_has_no_mode_parameter() -> None:
    # Mirrors compute_core_motion's own precedent: no args/debug/mode
    # parameter at all, so it cannot behave differently by mode.
    import inspect

    params = set(inspect.signature(compute_hazard_perception).parameters)
    assert not params & {"args", "debug", "mode"}
