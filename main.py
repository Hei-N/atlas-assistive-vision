"""Atlas Phase 1 entry point.

Wires together video input, YOLO detection, region analysis, and
visualization into a live (or file-based) detection loop.

Usage:
    python main.py --source 0
    python main.py --source data/input/test_video.mp4
    python main.py --source 0 --debug
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import yaml

from src.audio.audio_event_builder import AudioEventBuilder
from src.audio.audio_hazard_resolver import AudioHazardResolver
from src.audio.audio_sequencing_tracer import AudioSequencingTracer
from src.audio.audio_worker import AudioWorker
from src.audio.event_policy import EventPolicy
from src.audio.operating_mode_policy import (
    BALANCED,
    DETAILED,
    MINIMAL,
    PERSON_BROAD,
    PERSON_INELIGIBLE,
    PERSON_RELEVANT_ONLY,
    OperatingModePolicy,
    parse_operating_mode,
)
from src.audio.scene_summarizer import SceneSummarizer
from src.audio.speech_queue import SpeechQueue
from src.audio.startup_announcer import StartupAnnouncer
from src.audio.startup_audio_gate import StartupAudioGate
from src.conflict_imminence import ConflictImminenceEstimator
from src.models import (
    ApproachResult,
    AudioSequencingTrace,
    CompensatedMotion,
    ConflictImminence,
    DeliveryProfile,
    FilteredMotion,
    MotionEstimate,
    OperatingModeProfile,
    PathIntersectionResult,
    ResolvedMotion,
    SystemHealthTrace,
    ThreatAssessment,
    TrackedObject,
    TrajectoryPrediction,
)
from src.motion.approach_estimator import ApproachEstimator
from src.motion.compensation_csv_logger import (
    CompensationCsvLogger,
    PerObjectCsvLogger,
    build_csv_filename,
)
from src.motion.compensation_validator import (
    build_per_object_validation,
    build_validation_stats,
    should_log_validation_summary,
)
from src.motion.motion_compensator import MotionCompensator
from src.motion.motion_resolver import resolve_motions_for_tracks
from src.motion.motion_state_filter import MotionStateFilter
from src.motion.relative_proximity_estimator import RelativeProximityEstimator
from src.motion.visual_motion_estimator import VisualMotionEstimator
from src.object_detector import ObjectDetector
from src.object_tracker import ObjectTracker
from src.path_intersection_analyzer import PathIntersectionAnalyzer
from src.region_analyzer import RegionAnalyzer
from src.system.system_health_monitor import SystemHealthMonitor
from src.threat_assessment import ThreatAssessmentEngine
from src.trajectory_predictor import TrajectoryPredictor
from src.runtime_paths import get_config_path, get_model_path
from src.video_source import VideoSource, VideoSourceError
from src.visualizer import Visualizer

QUIT_KEY = "q"
DEFAULT_CONFIG_PATH = "config/settings.yaml"
DEFAULT_VALIDATION_OUTPUT_FPS = 20.0
COMPENSATION_CSV_LOG_DIR = Path("logs")
COMPENSATION_CSV_LOG_INTERVAL_FRAMES = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("atlas")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Atlas Phase 1: pedestrian hazard-awareness prototype."
    )
    parser.add_argument(
        "--source",
        required=True,
        help='Webcam index (e.g. "0") or path to a video file.',
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the YAML config file (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show FPS, frame dimensions, and region/zone boundaries.",
    )
    parser.add_argument(
        "--validate-compensation",
        action="store_true",
        help=(
            "Developer-only: show an aggregate + per-object camera-motion-"
            "compensation validation panel, for evaluating compensation on "
            "real footage before it's wired into trajectory prediction. "
            "Implies --debug."
        ),
    )
    parser.add_argument(
        "--validation-output",
        default=None,
        metavar="PATH",
        help=(
            "Optional: save the annotated --validate-compensation output "
            "to this video path. Ignored unless --validate-compensation "
            "is also passed."
        ),
    )
    parser.add_argument(
        "--export-per-object-csv",
        default=None,
        metavar="PATH",
        help=(
            "Optional: export complete per-object, per-frame camera-"
            "motion/compensation diagnostics to this CSV path -- intended "
            "for validating a controlled prerecorded clip. Ignored unless "
            "--validate-compensation is also passed."
        ),
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable all spoken/audio output, including the startup announcement.",
    )
    parser.add_argument(
        "--mode",
        type=str.lower,
        choices=["minimal", "balanced", "detailed"],
        default=None,
        help=(
            "Operating mode: how much is announced and how. Overrides "
            "operating_mode.default in config. Falls back to 'balanced' "
            "if neither is set."
        ),
    )
    parser.add_argument(
        "--startup-audio-voice",
        default=None,
        metavar="NAME",
        help=(
            "macOS `say` voice name for the startup announcement only. "
            "Overrides audio.startup_announcement.voice and --audio-voice."
        ),
    )
    parser.add_argument(
        "--startup-audio-rate",
        type=int,
        default=None,
        metavar="WPM",
        help=(
            "Words-per-minute for the startup announcement only. "
            "Overrides audio.startup_announcement.rate_wpm and --audio-rate."
        ),
    )
    parser.add_argument(
        "--audio-voice",
        default=None,
        metavar="NAME",
        help=(
            "General fallback macOS `say` voice name, used only where a "
            "more specific voice (e.g. --startup-audio-voice) isn't set."
        ),
    )
    parser.add_argument(
        "--audio-rate",
        type=int,
        default=None,
        metavar="WPM",
        help=(
            "General fallback words-per-minute, used only where a more "
            "specific rate (e.g. --startup-audio-rate) isn't set."
        ),
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and return the YAML configuration file.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is not valid YAML.
    """
    path = get_config_path() if config_path == DEFAULT_CONFIG_PATH else Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path!r}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_region_analyzer(config: dict) -> RegionAnalyzer:
    """Construct a RegionAnalyzer from the loaded configuration."""
    regions = config["regions"]
    zone = config["attention_zone"]
    return RegionAnalyzer(
        left_boundary=regions["left_boundary"],
        right_boundary=regions["right_boundary"],
        attention_zone_fractions=(
            zone["x_min_fraction"],
            zone["x_max_fraction"],
            zone["y_min_fraction"],
            zone["y_max_fraction"],
        ),
    )


def build_object_detector(
    config: dict, region_analyzer: RegionAnalyzer
) -> ObjectDetector:
    """Construct an ObjectDetector from the loaded configuration."""
    model_cfg = config["model"]
    allowed_classes = {int(k): v for k, v in config["classes"].items()}
    return ObjectDetector(
        model_name=get_model_path(model_cfg["name"]),
        confidence_threshold=model_cfg["confidence_threshold"],
        allowed_classes=allowed_classes,
        region_analyzer=region_analyzer,
    )


def build_object_tracker(config: dict) -> ObjectTracker:
    """Construct an ObjectTracker from the loaded configuration."""
    tracking_cfg = config["tracking"]
    return ObjectTracker(
        max_match_distance_px=tracking_cfg["max_match_distance_px"],
        max_disappeared_frames=tracking_cfg["max_disappeared_frames"],
        min_match_iou=tracking_cfg["min_match_iou"],
        history_length=tracking_cfg["history_length"],
        stationary_threshold_px=tracking_cfg["stationary_threshold_px"],
        size_change_threshold_fraction=tracking_cfg[
            "size_change_threshold_fraction"
        ],
    )


def build_trajectory_predictor(config: dict) -> TrajectoryPredictor:
    """Construct a TrajectoryPredictor from the loaded configuration."""
    trajectory_cfg = config["trajectory"]
    return TrajectoryPredictor(
        min_observations=trajectory_cfg["trajectory_min_observations"],
        prediction_horizon_frames=trajectory_cfg[
            "trajectory_prediction_horizon_frames"
        ],
        stationary_threshold_px_per_frame=trajectory_cfg[
            "trajectory_stationary_threshold_px_per_frame"
        ],
        max_missed_frames=trajectory_cfg["trajectory_max_missed_frames"],
        history_window=trajectory_cfg["trajectory_history_length"],
    )


def build_motion_state_filter(config: dict) -> MotionStateFilter:
    """Construct a MotionStateFilter from the loaded configuration.

    Unlike build_motion_estimator, a MISSING motion_filter section does
    not disable filtering -- it's filled in with the documented defaults
    below, mirroring build_motion_compensator's "missing section = safe
    defaults" convention, so Atlas still starts normally on an old
    config file that predates this section.
    """
    filter_cfg = config.get("motion_filter", {})
    return MotionStateFilter(
        stationary_enter_speed=filter_cfg.get("stationary_enter_speed", 2.5),
        moving_enter_speed=filter_cfg.get("moving_enter_speed", 4.0),
        moving_confirmation_frames=filter_cfg.get("moving_confirmation_frames", 5),
        stationary_confirmation_frames=filter_cfg.get(
            "stationary_confirmation_frames", 5
        ),
        minimum_history_samples=filter_cfg.get("minimum_history_samples", 5),
        smoothing_alpha=filter_cfg.get("smoothing_alpha", 0.3),
        suppress_raw_motion_during_camera_motion=filter_cfg.get(
            "suppress_raw_motion_during_camera_motion", True
        ),
    )


class CoreMotionResult(NamedTuple):
    """Output of compute_core_motion() -- see its docstring. A plain
    orchestration bundle (not a shared per-object domain model, so it
    lives here rather than in src/models.py), mirroring scripts/
    benchmark_detectors.py's own precedent of a small, locally-defined
    result type.
    """

    predictions: dict[int, TrajectoryPrediction]
    resolved_motions: dict[int, ResolvedMotion]
    filtered_motions: dict[int, FilteredMotion]
    resolved_predictions: dict[int, TrajectoryPrediction]
    resolved_intersection_results: dict[int, PathIntersectionResult]


def compute_core_motion(
    tracked_objects: list[TrackedObject],
    compensated_motions: dict[int, CompensatedMotion],
    motion_estimate: MotionEstimate | None,
    trajectory_predictor: TrajectoryPredictor,
    motion_state_filter: MotionStateFilter,
    path_intersection_analyzer: PathIntersectionAnalyzer,
    frame_width: int,
    frame_height: int,
) -> CoreMotionResult:
    """Always-on core perception step: raw trajectory prediction,
    resolved-motion selection, stationary-motion filtering, resolved
    trajectory prediction, and the resulting corridor decision.

    Deliberately takes no CLI-flag/mode parameter at all -- core
    perception and safety-relevant computation must run identically in
    every mode; callers (run()) decide what ADDITIONAL diagnostics/
    rendering to layer on top, never whether this computation itself
    runs. `predictions` (the raw, regression-fit trajectory) is included
    here rather than treated as debug-only because resolve_motions_for_
    tracks requires it as the RAW_FALLBACK velocity source whenever
    camera-motion compensation is UNAVAILABLE -- it is a real input to
    this core pipeline, not merely a comparison value. The raw corridor-
    intersection check (PathIntersectionAnalyzer.analyze on `predictions`
    directly) is NOT computed here: nothing in the core pipeline consumes
    it, it exists solely for the --validate-compensation panel's debug
    comparison column, so it stays the caller's responsibility.

    Safe for zero tracked_objects (returns all-empty dicts) and for
    compensated_motions={}/motion_estimate=None (motion compensation not
    configured -- every track resolves via RAW_FALLBACK, never crashes).
    """
    predictions = {
        obj.track_id: trajectory_predictor.predict(
            obj.position_history,
            obj.frames_since_seen,
            frame_width,
            frame_height,
        )
        for obj in tracked_objects
    }
    resolved_motions = resolve_motions_for_tracks(
        tracked_objects, compensated_motions, motion_estimate, predictions
    )
    filtered_motions = motion_state_filter.filter_for_tracks(
        tracked_objects, resolved_motions
    )
    resolved_predictions = {
        obj.track_id: trajectory_predictor.predict_from_resolved_motion(
            obj.center,
            filtered_motions[obj.track_id].velocity_x,
            filtered_motions[obj.track_id].velocity_y,
            len(obj.position_history),
            frame_width,
            frame_height,
            filtered_motions[obj.track_id].uncertain,
        )
        for obj in tracked_objects
    }
    resolved_intersection_results = {
        obj.track_id: path_intersection_analyzer.analyze(
            obj.track_id, resolved_predictions[obj.track_id],
            frame_width, frame_height,
        )
        for obj in tracked_objects
    }
    return CoreMotionResult(
        predictions=predictions,
        resolved_motions=resolved_motions,
        filtered_motions=filtered_motions,
        resolved_predictions=resolved_predictions,
        resolved_intersection_results=resolved_intersection_results,
    )


def build_path_intersection_analyzer(config: dict) -> PathIntersectionAnalyzer:
    """Construct a PathIntersectionAnalyzer from the loaded configuration."""
    corridor_cfg = config["corridor"]
    return PathIntersectionAnalyzer(
        corridor_bottom_left_x=corridor_cfg["corridor_bottom_left_x"],
        corridor_bottom_right_x=corridor_cfg["corridor_bottom_right_x"],
        corridor_bottom_y=corridor_cfg["corridor_bottom_y"],
        corridor_top_left_x=corridor_cfg["corridor_top_left_x"],
        corridor_top_right_x=corridor_cfg["corridor_top_right_x"],
        corridor_top_y=corridor_cfg["corridor_top_y"],
    )


def build_motion_estimator(config: dict) -> VisualMotionEstimator | None:
    """Construct a VisualMotionEstimator from the loaded configuration.

    Returns None (motion estimation disabled) when the motion_estimation
    section is missing entirely, or when it's present with enabled: false
    -- main.py treats both the same way rather than failing.
    """
    motion_cfg = config.get("motion_estimation")
    if motion_cfg is None or not motion_cfg.get("enabled", False):
        return None

    return VisualMotionEstimator(
        max_features=motion_cfg["max_features"],
        quality_level=motion_cfg["quality_level"],
        min_distance_px=motion_cfg["min_distance_px"],
        block_size=motion_cfg["block_size"],
        lk_window_size=motion_cfg["lk_window_size"],
        lk_max_level=motion_cfg["lk_max_level"],
        ransac_threshold_px=motion_cfg["ransac_threshold_px"],
        min_tracked_features=motion_cfg["min_tracked_features"],
        min_inliers=motion_cfg["min_inliers"],
        exclude_foreground=motion_cfg["exclude_foreground"],
        exclusion_padding_px=motion_cfg["exclusion_padding_px"],
        smoothing_enabled=motion_cfg["smoothing_enabled"],
        smoothing_alpha=motion_cfg["smoothing_alpha"],
        max_consecutive_invalid_frames=motion_cfg["max_consecutive_invalid_frames"],
        low_confidence_threshold=motion_cfg["low_confidence_threshold"],
        max_translation_px=motion_cfg["max_translation_px"],
        max_rotation_degrees=motion_cfg["max_rotation_degrees"],
    )


def build_motion_compensator(config: dict) -> MotionCompensator | None:
    """Construct a MotionCompensator from the loaded configuration.

    Unlike build_motion_estimator, a MISSING motion_compensation section
    does not mean disabled -- it's filled in with the documented defaults
    below so Atlas still starts normally on an old/incomplete config file.
    Returns None only when enabled is false (explicitly, or because the
    section is present but omits enabled and the caller wants the
    default) -- main.py then skips compensation for that run, matching
    build_motion_estimator's None-means-skip convention.
    """
    compensation_cfg = config.get("motion_compensation", {})
    if not compensation_cfg.get("enabled", True):
        return None

    return MotionCompensator(
        min_camera_confidence=compensation_cfg.get("min_camera_confidence", 0.6),
        stationary_threshold_px=compensation_cfg.get("stationary_threshold_px", 2.0),
    )


def build_validation_writer(
    output_path: str | None,
    frame_width: int,
    frame_height: int,
    fps: float,
) -> cv2.VideoWriter | None:
    """Construct the optional --validation-output video writer.

    Returns None (nothing recorded) when output_path is None -- the
    --validate-compensation panel still renders on screen either way.

    A source that doesn't report a usable FPS (<=0 or non-finite) falls
    back to DEFAULT_VALIDATION_OUTPUT_FPS rather than producing a broken
    (0 fps) file. If the writer still fails to open (e.g. an invalid or
    unwritable path), this logs a clear error and returns None rather
    than raising -- recording is an optional bonus feature, so the rest
    of the run continues normally without saving instead of crashing.
    """
    if output_path is None:
        return None

    if fps <= 0 or not math.isfinite(fps):
        logger.warning(
            "Source did not report a usable FPS for --validation-output; "
            "falling back to %.0f fps.",
            DEFAULT_VALIDATION_OUTPUT_FPS,
        )
        fps = DEFAULT_VALIDATION_OUTPUT_FPS

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        logger.error(
            "Could not open --validation-output path for writing: %r. "
            "Continuing without saving validation video.",
            output_path,
        )
        return None

    logger.info(
        "Recording --validate-compensation output to %s (%dx%d @ %.1f fps).",
        output_path,
        frame_width,
        frame_height,
        fps,
    )
    return writer


def build_compensation_csv_logger(
    validate_compensation: bool,
    logs_dir: Path,
    now: datetime,
) -> CompensationCsvLogger | None:
    """Construct the --validate-compensation structured CSV logger.

    Returns None (no file created) when validate_compensation is False --
    logging only ever activates alongside that flag. `now` is taken
    explicitly (not read internally via datetime.now()) so this stays a
    deterministic, directly testable function.
    """
    if not validate_compensation:
        return None

    csv_path = logs_dir / build_csv_filename(now)
    logger.info("Logging compensation validation data to %s", csv_path)
    return CompensationCsvLogger(csv_path)


def build_per_object_csv_logger(
    validate_compensation: bool,
    export_path: str | None,
) -> PerObjectCsvLogger | None:
    """Construct the optional --export-per-object-csv logger.

    Returns None (nothing exported) when export_path is None, or when
    validate_compensation is False -- mirrors build_validation_writer's
    existing "only activates alongside --validate-compensation" pattern.
    """
    if not validate_compensation or export_path is None:
        return None

    logger.info("Exporting per-object compensation diagnostics to %s", export_path)
    return PerObjectCsvLogger(export_path)


def build_startup_announcer(config: dict, args: argparse.Namespace) -> StartupAnnouncer:
    """Construct the cinematic startup announcement from config + CLI.

    Resolves the required precedence chain (requirement 6): startup-
    specific CLI override -> startup-specific config -> general audio
    CLI override -> general audio config -> system default. Each tier's
    CLI-vs-config pair is collapsed here into a single "startup" value
    and a single "general" value; StartupAnnouncer itself then tries the
    startup value first (if actually installed), then the general value,
    matching requirement 7's installed-voice fallback order exactly.

    Unlike build_motion_estimator, a MISSING audio section does not
    disable the announcement -- it's filled in with the documented
    defaults below, mirroring build_motion_compensator's "missing
    section = safe defaults" convention.
    """
    audio_cfg = config.get("audio", {})
    startup_cfg = audio_cfg.get("startup_announcement", {})

    startup_voice = args.startup_audio_voice or startup_cfg.get("voice")
    general_voice = args.audio_voice or audio_cfg.get("voice")

    startup_rate = args.startup_audio_rate or startup_cfg.get("rate_wpm", 165)
    general_rate = args.audio_rate or audio_cfg.get("rate_wpm")
    rate_wpm = startup_rate if startup_rate is not None else general_rate

    return StartupAnnouncer(
        enabled=not args.no_audio and startup_cfg.get("enabled", True),
        message=startup_cfg.get(
            "message", "Atlas online. Detection system activated."
        ),
        play_once=startup_cfg.get("play_once", True),
        delay_seconds=startup_cfg.get("delay_seconds", 0.25),
        require_first_valid_frame=startup_cfg.get("require_first_valid_frame", True),
        configured_voice=startup_voice,
        fallback_voice=general_voice,
        rate_wpm=rate_wpm,
        sound_enabled=startup_cfg.get("sound_enabled", False),
        sound_path=startup_cfg.get("sound_path"),
        sound_volume=startup_cfg.get("sound_volume", 0.7),
    )


def build_audio_event_builder(config: dict) -> AudioEventBuilder:
    """Construct the AudioEventBuilder -- stateless, config-independent
    today (the spoken vocabulary/templates are fixed, not configurable),
    but routed through a build_* factory for consistency with every
    other component and to leave room for future configurability."""
    return AudioEventBuilder()


def build_event_policy(config: dict) -> EventPolicy:
    """Construct the EventPolicy cooldown gate from config.

    Unlike build_motion_estimator, a MISSING audio.warnings section does
    not disable warnings -- it's filled in with the documented defaults
    below, matching build_motion_compensator's "missing section = safe
    defaults" convention.
    """
    warnings_cfg = config.get("audio", {}).get("warnings", {})
    return EventPolicy(
        cooldown_same_event_seconds=warnings_cfg.get("cooldown_same_event_seconds", 4.0),
        cooldown_informational_seconds=warnings_cfg.get(
            "cooldown_informational_seconds", 8.0
        ),
    )


def build_speech_queue(config: dict) -> SpeechQueue:
    """Construct the bounded SpeechQueue from config."""
    warnings_cfg = config.get("audio", {}).get("warnings", {})
    return SpeechQueue(
        max_size=warnings_cfg.get("max_queue_size", 5),
        max_age_seconds=warnings_cfg.get("max_event_age_seconds", 3.0),
    )


def build_scene_summarizer(config: dict) -> SceneSummarizer:
    """Construct the SceneSummarizer from config.

    Unlike build_motion_estimator, a MISSING audio.warnings.scene_summary
    section does not disable summarization -- it's filled in with the
    documented defaults below, matching build_event_policy's "missing
    section = safe defaults" convention.
    """
    scene_summary_cfg = config.get("audio", {}).get("warnings", {}).get("scene_summary", {})
    return SceneSummarizer(
        enabled=scene_summary_cfg.get("enabled", True),
        minimum_events=scene_summary_cfg.get("minimum_events", 2),
        max_objects_named=scene_summary_cfg.get("max_objects_named", 3),
        same_frame_only=scene_summary_cfg.get("same_frame_only", True),
        prefer_single_summary=scene_summary_cfg.get("prefer_single_summary", True),
        fallback_message=scene_summary_cfg.get(
            "fallback_message", "Multiple hazards ahead. Please wait."
        ),
    )


def build_relative_proximity_estimator(config: dict) -> RelativeProximityEstimator:
    """Construct the RelativeProximityEstimator from config.

    Unlike build_motion_estimator, a MISSING audio.warnings.proximity
    section does not disable proximity classification -- it's filled in
    with the documented defaults below, matching build_event_policy's
    "missing section = safe defaults" convention.
    """
    proximity_cfg = config.get("audio", {}).get("warnings", {}).get("proximity", {})
    return RelativeProximityEstimator(
        far_boundary=proximity_cfg.get("far_boundary", 0.55),
        near_boundary=proximity_cfg.get("near_boundary", 0.80),
    )


def build_approach_estimator(
    config: dict, proximity_estimator: RelativeProximityEstimator
) -> ApproachEstimator:
    """Construct the ApproachEstimator from config.

    Unlike build_motion_estimator, a MISSING audio.warnings.hazard_levels
    section does not disable approach detection -- it's filled in with
    the documented defaults below.
    """
    hazard_cfg = config.get("audio", {}).get("warnings", {}).get("hazard_levels", {})
    return ApproachEstimator(
        proximity_estimator=proximity_estimator,
        minimum_history_samples=hazard_cfg.get("minimum_history_samples", 5),
        minimum_approach_cues=hazard_cfg.get("minimum_approach_cues", 2),
        approaching_confirmation_frames=hazard_cfg.get("approaching_confirmation_frames", 3),
        ground_point_trend_threshold=hazard_cfg.get("ground_point_trend_threshold", 0.5),
        scale_growth_threshold_fraction=hazard_cfg.get("scale_growth_threshold_fraction", 0.02),
        corridor_distance_trend_threshold=hazard_cfg.get("corridor_distance_trend_threshold", 0.5),
    )


_DELIVERY_PROFILE_DEFAULTS = {
    "level_1": {"rate_wpm": 170, "cue_enabled": False, "cue_path": None, "cue_gain": 0.0},
    "level_2": {"rate_wpm": 178, "cue_enabled": False, "cue_path": None, "cue_gain": 0.0},
    "level_3": {"rate_wpm": 188, "cue_enabled": False, "cue_path": None, "cue_gain": 0.0},
    "level_4": {"rate_wpm": 195, "cue_enabled": True, "cue_path": None, "cue_gain": 0.75},
    "level_5": {"rate_wpm": 200, "cue_enabled": True, "cue_path": None, "cue_gain": 1.0},
}


def build_delivery_profiles(config: dict) -> dict[str, DeliveryProfile]:
    """Construct the per-hazard-level DeliveryProfile map from config.

    Unlike build_motion_estimator, a MISSING audio.warnings.delivery_
    profiles section (or an individual missing level within it) does not
    disable delivery profiles -- each level falls back to the documented
    defaults below.
    """
    profiles_cfg = config.get("audio", {}).get("warnings", {}).get("delivery_profiles", {})
    profiles: dict[str, DeliveryProfile] = {}
    for level_key, defaults in _DELIVERY_PROFILE_DEFAULTS.items():
        level_cfg = profiles_cfg.get(level_key, {})
        profiles[level_key] = DeliveryProfile(
            rate_wpm=level_cfg.get("rate_wpm", defaults["rate_wpm"]),
            cue_enabled=level_cfg.get("cue_enabled", defaults["cue_enabled"]),
            cue_path=level_cfg.get("cue_path", defaults["cue_path"]),
            cue_gain=level_cfg.get("cue_gain", defaults["cue_gain"]),
        )
    return profiles


def build_audio_hazard_resolver(
    config: dict, delivery_profiles: dict[str, DeliveryProfile]
) -> AudioHazardResolver:
    """Construct the AudioHazardResolver from config.

    No longer reads/owns a de-escalation-hold setting -- that canonical
    timing lives entirely in ThreatAssessmentEngine (threat_assessment.*
    config) now. See build_threat_assessment_engine.
    """
    hazard_cfg = config.get("audio", {}).get("warnings", {}).get("hazard_levels", {})
    return AudioHazardResolver(
        announce_level_1=hazard_cfg.get("announce_level_1", True),
        announce_level_2=hazard_cfg.get("announce_level_2", True),
        delivery_profiles=delivery_profiles,
    )


def build_threat_assessment_engine(config: dict) -> ThreatAssessmentEngine | None:
    """Construct the canonical ThreatAssessmentEngine from config, or None
    if threat_assessment.enabled is explicitly false (missing section or
    missing key defaults to enabled, matching this codebase's "missing
    config = safe/on defaults" convention elsewhere).

    Independent of AudioHazardResolver (built separately above) -- see
    src/threat_assessment.py's module docstring for why the two keep
    entirely separate state despite sharing the same underlying
    single-frame decision tree.
    """
    threat_cfg = config.get("threat_assessment", {})
    if not threat_cfg.get("enabled", True):
        return None
    return ThreatAssessmentEngine(
        path_conflict_confirmation_frames=threat_cfg.get("path_conflict_confirmation_frames", 3),
        deescalation_seconds=threat_cfg.get("deescalation_seconds", 0.75),
        immediate_danger_deescalation_seconds=threat_cfg.get(
            "immediate_danger_deescalation_seconds", 1.5
        ),
        disappearance_grace_seconds=threat_cfg.get("disappearance_grace_seconds", 1.0),
    )


def build_conflict_imminence_estimator(config: dict) -> ConflictImminenceEstimator | None:
    """Construct the ConflictImminenceEstimator from config, or None if
    conflict_imminence.enabled is explicitly false (missing section or
    missing key defaults to enabled, matching build_threat_assessment_
    engine's own "missing config = safe/on defaults" convention).

    Independent of ThreatAssessmentEngine (built separately above) --
    its output is threaded INTO ThreatAssessmentEngine.assess_for_tracks
    as an optional argument, not merged with it; see
    src/conflict_imminence.py's module docstring.
    """
    imminence_cfg = config.get("conflict_imminence", {})
    if not imminence_cfg.get("enabled", True):
        return None
    return ConflictImminenceEstimator(
        imminent_step_frames=imminence_cfg.get("imminent_step_frames", 3),
        soon_step_frames=imminence_cfg.get("soon_step_frames", 7),
        confirmation_frames=imminence_cfg.get("confirmation_frames", 3),
        deescalation_seconds=imminence_cfg.get("deescalation_seconds", 0.75),
        disappearance_grace_seconds=imminence_cfg.get("disappearance_grace_seconds", 1.0),
    )


class HazardPerceptionResult(NamedTuple):
    """Output of compute_hazard_perception() -- see its docstring."""

    proximity_by_track: dict[int, str]
    approach_results: dict[int, ApproachResult]


def compute_hazard_perception(
    tracked_objects: list[TrackedObject],
    core: CoreMotionResult,
    proximity_estimator: RelativeProximityEstimator,
    approach_estimator: ApproachEstimator,
    path_intersection_analyzer: PathIntersectionAnalyzer,
    frame_width: int,
    frame_height: int,
) -> HazardPerceptionResult:
    """Always-on relative-proximity and confirmed-approach computation:
    orchestrates the existing RelativeProximityEstimator and
    ApproachEstimator in sequence -- makes no hazard-level decision
    itself (that's AudioHazardResolver's job, downstream). Mirrors
    compute_core_motion's own "no mode parameter, perception runs in
    every mode" precedent -- called unconditionally every frame,
    regardless of whether the startup announcement is still speaking, so
    approach-confirmation streaks keep building continuously rather than
    pausing during startup silence.
    """
    proximity_by_track = proximity_estimator.estimate_for_tracks(tracked_objects, frame_height)
    corridor_polygon = path_intersection_analyzer.build_corridor_polygon(
        frame_width, frame_height
    )
    approach_results = approach_estimator.estimate_for_tracks(
        tracked_objects, core.filtered_motions, core.resolved_predictions,
        core.resolved_intersection_results, proximity_by_track, corridor_polygon, frame_height,
    )
    return HazardPerceptionResult(
        proximity_by_track=proximity_by_track, approach_results=approach_results
    )


def resolve_operating_mode(config: dict, args: argparse.Namespace) -> tuple[str, str]:
    """Resolve the active operating mode and where it came from.

    Precedence: --mode CLI flag -> operating_mode.default config ->
    hard default "balanced". Returns (mode, source) where source is
    "CLI", "CONFIG", or "DEFAULT" -- carried into every OperatingModeTrace
    for observability. --mode's argparse `choices=` already rejects an
    invalid CLI value before this function ever runs; an invalid CONFIG
    value is rejected here via parse_operating_mode() (raises, matching
    this codebase's "fail loudly on bad config" convention elsewhere).
    """
    if args.mode is not None:
        return parse_operating_mode(args.mode), "CLI"

    configured = config.get("operating_mode", {}).get("default")
    if configured is not None:
        return parse_operating_mode(configured), "CONFIG"

    return BALANCED, "DEFAULT"


def build_operating_mode_policy(config: dict, args: argparse.Namespace) -> OperatingModePolicy:
    """Construct the OperatingModePolicy from config + CLI.

    Unlike build_motion_estimator, a MISSING operating_mode section (or
    an individual missing mode within it) does not disable mode
    filtering -- each mode falls back to the documented defaults below.
    """
    mode, source = resolve_operating_mode(config, args)
    operating_mode_cfg = config.get("operating_mode", {})

    minimal_cfg = operating_mode_cfg.get("minimal", {})
    balanced_cfg = operating_mode_cfg.get("balanced", {})
    detailed_cfg = operating_mode_cfg.get("detailed", {})

    profiles = {
        MINIMAL: OperatingModeProfile(
            mode=MINIMAL,
            announce_traffic=minimal_cfg.get("announce_traffic", True),
            person_eligibility=(
                PERSON_RELEVANT_ONLY if minimal_cfg.get("announce_people", False) else PERSON_INELIGIBLE
            ),
            highest_danger_repeat_enabled=minimal_cfg.get("highest_danger_repeat_enabled", True),
            highest_danger_repeat_delay_seconds=minimal_cfg.get(
                "highest_danger_repeat_delay_seconds", 3.0
            ),
            highest_danger_max_cycles=minimal_cfg.get("highest_danger_max_cycles", 2),
            structured_scene_summaries=minimal_cfg.get("detailed_scene_summaries", False),
            max_summary_object_groups=minimal_cfg.get("max_summary_object_groups", 2),
            max_summary_words=minimal_cfg.get("max_summary_words", 15),
        ),
        BALANCED: OperatingModeProfile(
            mode=BALANCED,
            announce_traffic=balanced_cfg.get("announce_traffic", True),
            person_eligibility=(
                PERSON_RELEVANT_ONLY
                if (
                    balanced_cfg.get("announce_people_when_close", True)
                    or balanced_cfg.get("announce_people_when_approaching", True)
                    or balanced_cfg.get("announce_people_when_crossing_path", True)
                )
                else PERSON_INELIGIBLE
            ),
            highest_danger_repeat_enabled=balanced_cfg.get("highest_danger_repeat_enabled", True),
            highest_danger_repeat_delay_seconds=balanced_cfg.get(
                "highest_danger_repeat_delay_seconds", 3.0
            ),
            highest_danger_max_cycles=balanced_cfg.get("highest_danger_max_cycles", 2),
            structured_scene_summaries=balanced_cfg.get("detailed_scene_summaries", False),
            max_summary_object_groups=balanced_cfg.get("max_summary_object_groups", 2),
            max_summary_words=balanced_cfg.get("max_summary_words", 15),
        ),
        DETAILED: OperatingModeProfile(
            mode=DETAILED,
            announce_traffic=detailed_cfg.get("announce_traffic", True),
            person_eligibility=(
                PERSON_BROAD if detailed_cfg.get("announce_people", True) else PERSON_INELIGIBLE
            ),
            highest_danger_repeat_enabled=detailed_cfg.get("highest_danger_repeat_enabled", False),
            highest_danger_repeat_delay_seconds=detailed_cfg.get(
                "highest_danger_repeat_delay_seconds", 3.0
            ),
            highest_danger_max_cycles=detailed_cfg.get("highest_danger_max_cycles", 1),
            structured_scene_summaries=detailed_cfg.get("structured_scene_summaries", True),
            max_summary_object_groups=detailed_cfg.get("max_summary_object_groups", 2),
            max_summary_words=detailed_cfg.get("max_summary_words", 15),
        ),
    }

    return OperatingModePolicy(mode=mode, mode_source=source, profiles=profiles)


_PRIORITY_RANK = {"WARNING": 0, "INFORMATIONAL": 1}


def process_audio_events(
    startup_announcer: StartupAnnouncer,
    startup_audio_gate: StartupAudioGate,
    audio_hazard_resolver: AudioHazardResolver,
    operating_mode_policy: OperatingModePolicy,
    audio_event_builder: AudioEventBuilder,
    scene_summarizer: SceneSummarizer,
    audio_worker: AudioWorker,
    speech_queue: SpeechQueue,
    tracer: AudioSequencingTracer,
    tracked_objects: list[TrackedObject],
    threat_assessments: dict[int, ThreatAssessment],
    timestamp: float,
) -> AudioSequencingTrace:
    """Resolves hazard levels, builds, summarizes, and enqueues this
    frame's warning events -- unless the cinematic startup announcement
    still owns the audio system (see StartupAudioGate), in which case
    this is a complete no-op: no hazard levels are resolved, no events
    are built, queued, or spoken here, and nothing is buffered for
    later. AudioHazardResolver.resolve_for_tracks() is a stateless
    translation of the ALREADY-canonical threat_assessments (computed by
    ThreatAssessmentEngine every frame regardless of this gate -- see
    below) and AudioEventBuilder.build_events() is stateless too, so
    simply skipping this function while blocked is sufficient -- there is
    nothing accumulated to discard. Once unblocked, the very next call
    evaluates the current frame fresh, never anything from during startup
    speech.

    On the very first frame the gate reports READY (StartupAudioGate.
    is_first_ready_cycle()), at most the single highest-priority fresh
    event is admitted -- even if several objects are already in frame --
    so the transition out of startup silence doesn't itself sound like a
    burst of speech. Every frame after that admits events normally.

    Perception and canonical threat classification (`threat_assessments`,
    from ThreatAssessmentEngine.assess_for_tracks() -- see src/
    threat_assessment.py) are computed unconditionally by the caller
    regardless of this gate -- this function has no detector/tracker/
    motion-estimator parameter at all, so it cannot gate
    perception even in principle. Only hazard-event RESOLUTION/speaking is
    gated here, not the underlying perception or threat classification --
    ThreatAssessmentEngine's own escalation/de-escalation/persistence
    state keeps advancing regardless of whether this function runs.

    Returns an AudioSequencingTrace snapshot of this frame's sequencing
    state, for runtime verification (see src/models.py /
    src/audio/audio_sequencing_tracer.py) rather than only judging
    correctness by listening to spoken output. Per-object hazard
    decisions are separately logged at DEBUG level by
    AudioHazardResolver itself (see HazardTrace).
    """
    blocked = startup_audio_gate.warnings_blocked
    candidates_built = False
    if not blocked:
        hazard_results, _hazard_traces = audio_hazard_resolver.resolve_for_tracks(
            threat_assessments, timestamp,
        )
        # Stage 1: class/person eligibility by operating mode -- before
        # any wording exists. See src/audio/operating_mode_policy.py.
        eligible_results, _mode_filter_traces = operating_mode_policy.filter_events(
            hazard_results, timestamp
        )
        candidate_events = audio_event_builder.build_events(eligible_results, timestamp)
        candidates_built = True
        final_events = scene_summarizer.summarize(candidate_events, tracked_objects, timestamp)
        # Stage 2: Level-5 message form + danger-episode repeat policy +
        # Detailed-mode summary group/word caps.
        final_events, _mode_finalize_traces = operating_mode_policy.finalize_events(
            final_events, timestamp
        )

        if startup_audio_gate.is_first_ready_cycle() and len(final_events) > 1:
            final_events = [
                min(final_events, key=lambda e: (_PRIORITY_RANK.get(e.priority, 1), e.created_at))
            ]

        audio_worker.enqueue_events(final_events, timestamp)
        audio_worker.tick(timestamp)

    return tracer.record_frame(
        startup_announcer, startup_audio_gate, speech_queue, audio_worker,
        timestamp, candidates_built,
    )


def build_system_health_monitor(config: dict) -> SystemHealthMonitor:
    """Construct the SystemHealthMonitor from config's system_health:
    section. See src/system/system_health_monitor.py and
    docs/SYSTEM_HEALTH_AUDIO.md for the exact thresholds and rationale.
    """
    health_cfg = config.get("system_health", {})
    blocked_cfg = health_cfg.get("camera_blocked", {})
    feed_lost_cfg = health_cfg.get("camera_feed_lost", {})
    detection_cfg = health_cfg.get("detection_unavailable", {})

    return SystemHealthMonitor(
        enabled=health_cfg.get("enabled", True),
        camera_blocked_confirmation_seconds=blocked_cfg.get("confirmation_seconds", 1.5),
        variance_threshold=blocked_cfg.get("variance_threshold", 50.0),
        edge_density_threshold=blocked_cfg.get("edge_density_threshold", 0.02),
        camera_feed_lost_consecutive_failed_reads=feed_lost_cfg.get(
            "consecutive_failed_reads", 3
        ),
        detection_unavailable_consecutive_failures=detection_cfg.get(
            "consecutive_failures", 3
        ),
        failure_repeat_seconds=health_cfg.get("failure_repeat_seconds", 5.0),
        failure_max_cycles=health_cfg.get("failure_max_cycles", 2),
        recovery_confirmation_seconds=health_cfg.get("recovery_confirmation_seconds", 1.5),
    )


def process_system_health(
    monitor: SystemHealthMonitor,
    event_policy: EventPolicy,
    speech_queue: SpeechQueue,
    audio_worker: AudioWorker,
    timestamp: float,
) -> SystemHealthTrace:
    """Evaluates this frame's critical system-health state and, if a
    failure/repeat/recovery message is due, clears every OTHER pending
    operational entry (informational events, scene summaries, uncertain
    object messages alike) before speaking it -- a second, independent
    gate over the audio pipeline, structurally parallel to but not
    merged with StartupAudioGate (see process_audio_events). Ticks
    audio_worker every call (not only when a message was just admitted)
    so any in-flight cue/speech keeps progressing even while
    process_audio_events() is being skipped for this frame.

    Callers must check the returned trace's operational_audio_suppressed
    field and skip process_audio_events() for this frame when True --
    this function does not call it itself, keeping perception/event
    generation and system-health gating strictly separate.
    """
    event, trace = monitor.evaluate(timestamp)
    if event is not None:
        cleared = speech_queue.clear()
        for admitted in event_policy.filter([event], timestamp):
            speech_queue.enqueue(admitted)
        trace = dataclasses.replace(trace, operational_queue_items_cleared=cleared)
    audio_worker.tick(timestamp)
    return trace


def build_audio_worker(
    config: dict,
    args: argparse.Namespace,
    event_policy: EventPolicy,
    speech_queue: SpeechQueue,
    delivery_profiles: dict[str, DeliveryProfile],
) -> AudioWorker:
    """Construct the AudioWorker from config + CLI.

    Voice/rate precedence mirrors build_startup_announcer's: a
    warnings-specific config value, falling back to the general
    audio.voice/audio.rate_wpm config (--audio-voice/--audio-rate CLI
    overrides apply to both; there are no warnings-specific CLI flags,
    and --no-audio disables startup and warning audio alike). rate_wpm here is only the
    fallback used when an event's own delivery_profile isn't found in
    delivery_profiles -- normal operation resolves rate per hazard level.
    """
    audio_cfg = config.get("audio", {})
    warnings_cfg = audio_cfg.get("warnings", {})

    warnings_voice = warnings_cfg.get("voice") or args.audio_voice
    general_voice = audio_cfg.get("voice") or args.audio_voice

    warnings_rate = warnings_cfg.get("rate_wpm", 170) or args.audio_rate
    general_rate = audio_cfg.get("rate_wpm") or args.audio_rate
    rate_wpm = warnings_rate if warnings_rate is not None else general_rate

    return AudioWorker(
        enabled=not args.no_audio and warnings_cfg.get("enabled", True),
        queue=speech_queue,
        event_policy=event_policy,
        configured_voice=warnings_voice,
        fallback_voice=general_voice,
        rate_wpm=rate_wpm,
        delivery_profiles=delivery_profiles,
    )


def run(args: argparse.Namespace) -> None:
    """Run the main detection loop."""
    if args.validate_compensation:
        # --validate-compensation implies --debug: the panel is only
        # meaningful alongside the rest of the debug overlays, and this
        # avoids requiring the user to pass both flags redundantly.
        args.debug = True
    if args.debug:
        logger.setLevel(logging.DEBUG)

    config = load_config(args.config)

    region_analyzer = build_region_analyzer(config)
    try:
        detector = build_object_detector(config, region_analyzer)
    except Exception as exc:
        logger.error("Failed to load detection model: %s", exc)
        return
    tracker = build_object_tracker(config)
    trajectory_predictor = build_trajectory_predictor(config)
    trajectory_cfg = config["trajectory"]
    path_intersection_analyzer = build_path_intersection_analyzer(config)
    corridor_cfg = config["corridor"]
    motion_estimator = build_motion_estimator(config)
    motion_cfg = config.get("motion_estimation", {})
    if motion_estimator is not None:
        # No existing code path in run() reopens/swaps VideoSource mid-run
        # today, so there's no real "video source reset" event to hook
        # into yet -- this call is a defensive no-op (the estimator is
        # already fresh). Any future code path that reopens/switches
        # VideoSource must call motion_estimator.reset() there too.
        motion_estimator.reset()
    motion_compensator = build_motion_compensator(config)
    compensation_cfg = config.get("motion_compensation", {})
    compensation_stationary_threshold_px = compensation_cfg.get(
        "stationary_threshold_px", 2.0
    )
    motion_state_filter = build_motion_state_filter(config)
    window_name = config["display"]["window_name"]
    visualizer = Visualizer(window_name)

    logger.info(
        "Loading detection model. On the first run after a fresh install, "
        "macOS may take up to a minute to verify newly-installed native "
        "libraries before this finishes -- this is a one-time OS check, "
        "not a hang. Subsequent runs are fast."
    )
    try:
        detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
    except Exception as exc:
        logger.error("Detection model failed during warm-up: %s", exc)
        return
    logger.info("Detection model ready.")

    try:
        video = VideoSource(args.source)
    except VideoSourceError as exc:
        logger.error(str(exc))
        return

    validation_writer = None
    if args.validate_compensation and args.validation_output:
        writer_width, writer_height = video.get_frame_size()
        validation_writer = build_validation_writer(
            args.validation_output, writer_width, writer_height, video.get_fps()
        )

    compensation_csv_logger = build_compensation_csv_logger(
        args.validate_compensation, COMPENSATION_CSV_LOG_DIR, datetime.now()
    )
    per_object_csv_logger = build_per_object_csv_logger(
        args.validate_compensation, args.export_per_object_csv
    )

    # Reaching this point already proves the camera opened successfully
    # and every perception component above constructed without error, so
    # marking the announcer "camera ready" here directly satisfies
    # requirement 2's readiness conditions -- no separate check needed.
    startup_announcer = build_startup_announcer(config, args)
    startup_announcer.mark_camera_ready()
    # Startup speech has exclusive ownership of the audio system until
    # it fully finishes -- the warning pipeline below is gated on this
    # every frame so the two can never overlap. See
    # src/audio/startup_audio_gate.py.
    startup_audio_gate = StartupAudioGate(startup_announcer)
    audio_sequencing_tracer = AudioSequencingTracer()

    # Real-time spoken warnings, built from resolved perception results
    # only. Runs unconditionally in every mode (see the compute_core_
    # motion() call below) -- CLI flags never change what's computed,
    # only diagnostics/rendering.
    audio_event_builder = build_audio_event_builder(config)
    scene_summarizer = build_scene_summarizer(config)
    event_policy = build_event_policy(config)
    speech_queue = build_speech_queue(config)
    delivery_profiles = build_delivery_profiles(config)
    audio_worker = build_audio_worker(config, args, event_policy, speech_queue, delivery_profiles)

    # Five-level audio hazard system: perspective-aware proximity ->
    # confirmed approach -> hazard-level resolution. See
    # src/audio/audio_hazard_resolver.py and docs/AUDIO_ALERT_LEVELS.md.
    relative_proximity_estimator = build_relative_proximity_estimator(config)
    approach_estimator = build_approach_estimator(config, relative_proximity_estimator)
    audio_hazard_resolver = build_audio_hazard_resolver(config, delivery_profiles)

    # Canonical Level 1-5 threat classification (Threat Assessment Engine
    # v1): audio-independent from audio_hazard_resolver above -- see
    # src/threat_assessment.py. Computed unconditionally every frame,
    # like hazard_perception below, and is the single source of truth
    # AudioHazardResolver consumes for Level 1-5.
    threat_assessment_engine = build_threat_assessment_engine(config)
    # Conflict Imminence V1: a conservative, image-space, frame-step
    # -based signal feeding a second Level 5 pathway inside
    # ThreatAssessmentEngine -- see src/conflict_imminence.py. Computed
    # unconditionally every frame, alongside threat_assessment_engine.
    conflict_imminence_estimator = build_conflict_imminence_estimator(config)

    # User-selectable operating mode (Minimal/Balanced/Detailed): filters
    # and reshapes the already-resolved hazard events above -- perception
    # and hazard-level decisions are identical across all three modes.
    # See src/audio/operating_mode_policy.py and docs/OPERATING_MODES.md.
    operating_mode_policy = build_operating_mode_policy(config, args)
    logger.info(
        "Operating mode: %s (source: %s)",
        operating_mode_policy.mode, operating_mode_policy.mode_source,
    )

    # Critical system-health monitoring (camera blocked, camera/video
    # feed lost, detector unavailable): a second, independent gate over
    # the audio pipeline, alongside -- not merged with -- StartupAudioGate.
    # See src/system/system_health_monitor.py and
    # docs/SYSTEM_HEALTH_AUDIO.md.
    system_health_monitor = build_system_health_monitor(config)

    logger.info("Atlas started. Press '%s' to quit.", QUIT_KEY)

    prev_time = time.time()
    previous_raw_frame = None
    frame_count = 0
    try:
        while True:
            try:
                frame = video.read_frame()
            except VideoSourceError as exc:
                if not video.is_live_source():
                    # A video FILE reaching normal EOF is expected, not a
                    # failure -- exit exactly as before, never announced
                    # as a camera failure. See VideoSource.is_live_source().
                    logger.error(str(exc))
                    break
                # A LIVE source failing a read is a genuine candidate for
                # CAMERA_FEED_LOST -- record it and keep retrying rather
                # than exiting on the very first failed read; the monitor
                # requires several consecutive failures before confirming
                # and speaking anything.
                logger.debug("Live camera frame read failed: %s", exc)
                system_health_monitor.record_frame_read_result(False)
                system_health_monitor.record_frame_for_blocked_check(None, time.time())
                process_system_health(
                    system_health_monitor, event_policy, speech_queue, audio_worker,
                    time.time(),
                )
                time.sleep(0.05)
                continue

            system_health_monitor.record_frame_read_result(True)

            startup_announcer.notify_frame_read()
            startup_announcer.tick()

            frame_height, frame_width = frame.shape[:2]
            system_health_monitor.record_frame_for_blocked_check(frame, time.time())
            try:
                detections = detector.detect(frame)
            except Exception as exc:
                logger.debug("Detector call failed: %s", exc)
                system_health_monitor.record_detector_result(False)
                detections = []
            else:
                system_health_monitor.record_detector_result(True)
            tracked_objects = tracker.update(detections)

            motion_estimate = None
            if motion_estimator is not None and previous_raw_frame is not None:
                excluded_regions = (
                    [obj.bbox for obj in tracked_objects]
                    if motion_cfg.get("exclude_foreground", False)
                    else None
                )
                motion_estimate = motion_estimator.estimate(
                    previous_raw_frame, frame, excluded_regions
                )
                frame_count += 1
                if frame_count % 30 == 0:
                    logger.debug(
                        "Camera motion: dx=%.2f dy=%.2f rotation=%.2fdeg "
                        "confidence=%.2f tracked=%d inliers=%d valid=%s%s",
                        motion_estimate.dx,
                        motion_estimate.dy,
                        motion_estimate.rotation_degrees,
                        motion_estimate.confidence,
                        motion_estimate.tracked_feature_count,
                        motion_estimate.inlier_count,
                        motion_estimate.valid,
                        f" reason={motion_estimate.reason_invalid}"
                        if not motion_estimate.valid
                        else "",
                    )

            # Parallel data stream: computed every frame alongside
            # motion_estimate but not fed into trajectory_predictor or
            # path_intersection_analyzer, which still use
            # TrackedObject.position_history. Only consumed by the
            # debug-only draw_compensated_motion call below.
            compensated_motions = {}
            if motion_compensator is not None:
                compensated_motions = motion_compensator.compensate(
                    tracked_objects, motion_estimate
                )

            # Core perception: always computed in every mode; CLI flags
            # control diagnostics and rendering only. Raw trajectory
            # prediction is included because resolve_motions_for_tracks
            # needs it as the RAW_FALLBACK velocity source (see
            # compute_core_motion). Reused by the debug block below.
            core = compute_core_motion(
                tracked_objects, compensated_motions, motion_estimate,
                trajectory_predictor, motion_state_filter,
                path_intersection_analyzer, frame_width, frame_height,
            )

            # Relative proximity + confirmed approach: also unconditional,
            # every mode -- perception continues regardless of whether
            # startup speech is still owning the audio system.
            hazard_perception = compute_hazard_perception(
                tracked_objects, core, relative_proximity_estimator, approach_estimator,
                path_intersection_analyzer, frame_width, frame_height,
            )

            # Conflict Imminence V1: also unconditional, every mode,
            # alongside hazard_perception above -- a conservative,
            # image-space, frame-step-based signal (see
            # src/conflict_imminence.py) feeding ThreatAssessmentEngine's
            # second Level 5 pathway below. Independent of
            # threat_assessment_engine's own state.
            conflict_imminence_by_track: dict[int, ConflictImminence] = {}
            if conflict_imminence_estimator is not None:
                conflict_imminence_by_track = conflict_imminence_estimator.estimate_for_tracks(
                    tracked_objects, core.filtered_motions, core.resolved_predictions,
                    core.resolved_intersection_results, time.time(), frame_count,
                )

            # Canonical threat classification: also unconditional, every
            # mode, alongside hazard_perception above -- the single
            # source of truth for Level 1-5 (see src/threat_assessment.py).
            # AudioHazardResolver below consumes this directly rather than
            # recomputing its own classification/temporal state. When
            # threat assessment is disabled (threat_assessment.enabled:
            # false), there is no fallback classifier -- hazard audio is
            # silent too, since there is deliberately only one source of
            # truth now.
            threat_assessments: dict[int, ThreatAssessment] = {}
            if threat_assessment_engine is not None:
                threat_assessments = threat_assessment_engine.assess_for_tracks(
                    tracked_objects, core.filtered_motions, hazard_perception.approach_results,
                    core.resolved_intersection_results, hazard_perception.proximity_by_track,
                    time.time(), frame_count, conflict_imminence_by_track,
                )

            # Critical system-health check runs first every frame -- if a
            # camera/detector failure is active, ordinary operational
            # audio (hazard warnings, scene summaries, uncertain-object
            # messages) is suppressed entirely for this frame so a
            # system-failure message is never drowned out or followed by
            # now-unreliable perception audio. See
            # src/system/system_health_monitor.py.
            health_trace = process_system_health(
                system_health_monitor, event_policy, speech_queue, audio_worker, time.time(),
            )

            # Real-time spoken warnings, from resolved perception results
            # only -- hazard-level RESOLUTION and speech are blocked
            # entirely while the startup announcement still owns the
            # audio system -- see process_audio_events()/StartupAudioGate.
            # Perception, event generation, and speech playback stay
            # strictly separated: main.py never builds spoken text itself.
            if not health_trace.operational_audio_suppressed:
                process_audio_events(
                    startup_announcer, startup_audio_gate, audio_hazard_resolver,
                    operating_mode_policy, audio_event_builder, scene_summarizer, audio_worker,
                    speech_queue, audio_sequencing_tracer, tracked_objects, threat_assessments,
                    time.time(),
                )

            # Captured BEFORE any drawing below, which mutates `frame` in
            # place -- next iteration's motion estimate must be computed
            # from clean pixels, not this frame's drawn boxes/text.
            previous_raw_frame = frame.copy()

            visualizer.draw_tracked_objects(frame, tracked_objects, core.filtered_motions)
            left_px, right_px = region_analyzer.get_region_boundaries_px(
                frame_width
            )
            visualizer.draw_region_lines(frame, left_px, right_px)

            zone = region_analyzer.compute_attention_zone(
                frame_width, frame_height
            )
            visualizer.draw_attention_zone(frame, zone)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            if args.debug:
                visualizer.draw_trajectories(
                    frame,
                    tracked_objects,
                    core.predictions,
                    trajectory_cfg["trajectory_draw_history"],
                    trajectory_cfg["trajectory_history_points_to_draw"],
                )

                # RAW corridor check -- debug/validation comparison only.
                # Nothing in compute_core_motion's core pipeline consumes
                # this (unlike core.resolved_intersection_results, the
                # actual decision), so it's computed here, not promoted.
                corridor_polygon = path_intersection_analyzer.build_corridor_polygon(
                    frame_width, frame_height
                )
                intersection_results = {
                    obj.track_id: path_intersection_analyzer.analyze(
                        obj.track_id, core.predictions[obj.track_id],
                        frame_width, frame_height,
                    )
                    for obj in tracked_objects
                }
                visualizer.draw_corridor(
                    frame, corridor_polygon, corridor_cfg["corridor_draw_enabled"]
                )
                visualizer.draw_path_intersections(
                    frame,
                    tracked_objects,
                    core.resolved_intersection_results,
                    corridor_cfg["intersection_marker_radius"],
                )

                if motion_cfg.get("draw_debug", False):
                    visualizer.draw_motion_estimate(frame, motion_estimate)

                if args.validate_compensation:
                    # Developer-only validation mode: a richer per-object
                    # block (status + camera confidence, superseding the
                    # plainer draw_compensated_motion below) plus an
                    # aggregate summary panel. Never read by
                    # trajectory_predictor/path_intersection_analyzer above.
                    per_object_validation = build_per_object_validation(
                        tracked_objects,
                        compensated_motions,
                        motion_estimate,
                        compensation_stationary_threshold_px,
                        core.resolved_motions,
                        intersection_results,
                        core.resolved_intersection_results,
                        core.filtered_motions,
                    )
                    validation_stats = build_validation_stats(
                        per_object_validation, motion_estimate
                    )
                    visualizer.draw_validation_panel(
                        frame, tracked_objects, validation_stats,
                        per_object_validation, core.resolved_predictions,
                    )

                    if should_log_validation_summary(frame_count):
                        logger.info(
                            "frame=%d tracks=%d camera=(%.1f,%.1f) conf=%.2f "
                            "raw_avg=%.1f comp_avg=%.1f reduction=%.1f%% "
                            "stationary=%d moving=%d camera_dominated=%d "
                            "uncertain=%d",
                            frame_count,
                            validation_stats.active_track_count,
                            validation_stats.camera_dx,
                            validation_stats.camera_dy,
                            validation_stats.camera_confidence,
                            validation_stats.average_raw_speed,
                            validation_stats.average_compensated_speed,
                            validation_stats.reduction_percent,
                            validation_stats.stationary_count,
                            validation_stats.moving_count,
                            validation_stats.camera_dominated_count,
                            validation_stats.uncertain_count,
                        )

                    if compensation_csv_logger is not None and should_log_validation_summary(
                        frame_count, interval=COMPENSATION_CSV_LOG_INTERVAL_FRAMES
                    ):
                        compensation_csv_logger.log_row(
                            timestamp=datetime.now().isoformat(),
                            frame_number=frame_count,
                            camera_motion=motion_estimate,
                            stats=validation_stats,
                        )

                    if per_object_csv_logger is not None:
                        # Unsampled (every frame) -- a controlled-clip
                        # validation export needs complete data, unlike
                        # the sampled aggregate CSV/console log above.
                        per_object_csv_logger.log_rows(
                            frame_index=frame_count,
                            timestamp=datetime.now().isoformat(),
                            camera_motion=motion_estimate,
                            per_object=per_object_validation,
                        )
                elif motion_compensator is not None:
                    visualizer.draw_compensated_motion(
                        frame, tracked_objects, compensated_motions, core.filtered_motions
                    )

                visualizer.draw_motion_legend(frame)

                visualizer.draw_debug_info(
                    frame,
                    fps,
                    frame_width,
                    frame_height,
                    left_px,
                    right_px,
                    zone,
                )

            visualizer.show(frame)

            if validation_writer is not None:
                validation_writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(QUIT_KEY):
                logger.info("Quit key pressed. Shutting down.")
                break
    finally:
        video.release()
        if validation_writer is not None:
            validation_writer.release()
        if compensation_csv_logger is not None:
            compensation_csv_logger.close()
        if per_object_csv_logger is not None:
            per_object_csv_logger.close()
        # Terminates any in-flight `say`/`afplay` process so quitting
        # mid-speech never leaves an orphaned process behind.
        startup_announcer.shutdown()
        audio_worker.shutdown()
        visualizer.close()
        logger.info("Resources released. Exited cleanly.")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
