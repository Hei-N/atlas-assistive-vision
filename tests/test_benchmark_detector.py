"""Unit/integration tests for scripts/benchmark_detector.py.

No real YOLO weights are loaded and no network access happens: `src.
object_detector.YOLO` is monkeypatched with a tiny fake returning
pre-scripted per-call results, so ObjectDetector.detect() -- the REAL,
unmodified production method -- runs end-to-end deterministically.
Tiny synthetic images/videos are generated into pytest's tmp_path
rather than touching any real repo asset. Every test that writes
output passes an explicit --output-dir inside tmp_path so no test run
ever creates or modifies anything under the real logs/ directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from scripts.benchmark_detector import (
    DETECTIONS_CSV_COLUMNS,
    BenchmarkManifestError,
    BenchmarkOutputExistsError,
    BenchmarkSourceError,
    iou,
    mean_or_none,
    median_or_none,
    parse_args,
    percentile_or_none,
    run_benchmark,
)

COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
REAL_CONFIG_PATH = "config/settings.yaml"


class FakeBox:
    def __init__(self, cls_id: int, conf: float, xyxy=(1, 1, 10, 10)):
        self.cls = np.array([cls_id])
        self.conf = np.array([conf])
        self.xyxy = np.array([list(xyxy)], dtype=float)


class FakeResult:
    def __init__(self, boxes: list[FakeBox]):
        self.boxes = boxes


@pytest.fixture
def fake_yolo(monkeypatch):
    """Patches src.object_detector.YOLO. Set fake_yolo["queue"] to a
    list of per-frame box lists before running the benchmark -- frame N
    gets queue[N] (or an empty result once the queue is exhausted)."""
    holder = {"queue": []}

    class _FakeYOLO:
        def __init__(self, model_name):
            self.model_name = model_name
            self.names = dict(COCO_NAMES)
            self.device = "cpu"
            self.call_count = 0

        def __call__(self, frame, verbose=False):
            boxes = holder["queue"][self.call_count] if self.call_count < len(holder["queue"]) else []
            self.call_count += 1
            return [FakeResult(boxes)]

        def to(self, device):
            self.device = device
            return self

    monkeypatch.setattr("src.object_detector.YOLO", _FakeYOLO)
    return holder


def make_tiny_video(path: Path, num_frames: int, width: int = 32, height: int = 24, fps: float = 10.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        writer.write(np.full((height, width, 3), i % 255, dtype=np.uint8))
    writer.release()


def make_tiny_image(path: Path, width: int = 32, height: int = 24) -> None:
    cv2.imwrite(str(path), np.full((height, width, 3), 100, dtype=np.uint8))


def read_csv_rows(output_dir: Path) -> list[dict]:
    with (output_dir / "detections.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(output_dir: Path, name: str) -> dict:
    return json.loads((output_dir / name).read_text())


def make_args(tmp_path: Path, source: Path, **overrides) -> "argparse.Namespace":
    argv = ["--source", str(source), "--output-dir", str(tmp_path / "out"), "--config", REAL_CONFIG_PATH]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is False:
            continue
        else:
            argv += [flag, str(value)]
    return parse_args(argv)


# --- pure helper functions --------------------------------------------------


def test_mean_or_none_empty_and_values() -> None:
    assert mean_or_none([]) is None
    assert mean_or_none([2.0, 4.0]) == 3.0


def test_median_or_none_odd_and_even() -> None:
    assert median_or_none([]) is None
    assert median_or_none([1.0, 2.0, 3.0]) == 2.0
    assert median_or_none([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_percentile_or_none_matches_hand_computed_values() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile_or_none([], 50) is None
    assert percentile_or_none([7.0], 95) == 7.0
    assert percentile_or_none(values, 50) == 3.0
    assert percentile_or_none(values, 95) == pytest.approx(4.8)


def test_iou_full_overlap_is_one() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_no_overlap_is_zero() -> None:
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap() -> None:
    # Two 10x10 boxes overlapping in a 5x10 strip: intersection=50,
    # union = 100+100-50 = 150.
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


# --- 1/2: detector configuration reuse and override -------------------------


def test_default_config_matches_production_settings(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[]]

    args = make_args(tmp_path, image_path)
    output_dir = run_benchmark(args)

    with open(REAL_CONFIG_PATH) as f:
        real_config = yaml.safe_load(f)
    metadata = read_json(output_dir, "run_metadata.json")
    assert metadata["model_name"] == real_config["model"]["name"]
    assert metadata["confidence_threshold"] == real_config["model"]["confidence_threshold"]


def test_explicit_overrides_do_not_modify_production_config(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[]]
    before = Path(REAL_CONFIG_PATH).read_text()

    args = make_args(tmp_path, image_path, model="yolov8n.pt", confidence=0.9)
    output_dir = run_benchmark(args)

    metadata = read_json(output_dir, "run_metadata.json")
    assert metadata["model_name"] == "yolov8n.pt"
    assert metadata["confidence_threshold"] == 0.9
    after = Path(REAL_CONFIG_PATH).read_text()
    assert before == after


# --- 3/4/5: supported source kinds ------------------------------------------


def test_single_image_source_is_supported(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[FakeBox(2, 0.9)]]

    output_dir = run_benchmark(make_args(tmp_path, image_path))

    rows = read_csv_rows(output_dir)
    assert len(rows) == 1
    assert rows[0]["raw_class_name"] == "car"


def test_single_video_source_is_supported(tmp_path, fake_yolo) -> None:
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=3)
    fake_yolo["queue"] = [[FakeBox(2, 0.9)], [], [FakeBox(5, 0.8)]]

    output_dir = run_benchmark(make_args(tmp_path, video_path))

    summary = read_json(output_dir, "summary.json")
    assert summary["processed_frames"] == 3
    assert summary["total_detections"] == 2


def test_directory_source_is_supported(tmp_path, fake_yolo) -> None:
    directory = tmp_path / "clips"
    directory.mkdir()
    make_tiny_image(directory / "a.jpg")
    make_tiny_image(directory / "b.png")
    fake_yolo["queue"] = [[FakeBox(2, 0.9)], [FakeBox(1, 0.6)]]

    output_dir = run_benchmark(make_args(tmp_path, directory))

    rows = read_csv_rows(output_dir)
    source_ids = {row["source_id"] for row in rows}
    assert source_ids == {"a", "b"}


# --- 6/7: source errors ------------------------------------------------------


def test_missing_source_produces_a_clear_error(tmp_path, fake_yolo) -> None:
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(BenchmarkSourceError, match="does not exist"):
        run_benchmark(make_args(tmp_path, missing))


def test_unsupported_file_type_produces_a_clear_error(tmp_path, fake_yolo) -> None:
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not a supported source type")
    with pytest.raises(BenchmarkSourceError, match="Unsupported file type"):
        run_benchmark(make_args(tmp_path, unsupported))


# --- 8/9: output directory handling ------------------------------------------


def test_existing_result_directory_is_not_overwritten_by_default(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[]]

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "preexisting.txt").write_text("do not touch")

    args = make_args(tmp_path, image_path)
    with pytest.raises(BenchmarkOutputExistsError):
        run_benchmark(args)
    assert (output_dir / "preexisting.txt").exists()

    args_overwrite = make_args(tmp_path, image_path, overwrite=True)
    run_benchmark(args_overwrite)  # succeeds with --overwrite
    assert (output_dir / "run_metadata.json").exists()


def test_unique_result_directory_is_created(tmp_path, fake_yolo, monkeypatch) -> None:
    # --config must be absolute since we're about to chdir away from the
    # repo root -- the relative default would no longer resolve.
    absolute_config = str(Path(__file__).resolve().parent.parent / REAL_CONFIG_PATH)
    monkeypatch.chdir(tmp_path)
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[]]

    args = parse_args(["--source", str(image_path), "--config", absolute_config])
    output_dir = run_benchmark(args)

    assert output_dir.exists()
    assert str(output_dir).startswith("logs/detector_benchmarks")
    assert output_dir.resolve().is_relative_to(tmp_path)


# --- 10/11/12: output schemas -----------------------------------------------


def test_run_metadata_schema(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[]]

    output_dir = run_benchmark(make_args(tmp_path, image_path))
    metadata = read_json(output_dir, "run_metadata.json")

    expected_keys = {
        "schema_version", "benchmark_tool_version", "timestamp", "source_path",
        "source_type", "source_entry_count", "model_name", "model_file_sha256",
        "confidence_threshold", "iou_threshold", "input_resolution", "device",
        "ultralytics_version", "opencv_version", "python_version", "operating_system",
        "frame_step", "max_frames", "total_frames_processed", "command_line_arguments",
        "git_commit", "git_commit_status",
    }
    assert expected_keys.issubset(metadata.keys())
    assert metadata["git_commit"] is None
    assert "no .git repository" in metadata["git_commit_status"]


def test_detections_csv_schema(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[FakeBox(2, 0.9)]]

    output_dir = run_benchmark(make_args(tmp_path, image_path))
    with (output_dir / "detections.csv").open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == DETECTIONS_CSV_COLUMNS


def test_summary_json_schema(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[]]

    output_dir = run_benchmark(make_args(tmp_path, image_path))
    summary = read_json(output_dir, "summary.json")

    expected_keys = {
        "observational_baseline", "requires_ground_truth_for", "protected_classes",
        "priority_classes", "total_frames", "processed_frames", "failed_frames",
        "total_detections", "detections_per_raw_class", "detections_per_normalized_class",
        "mean_confidence_by_raw_class", "median_confidence_by_raw_class",
        "confidence_quantiles_by_raw_class", "average_detections_per_frame",
        "percent_frames_containing_class", "mean_inference_time_ms",
        "median_inference_time_ms", "p95_inference_time_ms", "mean_total_frame_time_ms",
        "effective_processing_fps", "source_fps", "real_time_factor",
        "frame_time_budget_ms", "frames_exceeding_frame_time_budget",
        "temporal_consistency_proxies",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["observational_baseline"] is True
    for forbidden in ("precision", "recall", "f1_score", "mean_average_precision"):
        assert forbidden in summary["requires_ground_truth_for"]


# --- 13/14: protected/priority classes present at zero count ----------------


def test_protected_classes_present_at_zero_count(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[FakeBox(1, 0.6)]]  # only a bicycle this run

    output_dir = run_benchmark(make_args(tmp_path, image_path))
    summary = read_json(output_dir, "summary.json")

    for name in ("car", "bus", "truck"):
        assert summary["detections_per_raw_class"][name] == 0


def test_priority_classes_present_at_zero_count(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[FakeBox(2, 0.9)]]  # only a car this run

    output_dir = run_benchmark(make_args(tmp_path, image_path))
    summary = read_json(output_dir, "summary.json")

    for name in ("bicycle", "motorcycle"):
        assert summary["detections_per_raw_class"][name] == 0


# --- 15/16/17: class naming honesty ------------------------------------------


def test_raw_and_normalized_class_names_remain_distinct(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[FakeBox(2, 0.9)]]  # car

    output_dir = run_benchmark(make_args(tmp_path, image_path))
    row = read_csv_rows(output_dir)[0]

    assert row["raw_class_name"] == "car"
    assert row["atlas_config_class_name"] == "car"
    assert row["atlas_normalized_class_name"] == "Vehicle"
    assert row["raw_class_name"] != row["atlas_normalized_class_name"]


def test_bicycle_is_not_relabeled_as_ebike(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[FakeBox(1, 0.6)]]  # bicycle

    output_dir = run_benchmark(make_args(tmp_path, image_path))
    row = read_csv_rows(output_dir)[0]

    assert row["raw_class_name"] == "bicycle"
    assert row["atlas_normalized_class_name"] == "Bicycle"
    assert "e-bike" not in row["atlas_normalized_class_name"].lower()
    assert "ebike" not in row["atlas_normalized_class_name"].lower()
    assert row["micromobility_candidate"] == "True"


def test_motorcycle_is_not_relabeled_as_scooter(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[FakeBox(3, 0.6)]]  # motorcycle

    output_dir = run_benchmark(make_args(tmp_path, image_path))
    row = read_csv_rows(output_dir)[0]

    assert row["raw_class_name"] == "motorcycle"
    assert row["atlas_normalized_class_name"] == "Motorcycle"
    assert "scooter" not in row["atlas_normalized_class_name"].lower()
    assert "moped" not in row["atlas_normalized_class_name"].lower()


# --- 18/19: empty vs. failed frames ------------------------------------------


def test_empty_frames_are_valid_and_not_detector_failures(tmp_path, fake_yolo) -> None:
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=3)
    fake_yolo["queue"] = [[], [], []]  # every frame detects nothing

    output_dir = run_benchmark(make_args(tmp_path, video_path))
    summary = read_json(output_dir, "summary.json")

    assert summary["processed_frames"] == 3
    assert summary["failed_frames"] == 0
    assert summary["total_detections"] == 0
    rows = read_csv_rows(output_dir)
    assert len(rows) == 3  # one frame-level row per empty frame
    assert all(row["detections_in_frame"] == "0" for row in rows)


def test_failed_frame_reads_are_counted(tmp_path, fake_yolo) -> None:
    corrupt_image = tmp_path / "corrupt.jpg"
    corrupt_image.write_bytes(b"not a real jpeg")
    fake_yolo["queue"] = []

    output_dir = run_benchmark(make_args(tmp_path, corrupt_image))
    summary = read_json(output_dir, "summary.json")

    assert summary["failed_frames"] == 1
    assert summary["processed_frames"] == 0


# --- 20/21/22: timing ---------------------------------------------------------


def test_inference_timing_is_recorded(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[FakeBox(2, 0.9)]]

    output_dir = run_benchmark(make_args(tmp_path, image_path))
    row = read_csv_rows(output_dir)[0]

    assert float(row["inference_time_ms"]) >= 0.0
    assert float(row["total_frame_time_ms"]) >= float(row["inference_time_ms"])


def test_average_median_p95_timing_in_summary_are_consistent(tmp_path, fake_yolo) -> None:
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=5)
    fake_yolo["queue"] = [[]] * 5

    output_dir = run_benchmark(make_args(tmp_path, video_path))
    summary = read_json(output_dir, "summary.json")

    assert summary["mean_inference_time_ms"] is not None
    assert summary["median_inference_time_ms"] is not None
    assert summary["p95_inference_time_ms"] is not None
    assert summary["p95_inference_time_ms"] >= summary["median_inference_time_ms"] >= 0.0


def test_effective_fps_calculation_is_correct() -> None:
    """Exact arithmetic check via a controlled accumulator (real
    wall-clock frame times from a mocked detector are microsecond-scale
    and too noisy to reverse-engineer exactly through JSON rounding --
    see build_summary(), which correctly derives effective_fps from the
    UNROUNDED mean, only rounding for display)."""
    from scripts.benchmark_detector import RunAccumulator, TemporalConsistencyProxy, build_summary

    accumulator = RunAccumulator()
    accumulator.seed_classes(["car"], ["Vehicle"])
    accumulator.total_frame_times_ms = [10.0, 20.0, 30.0]  # mean = 20.0ms -> 50.0 fps
    accumulator.inference_times_ms = [5.0, 5.0, 5.0]
    accumulator.processed_frames = 3
    accumulator.total_frames = 3

    summary = build_summary(accumulator, TemporalConsistencyProxy())

    assert summary["mean_total_frame_time_ms"] == 20.0
    assert summary["effective_processing_fps"] == 50.0


# --- 23/24: sampling ------------------------------------------------------------


def test_frame_step_sampling_works(tmp_path, fake_yolo) -> None:
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=6)
    fake_yolo["queue"] = [[]] * 6

    output_dir = run_benchmark(make_args(tmp_path, video_path, frame_step=2))
    summary = read_json(output_dir, "summary.json")

    assert summary["processed_frames"] == 3  # frames 0, 2, 4


def test_max_frames_limit_works(tmp_path, fake_yolo) -> None:
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=5)
    fake_yolo["queue"] = [[]] * 5

    output_dir = run_benchmark(make_args(tmp_path, video_path, max_frames=2))
    summary = read_json(output_dir, "summary.json")

    assert summary["processed_frames"] == 2


# --- 25/26: annotation --------------------------------------------------------


def test_annotation_output_disabled_by_default(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[FakeBox(2, 0.9)]]

    output_dir = run_benchmark(make_args(tmp_path, image_path))
    assert not (output_dir / "annotated").exists()


def test_annotation_output_works_when_explicitly_enabled(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[FakeBox(2, 0.9)]]

    output_dir = run_benchmark(make_args(tmp_path, image_path, annotate=True))
    annotated_files = list((output_dir / "annotated").glob("*.jpg"))
    assert len(annotated_files) == 1


# --- 27: no invented track IDs -----------------------------------------------


def test_no_track_id_invented_when_tracking_disabled(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[FakeBox(2, 0.9)]]

    output_dir = run_benchmark(make_args(tmp_path, image_path))
    row = read_csv_rows(output_dir)[0]
    assert row["track_id"] == ""


def test_track_id_populated_when_tracking_enabled(tmp_path, fake_yolo) -> None:
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=2)
    fake_yolo["queue"] = [[FakeBox(2, 0.9)], [FakeBox(2, 0.9)]]

    output_dir = run_benchmark(make_args(tmp_path, video_path, track=True))
    rows = read_csv_rows(output_dir)
    assert all(row["track_id"] != "" for row in rows)


# --- 28/29: manifest -----------------------------------------------------------


def test_manifest_parsing_works(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "clip_001", "path": str(image_path), "source_type": "image",
                    "scene_tags": ["daytime"], "lighting": "bright",
                    "camera_motion": "static", "expected_relevant_classes": ["car"],
                    "notes": "test entry",
                }
            ]
        )
    )
    fake_yolo["queue"] = [[FakeBox(2, 0.9)]]

    output_dir = run_benchmark(make_args(tmp_path, manifest_path))
    rows = read_csv_rows(output_dir)
    assert rows[0]["source_id"] == "clip_001"


@pytest.mark.parametrize(
    "entries",
    [
        [{"path": "x.jpg", "source_type": "image"}],  # missing id
        [{"id": "a", "path": "x.jpg", "source_type": "audio"}],  # invalid source_type
        [{"id": "a", "path": "does_not_exist.jpg", "source_type": "image"}],  # missing file
        [{"id": "a", "path": "x.jpg", "source_type": "image"}, {"id": "a", "path": "y.jpg", "source_type": "image"}],  # duplicate id
    ],
)
def test_invalid_manifest_entries_fail_clearly(tmp_path, fake_yolo, entries) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(entries))
    with pytest.raises(BenchmarkManifestError):
        run_benchmark(make_args(tmp_path, manifest_path))


# --- 30: source files remain unchanged ---------------------------------------


def test_source_files_remain_unchanged(tmp_path, fake_yolo) -> None:
    image_path = tmp_path / "frame.jpg"
    make_tiny_image(image_path)
    fake_yolo["queue"] = [[FakeBox(2, 0.9)]]
    before = image_path.read_bytes()

    run_benchmark(make_args(tmp_path, image_path))

    assert image_path.read_bytes() == before


# --- 31: production detector construction is unaffected --------------------


def test_production_detector_construction_still_works(tmp_path, fake_yolo) -> None:
    import main as atlas_main

    config = atlas_main.load_config(REAL_CONFIG_PATH)
    region_analyzer = atlas_main.build_region_analyzer(config)
    detector = atlas_main.build_object_detector(config, region_analyzer)
    fake_yolo["queue"] = [[FakeBox(2, 0.9)]]

    detections = detector.detect(np.zeros((24, 32, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].class_name == "car"


# --- synthetic acceptance test ------------------------------------------------


def test_synthetic_acceptance_mixed_classes_and_empty_frames(tmp_path, fake_yolo) -> None:
    """Mirrors the milestone's required synthetic acceptance scenario:
    car/bus/truck/bicycle/motorcycle/person detections plus empty
    frames, verified end-to-end -- exact counts, confidence stats,
    timing summary, CSV rows, zero-count protected/priority classes,
    metadata, and no e-bike/scooter mislabeling. No production audio or
    perception component is imported or invoked anywhere in this tool.
    """
    video_path = tmp_path / "clip.mp4"
    make_tiny_video(video_path, num_frames=6)
    fake_yolo["queue"] = [
        [FakeBox(2, 0.90), FakeBox(5, 0.80)],  # frame 0: car, bus
        [FakeBox(7, 0.70)],                     # frame 1: truck
        [FakeBox(1, 0.60), FakeBox(3, 0.50)],   # frame 2: bicycle, motorcycle
        [FakeBox(0, 0.95)],                     # frame 3: person
        [],                                      # frame 4: nothing
        [FakeBox(2, 0.85)],                     # frame 5: car
    ]

    output_dir = run_benchmark(make_args(tmp_path, video_path))

    rows = read_csv_rows(output_dir)
    assert len(rows) == 8  # 7 detections + 1 empty-frame row for frame 4
    assert sum(1 for r in rows if r["detections_in_frame"] == "0") == 1

    summary = read_json(output_dir, "summary.json")
    assert summary["total_frames"] == 6
    assert summary["processed_frames"] == 6
    assert summary["failed_frames"] == 0
    assert summary["total_detections"] == 7

    assert summary["detections_per_raw_class"] == {
        "bicycle": 1, "bus": 1, "car": 2, "motorcycle": 1, "person": 1, "truck": 1,
    }
    assert summary["detections_per_normalized_class"] == {
        "Bicycle": 1, "Motorcycle": 1, "Person": 1, "Vehicle": 4,
    }
    # Zero-count protected/priority classes still present -- none are
    # zero in this scenario, but every configured class key must exist.
    for name in ("car", "bus", "truck", "bicycle", "motorcycle", "person"):
        assert name in summary["detections_per_raw_class"]

    assert summary["mean_confidence_by_raw_class"]["car"] == pytest.approx((0.90 + 0.85) / 2, abs=1e-4)
    assert summary["percent_frames_containing_class"]["car"] == pytest.approx(100.0 * 2 / 6, abs=1e-2)

    metadata = read_json(output_dir, "run_metadata.json")
    assert metadata["total_frames_processed"] == 6
    assert metadata["source_type"] == "VIDEO"

    for row in rows:
        assert "e-bike" not in row["atlas_normalized_class_name"].lower()
        assert "scooter" not in row["atlas_normalized_class_name"].lower()

    # No production audio/perception module is imported by this tool at all.
    import scripts.benchmark_detector as tool_module
    tool_source = Path(tool_module.__file__).read_text()
    for forbidden_import in ("src.audio", "src.motion", "src.system"):
        assert forbidden_import not in tool_source
