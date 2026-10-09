"""Unit tests for the PURE metric-computation helpers in
scripts/benchmark_detectors.py. No camera, model, network, or real YOLO
inference -- only the small, importable functions are tested here, with
synthetic position/frame data, matching this project's established
testing convention. The CLI/video-loop glue (run_model_benchmark) is
intentionally NOT exercised by these tests -- it requires a real video
and a real (possibly auto-downloading) YOLO model, which belongs to
manual/integration validation, not the unit-test suite.
"""

from scripts.benchmark_detectors import (
    BENCHMARK_CSV_COLUMNS,
    ModelBenchmarkResult,
    compute_bbox_jitter,
    compute_missed_frames,
    get_peak_memory_mb,
)

# --- compute_bbox_jitter -----------------------------------------------


def test_bbox_jitter_zero_for_perfectly_still_object() -> None:
    positions = [(100, 100)] * 5
    sizes = [(20, 20)] * 5
    assert compute_bbox_jitter(positions, sizes) == 0.0


def test_bbox_jitter_nonzero_for_moving_center() -> None:
    positions = [(100, 100), (110, 100), (120, 100)]
    sizes = [(20, 20), (20, 20), (20, 20)]
    jitter = compute_bbox_jitter(positions, sizes)
    assert jitter > 0.0


def test_bbox_jitter_normalized_by_box_diagonal() -> None:
    # Same 10px displacement, but a much larger box -- normalized jitter
    # should be smaller for the larger box.
    small_box_jitter = compute_bbox_jitter(
        [(100, 100), (110, 100)], [(20, 20), (20, 20)]
    )
    large_box_jitter = compute_bbox_jitter(
        [(100, 100), (110, 100)], [(200, 200), (200, 200)]
    )
    assert large_box_jitter < small_box_jitter


def test_bbox_jitter_returns_zero_for_fewer_than_two_samples() -> None:
    assert compute_bbox_jitter([(100, 100)], [(20, 20)]) == 0.0
    assert compute_bbox_jitter([], []) == 0.0


def test_bbox_jitter_handles_zero_diagonal_safely() -> None:
    # Degenerate zero-size boxes -- must not raise a division error.
    jitter = compute_bbox_jitter([(100, 100), (110, 100)], [(0, 0), (0, 0)])
    assert jitter == 0.0


# --- compute_missed_frames -----------------------------------------------


def test_missed_frames_zero_for_consecutive_appearances() -> None:
    assert compute_missed_frames([1, 2, 3, 4]) == 0


def test_missed_frames_counts_gaps() -> None:
    # Seen at 1, 2, then a gap (missed 3, 4), then seen at 5.
    assert compute_missed_frames([1, 2, 5]) == 2


def test_missed_frames_handles_unsorted_input() -> None:
    assert compute_missed_frames([5, 1, 2]) == 2


def test_missed_frames_returns_zero_for_fewer_than_two_samples() -> None:
    assert compute_missed_frames([1]) == 0
    assert compute_missed_frames([]) == 0


def test_missed_frames_sums_multiple_gaps() -> None:
    # Gap of 1 (frame 3) then gap of 3 (frames 5,6,7).
    assert compute_missed_frames([1, 2, 4, 8]) == 1 + 3


# --- get_peak_memory_mb ---------------------------------------------------


def test_get_peak_memory_mb_returns_a_positive_number_or_none() -> None:
    result = get_peak_memory_mb()
    assert result is None or result > 0.0


# --- ModelBenchmarkResult --------------------------------------------------


def test_model_benchmark_result_computes_derived_fields() -> None:
    result = ModelBenchmarkResult(
        model_name="yolov8n.pt",
        input_width=1920,
        input_height=1080,
        avg_inference_time_ms=40.0,
        avg_frame_time_ms=50.0,
        total_detections=100,
        frames_processed=50,
        missed_frames_vehicles=2,
        track_creation_count=3,
        avg_bbox_jitter_normalized=0.05,
        raw_moving_false_positive_frames=10,
        filtered_moving_false_positive_frames=1,
        uncertain_result_count=4,
        peak_memory_mb=500.0,
    )

    assert result.avg_fps == 20.0  # 1000 / 50
    assert result.avg_detections_per_frame == 2.0  # 100 / 50


def test_model_benchmark_result_avg_fps_zero_when_frame_time_zero() -> None:
    result = ModelBenchmarkResult(
        model_name="yolov8n.pt", input_width=100, input_height=100,
        avg_inference_time_ms=0.0, avg_frame_time_ms=0.0, total_detections=0,
        frames_processed=0, missed_frames_vehicles=0, track_creation_count=0,
        avg_bbox_jitter_normalized=0.0, raw_moving_false_positive_frames=0,
        filtered_moving_false_positive_frames=0, uncertain_result_count=0,
        peak_memory_mb=None,
    )
    assert result.avg_fps == 0.0
    assert result.avg_detections_per_frame == 0.0


def test_model_benchmark_result_as_csv_row_matches_column_count() -> None:
    result = ModelBenchmarkResult(
        model_name="yolov8s.pt", input_width=640, input_height=480,
        avg_inference_time_ms=30.0, avg_frame_time_ms=40.0, total_detections=20,
        frames_processed=10, missed_frames_vehicles=1, track_creation_count=2,
        avg_bbox_jitter_normalized=0.02, raw_moving_false_positive_frames=3,
        filtered_moving_false_positive_frames=0, uncertain_result_count=1,
        peak_memory_mb=None,
    )
    row = result.as_csv_row()
    assert len(row) == len(BENCHMARK_CSV_COLUMNS)
    assert row[0] == "yolov8s.pt"
    assert row[-1] == ""  # peak_memory_mb=None -> blank, not "None"
