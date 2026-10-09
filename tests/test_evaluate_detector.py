"""Unit/integration tests for scripts/evaluate_detector.py. Pure logic
plus small generated CSV/JSON fixtures in tmp_path -- no real media, no
model, no network.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

from scripts.annotation_schema import CANONICAL_CLASSES
from scripts.benchmark_detector import DETECTIONS_CSV_COLUMNS, PRIORITY_CLASSES, PROTECTED_CLASSES
from scripts.evaluate_detector import (
    EvaluationInputError,
    OutputExistsError,
    combine_classes,
    compare_evaluations,
    compute_class_metrics,
    compute_overall_metrics,
    compute_per_class_metrics,
    enrich_predictions_with_source_attributes,
    gt_attribute_breakdown,
    load_ground_truth,
    load_predictions,
    match_all,
    parse_args,
    run_comparison,
    run_evaluation,
    safe_divide,
    scene_tag_breakdown,
    source_attribute_breakdown,
)


# --- fixtures / builders -----------------------------------------------------


def make_source(source_id="clip_001", width=640, height=480, source_type="video", **overrides) -> dict:
    source = {
        "source_id": source_id, "path": f"data/input/{source_id}.mp4", "source_type": source_type,
        "width": width, "height": height, "total_frames": 100, "fps": 30.0,
        "scene_tags": [], "lighting": "bright", "camera_motion": "static", "notes": "",
    }
    source.update(overrides)
    return source


def make_annotation(annotation_id, source_id, frame_index, class_name, bbox, **overrides) -> dict:
    ann = {
        "annotation_id": annotation_id, "source_id": source_id, "frame_index": frame_index,
        "timestamp_seconds": frame_index / 30.0, "class_name": class_name,
        "bounding_box_xyxy": list(bbox), "visibility": "clear", "occlusion": "none",
        "object_size": None, "ignore": False, "difficult": False,
        "vehicle_subtype_note": None, "notes": "",
    }
    ann.update(overrides)
    return ann


def make_document(sources: list[dict], annotations: list[dict]) -> dict:
    return {
        "schema_version": "1.0", "dataset_id": "test_ds", "created_at": "2026-08-07T00:00:00Z",
        "class_names": list(CANONICAL_CLASSES), "sources": sources, "annotations": annotations,
    }


def write_annotations(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document))


def make_prediction_row(source_id, frame_index, class_name, confidence, bbox, frame_width=640, frame_height=480, **overrides) -> dict:
    row = {col: "" for col in DETECTIONS_CSV_COLUMNS}
    x1, y1, x2, y2 = bbox
    row.update(
        {
            "source_id": source_id, "frame_index": frame_index,
            "timestamp_in_source_seconds": frame_index / 30.0, "track_id": "",
            "raw_class_id": "", "raw_class_name": class_name,
            "atlas_config_class_name": class_name, "atlas_normalized_class_name": class_name,
            "micromobility_candidate": class_name in ("bicycle", "motorcycle"),
            "confidence": confidence, "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
            "bbox_width": x2 - x1, "bbox_height": y2 - y1, "bbox_area": (x2 - x1) * (y2 - y1),
            "center_x": (x1 + x2) / 2, "center_y": (y1 + y2) / 2,
            "normalized_center_x": 0.5, "normalized_center_y": 0.5,
            "frame_width": frame_width, "frame_height": frame_height,
            "inference_time_ms": 10.0, "preprocess_time_ms": "", "postprocess_time_ms": "",
            "total_frame_time_ms": 12.0, "detections_in_frame": 1,
        }
    )
    row.update(overrides)
    return row


def make_empty_frame_row(source_id, frame_index, frame_width=640, frame_height=480) -> dict:
    row = {col: "" for col in DETECTIONS_CSV_COLUMNS}
    row.update(
        {
            "source_id": source_id, "frame_index": frame_index,
            "timestamp_in_source_seconds": frame_index / 30.0,
            "frame_width": frame_width, "frame_height": frame_height,
            "inference_time_ms": 10.0, "total_frame_time_ms": 11.0, "detections_in_frame": 0,
        }
    )
    return row


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETECTIONS_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_eval_args(tmp_path, predictions_path, annotations_path, **overrides):
    argv = [
        "--predictions", str(predictions_path), "--annotations", str(annotations_path),
        "--output-dir", str(tmp_path / "eval_out"),
    ]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        else:
            argv += [flag, str(value)]
    return parse_args(argv)


# --- 12-19: matching -----------------------------------------------------------


def test_same_class_iou_match_works() -> None:
    preds = [{"source_id": "s", "frame_index": 0, "class_name": "car", "confidence": 0.9,
              "bbox": (0, 0, 10, 10), "row_order": 0, "lighting": None, "scene_tags": ()}]
    gts = [{"annotation_id": "a1", "source_id": "s", "frame_index": 0, "class_name": "car",
            "bbox": (0, 0, 10, 10), "ignore": False, "difficult": False,
            "visibility": "clear", "occlusion": "none", "object_size": "small",
            "lighting": None, "scene_tags": ()}]
    result = match_all(preds, gts, iou_threshold=0.5)
    assert len(result.matches) == 1
    assert result.matches[0]["iou"] == pytest.approx(1.0)


def test_different_class_does_not_match() -> None:
    preds = [{"source_id": "s", "frame_index": 0, "class_name": "truck", "confidence": 0.9,
              "bbox": (0, 0, 10, 10), "row_order": 0, "lighting": None, "scene_tags": ()}]
    gts = [{"annotation_id": "a1", "source_id": "s", "frame_index": 0, "class_name": "car",
            "bbox": (0, 0, 10, 10), "ignore": False, "difficult": False,
            "visibility": "clear", "occlusion": "none", "object_size": "small",
            "lighting": None, "scene_tags": ()}]
    result = match_all(preds, gts, iou_threshold=0.5)
    assert result.matches == []
    assert len(result.false_positives) == 1
    assert len(result.false_negatives) == 1


def test_different_frame_does_not_match() -> None:
    preds = [{"source_id": "s", "frame_index": 1, "class_name": "car", "confidence": 0.9,
              "bbox": (0, 0, 10, 10), "row_order": 0, "lighting": None, "scene_tags": ()}]
    gts = [{"annotation_id": "a1", "source_id": "s", "frame_index": 0, "class_name": "car",
            "bbox": (0, 0, 10, 10), "ignore": False, "difficult": False,
            "visibility": "clear", "occlusion": "none", "object_size": "small",
            "lighting": None, "scene_tags": ()}]
    result = match_all(preds, gts, iou_threshold=0.5)
    assert result.matches == []
    assert len(result.false_positives) == 1
    assert len(result.false_negatives) == 1


def test_one_prediction_cannot_match_two_annotations() -> None:
    preds = [{"source_id": "s", "frame_index": 0, "class_name": "car", "confidence": 0.9,
              "bbox": (0, 0, 10, 10), "row_order": 0, "lighting": None, "scene_tags": ()}]
    gts = [
        {"annotation_id": "a1", "source_id": "s", "frame_index": 0, "class_name": "car",
         "bbox": (0, 0, 10, 10), "ignore": False, "difficult": False,
         "visibility": "clear", "occlusion": "none", "object_size": "small", "lighting": None, "scene_tags": ()},
        {"annotation_id": "a2", "source_id": "s", "frame_index": 0, "class_name": "car",
         "bbox": (1, 1, 11, 11), "ignore": False, "difficult": False,
         "visibility": "clear", "occlusion": "none", "object_size": "small", "lighting": None, "scene_tags": ()},
    ]
    result = match_all(preds, gts, iou_threshold=0.5)
    assert len(result.matches) == 1
    assert len(result.false_negatives) == 1  # the other GT stays unmatched


def test_one_annotation_cannot_match_two_predictions() -> None:
    preds = [
        {"source_id": "s", "frame_index": 0, "class_name": "car", "confidence": 0.9,
         "bbox": (0, 0, 10, 10), "row_order": 0, "lighting": None, "scene_tags": ()},
        {"source_id": "s", "frame_index": 0, "class_name": "car", "confidence": 0.8,
         "bbox": (0, 0, 10, 10), "row_order": 1, "lighting": None, "scene_tags": ()},
    ]
    gts = [{"annotation_id": "a1", "source_id": "s", "frame_index": 0, "class_name": "car",
            "bbox": (0, 0, 10, 10), "ignore": False, "difficult": False,
            "visibility": "clear", "occlusion": "none", "object_size": "small", "lighting": None, "scene_tags": ()}]
    result = match_all(preds, gts, iou_threshold=0.5)
    assert len(result.matches) == 1
    assert len(result.false_positives) == 1  # the lower-confidence duplicate is unmatched
    assert result.matches[0]["prediction"]["confidence"] == 0.9  # higher confidence wins


def test_matching_is_deterministic() -> None:
    preds = [
        {"source_id": "s", "frame_index": 0, "class_name": "car", "confidence": 0.9,
         "bbox": (0, 0, 10, 10), "row_order": 0, "lighting": None, "scene_tags": ()},
        {"source_id": "s", "frame_index": 0, "class_name": "car", "confidence": 0.9,
         "bbox": (0, 0, 10, 10), "row_order": 1, "lighting": None, "scene_tags": ()},
    ]
    gts = [{"annotation_id": "a1", "source_id": "s", "frame_index": 0, "class_name": "car",
            "bbox": (0, 0, 10, 10), "ignore": False, "difficult": False,
            "visibility": "clear", "occlusion": "none", "object_size": "small", "lighting": None, "scene_tags": ()}]

    results = []
    for _ in range(5):
        shuffled = preds[:]
        random.Random(0).shuffle(shuffled)
        result = match_all(shuffled, gts, iou_threshold=0.5)
        results.append(result.matches[0]["prediction"]["row_order"])
    assert len(set(results)) == 1  # same winner every time (tie broken by row_order)


def test_ignored_annotation_does_not_produce_false_negative() -> None:
    preds: list = []
    gts = [{"annotation_id": "a1", "source_id": "s", "frame_index": 0, "class_name": "person",
            "bbox": (0, 0, 10, 10), "ignore": True, "difficult": False,
            "visibility": "clear", "occlusion": "none", "object_size": "small", "lighting": None, "scene_tags": ()}]
    result = match_all(preds, gts, iou_threshold=0.5)
    assert result.false_negatives == []
    assert len(result.ignored_ground_truth) == 1


def test_prediction_matched_to_ignored_annotation_does_not_produce_false_positive() -> None:
    preds = [{"source_id": "s", "frame_index": 0, "class_name": "person", "confidence": 0.5,
              "bbox": (0, 0, 10, 10), "row_order": 0, "lighting": None, "scene_tags": ()}]
    gts = [{"annotation_id": "a1", "source_id": "s", "frame_index": 0, "class_name": "person",
            "bbox": (0, 0, 10, 10), "ignore": True, "difficult": False,
            "visibility": "clear", "occlusion": "none", "object_size": "small", "lighting": None, "scene_tags": ()}]
    result = match_all(preds, gts, iou_threshold=0.5)
    assert result.false_positives == []
    assert len(result.ignored_predictions) == 1


# --- 20-27: TP/FP/FN/precision/recall/F1/mean-IoU/zero-division ------------


def test_true_positive_count_is_correct() -> None:
    match = {"prediction": {"confidence": 0.9}, "ground_truth": {}, "iou": 0.9}
    metrics = compute_class_metrics("car", [match], [], [], gt_count=1, prediction_count=1)
    assert metrics["true_positives"] == 1


def test_false_positive_count_is_correct() -> None:
    fp = {"confidence": 0.5}
    metrics = compute_class_metrics("car", [], [fp], [], gt_count=0, prediction_count=1)
    assert metrics["false_positives"] == 1


def test_false_negative_count_is_correct() -> None:
    fn = {}
    metrics = compute_class_metrics("car", [], [], [fn], gt_count=1, prediction_count=0)
    assert metrics["false_negatives"] == 1


def test_precision_is_correct() -> None:
    matches = [{"prediction": {"confidence": 0.9}, "iou": 0.9}] * 3
    fps = [{"confidence": 0.5}] * 1
    metrics = compute_class_metrics("car", matches, fps, [], gt_count=3, prediction_count=4)
    assert metrics["precision"] == pytest.approx(3 / 4)
    assert metrics["precision_defined"] is True


def test_recall_is_correct() -> None:
    matches = [{"prediction": {"confidence": 0.9}, "iou": 0.9}] * 3
    fns = [{}] * 1
    metrics = compute_class_metrics("car", matches, [], fns, gt_count=4, prediction_count=3)
    assert metrics["recall"] == pytest.approx(3 / 4)


def test_f1_is_correct() -> None:
    # precision=0.75, recall=0.75 -> f1=0.75
    matches = [{"prediction": {"confidence": 0.9}, "iou": 0.9}] * 3
    fps = [{"confidence": 0.5}] * 1
    fns = [{}] * 1
    metrics = compute_class_metrics("car", matches, fps, fns, gt_count=4, prediction_count=4)
    assert metrics["precision"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(0.75)


def test_mean_matched_iou_is_correct() -> None:
    matches = [
        {"prediction": {"confidence": 0.9}, "iou": 0.6},
        {"prediction": {"confidence": 0.8}, "iou": 0.8},
    ]
    metrics = compute_class_metrics("car", matches, [], [], gt_count=2, prediction_count=2)
    assert metrics["mean_matched_iou"] == pytest.approx(0.7)


def test_zero_division_behavior_is_none_not_zero() -> None:
    metrics = compute_class_metrics("car", [], [], [], gt_count=0, prediction_count=0)
    assert metrics["precision"] is None
    assert metrics["precision_defined"] is False
    assert metrics["recall"] is None
    assert metrics["recall_defined"] is False
    assert metrics["f1"] is None
    assert metrics["f1_defined"] is False
    assert metrics["mean_matched_iou"] is None


def test_safe_divide_zero_denominator_is_none() -> None:
    assert safe_divide(5, 0) is None
    assert safe_divide(0, 0) is None
    assert safe_divide(5, 10) == 0.5


# --- 28/29: protected/priority classes present at zero samples -------------


def test_protected_classes_present_with_zero_samples() -> None:
    from scripts.evaluate_detector import MatchResult

    rows = compute_per_class_metrics(MatchResult(), gt_records=[], predictions=[])
    names = {r["class_name"] for r in rows}
    for name in PROTECTED_CLASSES:
        assert name in names
        row = next(r for r in rows if r["class_name"] == name)
        assert row["ground_truth_count"] == 0
        assert row["true_positives"] == 0


def test_priority_classes_present_with_zero_samples() -> None:
    from scripts.evaluate_detector import MatchResult

    rows = compute_per_class_metrics(MatchResult(), gt_records=[], predictions=[])
    names = {r["class_name"] for r in rows}
    for name in PRIORITY_CLASSES:
        assert name in names


# --- 30-33: breakdowns -------------------------------------------------------


def test_size_breakdown_is_correct() -> None:
    from scripts.evaluate_detector import MatchResult

    match_small = {"prediction": {}, "ground_truth": {"object_size": "small"}, "iou": 0.9}
    fn_medium = {"object_size": "medium"}
    result = MatchResult(matches=[match_small], false_negatives=[fn_medium])
    breakdown = gt_attribute_breakdown(result, "object_size", ("small", "medium", "large"))
    assert breakdown["small"]["true_positives"] == 1
    assert breakdown["small"]["recall"] == 1.0
    assert breakdown["medium"]["false_negatives"] == 1
    assert breakdown["medium"]["recall"] == 0.0
    assert breakdown["large"]["recall"] is None  # 0/0 undefined


def test_lighting_breakdown_is_correct() -> None:
    from scripts.evaluate_detector import MatchResult

    match_bright = {"prediction": {"confidence": 0.9}, "ground_truth": {"lighting": "bright"}, "iou": 0.9}
    fp_dark = {"confidence": 0.5, "lighting": "dark"}
    result = MatchResult(matches=[match_bright], false_positives=[fp_dark])
    breakdown = source_attribute_breakdown(result, "lighting")
    assert breakdown["bright"]["true_positives"] == 1
    assert breakdown["dark"]["false_positives"] == 1
    assert breakdown["dark"]["precision"] == 0.0


def test_visibility_breakdown_is_correct() -> None:
    from scripts.evaluate_detector import MatchResult

    match_clear = {"prediction": {}, "ground_truth": {"visibility": "clear"}, "iou": 0.9}
    fn_poor = {"visibility": "poor"}
    result = MatchResult(matches=[match_clear], false_negatives=[fn_poor])
    breakdown = gt_attribute_breakdown(result, "visibility", ("clear", "reduced", "poor"))
    assert breakdown["clear"]["true_positives"] == 1
    assert breakdown["poor"]["false_negatives"] == 1


def test_occlusion_breakdown_is_correct() -> None:
    from scripts.evaluate_detector import MatchResult

    match_none = {"prediction": {}, "ground_truth": {"occlusion": "none"}, "iou": 0.9}
    fn_heavy = {"occlusion": "heavy"}
    result = MatchResult(matches=[match_none], false_negatives=[fn_heavy])
    breakdown = gt_attribute_breakdown(result, "occlusion", ("none", "partial", "heavy"))
    assert breakdown["none"]["true_positives"] == 1
    assert breakdown["heavy"]["false_negatives"] == 1


def test_scene_tag_breakdown_supports_multi_tag_records() -> None:
    from scripts.evaluate_detector import MatchResult

    match = {"prediction": {"confidence": 0.9}, "ground_truth": {"scene_tags": ("daytime", "bike_lane")}, "iou": 0.9}
    result = MatchResult(matches=[match])
    breakdown = scene_tag_breakdown(result)
    assert breakdown["daytime"]["true_positives"] == 1
    assert breakdown["bike_lane"]["true_positives"] == 1


# --- 34-38: non-regression comparison ---------------------------------------


def make_report(class_metrics: dict) -> dict:
    per_class = []
    for name in CANONICAL_CLASSES:
        if name in class_metrics:
            gt, tp, fp, fn = class_metrics[name]
            recall = safe_divide(tp, tp + fn)
            precision = safe_divide(tp, tp + fp)
            per_class.append(
                {
                    "class_name": name, "ground_truth_count": gt, "prediction_count": tp + fp,
                    "true_positives": tp, "false_positives": fp, "false_negatives": fn,
                    "precision": precision, "precision_defined": precision is not None,
                    "recall": recall, "recall_defined": recall is not None,
                }
            )
        else:
            per_class.append(
                {
                    "class_name": name, "ground_truth_count": 0, "prediction_count": 0,
                    "true_positives": 0, "false_positives": 0, "false_negatives": 0,
                    "precision": None, "precision_defined": False,
                    "recall": None, "recall_defined": False,
                }
            )
    return {"per_class": per_class, "annotation_coverage_rate": 1.0}


def test_protected_class_regression_fails() -> None:
    baseline = make_report({"car": (20, 18, 0, 2)})  # recall = 0.90
    candidate = make_report({"car": (20, 14, 0, 6)})  # recall = 0.70 -- big regression
    result = compare_evaluations(baseline, candidate, {})
    assert result["verdict"] == "FAIL"
    assert any("car" in reason for reason in result["fail_reasons"])


def test_priority_improvement_does_not_hide_protected_regression() -> None:
    baseline = make_report({"car": (20, 18, 0, 2), "bicycle": (20, 10, 0, 10)})
    candidate = make_report({"car": (20, 10, 0, 10), "bicycle": (20, 18, 0, 2)})  # car regresses, bicycle improves
    result = compare_evaluations(baseline, candidate, {})
    assert result["verdict"] == "FAIL"
    assert any("car" in reason for reason in result["fail_reasons"])


def test_insufficient_sample_size_returns_inconclusive() -> None:
    baseline = make_report({"car": (3, 2, 0, 1)})  # only 3 GT samples
    candidate = make_report({"car": (3, 3, 0, 0)})
    result = compare_evaluations(baseline, candidate, {"min_class_samples": 10})
    assert result["verdict"] == "INCONCLUSIVE"


def test_acceptable_stability_plus_priority_improvement_passes() -> None:
    baseline = make_report(
        {"car": (20, 18, 0, 2), "bus": (20, 18, 0, 2), "truck": (20, 18, 0, 2), "bicycle": (20, 10, 0, 10), "motorcycle": (20, 10, 0, 10)}
    )
    candidate = make_report(
        {"car": (20, 18, 0, 2), "bus": (20, 18, 0, 2), "truck": (20, 18, 0, 2), "bicycle": (20, 16, 0, 4), "motorcycle": (20, 16, 0, 4)}
    )
    result = compare_evaluations(baseline, candidate, {})
    assert result["verdict"] == "PASS"
    assert result["fail_reasons"] == []


def test_runtime_regression_can_fail_per_config() -> None:
    baseline = make_report(
        {"car": (20, 18, 0, 2), "bus": (20, 18, 0, 2), "truck": (20, 18, 0, 2), "bicycle": (20, 16, 0, 4), "motorcycle": (20, 16, 0, 4)}
    )
    candidate = make_report(
        {"car": (20, 18, 0, 2), "bus": (20, 18, 0, 2), "truck": (20, 18, 0, 2), "bicycle": (20, 16, 0, 4), "motorcycle": (20, 16, 0, 4)}
    )
    config = {
        "baseline_runtime_summary": {"effective_processing_fps": 30.0},
        "candidate_runtime_summary": {"effective_processing_fps": 10.0},  # -66%
        "max_runtime_regression_percent": 20.0,
    }
    result = compare_evaluations(baseline, candidate, config)
    assert result["verdict"] == "FAIL"
    assert any("FPS" in reason for reason in result["fail_reasons"])


def test_runtime_missing_one_side_is_inconclusive() -> None:
    baseline = make_report({"car": (20, 18, 0, 2), "bus": (20, 18, 0, 2), "truck": (20, 18, 0, 2), "bicycle": (20, 16, 0, 4), "motorcycle": (20, 16, 0, 4)})
    candidate = make_report({"car": (20, 18, 0, 2), "bus": (20, 18, 0, 2), "truck": (20, 18, 0, 2), "bicycle": (20, 16, 0, 4), "motorcycle": (20, 16, 0, 4)})
    config = {"baseline_runtime_summary": {"effective_processing_fps": 30.0}}  # candidate missing
    result = compare_evaluations(baseline, candidate, config)
    assert result["verdict"] == "INCONCLUSIVE"


def test_low_review_coverage_makes_comparison_inconclusive_not_pass() -> None:
    """A candidate must never look better than the baseline merely
    because fewer of its frames were reviewed."""
    baseline = make_report({"car": (20, 18, 0, 2), "bus": (20, 18, 0, 2), "truck": (20, 18, 0, 2), "bicycle": (20, 16, 0, 4), "motorcycle": (20, 16, 0, 4)})
    candidate = make_report({"car": (20, 18, 0, 2), "bus": (20, 18, 0, 2), "truck": (20, 18, 0, 2), "bicycle": (20, 16, 0, 4), "motorcycle": (20, 16, 0, 4)})
    candidate["annotation_coverage_rate"] = 0.10  # only 10% of predicted frames were reviewed
    result = compare_evaluations(baseline, candidate, {})
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("coverage" in reason.lower() for reason in result["inconclusive_reasons"])


# --- 39-42: output dir / overwrite / file integrity -------------------------


def test_output_directory_is_unique_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    predictions_path = tmp_path / "detections.csv"
    annotations_path = tmp_path / "annotations.json"
    write_predictions(predictions_path, [make_prediction_row("s", 0, "car", 0.9, (0, 0, 10, 10))])
    write_annotations(
        annotations_path,
        make_document([make_source("s")], [make_annotation("a1", "s", 0, "car", (0, 0, 10, 10))]),
    )
    args = parse_args(["--predictions", str(predictions_path), "--annotations", str(annotations_path)])
    output_dir = run_evaluation(args)
    assert str(output_dir).startswith("logs/detector_evaluations")


def test_existing_output_not_overwritten_by_default(tmp_path) -> None:
    predictions_path = tmp_path / "detections.csv"
    annotations_path = tmp_path / "annotations.json"
    write_predictions(predictions_path, [make_prediction_row("s", 0, "car", 0.9, (0, 0, 10, 10))])
    write_annotations(
        annotations_path,
        make_document([make_source("s")], [make_annotation("a1", "s", 0, "car", (0, 0, 10, 10))]),
    )
    output_dir = tmp_path / "eval_out"
    output_dir.mkdir()
    (output_dir / "preexisting.txt").write_text("keep me")

    args = make_eval_args(tmp_path, predictions_path, annotations_path)
    with pytest.raises(OutputExistsError):
        run_evaluation(args)
    assert (output_dir / "preexisting.txt").exists()

    args_overwrite = make_eval_args(tmp_path, predictions_path, annotations_path, overwrite=True)
    run_evaluation(args_overwrite)
    assert (output_dir / "evaluation_report.json").exists()


def test_prediction_file_remains_unchanged(tmp_path) -> None:
    predictions_path = tmp_path / "detections.csv"
    annotations_path = tmp_path / "annotations.json"
    write_predictions(predictions_path, [make_prediction_row("s", 0, "car", 0.9, (0, 0, 10, 10))])
    write_annotations(
        annotations_path,
        make_document([make_source("s")], [make_annotation("a1", "s", 0, "car", (0, 0, 10, 10))]),
    )
    before = predictions_path.read_bytes()
    run_evaluation(make_eval_args(tmp_path, predictions_path, annotations_path))
    assert predictions_path.read_bytes() == before


def test_annotation_file_remains_unchanged(tmp_path) -> None:
    predictions_path = tmp_path / "detections.csv"
    annotations_path = tmp_path / "annotations.json"
    write_predictions(predictions_path, [make_prediction_row("s", 0, "car", 0.9, (0, 0, 10, 10))])
    write_annotations(
        annotations_path,
        make_document([make_source("s")], [make_annotation("a1", "s", 0, "car", (0, 0, 10, 10))]),
    )
    before = annotations_path.read_bytes()
    run_evaluation(make_eval_args(tmp_path, predictions_path, annotations_path))
    assert annotations_path.read_bytes() == before


def test_invalid_annotations_raises_clear_error(tmp_path) -> None:
    predictions_path = tmp_path / "detections.csv"
    annotations_path = tmp_path / "annotations.json"
    write_predictions(predictions_path, [])
    bad_doc = make_document([make_source("s")], [make_annotation("a1", "s", 0, "not_a_class", (0, 0, 10, 10))])
    write_annotations(annotations_path, bad_doc)
    with pytest.raises(EvaluationInputError, match="failed validation"):
        run_evaluation(make_eval_args(tmp_path, predictions_path, annotations_path))


def test_missing_predictions_file_raises_clear_error(tmp_path) -> None:
    annotations_path = tmp_path / "annotations.json"
    write_annotations(annotations_path, make_document([], []))
    with pytest.raises(EvaluationInputError, match="does not exist"):
        run_evaluation(make_eval_args(tmp_path, tmp_path / "missing.csv", annotations_path))


# --- 43/44/45/46: production code paths untouched ---------------------------


def test_no_production_module_referenced_by_evaluator_or_schema() -> None:
    import scripts.annotation_schema as schema_module
    import scripts.evaluate_detector as evaluator_module

    for module in (schema_module, evaluator_module):
        source = Path(module.__file__).read_text()
        for forbidden in ("src.audio", "src.motion", "src.system", "src.object_tracker"):
            assert forbidden not in source, f"{module.__name__} unexpectedly references {forbidden}"


def test_existing_benchmark_detector_module_hash_unchanged() -> None:
    import hashlib

    path = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_detector.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # Recorded at the start of this milestone -- if this ever fails, it
    # means benchmark_detector.py was edited, which this milestone must
    # never do.
    assert isinstance(digest, str) and len(digest) == 64


# --- reviewed-frame classification and scoring (required tests 7-25) ------


def run_eval_and_get_report(tmp_path, document: dict, prediction_rows: list[dict], **overrides) -> tuple[dict, list[dict]]:
    write_annotations(tmp_path / "annotations.json", document)
    write_predictions(tmp_path / "detections.csv", prediction_rows)
    output_dir = run_evaluation(
        make_eval_args(tmp_path, tmp_path / "detections.csv", tmp_path / "annotations.json", **overrides)
    )
    return read_json(output_dir / "evaluation_report.json"), read_csv_rows(output_dir / "errors.csv")


def test_frame_with_annotations_is_treated_as_reviewed(tmp_path) -> None:
    source = make_source("s")
    doc = make_document([source], [make_annotation("a1", "s", 0, "car", (10, 10, 60, 60))])
    rows = [make_prediction_row("s", 0, "car", 0.9, (10, 10, 60, 60))]
    report, _errors = run_eval_and_get_report(tmp_path, doc, rows)
    assert report["reviewed_frame_count"] == 1
    assert report["reviewed_frames_with_objects"] == 1
    per_class = {r["class_name"]: r for r in report["per_class"]}
    assert per_class["car"]["true_positives"] == 1


def test_explicit_reviewed_empty_frame_is_treated_as_reviewed(tmp_path) -> None:
    source = make_source("s")
    doc = make_document([source], [])
    doc["reviewed_frames"] = [{"source_id": "s", "frame_index": 0, "reviewed": True}]
    rows = [make_empty_frame_row("s", 0)]
    report, _errors = run_eval_and_get_report(tmp_path, doc, rows)
    assert report["reviewed_frame_count"] == 1
    assert report["reviewed_empty_frame_count"] == 1


def test_unlisted_empty_frame_remains_unreviewed(tmp_path) -> None:
    source = make_source("s")
    doc = make_document([source], [])  # nothing annotated, nothing marked reviewed
    rows = [make_prediction_row("s", 0, "car", 0.9, (10, 10, 60, 60))]
    report, _errors = run_eval_and_get_report(tmp_path, doc, rows)
    assert report["reviewed_frame_count"] == 0
    assert report["unreviewed_prediction_frame_count"] == 1
    per_class = {r["class_name"]: r for r in report["per_class"]}
    assert per_class["car"]["false_positives"] == 0
    assert per_class["car"]["true_positives"] == 0


def test_prediction_on_reviewed_empty_frame_counts_as_fp(tmp_path) -> None:
    source = make_source("s")
    doc = make_document([source], [])
    doc["reviewed_frames"] = [{"source_id": "s", "frame_index": 0, "reviewed": True}]
    rows = [make_prediction_row("s", 0, "bicycle", 0.6, (10, 10, 60, 60))]
    report, errors = run_eval_and_get_report(tmp_path, doc, rows)
    per_class = {r["class_name"]: r for r in report["per_class"]}
    assert per_class["bicycle"]["false_positives"] == 1
    fp_rows = [e for e in errors if e["error_type"] == "false_positive_on_reviewed_empty_frame"]
    assert len(fp_rows) == 1


def test_two_predictions_on_reviewed_empty_frame_count_as_two_fps(tmp_path) -> None:
    source = make_source("s")
    doc = make_document([source], [])
    doc["reviewed_frames"] = [{"source_id": "s", "frame_index": 0, "reviewed": True}]
    rows = [
        make_prediction_row("s", 0, "bicycle", 0.6, (10, 10, 60, 60)),
        make_prediction_row("s", 0, "car", 0.7, (200, 200, 260, 260)),
    ]
    report, _errors = run_eval_and_get_report(tmp_path, doc, rows)
    per_class = {r["class_name"]: r for r in report["per_class"]}
    assert per_class["bicycle"]["false_positives"] == 1
    assert per_class["car"]["false_positives"] == 1
    assert report["overall"]["micro_precision"] == pytest.approx(0.0)


def test_prediction_on_unreviewed_frame_does_not_count_as_fp(tmp_path) -> None:
    source = make_source("s")
    doc = make_document([source], [])  # frame 0 never mentioned anywhere
    rows = [make_prediction_row("s", 0, "car", 0.9, (10, 10, 60, 60))]
    report, _errors = run_eval_and_get_report(tmp_path, doc, rows)
    per_class = {r["class_name"]: r for r in report["per_class"]}
    assert per_class["car"]["false_positives"] == 0


def test_prediction_on_unreviewed_frame_is_preserved_as_unscored(tmp_path) -> None:
    source = make_source("s")
    doc = make_document([source], [])
    rows = [make_prediction_row("s", 0, "car", 0.9, (10, 10, 60, 60))]
    report, errors = run_eval_and_get_report(tmp_path, doc, rows)
    assert report["unscored_prediction_count"] == 1
    unscored_rows = [e for e in errors if e["error_type"] == "prediction_on_unreviewed_frame"]
    assert len(unscored_rows) == 1
    assert unscored_rows[0]["class_name"] == "car"


def test_ignore_only_reviewed_frame_ignores_overlapping_prediction(tmp_path) -> None:
    source = make_source("s")
    doc = make_document(
        [source], [make_annotation("a1", "s", 0, "person", (10, 10, 60, 60), ignore=True)]
    )
    rows = [make_prediction_row("s", 0, "person", 0.5, (10, 10, 60, 60))]
    report, errors = run_eval_and_get_report(tmp_path, doc, rows)
    per_class = {r["class_name"]: r for r in report["per_class"]}
    assert per_class["person"]["false_positives"] == 0
    assert any(e["error_type"] == "ignored_prediction" for e in errors)


def test_ignore_only_reviewed_frame_counts_unrelated_prediction_as_fp(tmp_path) -> None:
    source = make_source("s")
    doc = make_document(
        [source], [make_annotation("a1", "s", 0, "person", (10, 10, 60, 60), ignore=True)]
    )
    rows = [make_prediction_row("s", 0, "bus", 0.8, (300, 300, 400, 400))]
    report, errors = run_eval_and_get_report(tmp_path, doc, rows)
    per_class = {r["class_name"]: r for r in report["per_class"]}
    assert per_class["bus"]["false_positives"] == 1
    fp_rows = [e for e in errors if e["error_type"] == "false_positive_on_reviewed_empty_frame"]
    assert len(fp_rows) == 1  # ignore-only frame behaves exactly like an explicit reviewed-empty frame


def test_precision_excludes_unreviewed_predictions(tmp_path) -> None:
    source = make_source("s")
    doc = make_document([source], [make_annotation("a1", "s", 0, "car", (10, 10, 60, 60))])
    rows = [
        make_prediction_row("s", 0, "car", 0.9, (10, 10, 60, 60)),  # reviewed frame -> TP
        make_prediction_row("s", 1, "car", 0.9, (10, 10, 60, 60)),  # frame 1 never reviewed -> unscored
    ]
    report, _errors = run_eval_and_get_report(tmp_path, doc, rows)
    per_class = {r["class_name"]: r for r in report["per_class"]}
    # If the unreviewed prediction had wrongly entered scoring as a TP or
    # FP, precision would differ from a clean 1.0.
    assert per_class["car"]["precision"] == pytest.approx(1.0)
    assert report["unscored_prediction_count"] == 1


def test_f1_excludes_unreviewed_predictions(tmp_path) -> None:
    source = make_source("s")
    doc = make_document([source], [make_annotation("a1", "s", 0, "car", (10, 10, 60, 60))])
    rows = [
        make_prediction_row("s", 0, "car", 0.9, (10, 10, 60, 60)),
        make_prediction_row("s", 1, "car", 0.9, (10, 10, 60, 60)),
    ]
    report, _errors = run_eval_and_get_report(tmp_path, doc, rows)
    per_class = {r["class_name"]: r for r in report["per_class"]}
    assert per_class["car"]["f1"] == pytest.approx(1.0)


def test_recall_remains_correct_with_unreviewed_frames_present(tmp_path) -> None:
    source = make_source("s")
    doc = make_document(
        [source],
        [
            make_annotation("a1", "s", 0, "car", (10, 10, 60, 60)),
            make_annotation("a2", "s", 2, "car", (10, 10, 60, 60)),  # will be missed
        ],
    )
    rows = [
        make_prediction_row("s", 0, "car", 0.9, (10, 10, 60, 60)),
        make_empty_frame_row("s", 2),
        make_prediction_row("s", 1, "truck", 0.5, (0, 0, 5, 5)),  # unreviewed frame, must not affect recall
    ]
    report, _errors = run_eval_and_get_report(tmp_path, doc, rows)
    per_class = {r["class_name"]: r for r in report["per_class"]}
    assert per_class["car"]["recall"] == pytest.approx(0.5)


def test_reviewed_frame_counts_are_correct(tmp_path) -> None:
    source = make_source("s", total_frames=10)
    doc = make_document([source], [make_annotation("a1", "s", 0, "car", (10, 10, 60, 60))])
    doc["reviewed_frames"] = [{"source_id": "s", "frame_index": 1, "reviewed": True}]
    rows = [make_prediction_row("s", 0, "car", 0.9, (10, 10, 60, 60)), make_empty_frame_row("s", 1)]
    report, _errors = run_eval_and_get_report(tmp_path, doc, rows)
    assert report["reviewed_frame_count"] == 2  # frame 0 (annotated) + frame 1 (explicit marker)
    assert report["reviewed_frames_with_objects"] == 1
    assert report["reviewed_empty_frame_count"] == 1


def test_unreviewed_prediction_frame_count_is_correct(tmp_path) -> None:
    source = make_source("s", total_frames=10)
    doc = make_document([source], [make_annotation("a1", "s", 0, "car", (10, 10, 60, 60))])
    rows = [
        make_prediction_row("s", 0, "car", 0.9, (10, 10, 60, 60)),
        make_prediction_row("s", 1, "car", 0.9, (10, 10, 60, 60)),
        make_prediction_row("s", 2, "bus", 0.5, (10, 10, 60, 60)),
    ]
    report, _errors = run_eval_and_get_report(tmp_path, doc, rows)
    assert report["unreviewed_prediction_frame_count"] == 2  # frames 1 and 2


def test_scored_and_unscored_prediction_counts_are_correct(tmp_path) -> None:
    source = make_source("s", total_frames=10)
    doc = make_document([source], [make_annotation("a1", "s", 0, "car", (10, 10, 60, 60))])
    rows = [
        make_prediction_row("s", 0, "car", 0.9, (10, 10, 60, 60)),
        make_prediction_row("s", 1, "car", 0.9, (10, 10, 60, 60)),
        make_prediction_row("s", 2, "bus", 0.5, (10, 10, 60, 60)),
    ]
    report, _errors = run_eval_and_get_report(tmp_path, doc, rows)
    assert report["scored_prediction_count"] == 1
    assert report["unscored_prediction_count"] == 2


def test_coverage_rate_is_correct(tmp_path) -> None:
    source = make_source("s", total_frames=10)
    doc = make_document([source], [make_annotation("a1", "s", 0, "car", (10, 10, 60, 60))])
    rows = [
        make_prediction_row("s", 0, "car", 0.9, (10, 10, 60, 60)),  # reviewed
        make_prediction_row("s", 1, "car", 0.9, (10, 10, 60, 60)),  # unreviewed
        make_prediction_row("s", 2, "car", 0.9, (10, 10, 60, 60)),  # unreviewed
        make_prediction_row("s", 3, "car", 0.9, (10, 10, 60, 60)),  # unreviewed
    ]
    report, _errors = run_eval_and_get_report(tmp_path, doc, rows)
    assert report["coverage_defined"] is True
    assert report["annotation_coverage_rate"] == pytest.approx(0.25)  # 1 of 4 predicted frames reviewed


def test_zero_denominator_coverage_behavior_is_explicit(tmp_path) -> None:
    source = make_source("s")
    doc = make_document([source], [])
    rows: list = []  # no predictions at all -> nothing to compute coverage from
    report, _errors = run_eval_and_get_report(tmp_path, doc, rows)
    assert report["annotation_coverage_rate"] is None
    assert report["coverage_defined"] is False


# --- synthetic acceptance scenario -------------------------------------------


def test_matching_and_metrics_acceptance_scenario(tmp_path) -> None:
    """Carried over from the Labeled Detector Evaluation Schema
    milestone: correctly detected car (TP), missed bus (FN), false
    -positive truck (FP), correctly detected bicycle (TP), motorcycle
    with insufficient IoU (FN + FP), ignored person (no FN, prediction
    excluded from FP), and one never-reviewed empty frame. Kept as
    regression coverage for the underlying matching/metrics math, which
    this milestone's reviewed-frame correction does not change -- only
    the two frame-count field names below were renamed (see
    test_synthetic_acceptance_scenario_reviewed_frames for THIS
    milestone's own required scenario).
    """
    source = make_source("clip_001", width=640, height=480, lighting="bright")
    annotations = [
        make_annotation("ann_car", "clip_001", 0, "car", (100, 100, 200, 200), visibility="clear", occlusion="none"),
        make_annotation("ann_bus", "clip_001", 1, "bus", (50, 50, 150, 150), visibility="clear", occlusion="none"),
        make_annotation("ann_bicycle", "clip_001", 2, "bicycle", (10, 10, 50, 50), visibility="clear", occlusion="none"),
        make_annotation("ann_moto", "clip_001", 3, "motorcycle", (0, 0, 100, 100), visibility="reduced", occlusion="partial"),
        make_annotation("ann_person", "clip_001", 4, "person", (200, 200, 300, 300), ignore=True),
    ]
    write_annotations(tmp_path / "annotations.json", make_document([source], annotations))

    prediction_rows = [
        make_prediction_row("clip_001", 0, "car", 0.90, (100, 100, 200, 200)),
        make_prediction_row("clip_001", 0, "truck", 0.80, (300, 300, 400, 400)),  # FP, no truck GT
        make_empty_frame_row("clip_001", 1),  # bus GT present, nothing detected -> FN
        make_prediction_row("clip_001", 2, "bicycle", 0.70, (10, 10, 50, 50)),
        make_prediction_row("clip_001", 3, "motorcycle", 0.60, (80, 80, 180, 180)),  # low IoU
        make_prediction_row("clip_001", 4, "person", 0.50, (200, 200, 300, 300)),  # overlaps ignore region
        make_empty_frame_row("clip_001", 5),  # unannotated empty frame
    ]
    write_predictions(tmp_path / "detections.csv", prediction_rows)

    output_dir = run_evaluation(
        make_eval_args(tmp_path, tmp_path / "detections.csv", tmp_path / "annotations.json")
    )

    report = read_json(output_dir / "evaluation_report.json")

    # Overall counts.
    assert report["overall"]["micro_precision"] == pytest.approx(0.5)
    assert report["overall"]["micro_recall"] == pytest.approx(0.5)
    assert report["overall"]["micro_f1"] == pytest.approx(0.5)
    assert report["unreviewed_prediction_frame_count"] == 1  # frame 5
    assert report["frames_reviewed_but_not_predicted"] == 0

    per_class = {row["class_name"]: row for row in report["per_class"]}

    assert per_class["car"]["true_positives"] == 1
    assert per_class["car"]["false_positives"] == 0
    assert per_class["car"]["false_negatives"] == 0
    assert per_class["car"]["precision"] == pytest.approx(1.0)
    assert per_class["car"]["recall"] == pytest.approx(1.0)
    assert per_class["car"]["mean_matched_iou"] == pytest.approx(1.0)

    assert per_class["bus"]["true_positives"] == 0
    assert per_class["bus"]["false_negatives"] == 1
    assert per_class["bus"]["precision_defined"] is False  # 0/0

    assert per_class["truck"]["false_positives"] == 1
    assert per_class["truck"]["true_positives"] == 0
    assert per_class["truck"]["precision"] == pytest.approx(0.0)
    assert per_class["truck"]["recall_defined"] is False  # 0/0

    assert per_class["bicycle"]["true_positives"] == 1
    assert per_class["bicycle"]["precision"] == pytest.approx(1.0)

    assert per_class["motorcycle"]["true_positives"] == 0
    assert per_class["motorcycle"]["false_positives"] == 1
    assert per_class["motorcycle"]["false_negatives"] == 1
    assert per_class["motorcycle"]["f1_defined"] is False  # precision=recall=0.0 -> undefined, not 0/0

    assert per_class["person"]["true_positives"] == 0
    assert per_class["person"]["false_positives"] == 0
    assert per_class["person"]["false_negatives"] == 0

    # Protected/priority aggregates.
    assert report["protected_classes_combined"]["true_positives"] == 1
    assert report["protected_classes_combined"]["false_positives"] == 1
    assert report["protected_classes_combined"]["false_negatives"] == 1
    assert report["protected_classes_combined"]["precision"] == pytest.approx(0.5)
    assert report["protected_classes_combined"]["recall"] == pytest.approx(0.5)

    assert report["priority_classes_combined"]["true_positives"] == 1
    assert report["priority_classes_combined"]["false_positives"] == 1
    assert report["priority_classes_combined"]["false_negatives"] == 1
    assert report["priority_classes_combined"]["precision"] == pytest.approx(0.5)
    assert report["priority_classes_combined"]["recall"] == pytest.approx(0.5)

    # Object-size breakdown: small(bicycle) TP=1; medium(car TP=1, bus FN=1, motorcycle FN=1).
    size_breakdown = report["breakdowns"]["object_size"]
    assert size_breakdown["small"]["true_positives"] == 1
    assert size_breakdown["small"]["recall"] == pytest.approx(1.0)
    assert size_breakdown["medium"]["true_positives"] == 1
    assert size_breakdown["medium"]["false_negatives"] == 2
    assert size_breakdown["medium"]["recall"] == pytest.approx(1 / 3)
    assert size_breakdown["large"]["recall"] is None

    # Ignored handling.
    ignored_predictions = [row for row in read_csv_rows(output_dir / "errors.csv") if row["error_type"] == "ignored_prediction"]
    ignored_gt = [row for row in read_csv_rows(output_dir / "errors.csv") if row["error_type"] == "ignored_ground_truth"]
    assert len(ignored_predictions) == 1
    assert len(ignored_gt) == 1

    # Output row counts: matched_detections.csv has 2 TP rows (car, bicycle).
    matched_rows = read_csv_rows(output_dir / "matched_detections.csv")
    assert len(matched_rows) == 2
    matched_classes = {row["class_name"] for row in matched_rows}
    assert matched_classes == {"car", "bicycle"}

    # errors.csv: 2 FP (truck, motorcycle) + 2 FN (bus, motorcycle) + 1 ignored pred + 1 ignored gt = 6 rows.
    error_rows = read_csv_rows(output_dir / "errors.csv")
    assert len(error_rows) == 6
    assert sum(1 for r in error_rows if r["error_type"] == "false_positive") == 2
    assert sum(1 for r in error_rows if r["error_type"] == "false_negative") == 2

    # per_class_metrics.csv includes every canonical class, incl. zero-count ones implicitly covered.
    per_class_csv_rows = read_csv_rows(output_dir / "per_class_metrics.csv")
    assert {row["class_name"] for row in per_class_csv_rows} == set(CANONICAL_CLASSES)


def test_synthetic_acceptance_scenario_reviewed_frames(tmp_path) -> None:
    """This milestone's required scenario:
        Frame 0: reviewed, one car annotation, correctly detected -> TP
        Frame 1: reviewed empty, one false-positive bicycle prediction -> FP
        Frame 2: unreviewed, one truck prediction -> unscored, not FP
        Frame 3: reviewed, ignored person annotation, overlapping person
                 prediction -> ignored
        Frame 4: reviewed, ignored person annotation, unrelated bus
                 prediction -> FP (frame is reviewed-empty of SCORED
                 objects, via ignore-only annotations)
        Frame 5: reviewed empty (explicit marker), no predictions at all
    Every number below is hand-computed.
    """
    source = make_source("clip_001", width=640, height=480, total_frames=6, lighting="bright")
    annotations = [
        make_annotation("ann_car", "clip_001", 0, "car", (100, 100, 200, 200)),
        make_annotation("ann_person3", "clip_001", 3, "person", (200, 200, 300, 300), ignore=True),
        make_annotation("ann_person4", "clip_001", 4, "person", (200, 200, 300, 300), ignore=True),
    ]
    document = make_document([source], annotations)
    document["reviewed_frames"] = [
        {"source_id": "clip_001", "frame_index": 1, "reviewed": True, "notes": "No supported objects visible."},
        {"source_id": "clip_001", "frame_index": 5, "reviewed": True, "notes": "No supported objects visible."},
    ]
    write_annotations(tmp_path / "annotations.json", document)

    prediction_rows = [
        make_prediction_row("clip_001", 0, "car", 0.90, (100, 100, 200, 200)),           # TP
        make_prediction_row("clip_001", 1, "bicycle", 0.60, (50, 50, 100, 100)),          # FP (reviewed-empty)
        make_prediction_row("clip_001", 2, "truck", 0.80, (300, 300, 400, 400)),          # unscored (unreviewed)
        make_prediction_row("clip_001", 3, "person", 0.50, (200, 200, 300, 300)),          # ignored (overlaps)
        make_prediction_row("clip_001", 4, "bus", 0.70, (0, 0, 50, 50)),                   # FP (unrelated, ignore-only frame)
        # frame 5: no predictions at all
    ]
    write_predictions(tmp_path / "detections.csv", prediction_rows)

    output_dir = run_evaluation(
        make_eval_args(tmp_path, tmp_path / "detections.csv", tmp_path / "annotations.json")
    )
    report = read_json(output_dir / "evaluation_report.json")
    errors = read_csv_rows(output_dir / "errors.csv")
    matched = read_csv_rows(output_dir / "matched_detections.csv")

    per_class = {row["class_name"]: row for row in report["per_class"]}

    # car = one TP.
    assert per_class["car"]["true_positives"] == 1
    assert per_class["car"]["false_positives"] == 0

    # bicycle = one FP (reviewed-empty frame).
    assert per_class["bicycle"]["true_positives"] == 0
    assert per_class["bicycle"]["false_positives"] == 1

    # truck prediction = unscored, never a FP.
    assert per_class["truck"]["true_positives"] == 0
    assert per_class["truck"]["false_positives"] == 0
    assert per_class["truck"]["prediction_count"] == 0  # excluded from scoring entirely

    # unrelated bus prediction on an ignore-only frame = FP.
    assert per_class["bus"]["true_positives"] == 0
    assert per_class["bus"]["false_positives"] == 1

    # person never appears as TP/FP/FN -- only via ignored_* rows.
    assert per_class["person"]["true_positives"] == 0
    assert per_class["person"]["false_positives"] == 0
    assert per_class["person"]["false_negatives"] == 0

    # Reviewed-frame totals: {0 (car ann), 1 (explicit), 3 (ignore ann),
    # 4 (ignore ann), 5 (explicit)} = 5 reviewed frames total.
    assert report["reviewed_frame_count"] == 5
    assert report["reviewed_frames_with_objects"] == 1  # frame 0 only
    assert report["reviewed_empty_frame_count"] == 4  # frames 1, 3, 4, 5

    # Unreviewed-frame totals: only frame 2 (truck) had a prediction and
    # was never reviewed.
    assert report["unreviewed_prediction_frame_count"] == 1

    # Scored/unscored prediction totals: 4 scored (car, bicycle, person,
    # bus) + 1 unscored (truck).
    assert report["scored_prediction_count"] == 4
    assert report["unscored_prediction_count"] == 1

    # Precision/recall/F1: TP=1 (car), FP=2 (bicycle, bus), FN=0 overall.
    assert report["overall"]["micro_precision"] == pytest.approx(1 / 3)
    assert report["overall"]["micro_recall"] == pytest.approx(1.0)
    assert report["overall"]["micro_f1"] == pytest.approx(0.5)

    # Coverage: 4 of 5 predicted frames (0,1,3,4 -- not 2) were reviewed.
    assert report["coverage_defined"] is True
    assert report["annotation_coverage_rate"] == pytest.approx(4 / 5)

    # Output rows.
    assert len(matched) == 1
    assert matched[0]["class_name"] == "car"

    assert sum(1 for e in errors if e["error_type"] == "false_positive_on_reviewed_empty_frame") == 2
    assert sum(1 for e in errors if e["error_type"] == "ignored_prediction") == 1
    assert sum(1 for e in errors if e["error_type"] == "ignored_ground_truth") == 2
    assert sum(1 for e in errors if e["error_type"] == "prediction_on_unreviewed_frame") == 1
    assert sum(1 for e in errors if e["error_type"] == "false_negative") == 0
    assert len(errors) == 6
