"""Labeled detector evaluation for Atlas -- matches scripts/
benchmark_detector.py's detections.csv predictions against a hand
-authored annotations.json (scripts/annotation_schema.py) ground truth,
producing real precision/recall/F1 metrics plus an optional baseline
-vs-candidate non-regression comparison. See docs/DETECTOR_ANNOTATIONS.md
and docs/DETECTOR_BENCHMARK.md.

Two CLI modes on one script:
    python -m scripts.evaluate_detector --predictions detections.csv \\
        --annotations annotations.json --output-dir <path>

    python -m scripts.evaluate_detector \\
        --compare-baseline baseline/evaluation_report.json \\
        --compare-candidate candidate/evaluation_report.json \\
        --output-dir <path>

Reuses (never duplicates) scripts/benchmark_detector.py's iou(),
PROTECTED_CLASSES, PRIORITY_CLASSES, mean_or_none/median_or_none/
percentile_or_none, sha256_file, and get_git_commit -- and scripts/
annotation_schema.py's validate_annotations()/classify_object_size().
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.annotation_schema import (  # noqa: E402
    CANONICAL_CLASSES,
    OBJECT_SIZE_CATEGORIES,
    OCCLUSION_CATEGORIES,
    VISIBILITY_CATEGORIES,
    classify_object_size,
    validate_annotations,
)
from scripts.benchmark_detector import (  # noqa: E402
    PRIORITY_CLASSES,
    PROTECTED_CLASSES,
    get_git_commit,
    iou,
    mean_or_none,
    sha256_file,
)

SCHEMA_VERSION = "1.0"
DEFAULT_IOU_THRESHOLD = 0.50
DEFAULT_MIN_CLASS_SAMPLES = 10
DEFAULT_MAX_PROTECTED_RECALL_REGRESSION = 0.02
DEFAULT_MIN_PRIORITY_RECALL_IMPROVEMENT = 0.02
DEFAULT_PRIORITY_REGRESSION_TOLERANCE = 0.0
DEFAULT_MAX_RUNTIME_REGRESSION_PERCENT = 20.0
DEFAULT_MIN_REVIEW_COVERAGE = 0.80

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"

PER_CLASS_METRICS_CSV_COLUMNS = [
    "class_name", "ground_truth_count", "prediction_count",
    "true_positives", "false_positives", "false_negatives",
    "precision", "precision_defined", "recall", "recall_defined",
    "f1", "f1_defined", "mean_matched_iou", "mean_matched_iou_defined",
    "mean_confidence_true_positives", "mean_confidence_true_positives_defined",
    "mean_confidence_false_positives", "mean_confidence_false_positives_defined",
]

MATCHED_DETECTIONS_CSV_COLUMNS = [
    "annotation_id", "source_id", "frame_index", "class_name",
    "iou", "confidence", "difficult",
]

ERRORS_CSV_COLUMNS = [
    "error_type", "source_id", "frame_index", "class_name",
    "annotation_id", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "difficult",
]


class EvaluationInputError(Exception):
    """Raised for a malformed/invalid --predictions or --annotations input."""


class OutputExistsError(Exception):
    """Raised when the resolved output directory already has content and
    --overwrite was not passed."""


def safe_divide(numerator: float, denominator: float) -> float | None:
    """None (undefined), never 0.0, when the denominator is zero -- see
    docs/DETECTOR_ANNOTATIONS.md's "zero-division behavior" section."""
    return None if denominator == 0 else numerator / denominator


# --- loading -----------------------------------------------------------------


def load_predictions(path: Path) -> tuple[list[dict], set[tuple[str, int]]]:
    """Returns (prediction_records, all_processed_frame_keys). A
    zero-detection row (blank atlas_config_class_name) contributes only
    to the frame-key set, never to prediction_records -- there is no
    prediction to match there, but the frame WAS processed."""
    predictions: list[dict] = []
    frame_keys: set[tuple[str, int]] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_order, row in enumerate(reader):
            source_id = row["source_id"]
            frame_index = int(row["frame_index"])
            frame_keys.add((source_id, frame_index))

            class_name = row.get("atlas_config_class_name") or ""
            if not class_name:
                continue
            predictions.append(
                {
                    "source_id": source_id,
                    "frame_index": frame_index,
                    "class_name": class_name,
                    "confidence": float(row["confidence"]),
                    "bbox": (
                        float(row["bbox_x1"]), float(row["bbox_y1"]),
                        float(row["bbox_x2"]), float(row["bbox_y2"]),
                    ),
                    "row_order": row_order,
                    "lighting": None,
                    "scene_tags": (),
                }
            )
    return predictions, frame_keys


def load_ground_truth(document: dict) -> tuple[list[dict], set[tuple[str, int]], dict]:
    """Returns (gt_records, annotated_frame_keys, source_by_id).
    object_size is always recomputed here from bounding_box_xyxy + the
    source's width/height (see classify_object_size) -- never trusted
    from the file, guaranteeing reproducibility."""
    source_by_id = {source["source_id"]: source for source in document["sources"]}
    gt_records: list[dict] = []
    frame_keys: set[tuple[str, int]] = set()

    for ann in document["annotations"]:
        source = source_by_id[ann["source_id"]]
        frame_key = (ann["source_id"], ann["frame_index"])
        frame_keys.add(frame_key)

        x1, y1, x2, y2 = ann["bounding_box_xyxy"]
        object_size = "medium"
        if "width" in source and "height" in source:
            object_size = classify_object_size((x2 - x1) * (y2 - y1), source["width"] * source["height"])

        gt_records.append(
            {
                "annotation_id": ann["annotation_id"],
                "source_id": ann["source_id"],
                "frame_index": ann["frame_index"],
                "class_name": ann["class_name"],
                "bbox": (x1, y1, x2, y2),
                "ignore": ann["ignore"],
                "difficult": ann["difficult"],
                "visibility": ann["visibility"],
                "occlusion": ann["occlusion"],
                "object_size": object_size,
                "lighting": source.get("lighting"),
                "scene_tags": tuple(source.get("scene_tags", [])),
            }
        )
    return gt_records, frame_keys, source_by_id


def load_reviewed_frames(document: dict) -> set[tuple[str, int]]:
    """Returns the set of (source_id, frame_index) explicitly marked
    reviewed via the optional top-level "reviewed_frames" array (absent
    -> empty set, matching an old schema-"1.0" file with no such key).
    Only reviewed=true entries can exist here at all -- see
    scripts/annotation_schema.py's validate_annotations()."""
    return {(entry["source_id"], entry["frame_index"]) for entry in document.get("reviewed_frames", [])}


def enrich_predictions_with_source_attributes(predictions: list[dict], source_by_id: dict) -> None:
    for pred in predictions:
        source = source_by_id.get(pred["source_id"])
        if source is not None:
            pred["lighting"] = source.get("lighting")
            pred["scene_tags"] = tuple(source.get("scene_tags", []))


# --- matching ------------------------------------------------------------------


@dataclass
class MatchResult:
    matches: list[dict] = field(default_factory=list)  # {"prediction":, "ground_truth":, "iou":}
    false_positives: list[dict] = field(default_factory=list)
    ignored_predictions: list[dict] = field(default_factory=list)
    false_negatives: list[dict] = field(default_factory=list)
    ignored_ground_truth: list[dict] = field(default_factory=list)


def match_all(predictions: list[dict], gt_records: list[dict], iou_threshold: float) -> MatchResult:
    """Class-aware, one-to-one, deterministic greedy-by-confidence
    matching per (source_id, frame_index). See docs/DETECTOR_ANNOTATIONS.
    md "Matching method" for the full algorithm description and its
    documented limitations relative to Hungarian assignment."""
    preds_by_frame: dict = defaultdict(list)
    for pred in predictions:
        preds_by_frame[(pred["source_id"], pred["frame_index"])].append(pred)
    gt_by_frame: dict = defaultdict(list)
    for gt in gt_records:
        gt_by_frame[(gt["source_id"], gt["frame_index"])].append(gt)

    result = MatchResult()

    for frame_key in sorted(set(preds_by_frame) | set(gt_by_frame)):
        frame_preds = preds_by_frame.get(frame_key, [])
        frame_gt = gt_by_frame.get(frame_key, [])
        scored_gt = [gt for gt in frame_gt if not gt["ignore"]]
        ignore_regions = [gt for gt in frame_gt if gt["ignore"]]
        result.ignored_ground_truth.extend(ignore_regions)

        classes = {gt["class_name"] for gt in scored_gt} | {p["class_name"] for p in frame_preds}
        matched_pred_ids: set[int] = set()

        for class_name in sorted(classes):
            class_preds = sorted(
                (p for p in frame_preds if p["class_name"] == class_name),
                key=lambda p: (-p["confidence"], p["row_order"]),
            )
            class_gt = [gt for gt in scored_gt if gt["class_name"] == class_name]
            matched_gt_ids: set[int] = set()

            for pred in class_preds:
                best_iou, best_gt = 0.0, None
                for gt in class_gt:
                    if id(gt) in matched_gt_ids:
                        continue
                    score = iou(pred["bbox"], gt["bbox"])
                    if score > best_iou:
                        best_iou, best_gt = score, gt
                if best_gt is not None and best_iou >= iou_threshold:
                    matched_gt_ids.add(id(best_gt))
                    matched_pred_ids.add(id(pred))
                    result.matches.append({"prediction": pred, "ground_truth": best_gt, "iou": best_iou})

            for gt in class_gt:
                if id(gt) not in matched_gt_ids:
                    result.false_negatives.append(gt)

        for pred in frame_preds:
            if id(pred) in matched_pred_ids:
                continue
            overlaps_ignore = any(
                iou(pred["bbox"], region["bbox"]) >= iou_threshold for region in ignore_regions
            )
            if overlaps_ignore:
                result.ignored_predictions.append(pred)
            else:
                result.false_positives.append(pred)

    return result


# --- metrics --------------------------------------------------------------------


def compute_class_metrics(
    class_name: str, tp_matches: list[dict], fp_list: list[dict], fn_list: list[dict],
    gt_count: int, prediction_count: int,
) -> dict:
    tp, fp, fn = len(tp_matches), len(fp_list), len(fn_list)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    mean_iou = mean_or_none([m["iou"] for m in tp_matches])
    mean_conf_tp = mean_or_none([m["prediction"]["confidence"] for m in tp_matches])
    mean_conf_fp = mean_or_none([p["confidence"] for p in fp_list])

    return {
        "class_name": class_name,
        "ground_truth_count": gt_count,
        "prediction_count": prediction_count,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision, "precision_defined": precision is not None,
        "recall": recall, "recall_defined": recall is not None,
        "f1": f1, "f1_defined": f1 is not None,
        "mean_matched_iou": mean_iou, "mean_matched_iou_defined": mean_iou is not None,
        "mean_confidence_true_positives": mean_conf_tp,
        "mean_confidence_true_positives_defined": mean_conf_tp is not None,
        "mean_confidence_false_positives": mean_conf_fp,
        "mean_confidence_false_positives_defined": mean_conf_fp is not None,
    }


def compute_per_class_metrics(match_result: MatchResult, gt_records: list[dict], predictions: list[dict]) -> list[dict]:
    gt_count_by_class: dict = defaultdict(int)
    for gt in gt_records:
        if not gt["ignore"]:
            gt_count_by_class[gt["class_name"]] += 1
    pred_count_by_class: dict = defaultdict(int)
    for pred in predictions:
        pred_count_by_class[pred["class_name"]] += 1

    rows = []
    for class_name in CANONICAL_CLASSES:
        tp_matches = [m for m in match_result.matches if m["ground_truth"]["class_name"] == class_name]
        fp_list = [p for p in match_result.false_positives if p["class_name"] == class_name]
        fn_list = [g for g in match_result.false_negatives if g["class_name"] == class_name]
        rows.append(
            compute_class_metrics(
                class_name, tp_matches, fp_list, fn_list,
                gt_count_by_class.get(class_name, 0), pred_count_by_class.get(class_name, 0),
            )
        )
    return rows


def combine_classes(label: str, class_names: tuple, match_result: MatchResult, gt_records: list[dict], predictions: list[dict]) -> dict:
    class_set = set(class_names)
    tp_matches = [m for m in match_result.matches if m["ground_truth"]["class_name"] in class_set]
    fp_list = [p for p in match_result.false_positives if p["class_name"] in class_set]
    fn_list = [g for g in match_result.false_negatives if g["class_name"] in class_set]
    gt_count = sum(1 for g in gt_records if not g["ignore"] and g["class_name"] in class_set)
    pred_count = sum(1 for p in predictions if p["class_name"] in class_set)
    return compute_class_metrics(label, tp_matches, fp_list, fn_list, gt_count, pred_count)


def compute_overall_metrics(per_class_rows: list[dict]) -> dict:
    total_tp = sum(r["true_positives"] for r in per_class_rows)
    total_fp = sum(r["false_positives"] for r in per_class_rows)
    total_fn = sum(r["false_negatives"] for r in per_class_rows)
    micro_precision = safe_divide(total_tp, total_tp + total_fp)
    micro_recall = safe_divide(total_tp, total_tp + total_fn)
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision is not None and micro_recall is not None and (micro_precision + micro_recall) > 0
        else None
    )

    defined_precisions = [r["precision"] for r in per_class_rows if r["precision_defined"]]
    defined_recalls = [r["recall"] for r in per_class_rows if r["recall_defined"]]
    defined_f1s = [r["f1"] for r in per_class_rows if r["f1_defined"]]

    return {
        "micro_precision": micro_precision, "micro_precision_defined": micro_precision is not None,
        "micro_recall": micro_recall, "micro_recall_defined": micro_recall is not None,
        "micro_f1": micro_f1, "micro_f1_defined": micro_f1 is not None,
        "macro_precision": mean_or_none(defined_precisions),
        "macro_precision_classes_used": len(defined_precisions),
        "macro_precision_classes_skipped": len(per_class_rows) - len(defined_precisions),
        "macro_recall": mean_or_none(defined_recalls),
        "macro_recall_classes_used": len(defined_recalls),
        "macro_recall_classes_skipped": len(per_class_rows) - len(defined_recalls),
        "macro_f1": mean_or_none(defined_f1s),
        "macro_f1_classes_used": len(defined_f1s),
        "macro_f1_classes_skipped": len(per_class_rows) - len(defined_f1s),
    }


def gt_attribute_breakdown(match_result: MatchResult, attribute: str, categories: tuple) -> dict:
    """TP/FN/recall/mean-matched-IoU only -- a false positive has no
    matched ground truth to bucket by a GT attribute (object_size/
    visibility/occlusion), so precision/F1 are deliberately not
    computed here. See docs/DETECTOR_ANNOTATIONS.md."""
    tp_by_value: dict = defaultdict(list)
    fn_by_value: dict = defaultdict(int)
    for m in match_result.matches:
        tp_by_value[m["ground_truth"][attribute]].append(m["iou"])
    for gt in match_result.false_negatives:
        fn_by_value[gt[attribute]] += 1

    breakdown = {}
    for value in categories:
        tp = len(tp_by_value.get(value, []))
        fn = fn_by_value.get(value, 0)
        recall = safe_divide(tp, tp + fn)
        breakdown[value] = {
            "true_positives": tp, "false_negatives": fn,
            "recall": recall, "recall_defined": recall is not None,
            "mean_matched_iou": mean_or_none(tp_by_value.get(value, [])),
        }
    return breakdown


def source_attribute_breakdown(match_result: MatchResult, attribute: str) -> dict:
    """Full precision/recall/F1 -- a source-level attribute (lighting)
    applies to false positives too, since every prediction belongs to
    a known source even when unmatched."""
    values = set()
    for m in match_result.matches:
        values.add(m["ground_truth"].get(attribute))
    for p in match_result.false_positives:
        values.add(p.get(attribute))
    for gt in match_result.false_negatives:
        values.add(gt.get(attribute))
    values.discard(None)

    breakdown = {}
    for value in sorted(values, key=str):
        tp_matches = [m for m in match_result.matches if m["ground_truth"].get(attribute) == value]
        fp_list = [p for p in match_result.false_positives if p.get(attribute) == value]
        fn_list = [g for g in match_result.false_negatives if g.get(attribute) == value]
        breakdown[str(value)] = compute_class_metrics(
            str(value), tp_matches, fp_list, fn_list,
            gt_count=len(tp_matches) + len(fn_list), prediction_count=len(tp_matches) + len(fp_list),
        )
    return breakdown


def scene_tag_breakdown(match_result: MatchResult) -> dict:
    """Same shape as source_attribute_breakdown, but scene_tags is a
    set per record rather than one value -- a record contributes to
    every tag it carries."""
    tags = set()
    for m in match_result.matches:
        tags.update(m["ground_truth"].get("scene_tags", ()))
    for p in match_result.false_positives:
        tags.update(p.get("scene_tags", ()))
    for gt in match_result.false_negatives:
        tags.update(gt.get("scene_tags", ()))

    breakdown = {}
    for tag in sorted(tags):
        tp_matches = [m for m in match_result.matches if tag in m["ground_truth"].get("scene_tags", ())]
        fp_list = [p for p in match_result.false_positives if tag in p.get("scene_tags", ())]
        fn_list = [g for g in match_result.false_negatives if tag in g.get("scene_tags", ())]
        breakdown[tag] = compute_class_metrics(
            tag, tp_matches, fp_list, fn_list,
            gt_count=len(tp_matches) + len(fn_list), prediction_count=len(tp_matches) + len(fp_list),
        )
    return breakdown


# --- output-dir resolution (mirrors benchmark_detector.py's convention,
#     but under logs/detector_evaluations/ -- a distinct concept) -------------


def resolve_output_dir(default_stem: str, output_dir_arg: str | None, overwrite: bool) -> Path:
    if output_dir_arg is not None:
        output_dir = Path(output_dir_arg)
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = Path("logs") / "detector_evaluations" / f"{default_stem}_{stamp}"

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise OutputExistsError(
            f"Output directory already has content: {output_dir}. "
            "Pass --overwrite to reuse it, or choose a different --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# --- orchestration: evaluate ------------------------------------------------


def run_evaluation(args: argparse.Namespace) -> Path:
    predictions_path = Path(args.predictions)
    annotations_path = Path(args.annotations)
    if not predictions_path.exists():
        raise EvaluationInputError(f"--predictions file does not exist: {predictions_path}")
    if not annotations_path.exists():
        raise EvaluationInputError(f"--annotations file does not exist: {annotations_path}")

    document = json.loads(annotations_path.read_text())
    validation = validate_annotations(document, base_dir=annotations_path.parent)
    if not validation.is_valid:
        raise EvaluationInputError(
            "annotations.json failed validation:\n" + "\n".join(f"  - {e}" for e in validation.errors)
        )

    predictions, predicted_frame_keys = load_predictions(predictions_path)
    gt_records, annotated_frame_keys, source_by_id = load_ground_truth(document)
    explicit_reviewed_frame_keys = load_reviewed_frames(document)
    enrich_predictions_with_source_attributes(predictions, source_by_id)

    # Three frame categories (see docs/DETECTOR_ANNOTATIONS.md "Reviewed
    # -frame status"): a frame with >=1 SCORED (non-ignore) annotation is
    # REVIEWED_WITH_OBJECTS; a frame that's reviewed (has any annotation
    # entry at all -- including ignore-only -- OR an explicit
    # reviewed_frames marker) but has zero scored objects is
    # REVIEWED_EMPTY; anything else is UNREVIEWED.
    scored_annotated_frame_keys = {
        (g["source_id"], g["frame_index"]) for g in gt_records if not g["ignore"]
    }
    reviewed_frame_keys = annotated_frame_keys | explicit_reviewed_frame_keys
    reviewed_empty_frame_keys = reviewed_frame_keys - scored_annotated_frame_keys

    evaluated_frame_keys = predicted_frame_keys & reviewed_frame_keys
    frames_reviewed_but_not_predicted = reviewed_frame_keys - predicted_frame_keys
    unreviewed_prediction_frame_keys = predicted_frame_keys - reviewed_frame_keys

    scored_predictions = [p for p in predictions if (p["source_id"], p["frame_index"]) in evaluated_frame_keys]
    # Preserved, never silently dropped -- see errors.csv's
    # "prediction_on_unreviewed_frame" rows below. Never enters
    # match_all(), so it can never become a TP or FP.
    unscored_predictions = [
        p for p in predictions if (p["source_id"], p["frame_index"]) in unreviewed_prediction_frame_keys
    ]
    scored_gt = [g for g in gt_records if (g["source_id"], g["frame_index"]) in evaluated_frame_keys]

    match_result = match_all(scored_predictions, scored_gt, args.iou_threshold)
    per_class_rows = compute_per_class_metrics(match_result, scored_gt, scored_predictions)
    overall = compute_overall_metrics(per_class_rows)

    protected_combined = combine_classes(
        "protected(" + "+".join(PROTECTED_CLASSES) + ")", PROTECTED_CLASSES,
        match_result, scored_gt, scored_predictions,
    )
    priority_combined = combine_classes(
        "priority(" + "+".join(PRIORITY_CLASSES) + ")", PRIORITY_CLASSES,
        match_result, scored_gt, scored_predictions,
    )

    evaluated_predicted_frame_count = len(predicted_frame_keys)
    coverage_rate = (
        len(evaluated_frame_keys) / evaluated_predicted_frame_count
        if evaluated_predicted_frame_count > 0
        else None
    )

    breakdowns = {
        "object_size": gt_attribute_breakdown(match_result, "object_size", OBJECT_SIZE_CATEGORIES),
        "visibility": gt_attribute_breakdown(match_result, "visibility", VISIBILITY_CATEGORIES),
        "occlusion": gt_attribute_breakdown(match_result, "occlusion", OCCLUSION_CATEGORIES),
        "lighting": source_attribute_breakdown(match_result, "lighting"),
        "scene_tags": scene_tag_breakdown(match_result),
    }

    output_dir = resolve_output_dir(predictions_path.stem, args.output_dir, args.overwrite)

    commit, commit_status = get_git_commit(_REPO_ROOT)
    report = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "annotations_path": str(annotations_path),
        "annotations_sha256": sha256_file(annotations_path),
        "iou_threshold": args.iou_threshold,
        "matching_method": "greedy_by_confidence_class_aware_per_frame",
        "difficult_object_policy": "scored identically to normal ground truth (not excluded)",
        "evaluated_frame_count": len(evaluated_frame_keys),
        "frames_reviewed_but_not_predicted": len(frames_reviewed_but_not_predicted),
        "reviewed_frame_count": len(reviewed_frame_keys),
        "reviewed_frames_with_objects": len(scored_annotated_frame_keys),
        "reviewed_empty_frame_count": len(reviewed_empty_frame_keys),
        "unreviewed_prediction_frame_count": len(unreviewed_prediction_frame_keys),
        "scored_prediction_count": len(scored_predictions),
        "unscored_prediction_count": len(unscored_predictions),
        "annotation_coverage_rate": coverage_rate,
        "coverage_defined": coverage_rate is not None,
        "protected_classes": list(PROTECTED_CLASSES),
        "priority_classes": list(PRIORITY_CLASSES),
        "overall": overall,
        "protected_classes_combined": protected_combined,
        "priority_classes_combined": priority_combined,
        "per_class": per_class_rows,
        "breakdowns": breakdowns,
        "annotation_validation_warnings": validation.warnings,
        "git_commit": commit,
        "git_commit_status": commit_status,
    }

    (output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    _write_per_class_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    _write_matched_detections_csv(output_dir / "matched_detections.csv", match_result)
    _write_errors_csv(output_dir / "errors.csv", match_result, unscored_predictions, reviewed_empty_frame_keys)

    return output_dir


def _write_per_class_csv(path: Path, per_class_rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_CLASS_METRICS_CSV_COLUMNS)
        writer.writeheader()
        for row in per_class_rows:
            writer.writerow({key: row.get(key, "") for key in PER_CLASS_METRICS_CSV_COLUMNS})


def _write_matched_detections_csv(path: Path, match_result: MatchResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MATCHED_DETECTIONS_CSV_COLUMNS)
        writer.writeheader()
        for m in match_result.matches:
            gt, pred = m["ground_truth"], m["prediction"]
            writer.writerow(
                {
                    "annotation_id": gt["annotation_id"], "source_id": gt["source_id"],
                    "frame_index": gt["frame_index"], "class_name": gt["class_name"],
                    "iou": round(m["iou"], 4), "confidence": round(pred["confidence"], 4),
                    "difficult": gt["difficult"],
                }
            )


def _write_errors_csv(
    path: Path,
    match_result: MatchResult,
    unscored_predictions: list[dict],
    reviewed_empty_frame_keys: set[tuple[str, int]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ERRORS_CSV_COLUMNS)
        writer.writeheader()

        for pred in match_result.false_positives:
            x1, y1, x2, y2 = pred["bbox"]
            # A false positive on a REVIEWED_EMPTY (or ignore-only) frame
            # is tagged distinctly from an ordinary false positive on a
            # frame that also contains real objects -- both count toward
            # the same false_positives total, but a reviewer filtering
            # errors.csv can tell the two situations apart.
            frame_key = (pred["source_id"], pred["frame_index"])
            error_type = (
                "false_positive_on_reviewed_empty_frame"
                if frame_key in reviewed_empty_frame_keys
                else "false_positive"
            )
            writer.writerow(
                {
                    "error_type": error_type, "source_id": pred["source_id"],
                    "frame_index": pred["frame_index"], "class_name": pred["class_name"],
                    "annotation_id": "", "confidence": round(pred["confidence"], 4),
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2, "difficult": "",
                }
            )
        for gt in match_result.false_negatives:
            x1, y1, x2, y2 = gt["bbox"]
            writer.writerow(
                {
                    "error_type": "false_negative", "source_id": gt["source_id"],
                    "frame_index": gt["frame_index"], "class_name": gt["class_name"],
                    "annotation_id": gt["annotation_id"], "confidence": "",
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                    "difficult": gt["difficult"],
                }
            )
        for pred in match_result.ignored_predictions:
            x1, y1, x2, y2 = pred["bbox"]
            writer.writerow(
                {
                    "error_type": "ignored_prediction", "source_id": pred["source_id"],
                    "frame_index": pred["frame_index"], "class_name": pred["class_name"],
                    "annotation_id": "", "confidence": round(pred["confidence"], 4),
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2, "difficult": "",
                }
            )
        for gt in match_result.ignored_ground_truth:
            x1, y1, x2, y2 = gt["bbox"]
            writer.writerow(
                {
                    "error_type": "ignored_ground_truth", "source_id": gt["source_id"],
                    "frame_index": gt["frame_index"], "class_name": gt["class_name"],
                    "annotation_id": gt["annotation_id"], "confidence": "",
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                    "difficult": gt["difficult"],
                }
            )
        for pred in unscored_predictions:
            # Preserved here rather than silently dropped -- see
            # docs/DETECTOR_ANNOTATIONS.md "Reviewed-frame status". Never
            # counted as a false positive: the frame was never reviewed,
            # so there is no way to honestly say this prediction is wrong.
            x1, y1, x2, y2 = pred["bbox"]
            writer.writerow(
                {
                    "error_type": "prediction_on_unreviewed_frame", "source_id": pred["source_id"],
                    "frame_index": pred["frame_index"], "class_name": pred["class_name"],
                    "annotation_id": "", "confidence": round(pred["confidence"], 4),
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2, "difficult": "",
                }
            )


# --- non-regression comparison ------------------------------------------------


def _class_row(report: dict, class_name: str) -> dict | None:
    for row in report["per_class"]:
        if row["class_name"] == class_name:
            return row
    return None


def compare_evaluations(baseline_report: dict, candidate_report: dict, config: dict) -> dict:
    """Pure function: compares two evaluation_report.json documents and
    returns a comparison dict (never a bare score) with a PASS/FAIL/
    INCONCLUSIVE verdict plus every per-class input that produced it.
    config keys (all have defaults, see parse_args): min_class_samples,
    max_protected_recall_regression, min_priority_recall_improvement,
    priority_regression_tolerance, max_runtime_regression_percent,
    baseline_runtime_summary/candidate_runtime_summary (optional dicts
    with an "effective_processing_fps" key, from benchmark_detector.py's
    summary.json)."""
    min_samples = config.get("min_class_samples", DEFAULT_MIN_CLASS_SAMPLES)
    max_protected_regression = config.get(
        "max_protected_recall_regression", DEFAULT_MAX_PROTECTED_RECALL_REGRESSION
    )
    min_priority_improvement = config.get(
        "min_priority_recall_improvement", DEFAULT_MIN_PRIORITY_RECALL_IMPROVEMENT
    )
    priority_tolerance = config.get("priority_regression_tolerance", DEFAULT_PRIORITY_REGRESSION_TOLERANCE)
    max_runtime_regression_percent = config.get(
        "max_runtime_regression_percent", DEFAULT_MAX_RUNTIME_REGRESSION_PERCENT
    )
    min_review_coverage = config.get("min_review_coverage", DEFAULT_MIN_REVIEW_COVERAGE)

    class_comparisons = {}
    protected_fail_reasons = []
    protected_inconclusive_reasons = []
    priority_fail_reasons = []
    priority_inconclusive_reasons = []
    coverage_inconclusive_reasons = []

    # A candidate must never look better than the baseline merely
    # because fewer of its frames were reviewed -- low review coverage
    # on EITHER side makes the whole comparison inconclusive, not just
    # the affected classes.
    for label, report in (("baseline", baseline_report), ("candidate", candidate_report)):
        coverage = report.get("annotation_coverage_rate")
        if coverage is None or coverage < min_review_coverage:
            coverage_inconclusive_reasons.append(
                f"{label} report's annotation_coverage_rate "
                f"({coverage if coverage is not None else 'undefined'}) is below the minimum "
                f"required review coverage ({min_review_coverage})."
            )

    def compare_one_class(class_name: str, is_protected: bool) -> None:
        baseline_row = _class_row(baseline_report, class_name)
        candidate_row = _class_row(candidate_report, class_name)
        baseline_samples = baseline_row["ground_truth_count"] if baseline_row else 0
        candidate_samples = candidate_row["ground_truth_count"] if candidate_row else 0
        enough_samples = baseline_samples >= min_samples and candidate_samples >= min_samples

        baseline_recall = baseline_row["recall"] if baseline_row and baseline_row["recall_defined"] else None
        candidate_recall = candidate_row["recall"] if candidate_row and candidate_row["recall_defined"] else None
        recall_change = (
            candidate_recall - baseline_recall if baseline_recall is not None and candidate_recall is not None else None
        )

        def _delta(field_name: str) -> float | None:
            b = baseline_row.get(field_name) if baseline_row else None
            c = candidate_row.get(field_name) if candidate_row else None
            if b is None or c is None:
                return None
            return c - b

        entry = {
            "class_name": class_name,
            "baseline_ground_truth_count": baseline_samples,
            "candidate_ground_truth_count": candidate_samples,
            "enough_samples": enough_samples,
            "min_class_samples_required": min_samples,
            "baseline_recall": baseline_recall,
            "candidate_recall": candidate_recall,
            "recall_change": recall_change,
            "precision_change": _delta("precision"),
            "f1_change": _delta("f1"),
            "false_negative_change": (
                (candidate_row["false_negatives"] - baseline_row["false_negatives"])
                if baseline_row and candidate_row else None
            ),
        }
        class_comparisons[class_name] = entry

        if not enough_samples:
            if is_protected:
                protected_inconclusive_reasons.append(
                    f"{class_name}: insufficient samples (baseline={baseline_samples}, "
                    f"candidate={candidate_samples}, required={min_samples})"
                )
            else:
                priority_inconclusive_reasons.append(
                    f"{class_name}: insufficient samples (baseline={baseline_samples}, "
                    f"candidate={candidate_samples}, required={min_samples})"
                )
            return

        if recall_change is None:
            (protected_inconclusive_reasons if is_protected else priority_inconclusive_reasons).append(
                f"{class_name}: recall undefined in baseline or candidate report"
            )
            return

        if is_protected:
            if recall_change < -max_protected_regression:
                protected_fail_reasons.append(
                    f"{class_name}: recall regressed by {-recall_change:.4f} "
                    f"(tolerance {max_protected_regression:.4f})"
                )
        else:
            if recall_change < -priority_tolerance:
                priority_fail_reasons.append(
                    f"{class_name}: recall regressed by {-recall_change:.4f} "
                    f"(tolerance {priority_tolerance:.4f})"
                )
            elif recall_change < min_priority_improvement:
                priority_inconclusive_reasons.append(
                    f"{class_name}: recall change {recall_change:.4f} is below the minimum "
                    f"improvement threshold {min_priority_improvement:.4f} but has not regressed "
                    f"beyond tolerance -- improvement not established"
                )

    for class_name in PROTECTED_CLASSES:
        compare_one_class(class_name, is_protected=True)
    for class_name in PRIORITY_CLASSES:
        compare_one_class(class_name, is_protected=False)

    runtime_comparison = None
    runtime_fail_reasons = []
    runtime_inconclusive_reasons = []
    baseline_runtime = config.get("baseline_runtime_summary")
    candidate_runtime = config.get("candidate_runtime_summary")
    if baseline_runtime is not None or candidate_runtime is not None:
        if baseline_runtime is None or candidate_runtime is None:
            runtime_inconclusive_reasons.append("Runtime comparison requested but one summary is missing.")
        else:
            baseline_fps = baseline_runtime.get("effective_processing_fps")
            candidate_fps = candidate_runtime.get("effective_processing_fps")
            if baseline_fps and candidate_fps:
                percent_change = 100.0 * (candidate_fps - baseline_fps) / baseline_fps
                runtime_comparison = {
                    "baseline_effective_fps": baseline_fps,
                    "candidate_effective_fps": candidate_fps,
                    "percent_change": percent_change,
                }
                if percent_change < -max_runtime_regression_percent:
                    runtime_fail_reasons.append(
                        f"Effective FPS regressed by {-percent_change:.1f}% "
                        f"(tolerance {max_runtime_regression_percent:.1f}%)"
                    )
            else:
                runtime_inconclusive_reasons.append("Runtime summary missing effective_processing_fps.")

    all_fail_reasons = protected_fail_reasons + priority_fail_reasons + runtime_fail_reasons
    all_inconclusive_reasons = (
        protected_inconclusive_reasons + priority_inconclusive_reasons
        + runtime_inconclusive_reasons + coverage_inconclusive_reasons
    )

    # A hard FAIL is never masked by insufficient data elsewhere --
    # including low review coverage. Coverage alone can only ever push
    # PASS down to INCONCLUSIVE, never turn a real regression into a
    # softer verdict.
    if all_fail_reasons:
        verdict = FAIL
    elif all_inconclusive_reasons:
        verdict = INCONCLUSIVE
    else:
        verdict = PASS

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "fail_reasons": all_fail_reasons,
        "inconclusive_reasons": all_inconclusive_reasons,
        "policy_config": {
            "min_class_samples": min_samples,
            "max_protected_recall_regression": max_protected_regression,
            "min_priority_recall_improvement": min_priority_improvement,
            "priority_regression_tolerance": priority_tolerance,
            "max_runtime_regression_percent": max_runtime_regression_percent,
            "min_review_coverage": min_review_coverage,
        },
        "protected_classes": class_comparisons_for(class_comparisons, PROTECTED_CLASSES),
        "priority_classes": class_comparisons_for(class_comparisons, PRIORITY_CLASSES),
        "runtime_comparison": runtime_comparison,
        "review_coverage": {
            "baseline_annotation_coverage_rate": baseline_report.get("annotation_coverage_rate"),
            "candidate_annotation_coverage_rate": candidate_report.get("annotation_coverage_rate"),
            "min_review_coverage_required": min_review_coverage,
        },
    }


def class_comparisons_for(class_comparisons: dict, class_names: tuple) -> dict:
    return {name: class_comparisons[name] for name in class_names if name in class_comparisons}


# --- CLI ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Atlas's detector predictions against hand-authored "
            "ground truth (precision/recall/F1), or compare two prior "
            "evaluation reports for non-regression. See "
            "docs/DETECTOR_ANNOTATIONS.md."
        )
    )
    parser.add_argument("--predictions", default=None, help="Path to a detections.csv from scripts/benchmark_detector.py.")
    parser.add_argument("--annotations", default=None, help="Path to an annotations.json (see docs/DETECTOR_ANNOTATIONS.md).")
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD, help="Minimum IoU to count as a match (default: 0.50).")

    parser.add_argument("--compare-baseline", default=None, help="Path to a baseline evaluation_report.json.")
    parser.add_argument("--compare-candidate", default=None, help="Path to a candidate evaluation_report.json.")
    parser.add_argument("--baseline-runtime-summary", default=None, help="Optional summary.json from benchmark_detector.py for the baseline run.")
    parser.add_argument("--candidate-runtime-summary", default=None, help="Optional summary.json from benchmark_detector.py for the candidate run.")
    parser.add_argument("--min-class-samples", type=int, default=DEFAULT_MIN_CLASS_SAMPLES)
    parser.add_argument("--max-protected-recall-regression", type=float, default=DEFAULT_MAX_PROTECTED_RECALL_REGRESSION)
    parser.add_argument("--min-priority-recall-improvement", type=float, default=DEFAULT_MIN_PRIORITY_RECALL_IMPROVEMENT)
    parser.add_argument("--priority-regression-tolerance", type=float, default=DEFAULT_PRIORITY_REGRESSION_TOLERANCE)
    parser.add_argument("--max-runtime-regression-percent", type=float, default=DEFAULT_MAX_RUNTIME_REGRESSION_PERCENT)
    parser.add_argument(
        "--min-review-coverage", type=float, default=DEFAULT_MIN_REVIEW_COVERAGE,
        help="Minimum annotation_coverage_rate required on BOTH reports, else INCONCLUSIVE (default: 0.80).",
    )

    parser.add_argument("--output-dir", default=None, help="Result directory (default: logs/detector_evaluations/<stem>_<timestamp>).")
    parser.add_argument("--overwrite", action="store_true", help="Allow reusing a non-empty --output-dir (off by default).")

    args = parser.parse_args(argv)

    evaluate_mode = args.predictions is not None or args.annotations is not None
    compare_mode = args.compare_baseline is not None or args.compare_candidate is not None
    if evaluate_mode and compare_mode:
        parser.error("Pass either --predictions/--annotations OR --compare-baseline/--compare-candidate, not both.")
    if not evaluate_mode and not compare_mode:
        parser.error("Pass --predictions and --annotations (evaluate mode) or --compare-baseline and --compare-candidate (compare mode).")
    if evaluate_mode and (args.predictions is None or args.annotations is None):
        parser.error("Evaluate mode requires both --predictions and --annotations.")
    if compare_mode and (args.compare_baseline is None or args.compare_candidate is None):
        parser.error("Compare mode requires both --compare-baseline and --compare-candidate.")
    if not (0.0 <= args.iou_threshold <= 1.0):
        parser.error("--iou-threshold must be within [0, 1]")

    return args


def run_comparison(args: argparse.Namespace) -> Path:
    baseline_report = json.loads(Path(args.compare_baseline).read_text())
    candidate_report = json.loads(Path(args.compare_candidate).read_text())

    config = {
        "min_class_samples": args.min_class_samples,
        "max_protected_recall_regression": args.max_protected_recall_regression,
        "min_priority_recall_improvement": args.min_priority_recall_improvement,
        "priority_regression_tolerance": args.priority_regression_tolerance,
        "max_runtime_regression_percent": args.max_runtime_regression_percent,
        "min_review_coverage": args.min_review_coverage,
        "baseline_runtime_summary": (
            json.loads(Path(args.baseline_runtime_summary).read_text())
            if args.baseline_runtime_summary else None
        ),
        "candidate_runtime_summary": (
            json.loads(Path(args.candidate_runtime_summary).read_text())
            if args.candidate_runtime_summary else None
        ),
    }
    comparison = compare_evaluations(baseline_report, candidate_report, config)

    output_dir = resolve_output_dir("comparison", args.output_dir, args.overwrite)
    (output_dir / "comparison_report.json").write_text(json.dumps(comparison, indent=2))
    return output_dir


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.compare_baseline is not None:
        output_dir = run_comparison(args)
    else:
        output_dir = run_evaluation(args)
    print(f"Evaluation complete. Results written to: {output_dir}")


if __name__ == "__main__":
    main()
