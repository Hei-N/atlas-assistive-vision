"""Unit tests for scripts/annotation_schema.py -- ground-truth
annotation schema validation and object-size classification. Pure
logic, no network/model/media access.
"""

from __future__ import annotations

import copy

import pytest

from scripts.annotation_schema import (
    CANONICAL_CLASSES,
    MEDIUM_LARGE_BOUNDARY,
    SMALL_MEDIUM_BOUNDARY,
    classify_object_size,
    coco_json_to_atlas_annotations,
    validate_annotations,
)


def make_valid_document() -> dict:
    return {
        "schema_version": "1.0",
        "dataset_id": "atlas_eval_v1",
        "created_at": "2026-08-07T00:00:00Z",
        "class_names": list(CANONICAL_CLASSES),
        "sources": [
            {
                "source_id": "clip_001", "path": "data/input/clip_001.mp4",
                "source_type": "video", "width": 640, "height": 480,
                "total_frames": 10, "fps": 30.0,
                "scene_tags": ["daytime"], "lighting": "bright",
                "camera_motion": "static", "notes": "",
            }
        ],
        "annotations": [
            {
                "annotation_id": "ann_001", "source_id": "clip_001", "frame_index": 0,
                "timestamp_seconds": 0.0, "class_name": "car",
                "bounding_box_xyxy": [10, 10, 100, 100],
                "visibility": "clear", "occlusion": "none",
                "object_size": None, "ignore": False, "difficult": False,
                "vehicle_subtype_note": None, "notes": "",
            }
        ],
    }


# --- 1: valid annotation file passes -----------------------------------


def test_valid_annotation_file_passes() -> None:
    result = validate_annotations(make_valid_document())
    assert result.is_valid
    assert result.errors == []


# --- 2: missing required field fails clearly ----------------------------


def test_missing_required_top_level_field_fails_clearly() -> None:
    doc = make_valid_document()
    del doc["dataset_id"]
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("dataset_id" in e for e in result.errors)


def test_missing_required_source_field_fails_clearly() -> None:
    doc = make_valid_document()
    del doc["sources"][0]["path"]
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("path" in e for e in result.errors)


def test_missing_required_annotation_field_fails_clearly() -> None:
    doc = make_valid_document()
    del doc["annotations"][0]["class_name"]
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("class_name" in e for e in result.errors)


# --- 3: unknown schema version fails --------------------------------------


def test_unknown_schema_version_fails() -> None:
    doc = make_valid_document()
    doc["schema_version"] = "99.0"
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("schema_version" in e for e in result.errors)


# --- 4: invalid class fails ------------------------------------------------


def test_invalid_class_fails() -> None:
    doc = make_valid_document()
    doc["annotations"][0]["class_name"] = "e-bike"
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("class_name" in e for e in result.errors)


# --- 5: invalid source reference fails ------------------------------------


def test_invalid_source_reference_fails() -> None:
    doc = make_valid_document()
    doc["annotations"][0]["source_id"] = "does_not_exist"
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("unknown source_id" in e for e in result.errors)


# --- 6: invalid frame index fails -------------------------------------------


def test_negative_frame_index_fails() -> None:
    doc = make_valid_document()
    doc["annotations"][0]["frame_index"] = -1
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("frame_index" in e for e in result.errors)


def test_frame_index_beyond_total_frames_fails() -> None:
    doc = make_valid_document()
    doc["annotations"][0]["frame_index"] = 999
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("total_frames" in e for e in result.errors)


def test_nonzero_frame_index_on_image_source_fails() -> None:
    doc = make_valid_document()
    doc["sources"][0]["source_type"] = "image"
    doc["annotations"][0]["frame_index"] = 3
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("image source" in e for e in result.errors)


# --- 7: invalid bounding box fails ------------------------------------------


def test_non_numeric_bounding_box_fails() -> None:
    doc = make_valid_document()
    doc["annotations"][0]["bounding_box_xyxy"] = [10, "x", 100, 100]
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("non-numeric" in e for e in result.errors)


def test_x1_not_less_than_x2_fails() -> None:
    doc = make_valid_document()
    doc["annotations"][0]["bounding_box_xyxy"] = [100, 10, 50, 100]
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("x1<x2" in e for e in result.errors)


def test_non_finite_bounding_box_fails() -> None:
    doc = make_valid_document()
    doc["annotations"][0]["bounding_box_xyxy"] = [10, 10, float("inf"), 100]
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("non-numeric/non-finite" in e for e in result.errors)


# --- 8: out-of-bounds box fails ---------------------------------------------


def test_out_of_bounds_box_fails() -> None:
    doc = make_valid_document()
    doc["annotations"][0]["bounding_box_xyxy"] = [10, 10, 10000, 100]
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("outside frame bounds" in e for e in result.errors)


# --- 9: duplicate annotation ID fails ---------------------------------------


def test_duplicate_annotation_id_fails() -> None:
    doc = make_valid_document()
    duplicate = copy.deepcopy(doc["annotations"][0])
    doc["annotations"].append(duplicate)
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("Duplicate annotation_id" in e for e in result.errors)


def test_duplicate_source_id_fails() -> None:
    doc = make_valid_document()
    doc["sources"].append(copy.deepcopy(doc["sources"][0]))
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("Duplicate source_id" in e for e in result.errors)


# --- 10: ignore/difficult validate ------------------------------------------


def test_ignore_and_difficult_accept_valid_booleans() -> None:
    doc = make_valid_document()
    doc["annotations"][0]["ignore"] = True
    doc["annotations"][0]["difficult"] = True
    result = validate_annotations(doc)
    assert result.is_valid


def test_ignore_field_must_be_boolean() -> None:
    doc = make_valid_document()
    doc["annotations"][0]["ignore"] = "yes"
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("'ignore'" in e for e in result.errors)


def test_difficult_field_must_be_boolean() -> None:
    doc = make_valid_document()
    doc["annotations"][0]["difficult"] = 1
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("'difficult'" in e for e in result.errors)


# --- visibility / occlusion / object_size validity --------------------------


def test_invalid_visibility_fails() -> None:
    doc = make_valid_document()
    doc["annotations"][0]["visibility"] = "blurry"
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("visibility" in e for e in result.errors)


def test_invalid_occlusion_fails() -> None:
    doc = make_valid_document()
    doc["annotations"][0]["occlusion"] = "total"
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("occlusion" in e for e in result.errors)


def test_invalid_object_size_if_present_fails() -> None:
    doc = make_valid_document()
    doc["annotations"][0]["object_size"] = "huge"
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("object_size" in e for e in result.errors)


def test_null_object_size_is_allowed() -> None:
    doc = make_valid_document()
    doc["annotations"][0]["object_size"] = None
    result = validate_annotations(doc)
    assert result.is_valid


# --- near-duplicate warning (not a hard error) ------------------------------


def test_near_duplicate_annotation_produces_warning_not_error() -> None:
    doc = make_valid_document()
    near_dup = copy.deepcopy(doc["annotations"][0])
    near_dup["annotation_id"] = "ann_002"
    near_dup["bounding_box_xyxy"] = [11, 11, 101, 101]  # near-identical box
    doc["annotations"].append(near_dup)
    result = validate_annotations(doc)
    assert result.is_valid  # not a hard error
    assert any("near-duplicate" in w for w in result.warnings)


def test_missing_source_path_produces_warning_when_base_dir_given(tmp_path) -> None:
    doc = make_valid_document()
    result = validate_annotations(doc, base_dir=tmp_path)
    assert result.is_valid
    assert any("does not exist" in w for w in result.warnings)


def test_missing_source_path_is_skipped_without_base_dir() -> None:
    doc = make_valid_document()
    result = validate_annotations(doc)
    assert result.warnings == []


# --- 11: object-size category calculation is correct ------------------------


def test_object_size_small_below_boundary() -> None:
    frame_area = 1000.0
    bbox_area = frame_area * (SMALL_MEDIUM_BOUNDARY - 0.001)
    assert classify_object_size(bbox_area, frame_area) == "small"


def test_object_size_medium_between_boundaries() -> None:
    frame_area = 1000.0
    bbox_area = frame_area * (SMALL_MEDIUM_BOUNDARY + 0.01)
    assert classify_object_size(bbox_area, frame_area) == "medium"


def test_object_size_large_above_boundary() -> None:
    frame_area = 1000.0
    bbox_area = frame_area * (MEDIUM_LARGE_BOUNDARY + 0.01)
    assert classify_object_size(bbox_area, frame_area) == "large"


def test_object_size_exact_boundary_values() -> None:
    frame_area = 10000.0
    assert classify_object_size(frame_area * SMALL_MEDIUM_BOUNDARY, frame_area) == "medium"
    assert classify_object_size(frame_area * MEDIUM_LARGE_BOUNDARY, frame_area) == "large"


def test_object_size_handles_non_positive_frame_area_safely() -> None:
    assert classify_object_size(100.0, 0.0) == "medium"
    assert classify_object_size(100.0, -5.0) == "medium"


# --- reviewed_frames: schema/validation (required tests 1-6) ---------------


def test_valid_reviewed_empty_frame_passes_validation() -> None:
    doc = make_valid_document()
    doc["reviewed_frames"] = [
        {"source_id": "clip_001", "frame_index": 5, "reviewed": True, "notes": "No supported objects visible."}
    ]
    result = validate_annotations(doc)
    assert result.is_valid, result.errors


def test_reviewed_frames_absent_is_valid_and_backward_compatible() -> None:
    doc = make_valid_document()
    assert "reviewed_frames" not in doc  # an old schema-"1.0" file
    result = validate_annotations(doc)
    assert result.is_valid


def test_unknown_source_in_reviewed_frames_fails() -> None:
    doc = make_valid_document()
    doc["reviewed_frames"] = [{"source_id": "does_not_exist", "frame_index": 0, "reviewed": True}]
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("unknown source_id" in e for e in result.errors)


def test_out_of_range_frame_index_in_reviewed_frames_fails() -> None:
    doc = make_valid_document()
    doc["reviewed_frames"] = [{"source_id": "clip_001", "frame_index": 999, "reviewed": True}]
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("total_frames" in e for e in result.errors)


def test_duplicate_reviewed_frame_entry_fails() -> None:
    doc = make_valid_document()
    doc["reviewed_frames"] = [
        {"source_id": "clip_001", "frame_index": 5, "reviewed": True},
        {"source_id": "clip_001", "frame_index": 5, "reviewed": True},
    ]
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("duplicates an earlier entry" in e for e in result.errors)


def test_non_boolean_reviewed_value_fails() -> None:
    doc = make_valid_document()
    doc["reviewed_frames"] = [{"source_id": "clip_001", "frame_index": 5, "reviewed": "yes"}]
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("must be the boolean true" in e for e in result.errors)


def test_reviewed_false_is_explicitly_rejected() -> None:
    doc = make_valid_document()
    doc["reviewed_frames"] = [{"source_id": "clip_001", "frame_index": 5, "reviewed": False}]
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("reviewed=false" in e for e in result.errors)


def test_reviewed_frames_missing_required_field_fails() -> None:
    doc = make_valid_document()
    doc["reviewed_frames"] = [{"source_id": "clip_001", "frame_index": 5}]  # missing "reviewed"
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("reviewed_frames[0] missing required field" in e for e in result.errors)


def test_reviewed_frames_invalid_timestamp_fails() -> None:
    doc = make_valid_document()
    doc["reviewed_frames"] = [
        {"source_id": "clip_001", "frame_index": 5, "reviewed": True, "timestamp_seconds": -1.0}
    ]
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("timestamp_seconds" in e for e in result.errors)


def test_reviewed_frames_image_source_nonzero_frame_index_fails() -> None:
    doc = make_valid_document()
    doc["sources"][0]["source_type"] = "image"
    doc["reviewed_frames"] = [{"source_id": "clip_001", "frame_index": 3, "reviewed": True}]
    result = validate_annotations(doc)
    assert not result.is_valid
    assert any("image source" in e for e in result.errors)


def test_reviewed_frames_and_real_annotation_on_same_frame_is_not_a_conflict() -> None:
    """Redundant, not an error -- a human confirming review AND
    recording an object is consistent, deterministic behavior."""
    doc = make_valid_document()
    doc["reviewed_frames"] = [{"source_id": "clip_001", "frame_index": 0, "reviewed": True}]
    result = validate_annotations(doc)
    assert result.is_valid


# --- COCO import (narrow, optional) -----------------------------------------


def test_coco_import_maps_known_categories() -> None:
    coco_data = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 3, "bbox": [10, 20, 30, 40], "iscrowd": 0},
        ],
    }
    document, warnings = coco_json_to_atlas_annotations(
        coco_data, class_name_map={3: "car"}, dataset_id="ds", created_at="2026-01-01T00:00:00Z",
    )
    assert warnings == []
    assert len(document["annotations"]) == 1
    ann = document["annotations"][0]
    assert ann["class_name"] == "car"
    assert ann["bounding_box_xyxy"] == [10, 20, 40, 60]  # x,y,w,h -> x1,y1,x2,y2
    assert document["sources"][0]["source_id"] == "img1.jpg"


def test_coco_import_skips_unmapped_categories_with_warning() -> None:
    coco_data = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 99, "bbox": [10, 20, 30, 40], "iscrowd": 0},
        ],
    }
    document, warnings = coco_json_to_atlas_annotations(
        coco_data, class_name_map={3: "car"}, dataset_id="ds", created_at="2026-01-01T00:00:00Z",
    )
    assert document["annotations"] == []
    assert len(warnings) == 1
    assert "99" in warnings[0]


def test_coco_import_result_passes_validation() -> None:
    coco_data = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 3, "bbox": [10, 20, 30, 40], "iscrowd": 0},
        ],
    }
    document, _warnings = coco_json_to_atlas_annotations(
        coco_data, class_name_map={3: "car"}, dataset_id="ds", created_at="2026-01-01T00:00:00Z",
    )
    result = validate_annotations(document)
    assert result.is_valid, result.errors
