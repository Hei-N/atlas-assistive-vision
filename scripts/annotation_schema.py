"""Ground-truth annotation schema, validation, and object-size
classification for Atlas's labeled detector evaluation (scripts/
evaluate_detector.py) -- see docs/DETECTOR_ANNOTATIONS.md for the full
specification.

Reuses (never duplicates) scripts/benchmark_detector.py's
PROTECTED_CLASSES/PRIORITY_CLASSES for the canonical class vocabulary
and its iou() helper for near-duplicate detection -- this module owns
schema/validation only, not detection/matching logic.

This is schema + validation infrastructure only -- no annotation
authoring GUI, no automatic ground-truth generation. Annotations are
expected to be hand-authored as JSON, or converted from a COCO JSON
export via coco_json_to_atlas_annotations() (the only supported import path).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmark_detector import PRIORITY_CLASSES, PROTECTED_CLASSES, iou  # noqa: E402

SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})

# The evaluation vocabulary is exactly Atlas's configured detector
# classes -- reused, not redefined, from scripts/benchmark_detector.py.
CANONICAL_CLASSES = tuple(sorted(set(PROTECTED_CLASSES) | set(PRIORITY_CLASSES) | {"person"}))

SOURCE_TYPES = ("image", "video")
VISIBILITY_CATEGORIES = ("clear", "reduced", "poor")
OCCLUSION_CATEGORIES = ("none", "partial", "heavy")
OBJECT_SIZE_CATEGORIES = ("small", "medium", "large")

# Atlas-specific NORMALIZED-area thresholds (bbox_area / frame_area) --
# deliberately not COCO's absolute-pixel thresholds (32^2/96^2), since
# Atlas processes arbitrary resolutions rather than one fixed image
# size. A judgment call, not a claim about physical distance -- object
# size in image pixels is not the same thing as real-world distance.
SMALL_MEDIUM_BOUNDARY = 0.02
MEDIUM_LARGE_BOUNDARY = 0.10

NEAR_DUPLICATE_IOU_THRESHOLD = 0.9

REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version", "dataset_id", "created_at", "class_names", "sources", "annotations",
)
REQUIRED_SOURCE_FIELDS = ("source_id", "path", "source_type")
REQUIRED_ANNOTATION_FIELDS = (
    "annotation_id", "source_id", "frame_index", "class_name",
    "bounding_box_xyxy", "visibility", "occlusion", "ignore", "difficult",
)

# "reviewed_frames" is a purely OPTIONAL, additive top-level field (see
# docs/DETECTOR_ANNOTATIONS.md "Reviewed-frame status") -- deliberately
# absent from REQUIRED_TOP_LEVEL_FIELDS so a pre-existing schema-"1.0"
# file with no such key continues to validate exactly as before. Each
# entry proves a human inspected a frame that has zero SCORED (non
# -ignore) ground-truth objects; a frame with one or more annotation
# entries (scored or ignore-only) is already implicitly reviewed and
# does not need one of these.
REQUIRED_REVIEWED_FRAME_FIELDS = ("source_id", "frame_index", "reviewed")


@dataclass
class ValidationResult:
    """errors block evaluation (validate_annotations() found the file
    structurally invalid); warnings never block evaluation but should
    be surfaced to the user (e.g. near-duplicate annotations, a source
    path that doesn't exist on disk)."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def classify_object_size(bbox_area: float, frame_area: float) -> str:
    """small/medium/large from bbox_area / frame_area -- see the module
    docstring's threshold constants. Returns "medium" (a safe, non
    -crashing default) for a non-positive frame_area rather than
    raising or dividing by zero -- this should not happen for any
    annotation that already passed validate_annotations()'s bounds
    checks, but this function is intentionally defensive on its own."""
    if frame_area <= 0:
        return "medium"
    ratio = bbox_area / frame_area
    if ratio < SMALL_MEDIUM_BOUNDARY:
        return "small"
    if ratio < MEDIUM_LARGE_BOUNDARY:
        return "medium"
    return "large"


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_frame_index(frame_index: object, source: dict | None, label: str) -> list[str]:
    """Shared by both `annotations[]` and `reviewed_frames[]` validation
    -- the exact same non-negative/in-bounds/image-source rules apply
    to both, so this is factored out once rather than duplicated."""
    errors: list[str] = []
    valid = isinstance(frame_index, int) and not isinstance(frame_index, bool) and frame_index >= 0
    if not valid:
        errors.append(f"{label} has invalid frame_index {frame_index!r}.")
        return errors
    if source is None:
        return errors
    if source["source_type"] == "image" and frame_index != 0:
        errors.append(f"{label} is on an image source but frame_index {frame_index} != 0.")
    total_frames = source.get("total_frames")
    if total_frames is not None and frame_index >= total_frames:
        errors.append(f"{label} frame_index {frame_index} >= source total_frames {total_frames}.")
    return errors


def validate_annotations(data: object, base_dir: Path | None = None) -> ValidationResult:
    """Validates an already-parsed annotations.json document (a dict).
    Never raises for malformed input -- always returns a ValidationResult
    so every problem can be reported at once rather than fixing one
    crash at a time. Does not read or modify any source media file;
    base_dir (if given) is used ONLY to check a source's `path` exists,
    reported as a warning (never a hard error -- evaluation itself only
    needs the width/height already recorded in the JSON)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict):
        return ValidationResult(errors=["Top-level annotations document must be a JSON object."])

    missing_top = [name for name in REQUIRED_TOP_LEVEL_FIELDS if name not in data]
    if missing_top:
        errors.append(f"Missing required top-level field(s): {missing_top}")
        return ValidationResult(errors=errors)

    if data["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(
            f"Unsupported schema_version {data['schema_version']!r} "
            f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
        )
        return ValidationResult(errors=errors)

    sources = data["sources"]
    annotations = data["annotations"]
    if not isinstance(sources, list) or not isinstance(annotations, list):
        errors.append("'sources' and 'annotations' must both be lists.")
        return ValidationResult(errors=errors)

    source_by_id: dict = {}
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            errors.append(f"sources[{index}] is not an object.")
            continue
        missing = [name for name in REQUIRED_SOURCE_FIELDS if name not in source]
        if missing:
            errors.append(f"sources[{index}] missing required field(s): {missing}")
            continue

        source_id = source["source_id"]
        if source_id in source_by_id:
            errors.append(f"Duplicate source_id {source_id!r} at sources[{index}].")
        source_by_id[source_id] = source

        if source["source_type"] not in SOURCE_TYPES:
            errors.append(
                f"sources[{index}] ({source_id!r}) has invalid source_type "
                f"{source['source_type']!r} (must be one of {SOURCE_TYPES})."
            )
        for dim_field in ("width", "height"):
            if dim_field in source and not (
                _is_finite_number(source[dim_field]) and source[dim_field] > 0
            ):
                errors.append(
                    f"sources[{index}] ({source_id!r}) has invalid {dim_field}: "
                    f"{source.get(dim_field)!r}"
                )
        if base_dir is not None:
            candidate = base_dir / str(source["path"])
            if not candidate.exists():
                warnings.append(
                    f"sources[{index}] ({source_id!r}) path does not exist: {candidate}"
                )

    seen_annotation_ids: set = set()
    seen_boxes_by_signature: dict[tuple, list[tuple]] = {}
    for index, ann in enumerate(annotations):
        if not isinstance(ann, dict):
            errors.append(f"annotations[{index}] is not an object.")
            continue
        missing = [name for name in REQUIRED_ANNOTATION_FIELDS if name not in ann]
        if missing:
            errors.append(f"annotations[{index}] missing required field(s): {missing}")
            continue

        ann_id = ann["annotation_id"]
        if ann_id in seen_annotation_ids:
            errors.append(f"Duplicate annotation_id {ann_id!r} at annotations[{index}].")
        seen_annotation_ids.add(ann_id)

        source_id = ann["source_id"]
        source = source_by_id.get(source_id)
        if source is None:
            errors.append(
                f"annotations[{index}] ({ann_id!r}) references unknown source_id {source_id!r}."
            )

        frame_index = ann["frame_index"]
        errors.extend(_validate_frame_index(frame_index, source, f"annotations[{index}] ({ann_id!r})"))

        class_name = ann["class_name"]
        if class_name not in CANONICAL_CLASSES:
            errors.append(
                f"annotations[{index}] ({ann_id!r}) has invalid class_name {class_name!r} "
                f"(allowed: {CANONICAL_CLASSES})."
            )

        bbox = ann["bounding_box_xyxy"]
        bbox_valid = (
            isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(_is_finite_number(v) for v in bbox)
        )
        if not bbox_valid:
            errors.append(
                f"annotations[{index}] ({ann_id!r}) has a non-numeric/non-finite "
                f"bounding_box_xyxy: {bbox!r}."
            )
        else:
            x1, y1, x2, y2 = bbox
            if not (x1 < x2 and y1 < y2):
                errors.append(
                    f"annotations[{index}] ({ann_id!r}) bounding box must satisfy x1<x2 "
                    f"and y1<y2: {bbox!r}."
                )
            if source is not None and "width" in source and "height" in source:
                width, height = source["width"], source["height"]
                if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    errors.append(
                        f"annotations[{index}] ({ann_id!r}) bounding box {bbox!r} is outside "
                        f"frame bounds ({width}x{height})."
                    )

        if ann["visibility"] not in VISIBILITY_CATEGORIES:
            errors.append(
                f"annotations[{index}] ({ann_id!r}) has invalid visibility {ann['visibility']!r} "
                f"(allowed: {VISIBILITY_CATEGORIES})."
            )
        if ann["occlusion"] not in OCCLUSION_CATEGORIES:
            errors.append(
                f"annotations[{index}] ({ann_id!r}) has invalid occlusion {ann['occlusion']!r} "
                f"(allowed: {OCCLUSION_CATEGORIES})."
            )
        object_size = ann.get("object_size")
        if object_size is not None and object_size not in OBJECT_SIZE_CATEGORIES:
            errors.append(
                f"annotations[{index}] ({ann_id!r}) has invalid object_size {object_size!r} "
                f"(allowed: {OBJECT_SIZE_CATEGORIES} or null -- it is always recomputed by "
                "the evaluator regardless)."
            )

        for bool_field in ("ignore", "difficult"):
            if not isinstance(ann[bool_field], bool):
                errors.append(
                    f"annotations[{index}] ({ann_id!r}) field {bool_field!r} must be a boolean, "
                    f"got {ann[bool_field]!r}."
                )

        if bbox_valid:
            signature = (source_id, frame_index, class_name)
            for prior_bbox in seen_boxes_by_signature.get(signature, []):
                if iou(tuple(bbox), tuple(prior_bbox)) >= NEAR_DUPLICATE_IOU_THRESHOLD:
                    warnings.append(
                        f"annotations[{index}] ({ann_id!r}) looks like a near-duplicate of "
                        f"another annotation on the same source/frame/class (IoU >= "
                        f"{NEAR_DUPLICATE_IOU_THRESHOLD})."
                    )
            seen_boxes_by_signature.setdefault(signature, []).append(tuple(bbox))

    reviewed_frames = data.get("reviewed_frames", [])
    if not isinstance(reviewed_frames, list):
        errors.append("'reviewed_frames', if present, must be a list.")
    else:
        seen_reviewed_keys: set[tuple] = set()
        for index, entry in enumerate(reviewed_frames):
            if not isinstance(entry, dict):
                errors.append(f"reviewed_frames[{index}] is not an object.")
                continue
            missing = [name for name in REQUIRED_REVIEWED_FRAME_FIELDS if name not in entry]
            if missing:
                errors.append(f"reviewed_frames[{index}] missing required field(s): {missing}")
                continue

            source_id = entry["source_id"]
            source = source_by_id.get(source_id)
            if source is None:
                errors.append(f"reviewed_frames[{index}] references unknown source_id {source_id!r}.")

            frame_index = entry["frame_index"]
            errors.extend(_validate_frame_index(frame_index, source, f"reviewed_frames[{index}]"))

            reviewed = entry["reviewed"]
            if reviewed is not True:
                # Recommended simplification (see docs/DETECTOR_ANNOTATIONS.md):
                # only reviewed=true entries are allowed. An unreviewed frame
                # is represented by ABSENCE from this list, not reviewed=false
                # -- allowing both would be two ways to say the same "not
                # reviewed" thing, which is exactly the kind of ambiguity
                # this field exists to remove.
                if isinstance(reviewed, bool):
                    errors.append(
                        f"reviewed_frames[{index}] has reviewed=false, which is not allowed -- "
                        "represent an unreviewed frame by omitting it from reviewed_frames "
                        "entirely, not by listing it with reviewed=false."
                    )
                else:
                    errors.append(
                        f"reviewed_frames[{index}] field 'reviewed' must be the boolean true, "
                        f"got {reviewed!r}."
                    )

            timestamp_seconds = entry.get("timestamp_seconds")
            if timestamp_seconds is not None and not (
                _is_finite_number(timestamp_seconds) and timestamp_seconds >= 0
            ):
                errors.append(
                    f"reviewed_frames[{index}] has invalid timestamp_seconds {timestamp_seconds!r} "
                    "(must be finite and non-negative)."
                )

            key = (source_id, frame_index)
            if key in seen_reviewed_keys:
                errors.append(
                    f"reviewed_frames[{index}] duplicates an earlier entry for "
                    f"source_id={source_id!r}, frame_index={frame_index!r}."
                )
            seen_reviewed_keys.add(key)

    return ValidationResult(errors=errors, warnings=warnings)


# --- narrow, optional COCO JSON import -------------------------------------


def coco_json_to_atlas_annotations(
    coco_data: dict,
    class_name_map: dict[int, str],
    *,
    dataset_id: str,
    created_at: str,
    source_type: str = "image",
) -> tuple[dict, list[str]]:
    """Converts a COCO-format JSON document (`images`/`annotations`/
    `categories`) into an Atlas annotations.json document. Narrow and
    one-directional -- COCO images become single-frame Atlas "image"
    sources (frame_index always 0); COCO has no video/frame-index
    concept, so a COCO-derived video dataset is out of scope here.

    class_name_map: COCO category_id -> Atlas canonical class name.
    Any COCO category NOT present in this map is skipped entirely (its
    annotations are dropped, never guessed into a wrong canonical
    class) -- every skip is collected into the returned warnings list,
    never silent.

    Returns (atlas_annotations_dict, warnings).
    """
    warnings: list[str] = []

    images_by_id = {image["id"]: image for image in coco_data.get("images", [])}
    sources = []
    source_id_by_image_id = {}
    for image in coco_data.get("images", []):
        source_id = str(image.get("file_name", image["id"]))
        source_id_by_image_id[image["id"]] = source_id
        sources.append(
            {
                "source_id": source_id,
                "path": str(image.get("file_name", "")),
                "source_type": source_type,
                "width": image.get("width"),
                "height": image.get("height"),
                "scene_tags": [],
                "lighting": "unknown",
                "camera_motion": "unknown",
                "notes": "Imported from COCO JSON.",
            }
        )

    annotations = []
    for index, coco_ann in enumerate(coco_data.get("annotations", [])):
        category_id = coco_ann.get("category_id")
        class_name = class_name_map.get(category_id)
        if class_name is None:
            warnings.append(
                f"Skipped COCO annotation {coco_ann.get('id', index)}: category_id "
                f"{category_id!r} is not in class_name_map."
            )
            continue
        if class_name not in CANONICAL_CLASSES:
            warnings.append(
                f"Skipped COCO annotation {coco_ann.get('id', index)}: mapped class_name "
                f"{class_name!r} is not a canonical Atlas class {CANONICAL_CLASSES}."
            )
            continue

        image_id = coco_ann.get("image_id")
        if image_id not in source_id_by_image_id:
            warnings.append(
                f"Skipped COCO annotation {coco_ann.get('id', index)}: unknown image_id {image_id!r}."
            )
            continue

        x, y, width, height = coco_ann["bbox"]  # COCO bbox is [x, y, width, height]
        annotations.append(
            {
                "annotation_id": f"coco_{coco_ann.get('id', index)}",
                "source_id": source_id_by_image_id[image_id],
                "frame_index": 0,
                "timestamp_seconds": None,
                "class_name": class_name,
                "bounding_box_xyxy": [x, y, x + width, y + height],
                "visibility": "clear",
                "occlusion": "none",
                "object_size": None,
                "ignore": bool(coco_ann.get("iscrowd", 0)),
                "difficult": False,
                "vehicle_subtype_note": None,
                "notes": "Imported from COCO JSON -- visibility/occlusion default to "
                "clear/none and should be reviewed by a human.",
            }
        )

    document = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "created_at": created_at,
        "class_names": list(CANONICAL_CLASSES),
        "sources": sources,
        "annotations": annotations,
    }
    return document, warnings
