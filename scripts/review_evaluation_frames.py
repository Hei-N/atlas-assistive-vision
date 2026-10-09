"""Makes manual human review of an Atlas evaluation package (e.g. `data/
evaluation/test7_v1/`) easier and checks whether labeling is complete.

Two modes, no detector involved in either:
    python -m scripts.review_evaluation_frames data/evaluation/test7_v1
        Opens a simple frame-by-frame image viewer (OpenCV window) --
        navigate with n/right (next), p/left (previous), q/Esc (quit).
        Shows filename/frame_index/timestamp/selected_reason as on
        -screen text. Never runs object detection, never draws a box,
        never guesses a label -- purely a viewer. Recording review
        status/notes is handled by hand-editing
        `HUMAN_LABELING_CHECKLIST.md` in the same directory, not by this
        script, to keep this tool narrow and avoid a second, competing
        status-tracking mechanism.

    python -m scripts.review_evaluation_frames data/evaluation/test7_v1 --validate
        No GUI. Reports: any of the 35 selected frames missing from
        disk, any frame_index not yet covered by an annotation or a
        `reviewed_frames` entry, and the full result of scripts.
        annotation_schema.validate_annotations() (duplicate IDs, unknown
        classes, malformed boxes, etc.) -- exits 0 if everything is
        complete and valid, 1 otherwise. Never modifies annotations.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.annotation_schema import validate_annotations  # noqa: E402

# cv2 is only needed for the interactive viewer, not for --validate --
# imported lazily inside run_viewer() so `--validate` never requires a
# display-capable environment.


def load_manifest(csv_path: Path) -> list[dict]:
    """Frame rows from frame_manifest.csv, sorted by frame_index."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return sorted(rows, key=lambda r: int(r["frame_index"]))


def format_overlay_text(row: dict, position: int, total: int) -> list[str]:
    return [
        f"[{position + 1}/{total}] {row['image_filename']}",
        f"frame_index={row['frame_index']}  t={row['timestamp_seconds']}s",
        f"selected_reason={row['selected_reason']}",
    ]


def clamp_index(index: int, total: int) -> int:
    """Never wraps -- stays at the first/last frame past either end,
    so repeated 'previous' at frame 1 or 'next' at the last frame is a
    harmless no-op rather than surprising the reviewer by jumping."""
    if total <= 0:
        return 0
    return max(0, min(index, total - 1))


# --- completeness / validation ---------------------------------------------


@dataclass
class CompletenessResult:
    missing_frame_files: list[str] = field(default_factory=list)
    unreviewed_frame_indices: list[int] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)
    schema_warnings: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not (self.missing_frame_files or self.unreviewed_frame_indices or self.schema_errors)


def check_completeness(eval_dir: Path) -> CompletenessResult:
    """Never modifies annotations.json -- read-only. Checks: every
    selected frame's image file exists on disk; every selected frame's
    frame_index is covered by either an annotation or a reviewed_frames
    entry (for that same source_id); the full annotation_schema
    validation result (which already catches duplicate IDs/records, bad
    boxes, unknown classes, etc.)."""
    manifest_rows = load_manifest(eval_dir / "frame_manifest.csv")
    frames_dir = eval_dir / "frames"
    missing = [
        row["image_filename"] for row in manifest_rows
        if not (frames_dir / row["image_filename"]).exists()
    ]

    document = json.loads((eval_dir / "annotations.json").read_text())
    validation = validate_annotations(document)

    reviewed_keys: set[tuple[str, int]] = set()
    for ann in document.get("annotations", []):
        if "source_id" in ann and "frame_index" in ann:
            reviewed_keys.add((ann["source_id"], ann["frame_index"]))
    for entry in document.get("reviewed_frames", []):
        if "source_id" in entry and "frame_index" in entry:
            reviewed_keys.add((entry["source_id"], entry["frame_index"]))

    unreviewed = sorted(
        int(row["frame_index"]) for row in manifest_rows
        if (row["source_id"], int(row["frame_index"])) not in reviewed_keys
    )

    return CompletenessResult(
        missing_frame_files=missing,
        unreviewed_frame_indices=unreviewed,
        schema_errors=validation.errors,
        schema_warnings=validation.warnings,
    )


def print_completeness_report(result: CompletenessResult) -> None:
    print(f"Missing frame files: {len(result.missing_frame_files)}")
    for name in result.missing_frame_files:
        print(f"  - {name}")
    print(f"Unreviewed frames: {len(result.unreviewed_frame_indices)}")
    for idx in result.unreviewed_frame_indices:
        print(f"  - frame_index {idx}")
    print(f"Schema errors: {len(result.schema_errors)}")
    for e in result.schema_errors:
        print(f"  - {e}")
    print(f"Schema warnings: {len(result.schema_warnings)}")
    for w in result.schema_warnings:
        print(f"  - {w}")
    print()
    print("COMPLETE" if result.is_complete else "INCOMPLETE")


# --- interactive viewer (not exercised by tests -- needs a real display,
#     same convention as scripts/benchmark_detectors.py's run_model_
#     benchmark loop) -------------------------------------------------------


def run_viewer(frames_dir: Path, manifest_rows: list[dict]) -> None:
    """Simple next/previous image browser. No detection, no boxes, no
    label guessing -- see the module docstring."""
    import cv2

    total = len(manifest_rows)
    index = 0
    window_name = "Atlas evaluation frame review (n=next, p=previous, q=quit)"
    while True:
        row = manifest_rows[index]
        frame = cv2.imread(str(frames_dir / row["image_filename"]))
        if frame is None:
            print(f"WARNING: could not read {row['image_filename']}")
        else:
            canvas = frame.copy()
            for line_index, line in enumerate(format_overlay_text(row, index, total)):
                cv2.putText(
                    canvas, line, (10, 30 + line_index * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
                )
            cv2.imshow(window_name, canvas)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break
        elif key in (ord("n"), ord("d"), 83):  # n / d / right-arrow
            index = clamp_index(index + 1, total)
        elif key in (ord("p"), ord("a"), 81):  # p / a / left-arrow
            index = clamp_index(index - 1, total)

    cv2.destroyAllWindows()


# --- CLI ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Browse an Atlas evaluation package's selected frames for manual "
            "human review, or check whether labeling is complete. Never runs "
            "a detector, never draws or guesses a label."
        )
    )
    parser.add_argument(
        "eval_dir",
        help="Path to an evaluation package directory, e.g. data/evaluation/test7_v1",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Report labeling completeness instead of opening the viewer (no GUI needed).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    eval_dir = Path(args.eval_dir)

    if args.validate:
        result = check_completeness(eval_dir)
        print_completeness_report(result)
        sys.exit(0 if result.is_complete else 1)

    manifest_rows = load_manifest(eval_dir / "frame_manifest.csv")
    run_viewer(eval_dir / "frames", manifest_rows)


if __name__ == "__main__":
    main()
