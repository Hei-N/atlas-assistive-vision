# Atlas Evaluation Clip #1 — Test7

`data/input/Test7.mp4` is Atlas's first real detector-evaluation source.
This directory holds a small, human-labelable subset of its frames — the
frames were selected for human annotation, not yet annotated.

**Disclosed limitation**: this clip is filmed footage of a laptop screen
(not a direct camera feed) — parts of it show Atlas itself running live,
including its own detection overlay boxes/labels burned into the image.
It is not a clean, unprocessed camera capture. It's usable for building
and exercising the annotation/evaluation pipeline end-to-end, but keep
this in mind when interpreting results — a future, cleaner clip (a
direct camera recording of real street traffic) would give a more
representative evaluation.

**Do NOT use the burned-in overlay as ground truth.** Some frames show
Atlas's own live detection boxes/labels (e.g. `car | STATIONARY`,
`bus | MOVING`). The overlay was only used, informally, to help *locate*
a few interesting frames while selecting this set (e.g. spotting the
school bus) — it must never be copied into `annotations.json`. In
particular:
- One selected frame (`selected_reason: bicycle_or_motorcycle_candidate`)
  shows a rider that Atlas's own overlay labels `motorcycle`, but that
  looks visually like a person on a bicycle. **This ambiguity is
  intentionally left for a human to resolve** — do not default to
  whatever the overlay says.
- One frame shows a white minivan that the overlay labels `car`. Judge
  its class yourself rather than trusting that label.
- More generally: Atlas is the system being evaluated here: using its
  own overlay as ground truth would make the evaluation circular.

## Canonical classes

Only these six are valid `class_name` values (see
`docs/DETECTOR_ANNOTATIONS.md` §3):

- person
- bicycle
- motorcycle
- car
- bus
- truck

**Never invent `e-bike`, `scooter`, or `moped` as a canonical class** —
none of them exist in Atlas's detector class set. If you visually
recognize what looks like an e-bike, record it as `class_name: "bicycle"`
with `"vehicle_subtype_note": "possible_e_bike"` instead.

## Review requirement (important)

**Every selected frame must be fully reviewed** — inspect the whole
frame, not just the parts that catch your eye.

- If the frame has a supported object: draw a box for every one of them
  (`bounding_box_xyxy`, `class_name`, `visibility`, `occlusion`).
- **If no supported object is present, add a `reviewed_frames` entry**
  (`{"source_id": "test7", "frame_index": ..., "reviewed": true}`)
  instead of leaving the frame with nothing. A frame with neither an
  annotation nor a `reviewed_frames` entry is indistinguishable from one
  nobody looked at — its predictions will be excluded from scoring
  entirely rather than counted as false positives. See
  `docs/DETECTOR_ANNOTATIONS.md` §6 for the exact rule.

`annotations.json` in this directory currently has an **empty**
`annotations` array and **no** `reviewed_frames` entries — every one of
the 35 selected frames is intentionally UNREVIEWED right now. Nothing in
it was fabricated; fill it in by hand (or via the COCO-import path in
`docs/DETECTOR_ANNOTATIONS.md` §15) as you label.

## Files

- `frames/` — 35 frames, JPEG, native 1920×1080 resolution (never resized).
- `frame_manifest.csv` — one row per frame: source/index/timestamp/
  filename/dimensions/why it was picked (`selected_reason`).
- `annotations.json` — the starter ground-truth file, schema-valid,
  currently empty (no boxes, no reviewed-empty markers — see above).
- This `README.md`.

## Next Step — Human Annotation

Labeling has not started yet — `annotations.json` is still empty and
this is the very next step, in order:

1. Open `contact_sheet.jpg` to get a quick overview of all 35 frames at once.
2. Inspect each frame individually at full resolution (open the files
   in `frames/` directly, or run
   `python -m scripts.review_evaluation_frames data/evaluation/test7_v1`
   for a simple next/previous image browser — it never runs detection,
   never draws a box, and never guesses a label; it's a viewer only).
3. Mark every supported object you see in a frame (see "Canonical
   classes" and "Review requirement" above) — do not just annotate the
   object named in `selected_reason`, and never copy the burned-in
   overlay's label.
4. If a frame has no supported object at all, mark it **reviewed
   empty** (a `reviewed_frames` entry) instead of leaving it blank.
5. Track progress with `HUMAN_LABELING_CHECKLIST.md` (one checkbox
   block per frame) and use `annotation_working_template.json` as a
   scratch pad while labeling — it is **not** the canonical file and is
   never read by `evaluate_detector.py`; copy finished entries from it
   into the real `annotations.json` yourself.
6. Once every frame is reviewed, validate the completed
   `annotations.json`:
   ```bash
   python -m scripts.review_evaluation_frames data/evaluation/test7_v1 --validate
   ```
   This confirms all 35 frame files exist, every frame has been
   reviewed (annotated or marked reviewed-empty), and the file is
   schema-valid — it exits non-zero if anything is missing or invalid.
7. Only after that validation passes, run the YOLOv8s baseline and
   evaluate against your labels (below).

## After labeling

Once frames are annotated (and empty frames marked reviewed), run the
existing detector benchmark, then evaluate against your labels:

```bash
# 1. Run YOLOv8s (the current production detector) on the full clip,
#    producing a fresh detections.csv.
python -m scripts.benchmark_detector --source data/input/Test7.mp4 \
    --output-dir logs/detector_benchmarks/test7_yolov8s_baseline

# 2. Evaluate those predictions against your labeled annotations.json.
python -m scripts.evaluate_detector \
    --predictions logs/detector_benchmarks/test7_yolov8s_baseline/detections.csv \
    --annotations data/evaluation/test7_v1/annotations.json \
    --output-dir logs/detector_evaluations/test7_yolov8s_baseline
```

`logs/detector_evaluations/test7_yolov8s_baseline/evaluation_report.json`
will have real precision/recall/F1 per class, plus the `per_class_metrics.csv`/
`matched_detections.csv`/`errors.csv` breakdown files described in
`docs/DETECTOR_ANNOTATIONS.md`.
