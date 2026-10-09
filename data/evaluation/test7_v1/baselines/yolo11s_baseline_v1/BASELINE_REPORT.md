# Test7 YOLO11s Candidate Baseline v1 — Final Off-the-Shelf Comparison

Candidate comparison run only — **YOLO11s is not production**, and this task
did not change `config/settings.yaml`. This is the fourth and final
pretrained/off-the-shelf detector tested for Test7 v1. Reference baselines:
`../yolov8s_baseline_v1/` (frozen production), `../yolov8m_baseline_v1/`,
`../yolo26s_baseline_v1/`.

## Important limitation — same as the other Test7 baselines

Test7 is a screen-recorded evaluation clip containing Atlas's own burned-in
overlay pixels, not pristine raw camera footage. Ground truth came from
independent human/AI-assisted review, never the overlay. These are
same-dataset comparison numbers only, on this one CPU — not generalized to
other environments or to future mobile/glasses hardware.

## Compatibility confirmation

`yolo11s.pt` is a recognized downloadable asset in the installed Ultralytics
`8.4.104` environment. Unlike YOLO26 (`end2end: True`, NMS-free head),
YOLO11's config carries no `end2end` flag — it uses traditional NMS, the same
output path as `yolov8s`/`yolov8m`. Confirmed compatible with `ObjectDetector`
with **zero wrapper or production-code changes**. Params: 9.46M / 21.7 GFLOPs
— the smallest of the four models tested (vs. yolov8s 11.17M/28.8, yolov8m
25.90M/79.3, yolo26s 10.01M/22.8).

## Run configuration (identical to the other three baselines except the model)

| | |
|---|---|
| Model | `yolo11s.pt` (auto-downloaded via Ultralytics, official release; **not** production) |
| Confidence threshold | `0.50` (same as production, not tuned) |
| Evaluation matching IoU | `0.50` |
| Device | `cpu` |
| Input resolution | `1920×1080` (native) |
| Evaluated frames | the same exact 35 frame indices |

## Candidate protected-class metrics

| class | GT | pred | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| car | 139 | 93 | 69 | 24 | 70 | 0.742 | 0.496 | 0.595 |
| bus | 6 | 5 | 5 | 0 | 1 | 1.000 | 0.833 | 0.909 |
| truck | 6 | 2 | 1 | 1 | 5 | 0.500 | 0.167 | 0.250 |
| **protected combined** | **151** | **100** | **75** | **25** | **76** | **0.750** | **0.497** | **0.598** |

## Candidate priority-class metrics

| class | GT | pred | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| bicycle | 4 | 0 | 0 | 0 | 4 | undefined | **0.000** | undefined |
| motorcycle | 0 | 1 | 0 | 1 | 0 | 0.000 | undefined | undefined |
| **priority combined** | 4 | 1 | 0 | 1 | 4 | 0.000 | **0.000** | undefined |

**Bicycle recall is zero** — same complete failure as YOLO26s, worse than
YOLOv8s/YOLOv8m's 1/4. One new motorcycle false positive appeared (see
bicycle case-by-case below — it's the same rider, wrong class).

## Person

| class | GT | pred | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| person | 3 | 1 | 1 | 0 | 2 | 1.000 | 0.333 | 0.500 |

Only the non-`difficult` person (frame 1498, IoU 0.555) was matched; both
`difficult=true` people (frames 1408, 2486) were missed. No new person false
positives (unlike YOLO26s's 2).

## Bicycle case-by-case (frames 1588 / 1648 / 1660 / 1678)

| frame | prediction near rider | confidence | IoU vs bicycle GT | result |
|---|---|---|---|---|
| 1588 | none — zero detections in the entire frame | — | — | **miss** |
| 1648 | `motorcycle` at [280,682,353,750] | 0.522 | 0.660 | **wrong-class** (bicycle FN + motorcycle FP) |
| 1660 | none (only `bus`) | — | — | **miss** |
| 1678 | none (only `bus`) | — | — | **miss** |

0 of 4 bicycle frames correctly detected — below the "at least 2/4" target,
and below all three prior models' 1/4 (YOLOv8s, YOLOv8m) result.

## Comparison with all three prior models

- **Car FN**: YOLOv8s 77 → YOLOv8m 60 → YOLO26s 69 → **YOLO11s 70**. Improved
  vs. YOLOv8s (−7) but well short of YOLOv8m's gain, essentially tied with
  YOLO26s.
- **Bus**: identical to YOLOv8m and YOLO26s — 5/6 (0.833 recall), 0 FP.
  Consistently the strongest class across all three newer candidates.
- **Truck**: still only 1/6 (frame 4104, the same non-difficult "ID57"
  pickup every model catches, best IoU yet at 0.697). One new truck FP
  appeared at **frame 3385**, IoU 0.692 against the ground-truth *car* at
  that location — this is, again, the recurring ambiguous white commercial
  box/cargo van (review group R02): YOLOv8s called it `bus`, YOLOv8m called
  it `truck` (frames 3475/3834), YOLO26s missed it outright, and now YOLO11s
  calls it `truck` too (frame 3385). A fourth model, a fourth wrong guess on
  the same genuinely hard object — not a new failure mode. None of the 5
  `difficult=true` R06 edge-cropped pickups were detected by any model
  tested so far, including this one.
- **Person**: regressed from YOLOv8s's 2/3 to 1/3 — matches YOLO26s's
  regression pattern (both newer/smaller models lose one of the two
  `difficult=true` misses that YOLOv8s and YOLOv8m still caught).
- **New major false positives**: 1 motorcycle (bicycle rider misclassified),
  1 truck (the recurring van, misclassified) — both explainable by
  already-known-hard objects, not clean new failures on easy footage.

## Runtime

| metric | YOLOv8s | YOLOv8m | YOLO26s | YOLO11s |
|---|---|---|---|---|
| mean inference | 72.10 ms | 155.04 ms | 46.63 ms | **68.54 ms** |
| median inference | 45.71 ms | 98.25 ms | 41.77 ms | 45.08 ms |
| p95 inference | 58.42 ms | 275.31 ms | 71.35 ms | 70.74 ms |
| effective FPS | 13.87 | 6.45 | 21.45 | **14.59** |
| real-time factor | 0.232 | 0.108 | 0.358 | **0.244** |

YOLO11s is the **only candidate that matches or slightly beats YOLOv8s's own
speed** (mean 68.5 ms vs. 72.1 ms, effective FPS 14.59 vs. 13.87) — meeting
the runtime part of the task's "strong outcome" bar. It is still well short
of YOLO26s's speed and far from real-time (24% of the 59.9 FPS source). Not
generalized to mobile/glasses hardware.

## Artifacts in this directory

- `run_metadata.json`, `detections.csv`, `summary.json`,
  `evaluation_report.json`, `per_class_metrics.csv`, `matched_detections.csv`,
  `errors.csv`, `BASELINE_REPORT.md` (this file)
