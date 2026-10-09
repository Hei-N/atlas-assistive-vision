# Test7 Faster R-CNN (ResNet50-FPN-v2) Accuracy-Reference Benchmark

**Not a production candidate.** This is a diagnostic-only run of a stronger,
slower two-stage detector to answer one question: is YOLO leaving real
accuracy on the table on Test7's hardest cases (bicycles, small/occluded
objects)? Faster R-CNN is never routed through `src/object_detector.py`,
never called by `main.py`, and `config/settings.yaml` is untouched.
Reference: `../yolov8s_baseline_v1/` (frozen production),
`../yolov8m_baseline_v1/`, `../yolo26s_baseline_v1/`, `../yolo11s_baseline_v1/`.

## Important limitation — same as every other Test7 baseline, plus one more

Test7 is a screen-recorded clip with Atlas's own burned-in overlay pixels,
35 frames, 4 bicycle GT instances, 0 motorcycle GT instances. This
benchmark is diagnostic, not final proof of anything at scale.
**Additional limitation specific to this run**: Test7's ground truth was
built from a single-pass AI-assisted human review (see
`../../HUMAN_REVIEW_GROUPS.md`/`HUMAN_DECISIONS.md`), not an exhaustive,
multiply-verified label set. Faster R-CNN's much higher raw sensitivity
(244 raw detections vs. YOLO's 84-117) surfaced some predictions that may
be genuine missed ground-truth objects rather than true false positives —
flagged explicitly below where found, **never edited into the ground
truth** (the standing rule: never change annotations after seeing detector
output).

## Model / weights

| | |
|---|---|
| Model | `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2` |
| Weights | `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1` (official pretrained COCO weights, unmodified) |
| Weights file SHA-256 | `dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf` |
| Parameters | 43,712,278 |
| torch / torchvision | 2.13.0 / 0.28.0 |
| Confidence threshold | `0.50` (same as every YOLO baseline, not tuned) |
| Evaluation matching IoU | `0.50` |
| Device | `cpu` |
| Class mapping | torchvision's own COCO-91 indices: `1→person, 2→bicycle, 3→car, 4→motorcycle, 6→bus, 8→truck` (all other COCO categories discarded) |
| Evaluated frames | the same exact 35 frame indices, read directly from the already-extracted JPGs |

## Protected-class metrics

| class | GT | pred | TP | FP | FN | precision | recall | F1 | mean IoU |
|---|---|---|---|---|---|---|---|---|---|
| car | 139 | 195 | 97 | 94 | 42 | 0.508 | 0.698 | 0.588 | 0.713 |
| bus | 6 | 11 | 4 | 7 | 2 | 0.364 | 0.667 | 0.471 | 0.692 |
| truck | 6 | 12 | 1 | 11 | 5 | 0.083 | 0.167 | 0.111 | 0.736 |
| **protected combined** | **151** | **218** | **102** | **112** | **49** | **0.477** | **0.675** | **0.559** | **0.712** |

## Priority-class metrics

| class | GT | pred | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| bicycle | 4 | 5 | **4** | 1 | 0 | 0.800 | **1.000** | 0.889 |
| motorcycle | 0 | 1 | 0 | 1 | 0 | 0.000 | undefined | undefined |
| **priority combined** | 4 | 6 | 4 | 2 | 0 | 0.667 | **1.000** | 0.800 |

**Bicycle recall is 4/4 — every single ground-truth bicycle frame is
correctly detected.** This is the standout result of the entire Test7
detector search (YOLOv8s/YOLOv8m best: 1/4; YOLO26s/YOLO11s: 0/4).

## Person

| class | GT | pred | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| person | 3 | 20 | 2 | 18 | 1 | 0.100 | 0.667 | 0.174 |

Recovers the `difficult=true` person at **frame 1408** (IoU 0.697,
confidence 0.99) — missed by every YOLO model tested. Still misses the
other `difficult=true` person (frame 2486). 18 false positives, mostly
small (10–40px) boxes scattered across many frames — some are plausibly
genuine spurious detections, others may be small background pedestrians
Test7's single-pass ground truth never labeled (not verified either way;
not edited).

## Bicycle case-by-case (frames 1588 / 1648 / 1660 / 1678) — critical check

| frame | prediction at rider | confidence | IoU vs bicycle GT | result |
|---|---|---|---|---|
| 1588 | `bicycle` at [400,687,474,738] | 0.981 | 0.864 | **TP** |
| 1648 | `bicycle` at [282,692,350,750] | 0.972 | 0.517 | **TP** |
| 1660 | `bicycle` at [256,694,324,750] (+ a duplicate `motorcycle` at nearly the same box, conf 0.618, unmatched → priority-class FP) | 0.832 | 0.574 | **TP** (bicycle) |
| 1678 | `bicycle` at [209,692,279,756] | 0.941 | 0.624 | **TP** |

**4 of 4 — every frame correctly detected as `bicycle`, all with high
confidence (0.83–0.98) and IoU comfortably above the 0.50 match
threshold.** No model in this search has come close to this result.

One extra note: a 5th `bicycle` prediction appears at **frame 1498**
(confidence 0.954, box [544,693,614,736]) — not one of the 4 official GT
frames, and not matched to any ground truth, so it's scored as a false
positive. This may be a genuine bicycle in the scene that Test7's
single-pass ground truth review didn't label — flagged for awareness,
**not verified, not edited into the ground truth**.

## Comparison with all four YOLO models

- **Car FN**: YOLOv8s 77 → YOLOv8m 60 → YOLO26s 69 → YOLO11s 70 →
  **Faster R-CNN 42** — the best (lowest) of all five models by a wide
  margin, at the cost of the worst car precision (94 FP vs. 21-29 for the
  YOLO models).
- **Truck**: still only 1/6 (the same non-difficult "ID57" pickup every
  model catches, now with the best IoU yet, 0.736). **None of the 5
  `difficult=true` R06 edge-cropped pickups were recovered** — this
  specific hard case resists every model tested so far, two-stage
  included. Truck precision is the worst of all five models (11 FP for a
  single TP) — several of these FPs are large boxes over the left-edge
  region containing the R06 pickup / R09 SUV / R05 minivan cluster,
  suggesting Faster R-CNN sometimes merges adjacent vehicles in that
  crowded region into one oversized "truck" box rather than resolving
  them individually.
- **Bus/van confusion**: not resolved — the recurring ambiguous white
  commercial van (frames 3385/3475/3834) is misclassified as `truck` here
  too (frame 3834, confidence 0.854, box overlapping the van's ground
  -truth region at ~0.65-0.80 IoU with the *car* GT). This is now the
  **fifth** model (after YOLOv8s→bus, YOLOv8m→truck, YOLO26s→missed,
  YOLO11s→truck) to mishandle this exact object, in a fifth different
  way — very strong convergent evidence the object itself is genuinely
  ambiguous, not a model-specific bug.
- **Person**: recovers the `difficult=true` frame-1408 person that every
  YOLO model missed (see above) — a genuine, meaningful difficult-object
  win, at a steep precision cost (18 FP vs. 0-2 for YOLO models).

## Difficult / occluded / small-object breakdown

Reused directly from `evaluation_report.json`'s existing breakdowns (same
mechanism as every other Test7 baseline) — sample sizes are small, no
statistical certainty is claimed:

| dimension | bucket | TP | FN | recall | mean IoU |
|---|---|---|---|---|---|
| object_size | small | 83 | 46 | 0.643 | 0.697 |
| object_size | medium | 25 | 4 | 0.862 | 0.746 |
| object_size | large | 0 | 0 | undefined (no large GT objects) | — |
| visibility | clear | 90 | 32 | 0.738 | 0.718 |
| visibility | reduced | 15 | 11 | 0.577 | 0.665 |
| visibility | poor | 3 | 7 | 0.300 | 0.623 |
| occlusion | none | 85 | 25 | 0.773 | 0.716 |
| occlusion | partial | 23 | 22 | 0.511 | 0.678 |
| occlusion | heavy | 0 | 3 | 0.000 | undefined |

Every recall figure here is higher than the corresponding YOLOv8s baseline
bucket where a same-bucket comparison is meaningful (e.g. YOLOv8s overall
small-object recall 0.364 vs. Faster R-CNN's 0.643) — but heavy occlusion
is still a complete miss (0/3) for Faster R-CNN too, same as every YOLO
model. Two-stage detection helps with size/partial-occlusion, not with
near-total occlusion.

## Runtime

| metric | value |
|---|---|
| Model load time | 2.78 s |
| Mean inference time | 3754.9 ms |
| Median inference time | 3778.8 ms |
| p95 inference time | 4038.8 ms |
| Effective FPS | 0.266 |
| Real-time factor | 0.0044 |
| Peak memory (RSS) | ~1.83 GB |
| Device | CPU |

| | YOLOv8s | YOLOv8m | YOLO26s | YOLO11s | **Faster R-CNN** |
|---|---|---|---|---|---|
| mean inference | 72.1 ms | 155.0 ms | 46.6 ms | 68.5 ms | **3754.9 ms** |
| effective FPS | 13.87 | 6.45 | 21.45 | 14.59 | **0.27** |
| real-time factor | 0.232 | 0.108 | 0.358 | 0.244 | **0.0044** |

**Faster R-CNN is roughly 52× slower than YOLOv8s** on this CPU — fully
expected for a two-stage detector with no GPU, and not remotely
production-viable here. This CPU-only result must not be generalized to
GPU-equipped hardware, where the accuracy/speed gap between one-stage and
two-stage detectors narrows substantially — nor to future mobile/glasses
hardware.

## Artifacts in this directory

- `run_metadata.json`, `detections.csv`, `summary.json`,
  `evaluation_report.json`, `per_class_metrics.csv`, `matched_detections.csv`,
  `errors.csv`, `BASELINE_REPORT.md` (this file)
