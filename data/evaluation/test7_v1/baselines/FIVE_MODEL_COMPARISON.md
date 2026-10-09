# Test7 Five-Model Comparison — YOLOv8s / YOLOv8m / YOLO26s / YOLO11s / Faster R-CNN

Final cross-reference across all five Test7 detector benchmarks — four
pretrained YOLO variants plus one two-stage accuracy reference (Faster
R-CNN ResNet50-FPN-v2). All five ran on the identical 35 frames, the same
`annotations.json`, confidence 0.50, evaluator IoU 0.50, CPU.
**YOLOv8s remains production** (`config/settings.yaml` unchanged
throughout every benchmark in this series). Faster R-CNN is a diagnostic
reference only — it is not, and will not become, a production candidate
through this comparison.

Test7 is a screen-recorded clip with Atlas's own burned-in overlay pixels
— not pristine raw footage — with a small, single-pass-reviewed ground
truth (35 frames, 4 bicycle GT, 0 motorcycle GT). All numbers below are
same-dataset, single-CPU comparisons and should not be generalized to
other environments, GPU hardware, or future mobile/glasses hardware.

## Protected classes (car / bus / truck)

| class | metric | YOLOv8s | YOLOv8m | YOLO26s | YOLO11s | Faster R-CNN |
|---|---|---|---|---|---|---|
| car | GT/TP/FP/FN | 139/62/21/77 | 139/79/22/60 | 139/70/29/69 | 139/69/24/70 | 139/97/94/**42** |
| car | P/R/F1 | 0.747/0.446/0.559 | 0.782/0.568/0.658 | 0.707/0.504/0.588 | 0.742/0.496/0.595 | 0.508/**0.698**/0.588 |
| bus | GT/TP/FP/FN | 6/4/5/2 | 6/5/0/1 | 6/5/0/1 | 6/5/0/1 | 6/4/7/2 |
| bus | P/R/F1 | 0.444/0.667/0.533 | 1.000/0.833/0.909 | 1.000/0.833/0.909 | 1.000/0.833/0.909 | 0.364/0.667/0.471 |
| truck | GT/TP/FP/FN | 6/1/0/5 | 6/1/3/5 | 6/1/0/5 | 6/1/1/5 | 6/1/**11**/5 |
| truck | P/R/F1 | 1.000/0.167/0.286 | 0.250/0.167/0.200 | 1.000/0.167/0.286 | 0.500/0.167/0.250 | 0.083/0.167/0.111 |
| **protected combined** | P/R/F1 | 0.720/0.444/0.549 | 0.773/0.563/0.651 | 0.724/0.503/0.594 | 0.750/0.497/0.598 | 0.477/**0.675**/0.559 |

## Priority classes (bicycle / motorcycle)

| class | metric | YOLOv8s | YOLOv8m | YOLO26s | YOLO11s | **Faster R-CNN** |
|---|---|---|---|---|---|---|
| bicycle | GT/TP/FP/FN | 4/1/0/3 | 4/1/0/3 | 4/0/0/4 | 4/0/0/4 | **4/4/1/0** |
| bicycle | P/R/F1 | 1.000/0.250/0.400 | 1.000/0.250/0.400 | undef/0.000/undef | undef/0.000/undef | 0.800/**1.000**/0.889 |
| motorcycle | GT/FP | 0/0 | 0/2 | 0/0 | 0/1 | 0/1 |
| **priority combined** | P/R/F1 | 1.000/0.250/0.400 | 0.333/0.250/0.286 | undef/0.000/undef | 0.000/0.000/undef | 0.667/**1.000**/0.800 |

**Faster R-CNN is the only model to detect every ground-truth bicycle.**
No other model exceeds 1 of 4.

## Person

| metric | YOLOv8s | YOLOv8m | YOLO26s | YOLO11s | Faster R-CNN |
|---|---|---|---|---|---|
| GT/TP/FP/FN | 3/2/0/1 | 3/2/1/1 | 3/1/2/2 | 3/1/0/2 | 3/2/18/1 |
| P/R/F1 | 1.000/0.667/0.800 | 0.667/0.667/0.667 | 0.333/0.333/0.333 | 1.000/0.333/0.500 | 0.100/0.667/0.174 |
| difficult frame 1408 recovered? | no | no | no | no | **yes** |

## Runtime

| metric | YOLOv8s | YOLOv8m | YOLO26s | YOLO11s | Faster R-CNN |
|---|---|---|---|---|---|
| mean inference | 72.10 ms | 155.04 ms | 46.63 ms | 68.54 ms | **3754.95 ms** |
| median inference | 45.71 ms | 98.25 ms | 41.77 ms | 45.08 ms | 3778.82 ms |
| p95 inference | 58.42 ms | 275.31 ms | 71.35 ms | 70.74 ms | 4038.76 ms |
| effective FPS | 13.87 | 6.45 | 21.45 | 14.59 | 0.27 |
| real-time factor | 0.232 | 0.108 | 0.358 | 0.244 | 0.0044 |

Faster R-CNN is ~52× slower than production YOLOv8s on this CPU — expected
for a two-stage detector without GPU acceleration, and clearly not
production-viable here.

## Bicycle case-by-case (frames 1588 / 1648 / 1660 / 1678)

| frame | YOLOv8s | YOLOv8m | YOLO26s | YOLO11s | **Faster R-CNN** |
|---|---|---|---|---|---|
| 1588 | miss | wrong-class: `motorcycle` (IoU 0.648) | miss | miss | **TP** (IoU 0.864, conf 0.981) |
| 1648 | **TP** (IoU 0.607) | wrong-class: `motorcycle` (IoU 0.561) | miss | wrong-class: `motorcycle` (IoU 0.660) | **TP** (IoU 0.517, conf 0.972) |
| 1660 | miss | miss | miss | miss | **TP** (IoU 0.574, conf 0.832; also a duplicate wrong-class `motorcycle` FP nearby) |
| 1678 | miss | **TP** (IoU 0.806) | miss | miss | **TP** (IoU 0.624, conf 0.941) |

## The recurring ambiguous white commercial van (car GT, frames 3385/3475/3834)

Now mishandled by **all five** models tested, in five different ways:

| model | what happens |
|---|---|
| YOLOv8s | misclassified as `bus` |
| YOLOv8m | misclassified as `truck` |
| YOLO26s | missed entirely |
| YOLO11s | misclassified as `truck` |
| Faster R-CNN | misclassified as `truck` (frame 3834, confidence 0.854) |

Five for five — this object is genuinely visually ambiguous, independent
of model architecture or generation.

## What Faster R-CNN reveals

- **Bicycle recall headroom is real**: 4/4 vs. every YOLO model's 0-1/4,
  at high confidence and comfortable IoU margins — not a borderline
  result.
- **Car recall headroom is real**: FN drops to 42 (vs. YOLOv8s's 77,
  best YOLO result 60) — the two-stage region-proposal approach finds
  substantially more of the small/distant background cars YOLO misses.
- **Difficult-object recall headroom is real but partial**: recovers the
  difficult person YOLO always missed (frame 1408), but still completely
  misses all `heavy`-occlusion ground truth (0/3) and all 5 difficult R06
  pickups — headroom exists for *some* hard cases, not all of them.
- **The cost is severe, across-the-board precision loss**: car FP 94,
  truck FP 11 (for 1 TP), person FP 18, bus FP 7 — all worse than every
  YOLO model. Some of this may reflect genuine gaps in Test7's
  single-pass ground truth (see the frame-1498 bicycle note in
  `fasterrcnn_resnet50_fpn_v2_baseline_v1/BASELINE_REPORT.md`), not
  purely model error — undetermined without further review, and nothing
  was edited to resolve it.
- **Runtime is not remotely production-viable** on this CPU (0.27 FPS),
  though this says nothing about GPU-accelerated deployment.

## Decision

**ACCURACY_REFERENCE_USEFUL.** Faster R-CNN clearly reveals real,
material accuracy headroom on exactly the cases this whole search was
chasing: bicycle recall (0-1/4 → 4/4), car false negatives (best-of-five,
42 vs. 60-77), and at least one previously-unrecoverable difficult object
(the frame-1408 person). This meets the task's own PASS bar for
"accuracy reference useful" even though runtime is far too slow for
production use — that tradeoff is expected and explicitly accepted for a
diagnostic-only reference run.

**Production model remains YOLOv8s.** This result does not, by itself,
justify fine-tuning or architecture changes — it answers a narrower
question (does headroom exist?) with a clear "yes," which is useful input
for a *future* fine-tuning milestone (per `docs/
ATLAS_DETECTOR_DATASET_V1.md`), not a decision to act on today.
