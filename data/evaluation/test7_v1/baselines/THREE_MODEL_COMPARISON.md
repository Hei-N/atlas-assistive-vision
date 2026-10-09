# Test7 Three-Model Comparison — YOLOv8s vs YOLOv8m vs YOLO26s

Concise cross-reference of all three frozen Test7 baselines. Full detail in
each model's own `BASELINE_REPORT.md`. All three ran on the identical 35
frames, `annotations.json`, confidence 0.50, evaluator IoU 0.50, CPU.
**YOLOv8s remains production** (`config/settings.yaml` unchanged throughout).

Test7 is a screen-recorded clip with Atlas's own burned-in overlay pixels —
not pristine raw footage. All numbers below are same-dataset comparisons
only, on this one CPU, and should not be generalized.

## Protected classes (car / bus / truck)

| class | metric | YOLOv8s | YOLOv8m | YOLO26s |
|---|---|---|---|---|
| car | GT / TP / FP / FN | 139 / 62 / 21 / 77 | 139 / 79 / 22 / 60 | 139 / 70 / 29 / 69 |
| car | precision / recall / F1 | 0.747 / 0.446 / 0.559 | 0.782 / 0.568 / 0.658 | 0.707 / 0.504 / 0.588 |
| bus | GT / TP / FP / FN | 6 / 4 / 5 / 2 | 6 / 5 / 0 / 1 | 6 / 5 / 0 / 1 |
| bus | precision / recall / F1 | 0.444 / 0.667 / 0.533 | 1.000 / 0.833 / 0.909 | 1.000 / 0.833 / 0.909 |
| truck | GT / TP / FP / FN | 6 / 1 / 0 / 5 | 6 / 1 / 3 / 5 | 6 / 1 / 0 / 5 |
| truck | precision / recall / F1 | 1.000 / 0.167 / 0.286 | 0.250 / 0.167 / 0.200 | 1.000 / 0.167 / 0.286 |
| **protected combined** | precision / recall / F1 | 0.720 / 0.444 / 0.549 | **0.773 / 0.563 / 0.651** | 0.724 / 0.503 / 0.594 |

## Priority classes (bicycle / motorcycle)

| class | metric | YOLOv8s | YOLOv8m | YOLO26s |
|---|---|---|---|---|
| bicycle | GT / TP / FP / FN | 4 / 1 / 0 / 3 | 4 / 1 / 0 / 3 | 4 / 0 / 0 / **4** |
| bicycle | precision / recall / F1 | 1.000 / 0.250 / 0.400 | 1.000 / 0.250 / 0.400 | undefined / **0.000** / undefined |
| motorcycle | GT / TP / FP / FN | 0 / 0 / 0 / 0 | 0 / 0 / **2** / 0 | 0 / 0 / 0 / 0 |
| motorcycle | recall | undefined (0 GT) | undefined (0 GT) | undefined (0 GT) |
| **priority combined** | precision / recall / F1 | 1.000 / 0.250 / 0.400 | 0.333 / 0.250 / 0.286 | undefined / **0.000** / undefined |

## Person

| metric | YOLOv8s | YOLOv8m | YOLO26s |
|---|---|---|---|
| GT / TP / FP / FN | 3 / 2 / 0 / 1 | 3 / 2 / 1 / 1 | 3 / 1 / 2 / 2 |
| precision / recall / F1 | 1.000 / 0.667 / 0.800 | 0.667 / 0.667 / 0.667 | 0.333 / 0.333 / 0.333 |

## Overall

| metric | YOLOv8s | YOLOv8m | YOLO26s |
|---|---|---|---|
| micro P / R / F1 | 0.729 / 0.443 / 0.551 | **0.759 / 0.557 / 0.642** | 0.713 / 0.487 / 0.579 |
| macro P / R / F1 | 0.838 / 0.439 / 0.516 | 0.616 / 0.497 / 0.567 | 0.760 / 0.367 / 0.529 |

## Runtime

| metric | YOLOv8s | YOLOv8m | YOLO26s |
|---|---|---|---|
| mean inference | 72.10 ms | 155.04 ms | **46.63 ms** |
| median inference | 45.71 ms | 98.25 ms | **41.77 ms** |
| p95 inference | 58.42 ms | 275.31 ms | 71.35 ms |
| effective FPS | 13.87 | 6.45 | **21.45** |
| real-time factor | 0.232 | 0.108 | **0.358** |

## Bicycle case-by-case (frames 1588 / 1648 / 1660 / 1678)

| frame | YOLOv8s | YOLOv8m | YOLO26s |
|---|---|---|---|
| 1588 | miss (no detection) | wrong-class: `motorcycle` (IoU 0.648) | miss (no detection at all in frame) |
| 1648 | **TP**: `bicycle`, IoU 0.607 | wrong-class: `motorcycle` (IoU 0.561) | miss |
| 1660 | miss | miss | miss |
| 1678 | miss | **TP**: `bicycle`, IoU 0.806 | miss |

Across all three models, the rider is correctly identified as `bicycle` in
at most 1 of 4 frames at any time — never more than one simultaneously. YOLO26s
is the only model that never even attempts a nearby prediction (right or
wrong class) in any of the 4 frames.

## Verdict summary

- **YOLOv8m** vs YOLOv8s: real gains on car/bus, flat bicycle, a small-sample
  truck-precision regression, ~2× slower → **INCONCLUSIVE** (prior task).
- **YOLO26s** vs YOLOv8s and YOLOv8m: fastest of the three by a clear margin
  and avoids YOLOv8m's truck false positives, but **bicycle recall
  regresses to zero** (the exact priority-class objective this comparison
  was chasing) and car false positives are the worst of the three → **FAIL**
  (this task; see below).

**YOLOv8s remains the recommended production model for now.**
