# Test7 Four-Model Comparison — YOLOv8s vs YOLOv8m vs YOLO26s vs YOLO11s

Final cross-reference of all four Test7 baselines tested (the complete
off-the-shelf/pretrained detector search for Test7 v1 — see "after this
test" note at the bottom). Full detail in each model's own
`BASELINE_REPORT.md`. All four ran on the identical 35 frames,
`annotations.json`, confidence 0.50, evaluator IoU 0.50, CPU.
**YOLOv8s remains production** (`config/settings.yaml` unchanged throughout).

Test7 is a screen-recorded clip with Atlas's own burned-in overlay pixels —
not pristine raw footage. All numbers below are same-dataset comparisons on
one CPU and should not be generalized to other environments or hardware.

## Protected classes (car / bus / truck)

| class | metric | YOLOv8s | YOLOv8m | YOLO26s | YOLO11s |
|---|---|---|---|---|---|
| car | GT/TP/FP/FN | 139/62/21/77 | 139/79/22/60 | 139/70/29/69 | 139/69/24/70 |
| car | P/R/F1 | 0.747/0.446/0.559 | 0.782/0.568/0.658 | 0.707/0.504/0.588 | 0.742/0.496/0.595 |
| bus | GT/TP/FP/FN | 6/4/5/2 | 6/5/0/1 | 6/5/0/1 | 6/5/0/1 |
| bus | P/R/F1 | 0.444/0.667/0.533 | 1.000/0.833/0.909 | 1.000/0.833/0.909 | 1.000/0.833/0.909 |
| truck | GT/TP/FP/FN | 6/1/0/5 | 6/1/3/5 | 6/1/0/5 | 6/1/1/5 |
| truck | P/R/F1 | 1.000/0.167/0.286 | 0.250/0.167/0.200 | 1.000/0.167/0.286 | 0.500/0.167/0.250 |
| **protected combined** | P/R/F1 | 0.720/0.444/0.549 | **0.773/0.563/0.651** | 0.724/0.503/0.594 | 0.750/0.497/0.598 |

## Priority classes (bicycle / motorcycle)

| class | metric | YOLOv8s | YOLOv8m | YOLO26s | YOLO11s |
|---|---|---|---|---|---|
| bicycle | GT/TP/FP/FN | 4/1/0/3 | 4/1/0/3 | 4/0/0/4 | 4/0/0/4 |
| bicycle | P/R/F1 | 1.000/0.250/0.400 | 1.000/0.250/0.400 | undef/**0.000**/undef | undef/**0.000**/undef |
| motorcycle | GT/TP/FP/FN | 0/0/0/0 | 0/0/2/0 | 0/0/0/0 | 0/0/1/0 |
| **priority combined** | P/R/F1 | 1.000/0.250/0.400 | 0.333/0.250/0.286 | undef/0.000/undef | 0.000/0.000/undef |

## Person

| metric | YOLOv8s | YOLOv8m | YOLO26s | YOLO11s |
|---|---|---|---|---|
| GT/TP/FP/FN | 3/2/0/1 | 3/2/1/1 | 3/1/2/2 | 3/1/0/2 |
| P/R/F1 | 1.000/0.667/0.800 | 0.667/0.667/0.667 | 0.333/0.333/0.333 | 1.000/0.333/0.500 |

## Overall

| metric | YOLOv8s | YOLOv8m | YOLO26s | YOLO11s |
|---|---|---|---|---|
| micro P/R/F1 | 0.729/0.443/0.551 | **0.759/0.557/0.642** | 0.713/0.487/0.579 | 0.745/0.481/0.585 |
| macro P/R/F1 | 0.838/0.439/0.516 | 0.616/0.497/0.567 | 0.760/0.367/0.529 | 0.648/0.366/0.563 |

## Runtime

| metric | YOLOv8s | YOLOv8m | YOLO26s | YOLO11s |
|---|---|---|---|---|
| mean inference | 72.10 ms | 155.04 ms | 46.63 ms | 68.54 ms |
| median inference | 45.71 ms | 98.25 ms | **41.77 ms** | 45.08 ms |
| p95 inference | 58.42 ms | 275.31 ms | 71.35 ms | 70.74 ms |
| effective FPS | 13.87 | 6.45 | **21.45** | 14.59 |
| real-time factor | 0.232 | 0.108 | **0.358** | 0.244 |

## Bicycle case-by-case (frames 1588 / 1648 / 1660 / 1678)

| frame | YOLOv8s | YOLOv8m | YOLO26s | YOLO11s |
|---|---|---|---|---|
| 1588 | miss | wrong-class: `motorcycle` (IoU 0.648) | miss (no detections in frame) | miss (no detections in frame) |
| 1648 | **TP**: `bicycle` (IoU 0.607) | wrong-class: `motorcycle` (IoU 0.561) | miss | wrong-class: `motorcycle` (IoU 0.660) |
| 1660 | miss | miss | miss | miss |
| 1678 | miss | **TP**: `bicycle` (IoU 0.806) | miss | miss |

No model ever detects the rider correctly in more than 1 of 4 frames, and
never more than one frame at a time. YOLO26s and YOLO11s both fail to detect
the rider as anything (right or wrong class) in 3 of their 4 frames; YOLO11s's
one non-miss is a wrong-class `motorcycle` guess, same failure mode as
YOLOv8m showed at two frames.

## The recurring ambiguous white commercial van (car GT, frames 3385/3475/3834)

Every single tested model mishandles this one object, in a different way:

| model | what happens |
|---|---|
| YOLOv8s | misclassified as `bus` (frames 3834, 4014, 4104) |
| YOLOv8m | misclassified as `truck` (frames 3475, 3834) |
| YOLO26s | missed entirely (no detection at all) |
| YOLO11s | misclassified as `truck` (frame 3385) |

This is strong, convergent evidence that the object itself is genuinely
visually ambiguous (consistent with the human reviewer's own car-vs-truck
uncertainty during ground-truth labeling), not a model-specific bug.

## Verdict summary

- **YOLOv8m** vs YOLOv8s: real car/bus gains, flat bicycle, small-sample
  truck-precision regression, ~2× slower → **INCONCLUSIVE**.
- **YOLO26s** vs YOLOv8s/m: fastest, avoids YOLOv8m's truck FPs, but bicycle
  recall regresses to zero and car FP is worst of all → **FAIL**.
- **YOLO11s** vs YOLOv8s/m/26s: smallest/fastest-competitive model, matches
  or slightly beats YOLOv8s's own speed, modest protected-combined recall
  gain, but bicycle recall also regresses to zero and person recall drops
  from 2/3 to 1/3 → **FAIL**.

**YOLOv8s remains the recommended production model.** All four off-the-shelf
pretrained candidates tested for Test7 v1 either fail to clearly beat it or
carry a regression on a priority/protected metric that isn't offset by a
strong enough gain elsewhere. Per the task's own "after this test" guidance,
no further pretrained model should be tested automatically — the next
detector milestone (not started in this task) should focus on expanding
clean labeled raw footage and dataset coverage for
bicycle/motorcycle/scooter/e-bike before considering custom fine-tuning.
