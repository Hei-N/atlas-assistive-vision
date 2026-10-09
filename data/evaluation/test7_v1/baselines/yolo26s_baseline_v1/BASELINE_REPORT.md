# Test7 YOLO26s Candidate Baseline v1 — Comparison Measurement

Candidate comparison run only — **YOLO26s is not production**, and this task
did not change `config/settings.yaml`. Selected because it was fully
supported (config + downloadable weights) by the installed Ultralytics
version without any `ObjectDetector` wrapper changes, per the preferred
selection order (yolo26s over yolo11s). Reference baselines:
`../yolov8s_baseline_v1/` (frozen production) and `../yolov8m_baseline_v1/`
(prior candidate, larger/slower, no bicycle gain).

## Important limitation — same as the other Test7 baselines

Test7 is a screen-recorded evaluation clip containing Atlas's own burned-in
overlay pixels, not pristine raw camera footage. Ground truth came from
independent human/AI-assisted review, never the overlay. These are
same-dataset comparison numbers only — not a general claim about YOLO26s
performance elsewhere, and CPU-only timing here should not be generalized to
phone/Meta-glasses hardware.

## Run configuration (identical to the other two baselines except the model)

| | |
|---|---|
| Model | `yolo26s.pt` (auto-downloaded via Ultralytics, official release; **not** production) |
| Model params / GFLOPs | ~10.01M params, 22.8 GFLOPs (vs. yolov8s: 11.17M / 28.8 GFLOPs) |
| Confidence threshold | `0.50` (same as production, not tuned) |
| Evaluation matching IoU | `0.50` |
| Device | `cpu` (same machine) |
| Input resolution | `1920×1080` (native) |
| Evaluated frames | the same exact 35 frame indices |

## Candidate protected-class metrics

| class | GT | pred | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| car | 139 | 99 | 70 | 29 | 69 | 0.707 | 0.504 | 0.588 |
| bus | 6 | 5 | 5 | 0 | 1 | 1.000 | 0.833 | 0.909 |
| truck | 6 | 1 | 1 | 0 | 5 | 1.000 | 0.167 | 0.286 |
| **protected combined** | **151** | **105** | **76** | **29** | **75** | **0.724** | **0.503** | **0.594** |

## Candidate priority-class metrics

| class | GT | pred | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| bicycle | 4 | 0 | 0 | 0 | 4 | undefined | **0.000** | undefined |
| motorcycle | 0 | 0 | 0 | 0 | 0 | undefined | undefined | undefined |
| **priority combined** | 4 | 0 | 0 | 0 | 4 | undefined | **0.000** | undefined |

**Bicycle recall regressed to zero** — the candidate produced no
bicycle/motorcycle prediction anywhere near the rider in any of the 4 ground
-truth frames (see the bicycle case-by-case section below). Motorcycle still
has 0 ground truth and is correctly reported as undefined (no motorcycle
predictions this run either — unlike YOLOv8m's 2 false positives).

## Person

| class | GT | pred | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| person | 3 | 3 | 1 | 2 | 2 | 0.333 | 0.333 | 0.333 |

Only the non-`difficult` person (frame 1498) was matched; both `difficult=true`
people (frames 1408, 2486) were missed, and 2 new false positives appeared
(a near-duplicate box at 1498 next to the correct match, and an unrelated
spurious detection at 2486).

## Bicycle case-by-case (frames 1588 / 1648 / 1660 / 1678)

| frame | prediction near rider | confidence | IoU vs bicycle GT | result |
|---|---|---|---|---|
| 1588 | **none** — zero detections in the entire frame | — | — | **miss** |
| 1648 | none (only `bus` at [302,572,480,686] and a distant background `car`) | — | — | **miss** |
| 1660 | none (only `bus` at [288,573,468,687]) | — | — | **miss** |
| 1678 | none (only `bus` at [260,579,442,691]) | — | — | **miss** |

Every single one of the 4 bicycle frames is a **plain miss** — not a
wrong-class prediction (unlike YOLOv8m's motorcycle confusion at 1588/1648).
The rider is simply not detected as anything at any confidence level.

## Comparison with YOLOv8m (secondary)

- **Car gains recovered?** Partially — recall improved over YOLOv8s (0.446 →
  0.504) but did **not** reach YOLOv8m's 0.568, and false positives are worse
  than both other models (car FP: v8s 21, v8m 22, **y26s 29**).
- **Bicycle recall beyond 1/4?** No — it went to **0/4**, worse than both
  YOLOv8s and YOLOv8m (both 1/4).
- **Avoids YOLOv8m's new truck false positives?** Yes — truck FP is 0 (same
  as YOLOv8s), versus YOLOv8m's 3.
- **Bus/van confusion**: improved in one sense, unresolved in another — bus
  FP is 0 (matches YOLOv8m, both better than YOLOv8s's 5) and truck FP is 0
  (better than YOLOv8m's 3). But the recurring ambiguous white commercial van
  (frames 3385, 3475, 3834) is not being *misclassified* as bus/truck anymore
  — it's simply **not detected at all** by this candidate (contributing
  directly to the car FN count). Trading a wrong-class prediction for a
  missed detection is not a clear win.
- **Materially faster than YOLOv8m?** Yes, substantially — mean inference
  46.6 ms vs. YOLOv8m's 155.0 ms (roughly 3.3× faster), and even faster than
  YOLOv8s (72.1 ms).

## Runtime

| metric | YOLOv8s | YOLOv8m | YOLO26s |
|---|---|---|---|
| mean inference | 72.10 ms | 155.04 ms | **46.63 ms** |
| median inference | 45.71 ms | 98.25 ms | **41.77 ms** |
| p95 inference | 58.42 ms | 275.31 ms | 71.35 ms |
| effective FPS | 13.87 | 6.45 | **21.45** |
| real-time factor | 0.232 | 0.108 | **0.358** |

YOLO26s is clearly the fastest of the three on this CPU — best mean, median,
and real-time factor. Still well short of real-time (0.358 = ~36% of the
59.9 FPS source), and this CPU-only result must not be generalized to other
hardware.

## Artifacts in this directory

- `run_metadata.json`, `detections.csv`, `summary.json`,
  `evaluation_report.json`, `per_class_metrics.csv`, `matched_detections.csv`,
  `errors.csv`, `BASELINE_REPORT.md` (this file)
