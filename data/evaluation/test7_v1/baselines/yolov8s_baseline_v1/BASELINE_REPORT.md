# Test7 YOLOv8s Baseline v1 — Frozen Measurement

This is the **first real precision/recall/F1 baseline** for Atlas's
production detector against human-reviewed Test7 ground truth. It is a
**measurement, not a tuning pass** — nothing was changed after seeing
these results.

## Important limitation — read before interpreting any number below

Test7 is a **screen-recorded evaluation clip**: it is filmed footage of
a laptop screen showing Atlas running live, including Atlas's own
previous detection overlay boxes/labels burned into the recorded
pixels. It is **not a pristine raw-camera dataset**. The overlay was
never used as ground truth (all 181 ground-truth annotations were
built from independent human/AI-assisted visual review — see
`../../HUMAN_REVIEW_GROUPS.md` and `../../HUMAN_DECISIONS.md`), and the
detector processed the pixels exactly as recorded because that is the
only Test7 source available. Treat every number in this report as an
**initial baseline only**; a future comparison should also use cleaner
raw street footage, not another screen recording.

## Run configuration (frozen — do not change without a new baseline)

| | |
|---|---|
| Model | `yolov8s.pt` (production, `config/settings.yaml`, unmodified) |
| Confidence threshold | `0.50` (production default, unmodified) |
| NMS IoU threshold | Ultralytics internal default (unconfigured/unexposed by `src/object_detector.py`) |
| Evaluation matching IoU | `0.50` (evaluator default, unmodified) |
| Device | `cpu` |
| Input resolution | `1920×1080` (native, never resized) |
| Source | `data/input/Test7.mp4` |
| Evaluated frames | exactly the 35 frame indices in `frame_manifest.csv` (not the full 4160-frame video) |
| Model file SHA-256 | `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a` |

## Protected-class baseline (non-regression reference)

| class | GT | pred | TP | FP | FN | precision | recall | F1 | mean IoU | mean conf. TP | mean conf. FP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| car | 139 | 84 | 62 | 21 | 77 | 0.747 | 0.446 | 0.559 | 0.704 | 0.732 | 0.688 |
| bus | 6 | 9 | 4 | 5 | 2 | 0.444 | 0.667 | 0.533 | 0.673 | 0.860 | 0.642 |
| truck | 6 | 1 | 1 | 0 | 5 | 1.000 | 0.167 | 0.286 | 0.667 | 0.609 | undefined (0 FP) |
| **protected combined** | **151** | **94** | **67** | **26** | **84** | **0.720** | **0.444** | **0.549** | **0.702** | **0.738** | **0.679** |

## Priority-class baseline (non-regression reference)

| class | GT | pred | TP | FP | FN | precision | recall | F1 | mean IoU |
|---|---|---|---|---|---|---|---|---|---|
| bicycle | 4 | 1 | 1 | 0 | 3 | 1.000 | 0.250 | 0.400 | 0.607 |
| motorcycle | **0** | 0 | 0 | 0 | 0 | undefined | **undefined** | undefined | undefined |
| **priority combined** | 4 | 1 | 1 | 0 | 3 | 1.000 | 0.250 | 0.400 | 0.607 |

**Motorcycle has zero ground-truth instances in Test7.** Its recall/precision/F1
are reported as `null`/undefined by the evaluator — this is **not** a
successful 0-for-0 detection and must never be read as "100% motorcycle
recall." It is inconclusive due to zero samples, full stop.

## Person

| class | GT | pred | TP | FP | FN | precision | recall | F1 | mean IoU |
|---|---|---|---|---|---|---|---|---|---|
| person | 3 | 2 | 2 | 0 | 1 | 1.000 | 0.667 | 0.800 | 0.615 |

## Overall

- **Micro**: precision 0.729, recall 0.443, F1 0.551
- **Macro**: precision 0.838, recall 0.439, F1 0.516 (5 classes used, motorcycle skipped — 0 samples)

## Ignored / difficult handling

- 23 ground-truth annotations are `ignore=true` (the ambiguous tree-occluded/overlapping
  background vehicle clusters from review groups R07, R08, and singletons S004/S006/S012/S027/S028)
  — all excluded from the scored ground-truth pool as intended (`car` GT count is
  162 total minus these 23 = 139 scored, matching the report exactly). One prediction
  overlapped an ignored region and was itself excluded from FP counting (`ignored_prediction`
  in `errors.csv`).
- 72 annotations are `difficult=true` — scored identically to normal ground truth (not
  excluded), per the evaluator's existing, unmodified policy. This was not changed for
  this run.

## Error review

- **Largest false-negative class**: car (77 FN) — driven mostly by small/distant
  background vehicles across the tree-occluded park scene and the far-side street cluster.
- **Largest false-positive class**: car (21 FP), followed by bus (5 FP).
- **Small-object failure pattern**: recall drops sharply by ground-truth object size —
  `small` objects: recall 0.364 (47 TP / 129 scored); `medium` objects: recall 0.793
  (23 TP / 29 scored); no `large` objects in this ground truth. Small/distant background
  vehicles are the dominant miss category.
- **Occlusion failure pattern**: recall falls steeply with occlusion —
  `none`: 0.509, `partial`: 0.311, `heavy`: 0.0 (0 TP / 3 scored). Every heavily-occluded
  ground-truth object was missed.
- **Class-confusion pattern found**: at frames 3834, 4014, and 4104, YOLO predicted `bus`
  where ground truth has `car` (IoU 0.51–0.65, just below/near the match threshold in
  one case) — all three are the same recurring large white commercial box/cargo van
  (review group R02) that a human reviewer also judged ambiguous enough to require a
  car-vs-truck decision. YOLO's independent bus-shaped confusion on this exact object
  corroborates that it is a genuinely visually ambiguous vehicle, not a labeling error.
- **Bicycle-specific misses**: 3 of 4 ground-truth bicycles missed (frames 1588, 1660,
  1678); the one true positive is at frame 1648 (IoU 0.607, confidence 0.535, low
  confidence near the production threshold of 0.50).
- **Bus/truck-specific misses**: bus recall is comparatively strong (0.667, 4/6) but with
  5 false positives (many likely the same car-vs-bus confusion above); truck recall is
  poor (0.167, 1/6) — the one true positive is the untouched high-confidence pickup at
  frame 4104 ("ID57"); all 5 R06 left-edge-cropped pickup annotations (frames 3745, 3834,
  3924, 4014, 4104) were missed, consistent with their `difficult=true` flag from human
  review (heavy edge cropping).
- **Ignored/difficult regions behaved as intended**: all 23 `ignore=true` annotations were
  excluded from scoring (verified: scored car GT count 139 = 162 total − 23 ignored); no
  ignored region was scored as a false negative.

## Runtime

| metric | value |
|---|---|
| Mean inference time | 72.10 ms |
| Median inference time | 45.71 ms |
| p95 inference time | 58.42 ms *(see note)* |
| Effective processing FPS | 13.87 |
| Source video FPS | 59.91 |
| Real-time factor | 0.232 |
| Device | CPU |

**Note on p95**: with only 35 sampled frames, the p95 percentile (58.42 ms) is lower than
the median-adjacent mean (72.10 ms) because one large first-frame cold-start cost (~967 ms,
model warm-up) dominates the mean but is a single outlier that the interpolated p95 mostly
excludes on this small sample — expected sample-size behavior, not a data error.

**Real-time verdict for this run**: YOLOv8s on this CPU-only machine processes Test7 at
roughly **23% of real-time** (13.9 effective FPS vs. a 59.9 FPS source) — it does **not**
meet real-time throughput on this development machine. This result is specific to this
CPU and must not be generalized to future phone or Meta-glasses hardware, which may have
very different (often better, sometimes GPU/NPU-accelerated) inference performance.

## Artifacts in this directory

- `run_metadata.json` — exact model/config/environment/frame-selection metadata
- `detections.csv` — 97 raw YOLOv8s detections across the 35 evaluated frames
- `summary.json` — observational detection-frequency/confidence/runtime summary
- `evaluation_report.json` — full precision/recall/F1 report (per-class, protected/priority
  combined, overall micro/macro, size/visibility/occlusion/scene breakdowns)
- `per_class_metrics.csv` — one row per canonical class
- `matched_detections.csv` — every true-positive match (IoU, confidence, difficult flag)
- `errors.csv` — every false positive, false negative, ignored prediction, and ignored
  ground truth, tagged by `error_type`
- `BASELINE_REPORT.md` — this file
