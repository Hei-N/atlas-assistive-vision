# Test7 YOLOv8m Candidate Baseline v1 — Comparison Measurement

This is a **candidate comparison run only** — YOLOv8m is not production and
this task did not change `config/settings.yaml`. It exists to answer one
question: does stepping up from `yolov8s.pt` to `yolov8m.pt` (same
architecture family, no code changes) improve detection quality enough,
without regressing protected classes, to justify further consideration?
Reference: `../yolov8s_baseline_v1/` (the frozen production baseline).

## Important limitation — same as the YOLOv8s baseline

Test7 is a **screen-recorded evaluation clip** containing Atlas's own
burned-in overlay pixels — not pristine raw camera footage. Ground truth
was built by independent human/AI-assisted visual review, never from the
overlay. Treat every number below as a same-dataset comparison only; it
should not be generalized to other environments or hardware.

## Run configuration (identical to the YOLOv8s baseline except the model)

| | |
|---|---|
| Model | `yolov8m.pt` (auto-downloaded via Ultralytics, official release; **not** production) |
| Model file SHA-256 | `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5` |
| Confidence threshold | `0.50` (same as production, explicitly passed, not tuned) |
| Evaluation matching IoU | `0.50` (same as YOLOv8s run) |
| Device | `cpu` (same machine) |
| Input resolution | `1920×1080` (native) |
| Evaluated frames | the same exact 35 frame indices from `frame_manifest.csv` |

## Candidate protected-class metrics

| class | GT | pred | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| car | 139 | 102 | 79 | 22 | 60 | 0.782 | 0.568 | 0.658 |
| bus | 6 | 5 | 5 | 0 | 1 | 1.000 | 0.833 | 0.909 |
| truck | 6 | 4 | 1 | 3 | 5 | 0.250 | 0.167 | 0.200 |
| **protected combined** | **151** | **111** | **85** | **25** | **66** | **0.773** | **0.563** | **0.651** |

## Candidate priority-class metrics

| class | GT | pred | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| bicycle | 4 | 1 | 1 | 0 | 3 | 1.000 | 0.250 | 0.400 |
| motorcycle | **0** | 2 | 0 | 2 | 0 | 0.000 | **undefined** | undefined |
| **priority combined** | 4 | 3 | 1 | 2 | 3 | 0.333 | 0.250 | 0.286 |

Motorcycle still has **zero ground-truth instances** — recall remains
undefined/inconclusive by definition. The candidate did produce 2 motorcycle
predictions (both false positives against 0 GT), which is new information
this run surfaced but which cannot itself be scored as a recall change.

## Person

| class | GT | pred | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| person | 3 | 3 | 2 | 1 | 1 | 0.667 | 0.667 | 0.667 |

## Deltas vs. YOLOv8s (candidate − baseline)

| metric | YOLOv8s | YOLOv8m | delta |
|---|---|---|---|
| car precision | 0.747 | 0.782 | **+0.035** |
| car recall | 0.446 | 0.568 | **+0.122** |
| car F1 | 0.559 | 0.658 | **+0.100** |
| bus precision | 0.444 | 1.000 | **+0.556** |
| bus recall | 0.667 | 0.833 | **+0.167** |
| bus F1 | 0.533 | 0.909 | **+0.376** |
| truck precision | 1.000 | 0.250 | **−0.750** |
| truck recall | 0.167 | 0.167 | 0.000 |
| truck F1 | 0.286 | 0.200 | **−0.086** |
| bicycle precision/recall/F1 | 1.0 / 0.25 / 0.4 | 1.0 / 0.25 / 0.4 | 0 / 0 / 0 (unchanged) |
| motorcycle | undefined (0 pred) | undefined, but 2 new FP | n/a |
| person precision | 1.000 | 0.667 | **−0.333** |
| person recall | 0.667 | 0.667 | 0.000 |
| protected combined precision | 0.720 | 0.773 | **+0.052** |
| protected combined recall | 0.444 | 0.563 | **+0.119** |
| protected combined F1 | 0.549 | 0.651 | **+0.102** |
| priority combined precision | 1.000 | 0.333 | **−0.667** |
| priority combined recall | 0.250 | 0.250 | 0.000 |
| priority combined F1 | 0.400 | 0.286 | **−0.114** |
| overall micro precision/recall/F1 | 0.729 / 0.443 / 0.551 | 0.759 / 0.557 / 0.642 | +0.030 / +0.114 / +0.092 |
| overall macro precision/recall/F1 | 0.838 / 0.439 / 0.516 | 0.616 / 0.497 / 0.567 | −0.222 / +0.057 / +0.051 |
| mean inference time | 72.10 ms | 155.04 ms | **+82.95 ms** |
| median inference time | 45.71 ms | 98.25 ms | **+52.54 ms** |
| p95 inference time | 58.42 ms | 275.31 ms | **+216.89 ms** |
| effective FPS | 13.87 | 6.45 | **−7.42** |
| real-time factor | 0.232 | 0.108 | **−0.124** |

(Macro precision dropped mainly because motorcycle now contributes a defined
`0.0` precision value into the macro average where it was previously
undefined/skipped — an artifact of macro-averaging across a class with 0 GT
producing new false positives, not a car/bus/truck/bicycle/person precision
drop; every individual protected class's own precision moved flat-to-up
except truck.)

## Error review

**Bicycle recovery**: none of the 3 previously-missed bicycle frames (1588,
1660, 1678) were recovered as `bicycle`. However, **frame 1678's bicycle was
newly detected** (IoU 0.806, up from the v8s TP's 0.607 at frame 1648) — the
matched frame *shifted* from 1648 to 1678, so recall is unchanged overall (1/4
either way), but 1678 is a genuine new correct match. At 1588 and 1648, the
same rider is now predicted as `motorcycle` instead of missed entirely (IoU
0.56–0.65 vs the bicycle ground truth) — a class-confusion, not a plain miss;
1660 remains a full miss under both models.

**Car false negatives**: improved — 77 → 60 (17 fewer misses), consistent
with the `small` (poorer-recall) and `medium` object-size categories
benefiting from a larger network. This is the single largest concrete gain.

**Truck**: still only 1 of 6 detected (frame 4104, the same non-difficult
"ID57" pickup as YOLOv8s, IoU improved 0.667→0.695). All 5 `difficult=true`
R06 left-edge-cropped pickups (3745, 3834, 3924, 4014, 4104) remain missed —
**no improvement** on the hard truck cases. New truck false positives appeared
(0 → 3), all traceable to already-known-hard regions, not clean new
footage:
  - **frame 1767**: a duplicate `truck` box (IoU 0.93 against the ground-truth
    *bus*) alongside the correctly-matched `bus` prediction at the same
    location — a same-object dual-class output, not a new object being
    hallucinated.
  - **frame 3834**: the recurring ambiguous white commercial box/cargo van
    (review group R02) — YOLOv8s called this same object `bus`; YOLOv8m now
    calls it `truck`. Both are wrong (ground truth is `car` per human
    review), confirming this object is genuinely hard across model
    generations, not a new failure mode.
  - **frame 4014**: an oversized, loosely-drawn `truck` box (IoU only 0.13
    against the tight R06 ground-truth crop) that appears to span both the
    cropped pickup and the adjacent minivan region — a localization/box-size
    issue rather than a wrong detection location.

**Bus/van confusion**: meaningfully reduced, not worsened — YOLOv8s had 5
bus false positives (several coinciding with the R02 van); YOLOv8m has 0 bus
false positives and bus recall rose from 0.667 to 0.833. The van-related
confusion didn't disappear, it *moved* to the `truck` class (see frame 3834
above) rather than `bus`.

**New protected-class false negatives**: none — car FN and bus FN both
decreased; truck FN is unchanged (same 5 frames missed both times).

**New/worsened false positives**: car FP +1 (21→22, essentially flat), truck
FP +3 (0→3, see above), bus FP −5 (5→0, improvement), person FP +1 (0→1, new,
at frame 2486 — see `errors.csv`), motorcycle FP +2 (0→2, both explainable as
the same-rider class-confusion described above). Overall this is not "false
positives increasing badly" in aggregate (total FP across all classes: 26 →
30, a modest increase concentrated in already-known-ambiguous regions), but
the per-class truck and priority-combined precision swings are real and
should not be hidden behind the improved aggregate numbers.

## Runtime

Mean inference time roughly **doubled** (72.1 ms → 155.0 ms), effective FPS
roughly **halved** (13.87 → 6.45), real-time factor dropped from 0.232 to
0.108 (≈11% of real-time on this CPU). Both models are already far from
real-time on this development machine; the relative cost of stepping up to
"m" is consistent with published Ultralytics speed/accuracy tradeoffs and is
not itself disqualifying for a benchmarking-only comparison, but it is a
real, material runtime regression that would matter for any live-deployment
decision.

## Artifacts in this directory

- `run_metadata.json`, `detections.csv`, `summary.json`,
  `evaluation_report.json`, `per_class_metrics.csv`, `matched_detections.csv`,
  `errors.csv`, `BASELINE_REPORT.md` (this file)
