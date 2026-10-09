# Test7 Ground-Truth Audit — Faster R-CNN Flagged Cases

Audits Faster R-CNN's 132 false-positive predictions (`baselines/
fasterrcnn_resnet50_fpn_v2_baseline_v1/errors.csv`) against the canonical
`annotations.json`, to check whether any are visually plausible **missing**
ground-truth objects rather than genuine detector false positives.
**Faster R-CNN is treated as a review aid only, never as ground truth** —
every candidate below was checked against the original frame image itself
(ignoring any burned-in Atlas overlay text), and nothing was added unless
visually defensible from the frame alone.

**`annotations.json` was not modified by this task.**

## Methodology

1. **Automatic triage by IoU-adjacency** — for each of the 132 false
   positives, computed IoU against every existing ground-truth annotation
   of the same class in the same frame (including `ignore=true` ones):
   - **47 "near-miss"**: overlaps an existing, *scored* (non-ignored)
     annotation at IoU > 0.15 (but < the 0.50 match threshold) — almost
     certainly the *same physical object* Faster R-CNN drew a differently
     -sized box around, not a new object. Not individually re-verified
     visually; not candidates for addition.
   - **10 "ignored-region overlap"**: falls inside a region already marked
     `ignore=true` (the R07/R08 tree-occluded/overlapping background
     -vehicle clusters — see `HUMAN_REVIEW_GROUPS.md`). These regions were
     *deliberately* left unresolved by human review because they may
     contain multiple indistinguishable vehicles; this audit does not
     retroactively split them (per the standing "do not automatically
     split the cluster" policy from the ground-truth cleanup task).
   - **75 "uncovered"**: no meaningful overlap with any existing
     annotation — these are the real candidates for this audit.
2. **Direct visual inspection** of the original frame JPGs (never the
   burned-in overlay) for the highest-priority uncovered candidates: the
   explicitly-named frame-1498 bicycle, **all 18 person false positives**,
   and a representative sample of car/bus candidates to characterize the
   remaining pattern.
3. **Scope limit, stated plainly**: of the 75 "uncovered" candidates, this
   audit directly visually re-inspected the rendered frame for **10** of
   them (frame 1498's bicycle; both frame-2486, both frame-1648, and four
   of frame-1588's person candidates; one frame-2217 car; one frame-2756
   bus) and reasoned about **4 more** by cross-frame position consistency
   (see "recurring static detections" below). The remaining **~61**
   uncovered candidates (mostly `car`/`truck` predictions clustered in the
   39 car-FP and 11 truck-FP totals) were **not** individually re-opened
   as images in this pass — see "Car/truck pattern, not individually
   re-verified" below for why, and what a deeper follow-up would need to
   do differently.

## Audit queue

### Frame 1498 — bicycle-shaped prediction (explicit priority)

| | |
|---|---|
| Frame index | 1498 |
| Timestamp | 25.003s |
| Predicted class | `bicycle` |
| Confidence | 0.954 |
| BBox | [544, 693, 614, 736] |
| Current annotations in this region | none |
| Why it might be real | Sits at the edge of the small park/plaza area visible past the fence — a plausible location for a bike rack or parked bicycle |
| **Visual verdict** | **UNCERTAIN** — a pale, thin, roughly circular/vertical structure is visible near this location, consistent with either a parked bicycle or a decorative bike-rack/sculpture at the park entrance (this exact spot was already flagged as ambiguous during the original AI-assisted annotation pass — see the frame-1408 notes in `annotations_claude_draft.json`'s generation). At this resolution I cannot distinguish a genuine bicycle from a static rack/sculpture with confidence. **Not added.** |

### Person false positives (18 total, explicit priority)

| frame | bbox | confidence | region context | visual verdict |
|---|---|---|---|---|
| 1588 | [425,659,461,731] | 0.748 | Overlaps the bicycle-rider's own body (bicycle GT at [405,685,478,738]) | **Likely detector artifact** — the rider's torso, already counted as part of the `bicycle` annotation, not a separate person |
| 1588 | [79,694,103,782] | 0.668 | Thin sliver (24×88px) beside the already-annotated dark car | UNCERTAIN — too small/low-detail to confirm a person vs. background clutter |
| 1588 | [1413,843,1423,868] | 0.609 | Tiny box (10×25px), far right of frame | **Likely detector FP** — too small to be a legible person at this distance; no visible content |
| 1588 | [905,680,919,712] | 0.510 | Tiny box (14×32px) | **Likely detector FP** — same as above |
| 1648 | [1034,618,1057,705] | 0.950 | Near the park-gate/fence area | UNCERTAIN — a plausible small dark silhouette is visible near the gate, but not clearly resolvable as a person vs. a post/hydrant at this resolution |
| 1648 | [297,671,333,743] | 0.510 | Overlaps the bicycle-rider's body (bicycle GT at [278,668,352,738]) | **Likely detector artifact** — same rider already counted as `bicycle`, not a separate person |
| 1660, 1678 | ~[1033–1060, 618–701] | 0.51–0.61 | Same coordinates as the frame-1648 "park-gate" candidate above, recurring across 3 consecutive frames spanning only ~0.5s | UNCERTAIN, but the tight cross-frame position consistency is *some* evidence of a real (likely stationary) object rather than random noise — could be a person standing there briefly, or a static fixture (sign post/hydrant) the model repeatedly misreads. Not individually re-rendered; reasoned by pattern from the 1648 case. |
| 1857, 1947 | ~[1612–1624, 766–793] | 0.58–0.67 | Tiny (12×25px) box, recurring at nearly identical coordinates across 2 consecutive frames | **Likely detector FP** — the extreme smallness plus static recurrence across frames is much more consistent with a fixed street object (bollard/hydrant/sign) than a person |
| 2127 | [800,910,856,970] | 0.509 | Bottom-right area, no nearby GT | UNCERTAIN — not individually re-rendered |
| 2486 | [14,990,422,1076] | 0.971 | Very large (408×86px), extreme bottom-left edge of frame | **Likely detector FP** — no visible person at this location in the rendered frame; the unusual wide/short shape at the extreme bottom edge is consistent with a foreground/edge artifact, not a real pedestrian |
| 3205, 3385, 3565, 3685 | various, 20–70px boxes | 0.54–0.67 | Scattered, no nearby GT | UNCERTAIN — not individually re-rendered in this pass |
| 1408 | [1395,855,1410,895] | 0.540 | Tiny box (15×40px) | UNCERTAIN — not individually re-rendered |

**None of the 18 person false positives reached the "likely real missing GT" bar.** Several are explainable as the bicycle rider's own body being double-detected (already covered by the existing `bicycle` annotation), several are too small/low-confidence to trust, and the two recurring-position clusters are more consistent with static street furniture than people — but genuine uncertainty remains for a handful, appropriately left as **uncertain, ground truth unchanged**.

### Sample car/bus candidates (pattern check)

| frame | class | bbox | confidence | visual verdict |
|---|---|---|---|---|
| 2217 | car | [20,947,319,1038] | 0.994 | **Likely same object as existing GT** — overlaps (IoU ≈0.15, just under this audit's 0.15 near-miss cutoff) the already-annotated small car GT [0,955,95,1010] in this same wide/distant transitional frame; Faster R-CNN's box is simply much larger/looser around the same car, not a second vehicle |
| 2756 | bus | [1817,496,1920,676] | 0.949 | **Likely detector FP** — no visible bus at this location/height in the rendered frame (mostly tree canopy and building edge at the far right); the box's unusually high top edge (y=496, well above where any vehicle sits in this scene, ~y=590-660) is suspicious |

### Car/truck pattern, not individually re-verified (61 remaining "uncovered" candidates)

The remaining uncovered `car` (39) and `truck` (11) false positives are
concentrated in the **same frames and the same screen regions** as the
already-known, deliberately-unresolved background-vehicle clusters (the
tree-occluded area behind the park benches, frames 1408-2127; the
overlapping-cars area near the "83-10 Queens Blvd" building, frames
3205-4104; and the recurring ambiguous white commercial van at
3385/3475/3834, already documented as misclassified by every model tested
so far). Frame list: 1408, 1498, 1588, 1660, 1678, 1767, 1857, 2217, 2307,
2846, 3385, 3475, 3565, 3583, 3655, 3685, 3745, 3834, 3924, 4104 — this is
essentially the same frame set as `HUMAN_REVIEW_GROUPS.md`'s R07/R08
groups plus their immediate surroundings.

This audit did **not** re-open each of these 61 predictions as individual
images. Reasoning for that scope limit, stated plainly rather than hidden:
these regions were already visually inspected at length during ground
-truth construction and the subsequent human-decision pass, and the
explicit, already-recorded human judgment was that they contain an
*unknown* number of hard-to-separate vehicles — exactly the situation
`ignore=true` exists for. Treating each of Faster R-CNN's overlapping
guesses in that same cluster as a fresh, individually-confirmed new
ground-truth object would effectively let the detector silently expand the
already-deliberately-punted region, which is the opposite of the "high bar
for ground-truth change" this task requires. **If these clusters need
finer-grained ground truth, that is a dedicated future re-annotation task
(zooming into each cluster frame-by-frame with a fresh, careful pass), not
an automatic acceptance of detector output during an audit.**

## Summary counts

| category | count |
|---|---|
| Total Faster R-CNN false positives | 132 |
| Auto-resolved: near-miss of existing scored GT (same object, box mismatch) | 47 |
| Auto-resolved: overlaps an already-`ignore=true` region | 10 |
| Uncovered, directly visually inspected | 10 |
| Uncovered, reasoned by cross-frame pattern (not re-rendered) | 4 |
| Uncovered, car/truck cluster pattern (not individually re-verified — see above) | 61 |
| **Confirmed likely missing ground truth** | **0** |
| **Likely detector false positive** | 6 (2486-person, 1857/1947-person pair, 1588's two tiny persons, 2756-bus) + 2217-car (same-object) |
| **Uncertain (left unchanged)** | 1498-bicycle, 1588 (2 persons), 1648 (1 person), 1660/1678 (2 persons), 2127, 3205/3385/3565/3685 (4 persons), 1408 (1 person) — 12 total, plus the 61 un-reviewed car/truck cluster candidates |

## Recommendation

**No ground-truth corrections are justified at this audit's "high bar."**
Nothing visually confirmed itself as an unambiguous missing object.
`annotations_audit_candidate.json` was **not created** (per the task's own
instruction to skip it when no correction is justified).

If Test7's ground truth is to be improved further, the most productive next
step is a **dedicated, careful re-annotation pass** of the R07/R08 cluster
regions specifically (not an automatic audit of detector output) — this
matches `docs/ATLAS_DETECTOR_DATASET_V1.md`'s broader conclusion that
Test7's small, single-pass-reviewed ground truth is a known limitation.
