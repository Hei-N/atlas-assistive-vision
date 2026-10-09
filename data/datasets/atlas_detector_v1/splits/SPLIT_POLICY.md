# Split policy — Atlas Detector Dataset v1

## The rule

**Every frame from one source clip stays in exactly one split.** A clip is
assigned to `train`, `validation`, or `test` (evaluation) as a whole in
`../manifests/clip_tracker.csv`'s `split` column — never partially.

## Why

Neighboring video frames are near-duplicates (same vehicles, same
pedestrians, same background, seconds apart). Splitting neighboring frames
of the same clip across train and test would let the model effectively
"see the answer" during evaluation — inflating measured accuracy without
any real generalization improvement. Whole-clip assignment is the simplest
reliable way to prevent this leakage; frame-level random splitting is
explicitly **not** used for this dataset.

## The three splits

- **`train`** — used only for future fine-tuning (if that milestone
  happens). Never scored for a frozen baseline comparison.
- **`validation`** — used only during any future fine-tuning run to tune
  hyperparameters/early-stopping. Not the frozen non-regression reference.
- **`test` / evaluation** — the frozen, held-out reference set. Once a clip
  is assigned to `test`, it is **never** used for training or
  hyperparameter tuning, ever, even in a later milestone. `data/evaluation/
  test7_v1/` is the first member of this split by definition (Evaluation
  Dataset v1) and stays there.

## Files

- `train.txt`, `validation.txt`, `test.txt` — one `clip_id` per line, once
  clips exist. Not created yet (no real clips to list).
- The authoritative split assignment lives in `../manifests/
  clip_tracker.csv`'s `split` column; these `.txt` files (when generated)
  are a convenience export, not a second source of truth.

## Enforcement

Before any future fine-tuning run, cross-check that no `clip_id` appears in
both `train.txt`/`validation.txt` and `test.txt` — this should be a
scripted assertion when that tooling is built (not part of this
planning-only milestone).
