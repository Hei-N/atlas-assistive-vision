# manifests/

- **`clip_tracker.csv`** — the dataset-level tracker: one row per clip,
  filled in as footage is collected (see field meanings in
  `docs/ATLAS_DETECTOR_DATASET_V1.md` §"Dataset tracker"). Currently header
  -only — no clips have been collected yet, and no fake rows were added.
- Per-clip **benchmark manifests** (the `id`/`path`/`source_type`/
  `scene_tags`/`lighting`/`camera_motion`/`expected_relevant_classes`/
  `notes` format from `docs/DETECTOR_BENCHMARK.md` §7a) are a separate,
  narrower artifact consumed directly by `scripts/benchmark_detector.py` —
  generate one from this tracker once clips exist and you're ready to run a
  detector benchmark, don't hand-maintain two overlapping manifests.
