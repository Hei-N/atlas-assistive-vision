# Test7 Evaluation Clip #1 -- Human Labeling Checklist

One entry per selected frame (35 total). Use this to track review progress; the actual labels go into `annotations.json` (see `README.md`'s "Next Step -- Human Annotation" section and `docs/DETECTOR_ANNOTATIONS.md`).

## Canonical classes (exact -- no others)

- car
- bus
- truck
- bicycle
- motorcycle
- person

## Rules

- Label **every** visible supported object in the frame -- do not label only the "interesting" object.
- Do **not** copy the burned-in Atlas/YOLO overlay label as truth -- judge each object yourself.
- Do **not** invent `e-bike`/`scooter`/`moped` as canonical classes -- use `bicycle`/`motorcycle` with an optional `vehicle_subtype_note`.
- An ambiguous bicycle-vs-motorcycle case must be resolved by human visual judgment, never defaulted to the overlay.
- If no supported object is visible anywhere in the frame, mark it **reviewed empty** (a `reviewed_frames` entry), not left blank.
- A partially visible object should still be labeled if its class is reasonably identifiable.
- Use `ignore`/`difficult` only per the existing schema guidance (`docs/DETECTOR_ANNOTATIONS.md` section 5) -- not as a shortcut to skip judgment calls.

## Frames (35)

### test7_frame_001408_t023.501.jpg

- **frame_index**: 1408
- **timestamp_seconds**: 23.501
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_001498_t025.003.jpg

- **frame_index**: 1498
- **timestamp_seconds**: 25.003
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_001588_t026.505.jpg

- **frame_index**: 1588
- **timestamp_seconds**: 26.505
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_001648_t027.507.jpg

- **frame_index**: 1648
- **timestamp_seconds**: 27.507
- **selected_reason**: school_bus
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_001660_t027.707.jpg

- **frame_index**: 1660
- **timestamp_seconds**: 27.707
- **selected_reason**: bicycle_or_motorcycle_candidate
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_001678_t028.008.jpg

- **frame_index**: 1678
- **timestamp_seconds**: 28.008
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_001767_t029.493.jpg

- **frame_index**: 1767
- **timestamp_seconds**: 29.493
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_001857_t030.995.jpg

- **frame_index**: 1857
- **timestamp_seconds**: 30.995
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_001947_t032.498.jpg

- **frame_index**: 1947
- **timestamp_seconds**: 32.498
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_002037_t034.000.jpg

- **frame_index**: 2037
- **timestamp_seconds**: 34.0
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_002127_t035.502.jpg

- **frame_index**: 2127
- **timestamp_seconds**: 35.502
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_002217_t037.004.jpg

- **frame_index**: 2217
- **timestamp_seconds**: 37.004
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_002307_t038.506.jpg

- **frame_index**: 2307
- **timestamp_seconds**: 38.506
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_002396_t039.992.jpg

- **frame_index**: 2396
- **timestamp_seconds**: 39.992
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_002486_t041.494.jpg

- **frame_index**: 2486
- **timestamp_seconds**: 41.494
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_002576_t042.996.jpg

- **frame_index**: 2576
- **timestamp_seconds**: 42.996
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_002666_t044.498.jpg

- **frame_index**: 2666
- **timestamp_seconds**: 44.498
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_002756_t046.001.jpg

- **frame_index**: 2756
- **timestamp_seconds**: 46.001
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_002846_t047.503.jpg

- **frame_index**: 2846
- **timestamp_seconds**: 47.503
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_002936_t049.005.jpg

- **frame_index**: 2936
- **timestamp_seconds**: 49.005
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_003026_t050.507.jpg

- **frame_index**: 3026
- **timestamp_seconds**: 50.507
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_003115_t051.993.jpg

- **frame_index**: 3115
- **timestamp_seconds**: 51.993
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_003205_t053.495.jpg

- **frame_index**: 3205
- **timestamp_seconds**: 53.495
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_003295_t054.997.jpg

- **frame_index**: 3295
- **timestamp_seconds**: 54.997
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_003385_t056.499.jpg

- **frame_index**: 3385
- **timestamp_seconds**: 56.499
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_003475_t058.002.jpg

- **frame_index**: 3475
- **timestamp_seconds**: 58.002
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_003565_t059.504.jpg

- **frame_index**: 3565
- **timestamp_seconds**: 59.504
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_003583_t059.804.jpg

- **frame_index**: 3583
- **timestamp_seconds**: 59.804
- **selected_reason**: moving_vehicle
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_003655_t061.006.jpg

- **frame_index**: 3655
- **timestamp_seconds**: 61.006
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_003685_t061.507.jpg

- **frame_index**: 3685
- **timestamp_seconds**: 61.507
- **selected_reason**: dense_multi_vehicle
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_003745_t062.508.jpg

- **frame_index**: 3745
- **timestamp_seconds**: 62.508
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_003834_t063.994.jpg

- **frame_index**: 3834
- **timestamp_seconds**: 63.994
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_003924_t065.496.jpg

- **frame_index**: 3924
- **timestamp_seconds**: 65.496
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_004014_t066.998.jpg

- **frame_index**: 4014
- **timestamp_seconds**: 66.998
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

### test7_frame_004104_t068.500.jpg

- **frame_index**: 4104
- **timestamp_seconds**: 68.5
- **selected_reason**: interval_sample
- **status**:
  - [ ] not reviewed
  - [ ] reviewed with objects
  - [ ] reviewed empty
- **notes**: 

