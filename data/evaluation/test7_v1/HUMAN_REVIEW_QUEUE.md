# Test7 AI-Assisted Draft -- Human Review Queue

Generated from `annotations_claude_draft.json` (Claude's AI-assisted draft labels). Lists every draft annotation Claude marked `medium` or `low` confidence -- i.e. everything that is NOT `auto_high_confidence`. A human should resolve each item (confirm, correct, or remove) before any of this draft is merged into the canonical `annotations.json`.

Total draft annotations: 181
High confidence (not listed below): 55
Medium confidence: 82
Low confidence: 44
Total requiring human review: 126

## Review items

### test7_frame_001408_t023.501.jpg

- frame_index: 1408
- timestamp_seconds: 23.501

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [130, 685, 330, 730]
  - **reason for uncertainty**: cluster of parked cars near the curb; possibly more than one vehicle overlapping -- verify count
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: person
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [175, 700, 215, 775]
  - **reason for uncertainty**: small, distant figure near the parked cars; Atlas's overlay also tags a person here but is not being trusted as truth -- verify this is actually a person
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1545, 655, 1900, 710]
  - **reason for uncertainty**: one or more cars across the street, partly hidden by tree branches; exact count/boxes uncertain
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_001498_t025.003.jpg

- frame_index: 1498
- timestamp_seconds: 25.003

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [110, 630, 330, 715]
  - **reason for uncertainty**: cluster of parked cars near the curb; possibly more than one vehicle overlapping -- verify count
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: person
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [110, 705, 160, 780]
  - **reason for uncertainty**: walking figure on the sidewalk; small/distant
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1545, 655, 1900, 710]
  - **reason for uncertainty**: one or more cars across the street, partly hidden by tree branches; exact count/boxes uncertain
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_001588_t026.505.jpg

- frame_index: 1588
- timestamp_seconds: 26.505

- **candidate class**: bicycle
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [405, 685, 478, 738]
  - **reason for uncertainty**: Atlas's own overlay labels this 'motorcycle'; independent visual read (thin wheels, upright pedaling posture, no visible engine bulk) looks more like a bicycle, but the rider is small/distant -- human must resolve, do not default to overlay
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [75, 685, 168, 725]
  - **reason for uncertainty**: small, near left edge
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1540, 655, 1900, 708]
  - **reason for uncertainty**: partly hidden by tree, count uncertain
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_001648_t027.507.jpg

- frame_index: 1648
- timestamp_seconds: 27.507

- **candidate class**: bicycle
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [278, 668, 352, 738]
  - **reason for uncertainty**: same bicycle/motorcycle ambiguity as 1588/1660 -- Atlas overlay says motorcycle, visual evidence points to bicycle
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [518, 595, 600, 648]
  - **reason for uncertainty**: mostly occluded by sign pole; class/box uncertain
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1545, 655, 1900, 708]
  - **reason for uncertainty**: partly hidden by tree, count uncertain
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_001660_t027.707.jpg

- frame_index: 1660
- timestamp_seconds: 27.707

- **candidate class**: bicycle
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [255, 665, 330, 748]
  - **reason for uncertainty**: SPECIAL CASE frame named explicitly in the task spec (bicycle_or_motorcycle_candidate). Atlas overlay labels this 'motorcycle | ID21'. Independent visual read: thin wheels, upright pedaling posture, no visible motor/engine bulk or license plate -- looks like a bicycle, but the object is small/distant enough that certainty is limited. Do not default to the overlay label either way -- human must resolve.
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [500, 590, 605, 650]
  - **reason for uncertainty**: mostly occluded by sign pole
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1545, 655, 1900, 708]
  - **reason for uncertainty**: partly hidden by tree, count uncertain
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_001678_t028.008.jpg

- frame_index: 1678
- timestamp_seconds: 28.008

- **candidate class**: bicycle
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [205, 705, 282, 748]
  - **reason for uncertainty**: bicycle shape visible; rider unclear / possibly dismounted
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [420, 590, 465, 628]
  - **reason for uncertainty**: white car next to bus
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [505, 590, 600, 650]
  - **reason for uncertainty**: dark vehicle mostly occluded by sign pole
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1545, 655, 1900, 708]
  - **reason for uncertainty**: partly hidden by tree, count uncertain
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_001767_t029.493.jpg

- frame_index: 1767
- timestamp_seconds: 29.493

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [355, 595, 460, 635]
  - **reason for uncertainty**: white car next to bus
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [460, 595, 620, 660]
  - **reason for uncertainty**: dark vehicle partially cut by pole; could be a pickup/SUV -- car vs truck uncertain
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [985, 650, 1160, 705]
  - **reason for uncertainty**: dark SUV/crossover
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [1160, 650, 1280, 700]
  - **reason for uncertainty**: white car beside the dark SUV
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1550, 655, 1750, 700]
  - **reason for uncertainty**: distant, partly treed
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_001857_t030.995.jpg

- frame_index: 1857
- timestamp_seconds: 30.995

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [330, 595, 465, 655]
  - **reason for uncertainty**: dark SUV near bus
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [855, 650, 1050, 715]
  - **reason for uncertainty**: boxy SUV/crossover shape -- car vs truck judgment call, leaning car
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1550, 655, 1750, 705]
  - **reason for uncertainty**: distant, partly treed
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_001947_t032.498.jpg

- frame_index: 1947
- timestamp_seconds: 32.498

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [185, 630, 340, 700]
  - **reason for uncertainty**: dark car
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [390, 595, 475, 635]
  - **reason for uncertainty**: white car, background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [805, 650, 1005, 715]
  - **reason for uncertainty**: dark SUV
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1550, 655, 1900, 705]
  - **reason for uncertainty**: distant cluster, partly treed
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_002037_t034.000.jpg

- frame_index: 2037
- timestamp_seconds: 34.0

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [95, 670, 250, 740]
  - **reason for uncertainty**: dark minivan/SUV shape
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [390, 630, 490, 670]
  - **reason for uncertainty**: white car, background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [805, 650, 1005, 715]
  - **reason for uncertainty**: dark SUV
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1550, 655, 1900, 705]
  - **reason for uncertainty**: distant cluster, partly treed
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_002127_t035.502.jpg

- frame_index: 2127
- timestamp_seconds: 35.502

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [175, 895, 260, 940]
  - **reason for uncertainty**: camera framing shifted this frame; small/distant
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [45, 940, 190, 1000]
  - **reason for uncertainty**: small/distant, cropped by tree
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [630, 895, 855, 1000]
  - **reason for uncertainty**: dark SUV
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [925, 880, 1170, 960]
  - **reason for uncertainty**: silver sedan
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1450, 870, 1900, 930]
  - **reason for uncertainty**: distant, partly treed
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_002217_t037.004.jpg

- frame_index: 2217
- timestamp_seconds: 37.004

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1030, 790, 1250, 860]
  - **reason for uncertainty**: tiny/distant; Atlas overlay tags 'car | ID31 | INSUFFICIENT_HISTORY' here. The recorded scene has jumped to a different, wider street view at this timestamp (different buildings visible) -- treated independently, not assumed to be the same framing as neighboring frames.
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [650, 895, 900, 940]
  - **reason for uncertainty**: white car, partly behind railing
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [0, 955, 95, 1010]
  - **reason for uncertainty**: cropped at left edge
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_002307_t038.506.jpg

- frame_index: 2307
- timestamp_seconds: 38.506

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [95, 705, 330, 780]
  - **reason for uncertainty**: dark SUV
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [780, 670, 1000, 725]
  - **reason for uncertainty**: small/distant
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [1275, 665, 1420, 720]
  - **reason for uncertainty**: dark car
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_002396_t039.992.jpg

- frame_index: 2396
- timestamp_seconds: 39.992

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [0, 595, 215, 680]
  - **reason for uncertainty**: dark SUV, cropped at left edge
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [890, 590, 1105, 650]
  - **reason for uncertainty**: small/distant
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [1225, 590, 1425, 650]
  - **reason for uncertainty**: dark car, labeled 'ID31' by overlay
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_002486_t041.494.jpg

- frame_index: 2486
- timestamp_seconds: 41.494

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [70, 590, 300, 680]
  - **reason for uncertainty**: dark SUV, 'ID27 stationary'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: person
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [280, 605, 330, 670]
  - **reason for uncertainty**: small figure beside the dark SUV -- possibly a pedestrian near the vehicle, verify
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [930, 605, 1150, 650]
  - **reason for uncertainty**: silver car, distant background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_002576_t042.996.jpg

- frame_index: 2576
- timestamp_seconds: 42.996

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [90, 595, 320, 685]
  - **reason for uncertainty**: dark SUV, 'ID27'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [940, 590, 1150, 650]
  - **reason for uncertainty**: silver car, distant background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_002666_t044.498.jpg

- frame_index: 2666
- timestamp_seconds: 44.498

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [70, 605, 300, 685]
  - **reason for uncertainty**: dark SUV, 'ID27'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [930, 610, 1145, 655]
  - **reason for uncertainty**: silver car, distant background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_002756_t046.001.jpg

- frame_index: 2756
- timestamp_seconds: 46.001

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [15, 605, 255, 685]
  - **reason for uncertainty**: dark SUV, 'ID27'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [880, 615, 1100, 660]
  - **reason for uncertainty**: silver/dark car, distant background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_002846_t047.503.jpg

- frame_index: 2846
- timestamp_seconds: 47.503

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [15, 610, 255, 690]
  - **reason for uncertainty**: dark SUV, 'ID27'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [880, 620, 1080, 665]
  - **reason for uncertainty**: dark/silver car, distant background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_002936_t049.005.jpg

- frame_index: 2936
- timestamp_seconds: 49.005

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [10, 610, 250, 690]
  - **reason for uncertainty**: dark SUV, 'ID27'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [875, 620, 1080, 665]
  - **reason for uncertainty**: dark/silver car, distant background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_003026_t050.507.jpg

- frame_index: 3026
- timestamp_seconds: 50.507

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [10, 610, 250, 690]
  - **reason for uncertainty**: dark SUV, background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [875, 620, 1080, 665]
  - **reason for uncertainty**: dark/silver car, distant background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [1215, 605, 1385, 660]
  - **reason for uncertainty**: dark car, background 'ID31'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_003115_t051.993.jpg

- frame_index: 3115
- timestamp_seconds: 51.993

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [10, 610, 250, 690]
  - **reason for uncertainty**: dark SUV, background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_003205_t053.495.jpg

- frame_index: 3205
- timestamp_seconds: 53.495

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [0, 630, 235, 705]
  - **reason for uncertainty**: dark SUV, background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [880, 630, 1085, 680]
  - **reason for uncertainty**: cluster of overlapping cars in the distance -- individual count/boxes uncertain
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1855, 685, 1920, 745]
  - **reason for uncertainty**: cropped at right edge
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_003295_t054.997.jpg

- frame_index: 3295
- timestamp_seconds: 54.997

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [880, 615, 1250, 770]
  - **reason for uncertainty**: large moving vehicle, center lane; class less certain due to angle/motion blur
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [1345, 670, 1790, 810]
  - **reason for uncertainty**: WHITE MINIVAN called out in the task spec -- sliding door and boxy wagon/minivan shape visible; classified as car per 'ordinary vans map to car' guidance, flagged for human confirmation of body style
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [350, 615, 590, 670]
  - **reason for uncertainty**: silver car, background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [1200, 615, 1350, 670]
  - **reason for uncertainty**: dark car, background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_003385_t056.499.jpg

- frame_index: 3385
- timestamp_seconds: 56.499

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [0, 670, 135, 810]
  - **reason for uncertainty**: red SUV, cropped at left edge
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [140, 615, 380, 770]
  - **reason for uncertainty**: dark SUV, 'ID38'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [880, 660, 1265, 810]
  - **reason for uncertainty**: recurring white minivan (same as 3295), near-stationary/slow across several frames
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: truck
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1370, 655, 1790, 800]
  - **reason for uncertainty**: LARGE WHITE COMMERCIAL BOX VAN -- tall flat-panel body, no rear side windows, boxy delivery-van shape. Atlas's own overlay calls this 'car' here but 'truck' for what looks like the same/similar vehicle in frame 3475 -- inconsistent, neither label is trusted. Canonical schema has no 'van' class; tentatively marked truck given the boxy commercial body, but human must decide car vs truck.
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1855, 615, 1920, 660]
  - **reason for uncertainty**: cropped at right edge
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [1240, 830, 1920, 960]
  - **reason for uncertainty**: large, close, partially cropped -- likely a black sedan passing near the camera
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_003475_t058.002.jpg

- frame_index: 3475
- timestamp_seconds: 58.002

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [15, 615, 280, 760]
  - **reason for uncertainty**: dark SUV, 'ID44 moving'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [345, 605, 565, 655]
  - **reason for uncertainty**: white car, background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: truck
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [605, 590, 1220, 770]
  - **reason for uncertainty**: SAME recurring large white commercial box van as 3385 -- Atlas's own overlay literally labels this one 'truck | ID43'. Genuinely ambiguous car-vs-truck commercial van body; human must decide.
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [1235, 605, 1590, 670]
  - **reason for uncertainty**: dark car, 'ID41 moving right'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [800, 780, 1420, 960]
  - **reason for uncertainty**: black sedan, foreground, close to camera
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1830, 730, 1920, 800]
  - **reason for uncertainty**: cropped at bottom-right edge
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_003565_t059.504.jpg

- frame_index: 3565
- timestamp_seconds: 59.504

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [0, 605, 190, 770]
  - **reason for uncertainty**: dark SUV, cropped at left edge
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [250, 605, 610, 800]
  - **reason for uncertainty**: recurring white minivan
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [890, 605, 1070, 660]
  - **reason for uncertainty**: dark car, background 'ID46'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_003583_t059.804.jpg

- frame_index: 3583
- timestamp_seconds: 59.804

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [0, 615, 175, 770]
  - **reason for uncertainty**: dark SUV, cropped at left edge
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [210, 615, 560, 800]
  - **reason for uncertainty**: recurring white minivan
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [890, 600, 1050, 660]
  - **reason for uncertainty**: dark blue car, background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_003655_t061.006.jpg

- frame_index: 3655
- timestamp_seconds: 61.006

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [0, 655, 115, 750]
  - **reason for uncertainty**: dark SUV, cropped at left edge
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [125, 610, 540, 800]
  - **reason for uncertainty**: recurring white minivan
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [880, 595, 1090, 650]
  - **reason for uncertainty**: dark car, background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [485, 790, 1050, 960]
  - **reason for uncertainty**: black sedan, foreground, close to camera
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_003685_t061.507.jpg

- frame_index: 3685
- timestamp_seconds: 61.507

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [0, 655, 90, 745]
  - **reason for uncertainty**: dark SUV, cropped at left edge
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [100, 610, 505, 800]
  - **reason for uncertainty**: recurring white minivan
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [860, 595, 1090, 650]
  - **reason for uncertainty**: cluster of overlapping vehicles in the distance; possibly 2 separate cars, exact count uncertain
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [1610, 605, 1850, 670]
  - **reason for uncertainty**: white SUV, moving, 'ID49'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [1015, 790, 1655, 960]
  - **reason for uncertainty**: black sedan, foreground, close to camera
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_003745_t062.508.jpg

- frame_index: 3745
- timestamp_seconds: 62.508

- **candidate class**: truck
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [0, 655, 100, 750]
  - **reason for uncertainty**: visible truck-bed shape at the left edge -- possible pickup truck, class uncertain (car vs truck), cropped
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [105, 615, 500, 810]
  - **reason for uncertainty**: recurring white minivan
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [395, 595, 580, 655]
  - **reason for uncertainty**: small/distant car, background
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [870, 595, 1100, 655]
  - **reason for uncertainty**: cluster of 2-3 overlapping cars in the distance; exact count/boxes uncertain
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1330, 595, 1550, 650]
  - **reason for uncertainty**: small/distant car, background 'ID52'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_003834_t063.994.jpg

- frame_index: 3834
- timestamp_seconds: 63.994

- **candidate class**: truck
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [0, 655, 90, 745]
  - **reason for uncertainty**: white pickup, cropped at left edge
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [95, 615, 390, 800]
  - **reason for uncertainty**: recurring white minivan, background-left
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [850, 590, 1100, 655]
  - **reason for uncertainty**: cluster of overlapping cars in the distance; count uncertain
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: truck
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [340, 780, 1150, 960]
  - **reason for uncertainty**: LARGE white box/cargo van filling much of the frame -- same recurring commercial-van ambiguity as 3385/3475. Clearly a boxy panel van (flat sides, no rear side windows), not a sedan. Car vs truck classification left for human judgment.
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [1035, 670, 1290, 800]
  - **reason for uncertainty**: dark SUV, moving, 'ID51'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1235, 650, 1480, 780]
  - **reason for uncertainty**: silver/gray sedan, partially cropped at right edge
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_003924_t065.496.jpg

- frame_index: 3924
- timestamp_seconds: 65.496

- **candidate class**: truck
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [0, 655, 90, 750]
  - **reason for uncertainty**: white pickup, cropped at left edge
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [95, 610, 500, 810]
  - **reason for uncertainty**: recurring white minivan
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [580, 595, 1080, 660]
  - **reason for uncertainty**: multiple overlapping vehicles in the distance; count uncertain
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [1510, 605, 1750, 660]
  - **reason for uncertainty**: dark car, 'ID54 moving right'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_004014_t066.998.jpg

- frame_index: 4014
- timestamp_seconds: 66.998

- **candidate class**: truck
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [0, 655, 95, 750]
  - **reason for uncertainty**: white pickup, cropped at left edge
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [100, 610, 500, 810]
  - **reason for uncertainty**: recurring white minivan
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [850, 595, 1080, 660]
  - **reason for uncertainty**: cluster of overlapping cars in the distance; count uncertain
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [1200, 590, 1385, 655]
  - **reason for uncertainty**: maroon car, background 'ID41 insufficient'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [1300, 605, 1520, 660]
  - **reason for uncertainty**: dark car, 'ID56'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1830, 685, 1920, 770]
  - **reason for uncertainty**: cropped at right edge, 'ID53 moving'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

### test7_frame_004104_t068.500.jpg

- frame_index: 4104
- timestamp_seconds: 68.5

- **candidate class**: truck
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [0, 655, 95, 750]
  - **reason for uncertainty**: white pickup, cropped at left edge, 'ID46'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [100, 610, 500, 810]
  - **reason for uncertainty**: recurring white minivan
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [850, 595, 1080, 660]
  - **reason for uncertainty**: cluster of overlapping cars in the distance; count uncertain
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: medium
  - **approximate bounding_box_xyxy**: [1200, 615, 1440, 670]
  - **reason for uncertainty**: dark car, moving, 'ID58'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

- **candidate class**: car
  - **confidence**: low
  - **approximate bounding_box_xyxy**: [1830, 685, 1920, 790]
  - **reason for uncertainty**: cropped at far right edge, 'ID53'
  - **what the human should decide**: confirm the class and tighten/correct the box, or remove this candidate if it is not actually a supported object.

