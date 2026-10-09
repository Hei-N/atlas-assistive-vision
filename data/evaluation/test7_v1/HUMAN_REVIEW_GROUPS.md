# Test7 AI-Assisted Draft -- Human Review Groups

Organizes the 126 items in `HUMAN_REVIEW_QUEUE.md` (from `annotations_claude_draft.json`) into recurring visual decision groups, to reduce the number of independent human decisions needed. Grouping was done by Claude visually re-inspecting the relevant frames in `frames/` -- **this is a human-review convenience grouping only, not a claim of true object re-identification.** Atlas's burned-in overlay was not used as truth for grouping or for any class judgment. No class was resolved and `annotations_claude_draft.json` was not modified by this task.

## Groups (sorted by priority)

Priority order: 1) class ambiguity, 2) possible missed/false class, 3) severe occlusion, 4) tiny/distant objects, 5) minor bbox uncertainty.

### GROUP R01 -- Bicycle/motorcycle rider

- **Priority**: 1 (class ambiguity)
- **Frames**: 1588, 1648, 1660, 1678
- **Timestamps (s)**: 26.505, 27.507, 27.707, 28.008
- **Current draft class(es)**: bicycle
- **Approximate bbox per frame**:
  - frame 1588 (test7_frame_001588_t026.505.jpg): [405, 685, 478, 738]
  - frame 1648 (test7_frame_001648_t027.507.jpg): [278, 668, 352, 738]
  - frame 1660 (test7_frame_001660_t027.707.jpg): [255, 665, 330, 748]
  - frame 1678 (test7_frame_001678_t028.008.jpg): [205, 705, 282, 748]
- **Reason for review**: Atlas's own overlay labels this rider 'motorcycle' in every frame; independent visual read (thin wheels, upright pedaling posture, no visible engine bulk) leans bicycle, but the rider is small/distant. This is the bicycle_or_motorcycle_candidate frame (1660) plus 3 visually adjacent frames showing what looks like the same rider.
- **Same physical object likely?**: Likely yes -- consistent position/motion across 4 consecutive sampled frames spanning ~1.5s (t=26.5-28.0s).
- **Question for the human**: “Is this rider a bicycle or a motorcycle?”

### GROUP R02 -- White commercial box van

- **Priority**: 1 (class ambiguity)
- **Frames**: 3385, 3475, 3834
- **Timestamps (s)**: 56.499, 58.002, 63.994
- **Current draft class(es)**: truck
- **Approximate bbox per frame**:
  - frame 3385 (test7_frame_003385_t056.499.jpg): [1370, 655, 1790, 800]
  - frame 3475 (test7_frame_003475_t058.002.jpg): [605, 590, 1220, 770]
  - frame 3834 (test7_frame_003834_t063.994.jpg): [340, 780, 1150, 960]
- **Reason for review**: Tall flat-panel body, no rear side windows -- a boxy commercial/delivery-van shape with no clean canonical class. Atlas's own overlay inconsistently calls this vehicle 'car' in one frame and 'truck' in another; draft tentatively used truck.
- **Same physical object likely?**: Likely yes -- same recurring commercial van across a ~7.5s window (t=56.5-64.0s).
- **Question for the human**: “Is this commercial box van a car/van or a truck?”

### GROUP R03 -- Dark vehicle occluded behind sign pole

- **Priority**: 1 (class ambiguity)
- **Frames**: 1648, 1660, 1678, 1767
- **Timestamps (s)**: 27.507, 27.707, 28.008, 29.493
- **Current draft class(es)**: car
- **Approximate bbox per frame**:
  - frame 1648 (test7_frame_001648_t027.507.jpg): [518, 595, 600, 648]
  - frame 1660 (test7_frame_001660_t027.707.jpg): [500, 590, 605, 650]
  - frame 1678 (test7_frame_001678_t028.008.jpg): [505, 590, 600, 650]
  - frame 1767 (test7_frame_001767_t029.493.jpg): [460, 595, 620, 660]
- **Reason for review**: Mostly hidden behind a street sign pole near the bus stop; one frame's note explicitly raises car vs. pickup/SUV uncertainty.
- **Same physical object likely?**: Likely yes -- same stationary/slow vehicle behind the pole across ~1.8s (t=27.5-29.5s).
- **Question for the human**: “Is this occluded vehicle a car, SUV, or truck?”

### GROUP R04 -- Boxy SUV/crossover near park benches

- **Priority**: 2 (possible missed/false class)
- **Frames**: 1767, 1857, 1947, 2037
- **Timestamps (s)**: 29.493, 30.995, 32.498, 34.0
- **Current draft class(es)**: car
- **Approximate bbox per frame**:
  - frame 1767 (test7_frame_001767_t029.493.jpg): [985, 650, 1160, 705]
  - frame 1857 (test7_frame_001857_t030.995.jpg): [855, 650, 1050, 715]
  - frame 1947 (test7_frame_001947_t032.498.jpg): [805, 650, 1005, 715]
  - frame 2037 (test7_frame_002037_t034.000.jpg): [805, 650, 1005, 715]
- **Reason for review**: Consistent screen position across 4 frames; one frame's note explicitly calls this a 'car vs truck judgment call' due to its boxy crossover shape.
- **Same physical object likely?**: Likely yes -- stationary vehicle at a near-fixed position across ~4.5s (t=29.5-34.0s).
- **Question for the human**: “Is this SUV/crossover correctly a car, or should its boxy shape push it toward truck?”

### GROUP R05 -- Recurring white minivan

- **Priority**: 2 (possible missed/false class)
- **Frames**: 3295, 3385, 3565, 3583, 3655, 3685, 3745, 3834, 3924, 4014, 4104
- **Timestamps (s)**: 54.997, 56.499, 59.504, 59.804, 61.006, 61.507, 62.508, 63.994, 65.496, 66.998, 68.5
- **Current draft class(es)**: car
- **Approximate bbox per frame**:
  - frame 3295 (test7_frame_003295_t054.997.jpg): [1345, 670, 1790, 810]
  - frame 3385 (test7_frame_003385_t056.499.jpg): [880, 660, 1265, 810]
  - frame 3565 (test7_frame_003565_t059.504.jpg): [250, 605, 610, 800]
  - frame 3583 (test7_frame_003583_t059.804.jpg): [210, 615, 560, 800]
  - frame 3655 (test7_frame_003655_t061.006.jpg): [125, 610, 540, 800]
  - frame 3685 (test7_frame_003685_t061.507.jpg): [100, 610, 505, 800]
  - frame 3745 (test7_frame_003745_t062.508.jpg): [105, 615, 500, 810]
  - frame 3834 (test7_frame_003834_t063.994.jpg): [95, 615, 390, 800]
  - frame 3924 (test7_frame_003924_t065.496.jpg): [95, 610, 500, 810]
  - frame 4014 (test7_frame_004014_t066.998.jpg): [100, 610, 500, 810]
  - frame 4104 (test7_frame_004104_t068.500.jpg): [100, 610, 500, 810]
- **Reason for review**: Sliding-door minivan body classified as car per 'ordinary vans map to car' guidance. Draft treats every appearance as the same near-stationary/slow vehicle -- this is a convenience grouping across a long span, not a confirmed re-identification.
- **Same physical object likely?**: Plausible across the whole stretch (near-identical position/appearance was observed on direct sequential review), but re-identification confidence is lower than the shorter, tighter groups above given the ~13.5s span (t=55.0-68.5s) involved.
- **Question for the human**: “Confirm this is a minivan correctly classified as car, and confirm it's one vehicle throughout rather than several similar vans.”

### GROUP R06 -- Recurring white pickup, cropped at left edge

- **Priority**: 2 (possible missed/false class)
- **Frames**: 3745, 3834, 3924, 4014, 4104
- **Timestamps (s)**: 62.508, 63.994, 65.496, 66.998, 68.5
- **Current draft class(es)**: truck
- **Approximate bbox per frame**:
  - frame 3745 (test7_frame_003745_t062.508.jpg): [0, 655, 100, 750]
  - frame 3834 (test7_frame_003834_t063.994.jpg): [0, 655, 90, 745]
  - frame 3924 (test7_frame_003924_t065.496.jpg): [0, 655, 90, 750]
  - frame 4014 (test7_frame_004014_t066.998.jpg): [0, 655, 95, 750]
  - frame 4104 (test7_frame_004104_t068.500.jpg): [0, 655, 95, 750]
- **Reason for review**: A visible truck-bed shape at the extreme left frame edge, but consistently cropped in every appearance.
- **Same physical object likely?**: Likely yes -- consistent position/size at the frame edge across ~5.5s (t=62.5-68.5s).
- **Question for the human**: “Confirm this cropped left-edge vehicle is genuinely a pickup truck despite being consistently cut off.”

### GROUP R07 -- Distant tree-occluded background vehicles (park scene, right side)

- **Priority**: 3 (severe occlusion)
- **Frames**: 1408, 1498, 1588, 1648, 1660, 1678, 1767, 1857, 1947, 2037, 2127
- **Timestamps (s)**: 23.501, 25.003, 26.505, 27.507, 27.707, 28.008, 29.493, 30.995, 32.498, 34.0, 35.502
- **Current draft class(es)**: car
- **Approximate bbox per frame**:
  - frame 1408 (test7_frame_001408_t023.501.jpg): [1545, 655, 1900, 710]
  - frame 1498 (test7_frame_001498_t025.003.jpg): [1545, 655, 1900, 710]
  - frame 1588 (test7_frame_001588_t026.505.jpg): [1540, 655, 1900, 708]
  - frame 1648 (test7_frame_001648_t027.507.jpg): [1545, 655, 1900, 708]
  - frame 1660 (test7_frame_001660_t027.707.jpg): [1545, 655, 1900, 708]
  - frame 1678 (test7_frame_001678_t028.008.jpg): [1545, 655, 1900, 708]
  - frame 1767 (test7_frame_001767_t029.493.jpg): [1550, 655, 1750, 700]
  - frame 1857 (test7_frame_001857_t030.995.jpg): [1550, 655, 1750, 705]
  - frame 1947 (test7_frame_001947_t032.498.jpg): [1550, 655, 1900, 705]
  - frame 2037 (test7_frame_002037_t034.000.jpg): [1550, 655, 1900, 705]
  - frame 2127 (test7_frame_002127_t035.502.jpg): [1450, 870, 1900, 930]
- **Reason for review**: Partly hidden by tree branches across the street; the number of distinct vehicles present is uncertain.
- **Same physical object likely?**: Same general parked-vehicle area throughout (t=23.5-35.5s), but this almost certainly contains more than one physical car -- grouped only as a shared review question, not a single-object claim.
- **Question for the human**: “How many distinct vehicles are actually present in this tree-occluded area, and are they all cars?”

### GROUP R08 -- Overlapping background vehicle cluster (street scene, center-distance)

- **Priority**: 3 (severe occlusion)
- **Frames**: 3205, 3685, 3745, 3834, 3924, 4014, 4104
- **Timestamps (s)**: 53.495, 61.507, 62.508, 63.994, 65.496, 66.998, 68.5
- **Current draft class(es)**: car
- **Approximate bbox per frame**:
  - frame 3205 (test7_frame_003205_t053.495.jpg): [880, 630, 1085, 680]
  - frame 3685 (test7_frame_003685_t061.507.jpg): [860, 595, 1090, 650]
  - frame 3745 (test7_frame_003745_t062.508.jpg): [870, 595, 1100, 655]
  - frame 3834 (test7_frame_003834_t063.994.jpg): [850, 590, 1100, 655]
  - frame 3924 (test7_frame_003924_t065.496.jpg): [580, 595, 1080, 660]
  - frame 4014 (test7_frame_004014_t066.998.jpg): [850, 595, 1080, 660]
  - frame 4104 (test7_frame_004104_t068.500.jpg): [850, 595, 1080, 660]
- **Reason for review**: Cluster of overlapping/occluded vehicles near the 83-10 Queens Blvd building; exact count uncertain.
- **Same physical object likely?**: Same general parking area throughout (t=53.5-68.5s), likely containing 2+ distinct vehicles -- grouped only as a shared review question, not a single-object claim.
- **Question for the human**: “How many distinct vehicles are actually present in this overlapping cluster?”

### GROUP R09 -- Left-edge cropped dark SUV (street scene)

- **Priority**: 3 (severe occlusion)
- **Frames**: 3565, 3583, 3655, 3685
- **Timestamps (s)**: 59.504, 59.804, 61.006, 61.507
- **Current draft class(es)**: car
- **Approximate bbox per frame**:
  - frame 3565 (test7_frame_003565_t059.504.jpg): [0, 605, 190, 770]
  - frame 3583 (test7_frame_003583_t059.804.jpg): [0, 615, 175, 770]
  - frame 3655 (test7_frame_003655_t061.006.jpg): [0, 655, 115, 750]
  - frame 3685 (test7_frame_003685_t061.507.jpg): [0, 655, 90, 745]
- **Reason for review**: Cropped at the extreme left frame edge in every appearance, with a shrinking visible slice.
- **Same physical object likely?**: Likely yes -- consistent shrinking-crop pattern suggests one vehicle across ~2s (t=59.5-61.5s).
- **Question for the human**: “Confirm this is one vehicle (not several different cars) and confirm car classification despite the heavy edge crop.”

### GROUP R10 -- Person near parked cars (park scene opening)

- **Priority**: 4 (tiny/distant objects)
- **Frames**: 1408, 1498
- **Timestamps (s)**: 23.501, 25.003
- **Current draft class(es)**: person
- **Approximate bbox per frame**:
  - frame 1408 (test7_frame_001408_t023.501.jpg): [175, 700, 215, 775]
  - frame 1498 (test7_frame_001498_t025.003.jpg): [110, 705, 160, 780]
- **Reason for review**: Small, distant walking figure; Atlas's own overlay also tags a person here but is not trusted as truth.
- **Same physical object likely?**: Likely yes -- same pedestrian, frames are 1.5s apart.
- **Question for the human**: “Confirm this small distant figure is actually a person, and whether it's the same person in both frames.”

## Singleton items (not grouped -- each is its own decision)

These 71 items did not have a clear recurring match in an adjacent/nearby frame, or the match was too uncertain to claim even as a convenience grouping. Listed individually, bucketed by the same priority order as the groups above so they can still be triaged in batches.

### Priority 2 -- possible missed/false class

- frame 3295 (test7_frame_003295_t054.997.jpg), t=54.997s -- **car** (medium) [880, 615, 1250, 770]: large moving vehicle, center lane; class less certain due to angle/motion blur
### Priority 3 -- severe occlusion

- frame 2127 (test7_frame_002127_t035.502.jpg), t=35.502s -- **car** (low) [45, 940, 190, 1000]: small/distant, cropped by tree
- frame 2217 (test7_frame_002217_t037.004.jpg), t=37.004s -- **car** (medium) [650, 895, 900, 940]: white car, partly behind railing
- frame 2217 (test7_frame_002217_t037.004.jpg), t=37.004s -- **car** (low) [0, 955, 95, 1010]: cropped at left edge
- frame 2396 (test7_frame_002396_t039.992.jpg), t=39.992s -- **car** (medium) [0, 595, 215, 680]: dark SUV, cropped at left edge
- frame 3205 (test7_frame_003205_t053.495.jpg), t=53.495s -- **car** (low) [1855, 685, 1920, 745]: cropped at right edge
- frame 3385 (test7_frame_003385_t056.499.jpg), t=56.499s -- **car** (medium) [0, 670, 135, 810]: red SUV, cropped at left edge
- frame 3385 (test7_frame_003385_t056.499.jpg), t=56.499s -- **car** (low) [1855, 615, 1920, 660]: cropped at right edge
- frame 3385 (test7_frame_003385_t056.499.jpg), t=56.499s -- **car** (medium) [1240, 830, 1920, 960]: large, close, partially cropped -- likely a black sedan passing near the camera
- frame 3475 (test7_frame_003475_t058.002.jpg), t=58.002s -- **car** (low) [1830, 730, 1920, 800]: cropped at bottom-right edge
- frame 3834 (test7_frame_003834_t063.994.jpg), t=63.994s -- **car** (low) [1235, 650, 1480, 780]: silver/gray sedan, partially cropped at right edge
- frame 4014 (test7_frame_004014_t066.998.jpg), t=66.998s -- **car** (low) [1830, 685, 1920, 770]: cropped at right edge, 'ID53 moving'
- frame 4104 (test7_frame_004104_t068.500.jpg), t=68.5s -- **car** (low) [1830, 685, 1920, 790]: cropped at far right edge, 'ID53'
### Priority 4 -- tiny/distant objects

- frame 2127 (test7_frame_002127_t035.502.jpg), t=35.502s -- **car** (low) [175, 895, 260, 940]: camera framing shifted this frame; small/distant
- frame 2217 (test7_frame_002217_t037.004.jpg), t=37.004s -- **car** (low) [1030, 790, 1250, 860]: tiny/distant; Atlas overlay tags 'car | ID31 | INSUFFICIENT_HISTORY' here. The recorded scene has jumped to a different, wider street view at this timestamp (different buildings visible) -- treated independently, not assumed to be the same framing as neighboring frames.
- frame 2307 (test7_frame_002307_t038.506.jpg), t=38.506s -- **car** (low) [780, 670, 1000, 725]: small/distant
- frame 2396 (test7_frame_002396_t039.992.jpg), t=39.992s -- **car** (low) [890, 590, 1105, 650]: small/distant
- frame 2486 (test7_frame_002486_t041.494.jpg), t=41.494s -- **car** (medium) [930, 605, 1150, 650]: silver car, distant background
- frame 2576 (test7_frame_002576_t042.996.jpg), t=42.996s -- **car** (medium) [940, 590, 1150, 650]: silver car, distant background
- frame 2666 (test7_frame_002666_t044.498.jpg), t=44.498s -- **car** (medium) [930, 610, 1145, 655]: silver car, distant background
- frame 2756 (test7_frame_002756_t046.001.jpg), t=46.001s -- **car** (medium) [880, 615, 1100, 660]: silver/dark car, distant background
- frame 2846 (test7_frame_002846_t047.503.jpg), t=47.503s -- **car** (medium) [880, 620, 1080, 665]: dark/silver car, distant background
- frame 2936 (test7_frame_002936_t049.005.jpg), t=49.005s -- **car** (medium) [875, 620, 1080, 665]: dark/silver car, distant background
- frame 3026 (test7_frame_003026_t050.507.jpg), t=50.507s -- **car** (medium) [875, 620, 1080, 665]: dark/silver car, distant background
- frame 3745 (test7_frame_003745_t062.508.jpg), t=62.508s -- **car** (low) [395, 595, 580, 655]: small/distant car, background
- frame 3745 (test7_frame_003745_t062.508.jpg), t=62.508s -- **car** (low) [1330, 595, 1550, 650]: small/distant car, background 'ID52'
### Priority 5 -- minor bbox uncertainty

- frame 1408 (test7_frame_001408_t023.501.jpg), t=23.501s -- **car** (medium) [130, 685, 330, 730]: cluster of parked cars near the curb; possibly more than one vehicle overlapping -- verify count
- frame 1498 (test7_frame_001498_t025.003.jpg), t=25.003s -- **car** (medium) [110, 630, 330, 715]: cluster of parked cars near the curb; possibly more than one vehicle overlapping -- verify count
- frame 1588 (test7_frame_001588_t026.505.jpg), t=26.505s -- **car** (medium) [75, 685, 168, 725]: small, near left edge
- frame 1678 (test7_frame_001678_t028.008.jpg), t=28.008s -- **car** (medium) [420, 590, 465, 628]: white car next to bus
- frame 1767 (test7_frame_001767_t029.493.jpg), t=29.493s -- **car** (medium) [355, 595, 460, 635]: white car next to bus
- frame 1767 (test7_frame_001767_t029.493.jpg), t=29.493s -- **car** (medium) [1160, 650, 1280, 700]: white car beside the dark SUV
- frame 1857 (test7_frame_001857_t030.995.jpg), t=30.995s -- **car** (medium) [330, 595, 465, 655]: dark SUV near bus
- frame 1947 (test7_frame_001947_t032.498.jpg), t=32.498s -- **car** (medium) [185, 630, 340, 700]: dark car
- frame 1947 (test7_frame_001947_t032.498.jpg), t=32.498s -- **car** (medium) [390, 595, 475, 635]: white car, background
- frame 2037 (test7_frame_002037_t034.000.jpg), t=34.0s -- **car** (medium) [95, 670, 250, 740]: dark minivan/SUV shape
- frame 2037 (test7_frame_002037_t034.000.jpg), t=34.0s -- **car** (medium) [390, 630, 490, 670]: white car, background
- frame 2127 (test7_frame_002127_t035.502.jpg), t=35.502s -- **car** (medium) [630, 895, 855, 1000]: dark SUV
- frame 2127 (test7_frame_002127_t035.502.jpg), t=35.502s -- **car** (medium) [925, 880, 1170, 960]: silver sedan
- frame 2307 (test7_frame_002307_t038.506.jpg), t=38.506s -- **car** (medium) [95, 705, 330, 780]: dark SUV
- frame 2307 (test7_frame_002307_t038.506.jpg), t=38.506s -- **car** (medium) [1275, 665, 1420, 720]: dark car
- frame 2396 (test7_frame_002396_t039.992.jpg), t=39.992s -- **car** (medium) [1225, 590, 1425, 650]: dark car, labeled 'ID31' by overlay
- frame 2486 (test7_frame_002486_t041.494.jpg), t=41.494s -- **car** (medium) [70, 590, 300, 680]: dark SUV, 'ID27 stationary'
- frame 2486 (test7_frame_002486_t041.494.jpg), t=41.494s -- **person** (low) [280, 605, 330, 670]: small figure beside the dark SUV -- possibly a pedestrian near the vehicle, verify
- frame 2576 (test7_frame_002576_t042.996.jpg), t=42.996s -- **car** (medium) [90, 595, 320, 685]: dark SUV, 'ID27'
- frame 2666 (test7_frame_002666_t044.498.jpg), t=44.498s -- **car** (medium) [70, 605, 300, 685]: dark SUV, 'ID27'
- frame 2756 (test7_frame_002756_t046.001.jpg), t=46.001s -- **car** (medium) [15, 605, 255, 685]: dark SUV, 'ID27'
- frame 2846 (test7_frame_002846_t047.503.jpg), t=47.503s -- **car** (medium) [15, 610, 255, 690]: dark SUV, 'ID27'
- frame 2936 (test7_frame_002936_t049.005.jpg), t=49.005s -- **car** (medium) [10, 610, 250, 690]: dark SUV, 'ID27'
- frame 3026 (test7_frame_003026_t050.507.jpg), t=50.507s -- **car** (medium) [10, 610, 250, 690]: dark SUV, background
- frame 3026 (test7_frame_003026_t050.507.jpg), t=50.507s -- **car** (medium) [1215, 605, 1385, 660]: dark car, background 'ID31'
- frame 3115 (test7_frame_003115_t051.993.jpg), t=51.993s -- **car** (medium) [10, 610, 250, 690]: dark SUV, background
- frame 3205 (test7_frame_003205_t053.495.jpg), t=53.495s -- **car** (medium) [0, 630, 235, 705]: dark SUV, background
- frame 3295 (test7_frame_003295_t054.997.jpg), t=54.997s -- **car** (medium) [350, 615, 590, 670]: silver car, background
- frame 3295 (test7_frame_003295_t054.997.jpg), t=54.997s -- **car** (medium) [1200, 615, 1350, 670]: dark car, background
- frame 3385 (test7_frame_003385_t056.499.jpg), t=56.499s -- **car** (medium) [140, 615, 380, 770]: dark SUV, 'ID38'
- frame 3475 (test7_frame_003475_t058.002.jpg), t=58.002s -- **car** (medium) [15, 615, 280, 760]: dark SUV, 'ID44 moving'
- frame 3475 (test7_frame_003475_t058.002.jpg), t=58.002s -- **car** (medium) [345, 605, 565, 655]: white car, background
- frame 3475 (test7_frame_003475_t058.002.jpg), t=58.002s -- **car** (medium) [1235, 605, 1590, 670]: dark car, 'ID41 moving right'
- frame 3475 (test7_frame_003475_t058.002.jpg), t=58.002s -- **car** (medium) [800, 780, 1420, 960]: black sedan, foreground, close to camera
- frame 3565 (test7_frame_003565_t059.504.jpg), t=59.504s -- **car** (medium) [890, 605, 1070, 660]: dark car, background 'ID46'
- frame 3583 (test7_frame_003583_t059.804.jpg), t=59.804s -- **car** (medium) [890, 600, 1050, 660]: dark blue car, background
- frame 3655 (test7_frame_003655_t061.006.jpg), t=61.006s -- **car** (medium) [880, 595, 1090, 650]: dark car, background
- frame 3655 (test7_frame_003655_t061.006.jpg), t=61.006s -- **car** (medium) [485, 790, 1050, 960]: black sedan, foreground, close to camera
- frame 3685 (test7_frame_003685_t061.507.jpg), t=61.507s -- **car** (medium) [1610, 605, 1850, 670]: white SUV, moving, 'ID49'
- frame 3685 (test7_frame_003685_t061.507.jpg), t=61.507s -- **car** (medium) [1015, 790, 1655, 960]: black sedan, foreground, close to camera
- frame 3834 (test7_frame_003834_t063.994.jpg), t=63.994s -- **car** (medium) [1035, 670, 1290, 800]: dark SUV, moving, 'ID51'
- frame 3924 (test7_frame_003924_t065.496.jpg), t=65.496s -- **car** (medium) [1510, 605, 1750, 660]: dark car, 'ID54 moving right'
- frame 4014 (test7_frame_004014_t066.998.jpg), t=66.998s -- **car** (medium) [1200, 590, 1385, 655]: maroon car, background 'ID41 insufficient'
- frame 4014 (test7_frame_004014_t066.998.jpg), t=66.998s -- **car** (medium) [1300, 605, 1520, 660]: dark car, 'ID56'
- frame 4104 (test7_frame_004104_t068.500.jpg), t=68.5s -- **car** (medium) [1200, 615, 1440, 670]: dark car, moving, 'ID58'

## Summary

- Total review items (from HUMAN_REVIEW_QUEUE.md): 126
- Items folded into recurring-object groups: 55 (across 10 groups)
- Singleton items (ungrouped): 71
- Estimated number of human decisions remaining: 81 (10 group decisions + 71 singleton decisions, down from 126 independent items)

