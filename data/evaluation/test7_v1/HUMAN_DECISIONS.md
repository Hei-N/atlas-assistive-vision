# Test7 AI-Assisted Draft -- Human Decisions

One decision entry per recurring group (R01-R10) and one per singleton item (S001-S071), transcribed from the ChatGPT-assisted review of `chatgpt_review_package.zip` and applied to build `annotations_candidate.json` (NOT yet merged into the canonical `annotations.json` -- that promotion is a separate, not-yet-approved step).

R01:
Decision: [x]
Notes: KEEP bicycle. Rider visually looks like a pedal bicycle, not a motorcycle. Existing visibility/occlusion metadata preserved as-is (no new difficult override added).

R02:
Decision: [x]
Notes: CHANGE truck -> car. Cargo/full-size commercial van; canonical mapping is van->car. Applied only to the 3 annotations representing this van (not a global truck->car change).

R03:
Decision: [x]
Notes: KEEP car + DIFFICULT. SUV/crossover/passenger-vehicle shape; heavy pole/sign occlusion.

R04:
Decision: [x]
Notes: KEEP car. Boxy SUV/crossover remains canonical car.

R05:
Decision: [x]
Notes: KEEP car. Minivan maps to canonical car. Class decision only -- no cross-frame re-identification encoded in the schema.

R06:
Decision: [x]
Notes: KEEP truck + DIFFICULT. Pickup-bed/body evidence supports truck; heavy left-edge crop. Applied only to these 5 annotations, not a global car->truck change.

R07:
Decision: [x]
Notes: IGNORE + DIFFICULT. Tree-occluded area may contain multiple distinct vehicles; not scored as one merged car. Clean, separately-boxed vehicles elsewhere in these frames are untouched.

R08:
Decision: [x]
Notes: IGNORE + DIFFICULT. Overlapping background vehicles cannot be reliably separated; not scored as one merged car. Clean, separately-boxed vehicles elsewhere in these frames are untouched.

R09:
Decision: [x]
Notes: KEEP car + DIFFICULT. Single SUV/crossover-like vehicle, heavily cropped at the left edge.

R10:
Decision: [x]
Notes: KEEP person.

S001:
Frame: 3295 (test7_frame_003295_t054.997.jpg, t=54.997s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S002:
Frame: 2127 (test7_frame_002127_t035.502.jpg, t=35.502s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S003:
Frame: 2217 (test7_frame_002217_t037.004.jpg, t=37.004s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S004:
Frame: 2217 (test7_frame_002217_t037.004.jpg, t=37.004s)
Current draft class: car
Decision: [x]
Notes: IGNORE + DIFFICULT (too little of the object visible / too ambiguous for reliable scored ground truth).

S005:
Frame: 2396 (test7_frame_002396_t039.992.jpg, t=39.992s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S006:
Frame: 3205 (test7_frame_003205_t053.495.jpg, t=53.495s)
Current draft class: car
Decision: [x]
Notes: IGNORE + DIFFICULT (too little of the object visible / too ambiguous for reliable scored ground truth).

S007:
Frame: 3385 (test7_frame_003385_t056.499.jpg, t=56.499s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S008:
Frame: 3385 (test7_frame_003385_t056.499.jpg, t=56.499s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S009:
Frame: 3385 (test7_frame_003385_t056.499.jpg, t=56.499s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S010:
Frame: 3475 (test7_frame_003475_t058.002.jpg, t=58.002s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S011:
Frame: 3834 (test7_frame_003834_t063.994.jpg, t=63.994s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S012:
Frame: 4014 (test7_frame_004014_t066.998.jpg, t=66.998s)
Current draft class: car
Decision: [x]
Notes: IGNORE + DIFFICULT (too little of the object visible / too ambiguous for reliable scored ground truth).

S013:
Frame: 4104 (test7_frame_004104_t068.500.jpg, t=68.5s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S014:
Frame: 2127 (test7_frame_002127_t035.502.jpg, t=35.502s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S015:
Frame: 2217 (test7_frame_002217_t037.004.jpg, t=37.004s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S016:
Frame: 2307 (test7_frame_002307_t038.506.jpg, t=38.506s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S017:
Frame: 2396 (test7_frame_002396_t039.992.jpg, t=39.992s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S018:
Frame: 2486 (test7_frame_002486_t041.494.jpg, t=41.494s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S019:
Frame: 2576 (test7_frame_002576_t042.996.jpg, t=42.996s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S020:
Frame: 2666 (test7_frame_002666_t044.498.jpg, t=44.498s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S021:
Frame: 2756 (test7_frame_002756_t046.001.jpg, t=46.001s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S022:
Frame: 2846 (test7_frame_002846_t047.503.jpg, t=47.503s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S023:
Frame: 2936 (test7_frame_002936_t049.005.jpg, t=49.005s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S024:
Frame: 3026 (test7_frame_003026_t050.507.jpg, t=50.507s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S025:
Frame: 3745 (test7_frame_003745_t062.508.jpg, t=62.508s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S026:
Frame: 3745 (test7_frame_003745_t062.508.jpg, t=62.508s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S027:
Frame: 1408 (test7_frame_001408_t023.501.jpg, t=23.501s)
Current draft class: car
Decision: [x]
Notes: IGNORE + DIFFICULT (too little of the object visible / too ambiguous for reliable scored ground truth).

S028:
Frame: 1498 (test7_frame_001498_t025.003.jpg, t=25.003s)
Current draft class: car
Decision: [x]
Notes: IGNORE + DIFFICULT (too little of the object visible / too ambiguous for reliable scored ground truth).

S029:
Frame: 1588 (test7_frame_001588_t026.505.jpg, t=26.505s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S030:
Frame: 1678 (test7_frame_001678_t028.008.jpg, t=28.008s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S031:
Frame: 1767 (test7_frame_001767_t029.493.jpg, t=29.493s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S032:
Frame: 1767 (test7_frame_001767_t029.493.jpg, t=29.493s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S033:
Frame: 1857 (test7_frame_001857_t030.995.jpg, t=30.995s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S034:
Frame: 1947 (test7_frame_001947_t032.498.jpg, t=32.498s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S035:
Frame: 1947 (test7_frame_001947_t032.498.jpg, t=32.498s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S036:
Frame: 2037 (test7_frame_002037_t034.000.jpg, t=34.0s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S037:
Frame: 2037 (test7_frame_002037_t034.000.jpg, t=34.0s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S038:
Frame: 2127 (test7_frame_002127_t035.502.jpg, t=35.502s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S039:
Frame: 2127 (test7_frame_002127_t035.502.jpg, t=35.502s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S040:
Frame: 2307 (test7_frame_002307_t038.506.jpg, t=38.506s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S041:
Frame: 2307 (test7_frame_002307_t038.506.jpg, t=38.506s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S042:
Frame: 2396 (test7_frame_002396_t039.992.jpg, t=39.992s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S043:
Frame: 2486 (test7_frame_002486_t041.494.jpg, t=41.494s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S044:
Frame: 2486 (test7_frame_002486_t041.494.jpg, t=41.494s)
Current draft class: person
Decision: [x]
Notes: KEEP person + DIFFICULT.

S045:
Frame: 2576 (test7_frame_002576_t042.996.jpg, t=42.996s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S046:
Frame: 2666 (test7_frame_002666_t044.498.jpg, t=44.498s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S047:
Frame: 2756 (test7_frame_002756_t046.001.jpg, t=46.001s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S048:
Frame: 2846 (test7_frame_002846_t047.503.jpg, t=47.503s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S049:
Frame: 2936 (test7_frame_002936_t049.005.jpg, t=49.005s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S050:
Frame: 3026 (test7_frame_003026_t050.507.jpg, t=50.507s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S051:
Frame: 3026 (test7_frame_003026_t050.507.jpg, t=50.507s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S052:
Frame: 3115 (test7_frame_003115_t051.993.jpg, t=51.993s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S053:
Frame: 3205 (test7_frame_003205_t053.495.jpg, t=53.495s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S054:
Frame: 3295 (test7_frame_003295_t054.997.jpg, t=54.997s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S055:
Frame: 3295 (test7_frame_003295_t054.997.jpg, t=54.997s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S056:
Frame: 3385 (test7_frame_003385_t056.499.jpg, t=56.499s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S057:
Frame: 3475 (test7_frame_003475_t058.002.jpg, t=58.002s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S058:
Frame: 3475 (test7_frame_003475_t058.002.jpg, t=58.002s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S059:
Frame: 3475 (test7_frame_003475_t058.002.jpg, t=58.002s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S060:
Frame: 3475 (test7_frame_003475_t058.002.jpg, t=58.002s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S061:
Frame: 3565 (test7_frame_003565_t059.504.jpg, t=59.504s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S062:
Frame: 3583 (test7_frame_003583_t059.804.jpg, t=59.804s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S063:
Frame: 3655 (test7_frame_003655_t061.006.jpg, t=61.006s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S064:
Frame: 3655 (test7_frame_003655_t061.006.jpg, t=61.006s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S065:
Frame: 3685 (test7_frame_003685_t061.507.jpg, t=61.507s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S066:
Frame: 3685 (test7_frame_003685_t061.507.jpg, t=61.507s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S067:
Frame: 3834 (test7_frame_003834_t063.994.jpg, t=63.994s)
Current draft class: car
Decision: [x]
Notes: KEEP car.

S068:
Frame: 3924 (test7_frame_003924_t065.496.jpg, t=65.496s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S069:
Frame: 4014 (test7_frame_004014_t066.998.jpg, t=66.998s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S070:
Frame: 4014 (test7_frame_004014_t066.998.jpg, t=66.998s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

S071:
Frame: 4104 (test7_frame_004104_t068.500.jpg, t=68.5s)
Current draft class: car
Decision: [x]
Notes: KEEP car + DIFFICULT.

