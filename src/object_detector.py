"""YOLO-based object detection for Atlas Phase 1.

Loads a small pretrained YOLO model once and produces filtered Detection
objects. Contains no drawing/visualization code.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from src.models import BoundingBox, Detection
from src.region_analyzer import RegionAnalyzer


class ObjectDetector:
    """Runs YOLO inference and returns filtered, region-tagged detections.

    Args:
        model_name: Ultralytics model name or path (e.g. "yolov8n.pt").
        confidence_threshold: Minimum confidence score to keep a detection.
        allowed_classes: Mapping of COCO class ID -> class name to keep.
            Any detection whose class ID is not a key in this mapping is
            discarded.
        region_analyzer: RegionAnalyzer used to tag each detection's
            horizontal region.
    """

    def __init__(
        self,
        model_name: str,
        confidence_threshold: float,
        allowed_classes: dict[int, str],
        region_analyzer: RegionAnalyzer,
    ) -> None:
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                "confidence_threshold must be within [0, 1], got "
                f"{confidence_threshold}"
            )
        self._model = YOLO(model_name)
        self._confidence_threshold = confidence_threshold
        self._allowed_classes = allowed_classes
        self._region_analyzer = region_analyzer

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single frame.

        Args:
            frame: BGR image as a numpy array (as returned by OpenCV).

        Returns:
            List of Detection objects for allowed classes above the
            confidence threshold, each tagged with its region.
        """
        frame_height, frame_width = frame.shape[:2]
        results = self._model(frame, verbose=False)

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])
                if class_id not in self._allowed_classes:
                    continue

                confidence = float(box.conf[0])
                if confidence < self._confidence_threshold:
                    continue

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                center_x, center_y = bbox.center
                region = self._region_analyzer.classify_point(
                    center_x, frame_width
                )

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=self._allowed_classes[class_id],
                        confidence=confidence,
                        bbox=bbox,
                        center=(center_x, center_y),
                        region=region,
                    )
                )

        return detections
