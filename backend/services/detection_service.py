from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from backend.core.config import CONFIG

logger = logging.getLogger(__name__)


class DetectionService:
    def __init__(self,
                 onnx_path: str | None = None,
                 conf_thres: float | None = None,
                 iou_thres: float | None = None):
        try:
            # Import the existing detector implementation
            from yolov8_face import YOLOv8_face  # type: ignore
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to import yolov8_face: %s", e)
            raise

        self.onnx_path = onnx_path or CONFIG.yolov8_onnx
        self.conf_thres = conf_thres or CONFIG.detection_conf
        self.iou_thres = iou_thres or CONFIG.detection_iou

        logger.info("Loading YOLOv8-face ONNX from %s", self.onnx_path)
        self.detector = YOLOv8_face(self.onnx_path, self.conf_thres, self.iou_thres)

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Detect faces; returns (boxes xywh, scores, classids, landmarks)."""
        return self.detector.detect(frame_bgr)
