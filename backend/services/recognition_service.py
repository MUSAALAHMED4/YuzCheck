from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from backend.core.config import CONFIG

logger = logging.getLogger(__name__)


class RecognitionService:
    def __init__(self,
                 model_name: str = "r100",
                 model_path: str | None = None,
                 features_noext: str | None = None,
                 input_size: int = 112,
                 min_score: float | None = None):
        try:
            from face_recognition_class import FaceRecognitionClass  # type: ignore
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to import FaceRecognitionClass: %s", e)
            raise

        self.model_path = model_path or CONFIG.arcface_weights
        self.features_noext = features_noext or CONFIG.face_features_noext
        self.min_score = min_score or CONFIG.recognition_min_score

        logger.info("Loading ArcFace model from %s", self.model_path)
        self.recognizer = FaceRecognitionClass(
            model_name=model_name,
            model_path=self.model_path,
            feature_path=self.features_noext,
            input_size=input_size,
            min_score=self.min_score,
        )

    def recognize(self, face_bgr: np.ndarray) -> Tuple[float | None, str]:
        return self.recognizer.recognize_face(face_bgr)
