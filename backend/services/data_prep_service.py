from __future__ import annotations

import logging
import os
from typing import Dict, List

import cv2

from backend.core.config import CONFIG
from backend.services.detection_service import DetectionService

logger = logging.getLogger(__name__)


class DataPrepService:
    def __init__(self):
        self.det = DetectionService()

    def extract_faces_from_frames(self, frames_dir: str, output_dir: str | None = None,
                                  min_face_size: int | None = None, padding: int = 10) -> int:
        output_dir = output_dir or os.path.join(CONFIG.tmp_dir, "detected_faces")
        os.makedirs(output_dir, exist_ok=True)
        min_face_size = min_face_size or CONFIG.min_face_size

        saved = 0
        for fname in sorted(os.listdir(frames_dir)):
            fpath = os.path.join(frames_dir, fname)
            if not os.path.isfile(fpath):
                continue
            frame = cv2.imread(fpath)
            if frame is None:
                continue
            boxes, scores, classids, kpts = self.det.detect(frame)
            for b in boxes:
                if b[2] < min_face_size or b[3] < min_face_size:
                    continue
                x, y, w, h = b.astype(int)
                x0 = max(0, x - padding)
                y0 = max(0, y - padding)
                x1 = min(frame.shape[1], x + w + padding)
                y1 = min(frame.shape[0], y + h + padding)
                face = frame[y0:y1, x0:x1]
                out = os.path.join(output_dir, f"face_{saved:05d}.jpg")
                cv2.imwrite(out, face)
                saved += 1
        return saved

    def add_person_images(self, person_name: str, images: List[str]) -> Dict[str, str]:
        """Save uploaded images under datasets/new_persons/<name>, then trigger feature update via existing logic."""
        person_dir = os.path.join(CONFIG.datasets_dir, "new_persons", person_name)
        os.makedirs(person_dir, exist_ok=True)
        saved_paths: Dict[str, str] = {}
        for i, src in enumerate(images):
            ext = os.path.splitext(src)[1] or ".jpg"
            dst = os.path.join(person_dir, f"{person_name}_{i}{ext}")
            # move/copy; to avoid heavy copies, read and save
            img = cv2.imread(src)
            if img is None:
                continue
            cv2.imwrite(dst, img)
            saved_paths[str(i)] = dst

        # call add new person pipeline to update features
        # Dynamically import the script with a space in its filename
        add_persons = None
        try:
            import importlib.util
            script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "add new person.py")
            if os.path.exists(script_path):
                spec = importlib.util.spec_from_file_location("add_new_person_space", script_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore
                    add_persons = getattr(mod, "add_persons", None)
        except Exception as e:
            logger.warning("Dynamic import failed for 'add new person.py': %s", e)
        if add_persons is None:
            try:
                from add_new_person import add_persons  # type: ignore
            except Exception as e:
                logger.warning("Could not import add_persons function: %s", e)
                return saved_paths

        backup_dir = os.path.join(CONFIG.datasets_dir, "backup")
        faces_save_dir = os.path.join(CONFIG.datasets_dir, "data")
        features_path = CONFIG.face_features_noext
        os.makedirs(backup_dir, exist_ok=True)
        os.makedirs(faces_save_dir, exist_ok=True)
        add_persons(backup_dir=backup_dir,
                    add_persons_dir=os.path.join(CONFIG.datasets_dir, "new_persons"),
                    faces_save_dir=faces_save_dir,
                    features_path=features_path)
        return saved_paths
