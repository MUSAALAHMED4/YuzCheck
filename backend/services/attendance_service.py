from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from backend.core.config import CONFIG
from backend.services.detection_service import DetectionService
from backend.services.recognition_service import RecognitionService

logger = logging.getLogger(__name__)


class AttendanceService:
    def __init__(self,
                 detection: DetectionService | None = None,
                 recognition: RecognitionService | None = None):
        self.det = detection or DetectionService()
        self.rec = recognition or RecognitionService()

    def _log_to_excel(self, name: str, excel_file: str, last_logged: Dict[str, datetime], cooldown_s: int) -> bool:
        now = datetime.now()
        last = last_logged.get(name)
        if last and (now - last) < timedelta(seconds=cooldown_s):
            return False
        last_logged[name] = now
        ts = now.strftime('%Y-%m-%d %H:%M:%S')
        data = {"Name": [name], "Timestamp": [ts]}
        try:
            if excel_file and excel_file.strip():
                if excel_file and os.path.exists(excel_file):
                    existing = pd.read_excel(excel_file)
                    updated = pd.concat([existing, pd.DataFrame(data)], ignore_index=True)
                else:
                    updated = pd.DataFrame(data)
                updated.to_excel(excel_file, index=False)
        except Exception as e:  # non-fatal
            logger.warning("Failed to write Excel log: %s", e)
        return True

    def run(self,
            video_path: str,
            detection_conf: float | None = None,
            detection_iou: float | None = None,
            min_face_size: int | None = None,
            recognition_min_score: float | None = None,
            cooldown_seconds: int | None = None) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
        import os
        assert os.path.exists(video_path), f"Video not found: {video_path}"

        # override config if provided
        if detection_conf is not None or detection_iou is not None:
            self.det = DetectionService(
                conf_thres=detection_conf or CONFIG.detection_conf,
                iou_thres=detection_iou or CONFIG.detection_iou,
            )
        if recognition_min_score is not None:
            self.rec = RecognitionService(min_score=recognition_min_score)

        min_face_size = min_face_size or CONFIG.min_face_size
        cooldown_seconds = cooldown_seconds or CONFIG.cooldown_seconds

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        last_logged_times: Dict[str, datetime] = {}
        recognized: List[Tuple[str, str]] = []  # (name, timestamp)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                boxes, scores, classids, kpts = self.det.detect(frame)
                if len(boxes) == 0:
                    continue

                # Filter small faces
                filtered = []
                for b in boxes:
                    if b[2] >= min_face_size and b[3] >= min_face_size:
                        filtered.append(b)

                for box in filtered:
                    x, y, w, h = box.astype(int)
                    face = frame[y:y+h, x:x+w]
                    score, name = self.rec.recognize(face)
                    if score is None:
                        continue
                    if score >= (self.rec.recognizer.min_score if hasattr(self.rec, 'recognizer') else CONFIG.recognition_min_score):
                        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # Cooldown logging
                        if name not in last_logged_times or (datetime.now() - last_logged_times[name]).seconds > cooldown_seconds:
                            try:
                                # write Excel
                                excel_file = CONFIG.attendance_log_xlsx
                                # ensure parent dir exists
                                os.makedirs(os.path.dirname(excel_file), exist_ok=True)
                                if os.path.exists(excel_file):
                                    existing = pd.read_excel(excel_file)
                                    dfnew = pd.concat([existing, pd.DataFrame({"Name": [name], "Timestamp": [ts]})], ignore_index=True)
                                else:
                                    dfnew = pd.DataFrame({"Name": [name], "Timestamp": [ts]})
                                dfnew.to_excel(excel_file, index=False)
                            except Exception as e:
                                logger.warning("Excel write failed: %s", e)
                            last_logged_times[name] = datetime.now()
                        recognized.append((name, ts))
        finally:
            cap.release()

        # Compute absentees
        present_names = sorted(set([n for n, _ in recognized]))
        expected_names: List[str] = []
        try:
            if CONFIG.students_xlsx and os.path.exists(CONFIG.students_xlsx):
                df = pd.read_excel(CONFIG.students_xlsx)
                if 'Student Name' in df.columns:
                    expected_names = df['Student Name'].dropna().astype(str).tolist()
        except Exception as e:
            logger.warning("Failed reading students list: %s", e)

        absent = sorted(list(set(expected_names) - set(present_names))) if expected_names else []
        return recognized, present_names, absent

    def run_live(self,
                 detection_conf: float | None = None,
                 detection_iou: float | None = None,
                 min_face_size: int | None = None,
                 recognition_min_score: float | None = None,
                 cooldown_seconds: int | None = None,
                 max_seconds: int | None = 30) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
        import time
        import os

        # override config if provided
        if detection_conf is not None or detection_iou is not None:
            self.det = DetectionService(
                conf_thres=detection_conf or CONFIG.detection_conf,
                iou_thres=detection_iou or CONFIG.detection_iou,
            )
        if recognition_min_score is not None:
            self.rec = RecognitionService(min_score=recognition_min_score)

        min_face_size = min_face_size or CONFIG.min_face_size
        cooldown_seconds = cooldown_seconds or CONFIG.cooldown_seconds
        max_seconds = max_seconds or 30

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0)")

        start_time = time.time()
        last_logged_times: Dict[str, datetime] = {}
        recognized: List[Tuple[str, str]] = []

        try:
            while True:
                if (time.time() - start_time) >= max_seconds:
                    break
                ret, frame = cap.read()
                if not ret:
                    continue

                boxes, scores, classids, kpts = self.det.detect(frame)
                if len(boxes) == 0:
                    continue

                # Filter small faces
                filtered = []
                for b in boxes:
                    if b[2] >= min_face_size and b[3] >= min_face_size:
                        filtered.append(b)

                for box in filtered:
                    x, y, w, h = box.astype(int)
                    face = frame[y:y+h, x:x+w]
                    score, name = self.rec.recognize(face)
                    if score is None:
                        continue
                    if score >= (self.rec.recognizer.min_score if hasattr(self.rec, 'recognizer') else CONFIG.recognition_min_score):
                        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # Cooldown logging
                        if name not in last_logged_times or (datetime.now() - last_logged_times[name]).seconds > cooldown_seconds:
                            try:
                                excel_file = CONFIG.attendance_log_xlsx
                                os.makedirs(os.path.dirname(excel_file), exist_ok=True)
                                if os.path.exists(excel_file):
                                    existing = pd.read_excel(excel_file)
                                    dfnew = pd.concat([existing, pd.DataFrame({"Name": [name], "Timestamp": [ts]})], ignore_index=True)
                                else:
                                    dfnew = pd.DataFrame({"Name": [name], "Timestamp": [ts]})
                                dfnew.to_excel(excel_file, index=False)
                            except Exception as e:
                                logger.warning("Excel write failed: %s", e)
                            last_logged_times[name] = datetime.now()
                        recognized.append((name, ts))
        finally:
            cap.release()

        present_names = sorted(set([n for n, _ in recognized]))
        expected_names: List[str] = []
        try:
            if CONFIG.students_xlsx and os.path.exists(CONFIG.students_xlsx):
                df = pd.read_excel(CONFIG.students_xlsx)
                if 'Student Name' in df.columns:
                    expected_names = df['Student Name'].dropna().astype(str).tolist()
        except Exception as e:
            logger.warning("Failed reading students list: %s", e)

        absent = sorted(list(set(expected_names) - set(present_names))) if expected_names else []
        return recognized, present_names, absent
