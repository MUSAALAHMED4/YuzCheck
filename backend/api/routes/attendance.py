from __future__ import annotations

import os
import logging
from typing import List

from fastapi import APIRouter, HTTPException

from backend.core.config import CONFIG
from backend.models.attendance_models import (
    AttendanceRunRequest,
    AttendanceRunResult,
    AttendanceSummary,
    RecognizedEntry,
    AttendanceLiveRequest,
)
from backend.services.attendance_service import AttendanceService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/run", response_model=AttendanceRunResult)
def run_attendance(req: AttendanceRunRequest) -> AttendanceRunResult:
    video_path = req.video_path
    if not video_path and req.video_id:
        # resolve id to path in tmp
        # The file name is unknown; scan for first matching prefix
        candidates = [f for f in os.listdir(CONFIG.tmp_dir) if f.startswith(req.video_id)]
        if not candidates:
            raise HTTPException(status_code=404, detail="Video not found for id")
        video_path = os.path.join(CONFIG.tmp_dir, candidates[0])

    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail="Valid video_path or video_id required")

    service = AttendanceService()
    recognized, present, absent = service.run(
        video_path=video_path,
        detection_conf=req.detection_conf,
        detection_iou=req.detection_iou,
        min_face_size=req.min_face_size,
        recognition_min_score=req.recognition_min_score,
        cooldown_seconds=req.cooldown_seconds,
    )

    rec_entries: List[RecognizedEntry] = [RecognizedEntry(name=n, timestamp=t) for n, t in recognized]
    return AttendanceRunResult(recognized=rec_entries, present=present, absent=absent)


@router.get("/summary", response_model=AttendanceSummary)
def get_summary() -> AttendanceSummary:
    import pandas as pd
    total = 0
    present: List[str] = []
    absent: List[str] = []
    try:
        if CONFIG.students_xlsx and os.path.exists(CONFIG.students_xlsx):
            df_s = pd.read_excel(CONFIG.students_xlsx)
            if 'Student Name' in df_s.columns:
                students = df_s['Student Name'].dropna().astype(str).tolist()
                total = len(students)
        else:
            students = []
        if os.path.exists(CONFIG.attendance_log_xlsx):
            df_a = pd.read_excel(CONFIG.attendance_log_xlsx)
            if 'Name' in df_a.columns:
                present = sorted(set(df_a['Name'].dropna().astype(str).tolist()))
        absent = sorted(list(set(students) - set(present))) if total else []
    except Exception:
        pass
    return AttendanceSummary(total=total, present=present, absent=absent)


@router.post("/live", response_model=AttendanceRunResult)
def run_attendance_live(req: AttendanceLiveRequest) -> AttendanceRunResult:
    service = AttendanceService()
    recognized, present, absent = service.run_live(
        detection_conf=req.detection_conf,
        detection_iou=req.detection_iou,
        min_face_size=req.min_face_size,
        recognition_min_score=req.recognition_min_score,
        cooldown_seconds=req.cooldown_seconds,
        max_seconds=req.max_seconds or 30,
    )

    rec_entries: List[RecognizedEntry] = [RecognizedEntry(name=n, timestamp=t) for n, t in recognized]
    return AttendanceRunResult(recognized=rec_entries, present=present, absent=absent)


@router.get("/latest")
def get_latest_recognition():
    """Get the latest recognized person from the attendance log"""
    import pandas as pd
    try:
        image_url = None
        detected_dir = os.path.join(os.path.dirname(CONFIG.tmp_dir), "detected_faces")
        face_path = os.path.join(detected_dir, "latest.jpg")
        if os.path.exists(face_path):
            image_url = "/static/detected/latest.jpg"
        if os.path.exists(CONFIG.attendance_log_xlsx):
            df = pd.read_excel(CONFIG.attendance_log_xlsx)
            if not df.empty and 'Name' in df.columns and 'Timestamp' in df.columns:
                # Get the last row
                latest = df.iloc[-1]
                return {
                    "name": str(latest['Name']),
                    "timestamp": str(latest['Timestamp']),
                    "image_url": image_url
                }
        return {"name": None, "timestamp": None, "image_url": image_url}
    except Exception as e:
        logger.error(f"Error reading latest recognition: {e}")
        return {"name": None, "timestamp": None, "image_url": None}
