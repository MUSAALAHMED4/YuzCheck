from typing import List, Optional
from pydantic import BaseModel


class VideoUploadResponse(BaseModel):
    id: str
    path: str


class AttendanceRunRequest(BaseModel):
    video_id: Optional[str] = None
    video_path: Optional[str] = None
    detection_conf: Optional[float] = None
    detection_iou: Optional[float] = None
    min_face_size: Optional[int] = None
    recognition_min_score: Optional[float] = None
    cooldown_seconds: Optional[int] = None


class RecognizedEntry(BaseModel):
    name: str
    timestamp: str


class AttendanceRunResult(BaseModel):
    recognized: List[RecognizedEntry]
    present: List[str]
    absent: List[str]


class AttendanceSummary(BaseModel):
    total: int
    present: List[str]
    absent: List[str]


class AttendanceLiveRequest(BaseModel):
    detection_conf: Optional[float] = None
    detection_iou: Optional[float] = None
    min_face_size: Optional[int] = None
    recognition_min_score: Optional[float] = None
    cooldown_seconds: Optional[int] = None
    max_seconds: Optional[int] = 30
