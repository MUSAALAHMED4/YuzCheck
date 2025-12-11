from __future__ import annotations

import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.core.config import CONFIG
from backend.models.attendance_models import VideoUploadResponse

router = APIRouter()


@router.post("/video/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    ext = os.path.splitext(file.filename)[1]
    file_id = str(uuid.uuid4())
    out_path = os.path.join(CONFIG.tmp_dir, f"{file_id}{ext}")
    os.makedirs(CONFIG.tmp_dir, exist_ok=True)
    content = await file.read()
    with open(out_path, "wb") as f:
        f.write(content)
    return VideoUploadResponse(id=file_id, path=out_path)
