from __future__ import annotations

import os
from fastapi import APIRouter, UploadFile, File

from backend.core.config import CONFIG
from backend.services.pdf_service import PDFService

router = APIRouter()


@router.post("/extract-faces")
async def extract_faces(file: UploadFile = File(...)):
    os.makedirs(CONFIG.tmp_dir, exist_ok=True)
    tmp_path = os.path.join(CONFIG.tmp_dir, file.filename)
    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    mapping = PDFService().extract_faces(tmp_path)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return {"mapping": mapping}
