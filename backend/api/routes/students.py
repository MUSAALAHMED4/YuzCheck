from __future__ import annotations

import os
from typing import List

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.core.config import CONFIG
from backend.models.student_models import Student, StudentsResponse

router = APIRouter()


@router.get("", response_model=StudentsResponse)
def list_students() -> StudentsResponse:
    students: List[Student] = []
    if CONFIG.students_xlsx and os.path.exists(CONFIG.students_xlsx):
        df = pd.read_excel(CONFIG.students_xlsx)
        col = 'Student Name' if 'Student Name' in df.columns else df.columns[0]
        for name in df[col].dropna().astype(str).tolist():
            students.append(Student(name=name))
    return StudentsResponse(students=students)


@router.post("/import", response_model=StudentsResponse)
async def import_students(file: UploadFile = File(...)) -> StudentsResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    os.makedirs(os.path.dirname(CONFIG.students_xlsx), exist_ok=True)
    content = await file.read()
    tmp_path = os.path.join(CONFIG.tmp_dir, file.filename)
    with open(tmp_path, "wb") as f:
        f.write(content)
    # Save/convert to expected path
    try:
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(tmp_path)
        else:
            df = pd.read_excel(tmp_path)
        df.to_excel(CONFIG.students_xlsx, index=False)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return list_students()
