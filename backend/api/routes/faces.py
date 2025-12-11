from __future__ import annotations

import os
from typing import List

from fastapi import APIRouter, UploadFile, File, Form, Request
from urllib.parse import quote

from backend.core.config import CONFIG
from backend.services.data_prep_service import DataPrepService

router = APIRouter()


@router.post("/add-person")
async def add_person(name: str = Form(...), files: List[UploadFile] = File(...)):
    # Save uploads to tmp, then pass paths to service
    os.makedirs(CONFIG.tmp_dir, exist_ok=True)
    paths: List[str] = []
    for i, f in enumerate(files):
        ext = os.path.splitext(f.filename)[1] or ".jpg"
        dst = os.path.join(CONFIG.tmp_dir, f"{name}_{i}{ext}")
        content = await f.read()
        with open(dst, "wb") as out:
            out.write(content)
        paths.append(dst)

    svc = DataPrepService()
    saved_map = svc.add_person_images(name, paths)

    # cleanup temp
    for p in paths:
        try:
            os.remove(p)
        except Exception:
            pass

    return {"saved": saved_map}

@router.get("/list")
def list_faces(request: Request):
    data_dir = os.path.join(CONFIG.datasets_dir, "data")
    result = {}
    if os.path.exists(data_dir):
        for person in os.listdir(data_dir):
            pdir = os.path.join(data_dir, person)
            if os.path.isdir(pdir):
                imgs = [
                    os.path.join(pdir, f)
                    for f in os.listdir(pdir)
                    if os.path.isfile(os.path.join(pdir, f))
                ]
                # Convert filesystem paths to HTTP URLs served under /static/images/{person}/{file}
                urls = []
                base = str(request.base_url).rstrip("/")
                for p in imgs[:20]:
                    fname = os.path.basename(p)
                    url = f"{base}/static/images/{quote(person)}/{quote(fname)}"
                    urls.append(url)
                result[person] = urls
    return result
