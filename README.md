# yolov8-face-landmarks-opencv-dnn

使用 OpenCV 部署 yolov8 检测人脸和关键点，包含 C++和 Python 两个版本的程序，只依赖 opencv 库就可以运行。

训练源码是https://github.com/derronqi/yolov8-face
如果想做车牌检测 4 个角点，那就把检测 5 个人脸关键点改成 4 个

此外，添加了人脸质量评估模型 fqa，需要结合人脸检测来使用，对应的程序是 main_fqa.py 和 main_fqa.cpp

---

# YuzCheck - Web Refactor

This repository now includes a modern web application split into a FastAPI backend and a React frontend, refactoring the existing Tkinter-based attendance system while preserving detection, recognition, attendance logging, and PDF/image processing.

## Backend (FastAPI)

- Entry: `uvicorn backend.app:app --reload`
- Code: `backend/` with services wrapping existing modules:
  - Detection: wraps `yolov8_face.py` (YOLOv8-face ONNX)
  - Recognition: wraps `face_recognition_class.py` (ArcFace r100)
  - Attendance: video processing + Excel logging + absentees
  - PDF: extract faces from PDFs (PyMuPDF)
  - Data Prep: extract faces and add new persons (feature updates)

### Install

```bash
cd "YuzCheck"
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### Run

```bash
uvicorn backend.app:app --reload --port 8000
```

### Key Endpoints

- `POST /api/video/upload` – upload video
- `POST /api/attendance/run` – run detection+recognition over video
- `GET /api/attendance/summary` – present/absent summary
- `GET /api/students` – list students from `ogrenci_listesi_birlesik.xlsx`
- `POST /api/students/import` – import students (xlsx/csv)
- `POST /api/faces/add-person` – add person by uploading multiple images
- `GET /api/faces/list` – list existing faces

### Configuration

Set env vars to override defaults (optional):

- `YUZCHECK_YOLOV8_ONNX` (default `weights/yolov8n-face.onnx`)
- `YUZCHECK_ARCFACE_WEIGHTS` (default `face_recognition/arcface/weights/arcface_r100.pth`)
- `YUZCHECK_FACE_FEATURES` (default `datasets/face_features/feature` without .npz)
- `YUZCHECK_DET_CONF` (default 0.45), `YUZCHECK_DET_IOU` (0.3)
- `YUZCHECK_MIN_FACE` (default 60 px), `YUZCHECK_REC_MIN` (0.4)
- `YUZCHECK_COOLDOWN` (default 60 seconds)

Excel paths (created on demand): `backend/data/excel/attendance_log.xlsx` and `ogrenci_listesi_birlesik.xlsx` at repo root.

## Frontend (Vite + React)

### Install & Run

```bash
cd "YuzCheck/frontend"
npm install
npm run dev
```

Set API base (optional) in `.env`:

```
VITE_API_BASE=http://localhost:8000
```

### Pages

- Dashboard: attendance summary
- Attendance: upload video, run attendance, results
- Students: list/import students
- Faces: add person, list faces

---

Notes:

- Existing model weights and datasets are reused in place; no large files are moved.
- Excel logging is preserved and can be replaced later by a DB.
- The faces listing shows paths; serving actual images would require a static files endpoint.
