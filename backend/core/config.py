import os
from pathlib import Path
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
DATA_DIR = BACKEND_DIR / "data"
TMP_DIR = DATA_DIR / "tmp"
WEIGHTS_DIR = ROOT / "weights"
DATASETS_DIR = ROOT / "datasets"
FACE_FEATURES_PATH_NOEXT = DATASETS_DIR / "face_features" / "feature"
EXCEL_DIR = DATA_DIR / "excel"


class AppConfig(BaseModel):
    # Model/weights
    yolov8_onnx: str = str(os.getenv("YUZCHECK_YOLOV8_ONNX", WEIGHTS_DIR / "yolov8n-face.onnx"))
    arcface_weights: str = str(
        os.getenv("YUZCHECK_ARCFACE_WEIGHTS", ROOT / "face_recognition" / "arcface" / "weights" / "arcface_r100.pth")
    )
    face_features_noext: str = str(os.getenv("YUZCHECK_FACE_FEATURES", FACE_FEATURES_PATH_NOEXT))

    # Thresholds
    detection_conf: float = float(os.getenv("YUZCHECK_DET_CONF", 0.45))
    detection_iou: float = float(os.getenv("YUZCHECK_DET_IOU", 0.3))
    min_face_size: int = int(os.getenv("YUZCHECK_MIN_FACE", 60))
    recognition_min_score: float = float(os.getenv("YUZCHECK_REC_MIN", 0.4))
    cooldown_seconds: int = int(os.getenv("YUZCHECK_COOLDOWN", 60))

    # Paths
    tmp_dir: str = str(TMP_DIR)
    datasets_dir: str = str(DATASETS_DIR)
    excel_dir: str = str(EXCEL_DIR)
    attendance_log_xlsx: str = str(EXCEL_DIR / "attendance_log.xlsx")
    students_xlsx: str = str(ROOT / "ogrenci_listesi_birlesik.xlsx")


def ensure_dirs():
    for d in [DATA_DIR, TMP_DIR, EXCEL_DIR]:
        os.makedirs(d, exist_ok=True)


CONFIG = AppConfig()
ensure_dirs()
