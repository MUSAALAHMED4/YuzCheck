import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.core.logging_config import configure_logging
from backend.api.routes.attendance import router as attendance_router
from backend.api.routes.students import router as students_router
from backend.api.routes.faces import router as faces_router
from backend.api.routes.utils import router as utils_router
from backend.api.routes.pdf import router as pdf_router
from backend.core.config import CONFIG


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="YuzCheck API", version="1.0.0")

    # CORS: allow local dev frontends
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(utils_router, prefix="/api")
    app.include_router(attendance_router, prefix="/api/attendance", tags=["attendance"])
    app.include_router(students_router, prefix="/api/students", tags=["students"])
    app.include_router(faces_router, prefix="/api/faces", tags=["faces"])
    app.include_router(pdf_router, prefix="/api/pdf", tags=["pdf"])

    # Static: serve dataset images for frontend previews
    images_dir = os.path.join(CONFIG.datasets_dir, "data")
    if os.path.isdir(images_dir):
        app.mount(
            "/static/images",
            StaticFiles(directory=images_dir, html=False),
            name="images",
        )

    # Static: serve temporary files and detected faces
    if os.path.isdir(CONFIG.tmp_dir):
        app.mount(
            "/static/tmp",
            StaticFiles(directory=CONFIG.tmp_dir, html=False),
            name="tmp",
        )

    detected_dir = os.path.join(os.path.dirname(CONFIG.tmp_dir), "detected_faces")
    if os.path.isdir(detected_dir):
        app.mount(
            "/static/detected",
            StaticFiles(directory=detected_dir, html=False),
            name="detected",
        )

    return app


app = create_app()
