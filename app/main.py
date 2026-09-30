"""FaceSync API (FastAPI).

Deployed on Render (small plan is enough). Face recognition runs in the user's browser on their own
GPU/CPU, so this server only stores data in Supabase Postgres and matches 512-number embeddings.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app import models
from app.config import settings
from app.database import Base, engine, secure_postgres_tables
from app.dependencies import get_current_teacher
from app.face_service import prepare_image
from app.inference import get_inference
from app.routers import auth, classes, reports, sessions, students
from app.warmup import warm_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("facesync")

VERSION = "4.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_for_startup()          # refuse to boot with an unsafe production config
    Base.metadata.create_all(bind=engine)    # no-op when the tables already exist
    secure_postgres_tables()                 # keep biometric data off Supabase's public REST API
    if settings.INFERENCE_MODE.lower() == "local":
        get_inference()                      # optional server-side mode: load models now, not on first request
    log.info("FaceSync API %s started (env=%s, inference=%s)", VERSION, settings.ENVIRONMENT, settings.INFERENCE_MODE)
    yield
    log.info("FaceSync API shutting down")


app = FastAPI(
    title="FaceSync API",
    description="Face-recognition attendance API. Google sign-in, Supabase Postgres, on-device recognition.",
    version=VERSION,
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None,
    openapi_url=None if settings.is_production else "/openapi.json",
)

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return a short message and never echo the submitted values (they may contain NaN or 512-number
    face embeddings, which are neither JSON-safe nor something we want in error responses)."""
    parts = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(x) for x in err.get("loc", []) if x != "body")
        parts.append(f"{loc}: {err.get('msg', 'invalid value')}" if loc else str(err.get("msg", "invalid value")))
    return JSONResponse(status_code=422, content={"detail": "; ".join(parts) or "Invalid request"})


# Auth uses a Bearer header (not cookies), so credentialed CORS is unnecessary and stays off.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.FRONTEND_ORIGIN_REGEX or None,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

app.include_router(auth.router)
app.include_router(classes.router)
app.include_router(students.router)
app.include_router(sessions.router)
app.include_router(reports.router)


@app.get("/health")
def health():
    """Cheap liveness probe (used by Render and by the frontend to wake the server)."""
    return {"status": "ok", "service": "FaceSync API", "version": VERSION}


@app.get("/health/db")
def health_db():
    """Deeper check: can we reach the database?"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        log.error("Database health check failed: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    return {"status": "ok", "database": "up"}


@app.post("/warmup", status_code=202)
def warmup(background: BackgroundTasks, _: models.Teacher = Depends(get_current_teacher)):
    """Signed-in users can call this to ensure server-side models are loaded (no-op in 'device' mode)."""
    background.add_task(warm_inference)
    return {"status": "ready", "inference_mode": settings.INFERENCE_MODE}


@app.get("/diagnostics")
def diagnostics(_: models.Teacher = Depends(get_current_teacher)):
    """Inference backend details (device, models loaded). Requires sign-in."""
    if settings.INFERENCE_MODE.lower() != "local":
        return {"inference_mode": "device", "inference": {"where": "user's browser (WebGPU/WebAssembly)"}}
    return {"inference_mode": settings.INFERENCE_MODE, "inference": get_inference().health()}


@app.post("/api/detect-face")
def detect_face_endpoint(
    frame: UploadFile = File(...),
    _: models.Teacher = Depends(get_current_teacher),
):
    """Server-side fallback only (INFERENCE_MODE=local): face boxes + landmarks for a frame."""
    image_bytes = prepare_image(frame.file.read(settings.MAX_UPLOAD_MB * 1024 * 1024 + 1))
    result = get_inference().analyze(image_bytes, embed=False, liveness=False)
    return {
        "faces": [
            {"norm_bbox": list(f.norm_bbox), "landmarks": f.landmarks, "score": round(f.score, 3)}
            for f in result.faces
        ],
        "latency_ms": result.latency_ms,
        "frame_width": result.frame_width,
        "frame_height": result.frame_height,
    }


# Optionally serve the static frontend (local development or single-server deploy).
frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
if settings.SERVE_FRONTEND and (frontend_dir / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
