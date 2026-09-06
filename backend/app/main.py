from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import Base, engine
from app.routers import auth, classes, students, sessions, reports

app = FastAPI(title="FaceSync API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(classes.router)
app.include_router(students.router)
app.include_router(sessions.router)
app.include_router(reports.router)


@app.on_event("startup")
def on_startup():
    # Convenience for first deploy — safe to leave in, it only creates tables
    # that don't already exist. supabase_schema.sql is the source of truth.
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"status": "ok"}
