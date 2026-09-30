"""Authentication routes: Google sign-in (production) and optional dev login (local only)."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.database import get_db
from app.dependencies import get_current_teacher
from app.schemas import AuthConfig, DevLoginRequest, GoogleLoginRequest, LoginResponse, TeacherOut
from app.security import create_access_token, verify_google_id_token

router = APIRouter(prefix="/auth", tags=["auth"])


def _demo_enabled() -> bool:
    return settings.DEMO_MODE and not settings.is_production


@router.get("/config", response_model=AuthConfig)
def auth_config():
    """Public, non-secret values the frontend needs to render the Google button."""
    return AuthConfig(
        google_client_id=settings.GOOGLE_CLIENT_ID,
        demo_mode=_demo_enabled(),
        server_inference=settings.INFERENCE_MODE.lower() == "local",
    )


@router.post("/google", response_model=LoginResponse)
def google_login(payload: GoogleLoginRequest, db: Session = Depends(get_db)):
    claims = verify_google_id_token(payload.id_token)
    google_sub = claims["sub"]
    email = claims["email"].lower()
    name = claims.get("name") or email.split("@")[0]
    picture = claims.get("picture")

    teacher = db.query(models.Teacher).filter(models.Teacher.google_sub == google_sub).first()
    if teacher is None:
        # First Google sign-in for this email: adopt an existing record (e.g. data migrated
        # from the old SQLite database, or a dev account) instead of creating a duplicate.
        teacher = db.query(models.Teacher).filter(models.Teacher.email == email).first()
        if teacher is not None:
            teacher.google_sub = google_sub

    if teacher is None:
        teacher = models.Teacher(google_sub=google_sub, email=email, name=name, picture=picture)
        db.add(teacher)
    else:
        teacher.name = name
        teacher.picture = picture or teacher.picture

    try:
        db.commit()
    except IntegrityError:  # two simultaneous first logins
        db.rollback()
        teacher = db.query(models.Teacher).filter(models.Teacher.google_sub == google_sub).first()
        if teacher is None:
            raise HTTPException(status.HTTP_409_CONFLICT, "Could not create account, please retry")
    db.refresh(teacher)

    return LoginResponse(access_token=create_access_token(str(teacher.id)), teacher=TeacherOut.model_validate(teacher))


@router.post("/dev-login", response_model=LoginResponse)
def dev_login(payload: DevLoginRequest = DevLoginRequest(), db: Session = Depends(get_db)):
    """Local development only. Returns 404 unless DEMO_MODE=true and ENVIRONMENT != production."""
    if not _demo_enabled():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")

    google_sub = f"dev-sub-{payload.email}"
    teacher = db.query(models.Teacher).filter(models.Teacher.google_sub == google_sub).first()
    if not teacher:
        teacher = models.Teacher(google_sub=google_sub, email=payload.email, name=payload.name, picture=None)
        db.add(teacher)
        db.commit()
        db.refresh(teacher)
    return LoginResponse(access_token=create_access_token(str(teacher.id)), teacher=TeacherOut.model_validate(teacher))


@router.get("/me", response_model=TeacherOut)
def me(teacher: models.Teacher = Depends(get_current_teacher)):
    return teacher
