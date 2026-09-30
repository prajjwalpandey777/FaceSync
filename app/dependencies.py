"""FastAPI dependencies: current teacher + ownership checks."""
import uuid
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app import models
from app.database import get_db
from app.security import decode_access_token

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_teacher(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> models.Teacher:
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing Authorization Bearer token")
    teacher_id = decode_access_token(credentials.credentials)
    try:
        teacher_uuid = uuid.UUID(teacher_id)
    except ValueError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Malformed teacher ID")

    teacher = db.query(models.Teacher).filter(models.Teacher.id == teacher_uuid).first()
    if not teacher:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Teacher not found")
    return teacher


def get_owned_class(class_id: int, db: Session, teacher: models.Teacher) -> models.SchoolClass:
    school_class = (
        db.query(models.SchoolClass)
        .filter(models.SchoolClass.id == class_id, models.SchoolClass.teacher_id == teacher.id)
        .first()
    )
    if not school_class:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Class not found or access denied")
    return school_class


def get_owned_session(session_id: int, db: Session, teacher: models.Teacher) -> models.AttendanceSession:
    session = (
        db.query(models.AttendanceSession)
        .join(models.SchoolClass, models.AttendanceSession.class_id == models.SchoolClass.id)
        .filter(models.AttendanceSession.id == session_id, models.SchoolClass.teacher_id == teacher.id)
        .first()
    )
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Attendance session not found or access denied")
    return session
