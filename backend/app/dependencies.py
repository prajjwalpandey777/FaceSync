import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.database import get_db
from app.security import decode_access_token
from app import models

bearer_scheme = HTTPBearer()


def get_current_teacher(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> models.Teacher:
    teacher_id = decode_access_token(credentials.credentials)
    try:
        teacher_uuid = uuid.UUID(teacher_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session token")

    teacher = db.query(models.Teacher).filter(models.Teacher.id == teacher_uuid).first()
    if not teacher:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Teacher not found")
    return teacher


def get_owned_class(class_id: int, db: Session, teacher: models.Teacher) -> models.SchoolClass:
    """Fetch a class and 404 if it doesn't exist or doesn't belong to this teacher."""
    school_class = (
        db.query(models.SchoolClass)
        .filter(models.SchoolClass.id == class_id, models.SchoolClass.teacher_id == teacher.id)
        .first()
    )
    if not school_class:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")
    return school_class
