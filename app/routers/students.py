"""Student enrollment and management routes."""
import json
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, defer

from app import models
from app.config import settings
from app.database import get_db
from app.dependencies import get_current_teacher, get_owned_class
from app.face_service import encode_single_face, normalize_embedding
from app.schemas import StudentOut

router = APIRouter(prefix="/classes/{class_id}/students", tags=["students"])


@router.post("", response_model=StudentOut, status_code=status.HTTP_201_CREATED)
def enroll_student(
    class_id: int,
    name: str = Form(...),
    reg_no: str = Form(...),
    photo: UploadFile = File(..., description="One clear, front-facing photo of the student"),
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    # Sync handler: FastAPI runs it in a worker thread, so the (blocking) call to the GPU
    # service does not stall other requests.
    get_owned_class(class_id, db, teacher)

    name, reg_no = name.strip(), reg_no.strip()
    if not name or not reg_no:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Name and registration number are required")
    if not photo.content_type or not photo.content_type.startswith("image/"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file must be an image (JPEG or PNG)")

    if (
        db.query(models.Student.id)
        .filter(models.Student.class_id == class_id, models.Student.reg_no == reg_no)
        .first()
    ):
        # Fail before spending GPU time on a duplicate.
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"A student with registration number '{reg_no}' already exists in this class"
        )

    image_bytes = photo.file.read(settings.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    encoding = encode_single_face(image_bytes)

    student = models.Student(class_id=class_id, name=name, reg_no=reg_no, face_encoding=encoding, photo_url=None)
    db.add(student)
    try:
        db.commit()
        db.refresh(student)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"A student with registration number '{reg_no}' already exists in this class"
        )
    return student


@router.post("/from-embedding", response_model=StudentOut, status_code=status.HTTP_201_CREATED)
def enroll_student_from_embedding(
    class_id: int,
    name: str = Form(...),
    reg_no: str = Form(...),
    embedding: str = Form(..., description="JSON array of 512 numbers computed in the browser"),
    faces_detected: int = Form(1),
    is_real: bool = Form(True),
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    """Enrollment when the photo was analysed on the teacher's own device (no photo is uploaded)."""
    get_owned_class(class_id, db, teacher)

    name, reg_no = name.strip(), reg_no.strip()
    if not name or not reg_no:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Name and registration number are required")
    if faces_detected == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No face detected in the enrollment photo")
    if faces_detected > 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Multiple faces detected - please use a photo with only the student's face"
        )
    if not is_real:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Enrollment photo failed the liveness check. Please use a real, live photo."
        )
    try:
        vec = [float(x) for x in json.loads(embedding)]
    except (ValueError, TypeError):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid embedding")
    encoding = normalize_embedding(vec)

    if (
        db.query(models.Student.id)
        .filter(models.Student.class_id == class_id, models.Student.reg_no == reg_no)
        .first()
    ):
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"A student with registration number '{reg_no}' already exists in this class"
        )

    student = models.Student(class_id=class_id, name=name, reg_no=reg_no, face_encoding=encoding, photo_url=None)
    db.add(student)
    try:
        db.commit()
        db.refresh(student)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"A student with registration number '{reg_no}' already exists in this class"
        )
    return student


@router.get("", response_model=List[StudentOut])
def list_students(
    class_id: int,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    get_owned_class(class_id, db, teacher)
    return (
        db.query(models.Student)
        .options(defer(models.Student.face_encoding))  # don't ship 512-d vectors from the DB just to list names
        .filter(models.Student.class_id == class_id)
        .order_by(models.Student.name)
        .all()
    )


@router.delete("/{student_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_student(
    class_id: int,
    student_id: int,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    get_owned_class(class_id, db, teacher)
    student = (
        db.query(models.Student)
        .filter(models.Student.id == student_id, models.Student.class_id == class_id)
        .first()
    )
    if not student:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student not found")
    db.delete(student)
    db.commit()
    return None
