from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_teacher, get_owned_class
from app.face_service import encode_single_face
from app.schemas import StudentOut
from app import models

router = APIRouter(prefix="/classes/{class_id}/students", tags=["students"])


@router.post("", response_model=StudentOut, status_code=status.HTTP_201_CREATED)
async def enroll_student(
    class_id: int,
    name: str = Form(...),
    reg_no: str = Form(...),
    photo: UploadFile = File(..., description="One clear, front-facing photo of the student"),
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    """
    Enrolling a student REQUIRES a face photo — this is what attendance scanning
    will match against later. Reject non-image uploads early.
    """
    get_owned_class(class_id, db, teacher)  # ownership check

    if not photo.content_type or not photo.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must be an image")

    image_bytes = await photo.read()
    encoding = encode_single_face(image_bytes)  # raises 400 if 0 or 2+ faces found

    student = models.Student(
        class_id=class_id,
        name=name.strip(),
        reg_no=reg_no.strip(),
        face_encoding=encoding,
    )
    db.add(student)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A student with this registration number already exists in this class")
    db.refresh(student)
    return student


@router.get("", response_model=List[StudentOut])
def list_students(
    class_id: int,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    school_class = get_owned_class(class_id, db, teacher)
    return school_class.students


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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    db.delete(student)
    db.commit()
