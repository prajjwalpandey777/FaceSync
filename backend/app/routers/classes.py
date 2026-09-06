from typing import List

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_teacher, get_owned_class
from app.schemas import ClassCreate, ClassOut
from app import models

router = APIRouter(prefix="/classes", tags=["classes"])


@router.post("", response_model=ClassOut, status_code=status.HTTP_201_CREATED)
def create_class(
    payload: ClassCreate,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    school_class = models.SchoolClass(teacher_id=teacher.id, **payload.model_dump())
    db.add(school_class)
    db.commit()
    db.refresh(school_class)
    return ClassOut(**school_class.__dict__, student_count=0)


@router.get("", response_model=List[ClassOut])
def list_classes(
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    classes = db.query(models.SchoolClass).filter(models.SchoolClass.teacher_id == teacher.id).all()
    return [ClassOut(**c.__dict__, student_count=len(c.students)) for c in classes]


@router.get("/{class_id}", response_model=ClassOut)
def get_class(
    class_id: int,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    c = get_owned_class(class_id, db, teacher)
    return ClassOut(**c.__dict__, student_count=len(c.students))


@router.delete("/{class_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_class(
    class_id: int,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    c = get_owned_class(class_id, db, teacher)
    db.delete(c)
    db.commit()
