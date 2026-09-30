"""Class management routes."""
from typing import List

from fastapi import APIRouter, Depends, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models
from app.database import get_db
from app.dependencies import get_current_teacher, get_owned_class
from app.schemas import ClassCreate, ClassOut

router = APIRouter(prefix="/classes", tags=["classes"])


def _to_out(c: models.SchoolClass, student_count: int) -> ClassOut:
    return ClassOut(
        id=c.id, name=c.name, code=c.code, semester=c.semester, section=c.section, student_count=student_count
    )


@router.post("", response_model=ClassOut, status_code=status.HTTP_201_CREATED)
def create_class(
    payload: ClassCreate,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    school_class = models.SchoolClass(
        teacher_id=teacher.id,
        name=payload.name.strip(),
        code=payload.code.strip(),
        semester=payload.semester.strip(),
        section=payload.section.strip(),
    )
    db.add(school_class)
    db.commit()
    db.refresh(school_class)
    return _to_out(school_class, 0)


@router.get("", response_model=List[ClassOut])
def list_classes(
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    classes = (
        db.query(models.SchoolClass)
        .filter(models.SchoolClass.teacher_id == teacher.id)
        .order_by(models.SchoolClass.created_at.desc())
        .all()
    )
    # One grouped COUNT instead of loading every student (and their 512-d embeddings).
    counts = dict(
        db.query(models.Student.class_id, func.count(models.Student.id))
        .filter(models.Student.class_id.in_([c.id for c in classes] or [0]))
        .group_by(models.Student.class_id)
        .all()
    )
    return [_to_out(c, counts.get(c.id, 0)) for c in classes]


@router.get("/{class_id}", response_model=ClassOut)
def get_class(
    class_id: int,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    c = get_owned_class(class_id, db, teacher)
    count = db.query(func.count(models.Student.id)).filter(models.Student.class_id == c.id).scalar() or 0
    return _to_out(c, count)


@router.delete("/{class_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_class(
    class_id: int,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    school_class = get_owned_class(class_id, db, teacher)
    db.delete(school_class)
    db.commit()
    return None
