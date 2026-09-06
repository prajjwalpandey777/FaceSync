from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_teacher, get_owned_class
from app.face_service import encode_all_faces, best_match
from app.schemas import ScanResponse, MatchResult, SessionOut, SessionStatusOut, StudentOut
from app import models

router = APIRouter(tags=["sessions"])


@router.post("/classes/{class_id}/sessions/start", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def start_session(
    class_id: int,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    school_class = get_owned_class(class_id, db, teacher)
    if not school_class.students:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Add students to this class before taking attendance")

    existing_active = (
        db.query(models.AttendanceSession)
        .filter(models.AttendanceSession.class_id == class_id, models.AttendanceSession.status == "active")
        .first()
    )
    if existing_active:
        return existing_active  # resume instead of creating a duplicate

    session = models.AttendanceSession(class_id=class_id, status="active")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.post("/sessions/{session_id}/scan", response_model=ScanResponse)
async def scan_frame(
    session_id: int,
    frame: UploadFile = File(..., description="A single camera frame (JPEG/PNG) from the browser"),
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    """
    Called repeatedly (e.g. every 1-2 seconds) by the frontend while a scan is live.
    Detects every face in the frame and matches each one against students in this
    class who are not already marked present.
    """
    session = _get_owned_session(session_id, db, teacher)
    if session.status != "active":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This session has already ended")

    image_bytes = await frame.read()
    face_encodings = encode_all_faces(image_bytes)

    students = db.query(models.Student).filter(models.Student.class_id == session.class_id).all()
    already_present_ids = {
        r.student_id
        for r in db.query(models.AttendanceRecord).filter(
            models.AttendanceRecord.session_id == session_id,
            models.AttendanceRecord.status == "present",
        )
    }
    candidates = [(s.id, s.face_encoding) for s in students if s.id not in already_present_ids]
    students_by_id = {s.id: s for s in students}

    matches: List[MatchResult] = []
    matched_ids_this_frame = set()

    for encoding in face_encodings:
        remaining = [c for c in candidates if c[0] not in matched_ids_this_frame]
        result = best_match(encoding, remaining, tolerance=settings.FACE_MATCH_TOLERANCE)
        if result is None:
            continue
        student_id, confidence = result
        matched_ids_this_frame.add(student_id)

        record = models.AttendanceRecord(
            session_id=session_id, student_id=student_id, status="present", confidence=confidence
        )
        db.add(record)
        student = students_by_id[student_id]
        matches.append(MatchResult(student_id=student.id, name=student.name, reg_no=student.reg_no, confidence=confidence, status="matched"))

    if matches:
        db.commit()

    present_count = len(already_present_ids) + len(matches)
    return ScanResponse(matches=matches, faces_detected=len(face_encodings), present_count=present_count, total_count=len(students))


@router.get("/sessions/{session_id}", response_model=SessionStatusOut)
def session_status(
    session_id: int,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    session = _get_owned_session(session_id, db, teacher)
    present_student_ids = [
        r.student_id
        for r in db.query(models.AttendanceRecord).filter(
            models.AttendanceRecord.session_id == session_id, models.AttendanceRecord.status == "present"
        )
    ]
    present_students = db.query(models.Student).filter(models.Student.id.in_(present_student_ids)).all()
    total = db.query(models.Student).filter(models.Student.class_id == session.class_id).count()
    return SessionStatusOut(session=session, present=present_students, total=total)


@router.post("/sessions/{session_id}/end", response_model=SessionOut)
def end_session(
    session_id: int,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    """Marks every student not already marked present as absent, then closes the session."""
    session = _get_owned_session(session_id, db, teacher)
    if session.status == "ended":
        return session

    students = db.query(models.Student).filter(models.Student.class_id == session.class_id).all()
    present_ids = {
        r.student_id
        for r in db.query(models.AttendanceRecord).filter(
            models.AttendanceRecord.session_id == session_id, models.AttendanceRecord.status == "present"
        )
    }
    for student in students:
        if student.id not in present_ids:
            db.add(models.AttendanceRecord(session_id=session_id, student_id=student.id, status="absent"))

    session.status = "ended"
    session.ended_at = datetime.utcnow()
    db.commit()
    db.refresh(session)
    return session


def _get_owned_session(session_id: int, db: Session, teacher: models.Teacher) -> models.AttendanceSession:
    session = (
        db.query(models.AttendanceSession)
        .join(models.SchoolClass, models.AttendanceSession.class_id == models.SchoolClass.id)
        .filter(models.AttendanceSession.id == session_id, models.SchoolClass.teacher_id == teacher.id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session
