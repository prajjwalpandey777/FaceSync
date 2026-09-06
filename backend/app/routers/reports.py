from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_teacher, get_owned_class
from app.schemas import ClassReport, SessionReport, RosterEntry, OverallAttendanceRow
from app import models

router = APIRouter(prefix="/classes/{class_id}/reports", tags=["reports"])


def _build_session_report(db: Session, session: models.AttendanceSession, students) -> SessionReport:
    records = {
        r.student_id: r
        for r in db.query(models.AttendanceRecord).filter(models.AttendanceRecord.session_id == session.id)
    }
    present, absent = [], []
    for s in students:
        record = records.get(s.id)
        status_ = record.status if record else "absent"
        entry = RosterEntry(id=s.id, name=s.name, reg_no=s.reg_no, status=status_, confidence=record.confidence if record else None)
        (present if status_ == "present" else absent).append(entry)

    total = len(students)
    rate = round(len(present) / total * 100, 1) if total else 0.0
    return SessionReport(
        session_id=session.id,
        date=session.started_at,
        present=present,
        absent=absent,
        total=total,
        present_count=len(present),
        attendance_rate=rate,
    )


@router.get("", response_model=ClassReport)
def class_report(
    class_id: int,
    session_id: Optional[int] = None,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    school_class = get_owned_class(class_id, db, teacher)
    students = school_class.students
    sessions = (
        db.query(models.AttendanceSession)
        .filter(models.AttendanceSession.class_id == class_id, models.AttendanceSession.status == "ended")
        .order_by(models.AttendanceSession.started_at.asc())
        .all()
    )

    target_session = None
    if session_id:
        target_session = next((s for s in sessions if s.id == session_id), None)
        if not target_session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found for this class")
    elif sessions:
        target_session = sessions[-1]

    latest_report = _build_session_report(db, target_session, students) if target_session else None

    overall = []
    for s in students:
        attended = sum(
            1 for sess in sessions
            if db.query(models.AttendanceRecord).filter(
                models.AttendanceRecord.session_id == sess.id,
                models.AttendanceRecord.student_id == s.id,
                models.AttendanceRecord.status == "present",
            ).first()
        )
        pct = round(attended / len(sessions) * 100, 1) if sessions else 0.0
        overall.append(OverallAttendanceRow(student_id=s.id, name=s.name, reg_no=s.reg_no, sessions_attended=attended, sessions_total=len(sessions), percentage=pct))

    return ClassReport(
        class_id=school_class.id,
        class_name=school_class.name,
        class_code=school_class.code,
        sessions_run=len(sessions),
        latest_session=latest_report,
        overall=overall,
    )
