"""Live attendance scanning and session lifecycle routes."""
import time
from datetime import datetime
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.database import get_db
from app.dependencies import get_current_teacher, get_owned_class, get_owned_session
from app.face_service import analyze_frame, best_match, normalize_embedding
from app.schemas import DeviceScanRequest, MatchResult, ScanResponse, SessionOut, SessionStatusOut
from app.warmup import warm_inference

router = APIRouter(tags=["sessions"])


def _persist_matches(db: Session, session_id: int, pending, already_present_ids: set):
    """Insert attendance rows. If two scans race for the same student the unique constraint
    fires; fall back to one-by-one inserts so a duplicate never fails the whole request."""
    if not pending:
        return []

    def _record(student_id, confidence):
        return models.AttendanceRecord(
            session_id=session_id, student_id=student_id, status="present", confidence=confidence
        )

    try:
        for student_id, confidence in pending:
            db.add(_record(student_id, confidence))
        db.commit()
        return list(pending)
    except IntegrityError:
        db.rollback()

    persisted = []
    for student_id, confidence in pending:
        try:
            db.add(_record(student_id, confidence))
            db.commit()
            persisted.append((student_id, confidence))
        except IntegrityError:
            db.rollback()
            already_present_ids.add(student_id)
    return persisted


@router.post("/classes/{class_id}/sessions/start", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def start_session(
    class_id: int,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    school_class = get_owned_class(class_id, db, teacher)
    student_total = db.query(func.count(models.Student.id)).filter(models.Student.class_id == class_id).scalar()
    if not student_total:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Add students to this class before taking attendance")

    background.add_task(warm_inference)   # no-op unless INFERENCE_MODE=local

    existing_active = (
        db.query(models.AttendanceSession)
        .filter(models.AttendanceSession.class_id == class_id, models.AttendanceSession.status == "active")
        .first()
    )
    if existing_active:
        return existing_active

    session = models.AttendanceSession(class_id=school_class.id, status="active")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _match_and_record(db: Session, session: models.AttendanceSession, faces, latency_ms: float, device: str) -> ScanResponse:
    """Shared by the on-device scan and the optional server-side scan. `faces` is any list of objects
    with `.is_real` and `.embedding`; matches them to students and stores attendance."""
    session_id = session.id
    already_present_ids = {
        row[0]
        for row in db.query(models.AttendanceRecord.student_id).filter(
            models.AttendanceRecord.session_id == session_id,
            models.AttendanceRecord.status == "present",
        )
    }
    total_count = (
        db.query(func.count(models.Student.id)).filter(models.Student.class_id == session.class_id).scalar() or 0
    )

    # Only load embeddings of students who still need to be matched.
    candidates_q = db.query(models.Student).filter(models.Student.class_id == session.class_id)
    if already_present_ids:
        candidates_q = candidates_q.filter(models.Student.id.notin_(already_present_ids))
    candidate_students = candidates_q.all()
    candidates = [(s.id, s.face_encoding) for s in candidate_students]
    # Copy display fields now: commit() expires ORM objects and re-loading them would re-fetch embeddings.
    student_info = {s.id: (s.name, s.reg_no) for s in candidate_students}

    pending = []  # (student_id, confidence) to persist
    matched_ids = set()
    spoofs_detected = 0

    for face in faces:
        if not face.is_real:
            spoofs_detected += 1
            continue
        if not face.embedding:
            continue
        remaining = [c for c in candidates if c[0] not in matched_ids]
        result = best_match(face.embedding, remaining, threshold=settings.FACE_MATCH_THRESHOLD)
        if result is None:
            continue
        student_id, confidence = result
        matched_ids.add(student_id)
        pending.append((student_id, confidence))

    persisted = _persist_matches(db, session_id, pending, already_present_ids)

    matches: List[MatchResult] = []
    for student_id, confidence in persisted:
        name, reg_no = student_info[student_id]
        matches.append(
            MatchResult(student_id=student_id, name=name, reg_no=reg_no, confidence=confidence, status="matched")
        )

    return ScanResponse(
        matches=matches,
        faces_detected=len(faces),
        spoofs_detected=spoofs_detected,
        present_count=len(already_present_ids) + len(persisted),
        total_count=total_count,
        latency_ms=round(latency_ms, 2),
        device=device,
    )


@router.post("/sessions/{session_id}/scan-embeddings", response_model=ScanResponse)
def scan_embeddings(
    session_id: int,
    payload: DeviceScanRequest,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    """Main scan route. The browser already found the faces and computed their embeddings on the user's
    own GPU/CPU, so no image is uploaded. Matching and attendance rules are identical to /scan."""
    session = get_owned_session(session_id, db, teacher)
    if session.status != "active":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This session has already ended")

    for face in payload.faces:
        if face.embedding:
            face.embedding = normalize_embedding(face.embedding)
    return _match_and_record(db, session, payload.faces, payload.latency_ms or 0.0, payload.device or "on-device")


@router.post("/sessions/{session_id}/scan", response_model=ScanResponse)
def scan_frame(
    session_id: int,
    frame: UploadFile = File(..., description="A single camera frame (JPEG/PNG) from the browser"),
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    """Optional fallback (INFERENCE_MODE=local only): the server analyses an uploaded frame itself."""
    session = get_owned_session(session_id, db, teacher)
    if session.status != "active":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This session has already ended")

    t0 = time.perf_counter()
    image_bytes = frame.file.read(settings.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    analysis = analyze_frame(image_bytes)          # raises 501 in the default 'device' mode
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return _match_and_record(db, session, analysis.faces, latency_ms, analysis.gpu_name or analysis.device)


@router.get("/sessions/{session_id}", response_model=SessionStatusOut)
def session_status(
    session_id: int,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    session = get_owned_session(session_id, db, teacher)
    present_ids = [
        row[0]
        for row in db.query(models.AttendanceRecord.student_id).filter(
            models.AttendanceRecord.session_id == session_id, models.AttendanceRecord.status == "present"
        )
    ]
    present_students = (
        db.query(models.Student).filter(models.Student.id.in_(present_ids)).all() if present_ids else []
    )
    total_students = db.query(func.count(models.Student.id)).filter(models.Student.class_id == session.class_id).scalar()

    return SessionStatusOut(session=session, present=present_students, total=total_students or 0)


@router.post("/sessions/{session_id}/end", response_model=SessionOut)
def end_session(
    session_id: int,
    db: Session = Depends(get_db),
    teacher: models.Teacher = Depends(get_current_teacher),
):
    session = get_owned_session(session_id, db, teacher)
    if session.status == "ended":
        return session

    session.status = "ended"
    session.ended_at = datetime.utcnow()

    marked_ids = {
        row[0]
        for row in db.query(models.AttendanceRecord.student_id).filter(
            models.AttendanceRecord.session_id == session_id
        )
    }
    for (student_id,) in db.query(models.Student.id).filter(models.Student.class_id == session.class_id):
        if student_id not in marked_ids:
            db.add(models.AttendanceRecord(session_id=session_id, student_id=student_id, status="absent"))

    db.commit()
    db.refresh(session)
    return session
