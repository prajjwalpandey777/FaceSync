import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


# ---------- Auth ----------
class GoogleLoginRequest(BaseModel):
    id_token: str


class TeacherOut(BaseModel):
    id: uuid.UUID
    email: str
    name: str
    picture: Optional[str] = None

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    teacher: TeacherOut


# ---------- Classes ----------
class ClassCreate(BaseModel):
    name: str
    code: str
    semester: str
    section: str


class ClassOut(BaseModel):
    id: int
    name: str
    code: str
    semester: str
    section: str
    student_count: int = 0

    class Config:
        from_attributes = True


# ---------- Students ----------
class StudentOut(BaseModel):
    id: int
    name: str
    reg_no: str
    photo_url: Optional[str] = None

    class Config:
        from_attributes = True


# ---------- Sessions ----------
class SessionOut(BaseModel):
    id: int
    class_id: int
    status: str
    started_at: datetime
    ended_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MatchResult(BaseModel):
    student_id: int
    name: str
    reg_no: str
    confidence: float
    status: str  # "matched" | "already_marked"


class ScanResponse(BaseModel):
    matches: List[MatchResult]
    faces_detected: int
    present_count: int
    total_count: int


class SessionStatusOut(BaseModel):
    session: SessionOut
    present: List[StudentOut]
    total: int


# ---------- Reports ----------
class RosterEntry(BaseModel):
    id: int
    name: str
    reg_no: str
    status: str
    confidence: Optional[float] = None


class SessionReport(BaseModel):
    session_id: int
    date: datetime
    present: List[RosterEntry]
    absent: List[RosterEntry]
    total: int
    present_count: int
    attendance_rate: float


class OverallAttendanceRow(BaseModel):
    student_id: int
    name: str
    reg_no: str
    sessions_attended: int
    sessions_total: int
    percentage: float


class ClassReport(BaseModel):
    class_id: int
    class_name: str
    class_code: str
    sessions_run: int
    latest_session: Optional[SessionReport] = None
    overall: List[OverallAttendanceRow]
