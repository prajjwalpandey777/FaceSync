"""Pydantic schemas matching FaceSync API specification."""
import uuid
from datetime import datetime, timezone
import math
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field, PlainSerializer, field_validator


def _to_utc_iso(value: datetime) -> str:
    """Timestamps are stored as naive UTC. Emit them with a 'Z' so browsers
    convert to the viewer's local time correctly (otherwise they are read as local)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


UTCDateTime = Annotated[datetime, PlainSerializer(_to_utc_iso, return_type=str, when_used="json")]

# Auth
class GoogleLoginRequest(BaseModel):
    id_token: str

class AuthConfig(BaseModel):
    google_client_id: str
    demo_mode: bool = False
    server_inference: bool = False   # True only when INFERENCE_MODE=local (server can analyse images as a fallback)


class DevLoginRequest(BaseModel):
    email: Optional[str] = "teacher@facesync.edu"
    name: Optional[str] = "Demo Instructor"

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

# Classes
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

# Students
class StudentOut(BaseModel):
    id: int
    name: str
    reg_no: str
    photo_url: Optional[str] = None

    class Config:
        from_attributes = True

# Sessions
class SessionOut(BaseModel):
    id: int
    class_id: int
    status: str
    started_at: UTCDateTime
    ended_at: Optional[UTCDateTime] = None

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
    spoofs_detected: int = 0
    present_count: int
    total_count: int
    latency_ms: Optional[float] = None
    device: Optional[str] = None

class DeviceFace(BaseModel):
    """One face analysed inside the teacher's browser: only the verdict and 512 numbers are sent."""
    is_real: bool = True
    liveness: float = 1.0
    embedding: Optional[List[float]] = Field(default=None, min_length=512, max_length=512)

    @field_validator("embedding")
    @classmethod
    def _finite(cls, v):
        if v is not None and not all(math.isfinite(x) for x in v):
            raise ValueError("embedding contains non-finite numbers")
        return v


class DeviceScanRequest(BaseModel):
    faces: List[DeviceFace] = Field(default_factory=list, max_length=30)
    latency_ms: Optional[float] = None
    device: Optional[str] = Field(default=None, max_length=60)


class SessionStatusOut(BaseModel):
    session: SessionOut
    present: List[StudentOut]
    total: int

# Reports
class RosterEntry(BaseModel):
    id: int
    name: str
    reg_no: str
    status: str
    confidence: Optional[float] = None

class SessionReport(BaseModel):
    session_id: int
    date: UTCDateTime
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
