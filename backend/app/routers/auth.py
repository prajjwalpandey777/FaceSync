from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import GoogleLoginRequest, LoginResponse
from app.security import verify_google_id_token, create_access_token
from app import models

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/google", response_model=LoginResponse)
def google_login(payload: GoogleLoginRequest, db: Session = Depends(get_db)):
    """
    Frontend flow:
    1. Google Identity Services renders the "Sign in with Google" button.
    2. On success it gives you a `credential` (an ID token JWT) in the browser.
    3. POST that credential here as {"id_token": credential}.
    4. We verify it with Google, create/find the teacher, and return our own
       session token for the teacher to use on every other request.
    """
    claims = verify_google_id_token(payload.id_token)

    google_sub = claims["sub"]
    email = claims["email"]
    name = claims.get("name", email)
    picture = claims.get("picture")

    teacher = db.query(models.Teacher).filter(models.Teacher.google_sub == google_sub).first()
    if not teacher:
        teacher = models.Teacher(google_sub=google_sub, email=email, name=name, picture=picture)
        db.add(teacher)
        db.commit()
        db.refresh(teacher)
    else:
        # Keep profile info fresh in case name/photo changed on Google's side.
        teacher.name = name
        teacher.picture = picture
        db.commit()

    token = create_access_token(str(teacher.id))
    return LoginResponse(access_token=token, teacher=teacher)
