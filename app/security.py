"""Session-token (JWT) helpers and Google ID-token verification."""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from fastapi import HTTPException, status
from jose import JWTError, jwt

from app.config import settings

JWT_ALGORITHM = "HS256"

# ---------------------------------------------------------------------------
# Our own session tokens
# ---------------------------------------------------------------------------

def create_access_token(subject: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(subject),
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    subject = payload.get("sub")
    if not subject:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token subject")
    return subject


# ---------------------------------------------------------------------------
# Google sign-in
# ---------------------------------------------------------------------------

_google_request = None


def _get_google_request():
    """A reusable transport. Uses an HTTP cache for Google's public certs when possible."""
    global _google_request
    if _google_request is None:
        import requests
        from google.auth.transport import requests as google_requests

        session = requests.Session()
        try:
            import cachecontrol

            session = cachecontrol.CacheControl(session)
        except ImportError:  # cache is an optimisation only
            pass
        _google_request = google_requests.Request(session=session)
    return _google_request


def verify_google_id_token(token: str) -> Dict[str, Any]:
    """Validate a Google ID token (the `credential` from Google Identity Services).

    Checks signature, expiry, issuer and that the token was minted for OUR client ID.
    Additionally requires a verified email and (optionally) an allowed email domain.
    """
    # Local-only shortcut for demos/tests. Impossible when ENVIRONMENT=production.
    if settings.DEMO_MODE and not settings.is_production and token in ("dev_token", "demo_token", "test_token"):
        return {
            "sub": "dev-google-sub-001",
            "email": "teacher@facesync.edu",
            "email_verified": True,
            "name": "Prof. FaceSync",
            "picture": None,
        }

    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Google sign-in is not configured on the server")

    try:
        from google.oauth2 import id_token as google_id_token

        claims = google_id_token.verify_oauth2_token(
            token,
            _get_google_request(),
            settings.GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=10,
        )
    except Exception as exc:  # ValueError for bad tokens, network errors for cert fetch
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Google token verification failed: {exc}")

    if not claims.get("email") or not claims.get("email_verified"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Your Google account email is not verified")

    allowed = settings.allowed_email_domains
    if allowed:
        domain = claims["email"].rsplit("@", 1)[-1].lower()
        if domain not in allowed:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Accounts from '{domain}' are not allowed. Use one of: {', '.join(allowed)}",
            )
    return claims
