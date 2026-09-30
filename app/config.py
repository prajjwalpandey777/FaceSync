"""Configuration for the FaceSync backend.

Everything is driven by environment variables (or a local `.env` file), so the
same code runs on your laptop, on Render, and against Supabase.
Face recognition runs in the user's browser (INFERENCE_MODE=device); no GPU service is needed.
"""
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_INSECURE_JWT_DEFAULT = "change-me-in-production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "development" (default) or "production". Production turns on strict checks.
    ENVIRONMENT: str = "development"

    # ---- Database ---------------------------------------------------------
    # Local default = SQLite. On Render set this to your Supabase Postgres URI.
    DATABASE_URL: str = f"sqlite:///{PROJECT_ROOT / 'facesync.db'}"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 5

    # ---- Google sign-in ---------------------------------------------------
    # OAuth 2.0 *Web application* Client ID from Google Cloud Console.
    GOOGLE_CLIENT_ID: str = ""
    # Optional: comma-separated list of allowed email domains, e.g. "myschool.edu".
    # Leave empty to allow any Google account.
    ALLOWED_EMAIL_DOMAINS: str = ""

    # ---- Session tokens (issued by this backend after Google sign-in) ------
    JWT_SECRET: str = _INSECURE_JWT_DEFAULT
    JWT_EXPIRE_MINUTES: int = 720

    # ---- CORS -------------------------------------------------------------
    # Comma-separated list of allowed frontend origins, e.g.
    # "https://facesync.vercel.app,http://localhost:5500". "*" = allow all (dev only).
    FRONTEND_ORIGINS: str = "*"
    # Also allow Vercel preview deployments of your project (regex on the origin).
    FRONTEND_ORIGIN_REGEX: str = ""

    # Serve ./frontend from this server (handy locally; disable on Render if Vercel hosts it).
    SERVE_FRONTEND: bool = True

    # ---- Dev / demo login (NEVER enable in production) ---------------------
    DEMO_MODE: bool = False

    # ---- Inference --------------------------------------------------------
    # "device" -> (default) faces are recognised INSIDE the user's browser (laptop GPU/CPU, phone RAM).
    #             The server never sees camera images and needs no torch/opencv - runs on a small Render plan.
    # "local"  -> optional extra: the server can ALSO run the models in-process (needs requirements-local.txt
    #             and ~1.5 GB RAM). The website uses it only as a fallback when a device cannot run the models.
    INFERENCE_MODE: str = "device"
    EMBEDDING_DIM: int = 512

    # ---- Matching ---------------------------------------------------------
    FACE_MATCH_THRESHOLD: float = 0.45
    MAX_UPLOAD_MB: int = 8

    # ---- Local-inference settings (INFERENCE_MODE=local) ------------------
    DEVICE: str = "auto"                  # auto | cuda | cpu
    DETECTOR_BACKEND: str = "auto"
    CHECKPOINT_PATH: str = str(PROJECT_ROOT / "weights" / "backbone.pth")
    YUNET_PATH: str = str(PROJECT_ROOT / "weights" / "face_detection_yunet_2023mar.onnx")
    ANTI_SPOOF_MODEL_PATH: str = str(PROJECT_ROOT / "weights" / "2.7_80x80_MiniFASNetV2.onnx")
    ANTI_SPOOF_THRESHOLD: float = 0.60
    ANTI_SPOOF_ENABLED: bool = True

    # ---- helpers ----------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip().rstrip("/") for o in self.FRONTEND_ORIGINS.split(",") if o.strip()]

    @property
    def allowed_email_domains(self) -> List[str]:
        return [d.strip().lower().lstrip("@") for d in self.ALLOWED_EMAIL_DOMAINS.split(",") if d.strip()]

    def validate_for_startup(self) -> None:
        """Fail fast on unsafe production configuration."""
        problems = []
        if self.is_production:
            if self.JWT_SECRET == _INSECURE_JWT_DEFAULT or len(self.JWT_SECRET) < 32:
                problems.append("JWT_SECRET must be set to a random string of 32+ characters")
            if not self.GOOGLE_CLIENT_ID:
                problems.append("GOOGLE_CLIENT_ID is required")
            if self.DEMO_MODE:
                problems.append("DEMO_MODE must be false in production")
            if not self.DATABASE_URL or self.DATABASE_URL.startswith("sqlite"):
                problems.append("DATABASE_URL must point to Supabase Postgres, not SQLite")
            if "*" in self.cors_origins:
                problems.append("FRONTEND_ORIGINS must list your Vercel URL(s), not '*'")
        if self.INFERENCE_MODE.lower() not in ("device", "local"):
            problems.append("INFERENCE_MODE must be 'device' (default) or 'local'")
        # Local model weight checks (only needed when the server itself runs the models)
        if self.INFERENCE_MODE.lower() == "local":
            import os
            for label, path in [
                ("backbone.pth (ArcFace weights)", self.CHECKPOINT_PATH),
                ("face_detection_yunet_2023mar.onnx", self.YUNET_PATH),
                ("2.7_80x80_MiniFASNetV2.onnx (anti-spoof)", self.ANTI_SPOOF_MODEL_PATH),
            ]:
                if not os.path.isfile(path):
                    problems.append(f"Model file not found: {label} at {path}")
        if problems:
            raise RuntimeError("Invalid FaceSync configuration:\n  - " + "\n  - ".join(problems))


settings = Settings()
