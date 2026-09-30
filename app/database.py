"""Database engine: SQLite locally, PostgreSQL (Supabase) in production."""
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings


def _normalize_url(url: str) -> str:
    # Supabase / Heroku-style URLs use postgres://, SQLAlchemy needs postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


db_url = _normalize_url(settings.DATABASE_URL)
is_sqlite = db_url.startswith("sqlite")

engine_kwargs: dict = {}
if is_sqlite:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs.update(
        pool_pre_ping=True,          # survive Supabase pooler dropping idle connections
        pool_recycle=300,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
    )
    is_local_pg = any(h in db_url for h in ("@localhost", "@127.0.0.1", "@postgres:"))
    if "sslmode=" not in db_url and not is_local_pg:
        engine_kwargs["connect_args"] = {"sslmode": "require"}

engine = create_engine(db_url, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def secure_postgres_tables() -> None:
    """Supabase publishes every `public` table through its REST API. Enabling Row Level
    Security (with no policies) blocks that route; the API server's own `postgres` role
    bypasses RLS, so nothing changes for FaceSync. Idempotent and best-effort."""
    if is_sqlite:
        return
    log = logging.getLogger("facesync.db")
    try:
        with engine.begin() as conn:
            for table in Base.metadata.tables:
                conn.execute(text(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY'))
    except Exception as exc:  # e.g. a restricted DB user - not fatal
        log.warning("Could not enable Row Level Security automatically: %s", exc)
