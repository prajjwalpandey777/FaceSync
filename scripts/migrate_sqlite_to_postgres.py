#!/usr/bin/env python
"""Copy your existing local FaceSync data (facesync.db, SQLite) into Supabase Postgres.

Your old accounts were "dev" logins, so their Google identity does not exist yet. Use
--assign-to-email to move ALL classes / students / sessions under the Gmail address you
will sign in with. The first time you sign in with Google using that address, FaceSync
links your Google account to that teacher record automatically.

Examples
    # preview only
    python scripts/migrate_sqlite_to_postgres.py --source facesync.db --assign-to-email you@gmail.com --dry-run

    # real run (target = your Supabase Session-pooler URL; or set DATABASE_URL_TARGET)
    python scripts/migrate_sqlite_to_postgres.py --source facesync.db \
        --target "postgresql://postgres.xxxx:PASSWORD@aws-0-ap-south-1.pooler.supabase.com:5432/postgres" \
        --assign-to-email you@gmail.com

Embeddings are copied as-is. If you changed the recognition model since enrolling those
students, re-enroll them instead (embeddings from different models are not comparable).
"""
import argparse
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["DATABASE_URL"] = "sqlite://"  # importing app.database must not touch a real database

from sqlalchemy import create_engine, func, insert, select, text  # noqa: E402

from app import models  # noqa: E402
from app.database import Base  # noqa: E402

T_TEACHERS = models.Teacher.__table__
T_CLASSES = models.SchoolClass.__table__
T_STUDENTS = models.Student.__table__
T_SESSIONS = models.AttendanceSession.__table__
T_RECORDS = models.AttendanceRecord.__table__
SERIAL_TABLES = [T_CLASSES, T_STUDENTS, T_SESSIONS, T_RECORDS]


def make_target_engine(url: str):
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    kwargs = {}
    if url.startswith("postgresql") and "sslmode=" not in url and "@localhost" not in url and "@127.0.0.1" not in url:
        kwargs["connect_args"] = {"sslmode": "require"}
    return create_engine(url, pool_pre_ping=True, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(ROOT / "facesync.db"), help="path to the SQLite file")
    ap.add_argument("--target", default=os.environ.get("DATABASE_URL_TARGET", ""), help="Postgres URL")
    ap.add_argument("--assign-to-email", default="", help="put every class under this (Gmail) address")
    ap.add_argument("--dry-run", action="store_true", help="show what would be copied, change nothing")
    ap.add_argument("--force", action="store_true", help="copy even if the target already has classes")
    args = ap.parse_args()

    src_path = Path(args.source)
    if not src_path.is_file():
        print(f"Source database not found: {src_path}")
        return 1
    if not args.target and not args.dry_run:
        print("Provide --target (Supabase connection string) or set DATABASE_URL_TARGET.")
        return 1

    src = create_engine(f"sqlite:///{src_path.resolve()}")
    with src.connect() as s:
        teachers = s.execute(select(T_TEACHERS)).mappings().all()
        classes = s.execute(select(T_CLASSES)).mappings().all()
        students = s.execute(select(T_STUDENTS)).mappings().all()
        sessions = s.execute(select(T_SESSIONS)).mappings().all()
        records = s.execute(select(T_RECORDS)).mappings().all()

    print("Source data:")
    print(f"  teachers={len(teachers)} classes={len(classes)} students={len(students)} "
          f"sessions={len(sessions)} records={len(records)}")
    if args.assign_to_email:
        print(f"  -> all classes will be assigned to {args.assign_to_email.lower()}")
    else:
        print("  -> teachers are copied as they are (their dev accounts cannot sign in with Google;"
              " consider --assign-to-email)")
    if args.dry_run:
        print("Dry run - nothing written.")
        return 0

    dst = make_target_engine(args.target)
    Base.metadata.create_all(dst)

    with dst.begin() as d:
        existing = d.execute(select(func.count()).select_from(T_CLASSES)).scalar() or 0
        if existing and not args.force:
            print(f"Target already has {existing} classes. Re-run with --force to add anyway "
                  "(ids may collide), or use an empty database.")
            return 2

        # ---- teachers ------------------------------------------------------
        teacher_map = {}
        if args.assign_to_email:
            email = args.assign_to_email.strip().lower()
            tid = d.execute(select(T_TEACHERS.c.id).where(T_TEACHERS.c.email == email)).scalar()
            if tid is None:
                tid = uuid.uuid4()
                d.execute(insert(T_TEACHERS).values(
                    id=tid,
                    google_sub=f"pending-google-link:{email}",   # replaced at first Google sign-in
                    email=email,
                    name=email.split("@")[0],
                    picture=None,
                    created_at=datetime.utcnow(),
                ))
            for t in teachers:
                teacher_map[t["id"]] = tid
        else:
            for t in teachers:
                found = d.execute(select(T_TEACHERS.c.id).where(T_TEACHERS.c.email == t["email"])).scalar()
                if found is None:
                    d.execute(insert(T_TEACHERS).values(**dict(t)))
                    found = t["id"]
                teacher_map[t["id"]] = found

        # ---- everything else (ids preserved so foreign keys stay valid) ------
        if classes:
            d.execute(insert(T_CLASSES), [{**dict(c), "teacher_id": teacher_map[c["teacher_id"]]} for c in classes])
        if students:
            d.execute(insert(T_STUDENTS), [dict(x) for x in students])
        if sessions:
            d.execute(insert(T_SESSIONS), [dict(x) for x in sessions])
        if records:
            d.execute(insert(T_RECORDS), [dict(x) for x in records])

        # Postgres sequences do not advance when we insert explicit ids -> fix them up.
        if d.dialect.name == "postgresql":
            for t in SERIAL_TABLES:
                d.execute(text(
                    f"SELECT setval(pg_get_serial_sequence('{t.name}', 'id'), "
                    f"COALESCE((SELECT MAX(id) FROM {t.name}), 0) + 1, false)"
                ))

    print("Done. Migrated:")
    print(f"  classes={len(classes)} students={len(students)} sessions={len(sessions)} records={len(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
