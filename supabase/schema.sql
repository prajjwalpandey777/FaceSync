-- FaceSync schema for Supabase (PostgreSQL).
-- Run once in: Supabase Dashboard -> SQL Editor -> New query -> paste -> Run.
-- (The API also creates missing tables on startup, so this file is optional -
--  but running it first gives you Row Level Security and indexes from day one.)

create table if not exists teachers (
    id          uuid primary key default gen_random_uuid(),
    google_sub  varchar not null unique,
    email       varchar not null unique,
    name        varchar not null,
    picture     varchar,
    created_at  timestamp default (now() at time zone 'utc')
);

create table if not exists classes (
    id          serial primary key,
    teacher_id  uuid not null references teachers(id) on delete cascade,
    name        varchar not null,
    code        varchar not null,
    semester    varchar not null,
    section     varchar not null,
    created_at  timestamp default (now() at time zone 'utc')
);
create index if not exists ix_classes_teacher_id on classes(teacher_id);

create table if not exists students (
    id             serial primary key,
    class_id       integer not null references classes(id) on delete cascade,
    name           varchar not null,
    reg_no         varchar not null,
    face_encoding  double precision[] not null,      -- 512-d ArcFace embedding (biometric data!)
    photo_url      varchar,
    created_at     timestamp default (now() at time zone 'utc'),
    constraint uq_student_class_reg unique (class_id, reg_no)
);
create index if not exists ix_students_class_id on students(class_id);

create table if not exists attendance_sessions (
    id          serial primary key,
    class_id    integer not null references classes(id) on delete cascade,
    status      varchar not null default 'active',   -- active | ended
    started_at  timestamp default (now() at time zone 'utc'),
    ended_at    timestamp
);
create index if not exists ix_attendance_sessions_class_id on attendance_sessions(class_id);

create table if not exists attendance_records (
    id          serial primary key,
    session_id  integer not null references attendance_sessions(id) on delete cascade,
    student_id  integer not null references students(id) on delete cascade,
    status      varchar not null,                    -- present | absent
    confidence  double precision,
    marked_at   timestamp default (now() at time zone 'utc'),
    constraint uq_record_session_student unique (session_id, student_id)
);
create index if not exists ix_attendance_records_session_id on attendance_records(session_id);
create index if not exists ix_attendance_records_student_id on attendance_records(student_id);

-- SECURITY: Supabase exposes every table in the `public` schema through its REST API.
-- Turning on Row Level Security with NO policies blocks that API completely (anon key can
-- read nothing). Our FastAPI server connects as the `postgres` role, which bypasses RLS,
-- so the app keeps working. Face embeddings are biometric data - keep this on.
alter table teachers            enable row level security;
alter table classes             enable row level security;
alter table students            enable row level security;
alter table attendance_sessions enable row level security;
alter table attendance_records  enable row level security;
