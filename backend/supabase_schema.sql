-- FaceSync database schema for Supabase (Postgres)
-- Run this once in the Supabase SQL editor (Project -> SQL Editor -> New query -> Run).

create extension if not exists "uuid-ossp";

-- ============ TEACHERS ============
create table if not exists teachers (
  id uuid primary key default uuid_generate_v4(),
  google_sub text unique not null,       -- Google's stable user id ("sub" claim)
  email text unique not null,
  name text not null,
  picture text,
  created_at timestamptz not null default now()
);

-- ============ CLASSES ============
create table if not exists classes (
  id serial primary key,
  teacher_id uuid not null references teachers(id) on delete cascade,
  name text not null,
  code text not null,
  semester text not null,
  section text not null,
  created_at timestamptz not null default now()
);
create index if not exists idx_classes_teacher on classes(teacher_id);

-- ============ STUDENTS ============
-- face_encoding stores the 128-dimension face embedding produced by face_recognition.
create table if not exists students (
  id serial primary key,
  class_id integer not null references classes(id) on delete cascade,
  name text not null,
  reg_no text not null,
  face_encoding double precision[] not null,
  photo_url text,
  created_at timestamptz not null default now(),
  unique(class_id, reg_no)
);
create index if not exists idx_students_class on students(class_id);

-- ============ ATTENDANCE SESSIONS ============
create table if not exists attendance_sessions (
  id serial primary key,
  class_id integer not null references classes(id) on delete cascade,
  status text not null default 'active' check (status in ('active','ended')),
  started_at timestamptz not null default now(),
  ended_at timestamptz
);
create index if not exists idx_sessions_class on attendance_sessions(class_id);

-- ============ ATTENDANCE RECORDS ============
create table if not exists attendance_records (
  id serial primary key,
  session_id integer not null references attendance_sessions(id) on delete cascade,
  student_id integer not null references students(id) on delete cascade,
  status text not null check (status in ('present','absent')),
  confidence double precision,
  marked_at timestamptz not null default now(),
  unique(session_id, student_id)
);
create index if not exists idx_records_session on attendance_records(session_id);
create index if not exists idx_records_student on attendance_records(student_id);
