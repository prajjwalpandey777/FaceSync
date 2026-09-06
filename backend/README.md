# FaceSync Backend

Python + FastAPI backend for the FaceSync attendance app. Uses `face_recognition`
(dlib) for server-side face matching, Supabase Postgres for storage, Google
Sign-In for teacher auth, and is set up to deploy on Render via Docker.

## Architecture

```
Browser (phone or laptop camera)
   │  captures a frame every ~2s
   ▼
FastAPI backend (this repo)
   │  detects faces (dlib) → computes 128-d embeddings
   │  compares against enrolled students in the class (numpy distance)
   ▼
Supabase Postgres
   (teachers, classes, students + face embeddings, sessions, attendance records)
```

## 1. Set up Supabase

1. Create a project at https://supabase.com.
2. Go to **SQL Editor** → paste the contents of `supabase_schema.sql` → **Run**.
3. Go to **Project Settings → Database → Connection string → URI**. Copy it —
   this is your `DATABASE_URL`. Append `?sslmode=require` if it isn't already there.

## 2. Set up Google Sign-In

1. Go to https://console.cloud.google.com/ → create/select a project.
2. **APIs & Services → OAuth consent screen** → configure it (External, add
   your email as a test user while in development).
3. **APIs & Services → Credentials → Create Credentials → OAuth client ID**
   → Application type: **Web application**.
4. Add your frontend URL to **Authorized JavaScript origins**
   (e.g. `https://your-frontend.onrender.com` and `http://localhost:5500` for local dev).
5. Copy the **Client ID** — this is your `GOOGLE_CLIENT_ID`. You'll use this
   same ID both in the backend `.env` and in the frontend's
   `data-client_id` (see `frontend_integration/INTEGRATION_GUIDE.md`).

## 3. Configure environment variables

Copy `.env.example` to `.env` and fill in the values from steps 1-2:

```bash
cp .env.example .env
```

## 4. Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt # dlib compiles from source — this takes a few minutes
uvicorn app.main:app --reload
```

Visit http://localhost:8000/docs for interactive API docs (Swagger UI) — useful
for testing endpoints (e.g. enrolling a student with a photo) before wiring up
the frontend.

> **dlib install issues locally:** dlib needs `cmake` and a C++ compiler.
> - macOS: `brew install cmake`
> - Ubuntu/Debian: `sudo apt install build-essential cmake libopenblas-dev liblapack-dev`
> - Windows: install "Desktop development with C++" via Visual Studio Build Tools, plus cmake.

## 5. Deploy on Render

Option A — using the included blueprint:
1. Push this repo to GitHub.
2. In Render: **New → Blueprint** → connect the repo → Render reads `render.yaml`.
3. Fill in the `sync: false` env vars in the Render dashboard (`DATABASE_URL`,
   `GOOGLE_CLIENT_ID`, `FRONTEND_ORIGIN`). `JWT_SECRET` is auto-generated.
4. Deploy. **First build takes ~10-15 minutes** since dlib compiles from source —
   this is normal, subsequent deploys are cached and much faster.

Option B — manual web service:
1. **New → Web Service** → connect repo → Environment: **Docker**.
2. Choose at least the **Standard** plan — dlib + face_recognition need more
   than 512MB RAM to build and run comfortably; the free tier will likely OOM
   during the build or under real attendance-scan load.
3. Add the same environment variables as above.

Your API will be live at `https://<your-service-name>.onrender.com`. Test with:
```bash
curl https://<your-service-name>.onrender.com/health
```

## 6. Connect your frontend

The `../frontend` folder already contains the fully wired UI (Google Sign-In,
real camera capture, student photo enrollment, live scanning, reports) — see
`../frontend/README.md` for the two values you need to set before deploying it.

Deploy the frontend as a Render **Static Site** (or Netlify/Vercel/GitHub
Pages) — just make sure its URL matches `FRONTEND_ORIGIN` in the backend's env
vars (for CORS) and is served over **HTTPS**, since camera access requires it.

## API reference

| Method | Path | Purpose |
|---|---|---|
| POST | `/auth/google` | Exchange a Google ID token for a FaceSync session token |
| POST | `/classes` | Create a class |
| GET | `/classes` | List the logged-in teacher's classes |
| GET | `/classes/{id}` | Get one class |
| DELETE | `/classes/{id}` | Delete a class |
| POST | `/classes/{id}/students` | Enroll a student (multipart: `name`, `reg_no`, `photo`) |
| GET | `/classes/{id}/students` | List a class's students |
| DELETE | `/classes/{id}/students/{student_id}` | Remove a student |
| POST | `/classes/{id}/sessions/start` | Start (or resume) an attendance session |
| POST | `/sessions/{id}/scan` | Submit one camera frame (multipart: `frame`) for face matching |
| GET | `/sessions/{id}` | Current live session status/roster |
| POST | `/sessions/{id}/end` | End session, mark remaining students absent |
| GET | `/classes/{id}/reports` | Latest (or `?session_id=`) session report + overall attendance |

All endpoints except `/auth/google` and `/health` require
`Authorization: Bearer <token>` from the login response.

## Notes on accuracy & tuning

- `FACE_MATCH_TOLERANCE` (default `0.5`) controls strictness: lower = fewer
  false positives but more missed matches; the `face_recognition` library's
  own default is `0.6`. Tune based on real test sessions in your classroom's
  lighting.
- Enrollment photos should be a single well-lit, front-facing face — the API
  rejects photos with zero or multiple faces detected.
- The `hog` face detection model is used (CPU-friendly, works fine on Render's
  standard plan). The more accurate `cnn` model needs a GPU and isn't
  practical to deploy here.
