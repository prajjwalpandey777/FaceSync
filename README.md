# FaceSync — full project

```
facesync-project/
├── backend/                    ← Python + FastAPI, deploys on Render as a Web Service
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── security.py
│   │   ├── dependencies.py
│   │   ├── face_service.py
│   │   └── routers/
│   │       ├── auth.py
│   │       ├── classes.py
│   │       ├── students.py
│   │       ├── sessions.py
│   │       └── reports.py
│   ├── supabase_schema.sql     ← run once in Supabase's SQL editor
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── render.yaml
│   ├── .env.example
│   ├── .gitignore
│   └── README.md
│
└── frontend/                   ← plain HTML/JS, deploys on Render as a Static Site
    ├── index.html
    ├── api.js
    ├── camera.js
    ├── render.yaml
    └── README.md
```

## Should this be one GitHub repo or two?

**One repo with two folders (as above) is simplest** — Render lets you point
a service at a subdirectory of a repo. Push this whole `facesync-project/`
folder as a single repo, then:

- Backend: Render → **New → Blueprint** or **Web Service** → connect the repo
  → set **Root Directory** to `backend`.
- Frontend: Render → **New → Static Site** → connect the same repo → set
  **Root Directory** to `frontend`.

(Two separate repos works identically if you'd rather keep them fully apart —
just split the two folders into their own repos and skip setting Root Directory.)

## Deploy order (do this once, in order)

1. **Supabase** — create project, run `backend/supabase_schema.sql`, copy the connection string.
2. **Google Cloud Console** — create an OAuth Client ID (Web application).
3. **Push to GitHub** — commit this whole folder.
4. **Deploy backend** on Render (Docker/Blueprint, root dir `backend`) with the env vars from `backend/.env.example`. Wait for it to finish building (dlib takes ~10-15 min on the first build) and note its URL.
5. **Deploy frontend** on Render (Static Site, root dir `frontend`) — but first edit two values as described in `frontend/README.md`:
   - `api.js` → `API_BASE` = your backend's Render URL (from step 4)
   - `index.html` → `data-client_id` = your Google OAuth Client ID (from step 2)
6. Go back to the backend's env vars on Render and set `FRONTEND_ORIGIN` to your frontend's Render URL (from step 5), then redeploy the backend so CORS allows it.
7. In Google Cloud Console, add the frontend's Render URL to **Authorized JavaScript origins** on your OAuth client.
8. Open the frontend URL, sign in with Google, create a class, enroll a couple of students with clear photos, and run a test attendance scan.

Full details for each step are in `backend/README.md` and `frontend/README.md`.
