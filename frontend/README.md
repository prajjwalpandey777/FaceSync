# FaceSync Frontend

Plain HTML/CSS/JS — no build step. This is your original UI, now wired to the
real FaceSync backend (see `../facesync-backend`):

- `index.html` — the app itself
- `api.js` — fetch wrapper for every backend endpoint
- `camera.js` — webcam/phone camera capture helper

## Before deploying

1. Open `api.js` and set `API_BASE` to your deployed backend URL:
   ```js
   const API_BASE = "https://your-backend.onrender.com";
   ```
2. Open `index.html`, find the `g_id_onload` div near the top of the login
   view, and set `data-client_id` to your real Google OAuth Client ID
   (same one used in the backend's `GOOGLE_CLIENT_ID`).

## Deploy on Render

**New → Static Site** → connect this folder/repo → build command: empty →
publish directory: `.` (or use the included `render.yaml` blueprint).

Camera access requires HTTPS — Render's static sites are served over HTTPS
automatically, so this works out of the box once deployed. For local testing,
serve over `https://` (e.g. via `ngrok`) rather than opening the file
directly or using plain `http://`, or Chrome/Safari will block camera access.

In Google Cloud Console, add this site's URL to **Authorized JavaScript
origins** for your OAuth client (see backend README step 2).
