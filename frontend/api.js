/**
 * FaceSync API client.
 * Include this AFTER the Google Identity Services script and BEFORE your
 * existing index.html <script> block. It replaces the mock functions in
 * that file — see INTEGRATION_GUIDE.md for exactly what to swap.
 */

const API_BASE = "https://facesync-rv38.onrender.com"; // <-- set this to your deployed Render URL

const Api = {
  token: localStorage.getItem("facesync_token") || null,

  setToken(token) {
    this.token = token;
    localStorage.setItem("facesync_token", token);
  },

  clearToken() {
    this.token = null;
    localStorage.removeItem("facesync_token");
  },

  async _request(path, { method = "GET", body, isForm = false } = {}) {
    const headers = {};
    if (this.token) headers["Authorization"] = `Bearer ${this.token}`;
    if (body && !isForm) headers["Content-Type"] = "application/json";

    const res = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: body ? (isForm ? body : JSON.stringify(body)) : undefined,
    });

    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    if (res.status === 204) return null;
    return res.json();
  },

  // ---------- Auth ----------
  loginWithGoogle(idToken) {
    return this._request("/auth/google", { method: "POST", body: { id_token: idToken } });
  },

  // ---------- Classes ----------
  listClasses() { return this._request("/classes"); },
  createClass(data) { return this._request("/classes", { method: "POST", body: data }); },
  getClass(id) { return this._request(`/classes/${id}`); },
  deleteClass(id) { return this._request(`/classes/${id}`, { method: "DELETE" }); },

  // ---------- Students ----------
  listStudents(classId) { return this._request(`/classes/${classId}/students`); },
  enrollStudent(classId, { name, regNo, photoBlob }) {
    const form = new FormData();
    form.append("name", name);
    form.append("reg_no", regNo);
    form.append("photo", photoBlob, "photo.jpg");
    return this._request(`/classes/${classId}/students`, { method: "POST", body: form, isForm: true });
  },
  removeStudent(classId, studentId) {
    return this._request(`/classes/${classId}/students/${studentId}`, { method: "DELETE" });
  },

  // ---------- Sessions ----------
  startSession(classId) {
    return this._request(`/classes/${classId}/sessions/start`, { method: "POST" });
  },
  scanFrame(sessionId, frameBlob) {
    const form = new FormData();
    form.append("frame", frameBlob, "frame.jpg");
    return this._request(`/sessions/${sessionId}/scan`, { method: "POST", body: form, isForm: true });
  },
  sessionStatus(sessionId) { return this._request(`/sessions/${sessionId}`); },
  endSession(sessionId) { return this._request(`/sessions/${sessionId}/end`, { method: "POST" }); },

  // ---------- Reports ----------
  classReport(classId, sessionId) {
    const q = sessionId ? `?session_id=${sessionId}` : "";
    return this._request(`/classes/${classId}/reports${q}`);
  },
};
