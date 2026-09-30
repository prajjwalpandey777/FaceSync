/**
 * FaceSync API client.
 * Loaded after config.js (which defines window.FACESYNC_CONFIG.API_BASE).
 */

// API_BASE comes from config.js (generated on Vercel from FACESYNC_API_URL).
// When it is empty we talk to the page's own origin (local dev: FastAPI serves this folder).
const API_BASE = ((window.FACESYNC_CONFIG && window.FACESYNC_CONFIG.API_BASE) || window.location.origin || "")
  .replace(/\/+$/, "");

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

  async _request(path, { method = "GET", body, isForm = false, auth = true } = {}) {
    const headers = {};
    if (this.token) headers["Authorization"] = `Bearer ${this.token}`;
    if (body && !isForm) headers["Content-Type"] = "application/json";

    let res;
    try {
      res = await fetch(`${API_BASE}${path}`, {
        method,
        headers,
        body: body ? (isForm ? body : JSON.stringify(body)) : undefined,
      });
    } catch (netErr) {
      throw new Error("Cannot reach the FaceSync server. If it was idle it may still be waking up - wait a few seconds and try again.");
    }

    if (!res.ok) {
      let detail = res.statusText || `HTTP ${res.status}`;
      try {
        const data = await res.json();
        if (typeof data.detail === "string") detail = data.detail;
        else if (Array.isArray(data.detail)) detail = data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
      } catch (_) {}
      // Expired / invalid session: drop the token and send the user back to sign-in.
      if (res.status === 401 && auth && this.token) {
        this.clearToken();
        if (typeof window.handleSessionExpired === "function") window.handleSessionExpired();
      }
      throw new Error(detail);
    }
    if (res.status === 204) return null;
    return res.json();
  },

  // ---------- Auth ----------
  authConfig() { return this._request("/auth/config", { auth: false }); },
  loginWithGoogle(idToken) {
    return this._request("/auth/google", { method: "POST", body: { id_token: idToken }, auth: false });
  },
  loginDev(email = "teacher@facesync.local", name = "Local Instructor") {
    return this._request("/auth/dev-login", { method: "POST", body: { email, name }, auth: false });
  },
  me() { return this._request("/auth/me"); },
  /** Optional: lets an INFERENCE_MODE=local server load its models early (harmless otherwise). */
  warmup() { return this._request("/warmup", { method: "POST" }).catch(() => {}); },

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
  /** Enrollment when the face was analysed in the browser: only 512 numbers are sent, never the photo. */
  enrollStudentFromEmbedding(classId, { name, regNo, embedding, facesDetected = 1, isReal = true }) {
    const form = new FormData();
    form.append("name", name);
    form.append("reg_no", regNo);
    form.append("embedding", JSON.stringify(embedding || []));
    form.append("faces_detected", String(facesDetected));
    form.append("is_real", isReal ? "true" : "false");
    return this._request(`/classes/${classId}/students/from-embedding`, { method: "POST", body: form, isForm: true });
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
  /** Attendance scan when faces were analysed in the browser (no image leaves the device). */
  scanEmbeddings(sessionId, faces, latencyMs, device) {
    return this._request(`/sessions/${sessionId}/scan-embeddings`, {
      method: "POST",
      body: { faces, latency_ms: latencyMs, device },
    });
  },
  sessionStatus(sessionId) { return this._request(`/sessions/${sessionId}`); },
  endSession(sessionId) { return this._request(`/sessions/${sessionId}/end`, { method: "POST" }); },

  // ---------- Reports ----------
  classReport(classId, sessionId) {
    const q = sessionId ? `?session_id=${sessionId}` : "";
    return this._request(`/classes/${classId}/reports${q}`);
  },
  async exportSessionExcel(classId, sessionId) {
    const headers = {};
    if (this.token) headers["Authorization"] = `Bearer ${this.token}`;
    let res;
    try {
      res = await fetch(`${API_BASE}/classes/${classId}/reports/${sessionId}/export`, { headers });
    } catch (_) {
      throw new Error("Cannot reach the FaceSync server");
    }
    if (!res.ok) throw new Error("Could not export attendance");
    return res.blob();
  },
};
