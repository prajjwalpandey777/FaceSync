/**
 * FaceSync on-device AI.
 *
 * Face detection, liveness and recognition run INSIDE the user's browser:
 *   - laptop / desktop : the graphics card (WebGPU) when the browser supports it, otherwise the CPU (WebAssembly)
 *   - phone / tablet   : the phone's own GPU (WebGPU) or CPU, using the phone's RAM
 * Only the 512 face numbers go to the server - never the camera image.
 *
 * Loaded after config.js, api.js, camera.js and face-engine.js.
 * Configuration (all optional) comes from window.FACESYNC_CONFIG, written by build-config.js on Vercel:
 *   ARCFACE_URL, YUNET_URL, SPOOF_URL, ORT_VERSION, ORT_BASE
 */
const DeviceAI = (() => {
  const cfg = window.FACESYNC_CONFIG || {};
  const ORT_VERSION = cfg.ORT_VERSION || "1.30.0";
  const ORT_BASE = (cfg.ORT_BASE || `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`).replace(/\/?$/, "/");
  const MODELS = {
    yunet: cfg.YUNET_URL || "models/face_detection_yunet_2023mar.onnx",
    spoof: cfg.SPOOF_URL || "models/2.7_80x80_MiniFASNetV2.onnx",
    arcface: cfg.ARCFACE_URL || "models/arcface_r100.onnx",
  };
  const SLOW_MS = 2500;   // a device slower than this per frame uses the server fallback (only if the server has one)

  const self = {
    mode: null,               // "device" | "server" | null (not prepared yet)
    serverFallback: false,    // set from /auth/config: true only when the API runs with INFERENCE_MODE=local
    info: { backend: "", benchmarkMs: null },
    lastError: null,
    _promise: null,
    _failures: 0,
  };

  // ------------------------------------------------------------ small UI (progress card)
  let card = null;
  function ui(text, pct) {
    if (!text) { if (card) card.style.display = "none"; return; }
    if (!card) {
      card = document.createElement("div");
      card.setAttribute("role", "status");
      card.style.cssText = "position:fixed;left:50%;bottom:22px;transform:translateX(-50%);z-index:9999;width:min(92vw,420px);" +
        "background:rgba(10,14,26,.94);border:1px solid rgba(79,209,197,.4);border-radius:14px;padding:14px 16px;" +
        "color:#E6EDF7;font:13px/1.45 var(--font-mono,ui-monospace,monospace);box-shadow:0 10px 40px rgba(0,0,0,.45);";
      card.innerHTML = '<div data-t></div><div style="height:6px;background:rgba(255,255,255,.12);border-radius:6px;margin-top:10px;overflow:hidden;">' +
        '<div data-b style="height:100%;width:0;background:#4FD1C5;transition:width .25s;"></div></div>';
      document.body.appendChild(card);
    }
    card.style.display = "block";
    card.querySelector("[data-t]").textContent = text;
    card.querySelector("[data-b]").style.width = (pct == null ? 100 : Math.max(3, Math.round(pct * 100))) + "%";
  }

  // ------------------------------------------------------------ helpers
  function connectionInfo() {
    const c = navigator.connection || {};
    const slow = ["slow-2g", "2g", "3g"].includes(c.effectiveType);
    return { metered: !!c.saveData || c.type === "cellular" || slow, saveData: !!c.saveData };
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const el = document.createElement("script");
      el.src = src; el.async = true;
      el.onload = resolve;
      el.onerror = () => reject(new Error("Could not load the on-device AI engine. Check your internet connection or ad-blocker and try again."));
      document.head.appendChild(el);
    });
  }

  async function loadOrt() {
    if (window.ort) return window.ort;
    if (typeof WebAssembly === "undefined") {
      throw new Error("This browser cannot run the on-device AI. Please update Chrome, Edge, Firefox or Safari.");
    }
    await loadScript(ORT_BASE + "ort.webgpu.min.js");
    if (!window.ort) throw new Error("The on-device AI engine did not start.");
    window.ort.env.wasm.wasmPaths = ORT_BASE;
    // Multi-threading needs special page headers that would break Google sign-in, so the CPU path is single-threaded.
    window.ort.env.wasm.numThreads = window.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    return window.ort;
  }

  function friendly(err) {
    const m = String((err && err.message) || err || "");
    if (/Could not download model \(404\)|\(404\)/.test(m)) {
      return "The face-recognition model file was not found (" + MODELS.arcface + "). The site admin needs to upload arcface_r100.onnx - see DEPLOYMENT.md, step 6.";
    }
    if (/Failed to fetch|NetworkError|Load failed/i.test(m)) {
      return "Could not download the face-recognition model. Check your internet connection (and that the model host allows this website), then try again.";
    }
    if (/memory|allocation|out of/i.test(m)) {
      return "This device ran out of memory while loading the face-recognition model. Close other tabs/apps and try again, or use a laptop.";
    }
    return m || "Could not start the on-device face recognition.";
  }

  // ------------------------------------------------------------ public API
  /** Load everything needed and decide how this device will be used. Safe to call many times. */
  self.prepare = function prepare({ interactive = true } = {}) {
    if (self.mode) return Promise.resolve(self.mode);
    if (self._promise) return self._promise;

    self._promise = (async () => {
      if (interactive) {
        const net = connectionInfo();
        if (net.metered && !self._confirmedDownload) {
          const ok = confirm("FaceSync needs to download its face-recognition model once (about 260 MB). " +
            "You appear to be on mobile data or data-saver. Continue?");
          if (!ok) throw new Error("Face-recognition download cancelled.");
          self._confirmedDownload = true;
        }
      }
      try {
        ui("Preparing face recognition on this device… (first time only)", 0.02);
        const ort = await loadOrt();
        await FaceEngine.Engine.init({
          ort,
          models: MODELS,
          providers: ["webgpu", "wasm"],
          onProgress: (label, p) => ui(`Downloading ${label} model… ${Math.round(p * 100)}%`, p),
        });
        ui("Checking this device's speed…", null);
        const ms = await FaceEngine.Engine.benchmark(2);
        self.info = { backend: FaceEngine.Engine.backend, benchmarkMs: ms };
        self.mode = "device";
        if (ms > SLOW_MS && self.serverFallback) {
          self.mode = "server";       // this device is too slow; the server can do it instead
        }
        self.lastError = null;
        return self.mode;
      } catch (err) {
        self.lastError = friendly(err);
        console.warn("On-device AI failed:", err);
        if (self.serverFallback) { self.mode = "server"; return self.mode; }
        throw new Error(self.lastError);
      } finally {
        ui(null);
        if (!self.mode) self._promise = null;     // allow a retry after a failure
      }
    })();
    return self._promise;
  };

  /** Start loading quietly in the background (skipped on mobile data / data-saver). */
  self.prefetch = function prefetch() {
    if (self.mode || self._promise) return;
    if (connectionInfo().metered) return;
    self.prepare({ interactive: false }).catch(() => {});
  };

  self.usesDevice = () => self.mode === "device";

  /** Text for the small badge on the camera view. */
  self.badge = () => {
    if (self.mode === "device") {
      const b = self.info.backend === "webgpu" ? "GPU" : "CPU";
      return `ON-DEVICE · ${b}` + (self.info.lastMs ? ` · ${Math.round(self.info.lastMs)} ms` : "");
    }
    return self.mode === "server" ? "SERVER" : "";
  };

  /** Analyse a photo (Blob/File). Returns the engine result ({faces:[...]}). */
  self.analyzePhoto = async function analyzePhoto(blob) {
    const img = await FaceEngine.imageFromBlob(blob, 1280);
    return FaceEngine.Engine.analyze(img, { embed: true, liveness: true });
  };

  /** Analyse the current camera frame. */
  self.analyzeCamera = async function analyzeCamera() {
    const img = Camera.captureImageData(640);
    const res = await FaceEngine.Engine.analyze(img, { embed: true, liveness: true });
    self.info.lastMs = res.latency_ms;
    self._failures = 0;
    return res;
  };

  /** Called when a scan throws. After repeated failures on WebGPU, silently switch to the CPU engine. */
  self.noteFailure = async function noteFailure() {
    self._failures += 1;
    if (self._failures < 2) return;
    self._failures = 0;
    if (FaceEngine.Engine.backend === "webgpu") {
      try {
        const E = FaceEngine.Engine;
        E.ready = false; E.loading = null; E.arc = null;
        ui("Switching to the CPU engine…", null);
        await E.init({ ort: window.ort, models: MODELS, providers: ["wasm"],
          onProgress: (label, p) => ui(`Loading ${label} model… ${Math.round(p * 100)}%`, p) });
        self.info.backend = E.backend;
      } catch (e) { console.warn("CPU engine switch failed:", e); }
      finally { ui(null); }
    } else if (self.serverFallback) {
      self.mode = "server";
    }
  };

  return self;
})();
window.DeviceAI = DeviceAI;
