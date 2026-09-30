/* FaceSync on-device face engine.
 *
 * Runs the three FaceSync models (YuNet, MiniFASNetV2, your ArcFace) inside the visitor's browser
 * (WebGPU when available, otherwise WebAssembly on the CPU) using ONNX Runtime Web:
 *     YuNet (detect + 5 landmarks)  ->  MiniFASNetV2 (liveness)  ->  ArcFace (512-d embedding)
 * Only the 512 numbers and the liveness verdict are sent to the server - never the camera image.
 *
 * All pre/post-processing is plain JavaScript on RGBA pixel arrays so the same file also runs
 * under Node.js (that is how it is tested against the Python pipeline).
 *
 * Image type used everywhere:  { data: Uint8ClampedArray (RGBA), width, height }
 */
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- constants (mirror the Python code)
  const DET_SIZE = 640;                // the YuNet ONNX has a fixed 640x640 input
  const DET_STRIDES = [8, 16, 32];
  const DET_SCORE_THRESHOLD = 0.5;
  const DET_NMS_THRESHOLD = 0.3;
  const MIN_FACE_AREA = 2000;          // px^2 in the source frame (same as face_detector.py)
  const SPOOF_SCALE = 2.7;
  const SPOOF_SIZE = 80;
  const ARC_SIZE = 112;
  const REF_POINTS = [                 // InsightFace 112x112 reference landmarks
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041],
  ];

  // ---------------------------------------------------------------- small image helpers
  /** OpenCV INTER_LINEAR-compatible bilinear sample of one output pixel channel. */
  function bilinearResize(img, x1, y1, x2, y2, outW, outH) {
    // Resize the crop [x1..x2) x [y1..y2) of `img` to outW x outH (RGBA).
    const cw = x2 - x1, ch = y2 - y1;
    const out = new Uint8ClampedArray(outW * outH * 4);
    const sx = cw / outW, sy = ch / outH;
    const W = img.width, H = img.height, src = img.data;
    for (let y = 0; y < outH; y++) {
      let fy = (y + 0.5) * sy - 0.5 + y1;
      let y0 = Math.floor(fy); const wy = fy - y0;
      let ya = Math.min(Math.max(y0, y1), y2 - 1), yb = Math.min(Math.max(y0 + 1, y1), y2 - 1);
      ya = Math.min(Math.max(ya, 0), H - 1); yb = Math.min(Math.max(yb, 0), H - 1);
      for (let x = 0; x < outW; x++) {
        let fx = (x + 0.5) * sx - 0.5 + x1;
        let x0 = Math.floor(fx); const wx = fx - x0;
        let xa = Math.min(Math.max(x0, x1), x2 - 1), xb = Math.min(Math.max(x0 + 1, x1), x2 - 1);
        xa = Math.min(Math.max(xa, 0), W - 1); xb = Math.min(Math.max(xb, 0), W - 1);
        const o = (y * outW + x) * 4;
        for (let c = 0; c < 3; c++) {
          const p00 = src[(ya * W + xa) * 4 + c], p01 = src[(ya * W + xb) * 4 + c];
          const p10 = src[(yb * W + xa) * 4 + c], p11 = src[(yb * W + xb) * 4 + c];
          const top = p00 + (p01 - p00) * wx, bot = p10 + (p11 - p10) * wx;
          out[o + c] = Math.round(top + (bot - top) * wy);
        }
        out[o + 3] = 255;
      }
    }
    return { data: out, width: outW, height: outH };
  }

  /** Downscale so the longest side is <= maxSide (no upscaling). */
  function limitSize(img, maxSide) {
    const m = Math.max(img.width, img.height);
    if (m <= maxSide) return { img, scale: 1 };
    const s = maxSide / m;
    const w = Math.max(1, Math.round(img.width * s)), h = Math.max(1, Math.round(img.height * s));
    return { img: bilinearResize(img, 0, 0, img.width, img.height, w, h), scale: w / img.width };
  }

  // ---------------------------------------------------------------- YuNet detection
  function makeDetInput(img) {
    // Place the (possibly downscaled) frame in the top-left of a zero-padded 640x640 BGR CHW tensor,
    // exactly like the Python OpenVINO backend pads to a multiple of 32.
    const { img: small, scale } = limitSize(img, DET_SIZE);
    const N = DET_SIZE * DET_SIZE;
    const t = new Float32Array(3 * N);              // zeros = padding
    const W = small.width, H = small.height, d = small.data;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const s = (y * W + x) * 4, p = y * DET_SIZE + x;
        t[p] = d[s + 2];             // B
        t[N + p] = d[s + 1];         // G
        t[2 * N + p] = d[s];         // R
      }
    }
    return { tensor: t, scale };
  }

  function iou(a, b) {
    const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const uni = a[2] * a[3] + b[2] * b[3] - inter;
    return uni > 0 ? inter / uni : 0;
  }

  function decodeDetections(out, scale) {
    const cands = [];
    for (const s of DET_STRIDES) {
      const cols = DET_SIZE / s;
      const cls = out["cls_" + s].data, obj = out["obj_" + s].data;
      const bbox = out["bbox_" + s].data, kps = out["kps_" + s].data;
      const n = cls.length;
      for (let i = 0; i < n; i++) {
        const sc = Math.sqrt(Math.min(Math.max(cls[i], 0), 1) * Math.min(Math.max(obj[i], 0), 1));
        if (sc < DET_SCORE_THRESHOLD) continue;
        const c = i % cols, r = Math.floor(i / cols);
        const cx = (c + bbox[i * 4]) * s, cy = (r + bbox[i * 4 + 1]) * s;
        const w = Math.exp(bbox[i * 4 + 2]) * s, h = Math.exp(bbox[i * 4 + 3]) * s;
        const lm = [];
        for (let k = 0; k < 5; k++) lm.push([(kps[i * 10 + 2 * k] + c) * s / scale, (kps[i * 10 + 2 * k + 1] + r) * s / scale]);
        cands.push({ bbox: [(cx - w / 2) / scale, (cy - h / 2) / scale, w / scale, h / scale], landmarks: lm, score: sc });
      }
    }
    cands.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const c of cands) {
      if (keep.every((k) => iou(k.bbox, c.bbox) <= DET_NMS_THRESHOLD)) keep.push(c);
    }
    return keep;
  }

  // ---------------------------------------------------------------- liveness (MiniFASNetV2)
  function spoofBox(srcW, srcH, bbox, scale) {
    let [x, y, bw, bh] = bbox;
    scale = Math.min((srcH - 1) / Math.max(bh, 1), (srcW - 1) / Math.max(bw, 1), scale);
    const nw = bw * scale, nh = bh * scale, cx = bw / 2 + x, cy = bh / 2 + y;
    let lx = cx - nw / 2, ly = cy - nh / 2, rx = cx + nw / 2, ry = cy + nh / 2;
    if (lx < 0) { rx -= lx; lx = 0; }
    if (ly < 0) { ry -= ly; ly = 0; }
    if (rx > srcW - 1) { lx -= rx - srcW + 1; rx = srcW - 1; }
    if (ry > srcH - 1) { ly -= ry - srcH + 1; ry = srcH - 1; }
    return [Math.trunc(Math.max(0, lx)), Math.trunc(Math.max(0, ly)),
      Math.trunc(Math.min(srcW - 1, rx)), Math.trunc(Math.min(srcH - 1, ry))];
  }

  function spoofInput(img, bbox) {
    const [x1, y1, x2, y2] = spoofBox(img.width, img.height, bbox.map(Math.trunc), SPOOF_SCALE);
    const crop = bilinearResize(img, x1, y1, x2 + 1, y2 + 1, SPOOF_SIZE, SPOOF_SIZE);
    const N = SPOOF_SIZE * SPOOF_SIZE, t = new Float32Array(3 * N), d = crop.data;
    for (let p = 0; p < N; p++) {           // BGR, raw 0..255
      t[p] = d[p * 4 + 2]; t[N + p] = d[p * 4 + 1]; t[2 * N + p] = d[p * 4];
    }
    return t;
  }

  function softmax(v) {
    const m = Math.max(...v), e = v.map((x) => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / s);
  }

  // ---------------------------------------------------------------- alignment for ArcFace
  /** Least-squares similarity transform (scale+rotation+translation) mapping src -> dst (Umeyama). */
  function similarityTransform(src, dst) {
    const n = src.length;
    let sx = 0, sy = 0, dx = 0, dy = 0;
    for (let i = 0; i < n; i++) { sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]; }
    sx /= n; sy /= n; dx /= n; dy /= n;
    let a = 0, b = 0, den = 0;
    for (let i = 0; i < n; i++) {
      const px = src[i][0] - sx, py = src[i][1] - sy, qx = dst[i][0] - dx, qy = dst[i][1] - dy;
      a += px * qx + py * qy;
      b += px * qy - py * qx;
      den += px * px + py * py;
    }
    if (den < 1e-9) return null;
    const A = a / den, B = b / den;                       // [ A -B ; B A ]
    return [A, -B, dx - (A * sx - B * sy), B, A, dy - (B * sx + A * sy)];
  }

  /** warpAffine (bilinear, zero border) of `img` by forward matrix M into a size x size image. */
  function warpAffine(img, M, size) {
    const [a, b, c, d, e, f] = M;
    const det = a * e - b * d;
    const ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
    const ic = -(ia * c + ib * f), iff = -(id * c + ie * f);
    const out = new Uint8ClampedArray(size * size * 4);
    const W = img.width, H = img.height, s = img.data;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const fx = ia * x + ib * y + ic, fy = id * x + ie * y + iff;
        const x0 = Math.floor(fx), y0 = Math.floor(fy), wx = fx - x0, wy = fy - y0;
        const o = (y * size + x) * 4;
        for (let ch = 0; ch < 3; ch++) {
          const px = (xx, yy) => (xx < 0 || yy < 0 || xx >= W || yy >= H) ? 0 : s[(yy * W + xx) * 4 + ch];
          const top = px(x0, y0) * (1 - wx) + px(x0 + 1, y0) * wx;
          const bot = px(x0, y0 + 1) * (1 - wx) + px(x0 + 1, y0 + 1) * wx;
          out[o + ch] = Math.round(top * (1 - wy) + bot * wy);
        }
        out[o + 3] = 255;
      }
    }
    return { data: out, width: size, height: size };
  }

  function alignFace(img, face) {
    const M = similarityTransform(face.landmarks, REF_POINTS);
    if (M) return warpAffine(img, M, ARC_SIZE);
    const [x, y, w, h] = face.bbox.map(Math.trunc);            // fallback: padded bbox crop
    const mx = Math.trunc(w * 0.2), my = Math.trunc(h * 0.2);
    return bilinearResize(img, Math.max(0, x - mx), Math.max(0, y - my),
      Math.min(img.width, x + w + mx), Math.min(img.height, y + h + my), ARC_SIZE, ARC_SIZE);
  }

  function arcInput(aligned) {                                   // RGB, (x-127.5)/128, CHW
    const N = ARC_SIZE * ARC_SIZE, t = new Float32Array(3 * N), d = aligned.data;
    for (let p = 0; p < N; p++) {
      t[p] = (d[p * 4] - 127.5) / 128; t[N + p] = (d[p * 4 + 1] - 127.5) / 128; t[2 * N + p] = (d[p * 4 + 2] - 127.5) / 128;
    }
    return t;
  }

  function l2normalize(v) {
    let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    s = Math.sqrt(s) || 1e-9;
    return Array.from(v, (x) => Math.round((x / s) * 1e6) / 1e6);
  }

  // ---------------------------------------------------------------- the engine
  const Engine = {
    ort: null, yunet: null, spoof: null, arc: null,
    backend: "none",            // "webgpu" | "wasm" | "node"
    ready: false, loading: null,
    livenessThreshold: 0.6, livenessEnabled: true,
    benchmarkMs: null,

    /** cfg: { ort, models: {yunet, spoof, arcface}, providers?, onProgress? }
     *  Each model is a URL (browser) or a path/Uint8Array (node). */
    init(cfg) {
      if (this.loading) return this.loading;
      this.loading = (async () => {
        this.ort = cfg.ort;
        const providers = cfg.providers || ["webgpu", "wasm"];
        const prog = (label) => (p) => cfg.onProgress && cfg.onProgress(label, p);
        // Browser: models are URLs (downloaded once, then kept in the browser cache). Node/tests: file bytes or paths.
        const getBytes = async (src, label) =>
          (typeof src === "string" && !cfg.node) ? fetchCached(src, prog(label)) : src;
        const make = (bytes, provs) =>
          this.ort.InferenceSession.create(bytes, { executionProviders: provs, graphOptimizationLevel: "all" });

        // The two small models always use WebAssembly (fast enough, and avoids WebGPU quirks).
        this.yunet = await make(await getBytes(cfg.models.yunet, "detector"), cfg.node ? ["cpu"] : ["wasm"]);
        this.spoof = await make(await getBytes(cfg.models.spoof, "liveness"), cfg.node ? ["cpu"] : ["wasm"]);

        // The big recognition model: download once, then try WebGPU first and fall back to WebAssembly.
        let arcBytes = await getBytes(cfg.models.arcface, "recognition");
        let lastErr = null;
        for (const p of (cfg.node ? [["cpu"]] : providers.map((x) => [x]))) {
          try {
            this.arc = await make(arcBytes, p);
            this.backend = cfg.node ? "node" : p[0];
            lastErr = null; break;
          } catch (e) { lastErr = e; this.arc = null; }
        }
        arcBytes = null;                       // let the browser free the 260 MB download buffer
        if (!this.arc) throw lastErr || new Error("Could not load the recognition model");
        this.ready = true;
      })();
      this.loading.catch(() => { this.loading = null; });
      return this.loading;
    },

    /** Run the whole pipeline on one RGBA image. opts: {embed, liveness} */
    analyze(img, opts = {}) {
      // Runs are queued one after another: two frames are never processed at the same time.
      const run = () => this._analyze(img, opts);
      const p = (this._queue || Promise.resolve()).then(run, run);
      this._queue = p.catch(() => {});
      return p;
    },

    async _analyze(img, opts = {}) {
      if (!this.ready) throw new Error("Engine not ready");
      const embed = opts.embed !== false, liveness = opts.liveness !== false && this.livenessEnabled;
      const t0 = (typeof performance !== "undefined" ? performance : Date).now();
      const ort = this.ort;

      const { tensor, scale } = makeDetInput(img);
      const out = await this.yunet.run({ [this.yunet.inputNames[0]]: new ort.Tensor("float32", tensor, [1, 3, DET_SIZE, DET_SIZE]) });
      let faces = decodeDetections(out, scale).filter((f) => f.bbox[2] * f.bbox[3] >= MIN_FACE_AREA);
      faces.sort((a, b) => b.bbox[2] * b.bbox[3] - a.bbox[2] * a.bbox[3]);

      for (const f of faces) {
        f.is_real = true; f.liveness = 1.0; f.embedding = null;
        if (liveness) {
          const r = await this.spoof.run({ [this.spoof.inputNames[0]]: new ort.Tensor("float32", spoofInput(img, f.bbox), [1, 3, SPOOF_SIZE, SPOOF_SIZE]) });
          const probs = softmax(Array.from(r[this.spoof.outputNames[0]].data));
          const live = probs[1];
          f.liveness = Math.round(live * 1e4) / 1e4;
          f.is_real = probs.indexOf(Math.max(...probs)) === 1 && live >= this.livenessThreshold;
        }
      }
      if (embed) {
        for (const f of faces) {
          if (!f.is_real) continue;                             // no GPU/CPU time on spoofs
          const aligned = alignFace(img, f);
          const r = await this.arc.run({ [this.arc.inputNames[0]]: new ort.Tensor("float32", arcInput(aligned), [1, 3, ARC_SIZE, ARC_SIZE]) });
          f.embedding = l2normalize(r[this.arc.outputNames[0]].data);
        }
      }
      return {
        faces: faces.map((f) => ({ bbox: f.bbox.map(Math.trunc), score: Math.round(f.score * 1e4) / 1e4,
          is_real: f.is_real, liveness: f.liveness, embedding: f.embedding, landmarks: f.landmarks })),
        frame_width: img.width, frame_height: img.height,
        latency_ms: Math.round(((typeof performance !== "undefined" ? performance : Date).now() - t0) * 10) / 10,
        liveness_enabled: liveness, liveness_threshold: this.livenessThreshold, device: this.backend,
      };
    },

    /** Time the full pipeline on a synthetic frame (includes one recognition pass). */
    async benchmark(runs = 3) {
      const w = 480, h = 360, data = new Uint8ClampedArray(w * h * 4).fill(128);
      const times = [];
      // Warm-up: first run compiles shaders / allocates memory and is not representative.
      await this.arc.run({ [this.arc.inputNames[0]]: new this.ort.Tensor("float32", new Float32Array(3 * ARC_SIZE * ARC_SIZE), [1, 3, ARC_SIZE, ARC_SIZE]) });
      for (let i = 0; i < runs; i++) {
        const t = (typeof performance !== "undefined" ? performance : Date).now();
        await this.yunet.run({ [this.yunet.inputNames[0]]: new this.ort.Tensor("float32", makeDetInput({ data, width: w, height: h }).tensor, [1, 3, DET_SIZE, DET_SIZE]) });
        await this.spoof.run({ [this.spoof.inputNames[0]]: new this.ort.Tensor("float32", new Float32Array(3 * SPOOF_SIZE * SPOOF_SIZE), [1, 3, SPOOF_SIZE, SPOOF_SIZE]) });
        await this.arc.run({ [this.arc.inputNames[0]]: new this.ort.Tensor("float32", new Float32Array(3 * ARC_SIZE * ARC_SIZE), [1, 3, ARC_SIZE, ARC_SIZE]) });
        times.push((typeof performance !== "undefined" ? performance : Date).now() - t);
      }
      times.sort((a, b) => a - b);
      this.benchmarkMs = Math.round(times[Math.floor(times.length / 2)]);
      return this.benchmarkMs;
    },
  };

  // ---------------------------------------------------------------- browser-only helpers
  /** Download a model once and keep it in the browser cache (so the big file is fetched a single time). */
  async function fetchCached(url, onProgress) {
    let cache = null;
    try { cache = await caches.open("facesync-models-v1"); } catch (_) {}
    if (cache) {
      const hit = await cache.match(url);
      if (hit) return new Uint8Array(await hit.arrayBuffer());
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not download model (${res.status}) from ${url}`);
    const total = Number(res.headers.get("content-length")) || 0;
    const reader = res.body.getReader(), chunks = []; let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); got += value.length;
      if (onProgress && total) onProgress(got / total);
    }
    const buf = new Uint8Array(got); let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    if (cache) { try { await cache.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } })); } catch (_) {} }
    return buf;
  }

  /** Convert a Blob/File (photo) into an RGBA image, honouring EXIF rotation. Browser only. */
  async function imageFromBlob(blob, maxSide = 1280) {
    const bmp = await createImageBitmap(blob, { imageOrientation: "from-image" });
    const s = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
    const w = Math.max(1, Math.round(bmp.width * s)), h = Math.max(1, Math.round(bmp.height * s));
    const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
    const ctx = cv.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0, w, h);
    if (bmp.close) bmp.close();
    const d = ctx.getImageData(0, 0, w, h);
    return { data: d.data, width: w, height: h };
  }

  const api = { Engine, imageFromBlob, _internals: {
    bilinearResize, makeDetInput, decodeDetections, spoofInput, similarityTransform, warpAffine, alignFace, arcInput, l2normalize, limitSize,
  } };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FaceEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
