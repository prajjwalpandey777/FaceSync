/**
 * Camera helper — works on both laptop webcams and phone cameras.
 * Wire a <video id="scan-video" autoplay playsinline muted></video> element
 * into your attendance-live view (see INTEGRATION_GUIDE.md).
 */

const Camera = {
  stream: null,
  videoEl: null,
  mirrored: true,

  /**
   * Starts the camera. On phones this defaults to user webcam/mirror mode
   * for a natural mirror experience; switchFacing toggles to environment.
   */
  async start(videoElementId, { facingMode = "user" } = {}) {
    this.videoEl = document.getElementById(videoElementId);
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera access is not supported in this browser");
    }

    // Try the requested camera first, then fall back to any camera if that
    // exact facing mode isn't available (common on laptops with one webcam).
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
    } catch (err) {
      this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }

    this.videoEl.srcObject = this.stream;
    await this.videoEl.play();

    // Mirror image for user-facing camera (or laptop webcams) for a natural selfie/mirror experience
    const track = this.stream.getVideoTracks()[0];
    const trackSettings = track?.getSettings ? track.getSettings() : null;
    const isUserFacing = trackSettings?.facingMode === "user" || (!trackSettings?.facingMode && facingMode === "user") || facingMode === "user";
    this.setMirrored(isUserFacing);

    this.canvasEl = document.createElement("canvas");
  },

  setMirrored(mirrored) {
    this.mirrored = mirrored;
    if (this.videoEl) {
      if (mirrored) {
        this.videoEl.style.transform = "scaleX(-1)";
        this.videoEl.style.webkitTransform = "scaleX(-1)";
      } else {
        this.videoEl.style.transform = "none";
        this.videoEl.style.webkitTransform = "none";
      }
    }
  },

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.videoEl) this.videoEl.srcObject = null;
  },

   /** Grabs the current video frame as a JPEG Blob, ready to send to the backend.
   *  Pass maxWidth to downscale first — smaller frames upload faster and the
   *  backend's face detector runs faster on them too. */
  captureFrameBlob(quality = 0.85, maxWidth = null) {
    if (!this.videoEl || !this.videoEl.videoWidth) {
      return Promise.reject(new Error("Camera not ready yet"));
    }
    let width = this.videoEl.videoWidth;
    let height = this.videoEl.videoHeight;
    if (maxWidth && width > maxWidth) {
      height = Math.round(height * (maxWidth / width));
      width = maxWidth;
    }
    this.canvasEl.width = width;
    this.canvasEl.height = height;
    const ctx = this.canvasEl.getContext("2d");
    ctx.drawImage(this.videoEl, 0, 0, width, height);
    return new Promise((resolve) => this.canvasEl.toBlob(resolve, "image/jpeg", quality));
  },
  
  /** Current video frame as raw RGBA pixels ({data, width, height}) for the on-device face engine.
   *  No JPEG encoding/decoding, so nothing is lost and nothing is uploaded. */
  captureImageData(maxWidth = 640) {
    if (!this.videoEl || !this.videoEl.videoWidth) throw new Error("Camera not ready yet");
    let width = this.videoEl.videoWidth;
    let height = this.videoEl.videoHeight;
    if (maxWidth && width > maxWidth) {
      height = Math.round(height * (maxWidth / width));
      width = maxWidth;
    }
    this.canvasEl.width = width;
    this.canvasEl.height = height;
    const ctx = this.canvasEl.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(this.videoEl, 0, 0, width, height);
    const d = ctx.getImageData(0, 0, width, height);
    return { data: d.data, width, height };
  },

  /** Lets the teacher switch between front/rear camera mid-session (mobile). */
  async switchFacing(videoElementId, currentFacing) {
    this.stop();
    const next = currentFacing === "environment" ? "user" : "environment";
    await this.start(videoElementId, { facingMode: next });
    return next;
  },
};
