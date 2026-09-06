/**
 * Camera helper — works on both laptop webcams and phone cameras.
 * Wire a <video id="scan-video" autoplay playsinline muted></video> element
 * into your attendance-live view (see INTEGRATION_GUIDE.md).
 */

const Camera = {
  stream: null,
  videoEl: null,
  canvasEl: null,

  /**
   * Starts the camera. On phones this requests the rear camera by default
   * (facingMode: "environment") since that's what you'd point at a room of
   * students; pass "user" for laptops using the front webcam if preferred.
   */
  async start(videoElementId, { facingMode = "environment" } = {}) {
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

    this.canvasEl = document.createElement("canvas");
  },

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.videoEl) this.videoEl.srcObject = null;
  },

  /** Grabs the current video frame as a JPEG Blob, ready to send to the backend. */
  captureFrameBlob(quality = 0.85) {
    if (!this.videoEl || !this.videoEl.videoWidth) {
      return Promise.reject(new Error("Camera not ready yet"));
    }
    this.canvasEl.width = this.videoEl.videoWidth;
    this.canvasEl.height = this.videoEl.videoHeight;
    const ctx = this.canvasEl.getContext("2d");
    ctx.drawImage(this.videoEl, 0, 0, this.canvasEl.width, this.canvasEl.height);
    return new Promise((resolve) => this.canvasEl.toBlob(resolve, "image/jpeg", quality));
  },

  /** Lets the teacher switch between front/rear camera mid-session (mobile). */
  async switchFacing(videoElementId, currentFacing) {
    this.stop();
    const next = currentFacing === "environment" ? "user" : "environment";
    await this.start(videoElementId, { facingMode: next });
    return next;
  },
};
