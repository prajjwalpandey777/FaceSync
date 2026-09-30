/* Runtime configuration for the FaceSync frontend.
 *
 * On Vercel this file is REGENERATED at build time by build-config.js from environment variables,
 * so you never edit it by hand there:
 *   FACESYNC_API_URL     = https://your-api.onrender.com                      (required on Vercel)
 *   FACESYNC_ARCFACE_URL = https://huggingface.co/<you>/<repo>/resolve/main/arcface_r100.onnx   (required on Vercel)
 *
 * API_BASE = ""  ->  use the same origin as the page (local dev, where FastAPI serves this folder).
 * ARCFACE_URL    ->  your converted model; locally it is frontend/models/arcface_r100.onnx.
 */
window.FACESYNC_CONFIG = {
  API_BASE: "",
  ARCFACE_URL: "models/arcface_r100.onnx"
};
