"""Face-recognition business logic.

Heavy lifting (detection, liveness, ArcFace embeddings) happens in the user's browser
(default) or, optionally, in-process on the server (INFERENCE_MODE=local). This module
validates embeddings, applies the enrollment / attendance rules and does the
cosine-similarity matching.
"""
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
from fastapi import HTTPException, status
from PIL import Image, ImageOps

from app.config import settings
from app.inference import AnalyzeResult, get_inference  # noqa: F401 (server-side fallback only)

# Reject decompression bombs (a tiny file that expands to gigapixels).
Image.MAX_IMAGE_PIXELS = 60_000_000
MAX_SIDE_PX = 1600  # larger photos are downscaled before being sent to the GPU


def prepare_image(image_bytes: bytes) -> bytes:
    """Validate an uploaded image and normalise it for inference.

    * rejects empty / oversized / non-image data
    * bakes in EXIF rotation (phone photos) and downsizes huge photos to save upload time
    * small, already-upright JPEGs (live camera frames) pass through untouched
    """
    if not image_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty image data")
    if len(image_bytes) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"Image is too large (max {settings.MAX_UPLOAD_MB} MB)"
        )
    try:
        img = Image.open(BytesIO(image_bytes))
        img.load()
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Could not decode image (unsupported format)")

    needs_rotation = img.getexif().get(0x0112, 1) not in (None, 1)
    too_big = max(img.size) > MAX_SIDE_PX
    if img.format == "JPEG" and not needs_rotation and not too_big:
        return image_bytes

    img = ImageOps.exif_transpose(img).convert("RGB")
    if too_big:
        img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.LANCZOS)
    out = BytesIO()
    img.save(out, format="JPEG", quality=92)
    return out.getvalue()


def encode_single_face(image_bytes: bytes) -> List[float]:
    """Enrollment: return a 512-d embedding for a photo that contains exactly one live face."""
    result = get_inference().analyze(prepare_image(image_bytes), embed=True, liveness=True)

    if len(result.faces) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No face detected in the enrollment photo")
    if len(result.faces) > 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Multiple faces detected - please use a photo with only the student's face",
        )

    face = result.faces[0]
    if not face.is_real:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Enrollment photo failed liveness check (liveness: {face.liveness:.2f} < "
            f"{result.liveness_threshold}). Please use a real, live photo.",
        )
    if not face.embedding:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Face-recognition service returned no embedding")
    return face.embedding


def analyze_frame(image_bytes: bytes) -> AnalyzeResult:
    """Live attendance: all faces in a camera frame, with liveness verdicts and embeddings."""
    return get_inference().analyze(prepare_image(image_bytes), embed=True, liveness=True)


def best_match(
    unknown_encoding: List[float],
    candidates: List[Tuple[int, List[float]]],
    threshold: float = 0.45,
) -> Optional[Tuple[int, float]]:
    """Cosine-similarity match of one embedding against (student_id, embedding) candidates.

    Returns (student_id, confidence_percent) or None when nothing reaches `threshold`.
    """
    valid = [c for c in candidates if len(c[1]) == len(unknown_encoding)]
    if not valid:
        return None

    query = np.asarray(unknown_encoding, dtype=np.float32)
    gallery = np.asarray([c[1] for c in valid], dtype=np.float32)
    # Re-normalise defensively so the dot product is a true cosine similarity.
    query = query / max(float(np.linalg.norm(query)), 1e-9)
    gallery = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-9)

    sims = gallery @ query
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim < threshold:
        return None

    # Map cosine similarity [threshold, 1.0] -> confidence percent [65.0, 99.9]
    normalized = (best_sim - threshold) / max(1e-5, 1.0 - threshold)
    confidence = round(float(np.clip(65.0 + normalized * 34.9, 65.0, 99.9)), 1)
    return valid[best_idx][0], confidence


def normalize_embedding(vec: List[float]) -> List[float]:
    """Validate an embedding computed in the browser and L2-normalise it."""
    if len(vec) != settings.EMBEDDING_DIM:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Embedding must have {settings.EMBEDDING_DIM} numbers")
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(arr).all() or norm < 1e-6:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid embedding")
    return (arr / norm).tolist()
