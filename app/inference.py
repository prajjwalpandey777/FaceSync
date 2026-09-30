"""Face-inference gateway (server side).

INFERENCE_MODE=device (default): the BROWSER runs detection, liveness and ArcFace on the user's own
GPU/CPU and sends only 512 numbers to the API, so the server has nothing to load here.

INFERENCE_MODE=local (optional): the server can also analyse uploaded images in-process with
PyTorch + OpenCV (see requirements-local.txt). The website uses this only as a fallback.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from fastapi import HTTPException, status

from app.config import settings

log = logging.getLogger("facesync.inference")


@dataclass
class FaceResult:
    bbox: Sequence[int]                      # x, y, w, h in pixels
    norm_bbox: Sequence[float]               # x, y, w, h as 0..1 fractions
    landmarks: List[List[float]]             # 5 points, normalised 0..1
    score: float                             # detector confidence
    is_real: bool                            # passed liveness (anti-spoof)
    liveness: float
    embedding: Optional[List[float]] = None  # 512-d, L2-normalised (None if not requested / spoof)


@dataclass
class AnalyzeResult:
    faces: List[FaceResult] = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0
    device: str = "unknown"
    gpu_name: str = ""
    latency_ms: float = 0.0
    liveness_enabled: bool = True
    liveness_threshold: float = 0.6


_client = None


def get_inference():
    """Return the configured inference backend (created lazily, once)."""
    global _client
    if _client is None:
        mode = settings.INFERENCE_MODE.lower()
        if mode == "local":
            from app.local_inference import LocalInferenceClient  # heavy imports live here
            _client = LocalInferenceClient()
        elif mode == "device":
            raise HTTPException(
                status.HTTP_501_NOT_IMPLEMENTED,
                "This server does not analyse images - recognition runs in your browser. "
                "(Set INFERENCE_MODE=local to enable server-side recognition.)",
            )
        else:
            raise RuntimeError(f"Unknown INFERENCE_MODE '{settings.INFERENCE_MODE}' (use 'device' or 'local').")
    return _client
