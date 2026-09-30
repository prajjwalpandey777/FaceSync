"""In-process inference (INFERENCE_MODE=local): torch + OpenVINO/OpenCV on your own machine.

This is the original FaceSync runtime (PyTorch + OpenVINO/OpenCV), used only when INFERENCE_MODE=local.
Nothing here is imported in production, so the Render server stays light.
"""
from __future__ import annotations

import time
from io import BytesIO
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import HTTPException, status
from PIL import Image, ImageOps

from app.config import settings
from app.device_runtime import get_runtime
from app.inference import AnalyzeResult, FaceResult


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty image data")
    try:
        pil_img = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes)))
        return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Could not decode image (unsupported format)")
        return img


class LocalInferenceClient:
    def __init__(self) -> None:
        self.runtime = get_runtime()

    def analyze(self, image_bytes: bytes, *, embed: bool = True, liveness: bool = True) -> AnalyzeResult:
        rt = self.runtime
        t0 = time.perf_counter()
        img = _decode_image_bytes(image_bytes)
        ih, iw = img.shape[:2]

        detected = rt.detector.detect_faces(img, device=rt.arcface_device)
        use_liveness = liveness and rt.anti_spoof is not None and settings.ANTI_SPOOF_ENABLED

        faces = []
        for f in detected:
            if use_liveness:
                is_real, live_score, _ = rt.anti_spoof.check_liveness(img, f["bbox"])
            else:
                is_real, live_score = True, 1.0
            bx, by, bw, bh = f["bbox"]
            lms = [] if f.get("landmarks") is None else [[float(p[0]) / iw, float(p[1]) / ih] for p in f["landmarks"]]
            faces.append(
                (
                    f,
                    FaceResult(
                        bbox=[int(bx), int(by), int(bw), int(bh)],
                        norm_bbox=[bx / iw, by / ih, bw / iw, bh / ih],
                        landmarks=lms,
                        score=round(float(f["score"]), 4),
                        is_real=bool(is_real),
                        liveness=float(live_score),
                    ),
                )
            )

        if embed:
            todo = [(raw, res) for raw, res in faces if res.is_real]
            if todo:
                batch = torch.cat([raw["tensor"].to(rt.arcface_device) for raw, _ in todo], dim=0)
                with torch.no_grad():
                    embs = F.normalize(rt.model(batch), p=2, dim=1).cpu().numpy()
                for (_, res), vec in zip(todo, embs):
                    res.embedding = vec.tolist()

        diag: Dict[str, Any] = rt.get_diagnostics()
        return AnalyzeResult(
            faces=[res for _, res in faces],
            frame_width=iw,
            frame_height=ih,
            device=str(rt.arcface_device),
            gpu_name=diag.get("gpu_name", ""),
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            liveness_enabled=use_liveness,
            liveness_threshold=settings.ANTI_SPOOF_THRESHOLD,
        )

    def health(self) -> Dict[str, Any]:
        return self.runtime.get_diagnostics()
