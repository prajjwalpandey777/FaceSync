"""MiniFASNetV2 ONNX Face Anti-Spoofing (Presentation Attack Detection) Module.

Provides fast, high-accuracy liveness detection using the MiniFASNetV2 1.8M model.
Input: (80, 80) BGR crop with 2.7x scale margin, raw float32 [0.0, 255.0].
Output: Class 1 = REAL FACE; Class 0 & 2 = SPOOF ATTACK.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np


class MiniFASNetV2AntiSpoof:
    """MiniFASNetV2 ONNX predictor for face presentation attack detection."""

    def __init__(
        self,
        model_path: Union[str, Path] = "weights/2.7_80x80_MiniFASNetV2.onnx",
        threshold: float = 0.60,
        scale: float = 2.7,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.scale = scale
        self.net = None

        if self.model_path.is_file():
            try:
                self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

                # Warm-up pass
                dummy = np.zeros((1, 3, 80, 80), dtype=np.float32)
                self.net.setInput(dummy)
                self.net.forward()
            except Exception as e:
                print(f"[!] Warning: Failed to load MiniFASNetV2 ONNX ({self.model_path}): {e}")
                self.net = None
        else:
            print(f"[!] Warning: MiniFASNetV2 ONNX model file not found at {self.model_path}")

    @staticmethod
    def _get_new_box(
        src_w: int, src_h: int, bbox: Tuple[int, int, int, int], scale: float
    ) -> Tuple[int, int, int, int]:
        """Compute 2.7x scaled bounding box with contextual margin."""
        x, y, box_w, box_h = bbox
        scale = min((src_h - 1) / max(box_h, 1), (src_w - 1) / max(box_w, 1), scale)

        new_width = box_w * scale
        new_height = box_h * scale
        center_x = box_w / 2.0 + x
        center_y = box_h / 2.0 + y

        left_top_x = center_x - new_width / 2.0
        left_top_y = center_y - new_height / 2.0
        right_bottom_x = center_x + new_width / 2.0
        right_bottom_y = center_y + new_height / 2.0

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1

        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        return (
            int(max(0, left_top_x)),
            int(max(0, left_top_y)),
            int(min(src_w - 1, right_bottom_x)),
            int(min(src_h - 1, right_bottom_y)),
        )

    def crop_face(
        self, img: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Extract 2.7x scaled face crop resized to 80x80 BGR."""
        src_h, src_w = img.shape[:2]
        x1, y1, x2, y2 = self._get_new_box(src_w, src_h, bbox, self.scale)
        crop = img[y1 : y2 + 1, x1 : x2 + 1]
        if crop.size == 0:
            crop = cv2.resize(img, (80, 80))
        else:
            crop = cv2.resize(crop, (80, 80), interpolation=cv2.INTER_LINEAR)
        return crop

    def check_liveness(
        self, img: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[bool, float, float]:
        """Check liveness of a detected face bounding box in a BGR image.

        Returns:
            (is_real, liveness_score, spoof_score)
        """
        if self.net is None or img is None or img.size == 0:
            # Fallback: if model is disabled or unavailable, default to authentic
            return True, 1.0, 0.0

        crop_80 = self.crop_face(img, bbox)

        # Transpose HWC BGR -> NCHW BGR float32 in [0.0, 255.0]
        blob = np.ascontiguousarray(
            crop_80.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32
        )

        self.net.setInput(blob)
        logits = self.net.forward().squeeze()

        # Numerically stable Softmax
        shift_logits = logits - np.max(logits)
        exps = np.exp(shift_logits)
        probs = exps / np.sum(exps)

        liveness_score = float(probs[1])
        spoof_score = float(probs[0] + probs[2])
        argmax_idx = int(np.argmax(probs))

        is_real = (argmax_idx == 1) and (liveness_score >= self.threshold)
        return is_real, round(liveness_score, 4), round(spoof_score, 4)
