"""Face detection, alignment, and preprocessing for ArcFace IResNet-100."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor


# Standard InsightFace / ArcFace 112x112 reference landmarks (right eye, left eye, nose, right mouth, left mouth)
REFERENCE_FACIAL_POINTS_112x112 = np.array(
    [
        [38.2946, 51.6963],  # right eye
        [73.5318, 51.5014],  # left eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # right mouth corner
        [70.7299, 92.2041],  # left mouth corner
    ],
    dtype=np.float32,
)

DEFAULT_YUNET_PATH = (
    Path(__file__).resolve().parent / "weights" / "face_detection_yunet_2023mar.onnx"
)


class OpenVINOYuNetBackend:
    """Hardware-accelerated YuNet face detector backend using Intel OpenVINO.

    Targets Intel GPU (e.g. Intel Arc 130V) or CPU via OpenVINO runtime.
    Decodes multi-scale anchor outputs (strides 8, 16, 32) and applies NMS.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "GPU",
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        import openvino as ov

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.device_name = device

        self.core = ov.Core()
        model = self.core.read_model(str(model_path))
        # Configure dynamic shape so any resolution is processed without recompilation
        model.reshape([-1, 3, -1, -1])
        self.compiled_model = self.core.compile_model(model, device)
        self.infer_request = self.compiled_model.create_infer_request()
        self.output_names = [out.get_any_name() for out in model.outputs]

        full_dev_name = self.core.get_property(device, "FULL_DEVICE_NAME")
        self.description = f"OpenVINO {device} ({full_dev_name})"

        # Warm-up inference pass to initialize GPU command queues and compile device kernels
        dummy_blob = np.zeros((1, 3, 480, 640), dtype=np.float32)
        self.infer_request.infer({0: dummy_blob})

    def detect(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Detect faces in BGR image.

        Returns np.ndarray of shape [N, 15] or None.
        Row format: [x, y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score]
        """
        ih, iw = img.shape[:2]
        pad_w = ((iw - 1) // 32 + 1) * 32
        pad_h = ((ih - 1) // 32 + 1) * 32

        if pad_w != iw or pad_h != ih:
            pad_img = cv2.copyMakeBorder(
                img, 0, pad_h - ih, 0, pad_w - iw, cv2.BORDER_CONSTANT, value=0
            )
        else:
            pad_img = img

        # Fast HWC BGR uint8 -> NCHW BGR float32
        blob = np.ascontiguousarray(
            pad_img.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32
        )
        self.infer_request.infer({0: blob})
        outputs = {
            name: self.infer_request.get_tensor(name).data
            for name in self.output_names
        }

        faces = []
        for s in (8, 16, 32):
            cols = pad_w // s
            rows = pad_h // s
            cls = outputs[f"cls_{s}"].reshape(-1)
            obj = outputs[f"obj_{s}"].reshape(-1)
            scores = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))
            cand_idx = np.flatnonzero(scores >= self.score_threshold)
            if len(cand_idx) == 0:
                continue

            c = cand_idx % cols
            r = cand_idx // cols
            bbox = outputs[f"bbox_{s}"].reshape(-1, 4)[cand_idx]
            kps = outputs[f"kps_{s}"].reshape(-1, 10)[cand_idx]

            cx = (c + bbox[:, 0]) * s
            cy = (r + bbox[:, 1]) * s
            w = np.exp(bbox[:, 2]) * s
            h = np.exp(bbox[:, 3]) * s

            batch = np.empty((len(cand_idx), 15), dtype=np.float32)
            batch[:, 0] = cx - w * 0.5
            batch[:, 1] = cy - h * 0.5
            batch[:, 2] = w
            batch[:, 3] = h
            for n in range(5):
                batch[:, 4 + 2 * n] = (kps[:, 2 * n] + c) * s
                batch[:, 5 + 2 * n] = (kps[:, 2 * n + 1] + r) * s
            batch[:, 14] = scores[cand_idx]
            faces.append(batch)

        if not faces:
            return None

        all_faces = np.vstack(faces)
        if len(all_faces) == 1:
            return all_faces

        boxes = [
            [int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in all_faces
        ]
        keep = cv2.dnn.NMSBoxes(
            boxes,
            all_faces[:, 14].tolist(),
            self.score_threshold,
            self.nms_threshold,
            top_k=self.top_k,
        )
        if len(keep) > 0:
            return all_faces[keep.flatten()]
        return None


class FaceDetector:
    """Robust face detector and 112x112 ArcFace aligner.

    Supports:
        - OpenVINO GPU backend (Intel Arc / Iris Xe GPU)
        - OpenCV YuNet CPU backend
        - OpenCV Haar Cascade CPU fallback
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_YUNET_PATH,
        backend: str = "auto",
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        min_face_area: float = 2000.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.min_face_area = min_face_area
        self.backend_name = "None"
        self.openvino_backend = None
        self.opencv_detector = None
        self.cascade = None

        backend_req = backend.lower() if backend else "auto"

        # 1. Try OpenVINO GPU backend if requested or auto
        if backend_req in ("auto", "openvino", "gpu", "xpu") and self.model_path.is_file():
            try:
                import openvino as ov
                core = ov.Core()
                if "GPU" in core.available_devices:
                    self.openvino_backend = OpenVINOYuNetBackend(
                        model_path=self.model_path,
                        device="GPU",
                        score_threshold=score_threshold,
                        nms_threshold=nms_threshold,
                    )
                    self.backend_name = self.openvino_backend.description
            except Exception as e:
                if backend_req in ("openvino", "gpu", "xpu"):
                    print(f"  [!] OpenVINO GPU requested but initialization failed: {e}")
                self.openvino_backend = None

        # 2. Fall back to OpenCV YuNet (CPU) if OpenVINO GPU was not initialized
        if self.openvino_backend is None and self.model_path.is_file():
            try:
                self.opencv_detector = cv2.FaceDetectorYN.create(
                    model=str(self.model_path),
                    config="",
                    input_size=(320, 320),
                    score_threshold=self.score_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=5000,
                )
                self.backend_name = "OpenCV YuNet (CPU)"
            except Exception as e:
                print(f"  [!] OpenCV YuNet failed: {e}")
                self.opencv_detector = None

        # 3. Fall back to Haar Cascade if neural detectors are unavailable
        if (
            self.openvino_backend is None
            and self.opencv_detector is None
            and hasattr(cv2, "CascadeClassifier")
        ):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            self.backend_name = "OpenCV Haar Cascade (CPU)"

    def _align_face_landmarks(
        self, img: np.ndarray, landmarks: np.ndarray
    ) -> Optional[np.ndarray]:
        """Align face to 112x112 using similarity transform on 5 landmarks."""
        try:
            tfm, _ = cv2.estimateAffinePartial2D(
                landmarks.astype(np.float32),
                REFERENCE_FACIAL_POINTS_112x112,
                method=cv2.LMEDS,
            )
            if tfm is None:
                return None
            aligned = cv2.warpAffine(
                img, tfm, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0.0
            )
            return aligned
        except Exception:
            return None

    def _crop_face_bbox(
        self, img: np.ndarray, bbox: Tuple[int, int, int, int], margin: float = 0.2
    ) -> np.ndarray:
        """Crop face bounding box with contextual margin and resize to 112x112."""
        ih, iw = img.shape[:2]
        x, y, w, h = bbox
        mx = int(w * margin)
        my = int(h * margin)

        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(iw, x + w + mx)
        y2 = min(ih, y + h + my)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = cv2.resize(img, (112, 112))
        else:
            crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        return crop

    def to_input_tensor(
        self, aligned_bgr: np.ndarray, device: Optional[torch.device] = None
    ) -> Tensor:
        """Convert 112x112 BGR face image to normalized ArcFace tensor [1, 3, 112, 112].

        ArcFace expects RGB format normalized to [-1.0, 1.0] via (x - 127.5) / 128.0.
        """
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        tensor = (tensor - 127.5) / 128.0
        tensor = tensor.unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def detect_faces(
        self,
        img: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> List[Dict]:
        """Detect all faces in an image.

        Returns a list of dicts with keys:
            - 'bbox': [x, y, w, h]
            - 'landmarks': 5x2 array (if available, else None)
            - 'score': confidence score float
            - 'area': bounding box area
            - 'tensor': PyTorch Tensor [1, 3, 112, 112]
            - 'aligned_bgr': 112x112 np.ndarray
        """
        if img is None or img.size == 0:
            return []

        ih, iw = img.shape[:2]
        results = []

        faces = None
        if self.openvino_backend is not None:
            faces = self.openvino_backend.detect(img)
        elif self.opencv_detector is not None:
            self.opencv_detector.setInputSize((iw, ih))
            _, faces = self.opencv_detector.detect(img)

        if faces is not None:
            for f in faces:
                x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                area = float(w * h)
                score = float(f[14])
                if area < self.min_face_area:
                    continue

                landmarks = np.array(
                    [
                        [f[4], f[5]],
                        [f[6], f[7]],
                        [f[8], f[9]],
                        [f[10], f[11]],
                        [f[12], f[13]],
                    ],
                    dtype=np.float32,
                )

                aligned = self._align_face_landmarks(img, landmarks)
                if aligned is None:
                    aligned = self._crop_face_bbox(img, (x, y, w, h))

                tensor = self.to_input_tensor(aligned, device=device)
                results.append(
                    {
                        "bbox": (x, y, w, h),
                        "landmarks": landmarks,
                        "score": score,
                        "area": area,
                        "aligned_bgr": aligned,
                        "tensor": tensor,
                    }
                )

        elif self.cascade is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30)
            )
            for (x, y, w, h) in faces:
                area = float(w * h)
                if area < self.min_face_area:
                    continue
                aligned = self._crop_face_bbox(img, (x, y, w, h))
                tensor = self.to_input_tensor(aligned, device=device)
                results.append(
                    {
                        "bbox": (int(x), int(y), int(w), int(h)),
                        "landmarks": None,
                        "score": 0.9,
                        "area": area,
                        "aligned_bgr": aligned,
                        "tensor": tensor,
                    }
                )

        # Sort by area descending (largest faces first)
        results.sort(key=lambda r: r["area"], reverse=True)
        return results

    def detect_primary_face(
        self,
        img: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> Optional[Dict]:
        """Detect the single most dominant face in the image.

        Filters out small/background faces:
        If multiple faces exist, selects the largest face that is prominently visible.
        """
        faces = self.detect_faces(img, device=device)
        if not faces:
            return None

        # Primary face is the largest face
        primary = faces[0]

        # In multi-face images, ensure primary face is substantial (not background)
        ih, iw = img.shape[:2]
        img_area = float(ih * iw)
        if primary["area"] < 0.005 * img_area and primary["area"] < 4000:
            # Face is negligibly tiny relative to image
            return None

        return primary
