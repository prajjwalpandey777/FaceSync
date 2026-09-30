"""Hardware runtime abstraction and model lifecycle manager.
Supports automatic selection between:
- Intel Arc GPU (OpenVINO GPU detector + PyTorch XPU ArcFace)
- NVIDIA GPU (OpenCV/ONNX detector + PyTorch CUDA ArcFace)
- CPU Fallback (OpenCV YuNet CPU detector + PyTorch CPU ArcFace)
"""

from pathlib import Path
import sys
from typing import Dict, Optional, Tuple
import torch
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backbone.load_model import load_iresnet100, resolve_device
from face_detector import FaceDetector
from anti_spoof import MiniFASNetV2AntiSpoof
from app.config import settings

class HardwareRuntime:
    """Singleton managing compute devices and pre-loaded models."""
    _instance: Optional["HardwareRuntime"] = None

    def __init__(self) -> None:
        self.device_name = "unknown"
        self.arcface_device: Optional[torch.device] = None
        self.detector: Optional[FaceDetector] = None
        self.model: Optional[torch.nn.Module] = None
        self.anti_spoof: Optional[MiniFASNetV2AntiSpoof] = None
        self.initialized = False

    @classmethod
    def get_instance(cls) -> "HardwareRuntime":
        if cls._instance is None:
            cls._instance = HardwareRuntime()
        return cls._instance

    def initialize(
        self,
        device_pref: str = settings.DEVICE,
        detector_backend_pref: str = settings.DETECTOR_BACKEND,
    ) -> None:
        if self.initialized:
            return

        print("[*] Initializing FaceSync Hardware Runtime...")

        # 1. Resolve PyTorch compute device for ArcFace
        try:
            self.arcface_device = resolve_device(device_pref)
        except Exception as e:
            print(f"  [!] Device preference '{device_pref}' failed: {e}. Falling back to CPU.")
            self.arcface_device = torch.device("cpu")

        # 2. Load ArcFace IResNet-100 model onto accelerator
        print(f"  [+] Loading ArcFace backbone on device: {self.arcface_device}...")
        self.model = load_iresnet100(
            checkpoint_path=settings.CHECKPOINT_PATH,
            device=self.arcface_device,
        )
        self.model.eval()

        # Warm-up pass on ArcFace
        dummy_input = torch.zeros(1, 3, 112, 112, device=self.arcface_device)
        with torch.no_grad():
            _ = self.model(dummy_input)
        if self.arcface_device.type == "xpu":
            torch.xpu.synchronize()
        elif self.arcface_device.type == "cuda":
            torch.cuda.synchronize()

        # 3. Load YuNet Face Detector
        print(f"  [+] Loading Face Detector with backend: {detector_backend_pref}...")
        self.detector = FaceDetector(
            model_path=settings.YUNET_PATH,
            backend=detector_backend_pref,
        )

        # 4. Load MiniFASNetV2 Anti-Spoofing Model
        if settings.ANTI_SPOOF_ENABLED:
            print(f"  [+] Loading MiniFASNetV2 Anti-Spoof model ({settings.ANTI_SPOOF_MODEL_PATH})...")
            self.anti_spoof = MiniFASNetV2AntiSpoof(
                model_path=settings.ANTI_SPOOF_MODEL_PATH,
                threshold=settings.ANTI_SPOOF_THRESHOLD,
            )

        self.initialized = True
        print(f"  [+] FaceSync Runtime Ready!")
        print(f"      - ArcFace Device: {self.arcface_device}")
        print(f"      - Detector Backend: {self.detector.backend_name}")
        if self.anti_spoof and self.anti_spoof.net:
            print(f"      - Anti-Spoof Liveness: Enabled (threshold={settings.ANTI_SPOOF_THRESHOLD})")

    def get_diagnostics(self) -> Dict:
        """Return runtime health and accelerator diagnostics."""
        diag = {
            "initialized": self.initialized,
            "arcface_device": str(self.arcface_device) if self.arcface_device else "None",
            "detector_backend": self.detector.backend_name if self.detector else "None",
            "anti_spoof_enabled": settings.ANTI_SPOOF_ENABLED,
            "anti_spoof_loaded": self.anti_spoof is not None and self.anti_spoof.net is not None,
            "anti_spoof_threshold": settings.ANTI_SPOOF_THRESHOLD,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "xpu_available": getattr(torch, "xpu", None) is not None and torch.xpu.is_available(),
        }

        if self.arcface_device and self.arcface_device.type == "xpu":
            diag["gpu_name"] = torch.xpu.get_device_name(0)
            diag["vram_allocated_mb"] = round(torch.xpu.memory_allocated() / (1024 * 1024), 2)
        elif self.arcface_device and self.arcface_device.type == "cuda":
            diag["gpu_name"] = torch.cuda.get_device_name(0)
            diag["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        else:
            diag["gpu_name"] = "CPU (No accelerator)"
            diag["vram_allocated_mb"] = 0.0

        return diag

def get_runtime() -> HardwareRuntime:
    runtime = HardwareRuntime.get_instance()
    if not runtime.initialized:
        runtime.initialize()
    return runtime
