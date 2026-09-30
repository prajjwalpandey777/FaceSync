"""Strict loader for the official InsightFace ArcFace-Torch IResNet-100 model.

Official checkpoint source (model-zoo folder):
https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d

Download only ``ms1mv3_arcface_r100_fp16/backbone.pth`` from that folder and
save it as ``weights/ms1mv3_arcface_r100_fp16_backbone.pth``.  The
``rank_*_softmax_weight.pt`` files are distributed training classifier shards,
not inference-backbone weights.
"""

from pathlib import Path
from typing import Mapping, Optional, Union
import sys

# Keep the module importable both with ``python -m backbone.load_model`` and
# through editors that execute this file directly.
if __package__ in (None, ""):
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    __package__ = "backbone"

import torch
from torch import Tensor, nn
from .iresnet import iresnet100


OFFICIAL_CHECKPOINT_URL = (
    "https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d"
)
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "weights"
    / "backbone.pth"
)


def select_device() -> torch.device:
    """Prefer CUDA, then Intel XPU, and otherwise use CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _state_dict_from_checkpoint(
    checkpoint: object,
) -> Mapping[str, Tensor]:
    """Accept a plain state dict or a checkpoint containing ``state_dict``."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping containing model tensors")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in state_dict.items()
    ):
        raise TypeError("checkpoint does not contain a valid PyTorch state dict")
    return state_dict


def verify_checkpoint_compatibility(
    model: nn.Module, state_dict: Mapping[str, Tensor]
) -> None:
    """Raise if checkpoint keys or tensor shapes differ from this IResNet-100."""
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))
    mismatched = sorted(
        f"{key}: checkpoint {tuple(state_dict[key].shape)} != "
        f"model {tuple(model_state[key].shape)}"
        for key in set(model_state).intersection(state_dict)
        if model_state[key].shape != state_dict[key].shape
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append("missing keys: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected keys: " + ", ".join(unexpected))
        if mismatched:
            details.append("shape mismatches: " + "; ".join(mismatched))
        raise RuntimeError(
            "Official checkpoint is not exactly compatible with the current "
            "IResNet-100 architecture; weights were not loaded. "
            + " | ".join(details)
        )


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve a device specification to a torch.device.

    Accepts:
        None or "auto" -> automatically selects XPU > CUDA > CPU
        "xpu"          -> Intel XPU (raises if unavailable)
        "cpu"          -> CPU
        "cuda"         -> CUDA GPU (raises if unavailable)
        torch.device   -> used directly
    """
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return select_device()

    if isinstance(device, torch.device):
        return device

    device_str = str(device).lower()
    if device_str == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            raise RuntimeError(
                "Requested device='xpu' but Intel XPU is not available on this machine. "
                "Install torch==2.14.0+xpu from https://download.pytorch.org/whl/xpu "
                "and ensure Intel GPU drivers are installed. Use device='cpu' as fallback."
            )
        return torch.device("xpu")

    return torch.device(device_str)


def load_iresnet100(
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """Load an exactly matching official IResNet-100 checkpoint for inference.

    Args:
        checkpoint_path: Path to backbone.pth checkpoint file.
        device: Target compute device. Accepts 'auto', 'xpu', 'cpu', 'cuda',
                a torch.device, or None (same as 'auto').
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Download backbone.pth from "
            f"the official source: {OFFICIAL_CHECKPOINT_URL}"
        )

    target_device = resolve_device(device)
    model = iresnet100(num_features=512, fp16=False)
    # Always load weights onto CPU first, then transfer to target device.
    # This avoids issues with XPU not supporting direct mmap loading.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _state_dict_from_checkpoint(checkpoint)
    verify_checkpoint_compatibility(model, state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.to(target_device).eval()
    return model


@torch.no_grad()
def smoke_test(model: nn.Module, device: Union[str, torch.device]) -> Tensor:
    """Run the requested 112x112 inference-shape check."""
    output = model(torch.randn(1, 3, 112, 112, device=device))
    if output.shape != (1, 512):
        raise RuntimeError(f"Expected output shape (1, 512), got {tuple(output.shape)}")
    return output


if __name__ == "__main__":
    inference_model = load_iresnet100()
    embeddings = smoke_test(inference_model, next(inference_model.parameters()).device)
    print(f"device={next(inference_model.parameters()).device}, output={tuple(embeddings.shape)}")
