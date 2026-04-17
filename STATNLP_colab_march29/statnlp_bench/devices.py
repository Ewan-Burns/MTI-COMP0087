# Resolve a user-supplied device string (e.g. "auto", "gpu", "cpu")
# into a concrete PyTorch device name. Preference order: CUDA > CPU.

from __future__ import annotations


def normalize_device_name(device: str | None) -> str:
    if not device or not device.strip():
        return "auto"
    name = device.strip().lower()
    if name == "gpu":
        return "auto"
    return name


def detect_best_torch_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_torch_device(device: str | None = None) -> str:
    normalized = normalize_device_name(device)
    if normalized == "auto":
        return detect_best_torch_device()
    return normalized
