'''
 Shared constants and helpers used by both the supervised training module
 (statnlp_bench.training.train_supervised) and the inference-time scoring in supervised.py
'''
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

LABEL_MAP = {0: "human", 1: "ai"} 


# Clean NaN/Inf scores
def sanitize_scores(scores: list[float], fallback: float = 0.5) -> list[float]:
    return [float(s) if np.isfinite(s) else fallback for s in scores]


# Pick a decision threshold from human-only scores to achieve a target false-positive rate.
# Used after training to calibrate the binary prediction boundary.
def select_decision_threshold(
    labels: list[int],
    scores: list[float],
    *,
    target_fpr: float,
) -> tuple[float, float]:
    human_scores = sorted(
        s for label, s in zip(labels, sanitize_scores(scores)) if int(label) == 0
    )
    if not human_scores:
        return 0.5, 0.0
    # Threshold = the (1 - target_fpr) quantile of human score distribution
    keep_fraction = max(0.0, min(1.0, 1.0 - float(target_fpr)))
    idx = min(int(len(human_scores) * keep_fraction), len(human_scores) - 1)
    threshold = float(human_scores[idx])
    actual_fpr = sum(s >= threshold for s in human_scores) / len(human_scores)
    return threshold, float(actual_fpr)


# Force float32 on CPU to avoid bfloat16 issues on non-CUDA backends.
def model_load_kwargs(force_float32: bool = True) -> dict[str, Any]:
    if not force_float32:
        return {}
    try:
        import torch
        return {"torch_dtype": torch.float32}
    except Exception:
        return {}


# Check whether a directory contains a usable HF model checkpoint (config + weights).
def has_local_classifier_checkpoint(checkpoint_dir: str | Path) -> bool:
    path = Path(checkpoint_dir).expanduser().resolve()
    return (
        path.is_dir()
        and (path / "config.json").exists()
        and (
            any(path.glob("*.safetensors"))
            or any(path.glob("pytorch_model*.bin"))
            or any(path.glob("*.bin"))
        )
    )
