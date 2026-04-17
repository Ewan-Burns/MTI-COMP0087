

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..devices import resolve_torch_device
from ..hf_cache import from_pretrained_local_first
from ..progress import progress_iter
from ..registry import register_detector
from ..results import read_jsonl
from ..types import DetectionExample, DetectorSpec
from ._supervised_common import (
    LABEL_MAP,
    has_local_classifier_checkpoint,
    model_load_kwargs,
    sanitize_scores,
    select_decision_threshold,
)

_sanitize_scores = sanitize_scores
_select_decision_threshold = select_decision_threshold
_model_load_kwargs = model_load_kwargs


def load_detection_examples(path: str | Path) -> list[DetectionExample]:
    return [
        DetectionExample(
            example_id=str(row["example_id"]),
            prompt_id=str(row["prompt_id"]),
            source_method=str(row["source_method"]),
            text=str(row["text"]),
            label=int(row["label"]),
            split=str(row["split"]),
            metadata=row.get("metadata", {}),
        )
        for row in read_jsonl(path)
    ]


def load_training_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_dir) / "training_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_decision_threshold(checkpoint_dir: str | Path) -> float:
    metadata = load_training_metadata(checkpoint_dir)
    return float(metadata.get("decision_threshold", 0.5))


@lru_cache(maxsize=32)
def _load_classifier_cached(checkpoint_dir: str, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = from_pretrained_local_first(AutoTokenizer, checkpoint_dir)
    model = from_pretrained_local_first(
        AutoModelForSequenceClassification,
        checkpoint_dir,
        **model_load_kwargs(force_float32=(device == "cpu")),
    )
    # Force float32 on CPU — some checkpoints save as bfloat16 which lacks CPU support
    if device == "cpu":
        model = model.float()
        try:
            model.config.torch_dtype = torch.float32
        except Exception:
            pass
    model = model.to(device)
    model.eval()
    return tokenizer, model


def _load_classifier(checkpoint_dir: str | Path, device: str = "cpu") -> tuple[Any, Any]:
    resolved_device = resolve_torch_device(device)
    resolved_checkpoint = str(Path(checkpoint_dir).expanduser().resolve())
    return _load_classifier_cached(resolved_checkpoint, resolved_device)


def _score_texts_with_loaded_classifier(
    texts: list[str],
    *,
    tokenizer: Any,
    model: Any,
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 512,
    desc: str,
) -> list[float]:
    import torch

    resolved_device = resolve_torch_device(device)
    model.eval()
    scores: list[float] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for start in progress_iter(
        range(0, len(texts), batch_size),
        desc=desc,
        total=total_batches,
        unit="batch",
        leave=False,
    ):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        encoded = {k: v.to(resolved_device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        # Index 1 = P(AI) per LABEL_MAP convention
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
        scores.extend(float(p) for p in probs)
    return sanitize_scores(scores)


def score_supervised_detector_texts(
    texts: list[str],
    *,
    checkpoint_dir: str | Path,
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 512,
) -> list[float]:
    resolved_device = resolve_torch_device(device)
    tokenizer, model = _load_classifier(checkpoint_dir, device=resolved_device)
    return _score_texts_with_loaded_classifier(
        texts,
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
        batch_size=batch_size,
        max_length=max_length,
        desc=f"Score {Path(checkpoint_dir).name}",
    )


# Apply calibrated threshold (from training) to convert scores → binary predictions.
def predict_supervised_detector_texts(**kwargs: Any) -> list[int]:
    checkpoint_dir = kwargs["checkpoint_dir"]
    threshold = float(kwargs.pop("threshold", load_decision_threshold(checkpoint_dir)))
    return [1 if s >= threshold else 0 for s in score_supervised_detector_texts(**kwargs)]


from ..training.train_supervised import train_supervised_detector  # noqa: E402, F401

# Register two supervised detector variants (RoBERTa and mDeBERTa).
register_detector(
    DetectorSpec(
        name="tuned-roberta-base",
        family="supervised",
        requires_training=True,
        train=train_supervised_detector,
        score_texts=score_supervised_detector_texts,
        predict_texts=predict_supervised_detector_texts,
        metadata={"architecture": "roberta-base"},
    )
)

register_detector(
    DetectorSpec(
        name="tuned-mdeberta-v3-base",
        family="supervised",
        requires_training=True,
        train=train_supervised_detector,
        score_texts=score_supervised_detector_texts,
        predict_texts=predict_supervised_detector_texts,
        metadata={"architecture": "microsoft/mdeberta-v3-base"},
    )
)
