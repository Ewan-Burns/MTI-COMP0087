# --------------------------------------------------------------------------- #
# task_baselines.py — FULL_PASS baseline for the task-efficiency track
#
# Runs every example through all transformer layers with no early exit.
# This is the "no tricks" baseline that RAEE and other efficiency methods
# are compared against — same model, same predictions, just no speedup.
# --------------------------------------------------------------------------- #
from __future__ import annotations

import time
from typing import Any

from ..devices import resolve_torch_device
from ..hf_cache import from_pretrained_local_first
from ..progress import progress_iter
from ..registry import register_method
from ..types import MethodRunResult, MethodSpec, TaskExample


def _pair_texts(examples: list[TaskExample]) -> tuple[list[str], list[str | None]]:
    return [example.text_a for example in examples], [example.text_b for example in examples]


def _tokenize_batch(tokenizer: Any, texts_a: list[str], texts_b: list[str | None]) -> dict[str, Any]:
    common_kwargs = {
        "padding": True,
        "truncation": True,
        "max_length": 512,
        "return_tensors": "pt",
    }
    if any(text_b is not None for text_b in texts_b):
        return tokenizer(texts_a, [text_b or "" for text_b in texts_b], **common_kwargs)
    return tokenizer(texts_a, **common_kwargs)


def _predict_batch(model: Any, tokenizer: Any, texts_a: list[str], texts_b: list[str | None], device: str) -> tuple[list[int], list[float]]:
    import torch

    encoded = _tokenize_batch(tokenizer, texts_a, texts_b)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)
    predictions = probs.argmax(dim=-1).detach().cpu().tolist()
    confidences = probs.max(dim=-1).values.detach().cpu().tolist()
    return predictions, confidences


def run_full_pass(
    *,
    examples: list[TaskExample],
    model_name: str,
    batch_size: int = 8,
    device: str = "auto",
) -> MethodRunResult:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    resolved_device = resolve_torch_device(device)
    tokenizer = from_pretrained_local_first(AutoTokenizer, model_name)
    model = from_pretrained_local_first(AutoModelForSequenceClassification, model_name).to(resolved_device)
    model.eval()
    predictions: list[int] = []
    scores: list[float] = []
    started = time.perf_counter()
    for batch_start in progress_iter(
        range(0, len(examples), batch_size),
        desc="FULL_PASS batches",
        total=(len(examples) + batch_size - 1) // batch_size,
        unit="batch",
        leave=False,
    ):
        batch = examples[batch_start : batch_start + batch_size]
        texts_a, texts_b = _pair_texts(batch)
        batch_predictions, batch_scores = _predict_batch(model, tokenizer, texts_a, texts_b, resolved_device)
        predictions.extend(batch_predictions)
        scores.extend(batch_scores)
    elapsed = (time.perf_counter() - started) * 1000.0
    # exit_layers is always None — this baseline never exits early
    return MethodRunResult(
        method_name="FULL_PASS",
        predictions=predictions,
        scores=scores,
        exit_layers=[None] * len(predictions),
        latency_ms=elapsed,
    )


register_method(
    MethodSpec(
        name="FULL_PASS",
        track="task_efficiency",
        family="sequence_classifier",
        supports_answer_voting=False,
        supports_dataset=lambda _: True,
        run=run_full_pass,
    )
)
