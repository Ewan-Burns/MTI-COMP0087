# Shared model-loading and batch-scoring infrastructure for detectors that
# compare logits from two causal LMs (e.g. Binoculars, FastDetectGPT).
# Both methods need the same forward-pass pipeline; only the final score
# computation differs — so that is injected via the `compute_scores` callback.

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from ..devices import resolve_torch_device
from ..hf_cache import from_pretrained_local_first
from ..progress import progress_iter


# Quick fingerprint to verify two tokenizers share the same vocabulary.
def _tokenizer_signature(tokenizer: Any) -> tuple[int, tuple[tuple[str, int], ...]]:
    vocab = tokenizer.get_vocab()
    sample = tuple(sorted(vocab.items())[:256])
    return len(vocab), sample


# Cache loaded model pairs so repeated calls (e.g. across datasets) don't reload weights.
# Two models are loaded: model_q (the "main"/scorer) and model_r (the "reference"/auxiliary).
# Both must share a tokenizer vocabulary — otherwise token-level comparisons are meaningless.
@lru_cache(maxsize=4)
def load_model_pair(
    main_model_name: str,
    aux_model_name: str,
    device: str,
) -> tuple[Any, Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = resolve_torch_device(device)
    tokenizer_q = from_pretrained_local_first(AutoTokenizer, main_model_name)
    tokenizer_r = from_pretrained_local_first(AutoTokenizer, aux_model_name)
    if _tokenizer_signature(tokenizer_q) != _tokenizer_signature(tokenizer_r):
        raise ValueError(f"Models {main_model_name} and {aux_model_name} have incompatible tokenizers")
    if tokenizer_q.pad_token is None:
        tokenizer_q.pad_token = tokenizer_q.eos_token
    if tokenizer_r.pad_token is None:
        tokenizer_r.pad_token = tokenizer_r.eos_token
    import torch
    # Load in bfloat16 to match Dubois et al. — scoring models don't need full precision
    model_q = from_pretrained_local_first(
        AutoModelForCausalLM, main_model_name, torch_dtype=torch.bfloat16,
    ).to(resolved_device)
    model_q.eval()
    model_r = from_pretrained_local_first(
        AutoModelForCausalLM, aux_model_name, torch_dtype=torch.bfloat16,
    ).to(resolved_device)
    model_r.eval()
    return tokenizer_q, model_q, model_r


# ---------------------------------------------------------------------------
# Shared batch-scoring loop: tokenize, run both models, then delegate to
# `compute_scores` for the detector-specific metric (Binoculars or FastDetectGPT).
# ---------------------------------------------------------------------------
def score_with_model_pair(
    texts: list[str],
    *,
    main_model_name: str,
    aux_model_name: str,
    device: str,
    max_length: int,
    batch_size: int,
    detector_name: str,
    compute_scores: Callable,
) -> list[float]:
    import torch

    resolved_device = resolve_torch_device(device)
    tokenizer, model_q, model_r = load_model_pair(
        main_model_name, aux_model_name, resolved_device,
    )
    scores: list[float] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for start in progress_iter(
        range(0, len(texts), batch_size),
        desc=f"{detector_name} batches",
        total=total_batches,
        unit="batch",
        leave=False,
    ):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        encoded = {k: v.to(resolved_device) for k, v in encoded.items()}
        with torch.no_grad():
            # Shift logits left (:-1) to align predictions with next-token labels (1:)
            logits_q = model_q(**encoded).logits[:, :-1, :]
            logits_r = model_r(**encoded).logits[:, :-1, :]

        labels = encoded["input_ids"][:, 1:]  # next-token targets
        mask = encoded["attention_mask"][:, 1:].float()  # ignore padding
        log_probs_q = torch.log_softmax(logits_q, dim=-1)
        probs_r = torch.softmax(logits_r, dim=-1)

        batch_scores = compute_scores(log_probs_q, probs_r, labels, mask)
        scores.extend(batch_scores.detach().cpu().tolist())
    return scores
