# --------------------------------------------------------------------------- #
# raee.py — Retrieval-Augmented Early Exit (RAEE) for task efficiency
#
# Idea: build a kNN index from training-set hidden states across all layers.
# At inference, exit early when a layer's nearest-neighbor vote agrees with
# the model's own prediction above confidence thresholds. This trades a small
# amount of accuracy for large speedups on "easy" examples.
#
# Two thresholds gate early exit:
#   retrieval_threshold — minimum kNN vote confidence
#   model_threshold     — minimum model softmax confidence for the voted label
# --------------------------------------------------------------------------- #
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..devices import resolve_torch_device
from ..hf_cache import from_pretrained_local_first
from ..progress import progress_iter
from ..registry import register_method
from ..types import MethodRunResult, MethodSpec, TaskExample


# Pre-computed index: per-layer normalised hidden states + gold labels
@dataclass(slots=True)
class RAEEIndex:
    label_ids: np.ndarray
    layer_vectors: list[np.ndarray]


# Mean-pool over non-padding tokens to get a single vector per example
def _pool_hidden_state(hidden_state: Any, attention_mask: Any) -> np.ndarray:
    mask = attention_mask.unsqueeze(-1)
    pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled.detach().cpu().numpy()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _tokenize_batch(tokenizer: Any, batch: list[TaskExample]) -> dict[str, Any]:
    texts_a = [ex.text_a for ex in batch]
    texts_b = [ex.text_b for ex in batch]
    kwargs = {"padding": True, "truncation": True, "max_length": 512, "return_tensors": "pt"}
    if any(b is not None for b in texts_b):
        return tokenizer(texts_a, [b or "" for b in texts_b], **kwargs)
    return tokenizer(texts_a, **kwargs)


def _progress_batches(total: int, batch_size: int, desc: str):
    return progress_iter(
        range(0, total, batch_size),
        desc=desc,
        total=(total + batch_size - 1) // batch_size,
        unit="batch",
        leave=False,
    )


def build_raee_index(
    *,
    examples: list[TaskExample],
    model_name: str,
    batch_size: int = 8,
    device: str = "auto",
) -> tuple[RAEEIndex, Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    resolved_device = resolve_torch_device(device)
    tokenizer = from_pretrained_local_first(AutoTokenizer, model_name)
    model = from_pretrained_local_first(
        AutoModelForSequenceClassification, model_name,
    ).to(resolved_device)
    model.eval()

    label_ids: list[int] = []
    layer_buckets: list[list[np.ndarray]] | None = None

    for start in _progress_batches(len(examples), batch_size, "RAEE index batches"):
        batch = examples[start : start + batch_size]
        encoded = _tokenize_batch(tokenizer, batch)
        encoded = {k: v.to(resolved_device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)

        hidden_states = outputs.hidden_states or ()
        if layer_buckets is None:
            layer_buckets = [[] for _ in hidden_states]

        mask = encoded["attention_mask"]
        for layer_idx, hs in enumerate(hidden_states):
            layer_buckets[layer_idx].append(_pool_hidden_state(hs, mask))
        label_ids.extend(ex.label for ex in batch)

    assert layer_buckets is not None
    layer_vectors = [
        _normalize_rows(np.concatenate(parts, axis=0)) for parts in layer_buckets
    ]
    index = RAEEIndex(
        label_ids=np.asarray(label_ids, dtype=np.int64),
        layer_vectors=layer_vectors,
    )
    return index, tokenizer, model


# kNN vote: find top_k most similar training examples, return majority label
# and its proportion as confidence.
def _neighbor_vote(
    labels: np.ndarray, similarities: np.ndarray, *, top_k: int,
) -> tuple[int, float]:
    if len(similarities) <= top_k:
        top_indices = np.argsort(similarities)[::-1]
    else:
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    top_labels = labels[top_indices]
    label, count = Counter(int(l) for l in top_labels).most_common(1)[0]
    return label, count / max(1, len(top_labels))


def run_raee(
    *,
    train_examples: list[TaskExample],
    eval_examples: list[TaskExample],
    model_name: str,
    batch_size: int = 8,
    device: str = "auto",
    retrieval_top_k: int = 8,
    retrieval_threshold: float = 0.75,
    model_threshold: float = 0.75,
) -> MethodRunResult:
    import torch

    index, tokenizer, model = build_raee_index(
        examples=train_examples,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
    resolved_device = resolve_torch_device(device)
    predictions: list[int] = []
    scores: list[float] = []
    exit_layers: list[int | None] = []
    started = time.perf_counter()

    for start in _progress_batches(len(eval_examples), batch_size, "RAEE eval batches"):
        batch = eval_examples[start : start + batch_size]
        encoded = _tokenize_batch(tokenizer, batch)
        encoded = {k: v.to(resolved_device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)

        hidden_states = outputs.hidden_states or ()
        probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
        mask = encoded["attention_mask"]
        pooled_layers = [
            _normalize_rows(_pool_hidden_state(hs, mask)) for hs in hidden_states
        ]

        for i in range(len(batch)):
            sample_probs = probs[i]
            chosen_pred = int(sample_probs.argmax())
            chosen_layer: int | None = len(pooled_layers) - 1 if pooled_layers else None
            chosen_score = float(sample_probs.max())

            # Walk layers bottom-up; exit as soon as kNN and model agree
            for layer_idx, layer_vecs in enumerate(pooled_layers):
                # Cosine similarity (vectors are pre-normalised)
                sims = index.layer_vectors[layer_idx] @ layer_vecs[i]
                vote_label, vote_conf = _neighbor_vote(
                    index.label_ids, sims, top_k=retrieval_top_k,
                )
                model_conf = float(sample_probs[vote_label]) if vote_label < len(sample_probs) else 0.0
                if vote_conf >= retrieval_threshold and model_conf >= model_threshold:
                    chosen_pred = vote_label
                    chosen_layer = layer_idx
                    chosen_score = vote_conf
                    break  # early exit — skip remaining layers

            predictions.append(chosen_pred)
            scores.append(chosen_score)
            exit_layers.append(chosen_layer)

    elapsed = (time.perf_counter() - started) * 1000.0
    return MethodRunResult(
        method_name="RAEE",
        predictions=predictions,
        scores=scores,
        exit_layers=exit_layers,
        latency_ms=elapsed,
        metadata={
            "retrieval_top_k": retrieval_top_k,
            "retrieval_threshold": retrieval_threshold,
            "model_threshold": model_threshold,
        },
    )


register_method(
    MethodSpec(
        name="RAEE",
        track="task_efficiency",
        family="retrieval_guided_early_exit",
        supports_answer_voting=False,
        supports_dataset=lambda _: True,
        run=run_raee,
    )
)
