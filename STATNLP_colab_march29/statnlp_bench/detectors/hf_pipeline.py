# Wrapper for pre-trained HuggingFace text-classification models used as
# AI-text detectors. These are off-the-shelf classifiers (e.g. RoBERTa fine-tuned
# on GPT outputs) that output "ai" vs "human" probabilities.
#
# Key challenge: different models use different label names and orderings.
# We auto-detect which output class means "AI" by matching label strings
# against known markers (see _AI_MARKERS / _HUMAN_MARKERS below).

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..devices import resolve_torch_device
from ..hf_cache import from_pretrained_local_first, snapshot_download_local_first
from ..progress import progress_iter, progress_write
from ..registry import register_detector
from ..types import DetectorSpec

DEFAULT_HF_DETECTOR_IDS = [
    "openai-community/roberta-base-openai-detector",
    "Hello-SimpleAI/chatgpt-detector-roberta",
    "Hello-SimpleAI/chatgpt-qa-detector-roberta",
    "vraj33/ai-text-detector-deberta",
]

# Substrings used to classify model output labels as AI or human.
_AI_MARKERS = ("ai", "fake", "generated", "machine", "chatgpt", "gpt")
_HUMAN_MARKERS = ("human", "real", "organic", "person")

_DOWNLOAD_PATTERNS = [
    "*.json", "*.safetensors", "*.bin", "*.txt",
    "*.model", "tokenizer*", "vocab*", "merges.txt",
]


def _has_local_model_files(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").exists()
        and (any(path.glob("*.safetensors")) or any(path.glob("*.bin")))
    )


@lru_cache(maxsize=32)
def ensure_local_detector(cache_dir: Path, repo_id: str, token: str | None = None) -> Path:
    detector_path = (cache_dir / repo_id.replace("/", "--")).resolve()
    if _has_local_model_files(detector_path):
        progress_write(f"Reusing cached HF detector at {detector_path}")
        return detector_path
    detector_path.mkdir(parents=True, exist_ok=True)
    snapshot_download_local_first(
        repo_id=repo_id,
        local_dir=str(detector_path),
        allow_patterns=_DOWNLOAD_PATTERNS,
        token=token,
    )
    if not _has_local_model_files(detector_path):
        raise RuntimeError(f"Detector download completed but required files not found in {detector_path}")
    return detector_path


def _get_id2label(model: Any) -> dict[int, str]:
    config = getattr(getattr(model, "model", None), "config", None) or getattr(model, "config", None)
    raw = getattr(config, "id2label", None)
    if not isinstance(raw, dict):
        return {}
    result: dict[int, str] = {}
    for idx, label in raw.items():
        try:
            result[int(idx)] = str(label)
        except (TypeError, ValueError):
            continue
    return result


# Map model's id2label config to AI vs human label index sets.
# Handles models that only label one side (e.g. only "human" → infer the other).
def infer_detector_label_sets(model: Any) -> tuple[set[int], set[int]]:
    id2label = _get_id2label(model)
    ai_ids: set[int] = set()
    human_ids: set[int] = set()
    for idx, label_text in id2label.items():
        normalized = re.sub(r"\s+", " ", label_text.strip().lower())
        if any(m in normalized for m in _AI_MARKERS):
            ai_ids.add(idx)
        elif any(m in normalized for m in _HUMAN_MARKERS):
            human_ids.add(idx)
    if not ai_ids and human_ids and len(id2label) == 2:
        ai_ids = set(id2label.keys()) - human_ids
    return ai_ids, human_ids


# Extract P(AI) from the softmax output. Fallback cascade:
# 1. Use AI label indices directly if known
# 2. Else use 1 - P(human) if human labels known
# 3. Else assume binary with AI = index 1
def _ai_probability(probs: list[float], ai_ids: set[int], human_ids: set[int]) -> float:
    if ai_ids:
        return max(probs[i] for i in ai_ids if i < len(probs))
    if human_ids:
        return 1.0 - max(probs[i] for i in human_ids if i < len(probs))
    if len(probs) == 2:
        return probs[1]
    return max(probs)


# Some tokenizers report absurd max_length values (e.g. 1e30); treat those as unset.
def _resolve_max_length(tokenizer: Any) -> int | None:
    max_length = getattr(tokenizer, "model_max_length", None)
    try:
        max_length = int(max_length)
    except (TypeError, ValueError):
        return None
    if max_length <= 0 or max_length >= 1_000_000:
        return None
    return max_length


@lru_cache(maxsize=32)
def _load_hf_detector(
    model_id_or_path: str,
    *,
    token: str | None = None,
    device: str = "auto",
) -> tuple[Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    resolved_device = resolve_torch_device(device)
    try:
        tokenizer = from_pretrained_local_first(
            AutoTokenizer, model_id_or_path, token=token, fix_mistral_regex=True,
        )
    except TypeError:
        tokenizer = from_pretrained_local_first(
            AutoTokenizer, model_id_or_path, token=token,
        )
    model = from_pretrained_local_first(
        AutoModelForSequenceClassification, model_id_or_path, token=token,
    )
    model = model.to(resolved_device)
    model.eval()
    return tokenizer, model


def _resolve_model_ref(model_id_or_path: str, cache_dir: str | Path | None, token: str | None) -> str:
    local_ref = Path(model_id_or_path).expanduser()
    if local_ref.exists():
        return str(local_ref.resolve())
    if cache_dir is not None:
        return str(ensure_local_detector(Path(cache_dir), model_id_or_path, token=token))
    return model_id_or_path


def score_hf_detector_texts(
    texts: list[str],
    *,
    model_id_or_path: str,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    max_chars: int = 2500,
    device: str = "auto",
    batch_size: int = 8,
) -> list[float]:
    import torch

    model_ref = _resolve_model_ref(model_id_or_path, cache_dir, token)
    resolved_device = resolve_torch_device(device)
    tokenizer, model = _load_hf_detector(model_ref, token=token, device=resolved_device)
    ai_ids, human_ids = infer_detector_label_sets(model)
    max_length = _resolve_max_length(tokenizer)

    # --- Batch scoring loop ---
    scores: list[float] = []
    detector_name = Path(model_ref).name if Path(model_ref).exists() else model_id_or_path
    batch_size = max(1, batch_size)
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for start in progress_iter(
        range(0, len(texts), batch_size),
        desc=f"HF {detector_name}",
        total=total_batches,
        unit="batch",
        leave=False,
    ):
        # Pre-truncate by chars before tokenization as a cheap safety net
        batch = [text[:max_chars] for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt",
        )
        encoded = {k: v.to(resolved_device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        for probs in probabilities:
            scores.append(_ai_probability([float(p) for p in probs], ai_ids, human_ids))
    return scores


def predict_hf_detector_texts(**kwargs: Any) -> list[int]:
    return [1 if score >= 0.5 else 0 for score in score_hf_detector_texts(**kwargs)]


# Register each default HF detector as a zero-shot (no training required) detector.
for detector_id in DEFAULT_HF_DETECTOR_IDS:
    register_detector(
        DetectorSpec(
            name=detector_id,
            family="hf_pipeline",
            requires_training=False,
            train=None,
            score_texts=score_hf_detector_texts,
            predict_texts=predict_hf_detector_texts,
            metadata={"model_id_or_path": detector_id},
        )
    )
