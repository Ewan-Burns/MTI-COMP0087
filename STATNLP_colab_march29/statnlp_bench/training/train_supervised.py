
# Supervised fine-tuning for binary AI-text detectors.
# Checks reasonable checkpoint, find classifier and tokeniser
# Tokenise outputs, fine-tune, calibrate, then save model


from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ..config import SupervisedTrainingConfig
from ..devices import resolve_torch_device
from ..hf_cache import from_pretrained_local_first
from ..progress import progress_iter, progress_task, progress_write
from ..results import write_json
from ..types import DetectionExample, TrainedDetectorRef
from ..detectors._supervised_common import (
    LABEL_MAP,
    has_local_classifier_checkpoint,
    model_load_kwargs,
    sanitize_scores,
    select_decision_threshold,
)


# Check if an existing checkpoint was trained with the exact same config.
# Every field must match — any mismatch triggers a full retrain.
def _training_data_hash(examples: list) -> str:
    """Quick content hash of training data to detect dataset changes."""
    import hashlib
    sample = repr([(e.example_id, e.label) for e in examples[:100]])
    return hashlib.sha256(sample.encode()).hexdigest()[:12]


def _training_metadata_matches(
    metadata: dict[str, Any],
    *,
    model_name_or_path: str,
    config: SupervisedTrainingConfig,
    num_train_examples: int,
    num_validation_examples: int,
    data_hash: str = "",
) -> bool:
    effective_device = resolve_torch_device(config.device)
    expected = {
        "source_model": model_name_or_path,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "max_length": config.max_length,
        "seed": config.seed,
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "weight_decay": config.weight_decay,
        "device": effective_device,
        "force_float32": config.force_float32,
        "calibrate_threshold": config.calibrate_threshold,
        "target_fpr": config.target_fpr,
        "internal_validation_ratio": config.internal_validation_ratio,
        "num_train_examples": num_train_examples,
        "num_validation_examples": num_validation_examples,
    }
    if data_hash:
        expected["data_hash"] = data_hash
    return all(metadata.get(key) == val for key, val in expected.items())


def _build_hf_dataset(examples: list[DetectionExample]) -> Dataset:
    return Dataset.from_list([{"text": e.text, "label": int(e.label)} for e in examples])


def _tokenize_dataset(dataset: Dataset, tokenizer: Any, max_length: int) -> Dataset:
    return dataset.map(
        lambda batch: tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=max_length,
        ),
        batched=True,
        desc="Tokenizing detector dataset",
    )


# HF Trainer callback: convert logits → probabilities via numerically stable softmax.
def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs_all = np.exp(shifted)
    probs_all /= probs_all.sum(axis=-1, keepdims=True)
    probs = probs_all[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auroc"] = 0.5
    return metrics


def _build_training_arguments(
    TrainingArguments: Any,
    *,
    output_path: Path,
    config: SupervisedTrainingConfig,
) -> Any:
    effective_device = resolve_torch_device(config.device)
    kwargs: dict[str, Any] = {
        "output_dir": str(output_path),
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "save_strategy": "no",
        "seed": config.seed,
        "weight_decay": config.weight_decay,
        "load_best_model_at_end": False,
        "report_to": [],
        "remove_unused_columns": True,
        "disable_tqdm": True,
        "logging_steps": 10000,
    }
    params = set(inspect.signature(TrainingArguments.__init__).parameters)
    _apply_version_specific_args(kwargs, params, effective_device)
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in params})


# Handle API differences across HF Transformers versions (e.g. evaluation_strategy
# vs eval_strategy, use_cpu vs no_cuda) by introspecting TrainingArguments.__init__.
def _apply_version_specific_args(
    kwargs: dict[str, Any], params: set[str], device: str,
) -> None:
    if "save_only_model" in params:
        kwargs["save_only_model"] = True
    if "save_total_limit" in params:
        kwargs["save_total_limit"] = 1
    # No evaluation during training — matches Dubois (always use final model)
    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "no"
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = "no"

    if device == "cpu":
        if "use_cpu" in params:
            kwargs["use_cpu"] = True
        elif "no_cuda" in params:
            kwargs["no_cuda"] = True

    if device != "cuda" and "dataloader_pin_memory" in params:
        kwargs["dataloader_pin_memory"] = False
    if "fp16" in params:
        kwargs["fp16"] = False
    if "bf16" in params:
        kwargs["bf16"] = False


def _build_trainer(
    Trainer: Any,
    *,
    model: Any,
    training_args: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: Any,
) -> Any:
    params = set(inspect.signature(Trainer.__init__).parameters)
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "compute_metrics": _compute_metrics,
    }
    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    return Trainer(**kwargs)


def _prune_training_checkpoints(output_path: Path) -> None:
    for checkpoint_dir in output_path.glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def _reset_output_dir(output_path: Path) -> None:
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        return
    for child in output_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


# Attempt to skip training by reusing an existing checkpoint.
# Returns None if no valid checkpoint exists, forcing a fresh train.
def _try_reuse_checkpoint(
    output_path: Path,
    model_name_or_path: str,
    config: SupervisedTrainingConfig,
    num_train: int,
    num_val: int,
    data_hash: str = "",
) -> TrainedDetectorRef | None:
    metadata_path = output_path / "training_metrics.json"
    existing_metadata = {}
    if metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if not has_local_classifier_checkpoint(output_path):
        return None
    if config.force_retrain:
        return None
    if not _training_metadata_matches(
        existing_metadata,
        model_name_or_path=model_name_or_path,
        config=config,
        num_train_examples=num_train,
        num_validation_examples=num_val,
        data_hash=data_hash,
    ):
        return None

    _prune_training_checkpoints(output_path)
    progress_write(f"Reusing tuned detector checkpoint at {output_path}")
    return TrainedDetectorRef(
        name=output_path.name,
        family="supervised",
        checkpoint_dir=output_path,
        label_map=LABEL_MAP,
        metadata=existing_metadata,
    )


def _load_and_prepare_model(
    model_name_or_path: str, config: SupervisedTrainingConfig,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = from_pretrained_local_first(AutoTokenizer, model_name_or_path)
    model = from_pretrained_local_first(
        AutoModelForSequenceClassification,
        model_name_or_path,
        num_labels=2,
        id2label=LABEL_MAP,
        label2id={"human": 0, "ai": 1},
        **model_load_kwargs(force_float32=config.force_float32),
    )
    if config.force_float32:
        model = model.float()
        try:
            model.config.torch_dtype = torch.float32
        except Exception:
            pass
    return tokenizer, model


def _score_for_calibration(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    config: SupervisedTrainingConfig,
    desc: str,
) -> list[float]:
    import torch

    effective_device = resolve_torch_device(config.device)
    model.eval()
    scores: list[float] = []
    total_batches = (len(texts) + config.eval_batch_size - 1) // config.eval_batch_size
    for start in progress_iter(
        range(0, len(texts), config.eval_batch_size),
        desc=desc,
        total=total_batches,
        unit="batch",
        leave=False,
    ):
        batch = texts[start : start + config.eval_batch_size]
        encoded = tokenizer(
            batch, truncation=True, padding="max_length",
            max_length=config.max_length, return_tensors="pt",
        )
        encoded = {k: v.to(effective_device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
        scores.extend(float(p) for p in probs)
    return sanitize_scores(scores)


# Post-training threshold calibration: score all validation examples, then find
# the threshold where FPR on human texts ≈ target_fpr. Skipped if disabled or no val data.
def _calibrate_threshold(
    validation_examples: list[DetectionExample],
    tokenizer: Any,
    model: Any,
    config: SupervisedTrainingConfig,
    output_name: str,
) -> tuple[float, float]:
    if not config.calibrate_threshold or not validation_examples:
        return 0.5, 0.0
    validation_scores = _score_for_calibration(
        [e.text for e in validation_examples],
        tokenizer, model, config,
        desc=f"Calibrate {output_name}",
    )
    return select_decision_threshold(
        [int(e.label) for e in validation_examples],
        validation_scores,
        target_fpr=config.target_fpr,
    )


def _save_training_record(
    output_path: Path,
    metrics: dict[str, Any],
    model_name_or_path: str,
    config: SupervisedTrainingConfig,
    decision_threshold: float,
    validation_fpr: float,
    num_train: int,
    num_val: int,
    data_hash: str = "",
) -> None:
    effective_device = resolve_torch_device(config.device)
    record = {
        "metrics": metrics,
        "label_map": LABEL_MAP,
        "source_model": model_name_or_path,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "max_length": config.max_length,
        "seed": config.seed,
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "weight_decay": config.weight_decay,
        "device": effective_device,
        "force_float32": config.force_float32,
        "calibrate_threshold": config.calibrate_threshold,
        "decision_threshold": decision_threshold,
        "target_fpr": config.target_fpr,
        "validation_fpr": validation_fpr,
        "internal_validation_ratio": config.internal_validation_ratio,
        "num_train_examples": num_train,
        "num_validation_examples": num_val,
        "data_hash": data_hash,
    }
    write_json(output_path / "training_metrics.json", record)


# Main entry point: fine-tune a sequence classifier as a human-vs-AI detector.
def train_supervised_detector(
    *,
    train_examples: list[DetectionExample],
    validation_examples: list[DetectionExample],
    model_name_or_path: str,
    output_dir: str | Path,
    config: SupervisedTrainingConfig,
) -> TrainedDetectorRef:
    from transformers import Trainer, TrainingArguments

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    data_hash = _training_data_hash(train_examples)
    cached = _try_reuse_checkpoint(
        output_path, model_name_or_path, config,
        len(train_examples), len(validation_examples),
        data_hash=data_hash,
    )
    if cached is not None:
        return cached

    with progress_task(total=5, desc=f"Detector {output_path.name}", unit="stage", leave=True) as progress:
        _reset_output_dir(output_path)

        progress.set_postfix_str("load tokenizer+model")
        tokenizer, model = _load_and_prepare_model(model_name_or_path, config)
        progress.update(1)

        progress.set_postfix_str("tokenize train")
        train_dataset = _tokenize_dataset(_build_hf_dataset(train_examples), tokenizer, config.max_length)
        progress.update(1)

        progress.set_postfix_str("tokenize validation")
        eval_dataset = _tokenize_dataset(_build_hf_dataset(validation_examples), tokenizer, config.max_length)
        progress.update(1)

        training_args = _build_training_arguments(TrainingArguments, output_path=output_path, config=config)
        trainer = _build_trainer(
            Trainer, model=model, training_args=training_args,
            train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer,
        )

        progress.set_postfix_str("train")
        trainer.train()
        progress.update(1)

        progress.set_postfix_str("evaluate+save")
        # Clean up intermediate checkpoints, save final model, then calibrate threshold.
        _prune_training_checkpoints(output_path)
        trainer.save_model(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        metrics = trainer.evaluate()

        decision_threshold, validation_fpr = _calibrate_threshold(
            validation_examples, tokenizer, trainer.model, config, output_path.name,
        )
        progress.update(1)

    _save_training_record(
        output_path, metrics, model_name_or_path, config,
        decision_threshold, validation_fpr,
        len(train_examples), len(validation_examples),
        data_hash=data_hash,
    )

    return TrainedDetectorRef(
        name=output_path.name,
        family="supervised",
        checkpoint_dir=output_path,
        label_map=LABEL_MAP,
        metadata={
            "source_model": model_name_or_path,
            "metrics": metrics,
            "decision_threshold": decision_threshold,
            "target_fpr": config.target_fpr,
            "validation_fpr": validation_fpr,
        },
    )
