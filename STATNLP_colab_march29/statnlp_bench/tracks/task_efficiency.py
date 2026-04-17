# ==========================================================================
# Task-Efficiency Track — measures accuracy vs. computational cost.
#
# Pipeline:
#   1. Prepare NLU datasets (e.g. sentiment, NLI) with controlled sizes
#   2. Run each method (FULL_PASS baseline, RAEE early-exit) on each dataset
#   3. Record accuracy, macro-F1, average exit layer, and latency
#
# The goal is to compare early-exit strategies (like RAEE) against full
# model inference: do we lose much accuracy by exiting at intermediate layers?
# ==========================================================================

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, f1_score

from ..config import TaskTrackConfig, dataclass_to_dict
from ..datasets.nlu import prepare_nlu_dataset
from ..methods.raee import run_raee
from ..methods.task_baselines import run_full_pass
from ..progress import progress_iter, progress_task
from ..results import read_jsonl, write_csv, write_json
from ..types import DatasetManifest, TaskExample, TaskRunResult

# Import side effects for registry population.
from ..datasets import nlu as _nlu  # noqa: F401
from ..methods import raee as _raee  # noqa: F401
from ..methods import task_baselines as _task_baselines  # noqa: F401


def load_task_examples(manifest: DatasetManifest) -> list[TaskExample]:
    return [
        TaskExample(
            example_id=str(row["example_id"]),
            dataset=str(row["dataset"]),
            split=str(row["split"]),
            label=int(row["label"]),
            text_a=str(row["text_a"]),
            text_b=row.get("text_b"),
            label_text=row.get("label_text"),
            metadata=row.get("metadata", {}),
        )
        for row in read_jsonl(manifest.records_path)
    ]


def _compute_metrics(labels: list[int], preds: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def _result_to_row(result: TaskRunResult) -> dict[str, Any]:
    return {
        "dataset": result.dataset,
        "method_name": result.method_name,
        "accuracy": result.metric_dict.get("accuracy"),
        "macro_f1": result.metric_dict.get("macro_f1"),
        "avg_exit_layer": result.avg_exit_layer,
        "latency_ms": result.latency_ms,
        "metadata": result.metadata,
    }


def _result_to_json(result: TaskRunResult) -> dict[str, Any]:
    return {
        "dataset": result.dataset,
        "method_name": result.method_name,
        "metric_dict": result.metric_dict,
        "avg_exit_layer": result.avg_exit_layer,
        "latency_ms": result.latency_ms,
        "metadata": result.metadata,
    }


def prepare_task_datasets(config: TaskTrackConfig) -> dict[str, DatasetManifest]:
    return {
        name: prepare_nlu_dataset(
            dataset_name=name,
            output_dir=config.artifacts.datasets / "task_efficiency" / name,
            max_train_items=config.max_train_items,
            max_eval_items=config.max_eval_items,
            seed=config.seed,
        )
        for name in progress_iter(config.datasets, desc="Task datasets", total=len(config.datasets), unit="dataset", leave=True)
    }


def _run_method(
    method_name: str,
    eval_examples: list[TaskExample],
    train_examples: list[TaskExample],
    model_name: str,
    dataset_name: str,
    config: TaskTrackConfig,
) -> TaskRunResult:
    labels = [e.label for e in eval_examples]

    # FULL_PASS: standard inference through all layers (the accuracy ceiling).
    if method_name == "FULL_PASS":
        result = run_full_pass(
            examples=eval_examples,
            model_name=model_name,
            batch_size=config.batch_size,
            device=config.device,
        )
        metrics = _compute_metrics(labels, result.predictions)
        return TaskRunResult(
            dataset=dataset_name,
            method_name="FULL_PASS",
            metric_dict=metrics,
            avg_exit_layer=None,
            latency_ms=result.latency_ms,
        )

    # RAEE: early-exit inference — may stop at an intermediate layer per example.
    if method_name == "RAEE":
        result = run_raee(
            train_examples=train_examples,
            eval_examples=eval_examples,
            model_name=model_name,
            batch_size=config.batch_size,
            device=config.device,
        )
        exit_layers = [layer for layer in result.exit_layers if layer is not None]
        metrics = _compute_metrics(labels, result.predictions)
        return TaskRunResult(
            dataset=dataset_name,
            method_name="RAEE",
            metric_dict=metrics,
            avg_exit_layer=(sum(exit_layers) / len(exit_layers)) if exit_layers else None,
            latency_ms=result.latency_ms,
            metadata=result.metadata,
        )

    raise ValueError(f"Unknown method: {method_name}")


# Main entry point: prepare datasets, run all methods, persist results.
def run_task_benchmark(
    *,
    config: TaskTrackConfig,
) -> dict[str, list[TaskRunResult]]:
    manifests = prepare_task_datasets(config)
    results_by_dataset: dict[str, list[TaskRunResult]] = {}
    result_rows: list[dict[str, Any]] = []

    for dataset_name, manifest in progress_iter(
        manifests.items(),
        desc="Task benchmark datasets",
        total=len(manifests),
        unit="dataset",
        leave=True,
    ):
        examples = load_task_examples(manifest)
        train_examples = [e for e in examples if e.split == "train"]
        eval_examples = [e for e in examples if e.split == "test"]
        model_name = config.model_name_by_dataset[dataset_name]
        dataset_results: list[TaskRunResult] = []

        with progress_task(total=len(config.methods), desc=f"{dataset_name} methods", unit="method", leave=False) as method_progress:
            for method_name in config.methods:
                method_progress.set_postfix_str(method_name)
                dataset_results.append(_run_method(
                    method_name, eval_examples, train_examples,
                    model_name, dataset_name, config,
                ))
                method_progress.update(1)

        results_by_dataset[dataset_name] = dataset_results
        result_rows.extend(_result_to_row(r) for r in dataset_results)

    result_dir = config.artifacts.results / "task_efficiency"
    write_json(
        result_dir / "task_results.json",
        {name: [_result_to_json(r) for r in results] for name, results in results_by_dataset.items()},
    )
    write_json(result_dir / "run_config.json", {"task_track_config": dataclass_to_dict(config)})
    write_csv(result_dir / "task_results.csv", result_rows)
    return results_by_dataset
