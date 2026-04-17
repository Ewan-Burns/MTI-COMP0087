'''
 Detection Scoring — builds and evaluates the transfer matrix.
 accuracy @ calibrated FPR (e.g. 5%).
'''

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ..config import QuickRunConfig, SupervisedTrainingConfig
from ..detectors.binoculars import score_binoculars_texts
from ..detectors.fastdetectgpt import score_fastdetectgpt_texts
from ..detectors.hf_pipeline import DEFAULT_HF_DETECTOR_IDS, score_hf_detector_texts
from ..detectors.supervised import (
    load_decision_threshold,
    load_training_metadata,
    score_supervised_detector_texts,
)
from ..progress import progress_task, progress_write
from ..results import read_json, write_csv, write_json, save_heatmap, render_matrices_by_detector
from ..types import DatasetManifest, MatrixCellResult

from ._detection_data import (
    _artifact_root_from_manifest,
    _load_generation_records_cached,
    _load_prompt_records_cached,
    load_prompt_records,
)




# Clean scores
def _sanitize_scores(scores: list[float], fallback: float = 0.5) -> list[float]:
    result = []
    for score in scores:
        v = float(score)
        if v != v or v in (float("inf"), float("-inf")):
            v = fallback
        result.append(v)
    return result


def _accuracy_fpr_tpr(labels: list[int], scores: list[float], *, threshold: float) -> tuple[float, float, float]:
    preds = [1 if s >= threshold else 0 for s in scores]
    accuracy = float(accuracy_score(labels, preds))
    human_total = max(1, labels.count(0))
    ai_total = max(1, labels.count(1))
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    return accuracy, float(fp / human_total), float(tp / ai_total)


# Calibrate a decision threshold so that the FPR on human-only scores ≈ target_fpr.
def _threshold_for_target_fpr(human_scores: list[float], *, target_fpr: float) -> tuple[float, float]:
    sanitized = _sanitize_scores(human_scores)
    if not sanitized:
        return 0.5, 0.0
    sorted_scores = sorted(sanitized)
    idx = min(max(int(len(sorted_scores) * max(0.0, min(1.0, 1.0 - target_fpr))), 0), len(sorted_scores) - 1)
    threshold = float(sorted_scores[idx])
    actual_fpr = float(sum(s >= threshold for s in sanitized) / len(sanitized))
    return threshold, actual_fpr


def _metrics_from_scores(labels: list[int], scores: list[float], *, threshold: float = 0.5) -> tuple[float, float, float, float, float]:
    sanitized = _sanitize_scores(scores)
    try:
        auroc = float(roc_auc_score(labels, sanitized))
    except ValueError:
        auroc = 0.5
    accuracy, fpr, tpr = _accuracy_fpr_tpr(labels, sanitized, threshold=threshold)
    preds = [1 if s >= threshold else 0 for s in sanitized]
    return auroc, accuracy, float(f1_score(labels, preds, zero_division=0)), fpr, tpr


def _mean_ai_probability(scores: list[float], num_human_texts: int) -> float:
    ai_scores = _sanitize_scores(scores)[num_human_texts:]
    return float(sum(ai_scores) / len(ai_scores)) if ai_scores else 0.0


# Scoring / evaluation

def _cell_result_to_dict(result: MatrixCellResult) -> dict[str, Any]:
    return {
        "train_method": result.train_method,
        "test_method": result.test_method,
        "detector_name": result.detector_name,
        "auroc": result.auroc,
        "accuracy": result.accuracy,
        "f1": result.f1,
        "mean_ai_prob": result.mean_ai_prob,
        "threshold": result.threshold,
        "fpr": result.fpr,
        "tpr": result.tpr,
        "metadata": result.metadata,
    }


def _evaluate_score_only_detector(
    *,
    detector_name: str,
    scoring_fn: Any,
    texts: list[str],
    labels: list[int],
    kwargs: dict[str, Any],
    target_method: str,
    score_multiplier: float = 1.0,
    threshold: float = 0.5,
) -> MatrixCellResult:
    raw_scores = [float(s) for s in scoring_fn(texts, **kwargs)]
    scores = [score_multiplier * s for s in raw_scores]
    auroc, accuracy, f1, fpr, tpr = _metrics_from_scores(labels, scores, threshold=threshold)
    return MatrixCellResult(
        train_method="score_only",
        test_method=target_method,
        detector_name=detector_name,
        auroc=auroc,
        accuracy=accuracy,
        f1=f1,
        mean_ai_prob=_mean_ai_probability(scores, labels.count(0)),
        threshold=threshold,
        fpr=fpr,
        tpr=tpr,
        metadata={"score_multiplier": score_multiplier, "threshold": threshold},
    )


# Collect human-authored texts for threshold calibration.
def _calibration_human_texts(prompt_manifest: DatasetManifest) -> list[str]:
    records = [r for r in load_prompt_records(prompt_manifest) if r.reference_text]
    validation_texts = [r.reference_text for r in records if r.split == "validation"]
    if validation_texts:
        return validation_texts
    train_texts = [r.reference_text for r in records if r.split == "train"]
    if not train_texts:
        return []
    return train_texts[: max(1, int(len(train_texts) * 0.1))]


def _default_unsupervised_model_pairs(quick_config: QuickRunConfig) -> dict[str, dict[str, str]]:
    gen_model = quick_config.model.publication_model_id
    if "-Instruct" in gen_model:
        main_model = gen_model.replace("-Instruct", "")
        aux_model = gen_model
    else:
        main_model = gen_model
        aux_model = gen_model + "-Instruct"
    return {
        "Binoculars": {"main_model_name": main_model, "aux_model_name": aux_model},
        "FastDetectGPT": {"main_model_name": main_model, "aux_model_name": aux_model},
    }


# Build the test-set text lists for one target method: paired human + AI texts, with labels.
def _evaluation_texts_for_target(
    prompt_manifest: DatasetManifest,
    generation_path: Path,
) -> tuple[list[str], list[str], list[int]]:
    human_texts, ai_texts, labels = _evaluation_texts_for_target_cached(
        str(prompt_manifest.records_path.expanduser().resolve()),
        str(generation_path.expanduser().resolve()),
    )
    return list(human_texts), list(ai_texts), list(labels)


@lru_cache(maxsize=128)
def _evaluation_texts_for_target_cached(
    prompt_records_path: str,
    generation_path: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    prompt_records = [
        r for r in _load_prompt_records_cached(prompt_records_path)
        if r.reference_text and r.split == "test"
    ]
    gen_by_id = {r.prompt_id: r for r in _load_generation_records_cached(generation_path)}
    human_texts: list[str] = []
    ai_texts: list[str] = []
    for pr in prompt_records:
        gen = gen_by_id.get(pr.prompt_id)
        if gen is None:
            continue
        human_texts.append(pr.reference_text)
        ai_texts.append(gen.text)
    labels = [0] * len(human_texts) + [1] * len(ai_texts)
    return tuple(human_texts), tuple(ai_texts), tuple(labels)


# Evaluate one cell of the supervised transfer matrix: a detector trained on
# train_method's data, tested against target_method's AI text.
def _evaluate_supervised_cell(
    *,
    human_texts: list[str],
    ai_texts: list[str],
    labels: list[int],
    train_method: str,
    target_method: str,
    architecture: str,
    checkpoint_dir: Path,
    training_config: SupervisedTrainingConfig,
) -> MatrixCellResult:
    texts = human_texts + ai_texts
    decision_threshold = load_decision_threshold(checkpoint_dir)
    scores = score_supervised_detector_texts(
        texts,
        checkpoint_dir=checkpoint_dir,
        device=training_config.device,
        batch_size=training_config.eval_batch_size,
        max_length=training_config.max_length,
    )
    auroc, accuracy, f1, fpr, tpr = _metrics_from_scores(labels, scores, threshold=decision_threshold)
    training_metadata = load_training_metadata(checkpoint_dir)
    return MatrixCellResult(
        train_method=train_method,
        test_method=target_method,
        detector_name=architecture,
        auroc=auroc,
        accuracy=accuracy,
        f1=f1,
        mean_ai_prob=_mean_ai_probability(scores, len(human_texts)),
        threshold=decision_threshold,
        fpr=fpr,
        tpr=tpr,
        metadata={
            "checkpoint_dir": str(checkpoint_dir),
            "decision_threshold": decision_threshold,
            "target_fpr": training_metadata.get("target_fpr", training_config.target_fpr),
            "validation_fpr": training_metadata.get("validation_fpr"),
        },
    )


# Fill the full supervised transfer matrix: for each (architecture, train_method, target_method).
def _run_supervised_matrix(
    evaluation_cache: dict[str, tuple[list[str], list[str], list[int]]],
    trained_checkpoints: dict[str, dict[str, Path]],
    generation_paths: dict[str, Path],
    training_config: SupervisedTrainingConfig,
) -> list[MatrixCellResult]:
    results: list[MatrixCellResult] = []
    total = sum(len(ckpts) for ckpts in trained_checkpoints.values()) * len(generation_paths)
    with progress_task(total=total, desc="Supervised matrix", unit="cell", leave=True) as progress:
        for architecture, checkpoints in trained_checkpoints.items():
            for train_method, checkpoint_dir in checkpoints.items():
                for target_method in generation_paths:
                    progress.set_postfix(architecture=architecture, train=train_method, test=target_method)
                    human_texts, ai_texts, labels = evaluation_cache[target_method]
                    results.append(_evaluate_supervised_cell(
                        human_texts=human_texts,
                        ai_texts=ai_texts,
                        labels=labels,
                        train_method=train_method,
                        target_method=target_method,
                        architecture=architecture,
                        checkpoint_dir=checkpoint_dir,
                        training_config=training_config,
                    ))
                    progress.update(1)
    return results


# Score-only path for HuggingFace pipeline detectors (pre-trained, no fine-tuning).
# Threshold is calibrated once per detector on human validation texts.
def _run_hf_matrix(
    evaluation_cache: dict[str, tuple[list[str], list[str], list[int]]],
    calibration_humans: list[str],
    hf_detector_ids: list[str],
    generation_paths: dict[str, Path],
    training_config: SupervisedTrainingConfig,
    quick_config: QuickRunConfig,
) -> list[MatrixCellResult]:
    results: list[MatrixCellResult] = []
    total = len(hf_detector_ids) * len(generation_paths)
    with progress_task(total=total, desc="HF detector matrix", unit="cell", leave=True) as progress:
        for detector_id in hf_detector_ids:
            cal_scores = score_hf_detector_texts(
                calibration_humans,
                model_id_or_path=detector_id,
                cache_dir=quick_config.ai_detector_cache_dir,
                token=quick_config.model.hf_token,
                max_chars=quick_config.ai_detector_max_chars,
                device=training_config.device,
                batch_size=training_config.eval_batch_size,
            )
            decision_threshold, cal_fpr = _threshold_for_target_fpr(cal_scores, target_fpr=training_config.target_fpr)

            for target_method in generation_paths:
                progress.set_postfix(detector=detector_id, test=target_method)
                human_texts, ai_texts, labels = evaluation_cache[target_method]
                scores = score_hf_detector_texts(
                    human_texts + ai_texts,
                    model_id_or_path=detector_id,
                    cache_dir=quick_config.ai_detector_cache_dir,
                    token=quick_config.model.hf_token,
                    max_chars=quick_config.ai_detector_max_chars,
                    device=training_config.device,
                    batch_size=training_config.eval_batch_size,
                )
                auroc, accuracy, f1, fpr, tpr = _metrics_from_scores(labels, scores, threshold=decision_threshold)
                results.append(MatrixCellResult(
                    train_method="score_only",
                    test_method=target_method,
                    detector_name=detector_id,
                    auroc=auroc,
                    accuracy=accuracy,
                    f1=f1,
                    mean_ai_prob=_mean_ai_probability(scores, len(human_texts)),
                    threshold=decision_threshold,
                    fpr=fpr,
                    tpr=tpr,
                    metadata={"target_fpr": training_config.target_fpr, "validation_fpr": cal_fpr},
                ))
                progress.update(1)
    return results


def _calibrate_unsupervised_detector(
    name: str,
    scoring_fn: Any,
    calibration_humans: list[str],
    model_pair: dict[str, str],
    training_config: SupervisedTrainingConfig,
    score_multiplier: float = 1.0,
) -> tuple[dict[str, Any], float, float]:
    kwargs = dict(model_pair)
    kwargs.setdefault("device", training_config.device)
    kwargs.setdefault("batch_size", training_config.eval_batch_size)
    raw = [float(s) for s in scoring_fn(calibration_humans, **kwargs)]
    cal_scores = [score_multiplier * s for s in raw]
    threshold, cal_fpr = _threshold_for_target_fpr(cal_scores, target_fpr=training_config.target_fpr)
    return kwargs, threshold, cal_fpr


# Score-only path for unsupervised detectors (Binoculars, FastDetectGPT).
# Binoculars scores are negated (lower = more AI-like) so we multiply by -1.
def _run_unsupervised_matrix(
    evaluation_cache: dict[str, tuple[list[str], list[str], list[int]]],
    calibration_humans: list[str],
    generation_paths: dict[str, Path],
    training_config: SupervisedTrainingConfig,
    quick_config: QuickRunConfig,
    unsupervised_model_pairs: dict[str, dict[str, str]] | None,
) -> list[MatrixCellResult]:
    model_pairs = unsupervised_model_pairs or _default_unsupervised_model_pairs(quick_config)
    detector_configs = {
        "Binoculars": (score_binoculars_texts, -1.0),
        "FastDetectGPT": (score_fastdetectgpt_texts, 1.0),
    }
    calibrated: dict[str, tuple[dict[str, Any], float, float]] = {}
    for name, (scoring_fn, multiplier) in detector_configs.items():
        if name in model_pairs:
            calibrated[name] = _calibrate_unsupervised_detector(
                name, scoring_fn, calibration_humans, model_pairs[name],
                training_config, score_multiplier=multiplier,
            )

    results: list[MatrixCellResult] = []
    total = len(calibrated) * len(generation_paths)
    with progress_task(total=total, desc="Score-only matrix", unit="cell", leave=True) as progress:
        for target_method in generation_paths:
            human_texts, ai_texts, labels = evaluation_cache[target_method]
            texts = human_texts + ai_texts
            for name, (kwargs, threshold, cal_fpr) in calibrated.items():
                scoring_fn, multiplier = detector_configs[name]
                progress.set_postfix(detector=name, test=target_method)
                result = _evaluate_score_only_detector(
                    detector_name=name,
                    scoring_fn=scoring_fn,
                    texts=texts,
                    labels=labels,
                    kwargs=kwargs,
                    target_method=target_method,
                    score_multiplier=multiplier,
                    threshold=threshold,
                )
                result.metadata["target_fpr"] = training_config.target_fpr
                result.metadata["validation_fpr"] = cal_fpr
                results.append(result)
                progress.update(1)
    return results


def run_detection_matrix(
    prompt_manifest: DatasetManifest,
    generation_paths: dict[str, Path],
    trained_checkpoints: dict[str, dict[str, Path]],
    *,
    training_config: SupervisedTrainingConfig,
    quick_config: QuickRunConfig | None = None,
    hf_detector_ids: list[str] | None = None,
    run_unsupervised: bool = False,
    unsupervised_model_pairs: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    quick_config = quick_config or QuickRunConfig.from_env()
    artifact_root = _artifact_root_from_manifest(prompt_manifest)
    hf_detector_ids = list(DEFAULT_HF_DETECTOR_IDS) if hf_detector_ids is None else list(hf_detector_ids)
    result_dir = artifact_root / "results" / "generative_detection" / prompt_manifest.name

    supervised_cache_path = result_dir / "matrix_supervised.json"
    # Invalidate supervised cache if retraining was requested
    if training_config.force_retrain and supervised_cache_path.exists():
        supervised_cache_path.unlink()
    # Reuse cached supervised matrix only if it exists and matches current method set
    if supervised_cache_path.exists():
        cached = read_json(supervised_cache_path)
        cached_methods = {row.get("test_method") for row in (cached or [])}
        expected_methods = set(generation_paths.keys())
        if cached and cached_methods == expected_methods:
            progress_write(f"Supervised matrix cached ({len(cached)} cells), skipping.")
            results_supervised = [
                MatrixCellResult(**{k: v for k, v in row.items() if k != "metadata"}, metadata=row.get("metadata", {}))
                for row in cached
            ]
        else:
            progress_write("Supervised matrix cache stale (method set changed), recomputing.")
            supervised_cache_path.unlink(missing_ok=True)
            cached = None
            results_supervised = None
    else:
        cached = None
        results_supervised = None

    if results_supervised is None:
        calibration_humans = _calibration_human_texts(prompt_manifest)
        evaluation_cache = {
            target: _evaluation_texts_for_target(prompt_manifest, path)
            for target, path in generation_paths.items()
        }
        results_supervised = _run_supervised_matrix(evaluation_cache, trained_checkpoints, generation_paths, training_config)
        # Save immediately so a later crash doesn't lose this work
        supervised_rows = [_cell_result_to_dict(r) for r in results_supervised]
        write_json(supervised_cache_path, supervised_rows)
        write_csv(result_dir / "matrix_supervised.csv", supervised_rows)
        progress_write(f"Supervised matrix saved ({len(supervised_rows)} cells)")

    # Score-only detectors (HF pipelines + unsupervised)
    calibration_humans = _calibration_human_texts(prompt_manifest)
    evaluation_cache = {
        target: _evaluation_texts_for_target(prompt_manifest, path)
        for target, path in generation_paths.items()
    }

    results_score_only = _run_hf_matrix(
        evaluation_cache, calibration_humans, hf_detector_ids,
        generation_paths, training_config, quick_config,
    )

    if run_unsupervised:
        # Free supervised classifier models before loading unsupervised scoring models
        import gc
        from ..detectors.supervised import _load_classifier_cached
        if hasattr(_load_classifier_cached, 'cache_clear'):
            _load_classifier_cached.cache_clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        results_score_only.extend(_run_unsupervised_matrix(
            evaluation_cache, calibration_humans, generation_paths,
            training_config, quick_config, unsupervised_model_pairs,
        ))

    supervised_rows = [_cell_result_to_dict(r) for r in results_supervised]
    score_only_rows = [_cell_result_to_dict(r) for r in results_score_only]

    for name, rows in [("supervised", supervised_rows), ("score_only", score_only_rows), ("unsupervised", score_only_rows)]:
        write_json(result_dir / f"matrix_{name}.json", rows)
        write_csv(result_dir / f"matrix_{name}.csv", rows)

    for detector_name in sorted({r.detector_name for r in results_supervised}):
        subset = [r for r in results_supervised if r.detector_name == detector_name]
        for metric in ("accuracy", "auroc"):
            title = (
                f"{detector_name} accuracy@{training_config.target_fpr:.0%}FPR"
                if metric == "accuracy"
                else f"{detector_name} {metric}"
            )
            save_heatmap(
                subset,
                artifact_root / "plots" / "generative_detection" / prompt_manifest.name / f"{detector_name}_{metric}.png",
                title=title,
                value_field=metric,
            )

    write_json(result_dir / "summary.json", {
        "supervised_accuracy_at_target_fpr_tables": render_matrices_by_detector(results_supervised, value_field="accuracy"),
        "supervised_auroc_tables": render_matrices_by_detector(results_supervised, value_field="auroc"),
        "score_only_accuracy_at_target_fpr_tables": render_matrices_by_detector(results_score_only, value_field="accuracy"),
        "score_only_auroc_tables": render_matrices_by_detector(results_score_only, value_field="auroc"),
        "target_fpr": training_config.target_fpr,
    })

    return {"supervised": results_supervised, "score_only": results_score_only}


# Test detector robustness
def evaluate_human_domain_shift(
    *,
    shift_manifest: DatasetManifest,
    prompt_manifest: DatasetManifest,
    generation_paths: dict[str, Path],
    trained_checkpoints: dict[str, dict[str, Path]],
    training_config: SupervisedTrainingConfig,
) -> list[dict[str, Any]]:
    human_texts = [r.reference_text for r in load_prompt_records(shift_manifest) if r.reference_text]
    evaluation_cache = {
        target: _evaluation_texts_for_target(prompt_manifest, path)
        for target, path in generation_paths.items()
    }
    results: list[dict[str, Any]] = []
    total = sum(len(ckpts) for ckpts in trained_checkpoints.values()) * len(generation_paths)

    with progress_task(total=total, desc="Human-domain shift", unit="cell", leave=True) as progress:
        for architecture, checkpoints in trained_checkpoints.items():
            for train_method, checkpoint_dir in checkpoints.items():
                for target_method in generation_paths:
                    progress.set_postfix(architecture=architecture, train=train_method, test=target_method)
                    _, ai_texts, _ = evaluation_cache[target_method]
                    n = min(len(human_texts), len(ai_texts))
                    texts = human_texts[:n] + ai_texts[:n]
                    labels = [0] * n + [1] * n
                    scores = score_supervised_detector_texts(
                        texts,
                        checkpoint_dir=checkpoint_dir,
                        device=training_config.device,
                        batch_size=training_config.eval_batch_size,
                        max_length=training_config.max_length,
                    )
                    decision_threshold = load_decision_threshold(checkpoint_dir)
                    auroc, accuracy, f1, fpr, tpr = _metrics_from_scores(labels, scores, threshold=decision_threshold)
                    results.append({
                        "detector_name": architecture,
                        "train_method": train_method,
                        "test_method": target_method,
                        "auroc": auroc,
                        "accuracy": accuracy,
                        "f1": f1,
                        "threshold": decision_threshold,
                        "fpr": fpr,
                        "tpr": tpr,
                        "mean_ai_prob": _mean_ai_probability(scores, n),
                    })
                    progress.update(1)

    result_dir = _artifact_root_from_manifest(prompt_manifest) / "results" / "generative_detection" / prompt_manifest.name
    write_json(result_dir / "human_domain_shift.json", results)
    write_csv(result_dir / "human_domain_shift.csv", results)
    return results
