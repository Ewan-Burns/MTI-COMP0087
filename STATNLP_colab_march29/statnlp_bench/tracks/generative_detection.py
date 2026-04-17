# ==========================================================================
# Generative Detection Track — 4-stage pipeline:
#   1. Data prep:   prepare prompt datasets (local MT-Bench + optional external)
#   2. Generation:  run each generation method to produce AI texts per prompt
#   3. Training:    train supervised detectors on (human, AI) text pairs
#   4. Evaluation:  build a transfer matrix — every detector × every method
#
# The "transfer matrix" is the core evaluation artifact: rows = training
# source (which method's AI text the detector was trained on), columns =
# test target (which method's AI text is being detected). This reveals how
# well detectors generalise across generation methods.
# ==========================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import (
    GenerativeDetectionConfig,
    QuickRunConfig,
    SupervisedTrainingConfig,
    dataclass_to_dict,
)
from ..datasets.human_shift import prepare_human_shift_dataset
from ..datasets.mt_bench import prepare_mt_bench_dataset
from ..datasets.raid_like import prepare_raid_like_dataset
from ..detectors.hf_pipeline import DEFAULT_HF_DETECTOR_IDS, score_hf_detector_texts
from ..detectors.supervised import (
    load_detection_examples,
    train_supervised_detector,
)
from ..methods.profiles import publication_lane_for_method, resolve_method_profile
from ..methods.publication import publication_metadata_for_method
from ..progress import progress_iter, progress_task, progress_write
from ..registry import get_method
from ..results import read_json, read_jsonl, write_json, write_jsonl
from ..types import DatasetManifest, DetectionExample, PromptRecord

# Import side effects — these modules register methods/detectors/datasets on import.
from .. import registry as _registry  # noqa: F401
from ..datasets import human_shift as _human_shift  # noqa: F401
from ..datasets import mt_bench as _mt_bench  # noqa: F401
from ..datasets import raid_like as _raid_like  # noqa: F401
from ..detectors import binoculars as _binoculars  # noqa: F401
from ..detectors import fastdetectgpt as _fastdetectgpt  # noqa: F401
from ..detectors import hf_pipeline as _hf_pipeline  # noqa: F401
from ..detectors import supervised as _supervised  # noqa: F401
from ..methods import self_consistency as _self_consistency  # noqa: F401


def _clear_model_caches() -> None:
    """Free GPU/CPU memory by clearing all cached model objects between pipeline stages."""
    import gc
    from ..detectors._model_loading import load_model_pair
    from ..detectors.supervised import _load_classifier_cached
    from ..detectors.hf_pipeline import _load_hf_detector
    from ..methods.publication import _load_publication_model_cached
    for cached_fn in [load_model_pair, _load_classifier_cached, _load_hf_detector, _load_publication_model_cached]:
        if hasattr(cached_fn, 'cache_clear'):
            cached_fn.cache_clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

from ._detection_data import (
    _artifact_root_from_manifest,
    load_generation_records,
    load_prompt_records,
)
from ._detection_scoring import (
    evaluate_human_domain_shift,
    run_detection_matrix,
)


# Resolve which generation methods to run, with a priority cascade:
# explicit list > config.methods > named profile > publication defaults > built-in defaults.
def _resolved_method_names(config: GenerativeDetectionConfig, explicit_methods: list[str] | None = None) -> list[str]:
    raw = (
        explicit_methods
        or config.methods
        or (resolve_method_profile(config.method_profile) if config.method_profile else None)
        or (resolve_method_profile("publication_core") if config.publication_mode else None)
        or resolve_method_profile("publication_core")
    )
    return list(raw)


# Combine multiple prompt sources (e.g. local MT-Bench + external RAID) into one dataset.
# Prompt IDs are namespaced as "{manifest_name}:{original_id}" to avoid collisions.
def _merge_prompt_manifests(
    manifests: list[DatasetManifest],
    *,
    output_dir: str | Path,
    dataset_name: str,
) -> DatasetManifest:
    total_rows = sum(len(read_jsonl(m.records_path)) for m in manifests)
    merged_rows: list[dict[str, Any]] = []
    with progress_task(total=total_rows, desc=f"Merge dataset {dataset_name}", unit="row", leave=False) as progress:
        for manifest in manifests:
            for row in read_jsonl(manifest.records_path):
                progress.set_postfix_str(manifest.name)
                progress.update(1)
                row = dict(row)
                row["prompt_id"] = f"{manifest.name}:{row['prompt_id']}"
                row.setdefault("metadata", {})
                row["metadata"]["source_manifest"] = manifest.name
                merged_rows.append(row)

    root_dir = Path(output_dir).expanduser().resolve()
    records_path = root_dir / "prompts.jsonl"
    write_jsonl(records_path, merged_rows)
    metadata = {
        "name": dataset_name,
        "track": "generative_detection",
        "external": any(m.external for m in manifests),
        "component_manifests": [m.name for m in manifests],
        "num_prompts": len(merged_rows),
    }
    write_json(root_dir / "manifest.json", metadata)
    return DatasetManifest(
        name=dataset_name,
        track="generative_detection",
        external=metadata["external"],
        root_dir=root_dir,
        records_path=records_path,
        metadata=metadata,
    )


# --- Stage 1: Data Preparation ---
# Builds the prompt dataset. If external data is enabled, merges local + external sources.
def prepare_prompt_dataset(
    config: GenerativeDetectionConfig,
    *,
    question_file: str | Path = "datasets/mt_bench_prompts/raw/question.jsonl",
    external_detection_kwargs: dict[str, Any] | None = None,
) -> DatasetManifest:
    use_external = config.use_external_detection_data and external_detection_kwargs is not None
    manifests: list[DatasetManifest] = []

    if config.include_local_mt_bench or not use_external:
        local_name = config.dataset_name if not use_external else "mt_bench_local"
        manifests.append(
            prepare_mt_bench_dataset(
                question_file=question_file,
                output_dir=config.artifacts.datasets / "generative_detection" / local_name,
                dataset_name=local_name,
                split_seed=config.split_seed,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                max_prompts=config.max_prompts,
                balance_categories=config.balance_categories,
            )
        )

    if use_external:
        ext_name = str(external_detection_kwargs.get("dataset_name") or "raid_like")
        manifests.append(
            prepare_raid_like_dataset(
                output_dir=config.artifacts.datasets / "generative_detection" / ext_name,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                **external_detection_kwargs,
            )
        )

    if len(manifests) == 1:
        return manifests[0]
    return _merge_prompt_manifests(
        manifests,
        output_dir=config.artifacts.datasets / "generative_detection" / config.dataset_name,
        dataset_name=config.dataset_name,
    )


# Dispatch a single batch to the registry-based generation method.
def _run_generation_batch(
    batch_records: list[PromptRecord],
    *,
    method_name: str,
    method_spec: Any,
    batch_start: int,
    quick_config: QuickRunConfig,
    dataset_name: str | None = None,
) -> Any:
    prompts = [r.prompt_text for r in batch_records]
    return method_spec.run(
        prompts=prompts,
        method_name=method_name,
        prompt_start_idx=batch_start + 1,
        run_idx=0,
        config=quick_config,
        dataset_name=dataset_name,
    )


# --- Stage 2: Generation ---
# For each method, generate AI text for every prompt and cache to disk as JSONL.
# Cached outputs are reused if the method manifest (model, config hash, etc.) matches.
def build_generation_cache(
    prompt_manifest: DatasetManifest,
    *,
    methods: list[str] | None = None,
    method_profile: str | None = None,
    quick_config: QuickRunConfig | None = None,
    publication_mode: bool = False,
    force: bool = False,
) -> dict[str, Path]:
    methods = list(methods or resolve_method_profile(method_profile) or resolve_method_profile("publication_core"))
    quick_config = quick_config or QuickRunConfig.from_env()
    records = load_prompt_records(prompt_manifest)
    output_dir = _artifact_root_from_manifest(prompt_manifest) / "generations" / prompt_manifest.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}

    write_json(output_dir / "manifest.json", {
        "dataset_name": prompt_manifest.name,
        "methods": methods,
        "num_prompts": len(records),
        "sample_seed": quick_config.sample_seed,
        "max_tokens": quick_config.max_tokens,
        "generation_batch_size": quick_config.generation_batch_size,
        "publication_model_id": quick_config.model.publication_model_id,
        "publication_model_revision": quick_config.model.publication_model_revision,
        "method_profile": method_profile,
    })

    base_batch_size = max(1, int(quick_config.generation_batch_size))

    # Contrastive stores hidden states per token; CFG runs 2x forward passes.
    # Both need reduced batch sizes to avoid OOM.
    # Reduced batch sizes for memory-heavy methods:
    #   contrastive: stores hidden states for similarity — ~8x more memory
    #   cfg: 2x forward passes per step — ~4x more memory
    #   mbr: generates num_candidates (16) sequences per prompt — ~16x more sequences
    _BATCH_SIZE_DIVISORS = {"contrastive": 8, "cfg": 4, "mbr": 16}

    def _effective_batch_size(method_name: str) -> int:
        from ..methods.publication import TRANSFORMERS_METHODS, MBR_METHODS, _CUSTOM_PROCESSOR_METHODS
        if method_name in MBR_METHODS:
            mode = "mbr"
        elif method_name in TRANSFORMERS_METHODS:
            mode = TRANSFORMERS_METHODS[method_name].get("mode", "sample")
        elif method_name in _CUSTOM_PROCESSOR_METHODS:
            mode = "sample"
        else:
            mode = "sample"
        divisor = _BATCH_SIZE_DIVISORS.get(mode, 1)
        return max(1, base_batch_size // divisor)

    with progress_task(total=len(methods), desc="Generation methods", unit="method", leave=True) as method_progress:
        for method_name in methods:
            output_path = output_dir / f"{method_name}.jsonl"
            output_paths[method_name] = output_path
            method_progress.set_postfix_str(method_name)

            method_spec = get_method(method_name)
            method_metadata = dict(method_spec.metadata)
            paper_lane = publication_lane_for_method(method_name)

            if not method_spec.supports_dataset(prompt_manifest.name):
                raise ValueError(f"Method {method_name} does not support dataset {prompt_manifest.name}")
            if method_metadata.get("backend") == "publication_torch":
                method_metadata.update(publication_metadata_for_method(method_name, quick_config))

            expected_manifest = {
                "method_name": method_name,
                "dataset_name": prompt_manifest.name,
                "backend": method_metadata.get("backend", "publication_torch"),
                "paper_lane": paper_lane,
                "source_implementation": method_metadata.get("source_implementation"),
                "source_url": method_metadata.get("source_url"),
                "source_version": method_metadata.get("source_version"),
                "model_id": method_metadata.get("model_id", quick_config.model.publication_model_id),
                "config_hash": method_metadata.get("config_hash"),
                "publication_fair": bool(method_metadata.get("publication_fair", False)),
                "third_party_manifest": method_metadata.get("third_party_manifest"),
            }
            method_manifest_path = output_dir / f"{method_name}.manifest.json"

            # Skip regeneration if a cached file exists and its manifest matches exactly.
            if output_path.exists() and not force:
                cached = {}
                if method_manifest_path.exists():
                    try:
                        cached = read_json(method_manifest_path)
                    except Exception:
                        cached = {}
                if all(cached.get(k) == v for k, v in expected_manifest.items()):
                    progress_write(f"Reusing cached generations for {method_name} at {output_path}")
                    method_progress.update(1)
                    continue
                progress_write(f"Regenerating {method_name}: cache metadata mismatch at {output_path}")

            rows = []
            batch_size = _effective_batch_size(method_name)
            total_batches = (len(records) + batch_size - 1) // batch_size
            for batch_start in progress_iter(
                range(0, len(records), batch_size),
                desc=f"Generate {method_name}",
                total=total_batches,
                unit="batch",
                leave=False,
            ):
                batch_records = records[batch_start : batch_start + batch_size]
                result = _run_generation_batch(
                    batch_records,
                    method_name=method_name,
                    method_spec=method_spec,
                    batch_start=batch_start,
                    quick_config=quick_config,
                    dataset_name=prompt_manifest.name,
                )
                per_text_meta = result.metadata.get("per_text", []) if isinstance(result.metadata.get("per_text"), list) else []
                for idx, (record, text) in enumerate(zip(batch_records, result.texts)):
                    row_meta = {"category": record.category, "paper_lane": paper_lane, **method_metadata}
                    if idx < len(per_text_meta) and isinstance(per_text_meta[idx], dict):
                        row_meta.update(per_text_meta[idx])
                    rows.append({
                        "prompt_id": record.prompt_id,
                        "method_name": method_name,
                        "run_id": 0,
                        "seed": row_meta.get("seed", quick_config.sample_seed),
                        "text": text,
                        "metadata": row_meta,
                    })

            write_jsonl(output_path, rows)
            write_json(method_manifest_path, expected_manifest)
            method_progress.update(1)

    return output_paths


def _human_row_for_prompt(record: PromptRecord, *, copy_idx: int, split: str) -> dict[str, Any]:
    return {
        "example_id": f"human:{record.prompt_id}:{copy_idx}",
        "prompt_id": record.prompt_id,
        "source_method": "human",
        "text": record.reference_text,
        "label": 0,
        "split": split,
        "metadata": {"category": record.category},
    }


# Build a balanced detection corpus: for each prompt, pair the human reference text
# (label=0) with AI-generated text (label=1). When source_methods=["mixture"], all
# available methods contribute AI texts — each prompt gets one human copy per method.
def build_detection_corpus(
    prompt_manifest: DatasetManifest,
    generation_paths: dict[str, Path],
    *,
    source_methods: list[str],
    output_dir: str | Path,
) -> Path:
    prompt_records = [r for r in load_prompt_records(prompt_manifest) if r.reference_text]
    is_mixture = source_methods == ["mixture"]
    generation_by_method = {
        name: {rec.prompt_id: rec for rec in load_generation_records(path)}
        for name, path in generation_paths.items()
        if name in source_methods or is_mixture
    }
    methods_to_use = list(generation_by_method) if is_mixture else source_methods

    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        split_records = [r for r in prompt_records if r.split == split]
        for record in split_records:
            for copy_idx, method_name in enumerate(methods_to_use):
                generation = generation_by_method.get(method_name, {}).get(record.prompt_id)
                if generation is None:
                    continue
                rows.append(_human_row_for_prompt(record, copy_idx=copy_idx, split=split))
                rows.append({
                    "example_id": f"{method_name}:{record.prompt_id}:{copy_idx}",
                    "prompt_id": record.prompt_id,
                    "source_method": method_name,
                    "text": generation.text,
                    "label": 1,
                    "split": split,
                    "metadata": generation.metadata,
                })

    # Dubois et al. mix generated texts only, shuffle with seed=47, cap at 1000 AI examples,
    # then pair each AI example with its corresponding human reference.
    if is_mixture:
        import random
        train_ai = [r for r in rows if r["split"] == "train" and r["label"] == 1]
        train_human_by_pid = {}
        for r in rows:
            if r["split"] == "train" and r["label"] == 0:
                train_human_by_pid[r["prompt_id"]] = r
        other_rows = [r for r in rows if r["split"] != "train"]
        rng = random.Random(47)
        rng.shuffle(train_ai)
        train_ai = train_ai[:1000]
        # Pair each selected AI example with its human reference (deduplicate humans)
        seen_human_pids: set[str] = set()
        train_rows = []
        for ai_row in train_ai:
            pid = ai_row["prompt_id"]
            if pid not in seen_human_pids:
                human = train_human_by_pid.get(pid)
                if human:
                    train_rows.append(human)
                    seen_human_pids.add(pid)
            train_rows.append(ai_row)
        rows = train_rows + other_rows

    root_dir = Path(output_dir).expanduser().resolve()
    records_path = root_dir / "examples.jsonl"
    split_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    write_jsonl(records_path, rows)
    write_json(root_dir / "manifest.json", {
        "source_methods": source_methods,
        "num_examples": len(rows),
        "num_prompts": len(prompt_records),
        "split_counts": split_counts,
    })
    return records_path


def _split_detection_examples(examples: list[DetectionExample]) -> tuple[list[DetectionExample], list[DetectionExample], list[DetectionExample]]:
    return (
        [e for e in examples if e.split == "train"],
        [e for e in examples if e.split == "validation"],
        [e for e in examples if e.split == "test"],
    )


# If no validation split exists, carve a small hold-out from training data
# (stratified by label) so we can calibrate the decision threshold.
def _ensure_validation_examples(
    train_examples: list[DetectionExample],
    validation_examples: list[DetectionExample],
    *,
    ratio: float,
) -> tuple[list[DetectionExample], list[DetectionExample]]:
    if validation_examples or not train_examples:
        return train_examples, validation_examples
    human = [e for e in train_examples if e.label == 0]
    ai = [e for e in train_examples if e.label == 1]
    holdout_human = max(1, int(len(human) * ratio)) if human else 0
    holdout_ai = max(1, int(len(ai) * ratio)) if ai else 0
    validation = human[:holdout_human] + ai[:holdout_ai]
    validation_ids = {e.example_id for e in validation}
    train = [e for e in train_examples if e.example_id not in validation_ids]
    return train, validation


# --- Stage 3: Training ---
# Train one detector per (architecture × source method), plus a "mixture" detector
# trained on all methods combined. This populates the rows of the transfer matrix.
def train_supervised_detectors(
    prompt_manifest: DatasetManifest,
    generation_paths: dict[str, Path],
    *,
    methods: list[str],
    training_config: SupervisedTrainingConfig,
    output_root: str | Path | None = None,
) -> dict[str, dict[str, Path]]:
    output_root = Path(output_root or "models/detectors").expanduser().resolve()
    trained: dict[str, dict[str, Path]] = {arch: {} for arch in training_config.architecture_model_ids}
    # Each method gets its own detector; "mixture" trains on all methods pooled.
    source_groups = [[m] for m in methods] + [["mixture"]]
    corpus_root = _artifact_root_from_manifest(prompt_manifest) / "corpora"

    with progress_task(total=len(source_groups), desc="Detector source corpora", unit="corpus", leave=True) as source_progress:
        for source_group in source_groups:
            source_name = source_group[0] if source_group != ["mixture"] else "mixture"
            source_progress.set_postfix_str(source_name)

            corpus_path = build_detection_corpus(
                prompt_manifest, generation_paths,
                source_methods=source_group,
                output_dir=corpus_root / prompt_manifest.name / source_name,
            )
            examples = load_detection_examples(corpus_path)
            train_ex, val_ex, _ = _split_detection_examples(examples)
            train_ex, val_ex = _ensure_validation_examples(train_ex, val_ex, ratio=training_config.internal_validation_ratio)

            for arch, model_name in progress_iter(
                training_config.architecture_model_ids.items(),
                desc=f"Train detectors {source_name}",
                total=len(training_config.architecture_model_ids),
                unit="detector",
                leave=False,
            ):
                ref = train_supervised_detector(
                    train_examples=train_ex,
                    validation_examples=val_ex,
                    model_name_or_path=model_name,
                    output_dir=output_root / f"tuned-{arch}" / source_name,
                    config=training_config,
                )
                trained[arch][source_name] = ref.checkpoint_dir
            source_progress.update(1)
    return trained


# Lightweight interactive experiment: generate + score a handful of prompts.
# Not part of the full pipeline — useful for quick sanity checks.
def run_quick_experiment(
    *,
    prompts: list[str],
    methods: list[str] | None = None,
    detector_ids: list[str] | None = None,
    quick_config: QuickRunConfig | None = None,
    print_generated_text: bool | None = None,
) -> dict[str, Any]:
    methods = methods or list(resolve_method_profile("publication_core"))
    detector_ids = detector_ids or DEFAULT_HF_DETECTOR_IDS
    quick_config = quick_config or QuickRunConfig.from_env()
    if print_generated_text is None:
        print_generated_text = quick_config.print_generated_text

    outputs_by_method: dict[str, list[str]] = {}
    detector_summary: dict[str, dict[str, list[float]]] = {d: {} for d in detector_ids}
    batch_size = max(1, int(quick_config.generation_batch_size))

    print(f"Prompts configured: {len(prompts)}")
    print(f"Methods configured: {', '.join(methods)}")
    print(f"Detectors configured: {', '.join(detector_ids)}")

    for method_name in progress_iter(methods, desc="Quick methods", total=len(methods), unit="method", leave=True):
        method_spec = get_method(method_name)

        method_outputs: list[str] = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        for batch_start in progress_iter(
            range(0, len(prompts), batch_size),
            desc=f"Generate {method_name}",
            total=total_batches,
            unit="batch",
            leave=False,
        ):
            batch = prompts[batch_start : batch_start + batch_size]
            result = method_spec.run(prompts=batch, method_name=method_name, prompt_start_idx=batch_start + 1, run_idx=0, config=quick_config)
            method_outputs.extend(result.texts)
        outputs_by_method[method_name] = method_outputs

        for detector_id in progress_iter(detector_ids, desc=f"Detectors for {method_name}", total=len(detector_ids), unit="detector", leave=False):
            detector_summary[detector_id][method_name] = score_hf_detector_texts(
                method_outputs,
                model_id_or_path=detector_id,
                cache_dir=quick_config.ai_detector_cache_dir,
                token=quick_config.model.hf_token,
                max_chars=quick_config.ai_detector_max_chars,
                device=quick_config.ai_detector_device,
                batch_size=quick_config.ai_detector_batch_size,
            )

    for prompt_idx, prompt in enumerate(prompts, start=1):
        print(f"\nPrompt {prompt_idx}/{len(prompts)}: {prompt}")
        for method_name in methods:
            text = outputs_by_method.get(method_name, [])[prompt_idx - 1]
            if print_generated_text:
                print(f"\n{method_name}:\n{text}")
            else:
                print(f"\n{method_name}: complete")
            for detector_id in detector_ids:
                score = detector_summary[detector_id][method_name][prompt_idx - 1]
                print(f"Detector {detector_id} AI-probability: {score:.3f}")

    print("\nDetector Summary (higher = more AI-like):")
    for detector_id, scores_by_method in detector_summary.items():
        print(f"\n{detector_id}:")
        for method_name in methods:
            scores = scores_by_method.get(method_name, [])
            if scores:
                print(f"{method_name}: mean={sum(scores) / len(scores):.3f}, n={len(scores)}")

    return {"outputs_by_method": outputs_by_method, "detector_summary": detector_summary}


# --- Stage 4 (orchestrator): Full pipeline ---
# Runs all four stages in sequence and optionally evaluates human domain shift
# (testing detectors on out-of-distribution human text to measure robustness).
def run_full_generative_detection_pipeline(
    *,
    config: GenerativeDetectionConfig,
    quick_config: QuickRunConfig | None = None,
    training_config: SupervisedTrainingConfig | None = None,
    question_file: str | Path = "datasets/mt_bench_prompts/raw/question.jsonl",
    external_detection_kwargs: dict[str, Any] | None = None,
    external_human_shift_kwargs: dict[str, Any] | None = None,
    hf_detector_ids: list[str] | None = None,
    run_unsupervised: bool = False,
) -> dict[str, Any]:
    quick_config = quick_config or QuickRunConfig.from_env()
    training_config = training_config or SupervisedTrainingConfig()

    prompt_manifest = prepare_prompt_dataset(
        config, question_file=question_file, external_detection_kwargs=external_detection_kwargs,
    )

    methods = _resolved_method_names(config)
    generation_paths = build_generation_cache(
        prompt_manifest,
        methods=methods,
        method_profile=config.method_profile,
        quick_config=quick_config,
        force=config.force_regenerate,
        publication_mode=config.publication_mode,
    )

    # Free generation models before loading detector models
    _clear_model_caches()

    trained = train_supervised_detectors(
        prompt_manifest, generation_paths,
        methods=list(generation_paths.keys()),
        training_config=training_config,
        output_root=_artifact_root_from_manifest(prompt_manifest) / "models" / "detectors",
    )

    # Free supervised detector models before loading unsupervised scoring models
    _clear_model_caches()

    # Evaluate: build the transfer matrix (supervised + unsupervised detectors)
    matrix_results = run_detection_matrix(
        prompt_manifest, generation_paths, trained,
        training_config=training_config,
        quick_config=quick_config,
        hf_detector_ids=hf_detector_ids,
        run_unsupervised=run_unsupervised,
    )

    shift_results = None
    if config.use_external_human_shift and external_human_shift_kwargs is not None:
        shift_manifest = prepare_human_shift_dataset(
            output_dir=config.artifacts.datasets / "generative_detection" / "human_shift",
            **external_human_shift_kwargs,
        )
        shift_results = evaluate_human_domain_shift(
            shift_manifest=shift_manifest,
            prompt_manifest=prompt_manifest,
            generation_paths=generation_paths,
            trained_checkpoints=trained,
            training_config=training_config,
        )

    result_dir = _artifact_root_from_manifest(prompt_manifest) / "results" / "generative_detection" / prompt_manifest.name
    write_json(result_dir / "run_config.json", {
        "generative_detection_config": dataclass_to_dict(config),
        "resolved_methods": methods,
        "quick_run_config": dataclass_to_dict(quick_config),
        "supervised_training_config": dataclass_to_dict(training_config),
        "external_detection_kwargs": external_detection_kwargs,
        "external_human_shift_kwargs": external_human_shift_kwargs,
        "hf_detector_ids": list(DEFAULT_HF_DETECTOR_IDS) if hf_detector_ids is None else list(hf_detector_ids),
        "run_unsupervised": run_unsupervised,
    })

    return {
        "prompt_manifest": prompt_manifest,
        "generation_paths": generation_paths,
        "trained_checkpoints": trained,
        "matrix_results": matrix_results,
        "human_shift_results": shift_results,
        "result_dir": result_dir,
    }
