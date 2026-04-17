# ==========================================================================
# Detection data loading utilities.
#
# Two record types flow through the detection pipeline:
#   - PromptRecord: a prompt + human reference text + train/val/test split
#     (stored as prompts.jsonl under each dataset directory)
#   - GenerationRecord: an AI-generated text for a given prompt + method
#     (stored as {method_name}.jsonl under the generations directory)
#
# Both are LRU-cached (keyed by resolved file path) to avoid redundant I/O
# when the same dataset is accessed across multiple pipeline stages.
# ==========================================================================

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..config import build_artifact_paths
from ..results import read_jsonl
from ..types import DatasetManifest, GenerationRecord, PromptRecord


def load_prompt_records(manifest: DatasetManifest) -> list[PromptRecord]:
    return list(_load_prompt_records_cached(str(manifest.records_path.expanduser().resolve())))


@lru_cache(maxsize=32)
def _load_prompt_records_cached(records_path: str) -> tuple[PromptRecord, ...]:
    return tuple(
        PromptRecord(
            prompt_id=str(row["prompt_id"]),
            prompt_text=str(row["prompt_text"]),
            category=str(row["category"]),
            reference_text=str(row.get("reference_text", "")),
            split=str(row["split"]),
            metadata=row.get("metadata", {}),
        )
        for row in read_jsonl(records_path)
    )


def load_generation_records(path: str | Path) -> list[GenerationRecord]:
    return list(_load_generation_records_cached(str(Path(path).expanduser().resolve())))


@lru_cache(maxsize=64)
def _load_generation_records_cached(path: str) -> tuple[GenerationRecord, ...]:
    return tuple(
        GenerationRecord(
            prompt_id=str(row["prompt_id"]),
            method_name=str(row["method_name"]),
            run_id=int(row["run_id"]),
            seed=row.get("seed"),
            text=str(row["text"]),
            metadata=row.get("metadata", {}),
        )
        for row in read_jsonl(path)
    )


# Walk up from the manifest's root_dir to find the top-level artifact directory.
# Convention: datasets live at {artifact_root}/datasets/{track}/{name}, so parents[2].
def _artifact_root_from_manifest(manifest: DatasetManifest) -> Path:
    metadata_root = manifest.metadata.get("artifact_root")
    if metadata_root:
        return Path(str(metadata_root)).expanduser().resolve()
    try:
        return manifest.root_dir.parents[2]
    except IndexError:
        return build_artifact_paths().root
