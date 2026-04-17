# Configuration dataclasses for the statnlp benchmark pipeline.
# Every setting can be overridden via environment variables 


from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Environment-variable helpers
# Each returns the parsed env value if set, otherwise the provided default.

def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw.strip())


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw.strip())


def env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip()


def env_str_or_none(name: str) -> str | None:
    raw = os.environ.get(name, "").strip()
    return raw or None


def env_list(name: str, default: list[str], sep: str = ",") -> list[str]:
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    return [item.strip() for item in raw.split(sep) if item.strip()]


def env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


# Try all known HF token env vars; the ecosystem has used several names over time.
def hf_token_from_env() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )


# Artifact paths 
# Standardised directory layout for all pipeline outputs (datasets, generated
# text, evaluation results, plots). Everything lives under a single root.

@dataclass(slots=True)
class ArtifactPaths:
    root: Path
    datasets: Path
    generations: Path
    corpora: Path
    results: Path
    plots: Path


def build_artifact_paths(root: str | Path = "artifacts") -> ArtifactPaths:
    r = Path(root).expanduser().resolve()
    return ArtifactPaths(
        root=r,
        datasets=r / "datasets",
        generations=r / "generations",
        corpora=r / "corpora",
        results=r / "results",
        plots=r / "plots",
    )


#Model settings

@dataclass(slots=True)
class ModelSettings:
    hf_token: str | None = None
    # Generation model
    publication_model_id: str = "Qwen/Qwen2.5-3B"
    publication_model_revision: str | None = None
    publication_device: str = "auto"
    publication_torch_dtype: str = "float32"
    publication_attn_implementation: str | None = None
    publication_negative_prompt: str = ""


# Quick-run config 
# End-to-end settings for a single generation+detection run (model, decoding,
# detector params, human baseline). All fields overridable via from_env().

@dataclass(slots=True)
class QuickRunConfig:
    model: ModelSettings = field(default_factory=ModelSettings)
    max_tokens: int = 512
    generation_batch_size: int = 384
    sample_seed: int = 42
    prompt_seed_stride: int = 1000  # offset between per-prompt RNG seeds for reproducibility
    beam_width: int = 4
    beam_length_penalty: float = 0.6
    ai_detector_cache_dir: Path = Path("./models/detectors")
    ai_detector_max_chars: int = 2500  # truncate texts before feeding to detector
    ai_detector_device: str = "auto"
    ai_detector_batch_size: int = 8
    print_generated_text: bool = False
    enable_human_baseline: bool = True
    human_baseline_texts: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "QuickRunConfig":
        baseline_default = (
            "I am a GP, and I have a clinic in the morning, "
            "what's my first patient most likely to be presenting with?"
        )
        return cls(
            model=ModelSettings(
                hf_token=hf_token_from_env(),
                publication_model_id=env_str(
                    "PUBLICATION_MODEL_ID", "Qwen/Qwen2.5-3B"
                ),
                publication_model_revision=env_str_or_none("PUBLICATION_MODEL_REVISION"),
                publication_device=env_str("PUBLICATION_DEVICE", "auto").lower() or "auto",
                publication_torch_dtype=env_str("PUBLICATION_TORCH_DTYPE", "float32").lower() or "float32",
                publication_attn_implementation=env_str_or_none("PUBLICATION_ATTN_IMPLEMENTATION"),
                publication_negative_prompt=env_str("PUBLICATION_NEGATIVE_PROMPT", ""),
            ),
            max_tokens=env_int("MAX_TOKENS", 512),
            generation_batch_size=env_int("GENERATION_BATCH_SIZE", 384),
            sample_seed=env_int("SAMPLE_SEED", 42),
            prompt_seed_stride=env_int("PROMPT_SEED_STRIDE", 1000),
            beam_width=env_int("BEAM_WIDTH", 4),
            beam_length_penalty=env_float("BEAM_LENGTH_PENALTY", 0.6),
            ai_detector_cache_dir=env_path("AI_DETECTOR_CACHE_DIR", "./models/detectors"),
            ai_detector_max_chars=env_int("AI_DETECTOR_MAX_CHARS", 2500),
            ai_detector_device=env_str("AI_DETECTOR_DEVICE", "auto").lower() or "auto",
            ai_detector_batch_size=env_int("AI_DETECTOR_BATCH_SIZE", 8),
            print_generated_text=env_bool("PRINT_GENERATED_TEXT", False),
            enable_human_baseline=env_bool("ENABLE_HUMAN_BASELINE", True),
            # Baseline texts separated by "||" so individual texts can contain commas
            human_baseline_texts=env_list("HUMAN_BASELINE_TEXTS", [baseline_default], sep="||"),
        )


# Generative-detection track config
# Controls the full generate-then-detect pipeline: which prompts to use,
# how to split data, which generation methods to evaluate, and at what FPR
# threshold to report detection results.

@dataclass(slots=True)
class GenerativeDetectionConfig:
    artifacts: ArtifactPaths = field(default_factory=build_artifact_paths)
    dataset_name: str = "mt_bench_local"
    split_seed: int = 47
    train_ratio: float = 0.5
    val_ratio: float = 0.1
    test_ratio: float = 0.4
    max_prompts: int | None = None  # subsample prompts for faster dev iterations
    balance_categories: bool = True
    methods: list[str] = field(default_factory=list)
    method_profile: str | None = None  # named preset selecting a group of methods
    sample_seed: int = 42
    sampling_runs_per_prompt: int = 1
    use_external_detection_data: bool = False
    include_local_mt_bench: bool = True
    use_external_human_shift: bool = False  # test with human text from a different domain
    force_regenerate: bool = False
    target_fpr: float = 0.05  # false-positive rate used to set detection threshold
    split_profile: str = "default"
    publication_mode: bool = False  # stricter settings for paper-reproducible runs


# Supervised detector training config
# Hyperparameters for fine-tuning classifier heads (e.g. RoBERTa, mDeBERTa)
# on top of the generated/human text to build supervised AI-text detectors.

@dataclass(slots=True)
class SupervisedTrainingConfig:
    epochs: int = 3
    learning_rate: float = 5e-5
    max_length: int = 512
    train_batch_size: int = 8
    eval_batch_size: int = 64
    weight_decay: float = 0.0
    seed: int = 26
    device: str = "auto"
    force_float32: bool = True  # ensures stable training on CPU fallback
    calibrate_threshold: bool = True  # tune decision threshold on val set to hit target_fpr
    force_retrain: bool = False
    target_fpr: float = 0.05
    internal_validation_ratio: float = 0.1  # held-out slice for threshold calibration
    # Maps short names to HF model IDs for each detector architecture
    architecture_model_ids: dict[str, str] = field(
        default_factory=lambda: {
            "roberta-base": "roberta-base",
            "mdeberta-v3-base": "microsoft/mdeberta-v3-base",
        }
    )




# Recursively serialise nested dataclasses to plain dicts (for JSON export).
def dataclass_to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: dataclass_to_dict(getattr(value, key))
            for key in value.__dataclass_fields__
        }
    if isinstance(value, dict):
        return {key: dataclass_to_dict(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
