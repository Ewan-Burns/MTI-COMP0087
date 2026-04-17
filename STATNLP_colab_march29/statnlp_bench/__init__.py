"""STATNLP benchmark package."""

from .config import (
    ArtifactPaths,
    GenerativeDetectionConfig,
    QuickRunConfig,
    SupervisedTrainingConfig,
    TaskTrackConfig,
    build_artifact_paths,
)
from .hf_cache import configure_hf_environment
from .registry import DATASET_REGISTRY, DETECTOR_REGISTRY, METHOD_REGISTRY

configure_hf_environment()

from .datasets import human_shift as _human_shift  # noqa: F401
from .datasets import mt_bench as _mt_bench  # noqa: F401
from .datasets import nlu as _nlu  # noqa: F401
from .datasets import raid_like as _raid_like  # noqa: F401
from .detectors import binoculars as _binoculars  # noqa: F401
from .detectors import fastdetectgpt as _fastdetectgpt  # noqa: F401
from .detectors import hf_pipeline as _hf_pipeline  # noqa: F401
from .detectors import supervised as _supervised  # noqa: F401
from .methods import publication as _publication  # noqa: F401


__all__ = [
    "ArtifactPaths",
    "DATASET_REGISTRY",
    "DETECTOR_REGISTRY",
    "GenerativeDetectionConfig",
    "METHOD_REGISTRY",
    "QuickRunConfig",
    "SupervisedTrainingConfig",
    "TaskTrackConfig",
    "build_artifact_paths",
]
