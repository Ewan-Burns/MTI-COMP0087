# Central plugin registry for generation methods, AI-text detectors, and datasets.
# Modules register their specs at import time; the pipeline looks them up by name
# at run time. This decouples the pipeline orchestrator from individual implementations.

from __future__ import annotations

from .types import DatasetSpec, DetectorSpec, MethodSpec

# Global registries — populated by register_* calls during module import
METHOD_REGISTRY: dict[str, MethodSpec] = {}
DETECTOR_REGISTRY: dict[str, DetectorSpec] = {}
DATASET_REGISTRY: dict[str, DatasetSpec] = {}


def register_method(spec: MethodSpec) -> MethodSpec:
    METHOD_REGISTRY[spec.name] = spec
    return spec


def register_detector(spec: DetectorSpec) -> DetectorSpec:
    DETECTOR_REGISTRY[spec.name] = spec
    return spec


def register_dataset(spec: DatasetSpec) -> DatasetSpec:
    DATASET_REGISTRY[spec.name] = spec
    return spec


def get_method(name: str) -> MethodSpec:
    return METHOD_REGISTRY[name]


def get_detector(name: str) -> DetectorSpec:
    return DETECTOR_REGISTRY[name]


def get_dataset(name: str) -> DatasetSpec:
    return DATASET_REGISTRY[name]
