# Core domain types for the benchmark pipeline.
# These dataclasses flow through the pipeline: prompts go in, generated texts
# come out, detectors score them, and results land in MatrixCellResult tables.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

# The two benchmark tracks that share this type system
TrackName = Literal["generative_detection", "task_efficiency"]

# Data records (serialised to JSONL between pipeline stages)

@dataclass(slots=True)
class PromptRecord:
    prompt_id: str
    prompt_text: str
    category: str
    reference_text: str
    split: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationRecord:
    prompt_id: str
    method_name: str
    run_id: int
    seed: int | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# A labelled text sample ready for detector training/evaluation.
# label: 1 = AI-generated, 0 = human-written.
@dataclass(slots=True)
class DetectionExample:
    example_id: str
    prompt_id: str
    source_method: str  # generation method that produced this text (or "human")
    text: str
    label: int
    split: str
    metadata: dict[str, Any] = field(default_factory=dict)


# One cell in the train-method x test-method evaluation matrix.
# Each cell captures how well a detector trained on one generation method
# generalises to text from another method.
@dataclass(slots=True)
class MatrixCellResult:
    train_method: str
    test_method: str
    detector_name: str
    auroc: float | None
    accuracy: float | None
    f1: float | None
    mean_ai_prob: float | None
    threshold: float | None = None
    fpr: float | None = None
    tpr: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


#Task-efficiency track records

@dataclass(slots=True)
class TaskExample:
    example_id: str
    dataset: str
    split: str
    label: int
    text_a: str
    text_b: str | None = None
    label_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Raw per-example outputs from running a method on a dataset.
# exit_layers tracks early-exit behaviour (None if the method runs all layers).
@dataclass(slots=True)
class MethodRunResult:
    method_name: str
    texts: list[str] = field(default_factory=list)
    predictions: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    exit_layers: list[int | None] = field(default_factory=list)
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskRunResult:
    dataset: str
    method_name: str
    metric_dict: dict[str, float]
    avg_exit_layer: float | None
    latency_ms: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


# Registry specs

@dataclass(slots=True)
class DatasetManifest:
    name: str
    track: TrackName
    external: bool
    root_dir: Path
    records_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainedDetectorRef:
    name: str
    family: str
    checkpoint_dir: Path
    label_map: dict[int, str]
    metadata: dict[str, Any] = field(default_factory=dict)


# A pluggable generation/inference method
@dataclass(slots=True)
class MethodSpec:
    name: str
    track: TrackName | list[TrackName]
    family: str
    supports_answer_voting: bool
    supports_dataset: Callable[[str], bool]
    run: Callable[..., MethodRunResult]
    metadata: dict[str, Any] = field(default_factory=dict)



@dataclass(slots=True)
class DetectorSpec:
    name: str
    family: Literal["hf_pipeline", "supervised", "binoculars", "fastdetectgpt"]
    requires_training: bool
    train: Callable[..., TrainedDetectorRef] | None
    score_texts: Callable[..., list[float]]
    predict_texts: Callable[..., list[int] | None] | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetSpec:
    name: str
    track: TrackName
    external: bool
    prepare: Callable[..., DatasetManifest]
    metadata: dict[str, Any] = field(default_factory=dict)
