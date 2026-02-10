"""
Configuration management for the MTI project.
Handles model configs, sampling parameters, and experiment settings.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for a language model."""
    name: str
    hf_model_id: str
    model_type: str  # "generator" or "detector"
    max_length: int = 512
    device_map: str = "auto"
    torch_dtype: str = "float16"  # "float16", "bfloat16", "float32"
    load_in_8bit: bool = False
    load_in_4bit: bool = False


@dataclass
class SamplingConfig:
    """Configuration for a sampling/decoding strategy."""
    name: str
    method: str  # "greedy", "top_k", "top_p", "temperature", "repetition_penalty", "eta"
    params: Dict[str, Any] = field(default_factory=dict)

    def to_generate_kwargs(self) -> Dict[str, Any]:
        """Convert to HuggingFace generate() kwargs."""
        kwargs = {
            "max_new_tokens": self.params.get("max_new_tokens", 256),
            "do_sample": self.method != "greedy",
        }

        if self.method == "greedy":
            kwargs["do_sample"] = False
        elif self.method == "top_k":
            kwargs["top_k"] = self.params.get("top_k", 50)
        elif self.method == "top_p":
            kwargs["top_p"] = self.params.get("top_p", 0.95)
        elif self.method == "temperature":
            kwargs["temperature"] = self.params.get("temperature", 1.0)
        elif self.method == "repetition_penalty":
            kwargs["repetition_penalty"] = self.params.get("repetition_penalty", 1.0)
        elif self.method == "eta":
            kwargs["eta_cutoff"] = self.params.get("eta", 1e-4)
        elif self.method == "typical":
            kwargs["typical_p"] = self.params.get("typical_p", 0.95)

        return kwargs


@dataclass
class DetectorConfig:
    """Configuration for a detector."""
    name: str
    detector_type: str  # "binoculars", "fastdetectgpt", "radar", "roberta"
    model_id: Optional[str] = None
    observer_model_id: Optional[str] = None  # For Binoculars
    threshold: float = 0.5


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str
    generator_model: ModelConfig
    sampling_configs: List[SamplingConfig]
    detector_configs: List[DetectorConfig]
    dataset_name: str = "raid"
    num_samples: int = 1000
    output_dir: str = "outputs"
    seed: int = 42


# =============================================================================
# Pre-defined configurations based on Dubois et al. (2025) and project notes
# =============================================================================

# Generator Models
GENERATOR_MODELS = {
    "llama-3.2-3b": ModelConfig(
        name="llama-3.2-3b",
        hf_model_id="meta-llama/Llama-3.2-3B-Instruct",
        model_type="generator",
        max_length=512,
    ),
    "mistral-7b": ModelConfig(
        name="mistral-7b",
        hf_model_id="mistralai/Mistral-7B-v0.1",
        model_type="generator",
        max_length=512,
    ),
    "qwen2-7b": ModelConfig(
        name="qwen2-7b",
        hf_model_id="Qwen/Qwen2-7B",
        model_type="generator",
        max_length=512,
    ),
    "deepseek-7b": ModelConfig(
        name="deepseek-7b",
        hf_model_id="deepseek-ai/deepseek-llm-7b-base",
        model_type="generator",
        max_length=512,
    ),
}

# Sampling configurations from Dubois et al. (2025) Table 1
SAMPLING_CONFIGS = {
    # Greedy
    "greedy": SamplingConfig(name="greedy", method="greedy", params={"max_new_tokens": 256}),

    # Temperature variations
    "temp_0.5": SamplingConfig(name="temp_0.5", method="temperature", params={"temperature": 0.5}),
    "temp_0.7": SamplingConfig(name="temp_0.7", method="temperature", params={"temperature": 0.7}),
    "temp_0.9": SamplingConfig(name="temp_0.9", method="temperature", params={"temperature": 0.9}),
    "temp_1.0": SamplingConfig(name="temp_1.0", method="temperature", params={"temperature": 1.0}),
    "temp_1.1": SamplingConfig(name="temp_1.1", method="temperature", params={"temperature": 1.1}),
    "temp_1.2": SamplingConfig(name="temp_1.2", method="temperature", params={"temperature": 1.2}),

    # Repetition penalty variations
    "rep_1.05": SamplingConfig(name="rep_1.05", method="repetition_penalty", params={"repetition_penalty": 1.05}),
    "rep_1.10": SamplingConfig(name="rep_1.10", method="repetition_penalty", params={"repetition_penalty": 1.10}),
    "rep_1.15": SamplingConfig(name="rep_1.15", method="repetition_penalty", params={"repetition_penalty": 1.15}),
    "rep_1.20": SamplingConfig(name="rep_1.20", method="repetition_penalty", params={"repetition_penalty": 1.20}),
    "rep_1.25": SamplingConfig(name="rep_1.25", method="repetition_penalty", params={"repetition_penalty": 1.25}),
    "rep_1.30": SamplingConfig(name="rep_1.30", method="repetition_penalty", params={"repetition_penalty": 1.30}),

    # Top-k variations
    "topk_10": SamplingConfig(name="topk_10", method="top_k", params={"top_k": 10}),
    "topk_20": SamplingConfig(name="topk_20", method="top_k", params={"top_k": 20}),
    "topk_50": SamplingConfig(name="topk_50", method="top_k", params={"top_k": 50}),
    "topk_75": SamplingConfig(name="topk_75", method="top_k", params={"top_k": 75}),
    "topk_100": SamplingConfig(name="topk_100", method="top_k", params={"top_k": 100}),
    "topk_1000": SamplingConfig(name="topk_1000", method="top_k", params={"top_k": 1000}),

    # Top-p (nucleus) variations
    "topp_0.3": SamplingConfig(name="topp_0.3", method="top_p", params={"top_p": 0.3}),
    "topp_0.5": SamplingConfig(name="topp_0.5", method="top_p", params={"top_p": 0.5}),
    "topp_0.7": SamplingConfig(name="topp_0.7", method="top_p", params={"top_p": 0.7}),
    "topp_0.8": SamplingConfig(name="topp_0.8", method="top_p", params={"top_p": 0.8}),
    "topp_0.9": SamplingConfig(name="topp_0.9", method="top_p", params={"top_p": 0.9}),
    "topp_0.95": SamplingConfig(name="topp_0.95", method="top_p", params={"top_p": 0.95}),

    # Typical sampling variations
    "typical_0.3": SamplingConfig(name="typical_0.3", method="typical", params={"typical_p": 0.3}),
    "typical_0.5": SamplingConfig(name="typical_0.5", method="typical", params={"typical_p": 0.5}),
    "typical_0.7": SamplingConfig(name="typical_0.7", method="typical", params={"typical_p": 0.7}),
    "typical_0.8": SamplingConfig(name="typical_0.8", method="typical", params={"typical_p": 0.8}),
    "typical_0.9": SamplingConfig(name="typical_0.9", method="typical", params={"typical_p": 0.9}),
    "typical_0.95": SamplingConfig(name="typical_0.95", method="typical", params={"typical_p": 0.95}),

    # Eta sampling variations
    "eta_1e-4": SamplingConfig(name="eta_1e-4", method="eta", params={"eta": 1e-4}),
    "eta_1e-3": SamplingConfig(name="eta_1e-3", method="eta", params={"eta": 1e-3}),
    "eta_5e-3": SamplingConfig(name="eta_5e-3", method="eta", params={"eta": 5e-3}),
    "eta_0.01": SamplingConfig(name="eta_0.01", method="eta", params={"eta": 0.01}),
    "eta_0.05": SamplingConfig(name="eta_0.05", method="eta", params={"eta": 0.05}),
    "eta_0.1": SamplingConfig(name="eta_0.1", method="eta", params={"eta": 0.1}),
}

# Detector configurations
DETECTOR_CONFIGS = {
    "binoculars": DetectorConfig(
        name="binoculars",
        detector_type="binoculars",
        model_id="meta-llama/Llama-3.2-3B",
        observer_model_id="meta-llama/Llama-3.2-3B-Instruct",
    ),
    "fastdetectgpt": DetectorConfig(
        name="fastdetectgpt",
        detector_type="fastdetectgpt",
        model_id="meta-llama/Llama-3.2-3B",
    ),
    "radar": DetectorConfig(
        name="radar",
        detector_type="radar",
        model_id="TrustSafeAI/RADAR-Vicuna-7B",
    ),
    "roberta": DetectorConfig(
        name="roberta",
        detector_type="roberta",
        model_id="roberta-base",
    ),
}


def get_hf_token() -> str:
    """Get HuggingFace token from environment."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in environment. Check your .env file.")
    return token


def get_sampling_subset(subset: str = "dubois_core") -> List[SamplingConfig]:
    """Get a predefined subset of sampling configurations.

    Subsets:
        - "dubois_core": Core settings closest to human diversity (green in Table 2)
        - "dubois_all": All 37 configurations from Dubois et al.
        - "quick_test": Small subset for quick testing
    """
    if subset == "dubois_core":
        # Settings closest to human diversity from Dubois Table 2
        return [
            SAMPLING_CONFIGS["greedy"],
            SAMPLING_CONFIGS["temp_0.9"],
            SAMPLING_CONFIGS["topk_100"],
            SAMPLING_CONFIGS["topp_0.95"],
            SAMPLING_CONFIGS["typical_0.95"],
            SAMPLING_CONFIGS["eta_1e-4"],
        ]
    elif subset == "dubois_all":
        return list(SAMPLING_CONFIGS.values())
    elif subset == "quick_test":
        return [
            SAMPLING_CONFIGS["greedy"],
            SAMPLING_CONFIGS["temp_0.7"],
            SAMPLING_CONFIGS["topp_0.9"],
        ]
    else:
        raise ValueError(f"Unknown subset: {subset}")
