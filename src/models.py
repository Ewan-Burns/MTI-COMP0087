"""
Model loading and management for the MTI project.
Handles loading of generator and detector models with proper authentication.
"""

import torch
from typing import Dict, Optional, Tuple, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from .config import ModelConfig, get_hf_token


class ModelManager:
    """Manages loading and caching of language models."""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self.hf_token = get_hf_token()

    def load_generator(
        self,
        config: ModelConfig,
        force_reload: bool = False
    ) -> Tuple[Any, Any]:
        """Load a generator model and tokenizer.

        Args:
            config: Model configuration
            force_reload: Force reload even if cached

        Returns:
            Tuple of (model, tokenizer)
        """
        cache_key = config.name

        if cache_key in self._models and not force_reload:
            return self._models[cache_key], self._tokenizers[cache_key]

        print(f"Loading generator model: {config.hf_model_id}")

        # Configure quantization if requested
        quantization_config = None
        if config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.hf_model_id,
            token=self.hf_token,
            trust_remote_code=True,
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_model_id,
            token=self.hf_token,
            device_map=config.device_map,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        # Cache
        self._models[cache_key] = model
        self._tokenizers[cache_key] = tokenizer

        print(f"Model loaded successfully: {config.name}")
        return model, tokenizer

    def load_detector_model(
        self,
        model_id: str,
        model_type: str = "causal",
        force_reload: bool = False
    ) -> Tuple[Any, Any]:
        """Load a model for detection purposes.

        Args:
            model_id: HuggingFace model ID
            model_type: "causal" for LM-based detectors, "classifier" for RoBERTa-style
            force_reload: Force reload even if cached

        Returns:
            Tuple of (model, tokenizer)
        """
        cache_key = f"detector_{model_id}"

        if cache_key in self._models and not force_reload:
            return self._models[cache_key], self._tokenizers[cache_key]

        print(f"Loading detector model: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=self.hf_token,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if model_type == "causal":
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=self.hf_token,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        elif model_type == "classifier":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                token=self.hf_token,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self._models[cache_key] = model
        self._tokenizers[cache_key] = tokenizer

        print(f"Detector model loaded: {model_id}")
        return model, tokenizer

    def clear_cache(self):
        """Clear all cached models to free memory."""
        self._models.clear()
        self._tokenizers.clear()
        torch.cuda.empty_cache()


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create the global model manager."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
