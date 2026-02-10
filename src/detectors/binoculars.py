"""
Binoculars detector implementation.
Based on Hans et al. (2024) "Spotting LLMs with Binoculars".

The Binoculars score is defined as the ratio of:
- Perplexity of text under the main model
- Cross-entropy between main model and observer model
"""

import torch
import numpy as np
from typing import List, Optional
from tqdm import tqdm

from .base import BaseDetector, DetectionResult
from ..models import get_model_manager


class BinocularsDetector(BaseDetector):
    """
    Binoculars detector for AI-generated text detection.

    Uses two models (main and observer) to compute a detection score
    based on the ratio of perplexity to cross-entropy.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-3B",
        observer_model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        threshold: float = 0.9,
        max_length: int = 512,
    ):
        """
        Initialize Binoculars detector.

        Args:
            model_id: Main model for perplexity computation
            observer_model_id: Observer model for cross-entropy
            threshold: Classification threshold
            max_length: Maximum sequence length
        """
        super().__init__("binoculars", threshold)
        self.model_id = model_id
        self.observer_model_id = observer_model_id
        self.max_length = max_length

        # Load models
        self.model_manager = get_model_manager()
        self.model, self.tokenizer = self.model_manager.load_detector_model(
            model_id, model_type="causal"
        )
        self.observer_model, _ = self.model_manager.load_detector_model(
            observer_model_id, model_type="causal"
        )

        self.model.eval()
        self.observer_model.eval()

    def _compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text under the main model."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()

        return np.exp(loss)

    def _compute_cross_entropy(self, text: str) -> float:
        """Compute cross-entropy between main and observer model."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get logits from both models
            main_outputs = self.model(**inputs)
            observer_outputs = self.observer_model(**inputs.copy())

            main_logprobs = torch.log_softmax(main_outputs.logits, dim=-1)
            observer_probs = torch.softmax(observer_outputs.logits, dim=-1)

            # Compute cross-entropy: -sum(p_observer * log(p_main))
            cross_entropy = -torch.sum(observer_probs * main_logprobs, dim=-1)
            cross_entropy = cross_entropy.mean().item()

        return cross_entropy

    def _compute_binoculars_score(self, text: str) -> float:
        """
        Compute the Binoculars score.

        Score = perplexity / cross_entropy
        Higher scores indicate human text, lower scores indicate AI.
        """
        perplexity = self._compute_perplexity(text)
        cross_entropy = self._compute_cross_entropy(text)

        # Avoid division by zero
        if cross_entropy < 1e-10:
            cross_entropy = 1e-10

        score = perplexity / cross_entropy
        return score

    def detect(self, text: str) -> DetectionResult:
        """Detect if text is AI-generated."""
        score = self._compute_binoculars_score(text)

        # Note: For Binoculars, LOWER scores indicate AI text
        # We invert for consistency with other detectors
        inverted_score = 1 / (1 + score)  # Transform to [0, 1]

        prediction, confidence = self._score_to_prediction(inverted_score)

        return DetectionResult(
            text=text,
            score=inverted_score,
            prediction=prediction,
            confidence=confidence,
            detector_name=self.name,
            raw_output={"binoculars_score": score},
        )

    def detect_batch(self, texts: List[str], show_progress: bool = True) -> List[DetectionResult]:
        """Detect if multiple texts are AI-generated."""
        results = []
        iterator = tqdm(texts, desc="Binoculars") if show_progress else texts

        for text in iterator:
            results.append(self.detect(text))

        return results
