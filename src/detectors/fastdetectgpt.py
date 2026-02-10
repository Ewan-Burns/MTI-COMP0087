"""
FastDetectGPT detector implementation.
Based on Bao et al. (2024) "Fast-DetectGPT: Efficient Zero-shot Detection
of Machine-Generated Text via Conditional Probability Curvature".

Uses conditional probability curvature to detect AI-generated text.
"""

import torch
import numpy as np
from typing import List, Optional
from tqdm import tqdm

from .base import BaseDetector, DetectionResult
from ..models import get_model_manager


class FastDetectGPTDetector(BaseDetector):
    """
    FastDetectGPT detector for AI-generated text detection.

    Uses conditional probability curvature (difference between
    log probability and sampled cross-entropy) normalized by std.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-3B",
        reference_model_id: Optional[str] = None,
        threshold: float = 0.0,
        max_length: int = 512,
        num_samples: int = 10,
    ):
        """
        Initialize FastDetectGPT detector.

        Args:
            model_id: Model for scoring
            reference_model_id: Reference model for sampling (uses same if None)
            threshold: Classification threshold
            max_length: Maximum sequence length
            num_samples: Number of Monte Carlo samples
        """
        super().__init__("fastdetectgpt", threshold)
        self.model_id = model_id
        self.reference_model_id = reference_model_id or model_id
        self.max_length = max_length
        self.num_samples = num_samples

        # Load models
        self.model_manager = get_model_manager()
        self.model, self.tokenizer = self.model_manager.load_detector_model(
            model_id, model_type="causal"
        )

        # Load reference model if different
        if self.reference_model_id != model_id:
            self.reference_model, _ = self.model_manager.load_detector_model(
                self.reference_model_id, model_type="causal"
            )
        else:
            self.reference_model = self.model

        self.model.eval()
        self.reference_model.eval()

    def _get_logprobs(self, text: str) -> torch.Tensor:
        """Get log probabilities for each token."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logprobs = torch.log_softmax(outputs.logits, dim=-1)

            # Get log prob of actual tokens (shifted by 1)
            token_logprobs = logprobs[:, :-1, :].gather(
                2, inputs["input_ids"][:, 1:].unsqueeze(-1)
            ).squeeze(-1)

        return token_logprobs

    def _sample_and_compute_ce(self, text: str) -> tuple:
        """Sample from reference model and compute cross-entropy."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        sampled_ces = []

        with torch.no_grad():
            # Get reference model distribution
            ref_outputs = self.reference_model(**inputs)
            ref_probs = torch.softmax(ref_outputs.logits[:, :-1, :], dim=-1)

            # Get scoring model logprobs
            score_outputs = self.model(**inputs)
            score_logprobs = torch.log_softmax(score_outputs.logits[:, :-1, :], dim=-1)

            # Monte Carlo sampling
            for _ in range(self.num_samples):
                # Sample from reference distribution
                sampled_tokens = torch.multinomial(
                    ref_probs.view(-1, ref_probs.size(-1)), 1
                ).view(ref_probs.size(0), ref_probs.size(1))

                # Get log prob of sampled tokens under scoring model
                sampled_logprobs = score_logprobs.gather(2, sampled_tokens.unsqueeze(-1)).squeeze(-1)
                sampled_ces.append(sampled_logprobs.mean().item())

        return np.mean(sampled_ces), np.std(sampled_ces)

    def _compute_fastdetect_score(self, text: str) -> float:
        """
        Compute FastDetectGPT score.

        Score = (mean_logprob - mean_sampled_ce) / std_sampled_ce
        Positive scores indicate AI, negative indicate human.
        """
        # Get log probs of actual text
        token_logprobs = self._get_logprobs(text)
        mean_logprob = token_logprobs.mean().item()

        # Get sampled cross-entropy
        mean_ce, std_ce = self._sample_and_compute_ce(text)

        # Avoid division by zero
        if std_ce < 1e-10:
            std_ce = 1e-10

        # Curvature score
        score = (mean_logprob - mean_ce) / std_ce
        return score

    def detect(self, text: str) -> DetectionResult:
        """Detect if text is AI-generated."""
        score = self._compute_fastdetect_score(text)

        # Transform score to [0, 1] using sigmoid
        normalized_score = 1 / (1 + np.exp(-score))

        prediction, confidence = self._score_to_prediction(normalized_score)

        return DetectionResult(
            text=text,
            score=normalized_score,
            prediction=prediction,
            confidence=confidence,
            detector_name=self.name,
            raw_output={"fastdetect_score": score},
        )

    def detect_batch(self, texts: List[str], show_progress: bool = True) -> List[DetectionResult]:
        """Detect if multiple texts are AI-generated."""
        results = []
        iterator = tqdm(texts, desc="FastDetectGPT") if show_progress else texts

        for text in iterator:
            results.append(self.detect(text))

        return results
