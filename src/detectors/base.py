"""
Base class for all detectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np


@dataclass
class DetectionResult:
    """Result from AI text detection."""
    text: str
    score: float  # Higher = more likely AI-generated (typically)
    prediction: str  # "human" or "ai"
    confidence: float  # Confidence in the prediction
    detector_name: str
    raw_output: Optional[dict] = None  # Any additional outputs


class BaseDetector(ABC):
    """
    Abstract base class for AI text detectors.

    All detectors should inherit from this class and implement
    the detect() and detect_batch() methods.
    """

    def __init__(self, name: str, threshold: float = 0.5):
        """
        Initialize the detector.

        Args:
            name: Name of the detector
            threshold: Classification threshold (above = AI, below = human)
        """
        self.name = name
        self.threshold = threshold

    @abstractmethod
    def detect(self, text: str) -> DetectionResult:
        """
        Detect if a single text is AI-generated.

        Args:
            text: Input text to classify

        Returns:
            DetectionResult with score, prediction, and confidence
        """
        pass

    @abstractmethod
    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        """
        Detect if multiple texts are AI-generated.

        Args:
            texts: List of texts to classify

        Returns:
            List of DetectionResult objects
        """
        pass

    def _score_to_prediction(self, score: float) -> tuple:
        """
        Convert a score to a prediction and confidence.

        Args:
            score: Detection score (higher = more likely AI)

        Returns:
            Tuple of (prediction, confidence)
        """
        if score >= self.threshold:
            prediction = "ai"
            confidence = score
        else:
            prediction = "human"
            confidence = 1 - score

        return prediction, confidence

    def evaluate(
        self,
        texts: List[str],
        labels: List[str],  # "human" or "ai"
    ) -> dict:
        """
        Evaluate detector performance on labeled data.

        Args:
            texts: List of texts
            labels: Ground truth labels

        Returns:
            Dictionary with evaluation metrics
        """
        results = self.detect_batch(texts)

        predictions = [r.prediction for r in results]
        scores = [r.score for r in results]

        # Calculate metrics
        correct = sum(p == l for p, l in zip(predictions, labels))
        accuracy = correct / len(labels)

        # Calculate AUROC if sklearn is available
        try:
            from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

            # Convert labels to binary (1 = AI, 0 = human)
            binary_labels = [1 if l == "ai" else 0 for l in labels]
            binary_preds = [1 if p == "ai" else 0 for p in predictions]

            auroc = roc_auc_score(binary_labels, scores)
            f1 = f1_score(binary_labels, binary_preds)
            precision = precision_score(binary_labels, binary_preds)
            recall = recall_score(binary_labels, binary_preds)
        except ImportError:
            auroc = f1 = precision = recall = None

        return {
            "accuracy": accuracy,
            "auroc": auroc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "num_samples": len(labels),
        }
