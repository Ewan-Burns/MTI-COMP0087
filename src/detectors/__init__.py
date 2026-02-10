"""
AI-generated text detectors.
Implements Binoculars, FastDetectGPT, RADAR, and RoBERTa-based detectors.
"""

from .base import BaseDetector, DetectionResult
from .binoculars import BinocularsDetector
from .fastdetectgpt import FastDetectGPTDetector
from .radar import RADARDetector, RADARWithParaphraser
from .roberta import RoBERTaDetector

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "BinocularsDetector",
    "FastDetectGPTDetector",
    "RADARDetector",
    "RADARWithParaphraser",
    "RoBERTaDetector",
]
