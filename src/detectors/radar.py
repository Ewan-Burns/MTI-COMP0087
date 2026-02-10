"""
RADAR detector implementation.
Based on Hu et al. (2023) "RADAR: Robust AI-Text Detection via Adversarial Learning".

Uses a RoBERTa-based classifier trained with adversarial paraphrasing
to improve robustness against text modifications.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import BaseDetector, DetectionResult
from ..config import get_hf_token


class RADARDetector(BaseDetector):
    """
    RADAR (Robust AI-Text Detection via Adversarial Learning) detector.

    A semi-supervised detector that combines:
    - RoBERTa-based text classification
    - Adversarial training with paraphrased examples
    - Robust feature learning

    Pre-trained models available on HuggingFace:
    - TrustSafeAI/RADAR-Vicuna-7B
    """

    # Known pre-trained RADAR models
    PRETRAINED_MODELS = {
        "radar-vicuna": "TrustSafeAI/RADAR-Vicuna-7B",
    }

    def __init__(
        self,
        model_id: str = "TrustSafeAI/RADAR-Vicuna-7B",
        threshold: float = 0.5,
        max_length: int = 512,
    ):
        """
        Initialize RADAR detector.

        Args:
            model_id: HuggingFace model ID for RADAR checkpoint
            threshold: Classification threshold
            max_length: Maximum sequence length
        """
        super().__init__("radar", threshold)
        self.model_id = model_id
        self.max_length = max_length
        self.hf_token = get_hf_token()

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=self.hf_token
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, token=self.hf_token
        )

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def detect(self, text: str) -> DetectionResult:
        """Detect if text is AI-generated using RADAR."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

            # RADAR typically uses: 0 = human, 1 = AI
            # Check model config for label mapping
            if hasattr(self.model.config, 'id2label'):
                # Find AI label index
                ai_idx = None
                for idx, label in self.model.config.id2label.items():
                    if 'ai' in label.lower() or 'machine' in label.lower() or 'generated' in label.lower():
                        ai_idx = int(idx)
                        break
                if ai_idx is None:
                    ai_idx = 1  # Default assumption
            else:
                ai_idx = 1

            ai_prob = probs[0, ai_idx].item()

        prediction, confidence = self._score_to_prediction(ai_prob)

        return DetectionResult(
            text=text,
            score=ai_prob,
            prediction=prediction,
            confidence=confidence,
            detector_name=self.name,
            raw_output={"logits": outputs.logits.cpu().numpy().tolist()},
        )

    def detect_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
        show_progress: bool = True
    ) -> List[DetectionResult]:
        """Detect if multiple texts are AI-generated."""
        results = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="RADAR Detection")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            # Determine AI label index
            if hasattr(self.model.config, 'id2label'):
                ai_idx = None
                for idx, label in self.model.config.id2label.items():
                    if 'ai' in label.lower() or 'machine' in label.lower() or 'generated' in label.lower():
                        ai_idx = int(idx)
                        break
                if ai_idx is None:
                    ai_idx = 1
            else:
                ai_idx = 1

            for j, text in enumerate(batch_texts):
                ai_prob = probs[j, ai_idx].item()
                prediction, confidence = self._score_to_prediction(ai_prob)

                results.append(DetectionResult(
                    text=text,
                    score=ai_prob,
                    prediction=prediction,
                    confidence=confidence,
                    detector_name=self.name,
                ))

        return results


class RADARWithParaphraser(RADARDetector):
    """
    RADAR detector with integrated paraphraser for adversarial evaluation.

    This variant can paraphrase input text to test detector robustness.
    """

    def __init__(
        self,
        model_id: str = "TrustSafeAI/RADAR-Vicuna-7B",
        paraphraser_model: str = "humarin/chatgpt_paraphraser_on_T5_base",
        threshold: float = 0.5,
        max_length: int = 512,
    ):
        """
        Initialize RADAR with paraphraser.

        Args:
            model_id: RADAR model ID
            paraphraser_model: Paraphraser model for adversarial testing
            threshold: Classification threshold
            max_length: Maximum sequence length
        """
        super().__init__(model_id, threshold, max_length)
        self.paraphraser_model_id = paraphraser_model
        self._paraphraser = None
        self._para_tokenizer = None

    def _load_paraphraser(self):
        """Lazy load paraphraser model."""
        if self._paraphraser is None:
            from transformers import AutoModelForSeq2SeqLM
            self._para_tokenizer = AutoTokenizer.from_pretrained(
                self.paraphraser_model_id, token=self.hf_token
            )
            self._paraphraser = AutoModelForSeq2SeqLM.from_pretrained(
                self.paraphraser_model_id, token=self.hf_token
            )
            self._paraphraser.to(self.device)
            self._paraphraser.eval()

    def paraphrase(self, text: str, num_return: int = 1) -> List[str]:
        """
        Paraphrase input text.

        Args:
            text: Input text to paraphrase
            num_return: Number of paraphrases to generate

        Returns:
            List of paraphrased texts
        """
        self._load_paraphraser()

        inputs = self._para_tokenizer(
            f"paraphrase: {text}",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._paraphraser.generate(
                **inputs,
                max_length=self.max_length,
                num_return_sequences=num_return,
                num_beams=num_return * 2,
                do_sample=True,
                temperature=0.7,
            )

        paraphrases = [
            self._para_tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]

        return paraphrases

    def detect_with_paraphrase(
        self,
        text: str,
        num_paraphrases: int = 3
    ) -> dict:
        """
        Detect AI text with robustness check via paraphrasing.

        Args:
            text: Input text
            num_paraphrases: Number of paraphrases to test

        Returns:
            Dictionary with original and paraphrase detection results
        """
        # Detect original
        original_result = self.detect(text)

        # Generate and detect paraphrases
        paraphrases = self.paraphrase(text, num_paraphrases)
        paraphrase_results = [self.detect(p) for p in paraphrases]

        # Compute robustness metrics
        original_pred = original_result.prediction == "ai"
        para_preds = [r.prediction == "ai" for r in paraphrase_results]
        consistency = sum(p == original_pred for p in para_preds) / len(para_preds)

        return {
            "original": original_result,
            "paraphrases": list(zip(paraphrases, paraphrase_results)),
            "consistency": consistency,
            "robust_prediction": sum(para_preds) > len(para_preds) / 2,
        }
