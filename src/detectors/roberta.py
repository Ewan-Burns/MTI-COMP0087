"""
RoBERTa-based supervised detector.
Fine-tuned classifier for AI text detection.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

from .base import BaseDetector, DetectionResult
from ..config import get_hf_token


class RoBERTaDetector(BaseDetector):
    """
    RoBERTa-based supervised detector for AI text detection.

    Can be used with pre-trained detection models or fine-tuned
    on custom datasets.
    """

    def __init__(
        self,
        model_id: str = "roberta-base",
        threshold: float = 0.5,
        max_length: int = 512,
        num_labels: int = 2,
    ):
        """
        Initialize RoBERTa detector.

        Args:
            model_id: HuggingFace model ID (base or fine-tuned)
            threshold: Classification threshold
            max_length: Maximum sequence length
            num_labels: Number of output classes
        """
        super().__init__("roberta", threshold)
        self.model_id = model_id
        self.max_length = max_length
        self.num_labels = num_labels
        self.hf_token = get_hf_token()

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=self.hf_token
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            token=self.hf_token,
        )

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def detect(self, text: str) -> DetectionResult:
        """Detect if text is AI-generated."""
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

            # Assuming label 1 = AI, label 0 = human
            ai_prob = probs[0, 1].item()

        prediction, confidence = self._score_to_prediction(ai_prob)

        return DetectionResult(
            text=text,
            score=ai_prob,
            prediction=prediction,
            confidence=confidence,
            detector_name=self.name,
            raw_output={"logits": outputs.logits.cpu().numpy().tolist()},
        )

    def detect_batch(self, texts: List[str], batch_size: int = 16, show_progress: bool = True) -> List[DetectionResult]:
        """Detect if multiple texts are AI-generated."""
        results = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="RoBERTa Detection")

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

            for j, text in enumerate(batch_texts):
                ai_prob = probs[j, 1].item()
                prediction, confidence = self._score_to_prediction(ai_prob)

                results.append(DetectionResult(
                    text=text,
                    score=ai_prob,
                    prediction=prediction,
                    confidence=confidence,
                    detector_name=self.name,
                ))

        return results

    def fine_tune(
        self,
        train_texts: List[str],
        train_labels: List[int],  # 0 = human, 1 = AI
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        output_dir: str = "outputs/roberta_detector",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
    ):
        """
        Fine-tune the RoBERTa model on labeled data.

        Args:
            train_texts: Training texts
            train_labels: Training labels (0=human, 1=AI)
            val_texts: Validation texts
            val_labels: Validation labels
            output_dir: Output directory for checkpoints
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        # Prepare datasets
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        train_dataset = Dataset.from_dict({
            **train_encodings,
            "labels": train_labels,
        })

        eval_dataset = None
        if val_texts and val_labels:
            val_encodings = self.tokenizer(
                val_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            eval_dataset = Dataset.from_dict({
                **val_encodings,
                "labels": val_labels,
            })

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            logging_steps=100,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Train
        trainer.train()

        # Save the model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"Model saved to {output_dir}")
