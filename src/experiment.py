"""
Main experiment runner for the MTI project.
Orchestrates generation, detection, and evaluation.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from .config import (
    ExperimentConfig,
    ModelConfig,
    SamplingConfig,
    DetectorConfig,
    GENERATOR_MODELS,
    SAMPLING_CONFIGS,
    DETECTOR_CONFIGS,
    get_sampling_subset,
)
from .generation import TextGenerator, GenerationResult
from .detectors import BinocularsDetector, FastDetectGPTDetector, RADARDetector, RoBERTaDetector
from .metrics import compute_all_metrics


class Experiment:
    """
    Main experiment class for running generation and detection experiments.
    """

    def __init__(
        self,
        name: str,
        output_dir: str = "outputs",
        seed: int = 42,
    ):
        """
        Initialize experiment.

        Args:
            name: Experiment name
            output_dir: Output directory for results
            seed: Random seed
        """
        self.name = name
        self.output_dir = Path(output_dir) / name
        self.seed = seed

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for results
        self.generation_results: Dict[str, Dict[str, List[GenerationResult]]] = {}
        self.detection_results: Dict[str, Dict[str, List[Any]]] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}

    def run_generation(
        self,
        prompts: List[str],
        model_configs: List[ModelConfig],
        sampling_configs: List[SamplingConfig],
        batch_size: int = 1,
    ):
        """
        Run text generation for all model/sampling combinations.

        Args:
            prompts: List of prompts to generate from
            model_configs: List of model configurations
            sampling_configs: List of sampling configurations
            batch_size: Generation batch size
        """
        for model_config in model_configs:
            print(f"\n{'='*60}")
            print(f"Generating with model: {model_config.name}")
            print(f"{'='*60}")

            generator = TextGenerator(model_config)

            self.generation_results[model_config.name] = {}

            for sampling_config in sampling_configs:
                print(f"\nSampling: {sampling_config.name}")

                results = generator.generate(
                    prompts,
                    sampling_config,
                    batch_size=batch_size,
                )

                self.generation_results[model_config.name][sampling_config.name] = results

                # Compute and store text metrics
                texts = [r.generated_text for r in results]
                metrics = compute_all_metrics(texts)
                self.metrics[f"{model_config.name}_{sampling_config.name}"] = metrics

                print(f"  Generated {len(results)} texts")
                print(f"  Avg length: {metrics['avg_length']:.1f} words")
                print(f"  Diversity: {metrics['diversity_mean']:.3f}")

        # Save generation results
        self._save_generation_results()

    def run_detection(
        self,
        detector_configs: List[DetectorConfig],
        human_texts: Optional[List[str]] = None,
    ):
        """
        Run detection on all generated texts.

        Args:
            detector_configs: List of detector configurations
            human_texts: Optional human texts for comparison
        """
        for detector_config in detector_configs:
            print(f"\n{'='*60}")
            print(f"Running detector: {detector_config.name}")
            print(f"{'='*60}")

            # Initialize detector
            detector = self._create_detector(detector_config)

            self.detection_results[detector_config.name] = {}

            # Detect on all generated texts
            for model_name, sampling_results in self.generation_results.items():
                for sampling_name, gen_results in sampling_results.items():
                    key = f"{model_name}_{sampling_name}"
                    print(f"\n  Detecting: {key}")

                    texts = [r.generated_text for r in gen_results]
                    det_results = detector.detect_batch(texts, show_progress=True)

                    self.detection_results[detector_config.name][key] = det_results

                    # Compute detection metrics
                    ai_scores = [r.score for r in det_results]
                    ai_preds = [r.prediction for r in det_results]
                    ai_rate = sum(1 for p in ai_preds if p == "ai") / len(ai_preds)

                    print(f"    AI detection rate: {ai_rate:.1%}")
                    print(f"    Mean score: {np.mean(ai_scores):.3f}")

            # Also detect human texts if provided
            if human_texts:
                print(f"\n  Detecting human texts...")
                human_det_results = detector.detect_batch(human_texts, show_progress=True)
                self.detection_results[detector_config.name]["human"] = human_det_results

                human_scores = [r.score for r in human_det_results]
                human_ai_rate = sum(1 for r in human_det_results if r.prediction == "ai") / len(human_det_results)
                print(f"    False positive rate: {human_ai_rate:.1%}")

        # Save detection results
        self._save_detection_results()

    def _create_detector(self, config: DetectorConfig):
        """Create a detector from configuration."""
        if config.detector_type == "binoculars":
            return BinocularsDetector(
                model_id=config.model_id,
                observer_model_id=config.observer_model_id,
                threshold=config.threshold,
            )
        elif config.detector_type == "fastdetectgpt":
            return FastDetectGPTDetector(
                model_id=config.model_id,
                threshold=config.threshold,
            )
        elif config.detector_type == "radar":
            return RADARDetector(
                model_id=config.model_id,
                threshold=config.threshold,
            )
        elif config.detector_type == "roberta":
            return RoBERTaDetector(
                model_id=config.model_id,
                threshold=config.threshold,
            )
        else:
            raise ValueError(f"Unknown detector type: {config.detector_type}")

    def compute_auroc_matrix(self) -> pd.DataFrame:
        """
        Compute AUROC for each detector/generation combination.

        Returns:
            DataFrame with AUROC values
        """
        from sklearn.metrics import roc_auc_score

        rows = []

        for detector_name, det_results in self.detection_results.items():
            if "human" not in det_results:
                continue

            human_scores = [r.score for r in det_results["human"]]

            for key, ai_results in det_results.items():
                if key == "human":
                    continue

                ai_scores = [r.score for r in ai_results]

                # Compute AUROC
                labels = [0] * len(human_scores) + [1] * len(ai_scores)
                scores = human_scores + ai_scores

                try:
                    auroc = roc_auc_score(labels, scores)
                except:
                    auroc = 0.5

                rows.append({
                    "detector": detector_name,
                    "generation": key,
                    "auroc": auroc,
                })

        return pd.DataFrame(rows)

    def _save_generation_results(self):
        """Save generation results to disk."""
        output_file = self.output_dir / "generation_results.json"

        # Convert to serializable format
        data = {}
        for model_name, sampling_results in self.generation_results.items():
            data[model_name] = {}
            for sampling_name, results in sampling_results.items():
                data[model_name][sampling_name] = [
                    {
                        "prompt": r.prompt,
                        "generated_text": r.generated_text,
                        "num_tokens": r.num_tokens_generated,
                    }
                    for r in results
                ]

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        # Save metrics
        metrics_file = self.output_dir / "generation_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

        print(f"\nGeneration results saved to {output_file}")

    def _save_detection_results(self):
        """Save detection results to disk."""
        output_file = self.output_dir / "detection_results.json"

        # Convert to serializable format
        data = {}
        for detector_name, det_results in self.detection_results.items():
            data[detector_name] = {}
            for key, results in det_results.items():
                data[detector_name][key] = [
                    {
                        "score": r.score,
                        "prediction": r.prediction,
                        "confidence": r.confidence,
                    }
                    for r in results
                ]

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        # Save AUROC matrix if possible
        try:
            auroc_df = self.compute_auroc_matrix()
            auroc_df.to_csv(self.output_dir / "auroc_matrix.csv", index=False)
        except:
            pass

        print(f"\nDetection results saved to {output_file}")

    def generate_report(self) -> str:
        """Generate a summary report of the experiment."""
        report = []
        report.append(f"# Experiment Report: {self.name}")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")

        # Generation summary
        report.append("## Generation Summary")
        for model_name, sampling_results in self.generation_results.items():
            report.append(f"\n### Model: {model_name}")
            for sampling_name, results in sampling_results.items():
                key = f"{model_name}_{sampling_name}"
                metrics = self.metrics.get(key, {})
                report.append(f"- {sampling_name}: {len(results)} texts")
                report.append(f"  - Avg length: {metrics.get('avg_length', 0):.1f}")
                report.append(f"  - Diversity: {metrics.get('diversity_mean', 0):.3f}")
                report.append(f"  - Readability: {metrics.get('readability_mean', 0):.1f}")

        # Detection summary
        if self.detection_results:
            report.append("\n## Detection Summary")
            try:
                auroc_df = self.compute_auroc_matrix()
                report.append("\n### AUROC Matrix")
                report.append(auroc_df.to_markdown())
            except:
                pass

        report_text = "\n".join(report)

        # Save report
        report_file = self.output_dir / "report.md"
        with open(report_file, "w") as f:
            f.write(report_text)

        return report_text


def run_quick_experiment(
    prompts: List[str],
    model_name: str = "llama-3.2-3b",
    sampling_subset: str = "quick_test",
    output_dir: str = "outputs",
):
    """
    Run a quick experiment with default settings.

    Args:
        prompts: List of prompts
        model_name: Name of the generator model
        sampling_subset: Sampling configuration subset
        output_dir: Output directory
    """
    import numpy as np

    # Create experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = Experiment(f"quick_{timestamp}", output_dir=output_dir)

    # Get configurations
    model_configs = [GENERATOR_MODELS[model_name]]
    sampling_configs = get_sampling_subset(sampling_subset)

    # Run generation
    exp.run_generation(prompts, model_configs, sampling_configs)

    # Generate report
    report = exp.generate_report()
    print("\n" + report)

    return exp


# Import numpy for the quick experiment function
import numpy as np
