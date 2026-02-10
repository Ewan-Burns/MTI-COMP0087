#!/usr/bin/env python3
"""
Example script for running MTI experiments.
Demonstrates generation, detection, and evaluation workflows.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    GENERATOR_MODELS,
    DETECTOR_CONFIGS,
    get_sampling_subset,
)
from src.experiment import Experiment, run_quick_experiment
from src.generation import quick_generate


def run_full_experiment():
    """Run a full experiment with all models and sampling configs."""
    # Sample prompts for generation
    prompts = [
        "The future of artificial intelligence in healthcare is",
        "Climate change presents significant challenges because",
        "The history of space exploration began when",
        "In modern literature, the concept of identity is often explored through",
        "The relationship between technology and privacy has become",
    ]

    # Create experiment
    exp = Experiment(
        name="full_experiment",
        output_dir="outputs",
        seed=42,
    )

    # Run with selected models and sampling configs
    model_configs = [
        GENERATOR_MODELS["llama-3.2-3b"],
        # Add more models as needed:
        # GENERATOR_MODELS["mistral-7b"],
        # GENERATOR_MODELS["qwen2-7b"],
    ]

    sampling_configs = get_sampling_subset("dubois_core")

    # Generate texts
    exp.run_generation(
        prompts=prompts,
        model_configs=model_configs,
        sampling_configs=sampling_configs,
        batch_size=1,
    )

    # Run detection
    detector_configs = [
        DETECTOR_CONFIGS["binoculars"],
        DETECTOR_CONFIGS["fastdetectgpt"],
        # DETECTOR_CONFIGS["radar"],  # Requires RADAR model
        # DETECTOR_CONFIGS["roberta"],  # Requires fine-tuned model
    ]

    # Provide human reference texts for AUROC computation
    human_texts = [
        "The development of renewable energy sources has become increasingly important in recent years.",
        "Many researchers believe that early intervention can significantly improve outcomes.",
        "The cultural impact of social media on younger generations is still being studied.",
    ]

    exp.run_detection(
        detector_configs=detector_configs,
        human_texts=human_texts,
    )

    # Generate report
    report = exp.generate_report()
    print("\n" + report)

    return exp


def run_quick_test():
    """Run a quick test with minimal configuration."""
    prompts = [
        "The importance of education in modern society cannot be",
        "Artificial intelligence will likely transform the way we",
    ]

    exp = run_quick_experiment(
        prompts=prompts,
        model_name="llama-3.2-3b",
        sampling_subset="quick_test",
    )

    return exp


def test_single_generation():
    """Test single text generation."""
    result = quick_generate(
        prompt="The future of renewable energy is",
        model_name="llama-3.2-3b",
        sampling_name="temp_0.7",
    )

    print(f"Prompt: {result.prompt}")
    print(f"Generated: {result.generated_text}")
    print(f"Tokens: {result.num_tokens_generated}")


def test_single_detection():
    """Test single detection."""
    from src.detectors import FastDetectGPTDetector

    detector = FastDetectGPTDetector(
        model_id="meta-llama/Llama-3.2-3B",
        threshold=0.5,
    )

    # Test with sample texts
    ai_text = "The implementation of machine learning algorithms in healthcare has revolutionized diagnostic procedures."
    human_text = "I went to the store yesterday and bought some groceries. The weather was nice."

    ai_result = detector.detect(ai_text)
    human_result = detector.detect(human_text)

    print(f"AI text score: {ai_result.score:.3f} ({ai_result.prediction})")
    print(f"Human text score: {human_result.score:.3f} ({human_result.prediction})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MTI experiments")
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "gen", "detect"],
        default="quick",
        help="Experiment mode: full, quick, gen (single generation), detect (single detection)",
    )

    args = parser.parse_args()

    if args.mode == "full":
        run_full_experiment()
    elif args.mode == "quick":
        run_quick_test()
    elif args.mode == "gen":
        test_single_generation()
    elif args.mode == "detect":
        test_single_detection()
