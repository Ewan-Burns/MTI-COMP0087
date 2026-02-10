"""
Text generation with various sampling/decoding strategies.
Implements the inference methods from Dubois et al. (2025).
"""

import torch
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm

from .config import SamplingConfig, ModelConfig
from .models import get_model_manager


@dataclass
class GenerationResult:
    """Result from text generation."""
    prompt: str
    generated_text: str
    full_text: str  # prompt + generated
    sampling_config: str
    model_name: str
    num_tokens_generated: int


class TextGenerator:
    """
    Text generator supporting multiple sampling strategies.
    Based on the decoding methods from Dubois et al. (2025).
    """

    def __init__(self, model_config: ModelConfig):
        """
        Initialize the text generator.

        Args:
            model_config: Configuration for the generator model
        """
        self.model_config = model_config
        self.model_manager = get_model_manager()
        self.model, self.tokenizer = self.model_manager.load_generator(model_config)

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_config: SamplingConfig,
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> List[GenerationResult]:
        """
        Generate text continuations for given prompts.

        Args:
            prompts: Single prompt or list of prompts
            sampling_config: Sampling configuration to use
            batch_size: Batch size for generation
            show_progress: Whether to show progress bar

        Returns:
            List of GenerationResult objects
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        results = []
        generate_kwargs = sampling_config.to_generate_kwargs()

        # Process in batches
        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Generating ({sampling_config.name})")

        for i in iterator:
            batch_prompts = prompts[i:i + batch_size]
            batch_results = self._generate_batch(batch_prompts, generate_kwargs, sampling_config)
            results.extend(batch_results)

        return results

    def _generate_batch(
        self,
        prompts: List[str],
        generate_kwargs: Dict[str, Any],
        sampling_config: SamplingConfig,
    ) -> List[GenerationResult]:
        """Generate for a batch of prompts."""
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config.max_length // 2,  # Leave room for generation
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs,
            )

        # Decode and create results
        results = []
        for idx, (prompt, output) in enumerate(zip(prompts, outputs)):
            # Get only the generated part (excluding prompt)
            prompt_length = inputs["input_ids"][idx].shape[0]
            generated_ids = output[prompt_length:]

            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            full_text = self.tokenizer.decode(output, skip_special_tokens=True)

            results.append(GenerationResult(
                prompt=prompt,
                generated_text=generated_text,
                full_text=full_text,
                sampling_config=sampling_config.name,
                model_name=self.model_config.name,
                num_tokens_generated=len(generated_ids),
            ))

        return results

    def generate_with_multiple_configs(
        self,
        prompts: Union[str, List[str]],
        sampling_configs: List[SamplingConfig],
        batch_size: int = 1,
    ) -> Dict[str, List[GenerationResult]]:
        """
        Generate text with multiple sampling configurations.

        Args:
            prompts: Prompts to generate from
            sampling_configs: List of sampling configurations
            batch_size: Batch size for generation

        Returns:
            Dictionary mapping config name to list of results
        """
        results = {}

        for config in sampling_configs:
            print(f"\nGenerating with: {config.name}")
            results[config.name] = self.generate(
                prompts, config, batch_size=batch_size
            )

        return results


def quick_generate(
    prompt: str,
    model_name: str = "llama-3.2-3b",
    sampling_method: str = "greedy",
    max_new_tokens: int = 256,
) -> str:
    """
    Quick utility function for simple text generation.

    Args:
        prompt: Input prompt
        model_name: Name of the model to use
        sampling_method: Sampling method (greedy, top_p, top_k, temperature)
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    from .config import GENERATOR_MODELS, SAMPLING_CONFIGS

    if model_name not in GENERATOR_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(GENERATOR_MODELS.keys())}")

    model_config = GENERATOR_MODELS[model_name]
    generator = TextGenerator(model_config)

    # Get or create sampling config
    if sampling_method in SAMPLING_CONFIGS:
        sampling_config = SAMPLING_CONFIGS[sampling_method]
    else:
        sampling_config = SamplingConfig(
            name=sampling_method,
            method=sampling_method,
            params={"max_new_tokens": max_new_tokens},
        )

    results = generator.generate(prompt, sampling_config, show_progress=False)
    return results[0].generated_text


# Example usage and testing
if __name__ == "__main__":
    from .config import GENERATOR_MODELS, get_sampling_subset

    # Test with a simple prompt
    test_prompt = "The following is the full text of a news article titled 'Climate Change Summit' from bbc.com:"

    # Quick test
    print("Testing quick_generate...")
    text = quick_generate(test_prompt, model_name="llama-3.2-3b", sampling_method="greedy")
    print(f"Generated: {text[:200]}...")

    # Full test with multiple sampling configs
    print("\nTesting with multiple sampling configs...")
    model_config = GENERATOR_MODELS["llama-3.2-3b"]
    generator = TextGenerator(model_config)

    sampling_configs = get_sampling_subset("quick_test")
    results = generator.generate_with_multiple_configs(
        [test_prompt],
        sampling_configs,
        batch_size=1,
    )

    for config_name, result_list in results.items():
        print(f"\n{config_name}: {result_list[0].generated_text[:100]}...")
