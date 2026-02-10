"""
API-based text generation using Groq (free tier).
Use this for quick testing without local GPU requirements.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIGenerationResult:
    """Result from API-based text generation."""
    prompt: str
    generated_text: str
    model: str
    sampling_params: dict


def generate_with_groq(
    prompt: str,
    model: str = "llama-3.2-3b-preview",
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 256,
    api_key: Optional[str] = None,
) -> APIGenerationResult:
    """
    Generate text using Groq's free API.

    Available models:
        - llama-3.2-3b-preview
        - llama-3.2-1b-preview
        - llama-3.1-8b-instant
        - mixtral-8x7b-32768

    Args:
        prompt: Input prompt
        model: Groq model name
        temperature: Sampling temperature (0.0 = greedy, higher = more random)
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        api_key: Groq API key (or set GROQ_API_KEY env var)

    Returns:
        APIGenerationResult with generated text
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Please install groq: pip install groq")

    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "Groq API key not found. Either pass api_key parameter or set GROQ_API_KEY in .env\n"
            "Get a free key at: https://console.groq.com/keys"
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    generated_text = response.choices[0].message.content

    return APIGenerationResult(
        prompt=prompt,
        generated_text=generated_text,
        model=model,
        sampling_params={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
    )


def quick_test(prompt: str = "What is the capital of France?") -> str:
    """Quick test function."""
    result = generate_with_groq(prompt, temperature=0.0)  # greedy
    return result.generated_text


if __name__ == "__main__":
    # Test
    result = generate_with_groq("What is the capital of France?", temperature=0.0)
    print(f"Prompt: {result.prompt}")
    print(f"Response: {result.generated_text}")
