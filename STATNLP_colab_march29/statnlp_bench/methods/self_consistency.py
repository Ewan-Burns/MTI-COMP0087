# --------------------------------------------------------------------------- #
# self_consistency.py — Self-Consistency decoding (Wang et al., 2022)
#
# Sample N responses to the same prompt, extract the final answer from each,
# and return the response whose answer appears most often (majority vote).
# Targets QA/math tasks where answers are short extractable strings.
# --------------------------------------------------------------------------- #
from __future__ import annotations

import re
from collections import Counter

from ..config import QuickRunConfig
from ..registry import register_method
from ..types import MethodRunResult, MethodSpec
from .publication import run_publication_generation_method

# Regex cascade for answer extraction — tries structured patterns first,
# falls back to taking the first line of the response.
_ANSWER_PATTERNS = [
    re.compile(r"the answer is[:\s]+(.+)", re.IGNORECASE),
    re.compile(r"answer[:\s]+(.+)", re.IGNORECASE),
    re.compile(r"final answer[:\s]+(.+)", re.IGNORECASE),
]

# LaTeX \boxed{} is common in math model outputs
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def _supports_reference_dataset(dataset_name: str) -> bool:
    name = dataset_name.lower()
    return "mt_bench" in name or "qa" in name


def normalize_answer(text: str) -> str:
    text = text.strip()
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip().lower()
    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip().splitlines()[0].lower()
    return re.sub(r"\s+", " ", text.splitlines()[0].strip()).lower()


def run_self_consistency(
    *,
    prompts: list[str],
    config: QuickRunConfig,
    base_method_name: str = "TOP_P_09",
    num_samples: int = 5,
) -> MethodRunResult:
    outputs: list[str] = []
    for prompt_idx, prompt in enumerate(prompts, start=1):
        candidates = [
            run_publication_generation_method(
                prompts=[prompt],
                method_name=base_method_name,
                config=config,
                prompt_start_idx=prompt_idx,
                run_idx=run_idx,
            ).texts[0]
            for run_idx in range(num_samples)
        ]
        # Majority vote on normalised answers, then return the first original
        # response that produced the winning answer (preserves full text).
        normalized = [normalize_answer(c) for c in candidates]
        best_answer, _ = Counter(normalized).most_common(1)[0]
        original = next(c for c, n in zip(candidates, normalized) if n == best_answer)
        outputs.append(original)
    return MethodRunResult(
        method_name="SELF_CONSISTENCY",
        texts=outputs,
        metadata={"base_method_name": base_method_name, "num_samples": num_samples},
    )


register_method(
    MethodSpec(
        name="SELF_CONSISTENCY",
        track="generative_detection",
        family="answer_voting",
        supports_answer_voting=True,
        supports_dataset=_supports_reference_dataset,
        run=run_self_consistency,
    )
)
