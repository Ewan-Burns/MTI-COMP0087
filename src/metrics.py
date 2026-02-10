"""
Text quality metrics for evaluating generated text.
Based on Dubois et al. (2025) and Drayson et al. (2025).
"""

import numpy as np
from typing import List, Dict, Any
from collections import Counter
import re


def compute_diversity(text: str, n_range: tuple = (2, 4)) -> float:
    """
    Compute n-gram diversity score.

    D(x) = prod_{n=2}^{4} (1 - unique_ngrams / total_ngrams)

    Higher scores indicate more diverse text.
    """
    tokens = text.split()
    if len(tokens) < n_range[1]:
        return 0.0

    diversity = 1.0
    for n in range(n_range[0], n_range[1] + 1):
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            continue
        unique_ratio = len(set(ngrams)) / len(ngrams)
        diversity *= (1 - (1 - unique_ratio))  # Keep high for diverse

    return diversity


def compute_self_bleu(texts: List[str], max_n: int = 4) -> float:
    """
    Compute Self-BLEU score for a set of texts.
    Lower scores indicate higher diversity across generations.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    if len(texts) < 2:
        return 0.0

    smoothing = SmoothingFunction().method1
    scores = []

    for i, text in enumerate(texts):
        hypothesis = text.split()
        references = [t.split() for j, t in enumerate(texts) if j != i]

        if not hypothesis or not references:
            continue

        score = sentence_bleu(
            references,
            hypothesis,
            weights=[1.0/max_n] * max_n,
            smoothing_function=smoothing,
        )
        scores.append(score)

    return np.mean(scores) if scores else 0.0


def compute_mtld(text: str, threshold: float = 0.72) -> float:
    """
    Compute Measure of Textual Lexical Diversity (MTLD).
    Higher values indicate more diverse vocabulary usage.
    """
    tokens = text.lower().split()
    if len(tokens) < 10:
        return 0.0

    def one_pass(tokens):
        segments = []
        segment_start = 0
        types = set()

        for i, token in enumerate(tokens):
            types.add(token)
            ttr = len(types) / (i - segment_start + 1)

            if ttr < threshold:
                segments.append(i - segment_start + 1)
                segment_start = i + 1
                types = set()

        # Handle remaining tokens
        if segment_start < len(tokens):
            remaining_ttr = len(types) / (len(tokens) - segment_start)
            partial = (1 - remaining_ttr) / (1 - threshold)
            segments.append(partial * (len(tokens) - segment_start))

        return np.mean(segments) if segments else len(tokens)

    forward = one_pass(tokens)
    backward = one_pass(tokens[::-1])

    return (forward + backward) / 2


def compute_hapax_ratio(text: str) -> float:
    """
    Compute Hapax Legomena Ratio.
    Proportion of words that occur exactly once.
    """
    tokens = text.lower().split()
    if not tokens:
        return 0.0

    freq = Counter(tokens)
    hapax = sum(1 for count in freq.values() if count == 1)

    return hapax / len(freq) if freq else 0.0


def compute_simpson_diversity(text: str) -> float:
    """
    Compute Simpson's Diversity Index.
    Lower values indicate more diverse text.
    """
    tokens = text.lower().split()
    if len(tokens) < 2:
        return 0.0

    freq = Counter(tokens)
    n = len(tokens)

    return sum((count / n) ** 2 for count in freq.values())


def compute_readability(text: str) -> float:
    """
    Compute Flesch-Kincaid Reading Ease score.
    Higher scores indicate easier to read text.
    """
    try:
        import textstat
        return textstat.flesch_reading_ease(text)
    except ImportError:
        # Simple approximation if textstat not available
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        syllables = sum(count_syllables(word) for word in text.split())

        if sentences == 0 or words == 0:
            return 0.0

        return 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)


def count_syllables(word: str) -> int:
    """Count syllables in a word (simple heuristic)."""
    word = word.lower()
    vowels = "aeiou"
    count = 0
    prev_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    # Handle silent e
    if word.endswith('e'):
        count -= 1

    return max(1, count)


def compute_zipf_exponent(text: str) -> float:
    """
    Compute Zipf's Law exponent (alpha).
    Measures how steeply frequency decays with rank.
    """
    tokens = text.lower().split()
    if len(tokens) < 10:
        return 0.0

    freq = Counter(tokens)
    sorted_freq = sorted(freq.values(), reverse=True)

    ranks = np.arange(1, len(sorted_freq) + 1)
    freqs = np.array(sorted_freq)

    # Fit log-log linear regression
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)

    # alpha = -cov(log_r, log_f) / var(log_r)
    cov = np.cov(log_ranks, log_freqs)[0, 1]
    var = np.var(log_ranks)

    return -cov / var if var > 0 else 0.0


def compute_heaps_exponent(texts: List[str]) -> float:
    """
    Compute Heaps' Law exponent (beta).
    Models rate at which new words appear.
    """
    if not texts:
        return 0.0

    total_tokens = []
    unique_tokens = []

    cumulative_text = ""
    for text in texts:
        cumulative_text += " " + text
        tokens = cumulative_text.lower().split()
        total_tokens.append(len(tokens))
        unique_tokens.append(len(set(tokens)))

    if len(total_tokens) < 2:
        return 0.0

    log_n = np.log(total_tokens)
    log_v = np.log(unique_tokens)

    # beta = cov(log_n, log_v) / var(log_n)
    cov = np.cov(log_n, log_v)[0, 1]
    var = np.var(log_n)

    return cov / var if var > 0 else 0.0


def compute_all_metrics(texts: List[str]) -> Dict[str, Any]:
    """
    Compute all text quality metrics for a list of texts.

    Args:
        texts: List of generated texts

    Returns:
        Dictionary with all metrics
    """
    if not texts:
        return {}

    # Per-text metrics
    diversities = [compute_diversity(t) for t in texts]
    mtlds = [compute_mtld(t) for t in texts]
    hapax_ratios = [compute_hapax_ratio(t) for t in texts]
    simpson_indices = [compute_simpson_diversity(t) for t in texts]
    readabilities = [compute_readability(t) for t in texts]
    zipf_exponents = [compute_zipf_exponent(t) for t in texts]

    # Corpus-level metrics
    self_bleu = compute_self_bleu(texts[:1000])  # Sample for efficiency
    heaps_exp = compute_heaps_exponent(texts)

    return {
        "diversity_mean": np.mean(diversities),
        "diversity_std": np.std(diversities),
        "mtld_mean": np.mean(mtlds),
        "mtld_std": np.std(mtlds),
        "hapax_ratio_mean": np.mean(hapax_ratios),
        "simpson_index_mean": np.mean(simpson_indices),
        "readability_mean": np.mean(readabilities),
        "readability_std": np.std(readabilities),
        "zipf_alpha_mean": np.mean(zipf_exponents),
        "self_bleu": self_bleu,
        "heaps_beta": heaps_exp,
        "num_texts": len(texts),
        "avg_length": np.mean([len(t.split()) for t in texts]),
    }


# Human reference values from Dubois et al. (2025) Table 2
HUMAN_REFERENCE = {
    "mtld": 94.60,
    "hapax_ratio": 0.3490,
    "simpson_index": 0.0066,
    "zipf_alpha": 1.20,
    "heaps_beta": 0.5946,
    "readability": 50.34,
    "self_bleu": 42.89,
}
