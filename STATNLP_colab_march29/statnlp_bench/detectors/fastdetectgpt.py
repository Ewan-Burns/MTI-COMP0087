# FastDetectGPT (Bao et al., 2023): "Fast-DetectGPT: Efficient Zero-Shot Detection
# of Machine-Generated Text via Conditional Probability Curvature"
# https://arxiv.org/abs/2310.05130


from __future__ import annotations

import torch

from ..registry import register_detector
from ..types import DetectorSpec
from ._model_loading import score_with_model_pair


def _fastdetectgpt_scores(
    log_probs_q: torch.Tensor,
    probs_r: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    # Matches Dubois et al. / Bao et al
    actual_logprob = torch.gather(log_probs_q, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    expected_logprob = (probs_r * log_probs_q).sum(dim=-1)
    variance = ((probs_r * log_probs_q**2).sum(dim=-1) - expected_logprob**2).clamp(min=1e-8)
    # Sum across tokens (masked), then compute single z-score per sequence
    numerator = ((actual_logprob - expected_logprob) * mask).sum(dim=-1)
    var_sum = (variance * mask).sum(dim=-1).clamp(min=1e-8)
    return numerator / torch.sqrt(var_sum)


def score_fastdetectgpt_texts(
    texts: list[str],
    *,
    main_model_name: str,
    aux_model_name: str,
    device: str = "auto",
    max_length: int = 512,
    batch_size: int = 8,
) -> list[float]:
    return score_with_model_pair(
        texts,
        main_model_name=main_model_name,
        aux_model_name=aux_model_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        detector_name="FastDetectGPT",
        compute_scores=_fastdetectgpt_scores,
    )


register_detector(
    DetectorSpec(
        name="FastDetectGPT",
        family="fastdetectgpt",
        requires_training=False,
        train=None,
        score_texts=score_fastdetectgpt_texts,
        predict_texts=None,
    )
)
