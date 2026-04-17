# Binoculars detector (Hans et al., 2024): "Spotting LLM-Generated Text with a Glance"
# https://arxiv.org/abs/2401.12070
#
# Core idea: compare how two LMs (observer & performer) score the same text.
# Human text shows a high cross-entropy ratio; LLM-generated text does not.
# Score = mean_NLL(observer) / mean_cross_entropy(observer, performer)
# Lower scores → more likely AI-generated.

from __future__ import annotations

import torch

from ..registry import register_detector
from ..types import DetectorSpec
from ._model_loading import score_with_model_pair


def _binoculars_scores(
    log_probs_q: torch.Tensor,
    probs_r: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    # NLL of the actual next token under the observer (model_q)
    token_nll = -torch.gather(
        log_probs_q, dim=-1, index=labels.unsqueeze(-1),
    ).squeeze(-1)
    # Cross-entropy: E_{p_r}[-log p_q] — expectation over performer's distribution
    cross_entropy = -(probs_r * log_probs_q).sum(dim=-1)
    seq_lengths = mask.sum(dim=-1).clamp(min=1.0)
    mean_nll = (token_nll * mask).sum(dim=-1) / seq_lengths
    mean_cross_entropy = (cross_entropy * mask).sum(dim=-1) / seq_lengths
    # Ratio < 1 when the performer "expects" the text → likely AI-generated
    return mean_nll / mean_cross_entropy.clamp(min=1e-8)


def score_binoculars_texts(
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
        detector_name="Binoculars",
        compute_scores=_binoculars_scores,
    )


register_detector(
    DetectorSpec(
        name="Binoculars",
        family="binoculars",
        requires_training=False,
        train=None,
        score_texts=score_binoculars_texts,
        predict_texts=None,
    )
)
