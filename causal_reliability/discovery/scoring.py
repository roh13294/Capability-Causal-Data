from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from causal_reliability.certificates.distances import js_divergence, logits_to_margin, margin_collapse, softmax
from causal_reliability.discovery.candidate_interventions import CandidateIntervention
from causal_reliability.discovery.validation import label_preservation_rate, specificity_score


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


@torch.no_grad()
def score_candidate(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    candidate: CandidateIntervention,
    task_type: str,
    input_shape: tuple[int, ...],
    metadata: dict[str, Any] | None = None,
) -> dict[str, float | str | bool]:
    metadata = metadata or {}
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    model.eval()
    logits = model(x)
    probs = softmax(logits)
    pred = probs.argmax(dim=1)
    base_confidence = probs.max(dim=1).values
    x_prime = candidate.apply(x, y=y, metadata=metadata)
    logits_prime = model(x_prime)
    probs_prime = softmax(logits_prime)
    pred_prime = probs_prime.argmax(dim=1)
    cf_confidence = probs_prime.max(dim=1).values

    prediction_instability = (pred_prime != pred).float().mean()
    margin_instability = margin_collapse(logits, logits_prime.unsqueeze(1)).clamp_min(0).mean()
    distribution_instability = js_divergence(probs, probs_prime.unsqueeze(1)).mean()
    confidence_preservation = (cf_confidence / base_confidence.clamp_min(1e-6)).clamp(0, 1).mean()
    label_rate = label_preservation_rate(task_type, x, x_prime, y, candidate, metadata)
    support = float(candidate.support_score)
    specificity = specificity_score(task_type, candidate, input_shape)
    instability_score = (
        _as_float(prediction_instability)
        + _as_float(margin_instability)
        + _as_float(distribution_instability)
    ) / 3.0
    instability_only = instability_score
    label_preserved_instability = instability_score * label_rate
    confidence_preserved_instability = label_preserved_instability * _as_float(confidence_preservation)
    full = confidence_preserved_instability * support * specificity
    return {
        "candidate_id": candidate.name,
        "candidate_type": candidate.factor_type,
        "description": candidate.description,
        "prediction_instability": _as_float(prediction_instability),
        "margin_instability": _as_float(margin_instability),
        "distribution_instability": _as_float(distribution_instability),
        "label_preservation_rate": float(label_rate),
        "support_score": support,
        "specificity_score": float(specificity),
        "confidence_preservation": _as_float(confidence_preservation),
        "instability_only": float(instability_only),
        "label_preserved_instability": float(label_preserved_instability),
        "confidence_preserved_instability": float(confidence_preserved_instability),
        "full_unknown_shortcut_score": float(full),
        "mean_margin": _as_float(logits_to_margin(logits).mean()),
    }
