from __future__ import annotations

from typing import Any

import numpy as np
import torch


def entropy(probabilities: torch.Tensor) -> np.ndarray:
    p = probabilities.detach().cpu().clamp_min(1e-8)
    return (-(p * p.log()).sum(dim=1)).numpy()


def negative_margin(probabilities: torch.Tensor) -> np.ndarray:
    values = torch.topk(probabilities.detach().cpu(), k=2, dim=1).values
    return (-(values[:, 0] - values[:, 1])).numpy()


def cic_from_predictions(original: dict[str, Any], counterfactual: dict[str, Any]) -> dict[str, np.ndarray]:
    p = original["probabilities"].detach().cpu()
    q = counterfactual["probabilities"].detach().cpu()
    pred = original["predictions"].detach().cpu()
    cf_pred = counterfactual["predictions"].detach().cpu()
    conf = original["confidence"].detach().cpu()
    cf_conf_for_pred = q.gather(1, pred.view(-1, 1)).squeeze(1)
    margin_drop = (conf - cf_conf_for_pred).clamp_min(0)
    flip = (pred != cf_pred).float()
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=1)
    js += 0.5 * (q * (q.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=1)
    cis = margin_drop + flip + 0.5 * js
    return {
        "cis": cis.numpy(),
        "shift_risk": (margin_drop + 0.5 * js + 0.25 * flip).numpy(),
        "label_flip_only": flip.numpy(),
        "cf_confidence_for_original_pred": cf_conf_for_pred.numpy(),
        "cf_prediction": cf_pred.numpy(),
        "cf_confidence": counterfactual["confidence"].detach().cpu().numpy(),
    }


def quadrant_label(confidence: float, cis: float, confidence_threshold: float = 0.8, stability_threshold: float = 0.5) -> str:
    stability = float(np.exp(-max(0.0, cis)))
    high_conf = confidence >= confidence_threshold
    high_stab = stability >= stability_threshold
    if high_conf and high_stab:
        return "Reliable prediction"
    if not high_conf and high_stab:
        return "Uncertain but stable"
    if not high_conf and not high_stab:
        return "Generally fragile"
    return "Dangerous shortcut reliance"
