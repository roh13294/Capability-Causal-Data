from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from causal_reliability.certificates.distances import js_divergence, label_flip, margin_collapse, softmax
from causal_reliability.certificates.shift_risk import ShiftRiskWeights


@dataclass(frozen=True)
class CISWeights:
    w_flip: float = 2.0
    w_margin: float = 0.5
    w_tail: float = 0.5
    w_js: float = 0.25


@torch.no_grad()
def compute_counterfactual_outputs(model: torch.nn.Module, x: torch.Tensor, counterfactuals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits = model(x)
    flat_cf = counterfactuals.reshape(-1, *counterfactuals.shape[2:])
    logits_cf = model(flat_cf).reshape(x.shape[0], counterfactuals.shape[1], -1)
    return logits, logits_cf


def compute_shift_risk(
    logits_original: torch.Tensor,
    logits_counterfactuals: torch.Tensor,
    weights: ShiftRiskWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or ShiftRiskWeights()
    collapse = margin_collapse(logits_original, logits_counterfactuals)
    flips = label_flip(logits_original, logits_counterfactuals)
    js = js_divergence(softmax(logits_original), softmax(logits_counterfactuals))
    q90 = torch.quantile(collapse, q=0.9, dim=1)
    risk = (
        weights.alpha * collapse.mean(dim=1)
        + weights.beta * q90
        + weights.gamma * js.mean(dim=1)
        + weights.delta * flips.mean(dim=1)
    )
    parts = {
        "margin_collapse_mean": collapse.mean(dim=1),
        "margin_collapse_q90": q90,
        "js_mean": js.mean(dim=1),
        "flip_mean": flips.mean(dim=1),
    }
    return risk, parts


def _unit_normalize(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    values = values.float()
    lo = values.min()
    hi = values.max()
    if torch.isclose(hi, lo):
        return torch.zeros_like(values)
    return (values - lo) / (hi - lo + eps)


def compute_counterfactual_instability_score(
    parts: dict[str, torch.Tensor],
    weights: CISWeights | None = None,
) -> torch.Tensor:
    weights = weights or CISWeights()
    flip = parts["flip_mean"].float()
    margin = _unit_normalize(parts["margin_collapse_mean"])
    tail = _unit_normalize(parts["margin_collapse_q90"])
    js = _unit_normalize(parts["js_mean"])
    return weights.w_flip * flip + weights.w_margin * margin + weights.w_tail * tail + weights.w_js * js


def compute_cis_reliability(cis: torch.Tensor) -> torch.Tensor:
    return torch.exp(-cis.clamp_min(0.0))


def compute_causal_reliability(shift_risk: torch.Tensor) -> torch.Tensor:
    return torch.exp(-shift_risk.clamp_min(0.0))


@torch.no_grad()
def batch_compute_certificates(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    make_counterfactuals: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    weights: ShiftRiskWeights | None = None,
) -> dict[str, torch.Tensor]:
    rows: dict[str, list[torch.Tensor]] = {
        "pred": [],
        "label": [],
        "confidence": [],
        "margin": [],
        "shift_risk": [],
        "causal_reliability": [],
        "cis": [],
        "cis_reliability": [],
        "margin_collapse_mean": [],
        "margin_collapse_q90": [],
        "js_mean": [],
        "flip_mean": [],
    }
    from causal_reliability.certificates.distances import confidence, logits_to_margin

    for x, y, _shortcut, _causal in loader:
        x = x.to(device)
        y = y.to(device)
        cf = make_counterfactuals(x).to(device)
        logits, logits_cf = compute_counterfactual_outputs(model, x, cf)
        probs = softmax(logits)
        risk, parts = compute_shift_risk(logits, logits_cf, weights)
        cis = compute_counterfactual_instability_score(parts)
        rows["pred"].append(logits.argmax(dim=1).cpu())
        rows["label"].append(y.cpu())
        rows["confidence"].append(confidence(probs).cpu())
        rows["margin"].append(logits_to_margin(logits).cpu())
        rows["shift_risk"].append(risk.cpu())
        rows["causal_reliability"].append(compute_causal_reliability(risk).cpu())
        rows["cis"].append(cis.cpu())
        rows["cis_reliability"].append(compute_cis_reliability(cis).cpu())
        for key, value in parts.items():
            rows[key].append(value.cpu())
    return {key: torch.cat(value) for key, value in rows.items()}
