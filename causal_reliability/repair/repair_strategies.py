from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RepairDecision:
    repaired_prediction: int | None
    repaired_confidence: float
    selected_intervention: str
    repair_strategy: str
    repair_action: str
    consensus_stability: float


def _as_prob_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    row_sums = arr.sum(axis=1, keepdims=True)
    return arr / np.clip(row_sums, 1e-12, None)


def _margin(probs: np.ndarray) -> np.ndarray:
    ordered = np.sort(probs, axis=1)
    return ordered[:, -1] - ordered[:, -2] if probs.shape[1] > 1 else np.ones(probs.shape[0])


def intervention_instability(original_probs: np.ndarray, counterfactual_probs: np.ndarray) -> np.ndarray:
    original = _as_prob_matrix(original_probs)
    cf = _as_prob_matrix(counterfactual_probs)
    flips = (original.argmax(axis=1) != cf.argmax(axis=1)).astype(float)
    shift = np.abs(original - cf).sum(axis=1) / 2.0
    collapse = np.maximum(0.0, _margin(original) - _margin(cf))
    return flips + 0.5 * collapse + 0.25 * shift


def shortcut_neutralized_prediction(
    original_probs: np.ndarray,
    counterfactual_probs: np.ndarray,
    intervention_names: list[str] | None = None,
) -> RepairDecision:
    cf = _as_prob_matrix(counterfactual_probs)
    names = intervention_names or [f"intervention_{i}" for i in range(len(cf))]
    instability = intervention_instability(np.repeat(_as_prob_matrix(original_probs), len(cf), axis=0), cf)
    idx = int(np.argmax(instability))
    probs = cf[idx]
    return RepairDecision(
        repaired_prediction=int(probs.argmax()),
        repaired_confidence=float(probs.max()),
        selected_intervention=names[idx],
        repair_strategy="shortcut_neutralized_prediction",
        repair_action="repair",
        consensus_stability=float(np.exp(-instability[idx])),
    )


def counterfactual_consensus(
    counterfactual_probs: np.ndarray,
    intervention_names: list[str] | None = None,
    mode: str = "probability_average",
) -> RepairDecision:
    cf = _as_prob_matrix(counterfactual_probs)
    names = intervention_names or [f"intervention_{i}" for i in range(len(cf))]
    if mode == "majority_vote":
        votes = cf.argmax(axis=1)
        counts = np.bincount(votes, minlength=cf.shape[1]).astype(float)
        probs = counts / max(1.0, counts.sum())
    else:
        probs = cf.mean(axis=0)
    pred = int(probs.argmax())
    agreement = float((cf.argmax(axis=1) == pred).mean()) if len(cf) else 0.0
    dispersion = float(np.mean(np.abs(cf - probs.reshape(1, -1)).sum(axis=1) / 2.0)) if len(cf) else 1.0
    stability = max(0.0, min(1.0, agreement * np.exp(-dispersion)))
    return RepairDecision(
        repaired_prediction=pred,
        repaired_confidence=float(probs.max()),
        selected_intervention=",".join(names),
        repair_strategy="counterfactual_consensus",
        repair_action="repair",
        consensus_stability=stability,
    )


def stability_weighted_prediction(
    original_probs: np.ndarray,
    counterfactual_probs: np.ndarray,
    intervention_names: list[str] | None = None,
) -> RepairDecision:
    original = _as_prob_matrix(original_probs)[0]
    cf = _as_prob_matrix(counterfactual_probs)
    instability = intervention_instability(np.repeat(original.reshape(1, -1), len(cf), axis=0), cf)
    weights = np.exp(-instability)
    all_probs = np.vstack([original.reshape(1, -1), cf])
    all_weights = np.concatenate([[1.0], weights])
    probs = (all_probs * all_weights.reshape(-1, 1)).sum(axis=0) / np.clip(all_weights.sum(), 1e-12, None)
    names = intervention_names or [f"intervention_{i}" for i in range(len(cf))]
    return RepairDecision(
        repaired_prediction=int(probs.argmax()),
        repaired_confidence=float(probs.max()),
        selected_intervention="weighted:" + ",".join(names),
        repair_strategy="stability_weighted_prediction",
        repair_action="repair",
        consensus_stability=float(weights.mean()) if len(weights) else 1.0,
    )


def abstention_decision(
    original_confidence: float,
    stability_score: float,
    consensus: RepairDecision | None = None,
    high_confidence_threshold: float = 0.8,
    low_stability_threshold: float = 0.5,
    stable_consensus_threshold: float = 0.6,
) -> RepairDecision:
    has_stable_consensus = consensus is not None and consensus.consensus_stability >= stable_consensus_threshold
    should_abstain = (
        original_confidence >= high_confidence_threshold
        and stability_score < low_stability_threshold
        and not has_stable_consensus
    )
    if should_abstain:
        return RepairDecision(
            repaired_prediction=None,
            repaired_confidence=0.0,
            selected_intervention=consensus.selected_intervention if consensus else "none",
            repair_strategy="abstention",
            repair_action="abstain",
            consensus_stability=consensus.consensus_stability if consensus else 0.0,
        )
    if consensus is not None:
        return RepairDecision(
            repaired_prediction=consensus.repaired_prediction,
            repaired_confidence=consensus.repaired_confidence,
            selected_intervention=consensus.selected_intervention,
            repair_strategy="cic_guided_abstention_policy",
            repair_action="repair",
            consensus_stability=consensus.consensus_stability,
        )
    return RepairDecision(
        repaired_prediction=None,
        repaired_confidence=0.0,
        selected_intervention="none",
        repair_strategy="abstention",
        repair_action="abstain",
        consensus_stability=0.0,
    )


def decision_to_dict(decision: RepairDecision) -> dict[str, Any]:
    return {
        "repaired_prediction": decision.repaired_prediction,
        "repaired_confidence": decision.repaired_confidence,
        "selected_intervention": decision.selected_intervention,
        "repair_strategy": decision.repair_strategy,
        "repair_action": decision.repair_action,
        "consensus_stability": decision.consensus_stability,
    }
