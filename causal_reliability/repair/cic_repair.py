from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from causal_reliability.repair.repair_strategies import (
    RepairDecision,
    abstention_decision,
    counterfactual_consensus,
    decision_to_dict,
    intervention_instability,
    shortcut_neutralized_prediction,
    stability_weighted_prediction,
)


def _quadrant(confidence: float, stability_score: float, high_confidence_threshold: float, low_stability_threshold: float) -> str:
    if confidence >= high_confidence_threshold and stability_score < low_stability_threshold:
        return "Dangerous shortcut reliance"
    if confidence >= high_confidence_threshold:
        return "Confident and stable"
    if stability_score < low_stability_threshold:
        return "Uncertain and unstable"
    return "Low confidence but stable"


def _decision_for_strategy(
    strategy: str,
    original_probs: np.ndarray,
    counterfactual_probs: np.ndarray,
    intervention_names: list[str],
    confidence: float,
    stability_score: float,
    high_confidence_threshold: float,
    low_stability_threshold: float,
) -> RepairDecision:
    if strategy == "shortcut_neutralized":
        return shortcut_neutralized_prediction(original_probs, counterfactual_probs, intervention_names)
    if strategy == "counterfactual_consensus":
        return counterfactual_consensus(counterfactual_probs, intervention_names)
    if strategy == "stability_weighted":
        return stability_weighted_prediction(original_probs, counterfactual_probs, intervention_names)
    if strategy == "abstention":
        consensus = counterfactual_consensus(counterfactual_probs, intervention_names)
        return abstention_decision(
            confidence,
            stability_score,
            consensus,
            high_confidence_threshold=high_confidence_threshold,
            low_stability_threshold=low_stability_threshold,
        )
    raise ValueError(f"unknown repair strategy: {strategy}")


def build_repair_certificate(
    *,
    example_id: str | int,
    label: int,
    original_probs: np.ndarray,
    counterfactual_probs: np.ndarray,
    intervention_names: list[str] | None = None,
    strategy: str = "shortcut_neutralized",
    high_confidence_threshold: float = 0.8,
    low_stability_threshold: float = 0.5,
) -> dict[str, Any]:
    original = np.asarray(original_probs, dtype=float).reshape(-1)
    original = original / np.clip(original.sum(), 1e-12, None)
    cf = np.asarray(counterfactual_probs, dtype=float)
    if cf.ndim == 1:
        cf = cf.reshape(1, -1)
    names = intervention_names or [f"intervention_{i}" for i in range(len(cf))]
    instabilities = intervention_instability(np.repeat(original.reshape(1, -1), len(cf), axis=0), cf)
    cic_score = float(instabilities.max()) if len(instabilities) else 0.0
    stability_score = float(np.exp(-cic_score))
    original_prediction = int(original.argmax())
    original_confidence = float(original.max())
    original_correctness = int(original_prediction == int(label))
    decision = _decision_for_strategy(
        strategy,
        original,
        cf,
        names,
        original_confidence,
        stability_score,
        high_confidence_threshold,
        low_stability_threshold,
    )
    repaired_prediction = decision.repaired_prediction
    repaired_correctness = int(repaired_prediction == int(label)) if repaired_prediction is not None else 0
    repaired_failure_fixed = bool(original_correctness == 0 and repaired_correctness == 1)
    repair_success = bool(repaired_failure_fixed or (decision.repair_action == "abstain" and original_correctness == 0))
    return {
        "example_id": example_id,
        "label": int(label),
        "original_prediction": original_prediction,
        "original_confidence": original_confidence,
        "original_correctness": original_correctness,
        "cic_score": cic_score,
        "stability_score": stability_score,
        "quadrant": _quadrant(original_confidence, stability_score, high_confidence_threshold, low_stability_threshold),
        **decision_to_dict(decision),
        "repaired_correctness": repaired_correctness,
        "repair_success": repair_success,
        "repair_fixed_prediction": repaired_failure_fixed,
    }


def repair_batch(
    *,
    example_ids: list[str] | list[int],
    labels: np.ndarray,
    original_probs: np.ndarray,
    counterfactual_probs: np.ndarray,
    intervention_names: list[str] | None = None,
    strategy: str = "shortcut_neutralized",
    high_confidence_threshold: float = 0.8,
    low_stability_threshold: float = 0.5,
) -> pd.DataFrame:
    original = np.asarray(original_probs, dtype=float)
    cf = np.asarray(counterfactual_probs, dtype=float)
    if cf.ndim != 3:
        raise ValueError("counterfactual_probs must have shape [n_examples, n_interventions, n_classes]")
    rows = []
    for i, example_id in enumerate(example_ids):
        rows.append(
            build_repair_certificate(
                example_id=example_id,
                label=int(labels[i]),
                original_probs=original[i],
                counterfactual_probs=cf[i],
                intervention_names=intervention_names,
                strategy=strategy,
                high_confidence_threshold=high_confidence_threshold,
                low_stability_threshold=low_stability_threshold,
            )
        )
    return pd.DataFrame(rows)
