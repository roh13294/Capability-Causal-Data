from __future__ import annotations

from typing import Any

import torch

from causal_reliability.discovery.candidate_interventions import CandidateIntervention


def label_preservation_rate(
    task_type: str,
    x: torch.Tensor,
    x_prime: torch.Tensor,
    y: torch.Tensor,
    candidate: CandidateIntervention,
    metadata: dict[str, Any] | None = None,
) -> float:
    del x, x_prime, y
    metadata = metadata or {}
    if task_type == "vector":
        causal_dims = set(int(v) for v in metadata.get("causal_dims", [0]))
        affected = {int(v) for v in candidate.affected_features if isinstance(v, int)}
        return 0.35 if affected & causal_dims else 1.0
    if task_type == "vision":
        return 0.88 if candidate.factor_type == "small_translation" else 1.0
    if task_type == "text":
        causal_positions = set(int(v) for v in metadata.get("causal_positions", [0, 1]))
        affected = {int(v) for v in candidate.affected_features if isinstance(v, int)}
        return 0.55 if affected & causal_positions else 1.0
    return 1.0 if candidate.preserves_label_hint else 0.5


def specificity_score(task_type: str, candidate: CandidateIntervention, input_shape: tuple[int, ...]) -> float:
    if task_type == "vector":
        n_features = max(int(input_shape[0]), 1)
        affected = [v for v in candidate.affected_features if isinstance(v, int)]
        if not affected:
            return 0.45
        return max(0.25, 1.0 - (len(affected) - 1) / n_features)
    if task_type == "vision":
        broad = {"brightness", "contrast", "additive_noise", "blur", "small_translation"}
        return 0.58 if candidate.factor_type in broad else 0.94
    if task_type == "text":
        affected = [v for v in candidate.affected_features if isinstance(v, int)]
        return 0.92 if len(affected) <= 1 else max(0.45, 1.0 - 0.12 * len(affected))
    return 0.75
