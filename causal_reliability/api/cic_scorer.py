from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_probs(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    row_sums = arr.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0) or np.any(~np.isfinite(row_sums)):
        raise ValueError("model_predict_fn must return nonnegative probability-like rows")
    return arr / row_sums


def _margin(probs: np.ndarray) -> np.ndarray:
    if probs.shape[1] < 2:
        return np.ones(probs.shape[0])
    ordered = np.sort(probs, axis=1)
    return ordered[:, -1] - ordered[:, -2]


@dataclass
class CICScorer:
    model_predict_fn: Callable[[list[Any]], Any]
    interventions: Iterable[Callable[[Any], Any]]

    def score_examples(self, examples: list[Any]) -> list[dict]:
        original = _as_probs(self.model_predict_fn(examples))
        preds = original.argmax(axis=1)
        confidence = original.max(axis=1)
        original_margin = _margin(original)
        interventions = list(self.interventions)
        rows = []
        for i, example in enumerate(examples):
            cf_examples = [fn(example) for fn in interventions]
            if cf_examples:
                cf_probs = _as_probs(self.model_predict_fn(cf_examples))
                cf_preds = cf_probs.argmax(axis=1)
                flip_rate = float((cf_preds != preds[i]).mean())
                cf_margin = _margin(cf_probs)
                margin_collapse = float(np.maximum(0.0, original_margin[i] - cf_margin).mean())
                l1_shift = float(np.abs(cf_probs - original[i]).sum(axis=1).mean() / 2.0)
            else:
                flip_rate = 0.0
                margin_collapse = 0.0
                l1_shift = 0.0
            cic_score = float(flip_rate + 0.5 * margin_collapse + 0.25 * l1_shift)
            stability = float(np.exp(-cic_score))
            true_label = example.get("label") if isinstance(example, dict) else None
            predicted_label = int(preds[i])
            correctness = None if true_label is None else int(predicted_label == int(true_label))
            rows.append(
                {
                    "example_id": example.get("example_id", i) if isinstance(example, dict) else i,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "correctness": correctness,
                    "confidence": float(confidence[i]),
                    "cic_score": cic_score,
                    "stability_score": stability,
                    "flip_rate": flip_rate,
                    "margin_collapse": margin_collapse,
                    "probability_shift": l1_shift,
                }
            )
        return rows
