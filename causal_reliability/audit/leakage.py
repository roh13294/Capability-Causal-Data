from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from causal_reliability.analysis.metrics import auroc, spearman


FORBIDDEN_CERTIFICATE_INPUT_KEYS = {
    "shifted_label",
    "shifted_labels",
    "shift_label",
    "shift_labels",
    "shifted_correct",
    "shift_correct",
    "correct",
    "failure",
    "failures",
    "failure_label",
    "failure_labels",
    "test_label",
    "test_labels",
    "y_test",
    "labels",
    "label",
    "y",
}

ALLOWED_CERTIFICATE_INPUT_KEYS = {
    "logits_original",
    "original_logits",
    "logits",
    "logits_counterfactuals",
    "counterfactual_logits",
    "cf_logits",
    "counterfactual_outputs",
    "intervention_outputs",
    "interventions",
    "weights",
}


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _dataset_tensors(dataset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(dataset, "tensors"):
        tensors = dataset.tensors
        if len(tensors) < 4:
            raise AssertionError("dataset must expose x, y, shortcut, and causal tensors")
        return tuple(_as_numpy(t) for t in tensors[:4])  # type: ignore[return-value]
    if all(hasattr(dataset, name) for name in ("x", "y", "shortcut", "causal")):
        return (
            _as_numpy(dataset.x),
            _as_numpy(dataset.y),
            _as_numpy(dataset.shortcut),
            _as_numpy(dataset.causal),
        )
    raise AssertionError("dataset must be a TensorDataset or expose x/y/shortcut/causal attributes")


def _counterfactual_array(counterfactuals: Any) -> np.ndarray:
    if isinstance(counterfactuals, Mapping):
        for key in ("x", "counterfactuals", "cf", "x_cf"):
            if key in counterfactuals:
                return _as_numpy(counterfactuals[key])
        raise AssertionError("counterfactual mapping must include x, counterfactuals, cf, or x_cf")
    return _as_numpy(counterfactuals)


def _metadata_array(counterfactuals: Any, key: str) -> np.ndarray | None:
    if isinstance(counterfactuals, Mapping) and key in counterfactuals:
        return _as_numpy(counterfactuals[key])
    return None


def _first_difference_axis(original: np.ndarray, counterfactuals: np.ndarray) -> np.ndarray:
    cf = counterfactuals
    if cf.ndim == original.ndim + 1:
        cf = cf[:, 0]
    if cf.shape[0] != original.shape[0]:
        raise AssertionError("counterfactual batch must align with dataset rows")
    return np.abs(cf - original)


def assert_no_shift_labels_in_certificate_inputs(
    inputs: Mapping[str, Any] | Sequence[str] | None = None,
    **kwargs: Any,
) -> None:
    """Assert that certificate scoring receives only pre-evaluation model/intervention outputs.

    Pass either a mapping of inputs used by the scoring function, a sequence of input names,
    or keyword arguments. Labels may still be used later for evaluation after scores exist.
    """
    if inputs is None:
        keys = set(kwargs)
    elif isinstance(inputs, Mapping):
        keys = set(inputs)
        keys.update(kwargs)
    else:
        keys = set(inputs)
        keys.update(kwargs)
    normalized = {str(key).lower() for key in keys}
    forbidden = normalized & FORBIDDEN_CERTIFICATE_INPUT_KEYS
    if forbidden:
        raise AssertionError(f"certificate inputs include leakage-prone fields: {sorted(forbidden)}")
    unknown = normalized - ALLOWED_CERTIFICATE_INPUT_KEYS
    if unknown:
        raise AssertionError(f"certificate inputs include unrecognized fields: {sorted(unknown)}")


def assert_counterfactual_label_preservation(dataset: Any, counterfactuals: Any) -> None:
    _x, y, _shortcut, _causal = _dataset_tensors(dataset)
    cf_y = _metadata_array(counterfactuals, "y")
    if cf_y is None:
        cf_y = _metadata_array(counterfactuals, "label")
    if cf_y is None:
        cf_y = _metadata_array(counterfactuals, "labels")
    if cf_y is None:
        return
    if cf_y.ndim > y.ndim:
        expected = np.expand_dims(y, tuple(range(y.ndim, cf_y.ndim)))
    else:
        expected = y
    if not np.array_equal(cf_y, np.broadcast_to(expected, cf_y.shape)):
        raise AssertionError("counterfactual labels changed under a label-preserving intervention")


def assert_shortcut_changed(dataset: Any, counterfactuals: Any) -> None:
    x, _y, shortcut, _causal = _dataset_tensors(dataset)
    cf_shortcut = _metadata_array(counterfactuals, "shortcut")
    if cf_shortcut is not None:
        original = shortcut[:, None] if cf_shortcut.ndim > shortcut.ndim else shortcut
        if np.all(cf_shortcut == original):
            raise AssertionError("counterfactual shortcut metadata did not change")
        return
    diff = _first_difference_axis(x, _counterfactual_array(counterfactuals))
    if diff.reshape(diff.shape[0], -1).sum(axis=1).min(initial=0.0) <= 0:
        raise AssertionError("at least one counterfactual leaves the input unchanged")


def assert_causal_feature_preserved(dataset: Any, counterfactuals: Any) -> None:
    _x, _y, _shortcut, causal = _dataset_tensors(dataset)
    cf_causal = _metadata_array(counterfactuals, "causal")
    if cf_causal is not None:
        original = causal[:, None] if cf_causal.ndim > causal.ndim else causal
        if not np.array_equal(cf_causal, np.broadcast_to(original, cf_causal.shape)):
            raise AssertionError("counterfactual causal feature metadata changed")
        return
    cf = _counterfactual_array(counterfactuals)
    if cf.ndim >= 3 and cf.shape[-1] >= 1 and causal.ndim == 1:
        causal_coord = cf[..., 0]
        expected = causal[:, None] if causal_coord.ndim == 2 else causal
        if np.allclose(causal_coord, np.broadcast_to(expected, causal_coord.shape), atol=1e-6):
            return
    return


def assert_metric_polarity(scores: Mapping[str, Any] | Any, failures: Any) -> None:
    failures_np = _as_numpy(failures).astype(int)
    if isinstance(scores, Mapping):
        risk_scores = scores.get("ShiftRisk", scores.get("shift_risk"))
        cr_scores = scores.get("CR", scores.get("causal_reliability"))
        if risk_scores is not None:
            _assert_positive_failure_direction(_as_numpy(risk_scores), failures_np, "ShiftRisk")
        if cr_scores is not None:
            _assert_negative_failure_direction(_as_numpy(cr_scores), failures_np, "CR")
        if risk_scores is None and cr_scores is None:
            raise AssertionError("scores must include ShiftRisk/shift_risk or CR/causal_reliability")
    else:
        _assert_positive_failure_direction(_as_numpy(scores), failures_np, "scores")


def _assert_positive_failure_direction(scores: np.ndarray, failures: np.ndarray, name: str) -> None:
    if len(np.unique(scores)) < 2 or len(np.unique(failures)) < 2:
        return
    auc = auroc(scores, failures)
    corr = spearman(scores, failures)
    if np.isfinite(auc) and auc < 0.5 and np.isfinite(corr) and corr < 0:
        raise AssertionError(f"{name} appears sign-inverted for failure risk: AUROC={auc:.3f}")


def _assert_negative_failure_direction(scores: np.ndarray, failures: np.ndarray, name: str) -> None:
    if len(np.unique(scores)) < 2 or len(np.unique(failures)) < 2:
        return
    auc = auroc(scores, failures)
    corr = spearman(scores, failures)
    if np.isfinite(auc) and auc > 0.5 and np.isfinite(corr) and corr > 0:
        raise AssertionError(f"{name} appears sign-inverted for reliability: AUROC={auc:.3f}")
