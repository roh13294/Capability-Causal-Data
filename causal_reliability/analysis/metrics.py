from __future__ import annotations

import numpy as np
import pandas as pd


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def auroc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = _rankdata(scores) + 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def auroc_with_reason(scores, labels, min_failures: int = 5) -> tuple[float, str]:
    labels = np.asarray(labels, dtype=int)
    n_failures = int((labels == 1).sum())
    n_correct = int((labels == 0).sum())
    if n_failures == 0 or n_correct == 0:
        return float("nan"), "AUROC undefined because shifted correctness contains only one class."
    value = auroc(scores, labels)
    if n_failures < min_failures:
        return value, "low failure count"
    return value, ""


def auprc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-scores)
    y = labels[order]
    if y.sum() == 0:
        return float("nan")
    precision = np.cumsum(y) / (np.arange(len(y)) + 1)
    recall_step = y / y.sum()
    return float((precision * recall_step).sum())


def spearman(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(np.unique(scores)) < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(np.corrcoef(_rankdata(scores), _rankdata(labels))[0, 1])


def failure_prediction_table(score_map: dict[str, np.ndarray], failure: np.ndarray) -> pd.DataFrame:
    rows = []
    failure = np.asarray(failure, dtype=int)
    n_failures = int(failure.sum())
    n_correct = int((1 - failure).sum())
    for name, scores in score_map.items():
        scores = np.asarray(scores, dtype=float)
        n_decile = max(1, len(scores) // 10)
        order = np.argsort(scores)
        bottom_values = failure[order[:n_decile]]
        top_values = failure[order[-n_decile:]]
        bottom = float(bottom_values.mean())
        top = float(top_values.mean())
        smoothed_top = (int(top_values.sum()) + 0.5) / (len(top_values) + 1)
        smoothed_bottom = (int(bottom_values.sum()) + 0.5) / (len(bottom_values) + 1)
        auc, reason = auroc_with_reason(scores, failure)
        rows.append(
            {
                "method": name,
                "failure_auroc": auc,
                "auroc_note": reason,
                "n_failures": n_failures,
                "n_correct": n_correct,
                "failure_auprc": auprc(scores, failure),
                "spearman": spearman(scores, failure),
                "top_decile_failure_rate": top,
                "bottom_decile_failure_rate": bottom,
                "top_decile_failure_rate_unsmoothed": top,
                "bottom_decile_failure_rate_unsmoothed": bottom,
                "risk_ratio": smoothed_top / smoothed_bottom,
            }
        )
    return pd.DataFrame(rows)


def accuracy(pred, y) -> float:
    pred = np.asarray(pred)
    y = np.asarray(y)
    return float((pred == y).mean())


def worst_group_accuracy(pred, y, group) -> float:
    pred, y, group = np.asarray(pred), np.asarray(y), np.asarray(group)
    accs = [(pred[group == g] == y[group == g]).mean() for g in np.unique(group) if (group == g).any()]
    return float(np.min(accs)) if accs else float("nan")
