from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def bootstrap_ci(values, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    arr = _finite(values)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1 or n_boot <= 0:
        value = float(arr.mean())
        return value, value
    gen = _rng()
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = gen.choice(arr, size=len(arr), replace=True)
        stats[i] = sample.mean()
    return tuple(float(x) for x in np.quantile(stats, [alpha / 2, 1 - alpha / 2]))


def bootstrap_metric_ci(y_true, y_score, metric_fn: Callable, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    y = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if len(y) != len(scores):
        raise ValueError("y_true and y_score must have the same length")
    finite = np.isfinite(scores)
    y = y[finite]
    scores = scores[finite]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    gen = _rng()
    stats = []
    for _ in range(n_boot):
        idx = gen.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        value = float(metric_fn(scores[idx], y[idx]))
        if np.isfinite(value):
            stats.append(value)
    if not stats:
        value = float(metric_fn(scores, y))
        return value, value
    return tuple(float(x) for x in np.quantile(stats, [alpha / 2, 1 - alpha / 2]))


def paired_bootstrap_auc_diff(
    y_true,
    score_a,
    score_b,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    from causal_reliability.analysis.metrics import auroc

    y = np.asarray(y_true)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    if len(y) != len(a) or len(y) != len(b):
        raise ValueError("all inputs must have the same length")
    finite = np.isfinite(a) & np.isfinite(b)
    y = y[finite]
    a = a[finite]
    b = b[finite]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(auroc(a, y) - auroc(b, y))
    gen = _rng()
    stats = []
    for _ in range(n_boot):
        idx = gen.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        value = float(auroc(a[idx], y[idx]) - auroc(b[idx], y[idx]))
        if np.isfinite(value):
            stats.append(value)
    if not stats:
        return point, point, point
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(low), float(high)


def risk_ratio_ci(top_failures, bottom_failures, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float, float]:
    top = np.asarray(top_failures, dtype=float)
    bottom = np.asarray(bottom_failures, dtype=float)
    if len(top) == 0 or len(bottom) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(top.mean() / max(bottom.mean(), 1e-8))
    gen = _rng()
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        top_sample = gen.choice(top, size=len(top), replace=True)
        bottom_sample = gen.choice(bottom, size=len(bottom), replace=True)
        stats[i] = top_sample.mean() / max(bottom_sample.mean(), 1e-8)
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(low), float(high)


def mean_std_summary(values) -> tuple[float, float]:
    arr = _finite(values)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def format_mean_std(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        return f"{mean:.3f} +/- NA"
    return f"{mean:.3f} +/- {std:.3f}"


def format_ci(low: float, high: float) -> str:
    if not np.isfinite(low) or not np.isfinite(high):
        return "NA"
    return f"[{low:.3f}, {high:.3f}]"


def count_classes(y_true) -> tuple[int, int]:
    y = np.asarray(y_true, dtype=int)
    return int((y == 1).sum()), int((y == 0).sum())


def defined_auroc_reason(y_true, min_failures: int = 1, min_correct: int = 1) -> str:
    n_failures, n_correct = count_classes(y_true)
    if n_failures < min_failures or n_correct < min_correct:
        return f"undefined AUROC: failures={n_failures}, correct={n_correct}"
    return ""
