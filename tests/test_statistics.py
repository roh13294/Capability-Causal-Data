import numpy as np

from causal_reliability.analysis.metrics import auroc
from causal_reliability.analysis.statistics import (
    bootstrap_ci,
    bootstrap_metric_ci,
    paired_bootstrap_auc_diff,
    risk_ratio_ci,
)


def test_bootstrap_ci_returns_valid_intervals():
    low, high = bootstrap_ci([1, 2, 3, 4], n_boot=100)
    assert low <= high
    assert 1 <= low <= 4
    assert 1 <= high <= 4


def test_bootstrap_metric_ci_returns_valid_intervals():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    low, high = bootstrap_metric_ci(y, scores, auroc, n_boot=100)
    assert low <= high
    assert 0 <= low <= 1
    assert 0 <= high <= 1


def test_paired_bootstrap_auc_diff_has_expected_sign():
    y = np.array([0, 0, 1, 1, 1, 0])
    good = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3])
    bad = 1 - good
    diff, low, high = paired_bootstrap_auc_diff(y, good, bad, n_boot=100)
    assert diff > 0
    assert low <= high


def test_risk_ratio_ci_works():
    ratio, low, high = risk_ratio_ci([1, 1, 0, 1], [0, 0, 1, 0], n_boot=100)
    assert ratio > 1
    assert low <= high
