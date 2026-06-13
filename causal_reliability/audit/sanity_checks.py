from __future__ import annotations

import numpy as np


def random_score_baseline(failures, seed: int = 0) -> np.ndarray:
    failures = np.asarray(failures)
    return np.random.default_rng(seed).random(len(failures))


def shuffled_score_baseline(scores, seed: int = 0) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).copy()
    np.random.default_rng(seed).shuffle(scores)
    return scores


def oracle_failure_score_baseline(failures) -> np.ndarray:
    return np.asarray(failures, dtype=float)


def constant_score_baseline(failures, value: float = 0.0) -> np.ndarray:
    failures = np.asarray(failures)
    return np.full(len(failures), float(value))
