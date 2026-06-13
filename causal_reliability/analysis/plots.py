from __future__ import annotations

import os
import tempfile
from pathlib import Path

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from causal_reliability.analysis.metrics import auroc


def _save(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def accuracy_by_environment(metrics: dict[str, float], path: str | Path) -> None:
    plt.figure(figsize=(5, 3))
    names = list(metrics)
    plt.bar(names, [metrics[k] for k in names], color=["#3b82f6", "#14b8a6", "#ef4444"][: len(names)])
    plt.ylim(0, 1)
    plt.ylabel("accuracy")
    _save(path)


def reliability_vs_failure(reliability, failure, path: str | Path) -> None:
    plt.figure(figsize=(5, 3))
    plt.scatter(reliability, failure, s=10, alpha=0.45)
    plt.xlabel("causal reliability")
    plt.ylabel("shifted failure")
    _save(path)


def confidence_vs_reliability(confidence, reliability, failure, path: str | Path) -> None:
    plt.figure(figsize=(5, 3))
    plt.scatter(confidence, reliability, c=failure, s=12, alpha=0.55, cmap="coolwarm")
    plt.xlabel("confidence")
    plt.ylabel("causal reliability")
    _save(path)


def shift_risk_histogram(risk, failure, path: str | Path) -> None:
    plt.figure(figsize=(5, 3))
    risk = np.asarray(risk)
    failure = np.asarray(failure).astype(bool)
    plt.hist(risk[~failure], alpha=0.6, label="correct")
    plt.hist(risk[failure], alpha=0.6, label="failed")
    plt.xlabel("shift risk")
    plt.legend()
    _save(path)


def roc_failure_prediction(scores_by_name: dict[str, np.ndarray], failure, path: str | Path) -> None:
    failure = np.asarray(failure).astype(int)
    plt.figure(figsize=(5, 3))
    for name, score in scores_by_name.items():
        score = np.asarray(score)
        thresholds = np.r_[np.inf, np.sort(score)[::-1], -np.inf]
        tpr, fpr = [], []
        for t in thresholds:
            pred = score >= t
            tp = ((pred == 1) & (failure == 1)).sum()
            fp = ((pred == 1) & (failure == 0)).sum()
            fn = ((pred == 0) & (failure == 1)).sum()
            tn = ((pred == 0) & (failure == 0)).sum()
            tpr.append(tp / max(tp + fn, 1))
            fpr.append(fp / max(fp + tn, 1))
        plt.plot(fpr, tpr, label=f"{name} ({auroc(score, failure):.2f})")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(fontsize=7)
    _save(path)


def risk_decile_failure(risk, failure, path: str | Path) -> None:
    risk = np.asarray(risk)
    failure = np.asarray(failure)
    order = np.argsort(risk)
    chunks = np.array_split(order, 10)
    rates = [failure[c].mean() if len(c) else 0 for c in chunks]
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, 11), rates, marker="o")
    plt.xlabel("risk decile")
    plt.ylabel("failure rate")
    _save(path)


def reliability_calibration(bins, path: str | Path) -> None:
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], color="#555", linewidth=1)
    plt.scatter(bins["mean_reliability"], bins["shifted_accuracy"], s=np.maximum(bins["count"], 1))
    plt.xlabel("mean reliability")
    plt.ylabel("shifted accuracy")
    _save(path)


def counterfactual_grid(images, path: str | Path, max_rows: int = 4) -> None:
    images = images[:max_rows].detach().cpu()
    rows, cols = images.shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    axes = np.asarray(axes).reshape(rows, cols)
    for r in range(rows):
        for c in range(cols):
            axes[r, c].imshow(images[r, c].permute(1, 2, 0).clamp(0, 1))
            axes[r, c].axis("off")
    _save(path)
