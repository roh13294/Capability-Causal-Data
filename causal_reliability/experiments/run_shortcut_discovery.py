from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def make_synthetic_shortcut_data(
    n: int,
    n_features: int,
    causal_dims: list[int],
    shortcut_dims: list[int],
    seed: int,
    shortcut_strength: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n, n_features))
    causal_signal = x[:, causal_dims].sum(axis=1) + rng.normal(0, 0.35, size=n)
    y = (causal_signal > np.median(causal_signal)).astype(int)
    signed_y = np.where(y == 1, 1.0, -1.0)
    for dim in shortcut_dims:
        x[:, dim] = shortcut_strength * signed_y + rng.normal(0, 0.25, size=n)
    feature_types = ["noise"] * n_features
    for dim in causal_dims:
        feature_types[dim] = "causal feature"
    for dim in shortcut_dims:
        feature_types[dim] = "shortcut feature"
    return x, y, feature_types


def fit_linear_erm_proxy(x: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    signed = np.where(y == 1, 1.0, -1.0)
    x_aug = np.c_[np.ones(len(x)), x]
    reg = l2 * np.eye(x_aug.shape[1])
    reg[0, 0] = 0.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ signed)


def predict_proba(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.c_[np.ones(len(x)), x]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        logits = x_aug @ weights
    return _sigmoid(2.5 * logits)


def feature_instability_ranking(
    x: np.ndarray,
    weights: np.ndarray,
    feature_types: list[str],
    shortcut_dims: list[int],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 101)
    base_prob = predict_proba(x, weights)
    base_pred = (base_prob >= 0.5).astype(int)
    rows = []
    for dim in range(x.shape[1]):
        x_cf = x.copy()
        x_cf[:, dim] = rng.permutation(x_cf[:, dim])
        cf_prob = predict_proba(x_cf, weights)
        cf_pred = (cf_prob >= 0.5).astype(int)
        rows.append(
            {
                "feature_dim": dim,
                "feature_type": feature_types[dim],
                "known_shortcut": dim in shortcut_dims,
                "average_instability": float(np.mean(np.abs(cf_prob - base_prob))),
                "prediction_flip_rate": float(np.mean(cf_pred != base_pred)),
            }
        )
    rankings = pd.DataFrame(rows).sort_values(["average_instability", "prediction_flip_rate"], ascending=False).reset_index(drop=True)
    rankings["rank"] = np.arange(1, len(rankings) + 1)
    shortcut_ranks = rankings.loc[rankings["known_shortcut"], "rank"].astype(int).to_list()
    metrics = pd.DataFrame(
        [
            {
                "task": "synthetic",
                "shortcut_rank": int(min(shortcut_ranks)) if shortcut_ranks else -1,
                "shortcut_top1_hit": bool(shortcut_ranks and min(shortcut_ranks) <= 1),
                "shortcut_top3_hit": bool(shortcut_ranks and min(shortcut_ranks) <= 3),
                "mean_shortcut_instability": float(rankings.loc[rankings["feature_type"] == "shortcut feature", "average_instability"].mean()),
                "mean_causal_instability": float(rankings.loc[rankings["feature_type"] == "causal feature", "average_instability"].mean()),
                "mean_noise_instability": float(rankings.loc[rankings["feature_type"] == "noise", "average_instability"].mean()),
            }
        ]
    )
    return rankings, metrics


def _plot_rankings(rankings: pd.DataFrame, path: Path) -> None:
    colors = rankings["feature_type"].map({"shortcut feature": "#d55e00", "causal feature": "#0072b2", "noise": "#999999"}).fillna("#999999")
    plt.figure(figsize=(7.2, 4.2))
    labels = [f"d{int(d)}" for d in rankings["feature_dim"]]
    plt.bar(labels, rankings["average_instability"], color=colors)
    plt.ylabel("Average instability")
    plt.xlabel("Feature dimension ranked by instability")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _plot_type_means(rankings: pd.DataFrame, path: Path) -> None:
    means = rankings.groupby("feature_type")["average_instability"].mean().reindex(["shortcut feature", "causal feature", "noise"])
    plt.figure(figsize=(5.8, 4.2))
    plt.bar(means.index, means.values, color=["#d55e00", "#0072b2", "#999999"])
    plt.ylabel("Average instability")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def run(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg.get("results_dir", "results"))
    out_dir = ensure_dir(root / "shortcut_discovery")
    plot_dir = ensure_dir(out_dir / "plots")
    synthetic = cfg.get("synthetic", {})
    n = int(synthetic.get("n_examples", 512))
    n_features = int(synthetic.get("n_features", 8))
    causal_dims = [int(v) for v in synthetic.get("causal_dims", [0, 1])]
    shortcut_dims = [int(v) for v in synthetic.get("shortcut_dims", [2])]
    seed = int(cfg.get("seed", 0))
    shortcut_strength = float(synthetic.get("shortcut_strength", 1.6))
    x, y, feature_types = make_synthetic_shortcut_data(n, n_features, causal_dims, shortcut_dims, seed, shortcut_strength)
    weights = fit_linear_erm_proxy(x, y, l2=float(synthetic.get("l2", 1e-3)))
    rankings, metrics = feature_instability_ranking(x, weights, feature_types, shortcut_dims, seed)
    metrics.to_csv(out_dir / "shortcut_discovery_metrics.csv", index=False)
    rankings.to_csv(out_dir / "shortcut_discovery_rankings.csv", index=False)
    _plot_rankings(rankings, plot_dir / "feature_instability_ranking.png")
    _plot_type_means(rankings, plot_dir / "shortcut_vs_noise_instability.png")
    summary = [
        "# Shortcut Discovery Pilot",
        "",
        "This is a controlled pilot, not a general shortcut-discovery method. It asks whether CIC-style interventions can identify which known input factors behave like shortcuts in a synthetic setting.",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
        "",
        "## Feature Ranking",
        "",
        _markdown_table(rankings),
        "",
        "Claim: in controlled settings, CIC can help identify which input factors behave like shortcuts. This does not imply discovery of arbitrary real-world causal variables.",
        "",
    ]
    (out_dir / "shortcut_discovery_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {
        "metrics": out_dir / "shortcut_discovery_metrics.csv",
        "rankings": out_dir / "shortcut_discovery_rankings.csv",
        "summary": out_dir / "shortcut_discovery_summary.md",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/shortcut_discovery.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
