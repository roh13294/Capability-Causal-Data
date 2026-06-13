from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

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
from causal_reliability.utils.io import ensure_dir


QUADRANT_LABELS = {
    (True, True): "Reliable prediction",
    (False, True): "Uncertain but causally stable",
    (False, False): "Generally fragile",
    (True, False): "Dangerous shortcut reliance",
}


def normalize_cic(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - lo) / (hi - lo)


def quadrant_label(confidence: float, stability: float, confidence_threshold: float = 0.8, stability_threshold: float = 0.5) -> str:
    high_confidence = float(confidence) >= confidence_threshold
    high_stability = float(stability) >= stability_threshold
    return QUADRANT_LABELS[(high_confidence, high_stability)]


def build_reliability_plane(
    certs: pd.DataFrame,
    confidence_threshold: float = 0.8,
    stability_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if certs.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    if "failure" not in certs.columns and {"pred", "label"}.issubset(certs.columns):
        certs = certs.copy()
        certs["failure"] = (certs["pred"] != certs["label"]).astype(int)
    if "cis" not in certs.columns:
        raise ValueError("reliability plane requires a 'cis' Counterfactual Instability score column")
    points = certs.copy()
    points["normalized_cic"] = normalize_cic(points["cis"])
    points["counterfactual_stability"] = 1.0 - points["normalized_cic"]
    threshold = float(points["counterfactual_stability"].median()) if stability_threshold is None else float(stability_threshold)
    points["confidence_threshold"] = confidence_threshold
    points["stability_threshold"] = threshold
    points["quadrant"] = [
        quadrant_label(conf, stab, confidence_threshold, threshold)
        for conf, stab in zip(points["confidence"], points["counterfactual_stability"])
    ]
    group_cols = [c for c in ["task", "regime", "quadrant"] if c in points.columns]
    quadrants = (
        points.groupby(group_cols, dropna=False)
        .agg(
            count=("failure", "size"),
            failure_count=("failure", "sum"),
            failure_rate=("failure", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_counterfactual_stability=("counterfactual_stability", "mean"),
            mean_cic=("cis", "mean"),
        )
        .reset_index()
        if group_cols
        else pd.DataFrame()
    )
    summary_cols = [c for c in ["regime", "quadrant"] if c in points.columns]
    summary = (
        points.groupby(summary_cols, dropna=False)
        .agg(
            count=("failure", "size"),
            failure_count=("failure", "sum"),
            failure_rate=("failure", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_counterfactual_stability=("counterfactual_stability", "mean"),
        )
        .reset_index()
        if summary_cols
        else pd.DataFrame()
    )
    return points, quadrants, summary


def _scatter(points: pd.DataFrame, regime: str, path: Path, confidence_threshold: float, stability_threshold: float) -> None:
    subset = points[points["regime"] == regime] if "regime" in points else points
    plt.figure(figsize=(6.0, 4.8))
    if subset.empty:
        plt.text(0.5, 0.5, f"No {regime} rows found", ha="center", va="center")
        plt.axis("off")
    else:
        colors = np.where(subset["failure"].astype(int) == 1, "#d55e00", "#0072b2")
        plt.scatter(subset["confidence"], subset["counterfactual_stability"], c=colors, alpha=0.72, s=28, edgecolors="none")
        plt.axvline(confidence_threshold, color="0.25", linestyle="--", linewidth=1)
        plt.axhline(stability_threshold, color="0.25", linestyle="--", linewidth=1)
        plt.text(0.98, 0.04, "Dangerous\nshortcut reliance", transform=plt.gca().transAxes, ha="right", va="bottom", fontsize=9)
        plt.xlabel("Confidence")
        plt.ylabel("Counterfactual stability (1 - normalized CIC)")
        plt.title(regime.replace("-", " ").title())
        plt.xlim(0, 1.02)
        plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _bar_failure_rates(quadrants: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8.2, 4.6))
    if quadrants.empty:
        plt.text(0.5, 0.5, "No quadrant rows found", ha="center", va="center")
        plt.axis("off")
    else:
        rates = quadrants.groupby("quadrant")["failure_rate"].mean().sort_values(ascending=False)
        plt.bar(rates.index, rates.values, color="#cc6677")
        plt.ylim(0, 1)
        plt.ylabel("Failure rate")
        plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _bar_counts(quadrants: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9.0, 4.8))
    if quadrants.empty or "regime" not in quadrants:
        plt.text(0.5, 0.5, "No quadrant rows found", ha="center", va="center")
        plt.axis("off")
    else:
        pivot = quadrants.pivot_table(index="regime", columns="quadrant", values="count", aggfunc="sum", fill_value=0)
        bottom = np.zeros(len(pivot))
        x = np.arange(len(pivot))
        for col in pivot.columns:
            plt.bar(x, pivot[col].to_numpy(), bottom=bottom, label=col)
            bottom += pivot[col].to_numpy()
        plt.xticks(x, pivot.index, rotation=15, ha="right")
        plt.ylabel("Examples")
        plt.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def run(results_dir: str | Path = "results", confidence_threshold: float = 0.8, stability_threshold: float | None = None) -> dict[str, Path]:
    root = Path(results_dir)
    cert_path = root / "final_validation" / "final_validation_certificates.csv"
    real_path = root / "real_model_validation" / "real_model_certificates.csv"
    if cert_path.exists():
        certs = pd.read_csv(cert_path)
        if real_path.exists():
            certs = pd.concat([certs, pd.read_csv(real_path)], ignore_index=True, sort=False)
    elif real_path.exists():
        certs = pd.read_csv(real_path)
    else:
        raise FileNotFoundError(f"missing final validation certificates: {cert_path}")
    out_dir = ensure_dir(root / "reliability_plane")
    plot_dir = ensure_dir(out_dir / "plots")
    points, quadrants, summary = build_reliability_plane(certs, confidence_threshold, stability_threshold)
    threshold = float(points["stability_threshold"].iloc[0]) if len(points) else (0.5 if stability_threshold is None else stability_threshold)
    points.to_csv(out_dir / "reliability_plane_points.csv", index=False)
    quadrants.to_csv(out_dir / "reliability_plane_quadrants.csv", index=False)
    summary.to_csv(out_dir / "reliability_plane_regime_summary.csv", index=False)
    _scatter(points, "confident-wrong", plot_dir / "reliability_plane_confident_wrong.png", confidence_threshold, threshold)
    _scatter(points, "confidence-solvable", plot_dir / "reliability_plane_confidence_solvable.png", confidence_threshold, threshold)
    _bar_failure_rates(quadrants, plot_dir / "quadrant_failure_rates.png")
    _bar_counts(quadrants, plot_dir / "quadrant_counts_by_regime.png")
    dangerous = quadrants[quadrants["quadrant"] == "Dangerous shortcut reliance"] if len(quadrants) else pd.DataFrame()
    text = [
        "# Reliability Plane Summary",
        "",
        f"Confidence threshold: `{confidence_threshold:.3f}`.",
        f"Stability threshold: `{threshold:.3f}`.",
        "",
        "The dangerous quadrant is high confidence plus low counterfactual stability.",
        "",
        "## Quadrant Counts And Failure Rates",
        "",
        _markdown_table(quadrants) if len(quadrants) else "No quadrant rows found.",
        "",
        "## Dangerous Quadrant",
        "",
        _markdown_table(dangerous) if len(dangerous) else "No dangerous quadrant rows found.",
        "",
    ]
    (out_dir / "reliability_plane_summary.md").write_text("\n".join(text), encoding="utf-8")
    return {
        "points": out_dir / "reliability_plane_points.csv",
        "quadrants": out_dir / "reliability_plane_quadrants.csv",
        "summary": out_dir / "reliability_plane_summary.md",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    parser.add_argument("--stability_threshold", type=float, default=None)
    args = parser.parse_args()
    print(run(args.results_dir, args.confidence_threshold, args.stability_threshold))


if __name__ == "__main__":
    main()
