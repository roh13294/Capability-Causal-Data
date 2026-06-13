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

from causal_reliability.utils.io import ensure_dir


def _panel_a(ax, results_dir: Path) -> None:
    path = results_dir / "final_validation" / "final_validation_summary.csv"
    if not path.exists():
        ax.text(0.5, 0.5, "Run final validation first", ha="center", va="center")
        ax.set_axis_off()
        return
    df = pd.read_csv(path)
    reg = df.groupby("regime")[["confidence_risk_auroc", "cis_auroc"]].mean().reindex(["confidence-solvable", "confident-wrong", "mixed"])
    x = np.arange(len(reg))
    width = 0.34
    conf_err = df.groupby("regime")["confidence_risk_auroc_std"].mean().reindex(reg.index) if "confidence_risk_auroc_std" in df else None
    cic_err = df.groupby("regime")["cis_auroc_std"].mean().reindex(reg.index) if "cis_auroc_std" in df else None
    ax.bar(x - width / 2, reg["confidence_risk_auroc"], width, yerr=conf_err, capsize=3, label="Confidence risk", color="#0072b2")
    ax.bar(x + width / 2, reg["cis_auroc"], width, yerr=cic_err, capsize=3, label="CIC", color="#d55e00")
    ax.set_xticks(x, ["confidence\nsolvable", "confident\nwrong", "mixed"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Failure AUROC")
    ax.set_title("A. Regime separation")
    ax.legend(fontsize=8)


def _panel_b(ax, results_dir: Path) -> None:
    path = results_dir / "reliability_plane" / "reliability_plane_points.csv"
    if not path.exists():
        ax.text(0.5, 0.5, "Run reliability plane first", ha="center", va="center")
        ax.set_axis_off()
        return
    points = pd.read_csv(path)
    subset = points[points["regime"] == "confident-wrong"] if "regime" in points else points
    colors = np.where(subset["failure"].astype(int) == 1, "#d55e00", "#0072b2")
    ax.scatter(subset["confidence"], subset["counterfactual_stability"], c=colors, alpha=0.70, s=24, edgecolors="none")
    cthr = float(points["confidence_threshold"].iloc[0]) if "confidence_threshold" in points and len(points) else 0.8
    sthr = float(points["stability_threshold"].iloc[0]) if "stability_threshold" in points and len(points) else 0.5
    ax.axvline(cthr, color="0.25", linestyle="--", linewidth=1)
    ax.axhline(sthr, color="0.25", linestyle="--", linewidth=1)
    ax.fill_between([cthr, 1.02], -0.02, sthr, color="#d55e00", alpha=0.12)
    ax.text(0.98, 0.05, "dangerous\nquadrant", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Counterfactual stability")
    ax.set_title("B. Two-axis reliability plane")


def _panel_c(ax, results_dir: Path) -> None:
    path = results_dir / "shortcut_discovery" / "shortcut_discovery_rankings.csv"
    if not path.exists():
        ax.text(0.5, 0.5, "Run shortcut discovery first", ha="center", va="center")
        ax.set_axis_off()
        return
    rankings = pd.read_csv(path)
    means = rankings.groupby("feature_type")["average_instability"].mean().reindex(["shortcut feature", "causal feature", "noise"])
    ax.bar(means.index, means.values, color=["#d55e00", "#0072b2", "#999999"])
    ax.set_ylabel("Average instability")
    ax.set_title("C. Shortcut discovery pilot")
    ax.tick_params(axis="x", rotation=18)


def _panel_d(ax, results_dir: Path) -> None:
    path = results_dir / "real_model_validation" / "real_model_metrics.csv"
    if not path.exists():
        ax.text(0.5, 0.5, "Real-model validation optional", ha="center", va="center")
        ax.set_axis_off()
        return
    metrics = pd.read_csv(path)
    subset = metrics[metrics["method"].isin(["confidence_risk", "CIC"])]
    ax.bar(subset["method"], subset["failure_auroc"], color=["#0072b2", "#d55e00"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Failure AUROC")
    ax.set_title("D. Real-model check")


def make_figure(results_dir: str | Path = "results") -> dict[str, Path]:
    root = Path(results_dir)
    ensure_dir(root)
    has_real = (root / "real_model_validation" / "real_model_metrics.csv").exists()
    fig, axes = plt.subplots(1, 4 if has_real else 3, figsize=(16.6 if has_real else 13.2, 4.2))
    _panel_a(axes[0], root)
    _panel_b(axes[1], root)
    _panel_c(axes[2], root)
    if has_real:
        _panel_d(axes[3], root)
    fig.tight_layout()
    png = root / "concept_figure.png"
    pdf = root / "concept_figure.pdf"
    caption = root / "concept_figure_caption.md"
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)
    caption.write_text(
        "Confidence and counterfactual stability measure different axes of reliability. "
        "Confidence identifies low-confidence failures, while counterfactual instability identifies "
        "high-confidence shortcut dependence. The optional real-model panel is supporting evidence from a controlled shortcut task, not proof of foundation-model generalization. Error bars show across-seed standard deviation where available.\n",
        encoding="utf-8",
    )
    return {"png": png, "pdf": pdf, "caption": caption}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    print(make_figure(args.results_dir))


if __name__ == "__main__":
    main()
