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


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _empty_panel(ax, title: str, path: Path) -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(0.5, 0.5, f"Missing input:\n{path}", ha="center", va="center", fontsize=9)
    ax.set_axis_off()


def _panel_a(ax, rankings: pd.DataFrame) -> None:
    ax.set_title("A. Candidate Search", loc="left", fontweight="bold")
    if rankings.empty:
        ax.text(0.5, 0.5, "Run unknown shortcut discovery first.", ha="center", va="center")
        ax.set_axis_off()
        return
    type_scores = rankings.groupby("candidate_type")["full_unknown_shortcut_score"].mean().sort_values(ascending=True).tail(8)
    ax.barh(type_scores.index, type_scores.values, color="#4c78a8")
    ax.set_xlabel("Mean discovery score")


def _panel_b(ax, rankings: pd.DataFrame) -> None:
    ax.set_title("B. Discovery Ranking", loc="left", fontweight="bold")
    if rankings.empty:
        ax.text(0.5, 0.5, "Run unknown shortcut discovery first.", ha="center", va="center")
        ax.set_axis_off()
        return
    groups = rankings.copy()
    groups["group"] = groups["ground_truth_factor_group"].replace({"true_shortcut": "true shortcut", "causal": "causal", "irrelevant": "noise/irrelevant"})
    order = ["true shortcut", "causal", "noise/irrelevant", "corruption", "missing"]
    means = groups.groupby("group")["full_unknown_shortcut_score"].mean().reindex([g for g in order if g in set(groups["group"])])
    ax.bar(means.index, means.values, color=["#d55e00", "#cc79a7", "#4c78a8", "#999999", "#777777"][: len(means)])
    ax.set_ylabel("Mean discovery score")
    ax.tick_params(axis="x", rotation=25)


def _panel_c(ax, metrics: pd.DataFrame) -> None:
    ax.set_title("C. Discovered CIC", loc="left", fontweight="bold")
    if metrics.empty:
        ax.text(0.5, 0.5, "Run discovered CIC first.", ha="center", va="center")
        ax.set_axis_off()
        return
    columns = ["confidence_auroc", "oracle_cic_auroc", "discovered_cic_top1_auroc", "discovered_cic_top3_auroc"]
    labels = ["confidence", "oracle CIC", "disc. top-1", "disc. top-3"]
    x = np.arange(len(metrics))
    width = 0.18
    for i, (col, label) in enumerate(zip(columns, labels)):
        ax.bar(x + (i - 1.5) * width, metrics[col], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["task"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Failure AUROC")
    ax.legend(fontsize=7)


def _panel_d(ax, failure: pd.DataFrame) -> None:
    ax.set_title("D. Failure Controls", loc="left", fontweight="bold")
    if failure.empty:
        ax.text(0.5, 0.5, "Run failure cases first.", ha="center", va="center")
        ax.set_axis_off()
        return
    colors = failure["top3_true_shortcut_hit"].map({True: "#4c78a8", False: "#999999"})
    ax.bar(failure["case"], failure["dominance_gap"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Top score minus second")
    ax.tick_params(axis="x", rotation=35)


def run(results_dir: str | Path = "results") -> dict[str, Path]:
    root = Path(results_dir)
    rankings = _read(root / "unknown_shortcut_discovery" / "unknown_shortcut_rankings.csv")
    discovered = _read(root / "discovered_cic" / "discovered_cic_metrics.csv")
    failure = _read(root / "unknown_shortcut_discovery" / "failure_cases" / "failure_case_metrics.csv")
    ensure_dir(root)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    _panel_a(axes[0, 0], rankings)
    _panel_b(axes[0, 1], rankings)
    _panel_c(axes[1, 0], discovered)
    _panel_d(axes[1, 1], failure)
    fig.tight_layout()
    png = root / "moonshot_figure.png"
    pdf = root / "moonshot_figure.pdf"
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)

    caption = [
        "# Moonshot Figure Caption",
        "",
        "Candidate shortcut discovery pilot. Panel A ranks candidate intervention types by the candidate-shortcut score. Panel B compares true shortcut candidates against causal and noise/corruption candidates. Panel C tests whether CIC computed with discovered candidates from a finite candidate intervention class approaches oracle CIC and compares both to confidence. Panel D summarizes controls where discovery should fail or become cautious.",
        "",
        "This is an exploratory extension in controlled settings. It does not solve general causal discovery.",
        "",
    ]
    caption_path = root / "moonshot_figure_caption.md"
    caption_path.write_text("\n".join(caption), encoding="utf-8")
    return {"png": png, "pdf": pdf, "caption": caption_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    print(run(args.results_dir))


if __name__ == "__main__":
    main()
