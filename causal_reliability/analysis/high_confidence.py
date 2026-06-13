from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from causal_reliability.analysis.phase6_common import group_frame, read_certificate_files, safe_auroc, write_markdown_table
from causal_reliability.utils.io import ensure_dir

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THRESHOLDS = (0.6, 0.7, 0.8, 0.9)


def _row(meta: dict[str, object], df: pd.DataFrame, threshold: float) -> dict[str, object]:
    subset = df[df["confidence"] >= threshold]
    failure = subset["failure"].astype(int) if len(subset) else pd.Series(dtype=int)
    row = {
        **meta,
        "confidence_threshold": threshold,
        "n_examples": int(len(subset)),
        "n_failures": int(failure.sum()) if len(subset) else 0,
        "failure_rate": float(failure.mean()) if len(subset) else float("nan"),
        "confidence_auroc": safe_auroc(1.0 - subset["confidence"], failure) if len(subset) else float("nan"),
        "ShiftRisk_auroc": safe_auroc(subset["shift_risk"], failure) if len(subset) and "shift_risk" in subset.columns else float("nan"),
        "CIS_auroc": safe_auroc(subset["cis"], failure) if len(subset) and "cis" in subset.columns else float("nan"),
        "label_flip_only_auroc": safe_auroc(subset["flip_mean"], failure) if len(subset) and "flip_mean" in subset.columns else float("nan"),
        "calibrated_cis_auroc": safe_auroc(subset["calibrated_cis_score"], failure) if len(subset) and "calibrated_cis_score" in subset.columns else float("nan"),
    }
    return row


def build_summary(results_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path, df in read_certificate_files(results_dir):
        for meta, group in group_frame(path, df, results_dir):
            for threshold in THRESHOLDS:
                rows.append(_row(meta, group, threshold))
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    sort_cols = [col for col in ["task", "shift_type", "partial_flip_fraction", "confidence_threshold", "source_file"] if col in summary.columns]
    return summary.sort_values(sort_cols).reset_index(drop=True)


def _plot_auc(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    methods = ["confidence_auroc", "ShiftRisk_auroc", "CIS_auroc", "label_flip_only_auroc", "calibrated_cis_auroc"]
    mean = summary.groupby("confidence_threshold", observed=False)[methods].mean(numeric_only=True)
    plt.figure(figsize=(5.6, 3.6))
    for method in methods:
        if method in mean.columns:
            plt.plot(mean.index, mean[method], marker="o", label=method.replace("_auroc", ""))
    plt.ylim(0, 1)
    plt.xlabel("confidence threshold")
    plt.ylabel("failure AUROC")
    plt.legend(fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_failure(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    mean = summary.groupby("confidence_threshold", observed=False)["failure_rate"].mean()
    plt.figure(figsize=(5.2, 3.4))
    plt.plot(mean.index, mean.to_numpy(), marker="o")
    plt.xlabel("confidence threshold")
    plt.ylabel("failure rate")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_outputs(summary: pd.DataFrame, results_dir: str | Path) -> None:
    out_dir = ensure_dir(Path(results_dir) / "high_confidence_analysis")
    plot_dir = ensure_dir(out_dir / "plots")
    summary.to_csv(out_dir / "high_confidence_summary.csv", index=False)
    write_markdown_table(summary, out_dir / "high_confidence_summary.md", "High-Confidence Subset Summary")
    _plot_auc(summary, plot_dir / "high_confidence_auc_by_threshold.png")
    _plot_failure(summary, plot_dir / "high_confidence_failure_rate.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    summary = build_summary(args.results_dir)
    save_outputs(summary, args.results_dir)
    print(summary.to_string(index=False) if not summary.empty else "No eligible certificate files found.")


if __name__ == "__main__":
    main()
