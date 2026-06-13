from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from causal_reliability.analysis.phase6_common import (
    group_frame,
    merge_label_flip,
    read_certificate_files,
    safe_auroc,
    save_bar,
    save_scatter,
    write_markdown_table,
)
from causal_reliability.utils.io import ensure_dir


def classify_regime(confidence_auroc: float, mean_failed_confidence: float, mean_correct_confidence: float, shifted_failure_rate: float) -> str:
    if confidence_auroc >= 0.75 and mean_failed_confidence < mean_correct_confidence:
        return "confidence-solvable"
    if confidence_auroc <= 0.55 and mean_failed_confidence >= 0.75 and shifted_failure_rate >= 0.2:
        return "confident-wrong"
    return "mixed"


def _row(meta: dict[str, object], df: pd.DataFrame) -> dict[str, object]:
    failure = df["failure"].astype(int)
    failed = df[failure == 1]
    correct = df[failure == 0]
    high_conf = df[df["confidence"] >= 0.8]
    confidence_risk = 1.0 - df["confidence"]
    confidence_auc = safe_auroc(confidence_risk, failure)
    shift_risk_auc = safe_auroc(df["shift_risk"], failure) if "shift_risk" in df.columns else float("nan")
    cis_auc = safe_auroc(df["cis"], failure) if "cis" in df.columns else float("nan")
    shifted_failure_rate = float(failure.mean()) if len(df) else float("nan")
    mean_failed = float(failed["confidence"].mean()) if len(failed) else float("nan")
    mean_correct = float(correct["confidence"].mean()) if len(correct) else float("nan")
    row = {
        **meta,
        "n_examples": int(len(df)),
        "shifted_failure_rate": shifted_failure_rate,
        "mean_confidence_failures": mean_failed,
        "mean_confidence_correct": mean_correct,
        "confidence_auroc": confidence_auc,
        "ShiftRisk_auroc": shift_risk_auc,
        "CIS_auroc": cis_auc,
        "high_confidence_failure_count": int(((df["confidence"] >= 0.8) & (failure == 1)).sum()),
        "high_confidence_shifted_accuracy": float(high_conf["correct"].mean()) if len(high_conf) and "correct" in high_conf.columns else float("nan"),
    }
    row["regime_label"] = classify_regime(confidence_auc, mean_failed, mean_correct, shifted_failure_rate)
    return row


def build_summary(results_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path, df in read_certificate_files(results_dir):
        for meta, group in group_frame(path, df, results_dir):
            if len(group) and "failure" in group.columns:
                rows.append(_row(meta, group))
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = merge_label_flip(summary, results_dir)
    sort_cols = [col for col in ["task", "shift_type", "partial_flip_fraction", "experiment", "source_file"] if col in summary.columns]
    return summary.sort_values(sort_cols).reset_index(drop=True)


def save_outputs(summary: pd.DataFrame, results_dir: str | Path) -> None:
    out_dir = ensure_dir(Path(results_dir) / "regime_analysis")
    plot_dir = ensure_dir(out_dir / "plots")
    summary.to_csv(out_dir / "regime_summary.csv", index=False)
    write_markdown_table(summary, out_dir / "regime_summary.md", "Regime Classification Summary")
    save_scatter(summary, plot_dir / "confidence_failure_scatter.png", "mean_confidence_failures", "mean_confidence_correct", "regime_label")
    if not summary.empty:
        counts = summary["regime_label"].value_counts().rename_axis("regime_label").reset_index(name="count")
        save_bar(counts, plot_dir / "regime_bars.png", "regime_label", "count")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    summary = build_summary(args.results_dir)
    save_outputs(summary, args.results_dir)
    print(summary.to_string(index=False) if not summary.empty else "No eligible certificate files found.")


if __name__ == "__main__":
    main()
