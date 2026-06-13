from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from causal_reliability.analysis.phase6_common import group_frame, read_certificate_files, safe_auroc, write_markdown_table
from causal_reliability.utils.io import ensure_dir

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMPONENTS = {
    "mean_margin_collapse": "mean_margin_collapse",
    "tail_margin_collapse": "tail_margin_collapse",
    "JS": "JS",
    "label_flip": "label_flip",
    "full ShiftRisk": "shift_risk",
    "1-CR": "one_minus_cr",
    "confidence risk": "confidence_risk",
    "entropy": "entropy",
    "negative margin": "negative_margin",
    "CIS": "cis",
}


def _score_rows(meta: dict[str, object], df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    failure = df["failure"].astype(int)
    for name, col in COMPONENTS.items():
        if col not in df.columns:
            continue
        scores = pd.to_numeric(df[col], errors="coerce")
        auc = safe_auroc(scores, failure)
        inv_auc = safe_auroc(-scores, failure)
        rows.append(
            {
                **meta,
                "component": name,
                "mean_score_failures": float(scores[failure == 1].mean()) if (failure == 1).any() else float("nan"),
                "mean_score_correct": float(scores[failure == 0].mean()) if (failure == 0).any() else float("nan"),
                "auroc": auc,
                "inverted_auroc": inv_auc,
                "inverted_flag": bool(np.isfinite(inv_auc) and np.isfinite(auc) and inv_auc > auc + 0.1),
            }
        )
    return rows


def build_summary(results_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path, df in read_certificate_files(results_dir):
        for meta, group in group_frame(path, df, results_dir):
            rows.extend(_score_rows(meta, group))
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    sort_cols = [col for col in ["task", "shift_type", "partial_flip_fraction", "component", "source_file"] if col in summary.columns]
    return summary.sort_values(sort_cols).reset_index(drop=True)


def _plot_distributions(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    subset = summary[summary["component"].isin(["full ShiftRisk", "CIS", "confidence risk", "label_flip"])]
    if subset.empty:
        return
    x = np.arange(len(subset))
    plt.figure(figsize=(max(6, 0.25 * len(subset)), 4))
    plt.scatter(x, subset["mean_score_correct"], label="correct", s=18)
    plt.scatter(x, subset["mean_score_failures"], label="failures", s=18)
    plt.xticks(x, subset["component"].astype(str), rotation=60, ha="right", fontsize=7)
    plt.ylabel("mean score")
    plt.legend(fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_heatmap(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    index_cols = [col for col in ["task", "partial_flip_fraction", "source_file"] if col in summary.columns]
    if not index_cols:
        index_cols = ["source_file"]
    matrix = summary.pivot_table(index=index_cols, columns="component", values="auroc", aggfunc="mean")
    if matrix.empty:
        return
    plt.figure(figsize=(max(6, 0.55 * matrix.shape[1]), max(3.5, 0.24 * matrix.shape[0])))
    im = plt.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    plt.xticks(range(matrix.shape[1]), matrix.columns, rotation=45, ha="right", fontsize=7)
    plt.yticks(range(matrix.shape[0]), [" / ".join(map(str, idx if isinstance(idx, tuple) else (idx,))) for idx in matrix.index], fontsize=6)
    plt.colorbar(im, label="AUROC")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_flags(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    counts = summary.groupby("component", observed=False)["inverted_flag"].sum().sort_values(ascending=False)
    plt.figure(figsize=(6, 3.6))
    plt.bar(counts.index.astype(str), counts.to_numpy())
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.ylabel("inverted flags")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_outputs(summary: pd.DataFrame, results_dir: str | Path) -> None:
    out_dir = ensure_dir(Path(results_dir) / "score_diagnosis")
    plot_dir = ensure_dir(out_dir / "plots")
    summary.to_csv(out_dir / "component_direction_summary.csv", index=False)
    write_markdown_table(summary, out_dir / "component_direction_summary.md", "Component Direction Summary")
    _plot_distributions(summary, plot_dir / "score_distributions_by_correctness.png")
    _plot_heatmap(summary, plot_dir / "component_auc_heatmap.png")
    _plot_flags(summary, plot_dir / "inverted_score_flags.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    summary = build_summary(args.results_dir)
    save_outputs(summary, args.results_dir)
    print(summary.to_string(index=False) if not summary.empty else "No eligible certificate files found.")


if __name__ == "__main__":
    main()
