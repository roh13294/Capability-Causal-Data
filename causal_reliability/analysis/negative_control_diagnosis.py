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

CURRENT_CONTROLS = {
    "true_counterfactual",
    "random_labels",
    "irrelevant_counterfactual",
    "shuffled_any",
    "shuffled_within_class",
    "shuffled_same_shortcut",
    "shuffled_matched_confidence",
    "random_intervention_direction",
}


def _read_controls(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((results_dir / "negative_controls").glob("*_certificates.csv")):
        df = pd.read_csv(path)
        if "control" not in df:
            df.insert(0, "control", path.stem.replace("_certificates", ""))
        df = df[df["control"].isin(CURRENT_CONTROLS)]
        if df.empty:
            continue
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _corr(a: pd.Series, b: pd.Series) -> float:
    if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return float("nan")
    return float(a.corr(b, method="spearman"))


def _plot_hist(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(5.2, 3.4))
    for control in ("true_counterfactual", "shuffled_any", "shuffled_within_class", "shuffled_same_shortcut", "shuffled_matched_confidence"):
        sub = df[df["control"] == control]
        if len(sub):
            plt.hist(sub["shift_risk"], bins=16, alpha=0.45, label=control)
    plt.xlabel("ShiftRisk")
    plt.ylabel("examples")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_scatter(df: pd.DataFrame, x: str, y: str, path: Path) -> None:
    plt.figure(figsize=(4.7, 3.5))
    plt.scatter(df[x], df[y], c=df.get("failure", 0), cmap="coolwarm", s=18, alpha=0.75)
    plt.xlabel(x.replace("_", " "))
    plt.ylabel(y.replace("_", " "))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_components(df: pd.DataFrame, path: Path) -> None:
    cols = ["margin_collapse_mean", "margin_collapse_q90", "js_mean", "flip_mean"]
    means = df.groupby("control")[cols].mean(numeric_only=True)
    means.plot(kind="bar", figsize=(7.5, 3.8))
    plt.ylabel("mean component")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def build_diagnosis(results_dir: str | Path = "results") -> pd.DataFrame:
    results_path = Path(results_dir)
    df = _read_controls(results_path)
    rows = []
    if df.empty:
        return pd.DataFrame()
    true = df[df["control"] == "true_counterfactual"].reset_index(drop=True)
    true_risk = true["shift_risk"] if len(true) else pd.Series(dtype=float)
    for control, sub in df.groupby("control"):
        sub = sub.reset_index(drop=True)
        n = min(len(sub), len(true_risk))
        rows.append(
            {
                "control": control,
                "n": int(len(sub)),
                "mean_shift_risk": float(sub["shift_risk"].mean()),
                "mean_margin_collapse": float(sub["margin_collapse_mean"].mean()),
                "mean_confidence": float(sub["confidence"].mean()),
                "failure_rate": float(sub["failure"].mean()),
                "corr_with_confidence": _corr(sub["shift_risk"], sub["confidence"]),
                "corr_with_margin": _corr(sub["shift_risk"], sub["margin"]),
                "corr_with_input_difficulty": _corr(sub["shift_risk"], 1.0 - sub["confidence"]),
                "corr_with_shifted_failure": _corr(sub["shift_risk"], sub["failure"]),
                "corr_with_true_shift_risk": _corr(sub["shift_risk"].iloc[:n], true_risk.iloc[:n]) if n else float("nan"),
                "accidental_label_change_rate": 0.0 if "within_class" in control or "same_shortcut" in control else float("nan"),
                "mean_x_to_shuffled_distance": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def save_outputs(summary: pd.DataFrame, results_dir: str | Path = "results") -> None:
    results_path = Path(results_dir)
    out_dir = ensure_dir(results_path / "negative_control_diagnosis")
    plot_dir = ensure_dir(out_dir / "plots")
    summary.to_csv(out_dir / "diagnosis_summary.csv", index=False)
    lines = ["# Negative-Control Diagnosis", ""]
    if summary.empty:
        lines.append("No negative-control certificate files found.")
    else:
        cols = list(summary.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in summary.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                vals.append(f"{val:.3f}" if isinstance(val, float) and np.isfinite(val) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
    (out_dir / "diagnosis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    df = _read_controls(results_path)
    if not df.empty:
        _plot_hist(df, plot_dir / "true_vs_shuffled_shift_risk.png")
        shuffled = df[df["control"].str.contains("shuffled", na=False)]
        if len(shuffled):
            _plot_scatter(shuffled, "confidence", "shift_risk", plot_dir / "shuffled_risk_vs_confidence.png")
        true = df[df["control"] == "true_counterfactual"].reset_index(drop=True)
        any_shuf = df[df["control"] == "shuffled_any"].reset_index(drop=True)
        if len(true) and len(any_shuf):
            n = min(len(true), len(any_shuf))
            paired = pd.DataFrame({"true_shift_risk": true["shift_risk"].iloc[:n], "shuffled_shift_risk": any_shuf["shift_risk"].iloc[:n], "failure": true["failure"].iloc[:n]})
            _plot_scatter(paired, "true_shift_risk", "shuffled_shift_risk", plot_dir / "shuffled_risk_vs_true_risk.png")
        _plot_components(df, plot_dir / "margin_collapse_components.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    summary = build_diagnosis(args.results_dir)
    save_outputs(summary, args.results_dir)
    print(summary)


if __name__ == "__main__":
    main()
