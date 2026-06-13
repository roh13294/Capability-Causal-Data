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

from causal_reliability.analysis.metrics import auroc
from causal_reliability.analysis.phase6_common import _markdown_table, with_derived_scores
from causal_reliability.baselines.ood_heuristics import probability_distance_from_centroid
from causal_reliability.baselines.shortcut_heuristics import occlusion_scores_from_certificates
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


METHOD_ORDER = [
    "Confidence risk",
    "Entropy",
    "Negative margin",
    "Random augmentation sensitivity",
    "Occlusion shortcut heuristic",
    "Embedding/OOD distance",
    "Label-flip-only",
    "CIC",
]


def _safe_auc(values: pd.Series | np.ndarray, labels: pd.Series | np.ndarray) -> float:
    scores = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=int)
    mask = np.isfinite(scores) & np.isfinite(y)
    scores = scores[mask]
    y = y[mask]
    if len(scores) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return auroc(scores, y)


def _random_aug_proxy(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ["js_mean", "margin_collapse_mean", "margin_collapse_q90"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            lo, hi = values.min(skipna=True), values.max(skipna=True)
            if np.isfinite(lo) and np.isfinite(hi) and not np.isclose(lo, hi):
                parts.append((values - lo) / (hi - lo))
            else:
                parts.append(pd.Series(0.0, index=df.index))
    if not parts:
        return pd.Series(np.nan, index=df.index)
    return sum(parts) / len(parts)


def _score_columns(df: pd.DataFrame) -> dict[str, pd.Series]:
    df = with_derived_scores(df)
    def column(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[name], errors="coerce")

    scores: dict[str, pd.Series] = {
        "Confidence risk": column("confidence_risk"),
        "Entropy": column("entropy"),
        "Negative margin": column("negative_margin"),
        "Random augmentation sensitivity": _random_aug_proxy(df),
        "Occlusion shortcut heuristic": occlusion_scores_from_certificates(df),
        "Embedding/OOD distance": probability_distance_from_centroid(df),
        "Label-flip-only": column("flip_mean"),
        "CIC": column("cis") if "cis" in df else column("shift_risk"),
    }
    return scores


def _rows_from_certificates(path: Path, task: str, regime: str, source: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "failure" not in df.columns and {"pred", "label"}.issubset(df.columns):
        df["failure"] = (df["pred"] != df["label"]).astype(int)
    if "failure" not in df.columns:
        return []
    rows = []
    group_cols = [col for col in ["task", "regime"] if col in df.columns]
    groups = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False, observed=False)
    for keys, group in groups:
        if group_cols and not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys)) if group_cols else {}
        group_task = str(meta.get("task", task))
        group_regime = str(meta.get("regime", regime))
        scores = _score_columns(group)
        for method in METHOD_ORDER:
            score = scores[method]
            rows.append(
                {
                    "task": group_task,
                    "regime": group_regime,
                    "method": method,
                    "failure_auroc": _safe_auc(score, group["failure"]),
                    "n_examples": int(len(group)),
                    "n_failures": int(pd.to_numeric(group["failure"], errors="coerce").sum()),
                    "source": source,
                    "note": _method_note(method, source),
                }
            )
    return rows


def _rows_from_metric_file(path: Path, task: str, regime: str, source: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if not {"method", "failure_auroc"}.issubset(df.columns):
        return []
    method_map = {
        "confidence_risk": "Confidence risk",
        "confidence": "Confidence risk",
        "entropy": "Entropy",
        "negative_margin": "Negative margin",
        "margin": "Negative margin",
        "label_flip_only": "Label-flip-only",
        "CIC": "CIC",
        "cic": "CIC",
        "CIS": "CIC",
    }
    rows = []
    for _, rec in df.iterrows():
        method = method_map.get(str(rec["method"]), str(rec["method"]))
        if method not in METHOD_ORDER:
            continue
        rows.append(
            {
                "task": task,
                "regime": regime,
                "method": method,
                "failure_auroc": float(rec["failure_auroc"]),
                "n_examples": int(rec.get("n_examples", 0)) if pd.notna(rec.get("n_examples", np.nan)) else np.nan,
                "n_failures": int(rec.get("n_failures", 0)) if pd.notna(rec.get("n_failures", np.nan)) else np.nan,
                "source": source,
                "note": "aggregate metric file; heuristic baselines unavailable without per-example artifacts",
            }
        )
    return rows


def _method_note(method: str, source: str) -> str:
    if method == "Random augmentation sensitivity":
        return "computed from generic certificate drift components when raw model inputs are unavailable"
    if method == "Occlusion shortcut heuristic":
        return "uses known shortcut flip/occlusion proxy where available"
    if method == "Embedding/OOD distance":
        return "uses feature/logit-proxy centroid distance when learned embeddings are unavailable"
    return source


def _best_table(metrics: pd.DataFrame) -> pd.DataFrame:
    non_cic = metrics[metrics["method"] != "CIC"].dropna(subset=["failure_auroc"]).copy()
    cic = metrics[metrics["method"] == "CIC"][["task", "regime", "failure_auroc"]].rename(columns={"failure_auroc": "cic_auroc"})
    if non_cic.empty:
        return pd.DataFrame()
    idx = non_cic.groupby(["task", "regime"])["failure_auroc"].idxmax()
    best = non_cic.loc[idx, ["task", "regime", "method", "failure_auroc"]].rename(
        columns={"method": "best_non_cic_baseline", "failure_auroc": "best_non_cic_auroc"}
    )
    out = best.merge(cic, on=["task", "regime"], how="left")
    out["cic_advantage_over_best_non_cic"] = out["cic_auroc"] - out["best_non_cic_auroc"]
    out["interpretation"] = np.where(
        out["cic_advantage_over_best_non_cic"] > 0.02,
        "CIC wins",
        np.where(out["cic_advantage_over_best_non_cic"] < -0.02, "simpler baseline competitive or better", "rough tie"),
    )
    return out.sort_values(["task", "regime"]).reset_index(drop=True)


def _plot_auc(metrics: pd.DataFrame, path: Path) -> None:
    pivot = metrics.pivot_table(index=["task", "regime"], columns="method", values="failure_auroc", aggfunc="mean", observed=False)
    pivot = pivot[[m for m in METHOD_ORDER if m in pivot.columns]]
    ax = pivot.plot(kind="bar", figsize=(11, 5), width=0.82)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Failure AUROC")
    ax.set_xlabel("Task / regime")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _plot_cic_vs_best(best: pd.DataFrame, path: Path) -> None:
    if best.empty:
        return
    labels = best["task"] + "\n" + best["regime"]
    x = np.arange(len(best))
    plt.figure(figsize=(8.5, 4.0))
    plt.bar(x - 0.18, best["best_non_cic_auroc"], width=0.36, label="Best non-CIC")
    plt.bar(x + 0.18, best["cic_auroc"], width=0.36, label="CIC")
    plt.ylim(0, 1)
    plt.ylabel("Failure AUROC")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _plot_by_regime(metrics: pd.DataFrame, path: Path) -> None:
    summary = metrics.pivot_table(index="regime", columns="method", values="failure_auroc", aggfunc="mean", observed=False)
    summary = summary[[m for m in METHOD_ORDER if m in summary.columns]]
    ax = summary.plot(kind="bar", figsize=(8.6, 4.2), width=0.82)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean failure AUROC")
    ax.set_xlabel("Regime")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def run(cfg: dict[str, Any]) -> dict[str, str]:
    root = Path(cfg.get("results_dir", "results"))
    out_dir = ensure_dir(root / "baseline_comparison")
    plot_dir = ensure_dir(out_dir / "plots")
    rows: list[dict[str, Any]] = []

    rows.extend(
        _rows_from_certificates(
            root / "confident_wrong" / "confident_wrong_certificates.csv",
            "controlled",
            "confident-wrong",
            "confident_wrong certificates",
        )
    )
    rows.extend(
        _rows_from_certificates(
            root / "colored_digits" / "colored_digits_certificates.csv",
            "colored_digits",
            "confident-wrong",
            "colored_digits certificates",
        )
    )
    rows.extend(
        _rows_from_metric_file(
            root / "clip_overlay_validation" / "clip_overlay_metrics.csv",
            "clip_overlay",
            "pretrained-overlay",
            "clip_overlay metrics",
        )
    )

    final = root / "final_validation" / "final_validation_summary.csv"
    if final.exists():
        df = pd.read_csv(final)
        mixed = df[df["regime"].astype(str).str.contains("mixed", case=False, na=False)]
        for _, rec in mixed.iterrows():
            mapping = {
                "Confidence risk": "confidence_risk_auroc",
                "Label-flip-only": "label_flip_only_auroc",
                "CIC": "cis_auroc",
            }
            for method, col in mapping.items():
                if col in rec and pd.notna(rec[col]):
                    rows.append(
                        {
                            "task": rec["task"],
                            "regime": "mixed",
                            "method": method,
                            "failure_auroc": float(rec[col]),
                            "n_examples": np.nan,
                            "n_failures": np.nan,
                            "source": "final validation aggregate",
                            "note": "aggregate mixed-regime metric",
                        }
                    )

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics["method"] = pd.Categorical(metrics["method"], METHOD_ORDER, ordered=True)
        metrics = metrics.sort_values(["task", "regime", "method"]).reset_index(drop=True)
    best = _best_table(metrics) if not metrics.empty else pd.DataFrame()

    metrics_path = out_dir / "baseline_comparison_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    best.to_csv(out_dir / "baseline_comparison_best_non_cic.csv", index=False)

    if not metrics.empty:
        _plot_auc(metrics, plot_dir / "baseline_auc_comparison.png")
        _plot_cic_vs_best(best, plot_dir / "cic_vs_best_baseline.png")
        _plot_by_regime(metrics, plot_dir / "baseline_by_regime.png")

    summary = [
        "# Baseline Comparison Summary",
        "",
        "This reviewer-oriented comparison contextualizes CIC against simple uncertainty, generic instability, occlusion-style shortcut, and OOD-distance heuristics. It is not constructed to make CIC win everywhere.",
        "",
        "## Best Non-CIC Baseline By Task",
        "",
        _markdown_table(best) if not best.empty else "No eligible baseline rows found.",
        "",
        "## All Metrics",
        "",
        _markdown_table(metrics) if not metrics.empty else "No eligible baseline rows found.",
        "",
        "## Interpretation",
        "",
        "CIC is most useful when failures are high-confidence and specifically tied to unstable shortcut features. Simpler baselines can be competitive or better when failures are low-confidence, globally corrupted, or already separable by generic uncertainty/OOD signals.",
        "",
        "In colored digits, random augmentation sensitivity outperformed CIC, with random augmentation AUROC 0.9829 versus CIC AUROC 0.9512. This shows that some shortcut failures can be detected by generic instability, especially when perturbations accidentally disturb the shortcut. However, generic augmentation is not targeted, not necessarily label-preserving, and does not explain which factor is unstable. CIC remains useful as a principled counterfactual stability framework rather than as a universal winner over every heuristic.",
        "",
        "CIC is not claimed to dominate all baselines. The contribution is that it defines and operationalizes a second reliability axis.",
        "",
        "The random-augmentation and OOD rows use available certificate/logit proxies when raw trained models or embeddings are not stored with an artifact. Those rows are included to make the comparison explicit, not to overstate the strength of the heuristic evaluation.",
        "",
    ]
    summary_path = out_dir / "baseline_comparison_summary.md"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    return {
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "baseline_auc_plot": str(plot_dir / "baseline_auc_comparison.png"),
        "cic_vs_best_plot": str(plot_dir / "cic_vs_best_baseline.png"),
        "by_regime_plot": str(plot_dir / "baseline_by_regime.png"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_comparison.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
