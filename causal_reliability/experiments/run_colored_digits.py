from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_reliability.analysis.metrics import auroc
from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.analysis.statistics import bootstrap_metric_ci, defined_auroc_reason, format_ci, paired_bootstrap_auc_diff
from causal_reliability.data.colored_digits import dataset_info, make_colored_digits_task
from causal_reliability.experiments.common import run_task
from causal_reliability.experiments.stress_utils import score_map
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


def _binary_entropy(confidence: pd.Series) -> np.ndarray:
    p = confidence.clip(1e-8, 1 - 1e-8).to_numpy(dtype=float)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _summarize_certificates(cert_df: pd.DataFrame, summary: dict[str, Any], source: str) -> pd.DataFrame:
    if "failure" not in cert_df:
        cert_df = cert_df.copy()
        cert_df["failure"] = (cert_df["pred"] != cert_df["label"]).astype(int)
    scores = {
        "confidence": 1.0 - cert_df["confidence"].to_numpy(dtype=float),
        "entropy": _binary_entropy(cert_df["confidence"]),
        "margin": -cert_df["margin"].to_numpy(dtype=float),
        "old_shift_risk": cert_df["shift_risk"].to_numpy(dtype=float),
        "label_flip_only": cert_df["flip_mean"].to_numpy(dtype=float) if "flip_mean" in cert_df else cert_df["cis"].to_numpy(dtype=float),
        "cic": cert_df["cis"].to_numpy(dtype=float),
    }
    failure = cert_df["failure"].to_numpy(dtype=int)
    row: dict[str, Any] = {
        "task": "colored_digits",
        "dataset_source": source,
        "seed_count": 1,
        "id_accuracy": summary.get("id_accuracy", np.nan),
        "shifted_accuracy": summary.get("shifted_accuracy", np.nan),
        "failure_count": int(failure.sum()),
        "correct_count": int((1 - failure).sum()),
        "mean_failed_confidence": float(cert_df.loc[cert_df["failure"] == 1, "confidence"].mean()) if (failure == 1).any() else np.nan,
        "auroc_note": defined_auroc_reason(failure),
    }
    for name, values in scores.items():
        row[f"{name}_auroc"] = auroc(values, failure)
        low, high = bootstrap_metric_ci(failure, values, auroc, n_boot=300)
        row[f"{name}_auroc_95_ci"] = format_ci(low, high)
    diff, low, high = paired_bootstrap_auc_diff(failure, scores["cic"], scores["confidence"], n_boot=300)
    row["cic_minus_confidence_auroc"] = diff
    row["cic_minus_confidence_auroc_95_ci"] = format_ci(low, high)
    high = cert_df[cert_df["confidence"] >= 0.8]
    row["high_confidence_cic_auroc"] = auroc(high["cis"], high["failure"]) if len(high) and high["failure"].nunique() > 1 else np.nan
    row["high_confidence_cic_auroc_note"] = "" if np.isfinite(row["high_confidence_cic_auroc"]) else defined_auroc_reason(high["failure"] if "failure" in high else [])
    return pd.DataFrame([row])


def _plot_examples(bundle, out: Path) -> None:
    x, y, shortcut, _causal = bundle.shifted_test.tensors
    n = min(16, len(x))
    fig, axes = plt.subplots(2, 8, figsize=(10, 2.8))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        ax.imshow(x[i].permute(1, 2, 0).numpy())
        ax.set_title(f"y={int(y[i])}, c={int(shortcut[i])}", fontsize=7)
        ax.set_axis_off()
        if i + 1 >= n:
            break
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _copy_or_placeholder(src: Path, dst: Path, title: str) -> None:
    ensure_dir(dst.parent)
    if src.exists():
        shutil.copyfile(src, dst)
        return
    plt.figure(figsize=(4.8, 3.2))
    plt.text(0.5, 0.5, title, ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(dst, dpi=160)
    plt.close()


def _plot_confidence_hist(cert_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(5.4, 3.4))
    plt.hist(cert_df[cert_df["failure"] == 0]["confidence"], bins=16, alpha=0.65, label="correct")
    plt.hist(cert_df[cert_df["failure"] == 1]["confidence"], bins=16, alpha=0.65, label="failed")
    plt.xlabel("Confidence")
    plt.ylabel("Examples")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _plot_auc_bars(metrics: pd.DataFrame, path: Path) -> None:
    rec = metrics.iloc[0]
    labels = ["confidence", "entropy", "margin", "old ShiftRisk", "label flip", "CIC"]
    values = [rec.get(f"{name}_auroc", np.nan) for name in ["confidence", "entropy", "margin", "old_shift_risk", "label_flip_only", "cic"]]
    plt.figure(figsize=(6.2, 3.5))
    plt.bar(labels, values, color=["#0072b2", "#56b4e9", "#999999", "#cc79a7", "#f0e442", "#d55e00"])
    plt.ylim(0, 1)
    plt.ylabel("Failure AUROC")
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _plot_reliability_plane(cert_df: pd.DataFrame, path: Path) -> None:
    stability = np.exp(-cert_df["cis"].clip(lower=0).to_numpy(dtype=float))
    colors = np.where(cert_df["failure"].to_numpy(dtype=int) == 1, "#d55e00", "#0072b2")
    plt.figure(figsize=(4.8, 4.2))
    plt.scatter(cert_df["confidence"], stability, c=colors, alpha=0.7, s=22, edgecolors="none")
    plt.axvline(0.8, color="0.25", linestyle="--", linewidth=1)
    plt.axhline(np.nanmedian(stability), color="0.25", linestyle="--", linewidth=1)
    plt.xlabel("Confidence")
    plt.ylabel("Counterfactual stability")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def run(cfg: dict[str, Any]) -> dict[str, str]:
    cfg = dict(cfg)
    data_cfg = dict(cfg.get("data", {}))
    seed = int(cfg.get("seed", data_cfg.get("seed", 0)))
    data_cfg.setdefault("seed", seed)
    bundle = make_colored_digits_task(**data_cfg)
    cfg["data"] = data_cfg
    summary = run_task("colored_digits", bundle, cfg)

    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "colored_digits")
    plot_dir = ensure_dir(out_dir / "plots")
    cert_path = out_dir / "certificates.csv"
    cert_df = pd.read_csv(cert_path)
    cert_df["task"] = "colored_digits"
    cert_df["regime"] = "confident-wrong"
    cert_df["shift_type"] = "colored_digit_in_support_color_flip"
    cert_df.to_csv(out_dir / "colored_digits_certificates.csv", index=False)
    cert_df.to_csv(cert_path, index=False)

    source = dataset_info(seed=seed).source
    metrics = _summarize_certificates(cert_df, summary, source)
    metrics.to_csv(out_dir / "colored_digits_metrics.csv", index=False)

    summary_md = [
        "# Colored Digits Benchmark",
        "",
        f"Dataset source: {source}.",
        "",
        "This benchmark is supporting evidence. The true label is the digit class; the shortcut is object color. Training uses a high digit-color correlation and shifted evaluation uses an in-support color-label mapping flip.",
        "",
        _markdown_table(metrics),
        "",
        "CIC is evaluated as a color-changing, label-preserving intervention. This does not claim CIC always beats confidence; it tests whether the framework applies in a recognizable shortcut setting.",
        "",
    ]
    (out_dir / "colored_digits_summary.md").write_text("\n".join(summary_md), encoding="utf-8")

    _plot_examples(bundle, plot_dir / "colored_digits_examples.png")
    _plot_confidence_hist(cert_df, plot_dir / "confidence_hist_correct_vs_failed.png")
    _plot_auc_bars(metrics, plot_dir / "cic_vs_confidence_auc.png")
    _plot_reliability_plane(cert_df, plot_dir / "reliability_plane_colored_digits.png")

    return {
        "metrics": str(out_dir / "colored_digits_metrics.csv"),
        "certificates": str(out_dir / "colored_digits_certificates.csv"),
        "summary": str(out_dir / "colored_digits_summary.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colored_digits.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
