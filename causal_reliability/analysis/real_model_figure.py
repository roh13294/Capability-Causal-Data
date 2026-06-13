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


def make_figure(results_dir: str | Path = "results") -> dict[str, Path]:
    root = Path(results_dir)
    ensure_dir(root)
    metrics_path = root / "real_model_validation" / "real_model_metrics.csv"
    certs_path = root / "real_model_validation" / "real_model_certificates.csv"
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        subset = metrics[metrics["method"].isin(["confidence_risk", "CIC"])]
        axes[0].bar(subset["method"], subset["failure_auroc"], color=["#0072b2", "#d55e00"])
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Failure AUROC")
        axes[0].set_title("Real-model shortcut task")
    else:
        axes[0].text(0.5, 0.5, "Run real-model validation first", ha="center", va="center")
        axes[0].set_axis_off()
    if certs_path.exists():
        certs = pd.read_csv(certs_path)
        colors = np.where(certs["failure"].astype(int) == 1, "#d55e00", "#0072b2")
        axes[1].scatter(certs["confidence"], certs["cic_reliability"], c=colors, s=26, alpha=0.78)
        axes[1].set_xlabel("Confidence")
        axes[1].set_ylabel("Counterfactual stability")
        axes[1].set_xlim(0, 1.02)
        axes[1].set_ylim(0, 1.02)
    else:
        axes[1].text(0.5, 0.5, "No certificates", ha="center", va="center")
        axes[1].set_axis_off()
    fig.tight_layout()
    png = root / "real_model_figure.png"
    pdf = root / "real_model_figure.pdf"
    caption = root / "real_model_figure_caption.md"
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)
    caption.write_text(
        "Real-model validation on a controlled shortcut task. This figure compares confidence-risk and CIC failure AUROC and shows the reliability plane for shifted shortcut examples. It is supporting evidence, not proof of foundation-model generalization.\n",
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
