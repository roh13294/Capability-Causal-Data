from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from causal_reliability.data.synthetic_shapes import make_vector_task
from causal_reliability.experiments.stress_utils import (
    certificate_frame,
    failure_metrics,
    plot_metric_by_x,
    shift_risk_summary,
    train_for_bundle,
    write_json,
)
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


def run(cfg: dict) -> pd.DataFrame:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "synthetic" / "shortcut_sweep")
    correlations = cfg.get("shortcut_correlations", [0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
    base_seed = int(cfg.get("seed", 0))
    rows = []
    data_cfg = dict(cfg.get("data", {}))
    for i, corr in enumerate(correlations):
        sweep_cfg = dict(cfg)
        current_data = dict(data_cfg)
        current_data["train_corr"] = float(corr)
        current_data["id_corr"] = float(corr)
        bundle = make_vector_task(**current_data)
        model, id_metrics, shifted_metrics, device = train_for_bundle(bundle, sweep_cfg, base_seed + i)
        cert_df = certificate_frame(model, bundle, sweep_cfg, device)
        cert_df.to_csv(out_dir / f"certificates_corr_{corr:.2f}.csv", index=False)
        table = failure_metrics(cert_df)
        risk_summary = shift_risk_summary(cert_df)
        for _, metric_row in table.iterrows():
            rows.append(
                {
                    "shortcut_correlation": float(corr),
                    "method": str(metric_row["method"]),
                    "id_accuracy": id_metrics["accuracy"],
                    "shifted_accuracy": shifted_metrics["accuracy"],
                    "failure_auroc": metric_row["failure_auroc"],
                    "failure_auprc": metric_row["failure_auprc"],
                    "spearman": metric_row["spearman"],
                    "top_risk_decile_failure_rate": metric_row["top_decile_failure_rate"],
                    "bottom_risk_decile_failure_rate": metric_row["bottom_decile_failure_rate"],
                    "risk_ratio": metric_row["risk_ratio"],
                    "shift_risk_mean": risk_summary["mean_shift_risk"],
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "shortcut_sweep_metrics.csv", index=False)
    shift_rows = metrics[metrics["method"] == "ShiftRisk"].copy()
    write_json(
        out_dir / "shortcut_sweep_summary.json",
        {
            "shortcut_correlations": list(correlations),
            "mean_shifted_accuracy": float(shift_rows["shifted_accuracy"].mean()),
            "mean_shift_risk_auroc": float(shift_rows["failure_auroc"].mean()),
            "mean_shift_risk_ratio": float(shift_rows["risk_ratio"].mean()),
            "best_method_by_mean_auroc": metrics.groupby("method")["failure_auroc"].mean().sort_values(ascending=False).to_dict(),
        },
    )
    plot_metric_by_x(shift_rows, "shortcut_correlation", "shifted_accuracy", out_dir / "shortcut_vs_shift_accuracy.png", ylabel="shifted accuracy")
    plot_metric_by_x(metrics, "shortcut_correlation", "failure_auroc", out_dir / "shortcut_vs_failure_auroc.png", hue="method", ylabel="failure AUROC")
    plot_metric_by_x(shift_rows, "shortcut_correlation", "risk_ratio", out_dir / "shortcut_vs_risk_ratio.png", ylabel="ShiftRisk top/bottom decile failure ratio")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/shortcut_sweep.yaml")
    args = parser.parse_args()
    metrics = run(load_config(args.config))
    print(metrics.groupby("method")["failure_auroc"].mean().sort_values(ascending=False))


if __name__ == "__main__":
    main()
