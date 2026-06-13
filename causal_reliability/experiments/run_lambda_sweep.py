from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from causal_reliability.data.synthetic_shapes import make_vector_task
from causal_reliability.experiments.stress_utils import (
    certificate_frame,
    plot_metric_by_x,
    plot_tradeoff,
    shift_risk_summary,
    train_for_bundle,
    write_json,
)
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


def run(cfg: dict) -> pd.DataFrame:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "synthetic" / "lambda_sweep")
    lambdas = cfg.get("lambda_values", [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0])
    base_seed = int(cfg.get("seed", 0))
    rows = []
    for i, lam in enumerate(lambdas):
        bundle = make_vector_task(**cfg.get("data", {}))
        mode = "erm" if float(lam) == 0.0 else str(cfg.get("stability_mode", "combined"))
        model, id_metrics, shifted_metrics, device = train_for_bundle(bundle, cfg, base_seed + i, mode=mode, stability_lambda=float(lam))
        cert_df = certificate_frame(model, bundle, cfg, device)
        cert_df.to_csv(out_dir / f"certificates_lambda_{lam:.3g}.csv", index=False)
        summary = shift_risk_summary(cert_df)
        rows.append(
            {
                "lambda": float(lam),
                "id_accuracy": id_metrics["accuracy"],
                "shifted_accuracy": shifted_metrics["accuracy"],
                "worst_group_accuracy": summary["worst_group_accuracy"],
                "mean_shift_risk": summary["mean_shift_risk"],
                "failure_auroc": summary["shift_risk_failure_auroc"],
                "confidence_reliability_gap": summary["confidence_reliability_gap"],
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "lambda_sweep_metrics.csv", index=False)
    best = metrics.sort_values(["shifted_accuracy", "id_accuracy"], ascending=False).iloc[0].to_dict()
    write_json(
        out_dir / "lambda_sweep_summary.json",
        {
            "lambda_values": list(lambdas),
            "best_lambda_by_shifted_accuracy": best,
            "erm_row": metrics[metrics["lambda"] == 0.0].head(1).to_dict("records"),
        },
    )
    plot_metric_by_x(metrics, "lambda", "shifted_accuracy", out_dir / "lambda_vs_shift_accuracy.png", ylabel="shifted accuracy")
    plot_metric_by_x(metrics, "lambda", "id_accuracy", out_dir / "lambda_vs_id_accuracy.png", ylabel="ID accuracy")
    plot_metric_by_x(metrics, "lambda", "mean_shift_risk", out_dir / "lambda_vs_mean_shift_risk.png", ylabel="mean ShiftRisk")
    plot_tradeoff(metrics, out_dir / "lambda_tradeoff_curve.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lambda_sweep.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
