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


def _row_for_model(seed: int, model_name: str, id_metrics: dict, shifted_metrics: dict, cert_df: pd.DataFrame) -> dict:
    table = failure_metrics(cert_df).set_index("method")
    summary = shift_risk_summary(cert_df)
    return {
        "seed": seed,
        "model": model_name,
        "id_accuracy": id_metrics["accuracy"],
        "shifted_accuracy": shifted_metrics["accuracy"],
        "confidence_auroc": table.loc["confidence", "failure_auroc"],
        "entropy_auroc": table.loc["entropy", "failure_auroc"],
        "margin_auroc": table.loc["margin", "failure_auroc"],
        "shift_risk_auroc": table.loc["ShiftRisk", "failure_auroc"],
        "high_conf_low_reliability_failure_rate": summary["high_conf_low_reliability_failure_rate"],
    }


def run(cfg: dict) -> pd.DataFrame:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "synthetic" / "seed_variance")
    seeds = cfg.get("seeds", list(range(10)))
    rows = []
    for seed in seeds:
        bundle = make_vector_task(**cfg.get("data", {}))
        erm, erm_id, erm_shifted, erm_device = train_for_bundle(bundle, cfg, int(seed), mode="erm")
        erm_df = certificate_frame(erm, bundle, cfg, erm_device)
        erm_df.to_csv(out_dir / f"certificates_seed_{seed}_erm.csv", index=False)
        rows.append(_row_for_model(int(seed), "erm", erm_id, erm_shifted, erm_df))

        stability, stab_id, stab_shifted, stab_device = train_for_bundle(
            bundle,
            cfg,
            int(seed) + 10_000,
            mode=str(cfg.get("stability_mode", "combined")),
            stability_lambda=float(cfg.get("stability_lambda", 0.5)),
        )
        stab_df = certificate_frame(stability, bundle, cfg, stab_device)
        stab_df.to_csv(out_dir / f"certificates_seed_{seed}_stability.csv", index=False)
        rows.append(_row_for_model(int(seed), "stability", stab_id, stab_shifted, stab_df))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "seed_metrics.csv", index=False)
    summary = metrics.groupby("model").agg(["mean", "std"])
    summary.to_csv(out_dir / "seed_summary.csv")
    write_json(out_dir / "seed_summary.json", summary.reset_index().to_dict("records"))

    plot_metric_by_x(metrics, "seed", "shifted_accuracy", out_dir / "shifted_accuracy_by_seed.png", hue="model", ylabel="shifted accuracy")
    plot_metric_by_x(metrics, "seed", "shift_risk_auroc", out_dir / "failure_auroc_by_seed.png", hue="model", ylabel="ShiftRisk failure AUROC")
    means = metrics.groupby("model", as_index=False).agg(
        shifted_accuracy=("shifted_accuracy", "mean"),
        shifted_accuracy_std=("shifted_accuracy", "std"),
        shift_risk_auroc=("shift_risk_auroc", "mean"),
        shift_risk_auroc_std=("shift_risk_auroc", "std"),
    )
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5.2, 3.4))
    x = range(len(means))
    plt.errorbar(x, means["shifted_accuracy"], yerr=means["shifted_accuracy_std"], fmt="o", label="shifted accuracy")
    plt.errorbar(x, means["shift_risk_auroc"], yerr=means["shift_risk_auroc_std"], fmt="s", label="ShiftRisk AUROC")
    plt.xticks(list(x), means["model"])
    plt.ylim(0, 1)
    plt.ylabel("mean +/- std")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "metric_error_bars.png", dpi=140)
    plt.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/seed_variance.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)).groupby("model").mean(numeric_only=True))


if __name__ == "__main__":
    main()
