from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from causal_reliability.experiments.final_protocol import (
    add_certificate_cis,
    interpretation,
    locked_certificate_frame,
    metric_row,
    save_validation_plots,
    summarize_metrics,
    write_markdown,
)
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


def run(cfg: dict[str, Any]) -> dict[str, str]:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "final_validation")
    plot_dir = ensure_dir(out_dir / "plots")
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2])]
    tasks = list(cfg.get("tasks", ["synthetic", "vision", "text"]))
    regimes = list(cfg.get("regimes", ["confidence_solvable", "confident_wrong", "mixed"]))
    n_examples = int(cfg.get("n_examples", 96))

    cert_frames = []
    metric_rows = []
    for task in tasks:
        for regime in regimes:
            for seed in seeds:
                cert_df = locked_certificate_frame(task, regime, seed, n=n_examples)
                cert_frames.append(cert_df)
                metric_rows.append(metric_row(cert_df, task, regime, seed))

    by_seed = pd.DataFrame(metric_rows)
    metrics = by_seed.copy()
    certs = pd.concat(cert_frames, ignore_index=True)
    summary = add_certificate_cis(summarize_metrics(metrics), certs)
    summary["interpretation"] = summary.apply(interpretation, axis=1)

    metrics.to_csv(out_dir / "final_validation_metrics.csv", index=False)
    by_seed.to_csv(out_dir / "final_validation_by_seed.csv", index=False)
    summary.to_csv(out_dir / "final_validation_summary.csv", index=False)
    certs.to_csv(out_dir / "final_validation_certificates.csv", index=False)
    with (out_dir / "final_validation_config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    write_markdown(
        out_dir / "final_validation_summary.md",
        "Final Validation Summary",
        summary[
            [
                "task",
                "regime",
                "seeds",
                "id_accuracy_mean_std",
                "shifted_accuracy_mean_std",
                "mean_failed_confidence_mean_std",
                "confidence_risk_auroc_mean_std",
                "confidence_risk_auroc_95_ci",
                "shift_risk_auroc_mean_std",
                "label_flip_only_auroc_mean_std",
                "cis_auroc_mean_std",
                "cis_auroc_95_ci",
                "cic_minus_confidence_auroc_mean_std",
                "cic_minus_confidence_auroc_95_ci",
                "interpretation",
            ]
        ],
        "Locked settings are reported by regime. CIC denotes the Counterfactual Instability Certificate.",
    )
    save_validation_plots(metrics, certs, plot_dir)
    return {"out_dir": str(out_dir), "summary": str(out_dir / "final_validation_summary.md")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_validation.yaml")
    args = parser.parse_args()
    result = run(load_config(args.config))
    print(result)


if __name__ == "__main__":
    main()
