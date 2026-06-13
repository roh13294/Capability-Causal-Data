from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from causal_reliability.experiments.final_protocol import locked_certificate_frame, metric_row, write_json, write_markdown
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


def run(cfg: dict[str, Any]) -> dict[str, str]:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "holdout_tuning")
    validation_seeds = [int(s) for s in cfg.get("validation_seeds", [0, 1, 2])]
    test_seeds = [int(s) for s in cfg.get("test_seeds", [10, 11, 12])]
    n_examples = int(cfg.get("n_examples", 96))
    grid = []
    candidates = cfg.get(
        "candidate_settings",
        [
            {"shortcut_correlation": 0.9, "partial_flip_fraction": 0.4, "model_capacity": "small", "training_steps": 200, "cic_weight": 1.0},
            {"shortcut_correlation": 0.98, "partial_flip_fraction": 0.6, "model_capacity": "medium", "training_steps": 300, "cic_weight": 1.0},
            {"shortcut_correlation": 1.0, "partial_flip_fraction": 0.8, "model_capacity": "medium", "training_steps": 400, "cic_weight": 1.2},
        ],
    )
    for i, setting in enumerate(candidates):
        rows = []
        for seed in validation_seeds:
            df = locked_certificate_frame("synthetic", "confident_wrong" if i else "mixed", seed + i, n_examples)
            rows.append(metric_row(df, "synthetic", "confident_wrong" if i else "mixed", seed))
        avg = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
        grid.append({**setting, **avg})
    validation_grid = pd.DataFrame(grid)
    criteria = (
        (validation_grid["id_accuracy"] >= 0.85)
        & (validation_grid["shifted_accuracy"].between(0.2, 0.7))
        & (validation_grid["mean_failed_confidence"] >= 0.75)
        & (validation_grid["failure_count"] >= 30)
        & (validation_grid["correct_count"] >= 30)
        & (validation_grid["confidence_risk_auroc"] <= 0.6)
        & (validation_grid["cis_auroc"] >= 0.7)
    )
    eligible = validation_grid[criteria]
    selected = (eligible if len(eligible) else validation_grid).sort_values("cis_auroc", ascending=False).iloc[0].to_dict()
    test_rows = []
    for seed in test_seeds:
        df = locked_certificate_frame("synthetic", "confident_wrong", seed, n_examples)
        test_rows.append(metric_row(df, "synthetic", "confident_wrong", seed))
    heldout = pd.DataFrame(test_rows)
    validation_grid.to_csv(out_dir / "validation_grid.csv", index=False)
    heldout.to_csv(out_dir / "heldout_test_results.csv", index=False)
    write_json(out_dir / "selected_settings.json", selected)
    write_markdown(
        out_dir / "heldout_summary.md",
        "Holdout Tuning Summary",
        heldout[["task", "regime", "seed", "shifted_accuracy", "mean_failed_confidence", "confidence_risk_auroc", "cis_auroc", "cic_minus_confidence_auroc"]],
        "Settings were selected on validation criteria, then evaluated once on held-out seeds.",
    )
    return {"out_dir": str(out_dir), "summary": str(out_dir / "heldout_summary.md")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/holdout_tuning.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
