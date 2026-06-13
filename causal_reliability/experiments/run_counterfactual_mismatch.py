from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from causal_reliability.counterfactuals.base import vision_counterfactuals
from causal_reliability.data.synthetic_shapes import make_shape_task
from causal_reliability.experiments.stress_utils import (
    certificate_frame,
    plot_heatmap,
    shift_risk_summary,
    train_for_bundle,
    write_json,
)
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


def run(cfg: dict) -> pd.DataFrame:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "vision" / "counterfactual_mismatch")
    intervention_types = cfg.get("intervention_types", ["color", "background", "texture"])
    shift_types = cfg.get("shift_types", ["color", "background", "texture"])
    base_seed = int(cfg.get("seed", 0))
    rows = []
    data_cfg = dict(cfg.get("data", {}))
    for shift_i, shift_type in enumerate(shift_types):
        current_data = dict(data_cfg)
        current_data["shortcut_type"] = shift_type
        current_data["train_shortcut_type"] = shift_type
        current_data["id_shortcut_type"] = shift_type
        current_data["shift_shortcut_type"] = shift_type
        bundle = make_shape_task(**current_data)
        model, _id_metrics, shifted_metrics, device = train_for_bundle(bundle, cfg, base_seed + shift_i, mode="erm")
        for intervention_type in intervention_types:
            make_cf = lambda x, t=intervention_type: vision_counterfactuals(x, int(cfg.get("n_counterfactuals", 4)), intervention_type=t)
            cert_df = certificate_frame(model, bundle, cfg, device, make_cf=make_cf)
            cert_df.to_csv(out_dir / f"certificates_intervention_{intervention_type}_shift_{shift_type}.csv", index=False)
            summary = shift_risk_summary(cert_df)
            rows.append(
                {
                    "certificate_intervention_type": intervention_type,
                    "deployment_shift_type": shift_type,
                    "failure_auroc": summary["shift_risk_failure_auroc"],
                    "shifted_accuracy": shifted_metrics["accuracy"],
                    "top_risk_failure_rate": summary["top_risk_decile_failure_rate"],
                    "bottom_risk_failure_rate": summary["bottom_risk_decile_failure_rate"],
                    "risk_ratio": summary["risk_ratio"],
                }
            )
    matrix = pd.DataFrame(rows)
    matrix.to_csv(out_dir / "mismatch_matrix.csv", index=False)
    auroc_pivot = matrix.pivot(index="certificate_intervention_type", columns="deployment_shift_type", values="failure_auroc").loc[intervention_types, shift_types]
    ratio_pivot = matrix.pivot(index="certificate_intervention_type", columns="deployment_shift_type", values="risk_ratio").loc[intervention_types, shift_types]
    write_json(
        out_dir / "mismatch_summary.json",
        {
            "mean_diagonal_auroc": float(sum(auroc_pivot.loc[t, t] for t in set(intervention_types) & set(shift_types)) / max(len(set(intervention_types) & set(shift_types)), 1)),
            "mean_failure_auroc": float(matrix["failure_auroc"].mean()),
            "best_cells_by_auroc": matrix.sort_values("failure_auroc", ascending=False).head(5).to_dict("records"),
        },
    )
    plot_heatmap(auroc_pivot, out_dir / "mismatch_failure_auroc_heatmap.png", "Failure AUROC", "AUROC")
    plot_heatmap(ratio_pivot, out_dir / "mismatch_risk_ratio_heatmap.png", "Top/bottom risk failure ratio", "risk ratio")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/counterfactual_mismatch.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
