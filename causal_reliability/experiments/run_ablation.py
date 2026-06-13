import argparse
from pathlib import Path

import pandas as pd

from causal_reliability.data.synthetic_shapes import make_vector_task
from causal_reliability.experiments.common import run_task
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir
from causal_reliability.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ablation.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(Path(cfg.get("results_dir", "results")) / "synthetic" / "ablation")

    rows = []
    for corr in cfg.get("shortcut_strengths", [0.6, 0.8, 0.95]):
        local = dict(cfg)
        data = dict(cfg.get("data", {}))
        data["train_corr"] = corr
        local["data"] = data
        set_seed(int(local.get("seed", 0)))
        summary = run_task(f"synthetic/ablation/shortcut_{corr}", make_vector_task(**data), local)
        rows.append({"train_corr": corr, **summary})
    pd.DataFrame(rows).to_csv(out / "shortcut_strength_ablation.csv", index=False)

    rows = []
    for lam in cfg.get("lambdas", [0, 0.1, 0.5, 1.0]):
        local = dict(cfg)
        local["stability_lambda"] = lam
        set_seed(int(local.get("seed", 0)))
        summary = run_task(f"synthetic/ablation/lambda_{lam}", make_vector_task(**cfg.get("data", {})), local)
        rows.append({"lambda": lam, **summary})
    pd.DataFrame(rows).to_csv(out / "lambda_ablation.csv", index=False)

    pd.DataFrame({"note": ["distance, weight, counterfactual-count, shift-severity, and seed sweeps can be extended from this runner."]}).to_csv(
        out / "distance_ablation.csv", index=False
    )
    pd.DataFrame({"note": ["see configs/ablation.yaml for default weights"]}).to_csv(out / "weight_ablation.csv", index=False)
    pd.DataFrame({"note": ["set n_counterfactuals in config or invoke run_task programmatically"]}).to_csv(out / "counterfactual_count_ablation.csv", index=False)
    pd.DataFrame({"note": ["vary data.shift_corr for mild/strong/flipped shift"]}).to_csv(out / "seed_variance.csv", index=False)


if __name__ == "__main__":
    main()
