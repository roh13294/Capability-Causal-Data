#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-results/phase2_quick}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

python3 - <<PY
from pathlib import Path
import yaml

tmp = Path("$TMP_DIR")
base = {
    "seed": 7,
    "prefer_gpu": False,
    "results_dir": "$RESULTS_DIR",
    "epochs": 1,
    "batch_size": 16,
    "lr": 0.003,
    "n_counterfactuals": 2,
    "data": {"n_train": 48, "n_test": 24, "noise": 0.35, "shift_corr": 0.1},
}
configs = {
    "shortcut.yaml": {**base, "shortcut_correlations": [0.7, 0.95]},
    "lambda.yaml": {**base, "lambda_values": [0.0, 0.2], "stability_mode": "combined"},
    "seed.yaml": {**base, "seeds": [0, 1, 2], "stability_lambda": 0.2, "stability_mode": "combined"},
    "mismatch.yaml": {
        **base,
        "intervention_types": ["color"],
        "shift_types": ["color"],
        "data": {"n_train": 32, "n_test": 16, "image_size": 12, "train_corr": 0.9, "id_corr": 0.9, "shift_corr": 0.1},
    },
    "vision.yaml": {
        **base,
        "stability_lambda": 0.2,
        "stability_mode": "combined",
        "data": {"n_train": 32, "n_test": 16, "image_size": 12, "train_corr": 0.9, "id_corr": 0.9, "shift_corr": 0.1},
    },
    "text.yaml": {
        **base,
        "stability_lambda": 0.2,
        "stability_mode": "combined",
        "data": {"n_train": 48, "n_test": 24, "train_corr": 0.9, "id_corr": 0.9, "shift_corr": 0.1},
    },
}
for name, cfg in configs.items():
    (tmp / name).write_text(yaml.safe_dump(cfg), encoding="utf-8")
PY

python3 -m causal_reliability.experiments.run_shortcut_sweep --config "$TMP_DIR/shortcut.yaml"
python3 -m causal_reliability.experiments.run_lambda_sweep --config "$TMP_DIR/lambda.yaml"
python3 -m causal_reliability.experiments.run_seed_variance --config "$TMP_DIR/seed.yaml"
python3 -m causal_reliability.experiments.run_counterfactual_mismatch --config "$TMP_DIR/mismatch.yaml"
python3 -m causal_reliability.experiments.run_vision --config "$TMP_DIR/vision.yaml"
python3 -m causal_reliability.experiments.run_text --config "$TMP_DIR/text.yaml"
python3 -m causal_reliability.analysis.main_table --results_dir "$RESULTS_DIR"
python3 -m causal_reliability.analysis.sts_figure --results_dir "$RESULTS_DIR"
