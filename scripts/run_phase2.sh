#!/usr/bin/env bash
set -euo pipefail

python3 -m causal_reliability.experiments.run_shortcut_sweep --config configs/shortcut_sweep.yaml
python3 -m causal_reliability.experiments.run_lambda_sweep --config configs/lambda_sweep.yaml
python3 -m causal_reliability.experiments.run_seed_variance --config configs/seed_variance.yaml
python3 -m causal_reliability.experiments.run_counterfactual_mismatch --config configs/counterfactual_mismatch.yaml
python3 -m causal_reliability.experiments.run_vision --config configs/vision.yaml
python3 -m causal_reliability.experiments.run_text --config configs/text.yaml
python3 -m causal_reliability.analysis.main_table --results_dir results
python3 -m causal_reliability.analysis.sts_figure --results_dir results
