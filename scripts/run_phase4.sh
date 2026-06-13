#!/usr/bin/env bash
set -euo pipefail

python3 -m causal_reliability.experiments.run_confident_wrong --config configs/confident_wrong.yaml
python3 -m causal_reliability.analysis.metric_audit --results_dir results
python3 -m causal_reliability.experiments.run_negative_controls --config configs/negative_controls.yaml
python3 -m causal_reliability.analysis.main_table --results_dir results
python3 -m causal_reliability.analysis.sts_figure --results_dir results
