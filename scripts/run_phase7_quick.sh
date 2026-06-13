#!/usr/bin/env bash
set -euo pipefail

python3 -m causal_reliability.experiments.run_final_validation --config configs/final_validation.yaml
python3 -m causal_reliability.experiments.run_final_negative_controls --config configs/final_negative_controls.yaml
python3 -m causal_reliability.analysis.final_report --results_dir results
python3 -m causal_reliability.analysis.main_table --results_dir results
python3 -m causal_reliability.analysis.sts_figure --results_dir results
