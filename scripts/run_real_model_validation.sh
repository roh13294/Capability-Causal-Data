#!/usr/bin/env bash
set -euo pipefail

python3 -m causal_reliability.experiments.run_real_model_validation --config configs/real_model_validation.yaml
python3 -m causal_reliability.analysis.qualitative_examples --results_dir results
python3 -m causal_reliability.analysis.final_report --results_dir results
python3 -m causal_reliability.analysis.main_table --results_dir results
python3 -m causal_reliability.analysis.concept_figure --results_dir results
if [ -f causal_reliability/analysis/real_model_figure.py ]; then
  python3 -m causal_reliability.analysis.real_model_figure --results_dir results
fi
python3 -m pytest
