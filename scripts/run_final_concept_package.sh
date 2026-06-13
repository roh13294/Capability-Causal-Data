#!/usr/bin/env bash
set -euo pipefail

python3 -m causal_reliability.analysis.reliability_plane --results_dir results
python3 -m causal_reliability.experiments.run_shortcut_discovery --config configs/shortcut_discovery.yaml
python3 -m causal_reliability.analysis.concept_figure --results_dir results
python3 -m causal_reliability.analysis.final_report --results_dir results
python3 -m causal_reliability.analysis.main_table --results_dir results
python3 -m causal_reliability.analysis.sts_figure --results_dir results
