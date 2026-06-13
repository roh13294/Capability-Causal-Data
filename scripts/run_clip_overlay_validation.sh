#!/usr/bin/env bash
set -euo pipefail

python3 -m causal_reliability.experiments.run_clip_overlay_validation --config configs/clip_overlay_validation.yaml
python3 -m causal_reliability.analysis.qualitative_examples --results_dir results
python3 -m causal_reliability.analysis.final_report --results_dir results
python3 -m causal_reliability.analysis.main_table --results_dir results
