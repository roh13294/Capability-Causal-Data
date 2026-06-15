#!/usr/bin/env bash
set -euo pipefail

python3 -m causal_reliability.experiments.run_multidecoy_clip_repair --config configs/multidecoy_clip_repair.yaml
python3 -m causal_reliability.analysis.final_report --results_dir results
