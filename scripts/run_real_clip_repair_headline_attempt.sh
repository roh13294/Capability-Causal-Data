#!/usr/bin/env bash
set -euo pipefail

python3 -m causal_reliability.experiments.run_clip_overlay_repair --config configs/clip_overlay_repair.yaml
python3 -m causal_reliability.analysis.final_report --results_dir results
