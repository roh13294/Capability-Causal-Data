#!/usr/bin/env bash
set -euo pipefail

python3 -m causal_reliability.experiments.run_clip_overlay_repair --config configs/clip_overlay_repair.yaml
python3 -m causal_reliability.experiments.run_real_text_repair --config configs/real_text_repair.yaml
python3 -m causal_reliability.experiments.run_random_aug_failure_repair --config configs/random_aug_failure_repair.yaml
python3 -m causal_reliability.analysis.final_report --results_dir results

echo "CIC repair suite completed."
