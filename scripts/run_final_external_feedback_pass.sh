#!/usr/bin/env bash
set -euo pipefail

python3 -m causal_reliability.experiments.run_random_aug_failure_benchmark --config configs/random_aug_failure_benchmark.yaml
python3 -m causal_reliability.experiments.run_traffic_sign_shortcut_validation --config configs/traffic_sign_shortcut_validation.yaml
python3 -m causal_reliability.validation.export_label_preservation_packet --config configs/label_preservation_packet.yaml
python3 -m causal_reliability.analysis.final_report --results_dir results
