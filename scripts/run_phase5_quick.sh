#!/usr/bin/env bash
set -euo pipefail

python3 -m causal_reliability.experiments.run_partial_flip_sweep --config configs/partial_flip_sweep.yaml
python3 -m causal_reliability.experiments.run_certificate_ablation --config configs/certificate_ablation.yaml
python3 -m causal_reliability.experiments.run_negative_controls --config configs/negative_controls.yaml
python3 -m causal_reliability.analysis.negative_control_diagnosis --results_dir results
python3 -m causal_reliability.analysis.main_table --results_dir results
python3 -m causal_reliability.analysis.sts_figure --results_dir results
