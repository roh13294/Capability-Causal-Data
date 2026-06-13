#!/usr/bin/env bash
set -euo pipefail
python3 -m causal_reliability.experiments.run_ablation --config configs/ablation.yaml
