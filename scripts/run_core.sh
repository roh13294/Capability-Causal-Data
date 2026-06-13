#!/usr/bin/env bash
set -euo pipefail
python3 -m causal_reliability.experiments.run_synthetic --config configs/synthetic.yaml
python3 -m causal_reliability.experiments.run_tabular --config configs/tabular.yaml
