#!/usr/bin/env bash
set -euo pipefail

python3 -m causal_reliability.experiments.run_real_text_shortcut_validation --config configs/real_text_shortcut_validation.yaml
python3 -m causal_reliability.validation.export_label_preservation_packet --config configs/label_preservation_packet.yaml
python3 -m causal_reliability.audit.run_cic_audit --config configs/example_cic_audit.yaml
python3 -m causal_reliability.analysis.final_report --results_dir results
