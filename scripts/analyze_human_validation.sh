#!/usr/bin/env bash
set -euo pipefail

RESPONSES_CSV="${1:-}"

if [[ -z "${RESPONSES_CSV}" ]]; then
  python3 -m causal_reliability.validation.analyze_label_preservation_responses
else
  python3 -m causal_reliability.validation.analyze_label_preservation_responses --responses_csv "${RESPONSES_CSV}"
fi
