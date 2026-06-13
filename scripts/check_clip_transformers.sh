#!/usr/bin/env bash
set -euo pipefail

python3 -m causal_reliability.real_models.clip_zero_shot --check --allow-download --backend transformers --model-name openai/clip-vit-base-patch32
