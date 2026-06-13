#!/usr/bin/env bash
set -euo pipefail

python3 -m causal_reliability.real_models.clip_zero_shot --check --allow-download --backend open_clip --model-name ViT-B-32 --pretrained-tag laion2b_s34b_b79k
