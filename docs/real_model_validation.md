# Real Model Validation

## CLIP Text-Overlay Shortcut Validation

The CLIP overlay validation tests a controlled shortcut failure mode for a vision-language model. Each image contains a simple shape, such as a circle or square, plus an overlaid class word. The true label is always the actual shape. The shortcut is the text overlay.

The validation compares aligned overlays, misleading overlays, and mixed overlays. Misleading overlays stay in support because all overlay words are class words seen in the aligned condition, but the mapping is wrong. Counterfactuals remove the overlay, replace it with `object`, replace it with the correct class word, or replace it with another class word.

Run:

```bash
python3 -m pip install open_clip_torch
python3 -m causal_reliability.real_models.clip_zero_shot --check --allow-download --backend open_clip --model-name ViT-B-32 --pretrained-tag laion2b_s34b_b79k
bash scripts/run_clip_overlay_validation.sh
```

Alternative transformers backend:

```bash
python3 -m pip install transformers accelerate
python3 -m causal_reliability.real_models.clip_zero_shot --check --allow-download --backend transformers --model-name openai/clip-vit-base-patch32
bash scripts/run_clip_overlay_validation.sh
```

These dependencies are optional and are not required for the rest of the repository. Pretrained weights are downloaded only when `model.allow_pretrained_download: true` or `--allow-download` is explicitly set. If CLIP or pretrained weights are unavailable, the runner exits cleanly, writes an unavailable summary, and the result must not be used as pretrained-model evidence. Fallback local CNN results are smoke tests only, not real pretrained validation.
