# Real-Model Validation

This controlled visual shortcut task tests whether confidence can fail under shortcut flips while counterfactual stability provides complementary evidence of shortcut reliance.

- Model used: local_small_cnn
- Pretrained weights actually loaded: False
- CLIP available / zero-shot used: False
- Linear probe used: False
- Dataset type: background shortcut shapes
- ID accuracy: 1.000
- Shifted accuracy: 0.000
- Certificate examples: ID plus shifted shortcut-flip examples (`split` column records source).
- Mean failed confidence: 1.000
- Confidence AUROC: nan
- Confidence AUROC note: AUROC undefined because shifted correctness contains only one class.
- CIC AUROC: nan
- CIC AUROC note: AUROC undefined because shifted correctness contains only one class.
- High-confidence failure rate (confidence >= 0.8): 0.500

## Warning

Non-pretrained fallback: local small CNN trained only on the controlled ID shortcut data.

## Limitations

- This is a controlled visual shortcut task, not proof that CIC generalizes to all foundation models.
- Fallback results are marked explicitly and should not be used as headline pretrained evidence.
- Attribution outputs are sanity checks, not proof of mechanism.
