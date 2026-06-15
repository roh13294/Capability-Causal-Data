# Per-Input Class-Balance Validation (final theory gate)

The embedding-additivity validation tested a *stronger-than-necessary* condition (a single global, input-independent shortcut direction) and did not support it for OpenCLIP text overlays. The recovery theorem only requires a **weaker per-input condition**: after neutralization, the residual shortcut contribution to the logits should be approximately **class-independent for each individual image**. A class-independent logit offset cannot change the argmax, so if the residual class-dependent part is below half the clean causal margin, the repaired prediction equals the clean causal argmax. This experiment tests that weaker premise directly on the hard multi-decoy OpenCLIP text-overlay repair result.

Backend: open_clip. Model: ViT-B-32. Pretrained loaded: `True`. Fake backend: `False`. Class-balance threshold epsilon_B = 3.0 logits.

**per_input_class_balance_supported_for_text = `True`**

**clip_theory_support_status = `CLIP-supported via per-input class-balance`**

Per-input metrics, for each example and class y:

- `delta_neutralize_y = logit_y(x_neutralized) - logit_y(x_shortcut)` (PART 2 shift diagnostic), summarized by `shift_std`, `shift_range`, `max_centered_shift = max_y |delta - mean_y delta|`.
- `delta_to_clean_y = logit_y(x_neutralized) - logit_y(x_clean)`; the recovery-relevant per-input class-balance error is `residual = max_y |delta_to_clean_y - mean_y delta_to_clean|` (PART 3). Small residual relative to the clean margin means the neutralized shift is approximately class-independent for that input.
- `margin_condition_satisfied = margin_clean > 2 * residual`, compared against repair success.

## Class-balance by neutralization condition (PART 4/5)

### A. Oracle harmful-text neutralization

- n examples: 32
- mean / median shift_std: 2.865 / 2.716
- mean / median shift_range: 7.254 / 7.301
- mean / median max_centered_shift: 4.827 / 4.571
- mean / median residual-to-clean (recovery-relevant class-dependent residual): 2.301 / 2.464
- % examples satisfying class-balance threshold (residual <= epsilon_B): 0.844
- margin-condition satisfaction rate (m_clean > 2*residual): 0.719
- repair accuracy: 1.000
- repair success | margin satisfied vs violated: 1.000 (n=23) vs 1.000 (n=9)
- mean residual repaired vs failed: 2.301 vs NA

### B. CIC top-1 neutralization

- n examples: 32
- mean / median shift_std: 2.017 / 2.320
- mean / median shift_range: 5.168 / 5.996
- mean / median max_centered_shift: 3.361 / 3.867
- mean / median residual-to-clean (recovery-relevant class-dependent residual): 3.730 / 3.704
- % examples satisfying class-balance threshold (residual <= epsilon_B): 0.281
- margin-condition satisfaction rate (m_clean > 2*residual): 0.344
- repair accuracy: 0.781
- repair success | margin satisfied vs violated: 1.000 (n=11) vs 0.667 (n=21)
- mean residual repaired vs failed: 3.385 vs 4.961

### C. CIC top-3 consensus neutralization

- n examples: 32
- mean / median shift_std: 1.967 / 2.353
- mean / median shift_range: 4.995 / 5.946
- mean / median max_centered_shift: 3.294 / 3.944
- mean / median residual-to-clean (recovery-relevant class-dependent residual): 3.659 / 3.506
- % examples satisfying class-balance threshold (residual <= epsilon_B): 0.312
- margin-condition satisfaction rate (m_clean > 2*residual): 0.344
- repair accuracy: 0.781
- repair success | margin satisfied vs violated: 1.000 (n=11) vs 0.667 (n=21)
- mean residual repaired vs failed: 3.306 vs 4.919

### D. Matched random text-region neutralization (control)

- n examples: 32
- mean / median shift_std: 0.622 / 0.344
- mean / median shift_range: 1.615 / 0.857
- mean / median max_centered_shift: 0.967 / 0.532
- mean / median residual-to-clean (recovery-relevant class-dependent residual): 5.237 / 5.220
- % examples satisfying class-balance threshold (residual <= epsilon_B): 0.125
- margin-condition satisfaction rate (m_clean > 2*residual): 0.125
- repair accuracy: 0.406
- repair success | margin satisfied vs violated: 1.000 (n=4) vs 0.321 (n=28)
- mean residual repaired vs failed: 3.920 vs 6.139

### E. Watermark oracle neutralization (negative family)

- n examples: 24
- mean / median shift_std: 1.215 / 0.944
- mean / median shift_range: 3.036 / 2.362
- mean / median max_centered_shift: 2.042 / 1.587
- mean / median residual-to-clean (recovery-relevant class-dependent residual): 2.674 / 2.593
- % examples satisfying class-balance threshold (residual <= epsilon_B): 0.792
- margin-condition satisfaction rate (m_clean > 2*residual): 0.708
- repair accuracy: 0.750
- repair success | margin satisfied vs violated: 1.000 (n=17) vs 0.143 (n=7)
- mean residual repaired vs failed: 2.621 vs 2.834

## Decision (PART 6)

- pretrained CLIP loaded: `True`; fake backend: `False`
- oracle/CIC more class-balanced than random: `True` (random median residual-to-clean = 5.220)
- per_input_class_balance_supported_for_text = `True`
- clip_theory_support_status = `CLIP-supported via per-input class-balance`

Per-condition a-priori checks (more_balanced_than_random / margin_predicts_repair / repaired_fraction_balanced / failures_worse_balanced):

- oracle: more_balanced_than_random=`True`, margin_predicts_repair=`True`, repaired_fraction_balanced=`True`, failures_worse_balanced=`True`
- cic_top1: more_balanced_than_random=`True`, margin_predicts_repair=`True`, repaired_fraction_balanced=`False`, failures_worse_balanced=`True`
- cic_top3_consensus: more_balanced_than_random=`True`, margin_predicts_repair=`True`, repaired_fraction_balanced=`False`, failures_worse_balanced=`True`

## Object-entangled typographic shortcut effects

OpenCLIP's typographic shortcut effect is not a single global additive bias direction. The shift induced by overlay text is object-entangled: it contains a real shortcut component, but its direction varies substantially with the underlying object. This helps explain why generic global debiasing is unlikely to suffice, and why targeted per-input counterfactual region scoring can still repair failures.

## Scope and caveats

This is a finite-candidate, controlled validation of a weaker per-input premise. It does not establish open-world shortcut discovery, exact bounding-box localization, or general robustness.

This experiment does not change the repair policy, does not tune any threshold on the observed metrics, and does not claim open-world shortcut discovery, exact localization, or general robustness.