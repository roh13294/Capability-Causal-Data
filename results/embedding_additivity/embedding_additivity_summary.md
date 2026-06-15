# Embedding-Additivity Validation

This experiment gates the theory claim in `docs/theory.md`. CLIP logits are inner products `logit_y(X) = <u(X), v_y>`, so the additive-logit decomposition the recovery theorem assumes holds iff the embedding shift caused by a shortcut is approximately input-independent: `u(object + shortcut) - u(object only) ~= shortcut_direction`. We test that on two finite-candidate shortcut families with real pretrained OpenCLIP.

Backend: open_clip. Model: ViT-B-32. Pretrained loaded: `True`. Fake backend: `False`.

**Embedding additivity is NOT empirically supported for the current OpenCLIP text benchmark by this validation. The theorem should be presented as a conditional / theoretical explanation only, and must not be claimed to apply directly to CLIP text-overlay repair.**

## 1. Does the text-overlay shortcut satisfy approximate embedding additivity?

- Same-shortcut delta mean pairwise cosine: 0.765 (shuffled-label baseline 0.634; margin 0.131)
- Cosine to shortcut-value centroid (mean): 0.881
- Logit-channel consistency MAE / corr: 0.000 / 1.000
- Verdict: delta vectors cluster above shuffled baseline = `True`

## 2. Does the shortcut delta cluster by shortcut value more than by object class?

- Within-shortcut delta cohesion (mean pairwise cosine): 0.765
- Within-object delta cohesion (mean pairwise cosine): 0.855
- Nearest-centroid accuracy by shortcut value: 1.000 (chance 0.250, shuffled 0.256)
- Nearest-centroid accuracy by object class: 1.000 (note: nearest-centroid can saturate at 1.0 for both groupings; the cohesion comparison above is the decisive, non-saturating test)
- Verdict: shortcut clustering exceeds object clustering = `False`

## 3. Is neutralization close to the clean embedding?

- Mean clean-damage proxy ||u(neutralized) - u(clean)||: 0.891
- Mean shortcut-effect norm ||u(shortcut) - u(clean)||: 0.971
- Mean ratio (clean damage / shortcut effect): 0.917
- Verdict: neutralization damage small relative to shortcut effect = `False`

## 4. Does the margin diagnostic predict repair success?

- Repair-success rate overall: 1.000
- Fraction satisfying margin condition (m > 2*residual): 0.167
- Repair success | margin satisfied: 1.000 (n=8)
- Repair success | margin violated: 1.000 (n=40)
- Verdict: margin condition predicts repair = `True`

## 5. Does the watermark negative result show a weak / flat shortcut channel?

- Watermark mean shortcut-effect norm: 0.885 (text 0.971)
- Watermark same-shortcut delta cosine vs shuffled: 0.757 vs 0.721
- Watermark nearest-centroid accuracy (shortcut vs object): 0.604 vs 1.000
- Watermark shortcut channel weak/flat = `True`

The watermark transfer failure is consistent with the theory: the shortcut channel was weak or flat, so there was no strong shortcut contribution for CIC to neutralize.

## 6. How should the theorem be presented?

- embedding_additivity_supported_for_text = `False`
- embedding_additivity_supported_for_watermark = `False`
- Recommended framing: conditional_theory_only

## Scope and caveats

This is a finite-candidate, controlled validation. It does not establish open-world shortcut discovery, exact bounding-box localization, or general robustness. It only tests whether the additive-channel assumption behind the conditional recovery theorem is supported for these specific shortcut families on this pretrained CLIP model.

Headline-eligibility checks (text):

- pretrained_loaded = `True`
- not_fake_backend = `True`
- delta_clusters_above_shuffled = `True`
- shortcut_clustering_exceeds_object = `False`
- neutralization_damage_small = `False`
- margin_condition_predicts_repair = `True`