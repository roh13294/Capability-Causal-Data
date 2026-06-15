# Visual-Decoy Shortcut Pilot (second shortcut family)

Controlled gated pilot, isolated from the typographic text-overlay headline and the completed scale-and-multi-model audit.

- Shortcut family: `non_text_visual_decoy_patch` (central causal shape + competing-class corner patch, no written words)
- n_per_condition (held-out test, per regime): `64`
- Backend: `open_clip`; model: `ViT-B-32`; pretrained tag: `laion2b_s34b_b79k`; pretrained loaded: `True`
- Pilot/headline eligible: `False`

## Headline pilot numbers (held-out test, misleading regime unless noted)

- Clean / no-shortcut accuracy: `1.000`
- Misleading original accuracy: `0.578`
- Oracle decoy neutralization accuracy: `0.969`
- CIC top-1 repair accuracy: `0.734`
- CIC top-3 repair accuracy: `0.516`
- CIC clean-safe repair accuracy: `0.906`
- Matched random candidate-region accuracy: `0.438`
- CIC top-1 minus matched random (gap): `0.297`
- Clean-safe clean drop: `0.000`
- (Diagnostic) top-1 candidate IoU with oracle decoy region: `0.411`
- Validation-selected clean-safe score threshold: `0.685`

## Gate results

- PASS `pretrained_loaded`
- PASS `fake_backend_excluded`
- PASS `clean_accuracy_high`
- FAIL `misleading_original_le_0_40`
- PASS `oracle_repair_ge_0_85`
- PASS `cic_top1_minus_random_ge_0_15`
- PASS `clean_safe_drop_le_0_10`
- PASS `no_scorer_leakage`

Failed gates: ['misleading_original_le_0_40']

## Status (boundary evidence)

One or more strict pilot gates failed. Per the pre-registered protocol this run is recorded as honest
boundary evidence for the visual-decoy shortcut family and is NOT integrated as a positive result.
No further tuning was performed to force a pass.

## Scope and non-claims

- The non-oracle region scorer received only pixels, candidate boxes, and model probabilities. It did not receive
  the true label, correctness, shortcut type, or the oracle decoy box.
- The clean-safe score threshold was selected on a separate validation split, not the held-out test split.
- The oracle decoy box is used only for the oracle upper-bound baseline and for the diagnostic localization IoU.
- This pilot does not claim open-world discovery, general robustness, cross-shortcut transfer, exact localization,
  or universal shortcut repair. It is a single-model, single-family controlled pilot.
- It does not alter the text-overlay headline metrics or the completed scale-and-multi-model audit.