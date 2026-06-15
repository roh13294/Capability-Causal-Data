# Semantic-Decoy Shortcut Pilot (second shortcut family, final attempt)

One final principled attempt at a stronger non-text second shortcut family.
Controlled gated pilot, isolated from the typographic text-overlay headline, the completed
scale-and-multi-model audit, the Waterbirds diagnostic, and the earlier visual-decoy boundary result.

- Shortcut family: `non_text_semantic_decoy_icon` (central coloured causal icon + competing-class corner icon, no written words)
- n_per_condition (held-out test, per regime): `128`
- Backend: `open_clip`; model: `ViT-B-32`; pretrained tag: `laion2b_s34b_b79k`; pretrained loaded: `True`
- Pilot eligible: `True`

## Headline pilot numbers (held-out test, misleading regime unless noted)

- Clean / no-shortcut accuracy: `1.000`
- Misleading original accuracy: `0.297`
- Oracle decoy neutralization accuracy: `1.000`
- CIC top-1 repair accuracy: `0.711`
- CIC top-3 repair accuracy: `0.359`
- CIC clean-safe repair accuracy: `0.766`
- Matched random candidate-region accuracy: `0.258`
- CIC top-1 minus matched random (gap): `0.453`
- Clean-safe clean drop: `0.008`
- (Diagnostic) top-1 candidate IoU with oracle decoy region: `0.426`
- Validation-selected clean-safe score threshold: `0.125`

## Gate results

- PASS `pretrained_loaded`
- PASS `fake_backend_excluded`
- PASS `clean_accuracy_high`
- PASS `misleading_original_le_0_40`
- PASS `oracle_repair_ge_0_85`
- PASS `cic_top1_minus_random_ge_0_15`
- PASS `clean_safe_drop_le_0_10`
- PASS `no_scorer_leakage`

Failed gates: none

## Recommendation: SCALE to n=128 (pending explicit review/confirmation)

All strict pilot gates passed. These are pilot numbers for a second, non-text shortcut family.
Per the pre-registered protocol, scaling to n=128 (and any integration) requires explicit confirmation
and review before running. This run is NOT integrated into the headline result yet.

## Scope and non-claims

- The non-oracle region scorer received only pixels, candidate boxes, and model probabilities. It did not receive
  the true label, correctness, shortcut type, or the oracle decoy box (verified by signature inspection gate).
- The clean-safe score threshold was selected on a separate validation split, not the held-out test split.
- The oracle decoy box is used only for the oracle upper-bound baseline and for the diagnostic localization IoU.
- This pilot does not claim open-world discovery, general robustness, cross-shortcut transfer, exact localization,
  or universal shortcut repair. It is a single-model, single-family controlled pilot.
- It does not alter the text-overlay headline metrics, the scale/multi-model audit, the Waterbirds diagnostic,
  or the earlier visual-decoy boundary result.