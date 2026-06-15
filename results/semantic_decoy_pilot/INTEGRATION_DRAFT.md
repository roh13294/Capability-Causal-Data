# DRAFT — Proposed integration of the semantic-decoy second family

**Status: DRAFT FOR REVIEW. Nothing in the headline, final report, key-numbers JSON,
scale/multi-model audit, Waterbirds diagnostic, or visual-decoy boundary result has
been changed.** This file only proposes how the passing second-family pilot *would*
slot in if approved.

## What was run

The same finite-candidate CIC region method, applied to a new non-text shortcut
family with its own candidate intervention set: a central colored causal icon
(sun / heart / leaf / moon, true label) plus a larger, spatially separated,
competing-class corner icon (the decoy). No written words anywhere. The decoy's
oracle bbox is used only for the oracle upper bound and a diagnostic IoU; the
non-oracle scorer sees only pixels, candidate boxes, and model probabilities.

Two runs, identical config except size, real pretrained OpenCLIP
ViT-B-32 / laion2b_s34b_b79k (fake backend blocked):

| Metric (misleading regime unless noted) | n=64 pilot | n=128 scale | Gate |
|---|---|---|---|
| Clean / no-shortcut accuracy | 1.000 | 1.000 | ≥ 0.80 ✓ |
| Misleading original accuracy | 0.297 | 0.297 | ≤ 0.40 ✓ |
| Oracle decoy neutralization | 1.000 | 1.000 | ≥ 0.85 ✓ |
| CIC top-1 repair | 0.719 | 0.711 | — |
| CIC top-3 repair | 0.281 | 0.359 | — |
| CIC clean-safe repair | 0.719 | 0.766 | — |
| Matched random repair | 0.172 | 0.258 | — |
| CIC top-1 − matched random gap | +0.547 | +0.453 | ≥ 0.15 ✓ |
| Clean-safe clean drop | 0.000 | 0.008 | ≤ 0.10 ✓ |
| Scorer leakage check | safe | safe | no leakage ✓ |

Both runs pass all 8 strict gates. Numbers are stable across the 2× scale.

## Proposed key-numbers block (NOT yet written)

Mirrors the existing `cross_shortcut_*` / `clip_overlay_repair_*` conventions,
including the explicit `headline_eligible` / `include_in_headline` flags. Suggested
default: eligible = true, **include_in_headline = false** (a second-family
robustness corroboration, not the headline claim).

```json
{
  "semantic_decoy_family": "non_text_semantic_decoy_icon",
  "semantic_decoy_backend": "open_clip",
  "semantic_decoy_pretrained": true,
  "semantic_decoy_model": "ViT-B-32 / laion2b_s34b_b79k",
  "semantic_decoy_n_per_condition_pilot": 64,
  "semantic_decoy_n_per_condition_scale": 128,
  "semantic_decoy_clean_accuracy": 1.0,
  "semantic_decoy_misleading_accuracy": 0.296875,
  "semantic_decoy_oracle_repair": 1.0,
  "semantic_decoy_cic_top1": 0.7109375,
  "semantic_decoy_cic_top3": 0.359375,
  "semantic_decoy_cic_clean_safe": 0.765625,
  "semantic_decoy_matched_random": 0.2578125,
  "semantic_decoy_cic_minus_random_gap": 0.453125,
  "semantic_decoy_clean_safe_drop": 0.0078125,
  "semantic_decoy_all_gates_passed": true,
  "semantic_decoy_headline_eligible": true,
  "semantic_decoy_include_in_headline": false
}
```

(Scale-run values shown; the pilot block could carry both `_pilot` and `_scale`
variants if preferred.)

## Proposed prose for the final report (NOT yet written)

> **Second shortcut family (corroboration).** Beyond the typographic text-overlay
> headline, we ran the same finite-candidate CIC region method on an independent
> non-text shortcut family: a central colored causal icon with a larger competing
> -class corner icon and no written words. On real pretrained OpenCLIP ViT-B-32,
> the decoy drives misleading-regime accuracy to 0.30 while the central icon alone
> is perfectly recognized (clean 1.00) and oracle removal of the decoy fully
> restores it (1.00). Using only pixels, candidate boxes, and model probabilities
> — no label, correctness, shortcut-type, or oracle-box leakage — CIC top-1
> region repair recovers 0.71, versus 0.26 for an area-matched random candidate
> region (gap +0.45), with a 0.008 clean-regime drop under the validation-selected
> clean-safe policy. Results are stable from n=64 to n=128. This corroborates that
> the method generalizes across shortcut *modality* (typographic → semantic-icon),
> not just across instances of one family. It is a single-model, single-family
> controlled result and is **not** a claim of open-world discovery, cross-shortcut
> transfer, exact localization, or universal repair.

## Non-claims preserved

- Does not replace or alter the text-overlay headline metric.
- Does not alter the scale/multi-model audit, Waterbirds diagnostic, or the
  visual-decoy boundary result (which remains honest negative evidence for the
  *low-salience* flat-shape decoy).
- The decoy being larger than the central icon is the salience lever; the central
  icon is still the causal object (its identity is the label, confirmed by clean
  and oracle accuracy).

## To commit this (only on approval)

1. Add the `semantic_decoy_*` block to `results/final_report/final_key_numbers.json`
   (and the generator in `causal_reliability/analysis/final_report.py` if numbers
   should be produced programmatically rather than pasted).
2. Add the prose paragraph to `results/final_report/final_report.md`.
3. Add a row to `FINAL_ARTIFACT_INDEX.md` pointing at `results/semantic_decoy_pilot/`
   and `results/semantic_decoy_scale_n128/`.
4. Keep `include_in_headline = false` unless you decide it should be promoted.
