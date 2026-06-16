# Natural-text intervention/operator sweep (diagnostic)

Diagnoses whether strict natural-text repair is limited by the **intervention 
operator / masking strategy** rather than by CIC proposal selection. Holds the 
candidate geometry fixed (annotated text/logo boxes for the oracle ceiling; the 
existing CIC top-1 proposal for the method) and sweeps deterministic neutralization 
operators on the verified text-driven failures.

- Backend: `open_clip`. Model: `ViT-B-32`. Real pretrained: `True`.
- Verified text-driven failures: **29**.
- Operators evaluated: **13** (1 unavailable: telea_inpaint).
- cv2 / Telea inpaint available: `False`.

## Headline diagnostic answers

- Best **oracle strict** operator: `expanded_gray_fill_1.25` = **0.448**.
- Best **CIC strict** operator: `expanded_gray_fill_1.25` = **0.276**.
- Best **directional** operator: `gray_fill` = **1.000** (target-prob improvement rate).
- Oracle strict ceiling exceeds 0.50: **False**; exceeds 0.70: **False**.
- Oracle ceiling high enough for strict natural-text support: **False**.
- CIC bottleneck attribution: **residual natural-image ambiguity / label-set difficulty (oracle strict stays low even with a known text box, while directional improvement is high)**.
- Any strict gate could pass under a pre-declared global operator: **False**.

## Non-leakage / scope

- Operators are reported as a **diagnostic panel**; no operator is selected per-example using the true label or correctness.
- The only global operator choice uses a pre-declared, label-free aggregate criterion.
- `open_world_claim_allowed`: **False** (unchanged).
- Existing headline / final-report metrics are **not** modified by this experiment.

## Baseline reference (from the existing verified-failure run; unchanged)

```json
{
  "n_verified_text_driven_failures": 29,
  "oracle_text_box_repair_accuracy": 0.3103448275862069,
  "oracle_text_box_repair_or_improve_rate": 0.9655172413793104,
  "cic_top1_repair_accuracy": 0.2413793103448276,
  "matched_random_proposal_repair_accuracy": 0.13793103448275862,
  "selected_overlaps_text_box_rate": 0.7241379310344828,
  "natural_text_supported": false,
  "open_world_claim_allowed": false
}
```

## Oracle text-box ceiling sweep

Note: for this verified set the allowed labels per image are the visual target plus the 
text/logo distractors, so `alias_repair` (predicting any non-distractor label) collapses 
to strict repair here; it is reported for generality.

| operator | strict_repair | alias_repair | top3_recovery | top5_recovery | target_prob_improve_rate | median_target_prob_gain | target_rank_improve_rate | distractor_prob_decrease_rate | object_box_damage_proxy | content_preservation_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gray_fill | 0.310 | 0.310 | 0.897 | 0.966 | 1.000 | 0.077 | 0.621 | 1.000 | 0.040 | 0.850 |
| black_fill | 0.345 | 0.345 | 0.862 | 0.966 | 0.931 | 0.124 | 0.690 | 0.931 | 0.063 | 0.851 |
| white_fill | 0.414 | 0.414 | 0.828 | 0.966 | 0.966 | 0.074 | 0.655 | 0.966 | 0.064 | 0.865 |
| local_mean_fill | 0.310 | 0.310 | 0.862 | 0.966 | 0.966 | 0.085 | 0.655 | 0.966 | 0.032 | 0.850 |
| local_median_fill | 0.310 | 0.310 | 0.828 | 0.966 | 0.966 | 0.083 | 0.655 | 0.966 | 0.032 | 0.852 |
| gaussian_blur | 0.345 | 0.345 | 0.828 | 0.966 | 0.966 | 0.055 | 0.586 | 0.966 | 0.023 | 0.853 |
| pixelation | 0.345 | 0.345 | 0.828 | 0.966 | 0.966 | 0.063 | 0.586 | 0.966 | 0.025 | 0.862 |
| background_border_fill | 0.345 | 0.345 | 0.828 | 0.966 | 1.000 | 0.064 | 0.655 | 1.000 | 0.028 | 0.863 |
| telea_inpaint | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| expanded_gray_fill_1.10 | 0.414 | 0.414 | 0.828 | 0.966 | 0.966 | 0.206 | 0.655 | 0.966 | 0.051 | 0.820 |
| expanded_gray_fill_1.25 | 0.448 | 0.448 | 0.862 | 0.966 | 0.966 | 0.324 | 0.690 | 0.966 | 0.066 | 0.780 |
| expanded_blur_1.10 | 0.345 | 0.345 | 0.828 | 0.966 | 0.966 | 0.057 | 0.586 | 0.966 | 0.027 | 0.824 |
| expanded_blur_1.25 | 0.414 | 0.414 | 0.862 | 1.000 | 0.966 | 0.191 | 0.655 | 0.966 | 0.033 | 0.785 |

## CIC-selected-region operator sweep

| operator | strict_repair | target_prob_improve_rate | target_rank_improve_rate | distractor_prob_decrease_rate | cic_minus_random_gap | content_preservation_proxy | mean_area_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gray_fill | 0.241 | 0.793 | 0.345 | 0.793 | 0.103 | 0.930 | 0.070 |
| black_fill | 0.241 | 0.828 | 0.345 | 0.828 | 0.138 | 0.930 | 0.070 |
| white_fill | 0.241 | 0.931 | 0.310 | 0.931 | 0.103 | 0.942 | 0.070 |
| local_mean_fill | 0.241 | 0.931 | 0.345 | 0.931 | 0.103 | 0.930 | 0.070 |
| local_median_fill | 0.241 | 0.931 | 0.310 | 0.931 | 0.138 | 0.936 | 0.070 |
| gaussian_blur | 0.241 | 0.897 | 0.345 | 0.897 | 0.103 | 0.933 | 0.070 |
| pixelation | 0.241 | 0.862 | 0.379 | 0.862 | 0.069 | 0.940 | 0.070 |
| background_border_fill | 0.241 | 0.897 | 0.345 | 0.897 | 0.103 | 0.938 | 0.070 |
| telea_inpaint | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| expanded_gray_fill_1.10 | 0.241 | 0.897 | 0.345 | 0.897 | 0.103 | 0.914 | 0.086 |
| expanded_gray_fill_1.25 | 0.276 | 0.897 | 0.414 | 0.897 | 0.103 | 0.896 | 0.104 |
| expanded_blur_1.10 | 0.241 | 0.931 | 0.345 | 0.931 | 0.103 | 0.919 | 0.086 |
| expanded_blur_1.25 | 0.276 | 0.897 | 0.414 | 0.897 | 0.138 | 0.905 | 0.104 |

See `oracle_ceiling_analysis.md` for the ceiling interpretation, and the CSVs for the 
full metric set (including top-k union/independent results).