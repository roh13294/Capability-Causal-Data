# Natural-Text Directional Repair Metrics (diagnostic)

**Strict support gate remains failed; directional evidence is diagnostic only.**
These secondary metrics describe how repairs move the prediction; they do **not** replace the strict support gate and do not license any positive natural-text claim.

- Verified text-driven failures: 29
- `natural_text_supported` (strict, unchanged): False
- `open_proposal_supported` (strict, unchanged): False
- `open_world_claim_allowed`: False
- `natural_text_directional_evidence` (diagnostic flag): True
- Directional-evidence reasons (if unset): none

## Aggregate directional metrics by method

| method | n | strict_top1_repair_accuracy | alias_top1_repair_accuracy | target_prob_improvement_rate | median_target_prob_gain | target_rank_improvement_rate | median_target_rank_gain | text_distractor_prob_decrease_rate | median_text_distractor_prob_decrease | top3_target_recovery_rate | top5_target_recovery_rate | moved_away_from_text_rate | moved_toward_target_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 29 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.793 | 0.966 | 0.000 | 0.000 |
| oracle_text_box_repair | 29 | 0.310 | 0.310 | 0.966 | 0.058 | 0.655 | 1.000 | 0.897 | 0.384 | 0.828 | 0.966 | 0.897 | 0.966 |
| cic_top1 | 29 | 0.241 | 0.241 | 0.931 | 0.018 | 0.310 | 0.000 | 0.966 | 0.163 | 0.793 | 0.966 | 0.966 | 0.931 |
| cic_top3 | 29 | 0.207 | 0.207 | 0.966 | 0.014 | 0.310 | 0.000 | 0.966 | 0.149 | 0.828 | 0.966 | 0.966 | 0.966 |
| matched_random | 29 | 0.138 | 0.138 | 0.793 | 0.001 | 0.276 | 0.000 | 0.759 | 0.019 | 0.862 | 1.000 | 0.759 | 0.793 |
| largest_region | 29 | 0.172 | 0.172 | 0.828 | 0.005 | 0.310 | 0.000 | 0.862 | 0.053 | 0.828 | 1.000 | 0.862 | 0.828 |
| ocr_text_box_proposal | 13 | 0.231 | 0.231 | 0.769 | 0.022 | 0.538 | 1.000 | 0.769 | 0.105 | 0.692 | 1.000 | 0.769 | 0.769 |

## CIC repair conditional on selected-region text overlap

| text_overlap_bucket | n | cic_strict_repair_rate | cic_alias_repair_rate | cic_directional_rate |
| --- | --- | --- | --- | --- |
| coverage_ge_0.5 | 19 | 0.211 | 0.211 | 0.947 |
| iou_or_coverage_ge_0.3 | 1 | 1.000 | 1.000 | 1.000 |
| no_overlap | 6 | 0.000 | 0.000 | 0.833 |
| partial_overlap | 3 | 0.667 | 0.667 | 1.000 |

## Proposal-selection geometry (CIC top-1, over verified failures)

| metric | mean | median |
| --- | --- | --- |
| text_iou | 0.313 | 0.219 |
| object_iou | 0.073 | 0.075 |
| text_coverage | 0.566 | 0.622 |
| object_coverage | 0.535 | 0.588 |
| selected_area_fraction | 0.070 | 0.072 |
| overlaps_text_box_rate | 0.724 | nan |
| overlaps_object_box_rate | 0.310 | nan |
| overlaps_both_rate | 0.207 | nan |
| closer_to_text_rate | 0.793 | nan |

## Example categories

- cic_directional_only: 17
- cic_strict_repaired: 7
- oracle_strict_repaired: 3
- hard_no_clear_repair: 1
- oracle_directional_only: 1
