# COCO-Text Proposal-Localization Diagnostic

Diagnoses *why* the full COCO-Text CIC gates fail only on selected text-box overlap: 
is it **(A) a proposal-coverage failure** (text-overlapping proposals are not available) 
or **(B) a scoring/ranking failure** (text-overlapping proposals exist but CIC ranks 
object/background regions higher)? Geometry/ranking uses the frozen `coco_text_cic_full` 
proposals; repair uses the real CLIP backend on the verified directional/strict subsets. 
No existing metric, gate or final-report file is modified; `open_world_claim_allowed=False`.

- Backend: `open_clip` (real pretrained loaded: `True`); model ran: `True`.
- N: all=500, directional=57, strict=39, model-evaluated union=57.

## Headline diagnosis

- **Primary diagnosis: `ranking_failure`**
- Selected text-overlap@1 shortfall vs the 0.600 gate decomposes into **ranking_gap=0.436** (text proposals exist but rank below 1) and **coverage_gap=0.036** (no text proposal available at all).
- Coverage caps the gate even with a perfect ranker: True (recall ceiling 0.564 < threshold 0.600).
- Strict proposal recall (best text IoU >= 0.1): 0.564
- Directional proposal recall (best text IoU >= 0.1): 0.561
- Strict text-overlap@1 / @5 / @10: 0.128 / 0.308 / 0.359
- Strict median CIC rank of best text-overlapping proposal: 4.000
- Best text-overlapping proposal alias repair (ORACLE selection): 0.163
- Selected CIC alias repair: 0.439
- Best-text-proposal repair beats selected CIC: False

## Interpretation (answers to the diagnostic questions)

1. **Do open proposals contain text-overlapping regions?** Strict-subset recall at IoU>=0.1 is 0.564. See `proposal_recall_by_subset.csv` for IoU 0.1/0.3/0.5 and coverage 30/50/80%.
2. **Does CIC rank those text-overlapping regions highly?** text-overlap@1=0.128 but @10=0.359; median CIC rank of the best text-overlapping proposal is in `ranking_diagnostic_by_subset.csv`.
3. **Does repairing with the best text-overlapping proposal beat selected CIC?** best-text alias repair 0.163 vs selected 0.439 (this is an ORACLE selection - it consumes ground-truth text geometry and is not deployable).
4. **Coverage problem or ranking problem?** -> **ranking_failure** (see logic below).
5. **Does area normalization help?** See `area_normalized_scoring_diagnostic.csv`; `reward_text_oracle` is leakage/oracle-only.
6. **Does top-k union help?** See top-k rows in `ranking_diagnostic` / key numbers (topk text-overlap and target-prob improvement for k=1,3,5,10).

Diagnosis logic: a *coverage* failure means text-overlapping proposals are largely absent (low recall). A *ranking* failure means recall is adequate but the top-1 (and small top-k) rarely land on text because CIC scores object/background regions higher.

## Proposal recall by subset

| subset         | n   | has_text_iou_010_rate | has_text_iou_030_rate | has_text_iou_050_rate | has_text_coverage_030_rate | has_text_coverage_050_rate | has_text_coverage_080_rate | has_text_overlapping_proposal_rate | best_text_iou_mean | best_text_coverage_mean | best_object_iou_mean | n_text_overlapping_proposals_mean | n_object_overlapping_proposals_mean |
| -------------- | --- | --------------------- | --------------------- | --------------------- | -------------------------- | -------------------------- | -------------------------- | ---------------------------------- | ------------------ | ----------------------- | -------------------- | --------------------------------- | ----------------------------------- |
| all_500        | 500 | 0.512                 | 0.208                 | 0.07                  | 0.728                      | 0.688                      | 0.628                      | 0.512                              | 0.1655             | 0.6846                  | 0.2351               | 4.038                             | 15.56                               |
| directional_57 | 57  | 0.5614                | 0.2632                | 0.08772               | 0.807                      | 0.6842                     | 0.6316                     | 0.5614                             | 0.1878             | 0.727                   | 0.2329               | 4.807                             | 12.84                               |
| strict_39      | 39  | 0.5641                | 0.3077                | 0.1026                | 0.8205                     | 0.7179                     | 0.6923                     | 0.5641                             | 0.2028             | 0.7424                  | 0.2397               | 4.846                             | 13.79                               |

## Ranking diagnostic by subset

| subset         | n   | text_overlap_at_1_rate | text_overlap_at_3_rate | text_overlap_at_5_rate | text_overlap_at_10_rate | selected_overlaps_text_rate | has_text_overlapping_proposal_rate | rank_best_text_proposal_median | rank_best_object_proposal_median | selected_cic_score_median | best_text_proposal_cic_score_median |
| -------------- | --- | ---------------------- | ---------------------- | ---------------------- | ----------------------- | --------------------------- | ---------------------------------- | ------------------------------ | -------------------------------- | ------------------------- | ----------------------------------- |
| all_500        | 500 | 0.07                   | 0.118                  | 0.154                  | 0.218                   | 0.07                        | 0.512                              | 13                             | 1                                | 0.06437                   | 0.005576                            |
| directional_57 | 57  | 0.1404                 | 0.2456                 | 0.2807                 | 0.3333                  | 0.1404                      | 0.5614                             | 5                              | 2                                | 0.602                     | 0.1689                              |
| strict_39      | 39  | 0.1282                 | 0.2564                 | 0.3077                 | 0.359                   | 0.1282                      | 0.5641                             | 4                              | 2                                | 0.7037                    | 0.2035                              |

## Area-normalized scoring diagnostic (diagnostic only; `reward_text_oracle` = leakage)

| mode               | is_leakage | n  | new_top1_text_overlap_rate | mean_new_top1_text_iou | mean_new_top1_area_fraction | alias_repair_rate | target_prob_improvement_rate | median_target_prob_gain |
| ------------------ | ---------- | -- | -------------------------- | ---------------------- | --------------------------- | ----------------- | ---------------------------- | ----------------------- |
| original           | False      | 57 | 0.1404                     | 0.04475                | 0.05835                     | 0.4386            | 0.6842                       | 0.1034                  |
| div_sqrt_area      | False      | 57 | 0.1404                     | 0.04406                | 0.03555                     | 0.4386            | 0.7368                       | 0.09083                 |
| div_area_clip      | False      | 57 | 0.1404                     | 0.0425                 | 0.02975                     | 0.386             | 0.7368                       | 0.08867                 |
| penalize_object    | False      | 57 | 0.1404                     | 0.04359                | 0.04861                     | 0.4211            | 0.7193                       | 0.115                   |
| reward_text_oracle | True       | 57 | 0.1579                     | 0.05633                | 0.05638                     | 0.4386            | 0.6842                       | 0.1034                  |

## Inference-time text-dilated proposal diagnostic (no ground-truth boxes)

| family                  | n_examples | mean_n_proposals | mean_best_text_iou_available | text_recall_iou_010 | top1_text_overlap_rate | alias_repair_rate | target_prob_improvement_rate | median_target_prob_gain |
| ----------------------- | ---------- | ---------------- | ---------------------------- | ------------------- | ---------------------- | ----------------- | ---------------------------- | ----------------------- |
| text_like_cc            | 57         | 11.72            | 0.1792                       | 0.5263              | 0.1754                 | 0.3509            | 0.7368                       | 0.06046                 |
| dilated_text_like       | 57         | 11.04            | 0.1956                       | 0.4737              | 0.193                  | 0.3158            | 0.7193                       | 0.06438                 |
| thin_high_contrast_rect | 57         | 8.404            | 0.1444                       | 0.4211              | 0.1579                 | 0.2727            | 0.6182                       | 0.027                   |
| edge_dense_small        | 57         | 12               | 0.1884                       | 0.5263              | 0.2105                 | 0.3333            | 0.6491                       | 0.03821                 |

## Scientific recommendation

The low selected text-overlap is **primarily a scoring/ranking** problem: text-overlapping proposals are present in the open candidate set (recall 0.564) but CIC ranks them at median rank 4.000, so the top-1 lands on text only 0.128 of the time (ranking_gap 0.436 > coverage_gap 0.036). A secondary **coverage ceiling** also exists: even a perfect ranker could not clear the 0.600 gate because proposal recall is below it, so the proposal generator is a genuine but smaller limitation. Because CIC still produces large, real directional repair (target-prob up, text-distractor down) without localizing onto the annotated text, the honest framing is: **report COCO-Text as directional repair and treat the text-overlap shortfall as a documented localization limitation**, not evidence that CIC failed. Given that the variants tested (area normalization, top-k union, inference-time text-dilated proposals) are diagnostic rather than headline methods, this is best presented as an **appendix table plus a one-paragraph limitation** in the main text, not as a new headline result.

## Guardrails

- Wrote only under this output subdirectory; did not touch `results/coco_text_cic_full/`, `results/coco_text_cic_triage/`, or `results/final_report/`.
- Did not change any existing gate, metric, or support flag. `open_world_claim_allowed` stays False.
- Best-text-IoU / best-object-IoU proposal selection and `reward_text_oracle` are **oracle / leakage diagnostics** (they consume ground-truth geometry) and are reported as upper bounds, not deployable methods.