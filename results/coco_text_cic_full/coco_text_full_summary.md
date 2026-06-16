# COCO-Text Full Proposal-Based CIC (dataset-backed natural-image validation)

Full **shortcut-agnostic proposal-based CIC** / **open-candidate intervention search** over the 
verified COCO-Text subsets. Candidate scoring saw only pixels, proposal geometry, and model 
predictions — never the true label, correctness, oracle repair success, or subset membership. 
OCR/text-box proposals are reported as a **separate inference-time family** (we always report 
both *excluding* and *including* OCR boxes). Oracle text-box repair is an **eval-only upper bound**. 
This is **dataset-backed natural-image validation**, NOT full open-world shortcut discovery.

Backend: `open_clip`. Model: `ViT-B-32`. Real pretrained loaded: `True`. 
Fake backend: `False`. Data: loaded 500 local images.

## Support gates

- `coco_text_strict_support` (strict subset, n=39): **False**
  - reasons: ['CIC selected text-overlap rate 0.1282051282051282 < 0.60']
- `coco_text_directional_support` (directional subset, n=57): **False**
  - reasons: ['CIC selected text-overlap rate 0.14035087719298245 < 0.60']
- `open_world_claim_allowed`: False
- OCR-included materially improves over OCR-excluded: False
- Open proposals excluding OCR are sufficient (strict gate via excl-OCR CIC): False

## Per-subset metrics

### all_500 (n=500)

- Original strict / alias-aware accuracy: 0.76 / 0.76
- Oracle strict / alias-aware repair: 0.788 / 0.788
- Oracle top-3 / top-5 / pairwise recovery: 0.92 / 0.95 / 0.58
- CIC (excl OCR) strict / alias top-1: 0.714 / 0.714
- CIC (excl OCR) strict / alias top-3: 0.746 / 0.746
- CIC (incl OCR) strict / alias top-1: 0.716 / 0.716
- CIC top-5 recovery (excl OCR): 0.944
- CIC pairwise target-vs-text recovery (excl OCR): 0.134
- Matched-random / largest-region / OCR-proposal repair (alias): 0.754 / 0.734 / 0.7744360902255639
- CIC - random gap (alias): -0.040000000000000036
- Target-prob improvement rate (CIC / random / oracle): 0.154 / 0.168 / 0.242
- Median target-prob gain (CIC excl OCR): -0.0044976770877838135
- Target-rank improvement rate (CIC excl OCR): 0.12
- Text-distractor decrease rate (CIC / random): 0.254 / 0.222
- Selected text-box / object-box overlap rate (CIC excl OCR): 0.07 / 0.492
- Selected area fraction (CIC excl OCR): 0.07072803730867347

### directional_57 (n=57)

- Original strict / alias-aware accuracy: 0.0 / 0.0
- Oracle strict / alias-aware repair: 0.3157894736842105 / 0.3157894736842105
- Oracle top-3 / top-5 / pairwise recovery: 0.8596491228070176 / 0.9649122807017544 / 0.38596491228070173
- CIC (excl OCR) strict / alias top-1: 0.43859649122807015 / 0.43859649122807015
- CIC (excl OCR) strict / alias top-3: 0.43859649122807015 / 0.43859649122807015
- CIC (incl OCR) strict / alias top-1: 0.45614035087719296 / 0.45614035087719296
- CIC top-5 recovery (excl OCR): 0.8947368421052632
- CIC pairwise target-vs-text recovery (excl OCR): 0.45614035087719296
- Matched-random / largest-region / OCR-proposal repair (alias): 0.17543859649122806 / 0.38596491228070173 / 0.1836734693877551
- CIC - random gap (alias): 0.2631578947368421
- Target-prob improvement rate (CIC / random / oracle): 0.6842105263157895 / 0.47368421052631576 / 1.0
- Median target-prob gain (CIC excl OCR): 0.10340837761759758
- Target-rank improvement rate (CIC excl OCR): 0.5087719298245614
- Text-distractor decrease rate (CIC / random): 0.9649122807017544 / 0.6491228070175439
- Selected text-box / object-box overlap rate (CIC excl OCR): 0.14035087719298245 / 0.3508771929824561
- Selected area fraction (CIC excl OCR): 0.05835284360454708

### strict_39 (n=39)

- Original strict / alias-aware accuracy: 0.0 / 0.0
- Oracle strict / alias-aware repair: 0.46153846153846156 / 0.46153846153846156
- Oracle top-3 / top-5 / pairwise recovery: 0.8974358974358975 / 1.0 / 0.5641025641025641
- CIC (excl OCR) strict / alias top-1: 0.5384615384615384 / 0.5384615384615384
- CIC (excl OCR) strict / alias top-3: 0.5384615384615384 / 0.5384615384615384
- CIC (incl OCR) strict / alias top-1: 0.5641025641025641 / 0.5641025641025641
- CIC top-5 recovery (excl OCR): 0.8974358974358975
- CIC pairwise target-vs-text recovery (excl OCR): 0.5641025641025641
- Matched-random / largest-region / OCR-proposal repair (alias): 0.20512820512820512 / 0.46153846153846156 / 0.2647058823529412
- CIC - random gap (alias): 0.3333333333333333
- Target-prob improvement rate (CIC / random / oracle): 0.6923076923076923 / 0.48717948717948717 / 1.0
- Median target-prob gain (CIC excl OCR): 0.20474731922149658
- Target-rank improvement rate (CIC excl OCR): 0.5897435897435898
- Text-distractor decrease rate (CIC / random): 0.9487179487179487 / 0.717948717948718
- Selected text-box / object-box overlap rate (CIC excl OCR): 0.1282051282051282 / 0.358974358974359
- Selected area fraction (CIC excl OCR): 0.058969350961538464

## Content preservation (clean subset)

- Clean-subset content-preservation rate: 0.8852459016393442
- Clean-subset content-preservation drop: 0.11475409836065575
- Documented: False

## Leakage / scope guard

- No-oracle-leakage (scoring/proposal signatures clean): True
- OCR-included vs OCR-excluded reported separately: True
- Oracle operators are global and label-free; text boxes are eval-only geometry for the upper bound.
- Writes only under this output subdirectory; no final-report / Round-1 / triage artifact was touched.

## Full metrics table

| subset         | method                           | backend   | model_name | pretrained_loaded | oracle_upper_bound | n   | accuracy_strict | accuracy_alias | recovers_top3 | recovers_top5 | pairwise_recovery | target_prob_improvement_rate | median_target_prob_gain | target_rank_improvement_rate | text_distractor_decrease_rate | selected_text_overlap_rate | selected_object_overlap_rate | selected_area_fraction |
| -------------- | -------------------------------- | --------- | ---------- | ----------------- | ------------------ | --- | --------------- | -------------- | ------------- | ------------- | ----------------- | ---------------------------- | ----------------------- | ---------------------------- | ----------------------------- | -------------------------- | ---------------------------- | ---------------------- |
| all_500        | original_clip_prediction         | open_clip | ViT-B-32   | True              | False              | 500 | 0.76            | 0.76           | 0.914         | 0.938         | 0                 | 0                            | 0                       | 0                            | 0                             |                            |                              |                        |
| all_500        | oracle_text_box_repair           | open_clip | ViT-B-32   | True              | True               | 500 | 0.788           | 0.788          | 0.92          | 0.95          | 0.58              | 0.242                        | 0.0002093               | 0.086                        | 0.252                         |                            |                              |                        |
| all_500        | oracle_best_global_op            | open_clip | ViT-B-32   | True              | True               | 500 | 0.774           | 0.774          | 0.92          | 0.946         | 0.43              | 0.196                        | 6.199e-06               | 0.08                         | 0.228                         |                            |                              |                        |
| all_500        | cic_top1_repair_excl_ocr         | open_clip | ViT-B-32   | True              | False              | 500 | 0.714           | 0.714          | 0.886         | 0.944         | 0.134             | 0.154                        | -0.004498               | 0.12                         | 0.254                         | 0.07                       | 0.492                        | 0.07073                |
| all_500        | cic_top3_repair_excl_ocr         | open_clip | ViT-B-32   | True              | False              | 500 | 0.746           | 0.746          | 0.904         | 0.942         | 0.12              | 0.162                        | -0.003385               | 0.12                         | 0.248                         |                            |                              |                        |
| all_500        | cic_top1_repair_incl_ocr         | open_clip | ViT-B-32   | True              | False              | 500 | 0.716           | 0.716          | 0.888         | 0.944         | 0.136             | 0.156                        | -0.004412               | 0.124                        | 0.254                         | 0.088                      | 0.488                        | 0.06977                |
| all_500        | cic_top3_repair_incl_ocr         | open_clip | ViT-B-32   | True              | False              | 500 | 0.746           | 0.746          | 0.906         | 0.942         | 0.12              | 0.16                         | -0.003385               | 0.12                         | 0.246                         |                            |                              |                        |
| all_500        | cic_top1_best_global_op_excl_ocr | open_clip | ViT-B-32   | True              | False              | 500 | 0.716           | 0.716          | 0.89          | 0.934         | 0.184             | 0.172                        | -0.00249                | 0.128                        | 0.23                          |                            |                              |                        |
| all_500        | matched_random_proposal_repair   | open_clip | ViT-B-32   | True              | False              | 500 | 0.754           | 0.754          | 0.896         | 0.936         | 0.324             | 0.168                        | -0.0001115              | 0.062                        | 0.222                         |                            |                              |                        |
| all_500        | largest_region_repair            | open_clip | ViT-B-32   | True              | False              | 500 | 0.734           | 0.734          | 0.878         | 0.926         | 0.244             | 0.18                         | -0.001407               | 0.124                        | 0.262                         |                            |                              |                        |
| all_500        | ocr_proposal_repair              | open_clip | ViT-B-32   | True              | False              | 399 | 0.7744          | 0.7744         | 0.9098        | 0.9348        | 0.3509            | 0.1303                       | -2.861e-06              | 0.05514                      | 0.1529                        |                            |                              |                        |
| directional_57 | original_clip_prediction         | open_clip | ViT-B-32   | True              | False              | 57  | 0               | 0              | 0.7895        | 0.8596        | 0                 | 0                            | 0                       | 0                            | 0                             |                            |                              |                        |
| directional_57 | oracle_text_box_repair           | open_clip | ViT-B-32   | True              | True               | 57  | 0.3158          | 0.3158         | 0.8596        | 0.9649        | 0.386             | 1                            | 0.09763                 | 0.5965                       | 0.9123                        |                            |                              |                        |
| directional_57 | oracle_best_global_op            | open_clip | ViT-B-32   | True              | True               | 57  | 0.2982          | 0.2982         | 0.8596        | 0.9474        | 0.3684            | 0.9474                       | 0.05194                 | 0.5789                       | 0.9123                        |                            |                              |                        |
| directional_57 | cic_top1_repair_excl_ocr         | open_clip | ViT-B-32   | True              | False              | 57  | 0.4386          | 0.4386         | 0.6842        | 0.8947        | 0.4561            | 0.6842                       | 0.1034                  | 0.5088                       | 0.9649                        | 0.1404                     | 0.3509                       | 0.05835                |
| directional_57 | cic_top3_repair_excl_ocr         | open_clip | ViT-B-32   | True              | False              | 57  | 0.4386          | 0.4386         | 0.7719        | 0.8947        | 0.4737            | 0.7193                       | 0.1366                  | 0.5439                       | 0.9298                        |                            |                              |                        |
| directional_57 | cic_top1_repair_incl_ocr         | open_clip | ViT-B-32   | True              | False              | 57  | 0.4561          | 0.4561         | 0.7018        | 0.8947        | 0.4737            | 0.7018                       | 0.1034                  | 0.5263                       | 0.9649                        | 0.193                      | 0.3333                       | 0.05607                |
| directional_57 | cic_top3_repair_incl_ocr         | open_clip | ViT-B-32   | True              | False              | 57  | 0.4386          | 0.4386         | 0.7895        | 0.8947        | 0.4912            | 0.7193                       | 0.1366                  | 0.5439                       | 0.9298                        |                            |                              |                        |
| directional_57 | cic_top1_best_global_op_excl_ocr | open_clip | ViT-B-32   | True              | False              | 57  | 0.4035          | 0.4035         | 0.7719        | 0.8947        | 0.4211            | 0.6842                       | 0.07333                 | 0.4912                       | 0.8421                        |                            |                              |                        |
| directional_57 | matched_random_proposal_repair   | open_clip | ViT-B-32   | True              | False              | 57  | 0.1754          | 0.1754         | 0.7368        | 0.8772        | 0.2456            | 0.4737                       | 0.005123                | 0.2807                       | 0.6491                        |                            |                              |                        |
| directional_57 | largest_region_repair            | open_clip | ViT-B-32   | True              | False              | 57  | 0.386           | 0.386          | 0.7368        | 0.8421        | 0.4561            | 0.6842                       | 0.1081                  | 0.5439                       | 0.8246                        |                            |                              |                        |
| directional_57 | ocr_proposal_repair              | open_clip | ViT-B-32   | True              | False              | 49  | 0.1837          | 0.1837         | 0.7755        | 0.8776        | 0.2245            | 0.5918                       | 0.01813                 | 0.3061                       | 0.7551                        |                            |                              |                        |
| strict_39      | original_clip_prediction         | open_clip | ViT-B-32   | True              | False              | 39  | 0               | 0              | 0.8205        | 0.8718        | 0                 | 0                            | 0                       | 0                            | 0                             |                            |                              |                        |
| strict_39      | oracle_text_box_repair           | open_clip | ViT-B-32   | True              | True               | 39  | 0.4615          | 0.4615         | 0.8974        | 1             | 0.5641            | 1                            | 0.1406                  | 0.7179                       | 0.9744                        |                            |                              |                        |
| strict_39      | oracle_best_global_op            | open_clip | ViT-B-32   | True              | True               | 39  | 0.4359          | 0.4359         | 0.8974        | 0.9487        | 0.5385            | 0.9487                       | 0.0943                  | 0.6923                       | 1                             |                            |                              |                        |
| strict_39      | cic_top1_repair_excl_ocr         | open_clip | ViT-B-32   | True              | False              | 39  | 0.5385          | 0.5385         | 0.7436        | 0.8974        | 0.5641            | 0.6923                       | 0.2047                  | 0.5897                       | 0.9487                        | 0.1282                     | 0.359                        | 0.05897                |
| strict_39      | cic_top3_repair_excl_ocr         | open_clip | ViT-B-32   | True              | False              | 39  | 0.5385          | 0.5385         | 0.7949        | 0.8974        | 0.5897            | 0.7692                       | 0.1827                  | 0.6154                       | 0.9487                        |                            |                              |                        |
| strict_39      | cic_top1_repair_incl_ocr         | open_clip | ViT-B-32   | True              | False              | 39  | 0.5641          | 0.5641         | 0.7692        | 0.8974        | 0.5897            | 0.7179                       | 0.2047                  | 0.6154                       | 0.9487                        | 0.1795                     | 0.3333                       | 0.05521                |
| strict_39      | cic_top3_repair_incl_ocr         | open_clip | ViT-B-32   | True              | False              | 39  | 0.5385          | 0.5385         | 0.8205        | 0.8974        | 0.6154            | 0.7692                       | 0.1827                  | 0.6154                       | 0.9487                        |                            |                              |                        |
| strict_39      | cic_top1_best_global_op_excl_ocr | open_clip | ViT-B-32   | True              | False              | 39  | 0.5128          | 0.5128         | 0.7949        | 0.9231        | 0.5128            | 0.7436                       | 0.17                    | 0.6154                       | 0.8205                        |                            |                              |                        |
| strict_39      | matched_random_proposal_repair   | open_clip | ViT-B-32   | True              | False              | 39  | 0.2051          | 0.2051         | 0.7692        | 0.8718        | 0.2821            | 0.4872                       | 0.005174                | 0.2821                       | 0.7179                        |                            |                              |                        |
| strict_39      | largest_region_repair            | open_clip | ViT-B-32   | True              | False              | 39  | 0.4615          | 0.4615         | 0.7949        | 0.8462        | 0.5385            | 0.7179                       | 0.1892                  | 0.5897                       | 0.8718                        |                            |                              |                        |
| strict_39      | ocr_proposal_repair              | open_clip | ViT-B-32   | True              | False              | 34  | 0.2647          | 0.2647         | 0.8235        | 0.9118        | 0.3235            | 0.7647                       | 0.04118                 | 0.3529                       | 0.8235                        |                            |                              |                        |

## Best examples to inspect

- 7:truck (cic-success, oracle op=expanded_gray_fill_1.25)
- 56:cow (cic-success, oracle op=gaussian_blur)
- 76:orange (cic-success, oracle op=gaussian_blur)
- 77:car (cic-success, oracle op=expanded_blur_1.25)
- 121:cell phone (cic-success, oracle op=gray_fill)
- 145:laptop (cic-success, oracle op=expanded_gray_fill_1.25)
- 170:pizza (cic-success, oracle op=expanded_gray_fill_1.25)
- 214:laptop (cic-success, oracle op=expanded_gray_fill_1.25)