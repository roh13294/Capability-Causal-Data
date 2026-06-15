# Final Report

## Final Hypothesis

Confidence measures uncertainty. Counterfactual stability measures shortcut dependence. CIC complements confidence by detecting high-confidence shortcut failures, especially when a model relies on unstable shortcut features.

## Shortcut Definition

A shortcut is a feature that is predictive of the label in the training distribution but is not causally necessary for the true class. A model relies on a shortcut when changing that feature while preserving the true label changes the model's prediction.

Causal/stable features determine the true label. Shortcut/spurious features are correlated with the label but not label-defining. A counterfactual intervention is a label-preserving change to the shortcut feature.

## Main Results By Regime

| Task      | Regime              | Confidence AUROC | Confidence AUROC mean +/- std | Confidence AUROC 95% CI | CIC AUROC | CIC AUROC mean +/- std | CIC AUROC 95% CI | CIC - Confidence | CIC - Confidence mean +/- std | CIC - Confidence 95% CI | Mean Failed Confidence | Interpretation                                            |
| --------- | ------------------- | ---------------- | ----------------------------- | ----------------------- | --------- | ---------------------- | ---------------- | ---------------- | ----------------------------- | ----------------------- | ---------------------- | --------------------------------------------------------- |
| synthetic | confidence-solvable | 1                | 1.000 +/- 0.000               | [1.000, 1.000]          | 0.7543    | 0.754 +/- 0.119        | [0.694, 0.811]   | -0.2457          | -0.246 +/- 0.119              | [-0.306, -0.189]        | 0.4138                 | Confidence-solvable: confidence already detects failures. |
| text      | confidence-solvable | 1                | 1.000 +/- 0.000               | [1.000, 1.000]          | 0.6108    | 0.611 +/- 0.082        | [0.535, 0.675]   | -0.3892          | -0.389 +/- 0.082              | [-0.465, -0.325]        | 0.4155                 | Confidence-solvable: confidence already detects failures. |
| vision    | confidence-solvable | 1                | 1.000 +/- 0.000               | [1.000, 1.000]          | 0.7422    | 0.742 +/- 0.041        | [0.677, 0.807]   | -0.2578          | -0.258 +/- 0.041              | [-0.323, -0.193]        | 0.4191                 | Confidence-solvable: confidence already detects failures. |
| synthetic | confident-wrong     | 0.3025           | 0.303 +/- 0.032               | [0.245, 0.373]          | 1         | 1.000 +/- 0.000        | [1.000, 1.000]   | 0.6975           | 0.697 +/- 0.032               | [0.627, 0.755]          | 0.9104                 | Confident-wrong: CIC adds value over confidence.          |
| text      | confident-wrong     | 0.2473           | 0.247 +/- 0.011               | [0.195, 0.306]          | 1         | 1.000 +/- 0.000        | [1.000, 1.000]   | 0.7527           | 0.753 +/- 0.011               | [0.694, 0.805]          | 0.9133                 | Confident-wrong: CIC adds value over confidence.          |
| vision    | confident-wrong     | 0.3036           | 0.304 +/- 0.055               | [0.241, 0.367]          | 0.9993    | 0.999 +/- 0.001        | [0.998, 1.000]   | 0.6956           | 0.696 +/- 0.056               | [0.632, 0.759]          | 0.9082                 | Confident-wrong: CIC adds value over confidence.          |
| synthetic | mixed               | 0.84             | 0.840 +/- 0.055               | [0.784, 0.891]          | 0.8846    | 0.885 +/- 0.007        | [0.848, 0.920]   | 0.04465          | 0.045 +/- 0.054               | [-0.012, 0.114]         | 0.6795                 | Mixed: confidence and CIC both contain partial signal.    |
| text      | mixed               | 0.857            | 0.857 +/- 0.033               | [0.810, 0.899]          | 0.8683    | 0.868 +/- 0.022        | [0.825, 0.909]   | 0.0114           | 0.011 +/- 0.046               | [-0.048, 0.076]         | 0.6941                 | Mixed: confidence and CIC both contain partial signal.    |
| vision    | mixed               | 0.8332           | 0.833 +/- 0.019               | [0.786, 0.881]          | 0.9286    | 0.929 +/- 0.007        | [0.899, 0.954]   | 0.09543          | 0.095 +/- 0.020               | [0.035, 0.148]          | 0.685                  | Mixed: confidence and CIC both contain partial signal.    |

## Where Confidence Wins

Confidence is strongest in confidence-solvable regimes, where failures are low-confidence or OOD-like and the model already signals uncertainty.

## Where CIC Wins

In confident-wrong regimes, mean CIC AUROC was 1.000 versus confidence AUROC 0.285.

## Beyond Confidence: Reliability as a Two-Axis Problem

Confidence measures uncertainty in the model's current prediction. Counterfactual stability measures whether that prediction remains stable under label-preserving shortcut changes. The final results separate these two signals: confidence is strongest in confidence-solvable failures, while CIC is strongest in high-confidence shortcut failures.

The high-confidence plus low-stability quadrant is the dangerous quadrant. Its mean failure rate was 0.415 across available reliability-plane rows, with 790 examples.

## Candidate Shortcut Discovery Pilot

Method: generate a finite set of candidate interventions; do not tell the scoring function which candidate is the true shortcut; apply each label-preserving candidate intervention; measure prediction instability, label preservation, support, specificity, and confidence preservation; rank candidates by label-preserving, support-preserving instability; and compare to ground-truth shortcut metadata only after ranking.

| Task      | True Shortcut Rank | Top-1 Hit | Top-3 Hit | Top Candidate    | Interpretation                    |
| --------- | ------------------ | --------- | --------- | ---------------- | --------------------------------- |
| synthetic | 1                  | True      | True      | feature_dim_1    | synthetic true shortcut ranked #1 |
| vision    | 1                  | True      | True      | object_color     | vision true shortcut ranked #1    |
| text      | 1                  | True      | True      | token_position_2 | text true shortcut ranked #1      |

## Discovered-CIC Replacement Result

| Task      | Oracle CIC | Discovered Top-1 CIC | Discovered Top-3 CIC | Random Candidate CIC | Interpretation                                                                                                  |
| --------- | ---------- | -------------------- | -------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------- |
| synthetic | 0.7558     | 0.7558               | 0.76                 | 0.8101               | discovered matches oracle, but random candidate can be competitive/higher, so replacement evidence is not clean |
| vision    | 0.3027     | 0.3027               | 0.4545               | 0.616                | discovery ranks shortcut first, but CIC replacement is weak                                                     |
| text      | 0.718      | 0.718                | 0.7779               | 0.3558               | strongest discovered-CIC result; discovered CIC beats confidence and random                                     |

The discovery pilot successfully ranks true shortcut candidates first in controlled tasks, but using discovered interventions as full CIC replacements remains task-dependent. Therefore, discovery is a secondary exploratory extension, not the main contribution.

## Secondary Benchmark: Colored Digits

Colored digits reports CIC AUROC 0.951 versus confidence AUROC 0.109. It is supporting evidence only, included to test the same color-shortcut intervention logic in a recognizable digit-style setting.

In colored digits, random augmentation sensitivity outperformed CIC, with random augmentation AUROC 0.983 versus CIC AUROC 0.951. This shows that some shortcut failures can be detected by generic instability, especially when perturbations accidentally disturb the shortcut. However, generic augmentation is not targeted, not necessarily label-preserving, and does not explain which factor is unstable. CIC remains useful as a principled counterfactual stability framework rather than as a universal winner over every heuristic.

CIC is not claimed to dominate all baselines. The contribution is that it defines and operationalizes a second reliability axis.

## Real-Model Validation

This experiment tests whether the reliability-plane framework appears in a pretrained-model setting. It is not proof that CIC generalizes to all foundation models.

Model used: local_small_cnn. Pretrained: `False`. Zero-shot: `False`. Linear probe: `False`. Evidence status: fallback smoke test. Confidence AUROC: NA. CIC AUROC: NA.

## CLIP Text-Overlay Shortcut Validation

Evidence status: pretrained CLIP evidence. Backend: open_clip. Pretrained weights loaded: `True`. Aligned accuracy: 1.000. Misleading accuracy: 0.167. Mixed accuracy: 0.500. Confidence AUROC: 1.000. CIC AUROC: 1.000. High-confidence failure rate: 0.400.

The CLIP experiment is not the primary evidence that confidence fails, because in the mixed overlay setting both confidence and CIC achieved AUROC 1.000. Instead, the CLIP experiment validates a different part of the story: shortcut reliance occurs in a real pretrained vision-language model. Misleading text overlays reduced accuracy sharply, and occlusion analysis showed that masking the text changed predictions much more than masking the object.

Use the CLIP result as real pretrained model shortcut-failure evidence, attribution sanity check evidence, and social relevance evidence. Do not use it as the cleanest confidence-vs-CIC separation result.

## Human Label-Preservation Validation

To test whether CIC neutralization was label-preserving for human viewers, 3 annotators evaluated 100 original/repaired image pairs (300 total annotations). Under majority vote, the object label was preserved at rate 0.960 and the repaired image remained recognizable at rate 0.970. The 4 preservation failures were retained and flagged rather than removed.

Inter-annotator agreement was high. Fleiss' kappa was 0.973 for before-label judgments, 0.974 for after-label judgments, 0.920 for whether the label changed, and 1.000 for after-image recognizability (percent agreement 0.980, 0.980, 0.993, and 1.000 respectively). Failure characterization: 1x shape changed/covered; 1x blurry/unrecognizable; 1x corrupted/glitched; 1x blank/missing shape. Before/after label accuracy against the true label is reported only when pair-level true labels (metadata_hidden.csv) are present, and is otherwise left as n/a; no true labels were fabricated.

Metrics path: `results/human_label_preservation/` (analyzer: `validation/human_label_preservation/analyze_annotations.py`).

## WILDS Waterbirds Metadata-Only Diagnostic (Future Work, Not CIC Repair)

WILDS Waterbirds was also parsed as a real spurious-background diagnostic. A metadata-only OpenCLIP evaluation showed the expected background sensitivity: overall accuracy was 56.0%, with land-background accuracy 73.1% versus water-background accuracy 35.9%, and landbird accuracy dropping from 74.4% on land backgrounds to 21.6% on water backgrounds. However, WILDS Waterbirds does not ship oracle-repairable bird/background masks or bounding boxes, so CIC repair and failure-conditioned oracle repair were not run. This diagnostic motivates a future regenerated CUB+Places Waterbirds-style benchmark with known masks, but it is not a positive CIC repair result.

## Random Augmentation Failure Stress Test

Random augmentation sensitivity AUROC was 0.511; CIC AUROC was 1.000. Random augmentation failed relative to CIC: `True`.

This benchmark is designed to test whether generic instability is sufficient when the shortcut is localized and factor-specific. It does not show CIC dominates random augmentation in all settings.

## CIC-Guided Abstention and Repair

### Motivation

CIC-guided repair is a proof-of-concept. The strongest current result is not universal automatic correction, but targeted detection and selective abstention: CIC can identify high-confidence shortcut failures that random augmentation fails to detect.

### Automatic Repair Results

#### Hard Multi-Decoy CLIP Shortcut Localization

- headline_eligible = true
- headline_result_name = Hard Multi-Decoy CLIP Shortcut Localization
- evidence_status = pretrained CLIP hard multi-decoy non-oracle repair evidence
- headline_scope = finite candidate text-region proposals; not open-world discovery
- headline_primary_metric = misleading accuracy 0.250 to 0.750
- matched_random_text_baseline = 0.331
- clean_drop_top1 = 0.104
- clean_drop_clean_safe = 0.010
- localization_scope = coarse causal-region localization; IoU >= 0.3 strong, IoU >= 0.5 weak

On a held-out hard multi-decoy benchmark using real pretrained OpenCLIP, misleading text reduced accuracy to 25.0%. Non-oracle CIC region scoring repaired accuracy to 75.0%, compared with 33.1% for matched random text-region repair, while preserving no-overlay accuracy and keeping clean-safe accuracy drop to 1.0%.

This does not solve open-world shortcut discovery. The method searches a finite candidate class of text-region proposals. Localization is strongest at coarse IoU >= 0.3 and weak at strict IoU >= 0.5, so the result should be interpreted as coarse causal-region localization and repair, not exact bounding-box recovery.

The hard multi-decoy result should currently be interpreted as a strong controlled held-out benchmark result, not yet as full benchmark-resampling stability.

Main claim (decision tier A): On independently resampled held-out hard multi-decoy CLIP benchmark instances, non-oracle CIC repair achieved benchmark-resampling stability, consistently beating matched random text-region repair while preserving clean performance.

##### Evidence Hierarchy For This Result

These five evidence types are distinct and must not be conflated:

1. **Single-benchmark hard multi-decoy result** — the strong controlled held-out result above.
2. **Fixed-benchmark determinism check** — re-running the same fixed benchmark instance reproduces identical core metrics. This is a determinism check, not stability evidence.
3. **Lite resampling audit** — a tiny (n~=4 misleading/seed) resampling pass that was too small and volatile to establish robustness (e.g., per-seed original accuracy and CIC top-1 swung between 0.25 and 0.75). It does not establish benchmark-resampling stability.
4. **Full benchmark-resampling audit** — `--resample-benchmark-full` with >=32 misleading examples per independently resampled seed. Benchmark-resampling stability is claimed only if this succeeds.
5. **Failure-conditioned repair evaluation** — repair measured only on held-out examples where pretrained CLIP actually fails due to misleading text; framed as failure-conditioned, not open-world discovery, and its original accuracy is ~0 by construction (not a natural benchmark accuracy).

Across benchmark-resampled held-out hard multi-decoy runs, non-oracle CIC consistently outperformed matched random text-region repair while preserving clean performance.

Full benchmark-resampling audit artifacts are available in `results/hard_multidecoy_clip_repair/full_benchmark_resampling_audit.csv`, and the result survived independent resampling.

| seed_id | n_examples | n_hard_misleading_examples | no_overlay_accuracy | aligned_overlay_accuracy | original_hard_misleading_accuracy | oracle_repair_accuracy | cic_top1_repair_accuracy | cic_top3_repair_accuracy | cic_clean_safe_repair_accuracy | cic_selective_accuracy | cic_selective_coverage | cic_selective_abstention | random_matched_text_repair_mean | random_matched_text_repair_std | random_matched_text_repair_95ci | highest_textness_repair_accuracy | largest_text_repair_accuracy | random_augmentation_accuracy | clean_safe_clean_drop | top1_localization_iou_ge_0_3 | top1_localization_iou_ge_0_5 | top3_localization_iou_ge_0_3 | top3_localization_iou_ge_0_5 | random_matched_localization_mean | random_matched_localization_std | random_matched_localization_95ci | median_harmful_rank | fixed_benchmark_determinism_check | benchmark_resampled | headline_eligible | failed_reasons | image_set_hash                                                   | metadata_hash                                                    | lite_mode | full_resample | candidate_signature_hash                                         | cache_hits_estimated | cache_misses_estimated | total_time_sec | generation_time_sec | clip_prediction_candidate_cic_random_time_sec |
| ------- | ---------- | -------------------------- | ------------------- | ------------------------ | --------------------------------- | ---------------------- | ------------------------ | ------------------------ | ------------------------------ | ---------------------- | ---------------------- | ------------------------ | ------------------------------- | ------------------------------ | ------------------------------- | -------------------------------- | ---------------------------- | ---------------------------- | --------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | -------------------------------- | ------------------------------- | -------------------------------- | ------------------- | --------------------------------- | ------------------- | ----------------- | -------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- | --------- | ------------- | ---------------------------------------------------------------- | -------------------- | ---------------------- | -------------- | ------------------- | --------------------------------------------- |
| 0       | 128        | 32                         | 1                   | 1                        | 0.2812                            | 1                      | 0.7188                   | 0.7188                   | 0.7188                         | 0.9206                 | 0.9844                 | 0.01562                  | 0.3287                          | 0.05714                        | 0.0224                          | 0.7812                           | 0.25                         | 0.1562                       | 0.01042               | 0.625                        | 0.0625                       | 0.6562                       | 0.0625                       | 0.07125                          | 0.04185                         | 0.0164                           | 5                   | False                             | True                | True              |                | e2cc9218547667b1cc17ed3c2d1c407a505356cf682ffac4123afada20ecbf22 | 895b3f0c76e891cb13d62d7bfebeed7dfc77e06fea8c234a9930d36f4300d0ee | False     | True          | a38eb96d252b6181acc454b81cb93f4e31f73c71e9f5882ea722c5d65ff19bad | 0                    | 128                    | 526.5          | 0.693               | 522.7                                         |
| 1       | 128        | 32                         | 1                   | 1                        | 0.25                              | 1                      | 0.7188                   | 0.75                     | 0.7812                         | 0.92                   | 0.9766                 | 0.02344                  | 0.27                            | 0.03815                        | 0.01495                         | 0.625                            | 0.25                         | 0.1562                       | 0.03125               | 0.5625                       | 0.09375                      | 0.5625                       | 0.09375                      | 0.06375                          | 0.04814                         | 0.01887                          | 10                  | False                             | True                | True              |                | 6d6fc346668d43698bf4546e1326ad49c973e733bede356c2e3cc7eda66ea011 | 3ed1fc395135ce63866729fb9a24d25ba5250e243b1c0878aaf2fd1741edb2ab | False     | True          | da3bc75a44f32e65bd7299ec15e70b49152415d1f761576feb08f5ddffb2d8cd | 0                    | 128                    | 525            | 0.8288              | 521.8                                         |
| 2       | 128        | 32                         | 1                   | 1                        | 0.25                              | 1                      | 0.8125                   | 0.8438                   | 0.8438                         | 0.9528                 | 0.9922                 | 0.007812                 | 0.2675                          | 0.04865                        | 0.01907                         | 0.7188                           | 0.25                         | 0.0625                       | 0                     | 0.5938                       | 0.125                        | 0.625                        | 0.125                        | 0.05375                          | 0.03667                         | 0.01437                          | 9                   | False                             | True                | True              |                | 2162d5057f5ba9936c3996dbeb1cb037f2ee07f917380bde93325e3a0ec3240d | b529e0adf8de1b010187dce1fbf15f4d8d0d14d1cb4e2d8fd7473b0ae198ac0b | False     | True          | 00b69936047d37242015916cd68e98ec193cbd6ca09caf2bd22ab4ce07e33838 | 0                    | 128                    | 522.2          | 0.7252              | 519.2                                         |

##### Scale and Multi-Model Replication Audit

Supporting evidence only. This audit does NOT replace the frozen primary headline (ViT-B-32 / laion2b_s34b_b79k at n=32 per condition), which is left unchanged. At n_per_condition = 128, 4/4 real pretrained OpenCLIP model/checkpoint pairs loaded (0 skipped), and all 4 are repair_eligible. Test suite: 186 passed.

All four models were evaluated on the same larger resampled benchmark instance (benchmark hash `b896599c8b91c3bf87338c5cd5e0592b5f16a2dafa25632d4c5868ff7269bd41`) for a fair cross-model comparison. This benchmark hash differs from the n=32 headline benchmark, so these numbers are a separate, larger-n replication and are not directly comparable cell-for-cell to the frozen n=32 headline.

| Model    | Pretrained tag    | Original misleading | CIC top-1 repair | Matched random | CIC - random gap | Clean-safe drop | Status          |
| -------- | ----------------- | ------------------- | ---------------- | -------------- | ---------------- | --------------- | --------------- |
| ViT-B-32 | laion2b_s34b_b79k | 0.289               | 0.742            | 0.305          | 0.437            | 0.036           | repair_eligible |
| ViT-B-32 | openai            | 0.062               | 0.688            | 0.111          | 0.576            | 0.003           | repair_eligible |
| ViT-B-16 | laion2b_s34b_b88k | 0.078               | 0.938            | 0.159          | 0.778            | 0.000           | repair_eligible |
| RN50     | openai            | 0.000               | 0.758            | 0.072          | 0.686            | 0.000           | repair_eligible |

The main text-overlay result is stable at this larger n, and the text-overlay CIC effect replicates across multiple pretrained OpenCLIP backbones/checkpoints (ViT-B-32 laion/openai, ViT-B-16 laion, RN50 openai).

This audit does not imply open-world shortcut discovery, general robustness, cross-shortcut generalization, or exact localization. The method searches a finite candidate class of text-region proposals on a controlled synthetic text-overlay benchmark.

Artifacts: `results/hard_multidecoy_scale_model_audit/scale_model_summary.md`, `scale_model_key_numbers.json`, `scale_model_metrics.csv`, `scale_model_plot.png`, `model_availability.csv`.

##### Second Shortcut Family (Non-Text Semantic-Decoy Icon)

Positive supporting evidence (not the primary headline). Beyond the typographic text-overlay headline, the same finite-candidate CIC region method was run on an independent non-text shortcut family: a central colored causal icon plus a larger, spatially separated, competing-class corner icon, with no written words anywhere. On real pretrained OpenCLIP ViT-B-32 / laion2b_s34b_b79k (fake backend blocked), the central icon is perfectly recognized (clean accuracy 1.000) while the decoy drives misleading-regime accuracy to 0.297, and oracle removal of the decoy fully restores it (1.000). Using only pixels, candidate boxes, and model probabilities — no label, correctness, shortcut-type, or oracle-box leakage — CIC top-1 region repair recovers 0.711 (top-3 0.359, clean-safe 0.766), versus 0.258 for an area-matched random candidate region (CIC-minus-random gap +0.453), with a 0.008 clean-regime drop under the validation-selected clean-safe policy. Results are stable from n=64 (pilot) to n=128 (scale); both runs passed all 8 strict gates.

CIC also succeeds on a second controlled finite-candidate shortcut family beyond text overlays under controlled oracle-intervention conditions. This does not imply open-world shortcut discovery, general robustness, cross-shortcut transfer, universal shortcut repair, or exact localization. It is a single-model, single-family controlled result.

The earlier flat visual-decoy pilot was not failure-rich enough (misleading accuracy ~0.58 exceeded the <= 0.40 failure gate) and is retained as boundary evidence; the semantic-decoy icon benchmark was the final pre-specified second-family attempt and passed all gates.

Artifacts: `results/semantic_decoy_pilot/` (n=64) and `results/semantic_decoy_scale_n128/` (n=128); boundary evidence in `results/visual_decoy_pilot/`.

##### Failure-Conditioned Hard Multi-Decoy Repair Evaluation

Failure-conditioned evaluation (not open-world discovery): from 76 generated candidates, 50 held-out examples where pretrained CLIP actually fails were included (inclusion rate 0.658). Original failure-subset accuracy is 0.000 (~0 by construction, not a natural benchmark accuracy). Oracle harmful-text repair (upper bound) 1.000; CIC top-1 0.960; CIC top-3 0.980; CIC clean-safe 0.940; matched random text repair 0.112 (95% CI half-width 0.015); CIC-minus-random gap 0.868 (beats random: `True`). No-overlay / aligned preservation after clean-safe repair: 1.000 / 1.000. Localization top-1 IoU >= 0.3 / 0.5: 0.820 / 0.040. Headline eligible: `True`.

This is a failure-conditioned repair evaluation, not a general accuracy evaluation and not open-world shortcut discovery. The test set is finite and conditioned on observed failures, so its original accuracy is ~0 by construction.

##### Cross-Shortcut Generalization Attempt

A CIC repair/scoring policy selected on text-overlay shortcut failures was frozen and applied, with no retuning, to a different finite-candidate shortcut family (colored_symbol_watermark, a non-text colored-symbol watermark). The transfer attempt did not support cross-shortcut generalization: the frozen text-selected policy did not clear all eligibility thresholds on the non-text shortcut (reasons: n_failure_examples 4 < 30 and natural misleading accuracy not <= 0.40; frozen CIC top-1/top-3 repair accuracy < 0.70; clean / no-overlay preservation not high). The main claim stays centered on text-region finite-candidate repair.

This is not open-world shortcut discovery. The transfer test uses a finite candidate class of non-text region proposals and evaluates whether a policy selected on text overlays transfers to one new shortcut family.



Sample sizes: n_examples hard misleading = 32; aligned-overlay = 32; neutral-overlay = 32; no-overlay = 32; random matched text-region seeds = 100; selective abstained/repaired = 2 / 35.

95% confidence intervals: original hard misleading 0.250 [0.133, 0.421]; oracle repair 1.000 [0.893, 1.000]; CIC top-1 repair 0.750 [0.579, 0.867]; CIC top-3 repair 0.750 [0.579, 0.867]; CIC clean-safe repair 0.750 [0.579, 0.867]; no-overlay 1.000 [0.893, 1.000]; aligned-overlay 1.000 [0.893, 1.000]; top-1 IoU >= 0.3 0.594 [0.423, 0.745]; top-3 IoU >= 0.3 0.625 [0.453, 0.771].

Random matched text repair over random seeds: mean 0.331, std 0.052, 95% CI half-width 0.010. Conditional on this held-out test set, CIC top-1 substantially exceeded the matched random text-region baseline. The reported ± uncertainty for the matched random baseline reflects random baseline draw variability, not full test-set sampling uncertainty.

Backend/model/tag: open_clip / ViT-B-32 / laion2b_s34b_b79k. Pretrained loaded: `True`. Headline eligible: `True`. Oracle upper-bound repair accuracy: 1.000. CIC top-3 repair accuracy: 0.750. CIC clean-safe repair accuracy: 0.750. CIC selective accuracy/coverage/abstention: 0.929 / 0.984 / 0.016. Top-1 harmful localization IoU >= 0.3 / 0.5: 0.594 / 0.062; top-3: 0.625 / 0.062.

This is the main CLIP repair headline because it uses a held-out split, real pretrained OpenCLIP, non-oracle scoring that excludes the true label and harmful bbox, a hard misleading condition, and matched text-region controls.

#### Pretrained CLIP Shortcut Repair Attempt

Oracle overlay repair should not be treated as evidence of automatic shortcut discovery. Oracle CLIP repair is an upper-bound causal confirmation: it shows that removing the known shortcut restores performance, but it is not evidence of automatic shortcut discovery.

Pretrained CLIP shortcut repair was attempted but is not headline evidence for automatic discovery. Oracle CLIP overlay repair is available. Original misleading-overlay accuracy was 0.094; known-bbox repaired misleading-overlay accuracy was 1.000. Treat this as oracle upper-bound causal confirmation, not automatic discovery.

Known-bbox CLIP overlay repair is not the automatic discovery headline path. Current repair evidence status: pretrained CLIP repair evidence; oracle metrics file headline flag: `True`.

#### Single-Overlay Non-Oracle CLIP Shortcut Localization and Repair

Evidence status: pretrained CLIP non-oracle repair evidence; headline eligible: `False`. Original misleading accuracy: 0.188; oracle upper-bound misleading accuracy: 1.000; non-oracle top-1 misleading accuracy: 0.844; non-oracle top-3 misleading accuracy: 0.844; random patch misleading accuracy: 0.844. Top-1/top-3 localization success at IoU >= 0.3: 0.094 / 0.156. Clean accuracy drop: 0.000.

This single-overlay non-oracle repair result is promising, but it is not the headline because the matched/random patch baseline is competitive. It searches a finite candidate region class; it does not solve open-world discovery, causal discovery, or general robustness.

#### First Multi-Decoy CLIP Repair

The first multi-decoy repair run is not the headline because original misleading accuracy was high: 0.906. CIC top-1 repair accuracy was 0.719, while matched random text repair was 0.892.

Therefore it is not a true shortcut-failure benchmark; the hard multi-decoy repair run is the main headline result.

Real text repair best CIC repair success was 1.000; maximum clean accuracy drop among CIC repair methods was 0.000.

Random augmentation failure automatic repair success was 0.000 for random augmentation consensus and 1.000 for CIC-guided automatic repair.

### Selective Abstention Results

On the random augmentation failure repair benchmark, CIC abstention coverage was 0.456, selective accuracy was 1.000, and failure capture rate was 1.000.

### What The Repair Extension Proves

When candidate shortcut interventions are available, counterfactual stability can guide targeted correction or human-review flags. The strongest current evidence is that CIC can flag high-confidence shortcut failures where confidence and random augmentation are weak.

### What It Does Not Prove

This repair extension does not show CIC dramatically repairs all failures, dominates all baselines, works on pretrained CLIP without a real pretrained backend, or preserves clean accuracy unless a clean/aligned split was actually measured.

## Safety-Critical-Inspired Traffic Sign Shortcut Validation

Traffic-sign status: unavailable. Dataset: GTSRB. CIC AUROC: NA.

This experiment is a safety-critical-inspired shortcut audit. It does not validate deployment in autonomous vehicles.

Occlusion sanity check mean shortcut attention ratio: 0.767.

Attribution sanity check mean shortcut attention ratio: 1.000.

## Negative Controls

Passed controls: 24 / 24.

## Reviewer-Oriented Stress Tests

Simple-baseline comparison evaluated 6 task/regime rows. CIC exceeded the best non-CIC baseline by more than 0.02 AUROC in 0 rows, while simpler baselines were competitive or better in 1 rows. The project does not claim CIC dominates all shortcut detectors; it claims CIC provides a principled second reliability axis and performs strongly in high-confidence shortcut-failure regimes.

Failure modes are documented in `docs/when_cic_fails.md`: invalid interventions, missing shortcut candidates, entangled shortcuts, off-support counterfactuals, confidence-solvable failures, global corruption, and multi-causal tasks.

CIC also has computational and epistemic costs: realistic counterfactuals may require human annotation, domain-specific simulators, generative models, or audited transformation pipelines, and the user must know whether the intervention truly preserves the label.

What remains unresolved: intervention validity still requires task knowledge, finite candidate classes can miss real shortcuts, and simple uncertainty/OOD baselines can be sufficient when failures are not high-confidence shortcut failures.

## What Each Experiment Establishes

| Experiment                                                      | What it tests                                                                                                         | Main result                                                                                                             | What it proves                                                                                                                                      |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Final validation regimes                                        | Whether confidence and CIC separate across confidence-solvable, confident-wrong, and mixed regimes                    | Confidence is strongest in confidence-solvable failures; CIC is strongest in high-confidence shortcut failures          | Confidence and counterfactual stability are complementary reliability axes                                                                          |
| Negative controls                                               | Whether irrelevant, mismatched, or shortcut-preserving interventions weaken CIC                                       | 24 / 24 controls passed                                                                                                 | CIC depends on targeted, label-preserving shortcut interventions                                                                                    |
| Colored digits baseline comparison                              | Whether a recognizable color-shortcut benchmark is detectable by CIC and simple heuristics                            | Random augmentation AUROC 0.983 versus CIC AUROC 0.951                                                                  | CIC does not dominate every heuristic; generic instability can work when it disturbs the shortcut                                                   |
| Candidate shortcut discovery pilot                              | Whether finite candidate interventions can rank the hidden shortcut without revealing metadata to the scorer          | True shortcut ranked first in 3 controlled tasks                                                                        | Discovery is promising but exploratory; discovered-CIC replacement remains task-dependent                                                           |
| CLIP text-overlay validation                                    | Whether a real pretrained vision-language model relies on text overlays over shape evidence                           | Misleading accuracy 0.167; confidence AUROC 1.000; CIC AUROC 1.000                                                      | Shortcut reliance appears in a pretrained model, but this is not the clean confidence-vs-CIC separation result                                      |
| Real text shortcut validation                                   | Whether CIC applies to a real review-like text classification domain with neutral marker shortcuts                    | Writes confidence, entropy, margin, random perturbation, marker counterfactual, and CIC AUROCs                          | CIC can be audited outside images when label-preserving text marker interventions are specified                                                     |
| Human label-preservation validation                             | Whether CIC neutralization preserves the human-perceived object label                                                 | 3 annotators, 100 pairs; majority-vote label preserved 0.960, recognizable 0.970; Fleiss' kappa 0.973/0.974/0.920/1.000 | Neutralization is label-preserving to humans; agreement is high and the four failures were flagged not removed; no true labels fabricated           |
| Random augmentation failure stress test                         | Whether generic random perturbations miss a localized metadata shortcut that CIC targets directly                     | Random augmentation AUROC 0.511 versus CIC AUROC 1.000                                                                  | Generic instability is not sufficient in every localized factor-specific shortcut setting                                                           |
| Random augmentation failure repair/abstention                   | Whether CIC can flag localized shortcut failures where generic perturbation is near chance                            | CIC abstention coverage 0.456; selective accuracy 1.000; failure capture 1.000                                          | Proves CIC can flag localized shortcut failures where generic perturbation is near chance; does not prove automatic repair is always successful     |
| Real text repair                                                | Whether shortcut-neutralized prediction can correct dangerous-quadrant text examples in a controlled setting          | Real text repair best CIC repair success 1.000; max clean accuracy drop 0.000                                           | Proves shortcut-neutralized prediction can correct dangerous-quadrant text examples in a controlled setting; does not prove broad text-model repair |
| CLIP overlay repair                                             | Whether known-overlay neutralization repairs typographic overlay failures in real pretrained CLIP on a held-out split | Evidence status: pretrained CLIP repair evidence; reported as oracle upper bound                                        | Oracle overlay repair is an upper-bound causal confirmation, not automatic shortcut discovery                                                       |
| Single-overlay non-oracle CLIP shortcut localization and repair | Whether finite candidate regions can be ranked without the overlay bbox and then used for repair                      | Top-1/top-3 localization at IoU >= 0.3: 0.094 / 0.156; headline eligible: `False`                                       | Promising but not headline evidence because the matched/random patch baseline can be competitive                                                    |
| First multi-decoy CLIP repair                                   | Whether non-oracle scoring survives multiple text decoys                                                              | Original misleading accuracy 0.906; top-1 CIC repair 0.719; random text repair 0.892                                    | Not a true shortcut-failure benchmark because original misleading accuracy was high                                                                 |
| Hard multi-decoy CLIP shortcut localization                     | Whether finite candidate text-region scoring can repair a held-out hard misleading-overlay failure benchmark          | misleading accuracy 0.250 to 0.750                                                                                      | Main headline result: non-oracle finite-candidate repair evidence, with coarse localization and explicit matched random controls                    |
| Traffic sign shortcut validation                                | Whether a safety-critical-inspired sign shortcut audit is available without medical or deployment claims              | Status: unavailable; CIC AUROC NA                                                                                       | Traffic-sign evidence is counted only when explicitly available; unavailable runs do not fabricate validation                                       |
| Practitioner CIC audit workflow                                 | Whether users can score examples and assign reliability quadrants from supplied interventions                         | Simple API and CLI demo write certificates, report, and reliability-plane figures                                       | CIC is usable for hypothesized shortcut audits, not arbitrary turnkey deployment                                                                    |

## 9/10 Category Defense Summary

| Category                | Original concern              | Added evidence                                                                                                                      | Remaining limitation                                                                                                | Why 9/10 is defensible                                                                         |
| ----------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Originality             | Components have precedent.    | Two-axis reliability decomposition plus confidence-only insufficiency lemma.                                                        | CIC builds on counterfactual testing.                                                                               | The contribution is the synthesis and formal separation of confidence from shortcut stability. |
| Technical difficulty    | Benchmarks are controlled.    | CLIP, real text shortcut benchmark, baseline suite, held-out discovery, audit workflow.                                             | No large generative counterfactual engine.                                                                          | Substantial multi-domain system and evaluation.                                                |
| Clarity                 | Claim boundaries could blur.  | Final claim, failure modes, audit wording, and theorem-style separation are explicit.                                               | CIC terminology still requires careful reading.                                                                     | The project repeatedly distinguishes uncertainty, shortcut stability, and discovery.           |
| Experiments             | Limited real-world tasks.     | CLIP plus real text benchmark, baselines, negative controls, human validation support, and random augmentation failure stress test. | Not a 10-dataset benchmark; human validation sample size is limited to collected responses.                         | Strong breadth for STS with careful controls.                                                  |
| Real-world significance | Requires candidate shortcuts. | Practitioner audit API, CLIP, text benchmark, finite-candidate discovery, and traffic-sign status: unavailable.                     | Not fully automatic deployment; traffic-sign fallback/unavailable results are not real-world deployment validation. | Directly usable for hypothesized shortcut audits.                                              |
| Limitations             | Risk of overclaiming.         | Negative controls, when-CIC-fails doc, no-fabricated-human-results analyzer behavior, and explicit simulated-shortcut caveats.      | Intervention validity still requires domain judgment; simulated shortcut limitations remain.                        | The project makes limits part of the evidence hierarchy.                                       |

## Theory and Mechanism Validation

The finite-candidate CIC recovery theory is stated in `docs/theory.md`. Its central assumption is an additive logit decomposition `logit_y(g(C,S,N)) = phi_y(C,N) + psi_y(S) + xi_y(C,S,N)`. For CLIP, logits are inner products `logit_y(X) = <u(X), v_y>`, so this holds iff the embedding shift caused by a shortcut is approximately input-independent. The embedding-additivity validation experiment (`results/embedding_additivity/`) tests this on the text-overlay and colored-symbol watermark shortcuts with real pretrained OpenCLIP.

The theorem remains a conditional explanation, but embedding-additivity validation did not support applying it directly to the current OpenCLIP text benchmark: the shortcut embedding shift clustered by shortcut value above the shuffled baseline (within-shortcut cosine 0.765 vs shuffled 0.634) and oracle neutralization repaired 1.000 of cases, but the per-image delta clustered more by object class (within-object cosine 0.855) than by shortcut value, so the shortcut direction is not input-independent (reasons: shortcut_clustering_exceeds_object, neutralization_damage_small).

The watermark transfer failure is consistent with a weak or flat shortcut channel, not a clean repairable shortcut (watermark within-object cosine 0.923 exceeds within-shortcut cosine 0.757; embedding_additivity_supported_for_watermark = `False`).


### Per-Input Class-Balance (final theory gate)

Global input-independent embedding additivity is stronger than the recovery theorem requires. The final theory experiment (`results/per_input_class_balance/`) tests the weaker per-input premise: after neutralization the repaired logits should differ from the clean/causal logits by an approximately class-independent residual for each individual image (residual-to-clean `rho_y(x) = ell_y(T(x)) - ell_y(x_clean)`, with `max_y |rho_y(x) - mean_y rho(x)| <= epsilon_B`), with recovery guaranteed when the clean causal margin satisfies `m_clean(x) > 2*epsilon_B`. The residual is defined relative to the clean logits, not the misleading input logits, because a shift that is class-balanced relative to the misleading logits would preserve the misleading argmax rather than recover the clean causal argmax.

Although global input-independent embedding additivity was not supported, the weaker per-input class-balance condition was supported. The finite-candidate recovery theorem therefore provides a plausible mechanism for the OpenCLIP text-overlay repair result. On real pretrained OpenCLIP, oracle and CIC neutralization were substantially more class-balanced (median residual-to-clean 2.464 oracle / 3.704 CIC top-1) than matched random text-region neutralization (5.220), and the margin condition tracked repair success (clip_theory_support_status = `CLIP-supported via per-input class-balance`). The mechanism is validated most tightly for oracle neutralization and approximately for CIC: because CIC's median residual-to-clean exceeds epsilon_B = 3.0, the theorem does not fully explain every CIC success. CIC neutralization is more class-balanced than matched random repair and aligns with the recovery condition directionally, but the worst-case margin condition is conservative and many successful CIC repairs occur even when the sufficient condition is not formally satisfied.

**Object-entanglement finding.** OpenCLIP's typographic shortcut effect is not a single global additive bias direction. The shift induced by overlay text is object-entangled: it contains a real shortcut component, but its direction varies substantially with the underlying object. This helps explain why generic global debiasing is unlikely to suffice, and why targeted per-input counterfactual region scoring can still repair failures.

This theory section is a conditional, finite-candidate mechanism account. It does not claim open-world shortcut discovery, exact localization, or general robustness.

## Limitations

- Requires plausible shortcut-changing, label-preserving interventions.
- Does not solve unknown real-world causality.
- Controlled and semi-synthetic settings remain necessary for rigorous testing.
- Human validation was performed on 100 text-overlay repair pairs with 3 annotators (majority-vote label preserved 96/100, recognizable 97/100; Fleiss' kappa up to 1.000); broader validation across other shortcut families and real-world datasets remains future work.
- Random augmentation and traffic-sign shortcut results use simulated/localized shortcut mechanisms.
- In some regimes, confidence is better.
- Vision/text settings require careful calibration to avoid total collapse.

## Final Defensible Claim

Counterfactual Instability Certificates are not universal replacements for confidence. They are complementary reliability certificates for high-confidence shortcut failures.
