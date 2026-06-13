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

| Experiment                         | What it tests                                                                                                | Main result                                                                                                    | What it proves                                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Final validation regimes           | Whether confidence and CIC separate across confidence-solvable, confident-wrong, and mixed regimes           | Confidence is strongest in confidence-solvable failures; CIC is strongest in high-confidence shortcut failures | Confidence and counterfactual stability are complementary reliability axes                                     |
| Negative controls                  | Whether irrelevant, mismatched, or shortcut-preserving interventions weaken CIC                              | 24 / 24 controls passed                                                                                        | CIC depends on targeted, label-preserving shortcut interventions                                               |
| Colored digits baseline comparison | Whether a recognizable color-shortcut benchmark is detectable by CIC and simple heuristics                   | Random augmentation AUROC 0.983 versus CIC AUROC 0.951                                                         | CIC does not dominate every heuristic; generic instability can work when it disturbs the shortcut              |
| Candidate shortcut discovery pilot | Whether finite candidate interventions can rank the hidden shortcut without revealing metadata to the scorer | True shortcut ranked first in 3 controlled tasks                                                               | Discovery is promising but exploratory; discovered-CIC replacement remains task-dependent                      |
| CLIP text-overlay validation       | Whether a real pretrained vision-language model relies on text overlays over shape evidence                  | Misleading accuracy 0.167; confidence AUROC 1.000; CIC AUROC 1.000                                             | Shortcut reliance appears in a pretrained model, but this is not the clean confidence-vs-CIC separation result |

## Limitations

- Requires plausible shortcut-changing, label-preserving interventions.
- Does not solve unknown real-world causality.
- Controlled and semi-synthetic settings remain necessary for rigorous testing.
- In some regimes, confidence is better.
- Vision/text settings require careful calibration to avoid total collapse.

## Final Defensible Claim

Counterfactual Instability Certificates are not universal replacements for confidence. They are complementary reliability certificates for high-confidence shortcut failures.
