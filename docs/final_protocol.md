# Locked Final Protocol

This document is the locked final analysis plan for *Beyond Confidence: Counterfactual Stability as a Second Axis of Neural Network Reliability*.

## Final Claim

Reliability is two-dimensional. Confidence measures uncertainty; counterfactual stability measures shortcut dependence. CIC complements confidence by detecting high-confidence shortcut failures, not ordinary confidence-solvable failures.

The final claim was refined after stress testing. The project does not claim CIC always beats confidence; instead, it shows that confidence and counterfactual stability measure different reliability axes.

## Shortcut Definition

A shortcut is a feature that is predictive of the label in the training distribution but is not causally necessary for the true class. A model relies on a shortcut when changing that feature while preserving the true label changes the model's prediction.

Examples include background color predicting class without defining the object, a text overlay saying "square" on an image of a circle, or a marker word correlating with a text label without being part of the actual rule.

- causal/stable feature: a feature that determines the true label.
- shortcut/spurious feature: a feature correlated with the label but not label-defining.
- counterfactual intervention: a label-preserving change to the shortcut feature.

## Main Method

For each prediction, compute confidence and a Counterfactual Instability Certificate over a specified finite intervention set. The intervention set changes shortcut features while preserving the task label in controlled benchmark settings. CIC is compared with confidence risk, entropy, margin, old ShiftRisk, and label-flip-only ablations as failure-ranking scores.

## Main Regimes

- confidence-solvable: failures tend to be low-confidence or OOD-like, so confidence should remain a strong detector.
- confident-wrong: shifted failures can remain high-confidence because the shifted shortcut values are in support but remapped.
- mixed: both confidence and counterfactual instability contain partial signal.

## Final Datasets And Tasks

- synthetic vector task with known causal and shortcut coordinates.
- generated vision shapes with controlled color/background/texture shortcuts.
- rule-based text shortcut task.
- tabular proxy task for semi-synthetic structured shortcuts.
- colored digits benchmark as secondary evidence if available. It uses sklearn digits when installed and a generated digit-like fallback otherwise.

## Final Metrics

- ID accuracy and shifted accuracy.
- mean failed confidence and high-confidence failure fractions.
- failure AUROC for confidence risk, entropy, negative margin, old ShiftRisk, label-flip-only, and CIC.
- high-confidence CIC AUROC when both failures and correct examples exist in the high-confidence subset.
- seed count, mean +/- std across seeds, bootstrap 95% CI for AUROC when valid, and paired bootstrap 95% CI for CIC AUROC minus confidence AUROC.

AUROC is not reported as meaningful when all examples are correct, all examples fail, or there are too few failures/correct examples for the intended subset.

## Baselines

The locked baselines are confidence risk, entropy, negative margin, old ShiftRisk, label-flip-only CIC ablation, shuffled or mismatched counterfactual controls, stability training, and task-specific negative controls.

## Seeds

Final validation uses the seeds listed in `configs/final_validation.yaml`. Seeded summaries report the number of distinct seeds included. Single-run supporting benchmarks, including colored digits, are labeled as single-seed supporting evidence unless explicitly repeated.

## Evidence Status

Main evidence:

- final validation by regime
- negative controls
- reliability plane
- qualitative examples

Secondary evidence:

- shortcut discovery pilot
- discovered-CIC comparison
- colored digits benchmark if added

Exploratory only:

- unknown shortcut discovery
- moonshot extension

## Supporting Evidence

Evidence supports the final claim when CIC improves failure ranking over confidence in confident-wrong regimes, the dangerous quadrant has elevated shifted failure rate, negative controls weaken appropriately, and qualitative examples show high-confidence low-stability failures that are not simply low-confidence uncertainty.

## Weakening Evidence

Evidence weakens the final claim when confidence matches or beats CIC in the confident-wrong regime, CIC performs similarly on shuffled or irrelevant controls, high-confidence low-stability examples do not fail more often, or the result only holds for one synthetic generator.

## Negative Controls

Negative controls must include interventions that preserve the shortcut, shuffle or mismatch counterfactuals, perturb irrelevant features, or otherwise break the intended label-preserving shortcut change. These controls should reduce CIC's failure-ranking advantage.

## Exploratory-Only Results

Candidate shortcut discovery searches a finite candidate intervention class in controlled settings. It is an exploratory extension and does not solve general causal discovery.

The pilot method is locked as follows: generate a finite candidate intervention set; do not tell the scoring function which candidate is the true shortcut; apply each label-preserving candidate intervention; measure prediction instability, label preservation, support, specificity, and confidence preservation; rank candidates by label-preserving, support-preserving instability; and compare to ground-truth shortcut metadata only after ranking.

Current controlled results rank the true shortcut first for synthetic, vision, and text tasks. Discovered-CIC replacement is weaker: synthetic discovered CIC matches oracle but random candidates can be competitive or higher, vision replacement is weak even though discovery ranks the shortcut first, and text is the strongest discovered-CIC case. Discovery remains secondary exploratory evidence, not a replacement for the main oracle-intervention experiments.

## Limitations

CIC depends on the quality of the intervention set. It does not discover arbitrary unknown causes, does not guarantee deployment reliability, and does not replace confidence. Confidence is expected to win in confidence-solvable failures. Positive results are evidence for a reliability axis under controlled shortcut interventions, not proof of general unknown causality.
