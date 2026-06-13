# Causal Reliability Certificates

This project tests whether counterfactual prediction instability can certify when neural networks are "right for the wrong reason." The experiments show that confidence is sufficient for some low-confidence/OOD-like failures, but can fail in high-confidence shortcut failures. Counterfactual Instability Certificates target that blind spot by measuring whether predictions remain stable under label-preserving shortcut interventions.

## Beyond Confidence

This project argues that model reliability should not be treated as one-dimensional. Confidence and counterfactual stability capture different failure modes. Confidence detects ordinary uncertainty; Counterfactual Instability Certificates detect shortcut dependence, especially when a model is confidently wrong.

## Why This Matters

Neural networks can be accurate in-distribution while relying on features that are unstable across environments. Accuracy tells us whether a model was right. Causal reliability asks whether it was right for a reason that is expected to survive when shortcut features change.

This is a rigorous research framework for shortcut reliance, not a claim to solve distribution shift or replace OOD detection.

## Shortcut Definition

A shortcut is a feature that is predictive of the label in the training distribution but is not causally necessary for the true class. A model relies on a shortcut when changing that feature while preserving the true label changes the model's prediction.

Examples include:

- background color predicts class but does not define the object
- text overlay says "square" on an image of a circle
- a marker word correlates with a text label but is not part of the actual rule

This project uses three distinct terms:

- causal/stable feature: a feature that determines the true label.
- shortcut/spurious feature: a feature correlated with the label but not label-defining.
- counterfactual intervention: a label-preserving change to the shortcut feature.

## Problem

For each input `x`, the model outputs:

1. prediction
2. confidence
3. Counterfactual Instability Score
4. CIC Reliability
5. old ShiftRisk as an ablated baseline

The certificate estimates whether the prediction remains stable when shortcut features are changed while stable/causal features are preserved.

## Certificate Definition

Let `f(x)` output logits `z(x)`, and let `y_hat = argmax_k z_k(x)`.

The logit margin is:

```text
m(x) = z_yhat(x) - max_{j != yhat} z_j(x)
```

For shortcut-changing counterfactuals `x' in I(x)`:

```text
D_margin(x, x') = max(0, m(x) - m(x'))
D_flip(x, x') = 1[argmax f(x) != argmax f(x')]
D_JS(x, x') = JS(softmax(f(x)), softmax(f(x')))
```

Old ShiftRisk is:

```text
ShiftRisk(x) =
  alpha * mean(D_margin)
+ beta  * quantile_90(D_margin)
+ gamma * mean(D_JS)
+ delta * mean(D_flip)
```

Defaults are `alpha=1.0`, `beta=1.0`, `gamma=0.5`, `delta=1.0`.

```text
CR(x) = exp(-ShiftRisk(x))
```

The final certificate is the Counterfactual Instability Certificate (CIC). The user-facing score is the Counterfactual Instability Score, with CIC Reliability reported as its stability-oriented counterpart. High CIC Reliability means the prediction is stable under shortcut-changing counterfactuals. Low CIC Reliability means the prediction is fragile under this intervention set.

## Why Margin Collapse Is Primary

Label flips are interpretable but sparse: a model can become fragile long before the argmax changes. Logit-margin collapse measures degradation in decision support under counterfactual shortcut changes, while JS divergence provides a distributional supporting signal.

## Difference From Confidence And OOD Detection

Confidence asks how strongly the model prefers its current prediction. OOD detection asks whether an input looks unlike training data. Causal reliability asks whether the prediction is sensitive to known shortcut interventions that should preserve the label.

## Paradigm-Level Takeaway

Reliability should be evaluated on two axes:

1. How confident is the model?
2. Would the prediction remain stable under label-preserving shortcut changes?

The resulting reliability plane has four quadrants:

| Confidence | Counterfactual Stability | Interpretation |
| --- | --- | --- |
| High | High | Reliable prediction |
| Low | High | Uncertain but causally stable |
| Low | Low | Generally fragile |
| High | Low | Dangerous shortcut reliance |

The dangerous quadrant is not merely low confidence. It is the high-confidence, low-stability case where a shortcut-reliant model can be confidently wrong.

## Datasets

- `synthetic`: vector data with known causal and shortcut features.
- `vision`: generated shape/color images where shape determines label and color is shortcut.
- `colored_digits`: recognizable colored digit-style shortcut benchmark. It uses `sklearn.datasets.load_digits` when scikit-learn is installed; otherwise it uses a self-contained generated seven-segment digit-like fallback. This is supporting evidence, not the main claim.
- `real_model_validation`: controlled shortcut images evaluated with a real pretrained vision or vision-language model when available. If CLIP or pretrained torchvision weights are unavailable, the code falls back to explicitly marked non-pretrained/local modes; those fallback results are not headline pretrained evidence.
- `text`: rule-based token classification where shortcut words shift.
- `tabular`: semi-synthetic proxy benchmark inspired by health, finance, and education settings.

The tabular task is a proxy demonstration, not a deployment-ready medical or financial model.

## Counterfactual Generation

Counterfactuals are deterministic, rule-based, and small:

- vector: change shortcut coordinate
- vision: recolor object pixels while preserving shape mask
- text: replace shortcut token
- tabular: swap proxy variable while preserving stable features

## Experiments

Each experiment:

1. sets deterministic seeds
2. generates data
3. trains ERM
4. evaluates ID and shifted accuracy
5. computes certificates
6. evaluates failure prediction
7. trains a stability-regularized model
8. saves CSVs and plots under `results/`

## Baselines

Implemented baselines include confidence risk, entropy, negative margin, a simple feature-distance OOD score helper, counterfactual augmentation, stability training, invariant-style variance penalty, and group DRO utilities.

## Comparison to Simple Heuristics

CIC is compared against uncertainty metrics, generic augmentation sensitivity, occlusion-style shortcut heuristics, and OOD/embedding distance. This comparison is meant to contextualize CIC, not force it to win everywhere. Confidence, entropy, margin, OOD distance, or simple occlusion can be competitive or better when failures are low-confidence, globally corrupted, or already separable by generic instability.

In colored digits, random augmentation sensitivity outperformed CIC, with random augmentation AUROC 0.9829 versus CIC AUROC 0.9512. This shows that some shortcut failures can be detected by generic instability, especially when perturbations accidentally disturb the shortcut. However, generic augmentation is not targeted, not necessarily label-preserving, and does not explain which factor is unstable. CIC remains useful as a principled counterfactual stability framework rather than as a universal winner over every heuristic.

CIC is not claimed to dominate all baselines. The contribution is that it defines and operationalizes a second reliability axis.

The central claim remains two-axis: confidence measures uncertainty; counterfactual stability measures shortcut dependence. CIC complements confidence by detecting high-confidence shortcut failures, especially when a model relies on unstable shortcut features.

## Commands

```bash
python3 -m causal_reliability.experiments.run_synthetic --config configs/synthetic.yaml
python3 -m causal_reliability.experiments.run_vision --config configs/vision.yaml
python3 -m causal_reliability.experiments.run_colored_digits --config configs/colored_digits.yaml
python3 -m causal_reliability.experiments.run_real_model_validation --config configs/real_model_validation.yaml
python3 -m causal_reliability.experiments.run_text --config configs/text.yaml
python3 -m causal_reliability.experiments.run_tabular --config configs/tabular.yaml
python3 -m causal_reliability.experiments.run_ablation --config configs/ablation.yaml
python3 -m causal_reliability.experiments.run_baseline_comparison --config configs/baseline_comparison.yaml
python3 -m causal_reliability.analysis.qualitative_examples --results_dir results
```

```bash
bash scripts/quickstart.sh
bash scripts/run_core.sh
bash scripts/run_all.sh
bash scripts/run_ablation.sh
bash scripts/run_real_model_validation.sh
```

## Real-Model Validation

To reduce dependence on custom benchmarks, the project includes a controlled shortcut task evaluated with a real pretrained vision or vision-language model when available. The goal is to test whether high-confidence shortcut failures can also occur outside the custom-trained models, and whether counterfactual stability provides complementary evidence under label-preserving shortcut flips.

The runner prefers CLIP zero-shot, then a torchvision pretrained classifier with a linear probe, and finally a local non-pretrained fallback. The output summary states exactly which model was used and whether pretrained weights were actually loaded. This experiment supports the two-axis reliability framing; it does not prove that CIC generalizes to all foundation models.

The CLIP text-overlay experiment is not the primary evidence that confidence fails, because in the mixed overlay setting both confidence and CIC achieved AUROC 1.000. Instead, it validates a different part of the story: shortcut reliance occurs in a real pretrained vision-language model. Misleading text overlays sharply reduced accuracy, and occlusion analysis showed that masking the text changed predictions much more than masking the object. It should be read as real pretrained model shortcut-failure evidence, attribution sanity check evidence, and social relevance evidence, not as the cleanest confidence-vs-CIC separation result.

## Outputs

For each task, results include:

- `train_metrics.csv`
- `test_metrics.csv`
- `certificates.csv`
- `failure_prediction.csv`
- `reliability_bins.csv`
- `summary.csv`
- `plots/accuracy_by_environment.png`
- `plots/reliability_vs_failure.png`
- `plots/roc_failure_prediction.png`
- `plots/confidence_vs_reliability.png`
- `plots/shift_risk_histogram.png`
- `plots/risk_decile_failure.png`
- `plots/reliability_calibration.png`

Phase 3 aggregation writes:

- `results/main_results_table.csv`
- `results/main_results_table.md`
- `results/main_results_summary.json`
- `results/sts_main_figure.png`
- `results/sts_main_figure.pdf`

The final concept package writes:

- `results/reliability_plane/reliability_plane_summary.md`
- `results/shortcut_discovery/shortcut_discovery_summary.md`
- `results/concept_figure.png`
- `results/concept_figure.pdf`
- `results/concept_figure_caption.md`

Paper/poster-ready artifacts are organized under:

- `docs/final_protocol.md`: locked final analysis plan and evidence hierarchy.
- `results/final_report/`: final narrative report, key numbers, and claim table with seed counts and CIs where valid.
- `results/colored_digits/`: secondary colored digit benchmark metrics, certificates, summary, and plots.
- `results/qualitative_examples/`: reliability-plane quadrant examples and image/table panels.
- `results/sts_main_figure.*`, `results/concept_figure.*`, `results/moonshot_figure.*`: presentation-ready figures.

The STS main figure has four panels: Panel A compares ERM in-distribution and shifted accuracy under in-support shortcut flips, Panel B shows confidence for correct versus failed shifted examples, Panel C compares failure-prediction ROC curves, and Panel D evaluates AUROC inside the high-confidence subset.

## Phase 4: Confident-Wrong Shortcut Failure

Earlier stress tests showed an important boundary condition: confidence, entropy, and margin can beat ShiftRisk when shifted examples become low-confidence or OOD-like. That result should not be hidden. It sharpens the claim: causal reliability is most valuable for high-confidence shortcut failures, not ordinary OOD failures where confidence already works.

Phase 4 adds `shift_mode: in_support_flip`. Training uses familiar shortcut values, such as class 0 with red and class 1 with blue. Shifted test examples still use red and blue, but the class-shortcut mapping flips. The examples are familiar in shortcut space, so a shortcut-reliant model can remain confidently wrong.

This differs from `ood_new_shortcut`, where shifted examples use shortcut values not seen during training. In that setting, confidence may already detect failure because the input is unfamiliar. The stricter final standard is conditional: CIC should add value in high-confidence shortcut failures, especially among examples with confidence at least 0.8 or 0.9, while confidence may remain the best detector in confidence-solvable regimes.

Run Phase 4 with:

```bash
bash scripts/run_phase4_quick.sh
```

Key outputs:

- `results/confident_wrong/confident_wrong_metrics.csv`
- `results/confident_wrong/confident_wrong_failure_prediction.csv`
- `results/confident_wrong/confident_wrong_high_conf_subset.csv`
- `results/metric_audit/metric_audit_summary.md`
- `results/negative_controls/negative_control_metrics.csv`
- `results/sts_main_figure.png`
- `results/main_results_table.md`

Interpret `metric_audit_summary.md` as a guardrail. It flags tasks where confidence already solves failure prediction, tasks where ShiftRisk adds value, and cases where AUROC is undefined because there are too few failures or too few correct examples.

## Phase 5: Calibrated Confident-Wrong Evaluation

Total shifted collapse is not enough because AUROC becomes undefined when every shifted example fails. Phase 5 adds `shift_mode: partial_in_support_flip`, where shortcut values are still familiar from training but only part of the class-shortcut mapping flips. This creates the stronger benchmark: some shifted examples remain correct, some fail, and many failures can still be high-confidence.

Partial flips matter because they test ranking, not just collapse. The high-confidence subset matters because Causal Reliability Certificates are most useful when confidence does not already warn us. Shuffled controls matter because true shortcut counterfactuals should beat controls that preserve labels, preserve shortcut values, match confidence, or perturb irrelevant directions.

Run Phase 5 with:

```bash
bash scripts/run_phase5_quick.sh
```

Key outputs:

- `results/partial_flip_sweep/partial_flip_metrics.csv`
- `results/certificate_ablation/certificate_ablation_metrics.csv`
- `results/negative_controls/negative_control_metrics.csv`
- `results/negative_control_diagnosis/diagnosis_summary.md`
- `results/sts_main_figure.png`
- `results/main_results_table.md`

Strong evidence means shifted accuracy lands between 0.2 and 0.7, there are many high-confidence failures, ShiftRisk AUROC is higher than confidence AUROC, true counterfactual ShiftRisk beats shuffled and matched controls, and the result holds across synthetic, vision, and text.

Weak evidence means all shifted examples fail, confidence predicts failure as well as ShiftRisk, shuffled controls perform nearly as well as true counterfactuals, or the effect disappears outside synthetic data.

## How To Interpret Results

The central table is `failure_prediction.csv`:

```text
Method | Failure AUROC | Top-Decile Failure Rate | Bottom-Decile Failure Rate | Risk Ratio
```

The final hypothesis is conditional: confidence detects ordinary uncertainty and many confidence-solvable failures, while CIC adds value in high-confidence shortcut failures.

## What Changed During The Research Process

- Initial ShiftRisk formula was too broad.
- Stress tests showed confidence sometimes outperformed ShiftRisk.
- Score diagnosis revealed full ShiftRisk could invert in some regimes.
- The certificate was redesigned around counterfactual prediction instability, especially label flips.
- Final claim became conditional and stronger.

## Final Defensible Claim

Counterfactual Instability Certificates are not universal replacements for confidence. They are complementary reliability certificates for high-confidence shortcut failures.

## Exploratory Extension

The candidate shortcut discovery pilot searches a finite candidate intervention class in controlled settings. It generates candidate interventions, ranks them by label-preserving, support-preserving prediction instability without telling the scorer which candidate is the true shortcut, and compares to ground-truth shortcut metadata only after ranking.

The pilot successfully ranks true shortcut candidates first in the controlled synthetic, vision, and text tasks, but using discovered interventions as full CIC replacements remains task-dependent. Therefore, discovery is a secondary exploratory extension, not the main contribution, and it does not solve general causal discovery.

## What Would Support The Refined Hypothesis

- CIC predicts shifted failure better than confidence, entropy, or margin in confident-wrong shortcut regimes
- high-confidence low-reliability examples fail often under shift
- stability training improves shifted accuracy without destroying ID accuracy
- effects replicate across synthetic, vision, text, and tabular proxy tasks
- results hold across seeds and ablations

## What Would Falsify The Hypothesis

- CIC performs no better than confidence in the confident-wrong shortcut regime
- low-reliability examples do not fail more often under shift
- certificate only works with perfectly known shortcuts
- stability training destroys ID accuracy

## Limitations

- Requires plausible shortcut-changing, label-preserving interventions.
- Does not solve unknown real-world causality.
- Controlled and semi-synthetic settings are necessary for rigorous testing.
- In some regimes, confidence is better.
- Vision/text settings require careful calibration to avoid total collapse.
- results only work on toy synthetic data

## Limitations

The certificate depends on the quality and relevance of `I(x)`, the counterfactual intervention set. If the shortcut intervention is misspecified, the score can be misleading. These experiments use generated settings where shortcut and causal features are known, so positive results should be interpreted as evidence for a mechanism, not as proof of broad deployment reliability.

## Quickstart

Install dependencies, then run:

```bash
python3 -m pytest
bash scripts/quickstart.sh
```

The quickstart runs the CPU-friendly synthetic benchmark and writes outputs to `results/synthetic/`.

Last verified quickstart output in this workspace:

```text
train_accuracy: 0.9765625
id_accuracy: 0.9765625
shifted_accuracy: 0.6796875
stability_shifted_accuracy: 1.0
worst_group_accuracy: 0.3888888888888889
mean_shift_risk: 0.475053608417511
reliability_ece: 0.2153733572922647
high_conf_low_reliability_failure_rate: 0.75
```

## Full Run

```bash
bash scripts/run_all.sh
```

The full run executes generated synthetic, vision, text, and tabular proxy benchmarks. GPU is used if available in the task configs.
