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
- `real_text_shortcut`: a real review-like text classification sample with neutral source-marker shortcut injection. The checked-in default is a small reproducible sample, not a large benchmark.
- `random_aug_failure`: a localized metadata shortcut stress test where generic random text perturbations can miss the factor that CIC targets directly.
- `traffic_sign_shortcut`: an optional safety-critical-inspired traffic-sign shortcut audit. If GTSRB is unavailable or disabled, the runner writes an unavailable summary and does not claim real dataset validation.
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

The random augmentation failure stress test is designed to test whether generic instability is sufficient when the shortcut is localized and factor-specific. It does not show CIC dominates random augmentation in all settings.

The central claim remains two-axis: confidence measures uncertainty; counterfactual stability measures shortcut dependence. CIC complements confidence by detecting high-confidence shortcut failures, especially when a model relies on unstable shortcut features.

## CIC-Guided Repair

CIC-guided repair is a proof-of-concept. The strongest current result is not universal automatic correction, but targeted detection and selective abstention: CIC can identify high-confidence shortcut failures that random augmentation fails to detect.

### Hard Multi-Decoy CLIP Headline

On a held-out hard multi-decoy benchmark using real pretrained OpenCLIP, misleading text reduced accuracy to 25.0%. Non-oracle CIC region scoring repaired accuracy to 75.0%, compared with 33.1% for matched random text-region repair, while preserving no-overlay accuracy and keeping clean-safe accuracy drop to 1.0%. (These authoritative numbers come from `results/final_report/final_key_numbers.json`; an earlier exploratory `21.9% → 87.5% / 28.8%` figure from a prior prediction path is stale and is not used — see the discrepancy note in `FINAL_ARTIFACT_INDEX.md`.)

Conditional on this held-out test set, CIC top-1 exceeded the matched random text-region baseline by 41.9 percentage points. The reported ± uncertainty for the matched random baseline reflects random baseline draw variability, not full test-set sampling uncertainty.

The main hard multi-decoy result is a strong single-benchmark result. A previous lite two-seed pass showed deterministic behavior on a fixed benchmark instance, not true benchmark-resampling stability. Robustness across independently resampled hard benchmark instances remains a limitation.

This does not solve open-world shortcut discovery. The method searches a finite candidate class of text-region proposals. Localization is strongest at coarse IoU >= 0.3 and weak at strict IoU >= 0.5, so the result should be interpreted as coarse causal-region localization and repair, not exact bounding-box recovery.

CLIP repair evidence should be read as a ladder: oracle CLIP repair is upper-bound causal confirmation; single-overlay non-oracle repair is promising but has a competitive matched/random patch baseline; the first multi-decoy repair is not a true shortcut-failure benchmark because original misleading accuracy was high; hard multi-decoy repair is the main headline result.

### Scale and Multi-Model Replication Audit

As supporting evidence (this does **not** replace the frozen primary headline, which remains ViT-B-32 / laion2b_s34b_b79k at n=32 per condition: misleading 0.250 → CIC top-1 0.750 vs. 0.331 matched random, clean-safe drop 0.010), a scale-and-multi-model replication audit re-ran the hard multi-decoy text-overlay benchmark at n_per_condition = 128 across four real pretrained OpenCLIP backbone/checkpoint pairs. All 4/4 model/checkpoint pairs loaded (0 skipped, no fake backend) and all four were `repair_eligible`; the test suite passes (186 tests). All four models were evaluated on the **same** larger resampled benchmark instance (one shared benchmark hash) for a fair cross-model comparison, and this benchmark hash differs from the n=32 headline benchmark, so these numbers are a separate larger-n replication and are not a cell-for-cell restatement of the headline.

| Model | Pretrained tag | Original misleading | CIC top-1 | Matched random | CIC − random gap | Clean-safe drop | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-B-32 | laion2b_s34b_b79k | 0.289 | 0.742 | 0.305 | 0.437 | 0.036 | repair_eligible |
| ViT-B-32 | openai | 0.062 | 0.688 | 0.111 | 0.576 | 0.003 | repair_eligible |
| ViT-B-16 | laion2b_s34b_b88k | 0.078 | 0.938 | 0.159 | 0.778 | 0.000 | repair_eligible |
| RN50 | openai | 0.000 | 0.758 | 0.072 | 0.686 | 0.000 | repair_eligible |

The main text-overlay result is stable at larger n, and the text-overlay CIC effect replicates across multiple pretrained OpenCLIP backbones/checkpoints. This audit does **not** imply open-world shortcut discovery, general robustness, cross-shortcut generalization, or exact localization: the method still searches a finite candidate class of text-region proposals on a controlled synthetic text-overlay benchmark. Artifacts: `results/hard_multidecoy_scale_model_audit/scale_model_summary.md`, `scale_model_key_numbers.json`, `scale_model_metrics.csv`, `scale_model_plot.png`, `model_availability.csv`.

### Second Shortcut Family (Non-Text Semantic-Decoy Icon)

Positive supporting evidence — a second controlled finite-candidate shortcut family, **not** a replacement for the text-overlay headline. The same finite-candidate CIC region method was run on an independent non-text shortcut family: a central colored causal icon (the true label) plus a larger, spatially separated, competing-class corner icon (the decoy), with **no written words anywhere**. On real pretrained OpenCLIP ViT-B-32 / laion2b_s34b_b79k (fake backend blocked), the central icon is perfectly recognized (clean accuracy 1.000) while the decoy drives misleading-regime accuracy down to 0.297, and oracle removal of the decoy fully restores it (1.000). Using only pixels, candidate boxes, and model probabilities — no label, correctness, shortcut-type, or oracle-box leakage — CIC top-1 region repair recovers 0.711 (top-3 0.359, clean-safe 0.766), versus 0.258 for an area-matched random candidate region (CIC − random gap +0.453), with a 0.008 clean-regime drop under the validation-selected clean-safe policy.

The n=64 pilot passed 8/8 gates and the n=128 scale run also passed 8/8 gates (no failed gates), so the result is stable across scale. CIC also succeeds on this second controlled finite-candidate shortcut family under controlled oracle-intervention conditions. This does **not** imply open-world shortcut discovery, general robustness, cross-shortcut transfer, universal shortcut repair, or exact localization. The earlier flat visual-decoy pilot was not failure-rich enough (misleading accuracy ~0.58 exceeded the ≤ 0.40 failure gate) and is retained as boundary evidence; the semantic-decoy icon benchmark was the final pre-specified second-family attempt and passed all gates. Artifacts: `results/semantic_decoy_pilot/` (n=64), `results/semantic_decoy_scale_n128/` (n=128); boundary evidence in `results/visual_decoy_pilot/`.

### Cross-Shortcut Generalization Attempt

As a generalization probe, the text-overlay-selected CIC repair/scoring policy was frozen and applied, with no retuning, reselection, or reweighting, to a different finite-candidate shortcut family: a non-text colored-symbol watermark (`run_cross_shortcut_generalization`). The non-oracle scorer received only image pixels, CLIP predictions, class prompts, and candidate proposals; the true label and harmful bbox were used only to define the held-out failure subset, the oracle upper bound, and localization metrics.

The transfer attempt did not support cross-shortcut generalization. Pretrained OpenCLIP is far less susceptible to the non-text colored-symbol watermark than to text overlays: no-overlay accuracy stays near 1.0 and the misleading watermark only flips a minority of predictions, so the benchmark does not produce a clean repairable failure regime (oracle neutralization rarely restores the label on the flipped cases, and the failure-conditioned inclusion count is far below the n >= 30 threshold). The result is therefore reported honestly as a negative/limiting transfer result rather than an S+ generalization figure, and `cross_shortcut_headline_eligible` is false. This is a finite-candidate transfer test to one new shortcut family, not open-world shortcut discovery and not a claim of general robustness. The main claim stays centered on text-region finite-candidate repair.

### Motivation

The repair extension tests what can be done after a prediction lands in the dangerous quadrant: high confidence with low counterfactual stability. The conservative use case is to flag the example for human review. Automatic correction is reported separately.

### Automatic Repair Results

Real text repair shows that shortcut-neutralized prediction can correct dangerous-quadrant text examples in a controlled setting. The absolute gain is small because original accuracy was already high, so dangerous-quadrant repair success is separated from total accuracy gain.

Random augmentation failure repair has modest automatic-correction accuracy gains. It does not prove automatic repair is always successful.

CLIP repair is a real pretrained-model headline attempt only when `evidence_status` is exactly `pretrained CLIP repair evidence` and `headline_eligible` is true. If pretrained CLIP cannot load, the runner writes an unavailable summary and no fake repair metrics are eligible for headline tables.

Non-oracle CLIP shortcut localization is the stricter repair test. The runner generates sliding-window, high-frequency/textness, random-control, and center-object-control candidate regions from pixels only, ranks them with CIC-style prediction instability, and compares to the true overlay bbox only after ranking. Oracle overlay neutralization is reported only as an oracle upper bound. Non-oracle repair can be a headline only when pretrained CLIP loaded, held-out misleading examples are sufficient, localization succeeds above controls, repair or selective abstention passes the configured criteria, and clean/no-overlay accuracy drop is small.

### Selective Abstention Results

The random augmentation failure repair/abstention benchmark is the strongest repair-extension result: CIC can flag localized shortcut failures where generic perturbation is near chance. Selective risk curves compare confidence abstention, random augmentation abstention, and CIC abstention.

### What The Repair Extension Proves

When candidate shortcut interventions are available, counterfactual stability can guide targeted correction or human-review flags. CIC is strongest here as a failure detection and selective-abstention signal.

### What It Does Not Prove

This does not claim CIC dramatically repairs all failures, discovers arbitrary causal structure, dominates all baselines, works on pretrained CLIP unless pretrained CLIP actually loaded, or preserves clean accuracy unless a clean/aligned split was actually measured. Abstention is not counted as automatic correction.

### Pretrained CLIP Shortcut Repair Attempt

The CLIP overlay repair runner builds validation and held-out test splits for aligned, misleading, neutral, and no-overlay shape images. Prompt and neutralization strategy are selected on validation only. The known-bbox repair result is an oracle upper-bound causal confirmation unless the non-oracle discovery runner also succeeds. The final CLIP repair headline can only come from `results/nonoracle_clip_repair/nonoracle_clip_repair_metrics.csv` with `headline_eligible` true; otherwise CLIP repair is reported as an attempted real-model validation and is not used as headline evidence.

Repair runners:

```bash
python3 -m causal_reliability.experiments.run_clip_overlay_repair --config configs/clip_overlay_repair.yaml
python3 -m causal_reliability.experiments.run_nonoracle_clip_repair --config configs/nonoracle_clip_repair.yaml
python3 -m causal_reliability.experiments.run_real_text_repair --config configs/real_text_repair.yaml
python3 -m causal_reliability.experiments.run_random_aug_failure_repair --config configs/random_aug_failure_repair.yaml
```

Final scoped repair claim: CIC-guided repair is a proof-of-concept. The stronger result is that CIC can detect and flag high-confidence shortcut failures where confidence and random augmentation fail. When automatic repair is uncertain, CIC supports selective abstention or human review.

Final scoped claim: confidence measures uncertainty. Counterfactual stability measures shortcut dependence. CIC complements confidence by detecting high-confidence shortcut failures when label-preserving shortcut interventions are available, hypothesized, or discovered from a finite candidate set.

Related work positioning is summarized in `docs/related_work.md`. This project does not claim shortcut learning or counterfactual invariance are new; the contribution is an operational per-example reliability certificate, a two-axis reliability plane, baseline-tested shortcut-failure regimes, and an audit/repair workflow for candidate shortcut interventions.

## Theory and Mechanism Validation

A conditional, finite-candidate recovery theory for CIC shortcut repair is stated in `docs/theory.md`. It assumes an additive logit decomposition `logit_y(g(C,S,N)) = phi_y(C,N) + psi_y(S) + xi_y(C,S,N)`; because CLIP logits are inner products `logit_y(X) = <u(X), v_y>`, the assumption holds iff the embedding shift caused by a shortcut is approximately input-independent. The embedding-additivity validation experiment tests this on real pretrained OpenCLIP:

```bash
python3 -m causal_reliability.experiments.run_embedding_additivity_validation --config configs/embedding_additivity_validation.yaml
```

The theorem remains a conditional explanation, but embedding-additivity validation did not support applying the *global* form directly to the current OpenCLIP text benchmark (`embedding_additivity_supported_for_text = false`): the shortcut embedding shift clusters by shortcut value above the shuffled baseline and oracle neutralization repairs the prediction, but the per-image delta clusters more tightly by object class than by shortcut value, so the shortcut direction is not input-independent. The colored-symbol watermark transfer failure is consistent with a weak or flat shortcut channel, not a clean repairable shortcut (`embedding_additivity_supported_for_watermark = false`). Outputs are written to `results/embedding_additivity/`.

Global input-independent additivity is, however, *stronger* than the recovery theorem requires. The **final theory gate** tests the weaker **per-input residual-to-clean class-balance** premise: after neutralization the repaired logits should differ from the clean/causal logits by an approximately class-independent residual for each individual image (residual-to-clean `rho_y(x) = ell_y(T(x)) - ell_y(x_clean)`, with `max_y |rho_y(x) - mean_y rho(x)| <= epsilon_B`), with recovery guaranteed when the clean causal margin satisfies `m_clean(x) > 2*epsilon_B`. The residual is defined relative to the clean logits, not the misleading input logits, because a shift that is class-balanced relative to the misleading logits would preserve the misleading argmax rather than recover the clean causal argmax (see `docs/theory.md`, Section 9 and the per-input corollary):

```bash
python3 -m causal_reliability.experiments.run_per_input_class_balance_validation --config configs/per_input_class_balance_validation.yaml
```

On real pretrained OpenCLIP, oracle and CIC neutralization are substantially more class-balanced (smaller per-input residual-to-clean logit shift) than matched random text-region neutralization, and the margin condition tracks repair success. The authoritative outcome — `per_input_class_balance_supported_for_text` (true/false/mixed) and `clip_theory_support_status` — is recorded in `results/per_input_class_balance/per_input_class_balance_key_numbers.json`. A positive structural finding accompanies it: the typographic shortcut effect is **object-entangled** (it contains a real shortcut component but its direction varies with the underlying object), which is why generic global debiasing is unlikely to suffice while targeted per-input region scoring can still repair failures. The theory is conditional and finite-candidate; it does not claim open-world shortcut discovery, exact localization, or general robustness.

## Commands

```bash
python3 -m causal_reliability.experiments.run_synthetic --config configs/synthetic.yaml
python3 -m causal_reliability.experiments.run_vision --config configs/vision.yaml
python3 -m causal_reliability.experiments.run_colored_digits --config configs/colored_digits.yaml
python3 -m causal_reliability.experiments.run_real_model_validation --config configs/real_model_validation.yaml
python3 -m causal_reliability.experiments.run_real_text_shortcut_validation --config configs/real_text_shortcut_validation.yaml
python3 -m causal_reliability.experiments.run_random_aug_failure_benchmark --config configs/random_aug_failure_benchmark.yaml
python3 -m causal_reliability.experiments.run_clip_overlay_repair --config configs/clip_overlay_repair.yaml
python3 -m causal_reliability.experiments.run_real_text_repair --config configs/real_text_repair.yaml
python3 -m causal_reliability.experiments.run_random_aug_failure_repair --config configs/random_aug_failure_repair.yaml
python3 -m causal_reliability.experiments.run_traffic_sign_shortcut_validation --config configs/traffic_sign_shortcut_validation.yaml
python3 -m causal_reliability.experiments.run_text --config configs/text.yaml
python3 -m causal_reliability.experiments.run_tabular --config configs/tabular.yaml
python3 -m causal_reliability.validation.export_label_preservation_packet --config configs/label_preservation_packet.yaml
python3 -m causal_reliability.audit.run_cic_audit --config configs/example_cic_audit.yaml
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

## Practitioner Audit Workflow

The package exposes a small API:

```python
from causal_reliability.api import CICScorer, ReliabilityPlane
```

The audit demo writes certificates, reliability quadrants, recommended actions, and report files under `results/audit_demo/`. This is a practitioner-facing audit workflow for settings where candidate shortcut interventions can be specified. It is not a turnkey solution for arbitrary models or unknown shortcuts.

## Human Label-Preservation Packet

The validation packet exporter creates 10-20 original/counterfactual pairs per domain for colored digits, CLIP overlay, and real text shortcuts under `results/label_preservation_packet/`. The analyzer accepts a response CSV and computes label-preservation and plausibility agreement rates, per-domain rates, annotator/example/judgment counts, and simple pairwise agreement when multiple annotators exist. If no response CSV is provided, it reports that no human validation responses have been provided yet.

Expected human response columns are `annotator_id,example_id,original_label_human,counterfactual_label_human,label_preserved_human,plausible_human,concerns`. Run `bash scripts/analyze_human_validation.sh path/to/responses.csv` after collecting responses.

### Human label-preservation validation

Three annotators evaluated 100 original/repaired image pairs (300 total annotations). Majority vote found that the object label was preserved in 96 of 100 pairs and that the repaired image remained recognizable in 97 of 100 pairs. Inter-annotator agreement was high: Fleiss' kappa was 0.973 for before-label judgments, 0.974 for after-label judgments, 0.920 for whether the label changed, and 1.000 for after-image recognizability (percent agreement 0.980, 0.980, 0.993, and 1.000 respectively). The four preservation failures were retained and flagged rather than removed: one apparent shape change, one blurry/unrecognizable repair, one corrupted/glitched repair, and one blank/missing-shape repair.

The analyzer `validation/human_label_preservation/analyze_annotations.py` computes majority-vote rates, percent agreement, and Fleiss' kappa (implemented directly, no extra dependency) for the before-label, after-label, label-change, and recognizability fields directly from the raw per-annotator responses in `validation/human_label_preservation/completed_annotations/`. Before/after label accuracy against the true label is reported only when `metadata_hidden.csv` is present, and is otherwise left as n/a (no true labels are fabricated). Outputs: `results/human_label_preservation/` (`human_validation_summary.md`, `human_validation_metrics.json`, `human_validation_flags.csv`).

## WILDS Waterbirds metadata-only diagnostic (future work, not CIC repair)

WILDS Waterbirds was also parsed as a real spurious-background diagnostic (11,788 examples). A metadata-only OpenCLIP evaluation showed the expected background sensitivity: overall accuracy 56.0%, with land-background accuracy 73.1% versus water-background accuracy 35.9%, and landbird accuracy dropping from 74.4% on land backgrounds to 21.6% on water backgrounds. However, WILDS Waterbirds does not ship oracle-repairable bird/background masks or bounding boxes, so CIC repair and failure-conditioned oracle repair were not run. This diagnostic motivates a future regenerated CUB+Places Waterbirds-style benchmark with known masks (see [docs/regenerated_waterbirds_pilot.md](docs/regenerated_waterbirds_pilot.md)), but it is not a positive CIC repair result and is never headline evidence.

## Safety-Critical-Inspired Traffic Sign Audit

The traffic-sign runner is optional and conservative. It prefers GTSRB only when explicitly enabled and available; otherwise it writes `results/traffic_sign_shortcut/traffic_sign_summary.md` explaining that the validation is unavailable. This experiment is a safety-critical-inspired shortcut audit. It does not validate deployment in autonomous vehicles.

## Regenerated Waterbirds-Style Spurious-Background Pilot

Optional supporting evidence for finite-candidate CIC repair on a real
spurious-background failure mode. It regenerates a Waterbirds-like benchmark by
compositing CUB-200-2011 birds (with their pixel-perfect segmentation masks)
onto Places land/water backgrounds, which yields an oracle background
intervention (neutralize the background, keep the bird). It is **not** a
replacement for the main OpenCLIP text-overlay headline result and **not**
open-world discovery. See [docs/regenerated_waterbirds_pilot.md](docs/regenerated_waterbirds_pilot.md)
for the full method, the leakage rules, and the headline-eligibility gate.

Place the assets (large data is never downloaded automatically):

```
data/cub/CUB_200_2011/{images.txt,image_class_labels.txt,classes.txt,bounding_boxes.txt,images/}
data/cub/segmentations/<class>/<file>.png   # CUB segmentation masks
data/places/<scene_name>/<file>.jpg         # folders named by scene (lake, forest, ...)
```

Run it (skips cleanly if any asset is missing, writing
`results/regenerated_waterbirds_cic/waterbirds_regeneration_summary.md`):

```bash
python3 -m causal_reliability.experiments.run_regenerated_waterbirds_cic \
  --config configs/regenerated_waterbirds_cic.yaml
```

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
- `results/real_text_shortcut/`: real text shortcut metrics, summary, certificates, examples, and reliability-plane figures.
- `results/label_preservation_packet/`: human validation packet and optional response analysis.
- `results/audit_demo/`: practitioner CIC audit certificates, report, and reliability-plane figures.
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
