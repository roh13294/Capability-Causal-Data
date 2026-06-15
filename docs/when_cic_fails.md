# When CIC Fails

Counterfactual Instability Certificates are only meaningful when the intervention is label-preserving, plausible, and factor-specific.

This document records the main failure modes for interpreting counterfactual stability. The purpose is to bound the claim: CIC complements confidence by measuring shortcut dependence under valid interventions; it is not a universal uncertainty score, shortcut detector, or causal discovery method.

## Invalid Intervention

What goes wrong: the counterfactual change accidentally changes the true label. For example, changing object shape while claiming the label is preserved invalidates a shape-label vision task.

Detection or mitigation: interventions are defined against task metadata, and negative controls test mismatched or label-damaging interventions.

Experiment/control: counterfactual mismatch controls, final negative controls, and task-specific intervention definitions.

Remaining limitation: in real data, label preservation may require domain judgment that the benchmark cannot fully automate.

## Missing Shortcut Candidate

What goes wrong: the true shortcut is not represented in the candidate intervention class, so CIC can remain low even when the model uses a different unstable feature.

Detection or mitigation: the project separates oracle shortcut interventions from candidate shortcut discovery and reports discovery as an exploratory extension.

Experiment/control: unknown shortcut discovery and discovered-CIC replacement analyses.

Remaining limitation: CIC can only test the intervention family supplied to it.

The practitioner API and audit demo make this explicit: the user supplies candidate shortcut interventions. The workflow is useful for hypothesized or finite-candidate shortcut audits, not for turnkey arbitrary deployment.

Repair consequence: CIC-guided repair cannot neutralize a shortcut that is absent from the candidate intervention set. In this case a repair policy may keep the wrong prediction or abstain, and that failure should be reported.

## Entangled Shortcuts

What goes wrong: shortcut and causal features cannot be independently edited. Changing the shortcut also damages the causal feature, so instability no longer isolates shortcut reliance.

Detection or mitigation: interventions are kept factor-specific in synthetic, vision, text, and colored-digits tasks; qualitative examples inspect whether edits preserve the object or token-level causal content.

Experiment/control: colored digits, qualitative examples, and negative controls.

Remaining limitation: real-world factors are often entangled, and factor-specific editing may be imperfect.

## Off-Support Counterfactual

What goes wrong: the intervention creates unrealistic examples. The model may become unstable because the input is implausible, not because a shortcut has been isolated.

Detection or mitigation: final validation emphasizes in-support shortcut flips and separates OOD-like regimes from confident-wrong shortcut regimes.

Experiment/control: final validation across confidence-solvable, confident-wrong, and mixed regimes.

Remaining limitation: plausibility is benchmark-dependent and may not transfer to arbitrary natural data.

## Confidence-Solvable Failures

What goes wrong: failures are already low-confidence, so confidence, entropy, or margin detects them without counterfactual stability.

Detection or mitigation: the final report explicitly includes confidence-solvable regimes where confidence wins.

Experiment/control: final validation confidence-solvable regime and metric audit.

Remaining limitation: CIC should not be presented as a replacement for confidence in these settings.

## Global Corruption

What goes wrong: perturbations destabilize the model by damaging the whole input rather than isolating a shortcut feature.

Detection or mitigation: generic augmentation sensitivity and negative controls compare CIC against non-targeted perturbations.

Experiment/control: baseline comparison runner, final negative controls, and the random augmentation failure stress test. The stress test is designed to test whether generic instability is sufficient when the shortcut is localized and factor-specific. It does not show CIC dominates random augmentation in all settings.

Remaining limitation: if an intervention is too broad, CIC can collapse into a generic robustness score.

Repair consequence: if the repair counterfactual damages non-shortcut content, the repaired prediction can become worse than the original. Clean accuracy drop and non-abstained accuracy should be reported alongside repair success.

## Multi-Causal Tasks

What goes wrong: the label depends on several features, so changing one apparently shortcut-like feature may not preserve the label under a nuanced causal graph.

Detection or mitigation: the project uses controlled tasks with explicit causal/shortcut metadata and limits claims for real-world causal structure.

Experiment/control: formal protocol, controlled final validation, and tabular proxy caveats.

Remaining limitation: multi-causal real tasks require stronger causal specification than this project attempts.

## Counterfactual Generation Cost

What goes wrong: CIC requires generating or evaluating label-preserving counterfactuals. This is cheap for controlled settings such as color, text overlays, and simple token edits, but can be expensive in realistic image, language, or medical settings.

Realistic counterfactuals may require human annotation, domain-specific simulators, generative models, or carefully audited transformation pipelines.

The cost is not only computational. It is also epistemic: one must know whether the intervention truly preserves the label.

Detection or mitigation: use CIC first in high-risk domains with known candidate shortcuts; use finite candidate intervention classes; audit label preservation; report when interventions are off-support or invalid; and treat automatic discovery as exploratory unless validated.

Experiment/control: final protocol intervention audits, negative controls, candidate discovery scoring audits, and explicit fallback/unavailable evidence labels.

Remaining limitation: high-quality counterfactual generation can dominate the practical cost of applying CIC outside controlled benchmarks.

## Repair-Specific Failure Modes

What goes wrong: a shortcut-neutralized counterfactual can still be misclassified, or multiple counterfactuals can disagree without producing a stable consensus.

Detection or mitigation: report repaired correctness, automatic correction success, abstention rate, coverage, selective accuracy, failure capture rate, false abstention rate, and accuracy on non-abstained examples. Count abstention separately from corrected predictions.

Clean accuracy drop should only be reported when a true clean/aligned split exists. If no such split is available, the metric should be null with an explicit reason rather than inferred from a shifted shortcut subset.

Experiment/control: CLIP overlay repair, real text shortcut repair, and random augmentation failure repair.

CLIP-specific caveat: known-bbox overlay neutralization is an oracle upper-bound causal confirmation. It is not evidence that the system discovered the shortcut region. The non-oracle CLIP repair runner separately tests whether finite candidate regions can be ranked without the overlay bbox or true label, and only that runner can support an automatic localization-and-repair headline.

Remaining limitation: CIC-guided repair is a proof-of-concept. The strongest current result is not universal automatic correction, but targeted detection and selective abstention: CIC can identify high-confidence shortcut failures that random augmentation fails to detect. It is not general model editing, open-world causal discovery, broad text-model repair, pretrained CLIP repair evidence without `headline_eligible`, or deployment certification.

## Cross-Shortcut Transfer

What goes wrong: a policy (score threshold, consensus threshold, scoring weights) tuned to one shortcut family may not transfer to a different shortcut family. The text-overlay scorer even carries an inherited "textness" weighting term, which has no reason to help a non-text shortcut and could hurt.

Detection or mitigation: the cross-shortcut generalization runner freezes the text-overlay-selected CIC policy and applies it, with no retuning, reselection, or reweighting, to a different finite-candidate shortcut family — a non-text colored-symbol watermark. It reports both a natural benchmark (Mode A) and a failure-conditioned transfer evaluation (Mode B), compares against a matched random region baseline with a 95% CI, and is only `cross_shortcut_headline_eligible` if the frozen policy actually beats matched random repair while preserving clean accuracy. An exploratory "frozen policy without textness" ablation reports whether the inherited textness term helps or hurts transfer, without tuning its weight. If the frozen policy fails to transfer, the runner says so and is not headline eligible.

Experiment/control: cross-shortcut generalization (`run_cross_shortcut_generalization`); matched random region repair, highest-contrast, largest-region, center-object, and random-augmentation controls.

Remaining limitation: this is a finite-candidate transfer test to one new shortcut family. It is not open-world shortcut discovery and not a claim of general robustness. The non-oracle scorer never sees the true label, harmful shortcut bbox, shortcut type, or correctness; those are used only to define the held-out failure subset, the oracle upper bound, and localization metrics.

## Weak Or Flat Shortcut Channel (Theory And Mechanism Validation)

What goes wrong: neutralizing a shortcut region can only repair a prediction if there is a strong, separable shortcut channel to cancel. If the shortcut channel is weak or flat — the model's score barely depends on the shortcut value — then CIC has little to neutralize, and repair (or transfer of a repair policy) will look like it failed even though nothing is broken. Separately, the recovery theorem in `docs/theory.md` assumes an additive logit decomposition whose shortcut term has an approximately input-independent direction; if the per-image shortcut shift is actually object-dependent, the additive-channel assumption is only partially met and the theorem should not be promoted to a mechanism claim for the model.

Detection or mitigation: the embedding-additivity validation experiment (`run_embedding_additivity_validation`, `results/embedding_additivity/`) measures, on real pretrained OpenCLIP, whether the shortcut embedding shift clusters by shortcut value (above a shuffled-label baseline) more tightly than by object class, whether neutralization stays close to the clean embedding, and whether a margin condition predicts repair success. It emits `embedding_additivity_supported_for_text` and `embedding_additivity_supported_for_watermark`, and flags `watermark_shortcut_channel_weak` when the watermark channel is weak/flat.

Observed result: for the hard multi-decoy text overlay, the shortcut delta clusters by shortcut value above the shuffled baseline and oracle neutralization repairs the prediction, but the per-image delta clusters more tightly by object class than by shortcut value, so the *global* additivity form is not supported. For the colored-symbol watermark, the shortcut channel is weak/flat, which is consistent with — and explains — the cross-shortcut transfer negative result rather than refuting the theory.

Weaker premise (final theory gate): global input-independent additivity is stronger than the recovery theorem requires. The per-input class-balance validation (`run_per_input_class_balance_validation`, `results/per_input_class_balance/`) tests the weaker premise — after neutralization the repaired logits need only differ from the clean/causal logits by an approximately class-independent residual *within each image* (residual-to-clean `rho_y(x) = ell_y(T(x)) - ell_y(x_clean)`, with `max_y |rho_y(x) - mean_y rho(x)| <= epsilon_B`), with recovery guaranteed when the clean causal margin satisfies `m_clean(x) > 2*epsilon_B`. On real pretrained OpenCLIP, oracle and CIC neutralization are substantially more class-balanced than matched random text-region neutralization, and the margin condition tracks repair success; `per_input_class_balance_supported_for_text` and `clip_theory_support_status` record the outcome. The structural reason the global form fails while the per-input form can hold is **object-entanglement**: the typographic shortcut effect carries a real shortcut component but its direction varies with the underlying object, so generic global debiasing is unlikely to suffice even though targeted per-input region scoring can still repair failures.

Remaining limitation: this validation is a finite-candidate, controlled test on the text-overlay (and watermark contrast) shortcut families and one pretrained CLIP model. It does not establish open-world shortcut discovery, exact localization, or general robustness.

## Safety-Critical-Inspired Shortcuts

What goes wrong: it is easy to overstate a simulated shortcut audit as deployment evidence.

Detection or mitigation: the optional traffic-sign runner reports whether GTSRB was actually used. If GTSRB is unavailable or disabled, it writes an unavailable summary. A synthetic fallback, when explicitly enabled, is labeled as synthetic safety-critical-inspired evidence.

Experiment/control: optional traffic-sign shortcut validation.

Remaining limitation: This experiment is a safety-critical-inspired shortcut audit. It does not validate deployment in autonomous vehicles.
