# Related Work Positioning

This project does not claim that shortcut learning, spurious correlations, or counterfactual invariance are new. Its contribution is an operational per-example reliability certificate, a two-axis reliability plane, baseline-tested shortcut-failure regimes, and an audit/repair workflow for candidate shortcut interventions.

## Shortcut Learning And Spurious Correlations

Shortcut learning and spurious correlations are well-established failure modes: a model can exploit background, texture, source, color, demographic, or annotation artifacts that correlate with labels in one environment and fail under shift. Benchmarks such as Colored MNIST, Waterbirds, and CelebA-style attribute shifts make this problem concrete by separating a stable label feature from a correlated nuisance feature.

## Counterfactual Invariance And Augmented Data

Counterfactual invariance and counterfactually augmented data ask whether predictions should remain stable under label-preserving changes. CIC uses that idea at evaluation time as a per-example certificate: if a hypothesized shortcut is changed while the label-defining content is preserved, a reliable prediction should stay stable. The project therefore builds on counterfactual testing rather than claiming to invent it.

## Robust Optimization Baselines

Methods such as Group DRO and IRM target invariance or group robustness during training. This repository includes baseline comparisons to contextualize CIC, but CIC is primarily a diagnostic and audit signal. It can be used with pretrained or fixed models when candidate interventions are available, rather than requiring retraining with group labels.

## CLIP Typographic Attacks

CLIP and related vision-language models are known to be vulnerable to typographic or text-overlay cues. The CLIP overlay experiments here use that phenomenon as a controlled shortcut benchmark. Oracle overlay neutralization is an upper bound; non-oracle region discovery is a finite-candidate localization test, not a claim of open-world causal discovery.

## Typographic-Attack Defenses And Disentangling Written Vs. Visual Concepts

Closest to the repair setting are defenses that try to separate written text from visual concepts in CLIP. Materzyńska, Torralba, and Bau (CVPR 2022, *Disentangling Visual and Written Concepts in CLIP*) study whether CLIP's image encoder entangles written words with natural-image concepts and propose representation-space procedures for isolating or suppressing spelling-related directions. CIC differs in objective and operating mode: it does not train or learn a global text-removal subspace, and it does not modify the model. Instead, CIC performs per-example, finite-candidate audit and repair by ranking candidate regions according to counterfactual effect. This distinction matters because our embedding-additivity validation finds that typographic shortcut shifts are object-entangled rather than a single global additive bias direction; the method therefore targets the per-input region whose neutralization most improves stability.

## Robustness And Augmentation Baselines

Generic augmentation, occlusion, entropy, confidence, and margin baselines can detect many failures, especially low-confidence or broadly corrupted examples. CIC is not claimed to dominate all baselines. Its intended advantage is in high-confidence shortcut failures where a targeted, label-preserving intervention exposes dependence on an unstable factor.

## Selective Prediction And Abstention

Selective prediction and abstention evaluate whether a model can defer on risky examples while maintaining high accuracy on covered examples. CIC-guided abstention uses counterfactual instability as one risk signal. Abstention is reported separately from automatic repair and should not be counted as corrected prediction.

## Theory And Mechanism Validation

Mechanistic accounts of when a localized intervention recovers a causal prediction typically assume some additive or modular structure in the model's scores. The finite-candidate CIC recovery theory (`docs/theory.md`) makes this explicit: it assumes an additive logit decomposition into a causal channel, a shortcut channel, and a small interaction residual. For CLIP, whose logits are inner products between an image embedding and class text embeddings, this reduces in its *global* form to the claim that the embedding shift caused by a shortcut is approximately input-independent. The embedding-additivity validation experiment tests that strong claim on real pretrained OpenCLIP and did not support applying it directly to the OpenCLIP text-overlay result (the per-image shortcut delta clusters more tightly by object class than by shortcut value, so the shortcut direction is not input-independent), while the colored-symbol watermark transfer failure is consistent with a weak or flat shortcut channel.

Global input-independent additivity is, however, stronger than the recovery theorem actually requires. A weaker, **per-input class-balance** premise suffices: after neutralization the residual shortcut contribution to the logits need only be approximately class-independent *within each individual image*, with recovery guaranteed when the clean causal margin exceeds twice the per-input class-balance error. The per-input class-balance validation experiment (`run_per_input_class_balance_validation.py`, `results/per_input_class_balance/`) is the final theory gate: on real pretrained OpenCLIP, oracle and CIC neutralization are substantially more class-balanced than matched random text-region neutralization, and the margin condition tracks repair success, with `clip_theory_support_status` recording the outcome. The accompanying positive structural finding is that the typographic shortcut effect is **object-entangled** — it carries a real shortcut component but its direction varies with the underlying object — which is why generic global debiasing is unlikely to suffice while targeted per-input region scoring can still repair failures. The theory is conditional and finite-candidate; it is not a claim of open-world shortcut discovery, exact localization, or general robustness.

## Cross-Shortcut Transfer And Generalization

A recurring question for shortcut-mitigation methods is whether a policy tuned on one shortcut family transfers to another without retuning. The cross-shortcut generalization test addresses a narrow version of this: a CIC repair/scoring policy selected on text-overlay shortcut failures is frozen and applied, with no retuning or reselection, to a different finite-candidate shortcut family — a non-text colored-symbol watermark. This is a finite-candidate transfer test to one new shortcut family, not open-world shortcut discovery and not a claim of general robustness. The frozen scorer inherits a "textness" weighting term whose effect on transfer is reported as an exploratory ablation rather than tuned. The result is only treated as supporting generalization evidence when the frozen policy beats a matched random region baseline (with a reported 95% CI) and preserves clean accuracy; otherwise the transfer attempt is reported honestly as unsupported, and the main claim stays centered on text-region finite-candidate repair.
