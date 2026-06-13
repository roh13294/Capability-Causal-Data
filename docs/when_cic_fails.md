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

Experiment/control: baseline comparison runner and final negative controls.

Remaining limitation: if an intervention is too broad, CIC can collapse into a generic robustness score.

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
