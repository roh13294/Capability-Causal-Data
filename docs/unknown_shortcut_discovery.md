# Candidate Shortcut Discovery

The original CIC experiments assume that the shortcut intervention is known. In that setting, CIC asks whether model predictions are stable under a counterfactual change to the known shortcut factor while the task label should stay fixed.

The candidate shortcut discovery extension relaxes that assumption in a controlled way. It searches over a finite candidate intervention class and ranks candidate factors by label-preserving prediction instability. The limited claim is:

Given a finite candidate intervention class, counterfactual instability can help identify candidate shortcut factors in controlled settings.

This is an exploratory extension. It is not general unknown causality, and it does not solve general causal discovery.

## Candidate Shortcut Discovery Pilot: Method

The pilot is self-contained and deliberately finite:

1. Generate a finite set of candidate interventions.
2. Do not tell the scoring function which candidate is the true shortcut.
3. For each candidate, apply a label-preserving candidate intervention.
4. Measure prediction instability, label preservation, support, specificity, and confidence preservation.
5. Rank candidates by label-preserving, support-preserving instability.
6. Compare to ground-truth shortcut metadata only after ranking.

The controlled result is that the true shortcut ranked first in synthetic, vision, and text tasks. The stronger replacement question is less clean: discovered interventions do not always behave like oracle CIC interventions, and random candidates can be competitive in some tasks.

## Algorithm

```text
Inputs:
  model f
  validation examples (x_i, y_i)
  candidate interventions I_1, ..., I_m
  task-specific label-preservation and support audits

For each candidate intervention I_j:
  1. Apply I_j to validation examples to produce x'_i.
  2. Compute original predictions f(x_i) and counterfactual predictions f(x'_i).
  3. Measure prediction instability:
       label flips, margin collapse, and distributional divergence.
  4. Estimate whether I_j should preserve the true label.
  5. Estimate whether I_j remains in support and is factor-specific.
  6. Score:
       instability
       times label-preservation rate
       times confidence-preservation
       times support score
       times specificity score.

Rank candidates by this score.
Select top-k candidates.
Compute discovered CIC using those top-k interventions.
Compare discovered CIC to oracle CIC, random candidate CIC, confidence risk,
entropy, negative margin, and the old ShiftRisk baseline.
```

## Scoring Intuition

Shortcut interventions should change a model that learned the shortcut, but they should not change the ground-truth label. The score therefore rewards interventions that destabilize predictions while preserving labels.

Causal feature interventions may also destabilize predictions, but changing causal features can change the true label. The label-preservation audit penalizes these candidates.

Corruptions such as blur, global noise, or broad brightness shifts may destabilize predictions without identifying a shortcut factor. The support and specificity audits penalize broad or low-support interventions.

## Assumptions

The true shortcut must be represented by at least one candidate intervention, or discovery should fail honestly.

The validation examples must be drawn from a controlled setting where label preservation can be audited or approximated for each candidate class.

The candidate interventions must be meaningful enough that a high score can be interpreted as evidence about a factor, not merely as arbitrary input damage.

The trained model must have learned enough of the shortcut for counterfactual instability to reveal it.

## Expected Failures

If there is no shortcut, no candidate should dominate strongly and CIC should not imply perfect shortcut discovery.

If multiple shortcuts are present, at least one true shortcut should ideally rank in the top-k; recovering every shortcut is a stronger requirement.

If a causal feature intervention is high-instability, it should be penalized by the label-preservation audit.

If a corruption intervention is high-instability, it should be penalized by support and specificity.

If the true shortcut is missing from the candidate intervention class, discovery should fail rather than claim a shortcut was found.

## Pilot Conclusion

The discovery pilot successfully ranks true shortcut candidates first in controlled tasks, but using discovered interventions as full CIC replacements remains task-dependent. Therefore, discovery is a secondary exploratory extension, not the main contribution.
