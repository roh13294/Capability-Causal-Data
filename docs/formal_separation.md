# Formal Separation: Confidence and Counterfactual Stability

This note states the conceptual separation used by the final experiments. It is not a universal dominance claim. It says that confidence and counterfactual stability can be informative in different distribution-shift regimes.

## Shortcut Definition

A shortcut is a feature that is predictive of the label in the training distribution but is not causally necessary for the true class. A model relies on a shortcut when changing that feature while preserving the true label changes the model's prediction.

A causal or stable feature determines the true label. A shortcut or spurious feature is correlated with the label but not label-defining. A counterfactual intervention is a label-preserving change to the shortcut feature.

## Setup

Let `f(x)` be a classifier and let `conf(x)` denote the model confidence in its predicted class. Let `I(x)` be a set of label-preserving shortcut interventions: each `x' in I(x)` changes a suspected shortcut factor while preserving the ground-truth label. Define the Counterfactual Instability Certificate score as

```text
CIC(x) = average instability of f over x' in I(x).
```

The instability can include prediction flips, margin collapse, or distributional change in the predictive distribution. Let

```text
F(x) = 1[f(x_shift) != y]
```

indicate failure under the shifted evaluation distribution.

## Proposition

Confidence and counterfactual stability are separable reliability signals. There exist distribution-shift regimes where confidence predicts failure and CIC adds little, and there exist regimes where confidence risk fails but CIC separates shortcut-dependent failures from stable predictions.

## Lemma: Confidence-Only Insufficiency For Shortcut Reliance

Setup: let `x = (c, s)`, with true label `y = g(c)`. The shortcut `s` is correlated with `y` in some environments but is not causally necessary for `y`. Let `u(f(x))` be any reliability score that depends only on the model's output probabilities on the observed input.

Claim: no confidence-only metric is sufficient to detect shortcut reliance.

Proof sketch: construct Model A and Model B. Model A predicts from `c`; Model B predicts from `s`. On the observed input, choose parameters so that both models output the same predicted class and the same probability vector, for example `[0.95, 0.05]` for class 0. Since `u` only sees this probability vector, it assigns the same reliability score to both models.

Now intervene on the shortcut: form `x' = (c, s')`, where `s'` changes the shortcut but preserves the causal content and therefore preserves `y`. Model A remains stable because `c` is unchanged. Model B changes its prediction, margin, or probability distribution because its decision used `s`. The two models therefore differ in shortcut reliance even though every confidence-only score assigns them the same score on `x`.

Conclusion: counterfactual stability provides information not contained in confidence. This does not imply that CIC detects every shortcut; it only establishes that confidence and shortcut stability are distinct reliability axes.

## Regime A: Confidence-Solvable Shift

In a confidence-solvable regime, shifted failures tend to be uncertain:

```text
E[conf(x) | F(x)=1] < E[conf(x) | F(x)=0].
```

Equivalently, confidence risk `1 - conf(x)` ranks failures above correct predictions. In this regime, a counterfactual shortcut intervention set may be unnecessary for failure detection. CIC can still carry signal, but the failure mode is already visible through ordinary uncertainty.

This is the empirical pattern in the final validation confidence-solvable regime: confidence AUROC is `1.000`, while CIC AUROC is lower, around `0.702` on average across the locked tasks.

## Regime B: Confident-Wrong Shortcut Shift

In a confident-wrong shortcut regime, the shifted examples remain familiar enough that a shortcut-reliant model can be highly confident while wrong:

```text
E[conf(x) | F(x)=1] >= E[conf(x) | F(x)=0].
```

Here confidence risk `1 - conf(x)` can become anti-predictive: the failures are not low-confidence warnings. If the shortcut intervention set `I(x)` targets the factor the model relies on, then failures can instead satisfy

```text
E[CIC(x) | F(x)=1] > E[CIC(x) | F(x)=0].
```

In that case, counterfactual instability separates shortcut-dependent failures from stable predictions even when confidence does not.

This is the empirical pattern in the final validation confident-wrong regime: confidence AUROC is about `0.285`, while CIC AUROC is about `1.000`.

## Mixed Regime

Mixed regimes contain both ordinary uncertainty and shortcut dependence. Confidence and CIC can both carry partial signal:

```text
E[1 - conf(x) | F(x)=1] > E[1 - conf(x) | F(x)=0]
```

for some failures, and

```text
E[CIC(x) | F(x)=1] > E[CIC(x) | F(x)=0]
```

for others.

The final validation mixed regime follows this pattern: confidence remains useful, and CIC also contributes signal.

## Takeaway

Confidence and counterfactual stability should be treated as two axes of reliability. Confidence measures the model's internal uncertainty about its current prediction. CIC measures whether that prediction is stable under label-preserving shortcut changes. The final claim is therefore complementary: Counterfactual Instability Certificates are most useful for high-confidence shortcut failures, not as replacements for confidence in confidence-solvable regimes.
