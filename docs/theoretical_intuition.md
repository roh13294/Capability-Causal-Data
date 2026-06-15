# Theoretical Intuition

This note gives a compact intuition for why confidence and counterfactual stability should be treated as separate reliability signals. It is not a full theorem proving causal discovery, and it does not claim that CIC always beats confidence.

## Setup

Let an input be `x = (c, s)`, where `c` is the causal feature and `s` is the shortcut feature. The true label is `y = g(c)`.

A model `f` may depend on `c`, on `s`, or on both. Let `conf(x)` be the model confidence in its current prediction. Let `I_s(x)` be a counterfactual intervention that changes `s` while preserving `c` and therefore preserving `y`. CIC measures prediction instability under `I_s`.

## Proposition 1: Confidence And Counterfactual Stability Are Separable

There exist regimes where confidence perfectly predicts failure and CIC is unnecessary, and regimes where confidence is uninformative or anti-predictive while CIC detects shortcut dependence.

### Constructive Example A: Confidence-Solvable Regime

Suppose failures are low-confidence. Correct examples have high `conf(x)`, while incorrect examples have low `conf(x)`. Then confidence risk, such as `1 - conf(x)`, separates correct and incorrect predictions. In this regime, CIC may add little for failure ranking because the model already signals ordinary uncertainty.

### Constructive Example B: Confident-Wrong Shortcut Regime

Suppose training makes `s` highly correlated with `y`, and the model learns `f(x) = s` with high confidence. At shifted test time, the relationship between `s` and `y` flips, but the shortcut values remain familiar. The model can remain high-confidence while becoming wrong.

Changing `s` while preserving `c` changes the prediction, so CIC is high. Confidence fails because the model is not uncertain; counterfactual stability detects that the prediction depends on an unstable shortcut feature.

## Lemma: No Confidence-Only Metric Is Sufficient To Detect Shortcut Reliance

Let the input be `x = (c, s)`, where `c` is causal content and `s` is a shortcut. The true label is `y = g(c)`, so `s` is not causally necessary for `y`. A model may rely on `c`, on `s`, or on both. A confidence-only metric `u(x)` depends only on the model's output probabilities at the observed input.

Construct two models on the same observed example. Model A relies on the causal feature `c`. Model B relies on the shortcut feature `s`. On the observed input, both models produce the same predicted label and the same confidence. Any confidence-only metric must assign them the same reliability score.

Now apply a label-preserving intervention that changes `s` while preserving `c`. Model A remains stable because the causal feature is unchanged. Model B changes prediction or loses decision support because the shortcut changed. Therefore confidence alone cannot distinguish causal reliance from shortcut reliance on this example.

This is a separation argument, not a universal guarantee that CIC detects every shortcut. It shows that counterfactual stability can contain information not present in confidence.

## Conclusion

CIC is not a replacement for confidence. It measures a different property: whether the prediction is stable under label-preserving shortcut changes.
