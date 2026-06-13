# Qualitative Examples

## Reliable prediction

- Original input: example_id=37
- Counterfactual input: shortcut/color intervention
- Prediction / true label: 0 / 0
- Confidence: 0.995
- CIC: 0.513
- Shifted prediction failed: False
- Explanation: High confidence and high stability: the prediction remains stable under shortcut changes.

## Uncertain but stable

- Original input: example_id=77
- Counterfactual input: shortcut/color intervention
- Prediction / true label: 1 / 1
- Confidence: 0.799
- CIC: 0.567
- Shifted prediction failed: False
- Explanation: Low confidence but high stability: uncertainty is visible, but the shortcut intervention does not destabilize the prediction.

## Generally fragile

- Original input: example_id=78
- Counterfactual input: shortcut/color intervention
- Prediction / true label: 0 / 1
- Confidence: 0.798
- CIC: 1.489
- Shifted prediction failed: True
- Explanation: Low confidence and low stability: both ordinary uncertainty and counterfactual fragility are present.

## Dangerous shortcut reliance

- Original input: example_id=49
- Counterfactual input: shortcut/color intervention
- Prediction / true label: 1 / 0
- Confidence: 0.995
- CIC: 1.180
- Shifted prediction failed: True
- Explanation: High confidence and low stability: a confident prediction depends on shortcut features.
