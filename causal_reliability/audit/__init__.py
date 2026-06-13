"""Scientific audit utilities for leakage and counterfactual sanity checks."""

from causal_reliability.audit.leakage import (
    assert_causal_feature_preserved,
    assert_counterfactual_label_preservation,
    assert_metric_polarity,
    assert_no_shift_labels_in_certificate_inputs,
    assert_shortcut_changed,
)

__all__ = [
    "assert_causal_feature_preserved",
    "assert_counterfactual_label_preservation",
    "assert_metric_polarity",
    "assert_no_shift_labels_in_certificate_inputs",
    "assert_shortcut_changed",
]
