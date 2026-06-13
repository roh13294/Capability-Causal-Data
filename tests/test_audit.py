import numpy as np
import pytest
import torch

from causal_reliability.audit.leakage import (
    assert_causal_feature_preserved,
    assert_counterfactual_label_preservation,
    assert_metric_polarity,
    assert_no_shift_labels_in_certificate_inputs,
    assert_shortcut_changed,
)
from causal_reliability.data.splits import tensor_dataset


def _dataset():
    x = torch.tensor([[0.0, -1.0, 0.2], [1.0, 1.0, 0.3]])
    y = torch.tensor([0, 1])
    shortcut = torch.tensor([0, 1])
    causal = torch.tensor([0, 1])
    return tensor_dataset(x, y, shortcut, causal)


def test_counterfactual_label_preservation():
    cfs = {"x": torch.zeros(2, 2, 3), "y": torch.tensor([[0, 0], [1, 1]])}
    assert_counterfactual_label_preservation(_dataset(), cfs)
    cfs["y"] = torch.tensor([[0, 1], [1, 1]])
    with pytest.raises(AssertionError):
        assert_counterfactual_label_preservation(_dataset(), cfs)


def test_shortcut_changed():
    cfs = {"x": torch.zeros(2, 2, 3), "shortcut": torch.tensor([[1, 1], [0, 0]])}
    assert_shortcut_changed(_dataset(), cfs)
    cfs["shortcut"] = torch.tensor([[0, 0], [1, 1]])
    with pytest.raises(AssertionError):
        assert_shortcut_changed(_dataset(), cfs)


def test_causal_feature_preserved():
    cfs = {"x": torch.zeros(2, 2, 3), "causal": torch.tensor([[0, 0], [1, 1]])}
    assert_causal_feature_preserved(_dataset(), cfs)
    cfs["causal"] = torch.tensor([[0, 1], [1, 1]])
    with pytest.raises(AssertionError):
        assert_causal_feature_preserved(_dataset(), cfs)


def test_no_shift_labels_used_in_certificate_scoring_if_detectable():
    assert_no_shift_labels_in_certificate_inputs(["logits_original", "logits_counterfactuals", "intervention_outputs"])
    with pytest.raises(AssertionError):
        assert_no_shift_labels_in_certificate_inputs({"logits_original": np.zeros(2), "failure": np.ones(2)})


def test_metric_polarity_check():
    failures = np.array([0, 0, 1, 1])
    assert_metric_polarity({"ShiftRisk": np.array([0.1, 0.2, 0.8, 0.9]), "CR": np.array([0.9, 0.8, 0.2, 0.1])}, failures)
    with pytest.raises(AssertionError):
        assert_metric_polarity({"ShiftRisk": np.array([0.9, 0.8, 0.2, 0.1])}, failures)
