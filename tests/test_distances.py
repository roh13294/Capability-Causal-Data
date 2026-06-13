import torch

from causal_reliability.certificates.distances import confidence, entropy, js_divergence, label_flip, logits_to_margin, margin_collapse, softmax


def test_distance_functions_valid_values():
    logits = torch.tensor([[3.0, 1.0], [0.2, 0.8]])
    cf = torch.tensor([[[2.0, 1.0], [0.0, 2.0]], [[0.1, 0.9], [1.0, 0.0]]])
    assert torch.allclose(logits_to_margin(logits), torch.tensor([2.0, 0.6]))
    assert margin_collapse(logits, cf).shape == (2, 2)
    assert label_flip(logits, cf).shape == (2, 2)
    probs = softmax(logits)
    assert (confidence(probs) <= 1).all()
    assert (entropy(probs) >= 0).all()
    assert (js_divergence(probs, softmax(cf)) >= 0).all()
