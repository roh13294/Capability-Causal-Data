from causal_reliability.certificates.distances import logits_to_margin


def negative_margin(logits):
    return -logits_to_margin(logits)
