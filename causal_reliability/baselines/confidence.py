import torch


def confidence_risk(probs: torch.Tensor) -> torch.Tensor:
    return 1.0 - probs.max(dim=-1).values
