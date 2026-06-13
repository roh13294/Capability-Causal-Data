import torch
import torch.nn.functional as F


def group_dro_loss(logits: torch.Tensor, y: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    losses = []
    for g in group.unique():
        mask = group == g
        if mask.any():
            losses.append(F.cross_entropy(logits[mask], y[mask]))
    return torch.stack(losses).max()
