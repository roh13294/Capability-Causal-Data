import torch


def feature_distance_score(x: torch.Tensor, train_mean: torch.Tensor) -> torch.Tensor:
    return (x.float().flatten(1) - train_mean.flatten()).pow(2).sum(dim=1).sqrt()
