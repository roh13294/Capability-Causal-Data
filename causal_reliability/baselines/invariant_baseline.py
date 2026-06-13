import torch


def environment_variance_penalty(logits_by_env: list[torch.Tensor]) -> torch.Tensor:
    means = torch.stack([logits.mean(dim=0) for logits in logits_by_env])
    return means.var(dim=0).mean()
