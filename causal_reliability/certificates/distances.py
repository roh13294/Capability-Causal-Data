import torch
import torch.nn.functional as F


def logits_to_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = logits.topk(k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def margin_collapse(logits_original: torch.Tensor, logits_cf: torch.Tensor) -> torch.Tensor:
    original_margin = logits_to_margin(logits_original).unsqueeze(-1)
    cf_margin = logits_to_margin(logits_cf)
    return torch.clamp(original_margin - cf_margin, min=0.0)


def label_flip(logits_original: torch.Tensor, logits_cf: torch.Tensor) -> torch.Tensor:
    original = logits_original.argmax(dim=-1).unsqueeze(-1)
    cf = logits_cf.argmax(dim=-1)
    return (original != cf).float()


def js_divergence(probs_original: torch.Tensor, probs_cf: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = probs_original.unsqueeze(-2).clamp_min(eps)
    q = probs_cf.clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p / m).log()).sum(dim=-1) + 0.5 * (q * (q / m).log()).sum(dim=-1)


def entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def confidence(probs: torch.Tensor) -> torch.Tensor:
    return probs.max(dim=-1).values


def softmax(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)
