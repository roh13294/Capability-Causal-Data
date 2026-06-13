from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from causal_reliability.certificates.distances import js_divergence, margin_collapse, softmax


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str = "erm",
    make_counterfactuals: Callable[[torch.Tensor], torch.Tensor] | None = None,
    stability_lambda: float = 0.5,
    eta: float = 0.5,
) -> float:
    model.train()
    losses = []
    for x, y, _shortcut, _causal in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if mode in {"augmentation", "stability", "combined"}:
            if make_counterfactuals is None:
                raise ValueError("make_counterfactuals is required for counterfactual training")
            cf = make_counterfactuals(x).to(device)
            flat = cf.reshape(-1, *cf.shape[2:])
            logits_cf = model(flat).reshape(x.shape[0], cf.shape[1], -1)
            if mode in {"augmentation", "combined"}:
                loss = loss + F.cross_entropy(logits_cf.reshape(-1, logits_cf.shape[-1]), y.repeat_interleave(cf.shape[1]))
            if mode in {"stability", "combined"}:
                stability = margin_collapse(logits, logits_cf).mean() + eta * js_divergence(softmax(logits), softmax(logits_cf)).mean()
                loss = loss + stability_lambda * stability
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(sum(losses) / max(len(losses), 1))


def train_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
    mode: str = "erm",
    make_counterfactuals: Callable[[torch.Tensor], torch.Tensor] | None = None,
    stability_lambda: float = 0.5,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return [
        train_epoch(model, loader, optimizer, device, mode, make_counterfactuals, stability_lambda)
        for _ in range(epochs)
    ]
