from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _as_numpy(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _predict_proba(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.softmax(logits, dim=-1)


def random_augmentation_sensitivity(
    model: torch.nn.Module,
    x: torch.Tensor,
    perturb: Callable[[torch.Tensor], torch.Tensor] | None = None,
    n_augmentations: int = 8,
) -> np.ndarray:
    """Prediction instability under generic, non-targeted perturbations."""
    if perturb is None:
        perturb = generic_tensor_perturbation
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        base = _predict_proba(model, x)
        base_pred = base.argmax(dim=-1)
        scores = torch.zeros(len(x), device=device)
        for _ in range(int(n_augmentations)):
            aug = perturb(x).to(device)
            probs = _predict_proba(model, aug)
            flip = (probs.argmax(dim=-1) != base_pred).float()
            drift = 0.5 * torch.abs(probs - base).sum(dim=-1)
            scores += 0.5 * flip + 0.5 * drift
        scores /= max(int(n_augmentations), 1)
    return _as_numpy(scores)


def generic_tensor_perturbation(x: torch.Tensor) -> torch.Tensor:
    """Small noise, brightness/contrast, and one-pixel image translation."""
    y = x.clone()
    if y.ndim >= 4:
        y = torch.roll(y, shifts=1, dims=-1)
        dims = tuple(range(1, y.ndim))
        mean = y.mean(dim=dims, keepdim=True)
        y = (y - mean) * 1.05 + mean + 0.03
    else:
        y = y + 0.03
    y = y + 0.02 * torch.randn_like(y)
    return y.clamp(0.0, 1.0) if x.min() >= 0 and x.max() <= 1 else y


def neutral_token_replacement(
    tokens: torch.Tensor,
    neutral_token_id: int,
    positions: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    y = tokens.clone()
    if y.ndim == 1:
        y = y.unsqueeze(0)
    if positions is None:
        positions = [y.shape[1] - 1]
    for pos in positions:
        if -y.shape[1] <= pos < y.shape[1]:
            y[:, pos] = int(neutral_token_id)
    return y


def occlusion_confidence_drop(
    model: torch.nn.Module,
    x: torch.Tensor,
    shortcut_mask: torch.Tensor | None = None,
    object_mask: torch.Tensor | None = None,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Confidence drops after masking known shortcut/object regions."""
    if shortcut_mask is None and object_mask is None:
        raise ValueError("At least one mask must be provided.")
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        base = _predict_proba(model, x)
        pred = base.argmax(dim=-1)
        base_conf = base.gather(1, pred[:, None]).squeeze(1)
        out: dict[str, np.ndarray] = {}
        for name, mask in [("shortcut", shortcut_mask), ("object", object_mask)]:
            if mask is None:
                continue
            mask = mask.to(device).float()
            masked = x * (1.0 - mask) + fill_value * mask
            probs = _predict_proba(model, masked)
            conf = probs.gather(1, pred[:, None]).squeeze(1)
            out[f"{name}_occlusion_drop"] = _as_numpy(base_conf - conf)
        if {"shortcut_occlusion_drop", "object_occlusion_drop"}.issubset(out):
            denom = np.abs(out["object_occlusion_drop"]) + 1e-8
            out["shortcut_object_occlusion_ratio"] = out["shortcut_occlusion_drop"] / denom
    return pd.DataFrame(out)


def occlusion_scores_from_certificates(df: pd.DataFrame) -> pd.Series:
    """Attribution-style shortcut proxy when only certificate artifacts exist."""
    if "flip_mean" in df.columns:
        return pd.to_numeric(df["flip_mean"], errors="coerce")
    if "shortcut_occlusion_drop" in df.columns:
        return pd.to_numeric(df["shortcut_occlusion_drop"], errors="coerce")
    if "cis" in df.columns:
        return pd.to_numeric(df["cis"], errors="coerce")
    return pd.Series(np.nan, index=df.index)

