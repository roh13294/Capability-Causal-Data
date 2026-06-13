from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch


def mask_region(image: torch.Tensor, mask: np.ndarray, fill: float = 0.5) -> torch.Tensor:
    out = image.clone()
    m = torch.as_tensor(mask, dtype=torch.bool, device=out.device)
    out[:, m] = fill
    return out


def _region_mask(ex: dict[str, Any], *names: str) -> np.ndarray:
    for name in names:
        if name in ex:
            return np.asarray(ex[name], dtype=bool)
    return np.zeros(np.asarray(ex["image"]).shape[:2], dtype=bool)


def occlusion_metrics(model: Any, examples: list[dict[str, Any]], predictions: torch.Tensor, confidence: torch.Tensor) -> pd.DataFrame:
    rows = []
    base_images = torch.from_numpy(np.stack([ex["image"] for ex in examples]).astype(np.float32)).permute(0, 3, 1, 2)
    for i, ex in enumerate(examples):
        pred = int(predictions[i])
        base_conf = float(confidence[i])
        object_img = mask_region(base_images[i], _region_mask(ex, "object_mask", "shape_mask")).unsqueeze(0)
        shortcut_img = mask_region(base_images[i], _region_mask(ex, "shortcut_mask", "text_mask")).unsqueeze(0)
        background_img = mask_region(base_images[i], _region_mask(ex, "background_mask")).unsqueeze(0)
        object_prob = model.predict(object_img)["probabilities"][0, pred].item()
        shortcut_prob = model.predict(shortcut_img)["probabilities"][0, pred].item()
        background_prob = model.predict(background_img)["probabilities"][0, pred].item()
        object_effect = max(0.0, base_conf - object_prob)
        shortcut_effect = max(0.0, base_conf - shortcut_prob)
        background_effect = max(0.0, base_conf - background_prob)
        ratio = shortcut_effect / (shortcut_effect + object_effect + 1e-8)
        rows.append(
            {
                "example_id": ex["example_id"],
                "label": ex["label"],
                "pred": pred,
                "confidence": base_conf,
                "occlusion_object_effect": object_effect,
                "occlusion_shortcut_effect": shortcut_effect,
                "object_occlusion_drop": object_effect,
                "text_occlusion_drop": shortcut_effect,
                "background_occlusion_drop": background_effect,
                "shortcut_attention_ratio": ratio,
            }
        )
    return pd.DataFrame(rows)
