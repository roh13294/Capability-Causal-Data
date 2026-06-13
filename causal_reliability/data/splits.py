from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset


@dataclass
class DatasetBundle:
    train: TensorDataset
    id_test: TensorDataset
    shifted_test: TensorDataset
    input_shape: tuple[int, ...]
    num_classes: int = 2
    task_type: str = "vector"
    vocab: dict[str, int] | None = None


def tensor_dataset(x: torch.Tensor, y: torch.Tensor, shortcut: torch.Tensor, causal: torch.Tensor) -> TensorDataset:
    return TensorDataset(x.float() if x.dtype.is_floating_point else x.long(), y.long(), shortcut.long(), causal.long())
