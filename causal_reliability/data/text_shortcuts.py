from __future__ import annotations

import torch

from causal_reliability.data.splits import DatasetBundle, tensor_dataset

VOCAB = {
    "<pad>": 0,
    "low": 1,
    "high": 2,
    "less": 3,
    "more": 4,
    "always": 5,
    "never": 6,
    "clearly": 7,
    "rarely": 8,
}
SHORTCUTS = [5, 6]
FILLERS = [7, 8]
SHIFT_MODES = ("ood_new_shortcut", "in_support_flip", "weak_shift", "mixed_shift", "partial_in_support_flip")


def _shortcut(y: torch.Tensor, corr: float) -> torch.Tensor:
    agree = torch.rand_like(y.float()) < corr
    positive_token = torch.full_like(y, SHORTCUTS[0])
    negative_token = torch.full_like(y, SHORTCUTS[1])
    return torch.where(torch.where(agree, y == 1, y == 0), positive_token, negative_token)


def _partial_flip_mask(
    y: torch.Tensor,
    partial_flip_fraction: float = 0.5,
    partial_flip_classes: list[int] | tuple[int, ...] | None = None,
    partial_flip_strength: float = 1.0,
) -> torch.Tensor:
    if not 0 <= partial_flip_fraction <= 1:
        raise ValueError("partial_flip_fraction must be in [0, 1]")
    if not 0 <= partial_flip_strength <= 1:
        raise ValueError("partial_flip_strength must be in [0, 1]")
    eligible = torch.ones_like(y, dtype=torch.bool)
    if partial_flip_classes is not None:
        allowed = torch.tensor(list(partial_flip_classes), device=y.device, dtype=y.dtype)
        eligible = (y.unsqueeze(1) == allowed.view(1, -1)).any(dim=1)
    return eligible & (torch.rand_like(y.float()) < partial_flip_fraction) & (torch.rand_like(y.float()) < partial_flip_strength)


def _shortcut_for_mode(
    y: torch.Tensor,
    corr: float,
    shift_mode: str = "weak_shift",
    partial_flip_fraction: float = 0.5,
    partial_flip_classes: list[int] | tuple[int, ...] | None = None,
    partial_flip_strength: float = 1.0,
) -> torch.Tensor:
    if shift_mode not in SHIFT_MODES:
        raise ValueError(f"unknown shift_mode: {shift_mode}")
    if shift_mode == "in_support_flip":
        return torch.where(y == 1, torch.tensor(SHORTCUTS[1]), torch.tensor(SHORTCUTS[0]))
    if shift_mode == "partial_in_support_flip":
        normal = _shortcut(y, corr)
        flipped = torch.where(y == 1, torch.tensor(SHORTCUTS[1]), torch.tensor(SHORTCUTS[0]))
        return torch.where(_partial_flip_mask(y, partial_flip_fraction, partial_flip_classes, partial_flip_strength), flipped, normal)
    if shift_mode == "mixed_shift":
        flip = torch.where(y == 1, torch.tensor(SHORTCUTS[1]), torch.tensor(SHORTCUTS[0]))
        use_flip = torch.rand_like(y.float()) < 0.5
        return torch.where(use_flip, flip, _shortcut(y, corr))
    return _shortcut(y, corr)


def _make_split(
    n: int,
    corr: float,
    seq_len: int = 6,
    shift_mode: str = "weak_shift",
    partial_flip_fraction: float = 0.5,
    partial_flip_classes: list[int] | tuple[int, ...] | None = None,
    partial_flip_strength: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.randint(0, 10, (n,))
    b = torch.randint(0, 10, (n,))
    y = (a > b).long()
    c = y.clone()
    shortcut = _shortcut_for_mode(y, corr, shift_mode, partial_flip_fraction, partial_flip_classes, partial_flip_strength)
    x = torch.zeros(n, seq_len, dtype=torch.long)
    x[:, 0] = torch.where(a > 4, torch.tensor(VOCAB["high"]), torch.tensor(VOCAB["low"]))
    x[:, 1] = torch.where(b > 4, torch.tensor(VOCAB["more"]), torch.tensor(VOCAB["less"]))
    x[:, 2] = shortcut
    x[:, 3] = torch.tensor(VOCAB["clearly"])
    x[:, 4] = (a % 2) + 1
    x[:, 5] = (b % 2) + 3
    return x, y, (shortcut == SHORTCUTS[0]).long(), c


def make_text_task(
    n_train: int = 512,
    n_test: int = 256,
    train_corr: float = 0.95,
    id_corr: float = 0.95,
    shift_corr: float = 0.1,
    shift_mode: str = "weak_shift",
    partial_flip_fraction: float = 0.5,
    partial_flip_classes: list[int] | tuple[int, ...] | None = None,
    partial_flip_strength: float = 1.0,
) -> DatasetBundle:
    return DatasetBundle(
        tensor_dataset(*_make_split(n_train, train_corr)),
        tensor_dataset(*_make_split(n_test, id_corr)),
        tensor_dataset(*_make_split(n_test, shift_corr, shift_mode=shift_mode, partial_flip_fraction=partial_flip_fraction, partial_flip_classes=partial_flip_classes, partial_flip_strength=partial_flip_strength)),
        input_shape=(6,),
        task_type="text",
        vocab=VOCAB,
    )
