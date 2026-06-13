from __future__ import annotations

import math

import torch

from causal_reliability.data.splits import DatasetBundle, tensor_dataset


SHIFT_MODES = ("ood_new_shortcut", "in_support_flip", "weak_shift", "mixed_shift", "partial_in_support_flip")


def _shortcut_from_label(y: torch.Tensor, corr: float) -> torch.Tensor:
    agree = torch.rand_like(y.float()) < corr
    return torch.where(agree, y, 1 - y)


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
    use_fraction = torch.rand_like(y.float()) < partial_flip_fraction
    use_strength = torch.rand_like(y.float()) < partial_flip_strength
    return eligible & use_fraction & use_strength


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
        return 1 - y
    if shift_mode == "partial_in_support_flip":
        normal = _shortcut_from_label(y, corr)
        flipped = 1 - y
        use_flip = _partial_flip_mask(y, partial_flip_fraction, partial_flip_classes, partial_flip_strength)
        return torch.where(use_flip, flipped, normal)
    if shift_mode == "mixed_shift":
        hard_flip = torch.rand_like(y.float()) < 0.5
        return torch.where(hard_flip, 1 - y, _shortcut_from_label(y, corr))
    return _shortcut_from_label(y, corr)


def _make_vector_split(
    n: int,
    corr: float,
    noise: float,
    shift_mode: str = "weak_shift",
    causal_strength: float = 1.0,
    shortcut_strength: float = 1.0,
    partial_flip_fraction: float = 0.5,
    partial_flip_classes: list[int] | tuple[int, ...] | None = None,
    partial_flip_strength: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    y = torch.randint(0, 2, (n,))
    c = y.clone()
    s = _shortcut_for_mode(y, corr, shift_mode, partial_flip_fraction, partial_flip_classes, partial_flip_strength)
    causal_feature = causal_strength * (2 * c.float() - 1) + noise * torch.randn(n)
    if shift_mode == "ood_new_shortcut":
        shortcut_feature = torch.full((n,), 2.6) + noise * torch.randn(n)
    else:
        shortcut_feature = shortcut_strength * (2 * s.float() - 1) + noise * torch.randn(n)
    nuisance = noise * torch.randn(n, 2)
    x = torch.stack([causal_feature, shortcut_feature], dim=1)
    x = torch.cat([x, nuisance], dim=1)
    return x, y, s, c


def make_vector_task(
    n_train: int = 512,
    n_test: int = 256,
    train_corr: float = 0.95,
    id_corr: float = 0.95,
    shift_corr: float = 0.1,
    noise: float = 0.35,
    shift_mode: str = "weak_shift",
    causal_strength: float = 1.0,
    shortcut_strength: float = 1.0,
    partial_flip_fraction: float = 0.5,
    partial_flip_classes: list[int] | tuple[int, ...] | None = None,
    partial_flip_strength: float = 1.0,
) -> DatasetBundle:
    train = tensor_dataset(*_make_vector_split(n_train, train_corr, noise, causal_strength=causal_strength, shortcut_strength=shortcut_strength))
    id_test = tensor_dataset(*_make_vector_split(n_test, id_corr, noise, causal_strength=causal_strength, shortcut_strength=shortcut_strength))
    shifted = tensor_dataset(
        *_make_vector_split(
            n_test,
            shift_corr,
            noise,
            shift_mode,
            causal_strength,
            shortcut_strength,
            partial_flip_fraction,
            partial_flip_classes,
            partial_flip_strength,
        )
    )
    return DatasetBundle(train, id_test, shifted, input_shape=(4,), task_type="vector")


def _texture_pattern(shortcut: int, size: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    if shortcut == 0:
        return ((xx // 2) % 2).float()
    return (((xx + yy) // 3) % 2).float()


def _render_shape(label: int, shortcut: int, size: int = 16, shortcut_type: str = "color") -> torch.Tensor:
    img = torch.zeros(3, size, size)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    cx = cy = (size - 1) / 2
    if label == 0:
        mask = ((xx - cx).abs() < size * 0.22) & ((yy - cy).abs() < size * 0.22)
    else:
        mask = (xx - cx).pow(2) + (yy - cy).pow(2) < (size * 0.24) ** 2
    shortcut_colors = (
        torch.tensor([0.95, 0.20, 0.20]),
        torch.tensor([0.15, 0.45, 0.95]),
    )
    neutral_object = torch.tensor([0.88, 0.88, 0.78])
    neutral_bg = torch.tensor([0.07, 0.07, 0.08])
    if shortcut_type == "color":
        color = shortcut_colors[shortcut]
        bg = neutral_bg
    elif shortcut_type == "background":
        color = neutral_object
        bg = shortcut_colors[shortcut] * 0.55
    elif shortcut_type == "texture":
        color = neutral_object
        bg = neutral_bg
    else:
        raise ValueError(f"unknown shortcut_type: {shortcut_type}")
    bg = bg.view(3, 1, 1)
    img[:] = bg
    img[:, mask] = color.view(3, 1)
    if shortcut_type == "texture":
        pattern = _texture_pattern(shortcut, size)
        texture = (0.55 + 0.35 * pattern).view(1, size, size)
        img[:, mask] = (neutral_object.view(3, 1) * texture[:, mask]).clamp(0, 1)
    img += 0.03 * torch.randn_like(img)
    return img.clamp(0, 1)


def _make_shape_split(
    n: int,
    corr: float,
    size: int,
    shortcut_type: str,
    shift_mode: str = "weak_shift",
    partial_flip_fraction: float = 0.5,
    partial_flip_classes: list[int] | tuple[int, ...] | None = None,
    partial_flip_strength: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    y = torch.randint(0, 2, (n,))
    c = y.clone()
    s = _shortcut_for_mode(y, corr, shift_mode, partial_flip_fraction, partial_flip_classes, partial_flip_strength)
    x = torch.stack([_render_shape(int(y[i]), int(s[i]), size, shortcut_type) for i in range(n)])
    return x, y, s, c


def make_shape_task(
    n_train: int = 384,
    n_test: int = 192,
    train_corr: float = 0.95,
    id_corr: float = 0.95,
    shift_corr: float = 0.1,
    image_size: int = 16,
    shortcut_type: str = "color",
    train_shortcut_type: str | None = None,
    id_shortcut_type: str | None = None,
    shift_shortcut_type: str | None = None,
    shift_mode: str = "weak_shift",
    partial_flip_fraction: float = 0.5,
    partial_flip_classes: list[int] | tuple[int, ...] | None = None,
    partial_flip_strength: float = 1.0,
) -> DatasetBundle:
    train_shortcut_type = train_shortcut_type or shortcut_type
    id_shortcut_type = id_shortcut_type or shortcut_type
    shift_shortcut_type = shift_shortcut_type or shortcut_type
    return DatasetBundle(
        tensor_dataset(*_make_shape_split(n_train, train_corr, image_size, train_shortcut_type)),
        tensor_dataset(*_make_shape_split(n_test, id_corr, image_size, id_shortcut_type)),
        tensor_dataset(
            *_make_shape_split(
                n_test,
                shift_corr,
                image_size,
                shift_shortcut_type,
                shift_mode,
                partial_flip_fraction,
                partial_flip_classes,
                partial_flip_strength,
            )
        ),
        input_shape=(3, image_size, image_size),
        task_type="vision",
    )
