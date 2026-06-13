from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from causal_reliability.data.splits import DatasetBundle, tensor_dataset


PALETTE = torch.tensor(
    [
        [0.92, 0.18, 0.16],
        [0.12, 0.42, 0.92],
        [0.15, 0.72, 0.28],
        [0.92, 0.76, 0.16],
    ],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class ColoredDigitsInfo:
    source: str
    shortcut_cardinality: int = 4
    label_space: str = "digit class"


def _try_sklearn_digits() -> tuple[np.ndarray, np.ndarray, str] | None:
    try:
        from sklearn.datasets import load_digits  # type: ignore
    except Exception:
        return None
    digits = load_digits()
    x = digits.images.astype("float32") / 16.0
    y = digits.target.astype("int64")
    return x, y, "sklearn digits"


def _segment_templates(size: int = 8) -> list[np.ndarray]:
    templates = []
    segments = {
        "top": (slice(0, 1), slice(2, 6)),
        "upper_left": (slice(1, 4), slice(1, 2)),
        "upper_right": (slice(1, 4), slice(6, 7)),
        "middle": (slice(3, 4), slice(2, 6)),
        "lower_left": (slice(4, 7), slice(1, 2)),
        "lower_right": (slice(4, 7), slice(6, 7)),
        "bottom": (slice(7, 8), slice(2, 6)),
    }
    active = [
        ("top", "upper_left", "upper_right", "lower_left", "lower_right", "bottom"),
        ("upper_right", "lower_right"),
        ("top", "upper_right", "middle", "lower_left", "bottom"),
        ("top", "upper_right", "middle", "lower_right", "bottom"),
        ("upper_left", "upper_right", "middle", "lower_right"),
        ("top", "upper_left", "middle", "lower_right", "bottom"),
        ("top", "upper_left", "middle", "lower_left", "lower_right", "bottom"),
        ("top", "upper_right", "lower_right"),
        ("top", "upper_left", "upper_right", "middle", "lower_left", "lower_right", "bottom"),
        ("top", "upper_left", "upper_right", "middle", "lower_right", "bottom"),
    ]
    for digit_segments in active:
        img = np.zeros((size, size), dtype="float32")
        for segment in digit_segments:
            img[segments[segment]] = 1.0
        templates.append(img)
    return templates


def _generated_digits(n: int, seed: int, size: int = 8) -> tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    templates = _segment_templates(size)
    labels = np.arange(n, dtype="int64") % 10
    rng.shuffle(labels)
    images = np.empty((n, size, size), dtype="float32")
    for i, label in enumerate(labels):
        img = templates[int(label)].copy()
        img = np.roll(img, shift=int(rng.integers(-1, 2)), axis=0)
        img = np.roll(img, shift=int(rng.integers(-1, 2)), axis=1)
        img += rng.normal(0.0, 0.10, img.shape).astype("float32")
        images[i] = np.clip(img, 0.0, 1.0)
    return images, labels, "generated seven-segment digit-like fallback"


def _base_pool(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    loaded = _try_sklearn_digits()
    if loaded is None:
        return _generated_digits(n, seed)
    x, y, source = loaded
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(len(y)), size=n, replace=n > len(y))
    return x[idx], y[idx], source


def _shortcut_for_labels(labels: np.ndarray, corr: float, rng: np.random.Generator, shift_mode: str) -> np.ndarray:
    base = labels % len(PALETTE)
    if shift_mode == "in_support_flip":
        return (base + 1) % len(PALETTE)
    if shift_mode != "correlated":
        raise ValueError(f"unknown colored-digits shift_mode: {shift_mode}")
    random_shortcut = rng.integers(0, len(PALETTE), size=len(labels))
    agree = rng.random(len(labels)) < corr
    return np.where(agree, base, random_shortcut)


def colorize_digits(gray: np.ndarray | torch.Tensor, shortcut: np.ndarray | torch.Tensor, background: float = 0.02) -> torch.Tensor:
    gray_t = torch.as_tensor(gray, dtype=torch.float32)
    if gray_t.ndim == 2:
        gray_t = gray_t.unsqueeze(0)
    shortcut_t = torch.as_tensor(shortcut, dtype=torch.long)
    colors = PALETTE[shortcut_t].view(-1, 3, 1, 1)
    mask = gray_t.unsqueeze(1).clamp(0, 1)
    bg = torch.full_like(colors.expand(-1, -1, gray_t.shape[-2], gray_t.shape[-1]), background)
    return (bg * (1 - mask) + colors * mask).clamp(0, 1)


def _split(
    n: int,
    corr: float,
    seed: int,
    shift_mode: str = "correlated",
    noise: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gray, labels, _source = _base_pool(n, seed)
    rng = np.random.default_rng(seed + 1009)
    shortcut = _shortcut_for_labels(labels, corr, rng, shift_mode)
    x = colorize_digits(gray, shortcut)
    if noise > 0:
        gen = torch.Generator().manual_seed(seed + 7919)
        x = (x + noise * torch.randn(x.shape, generator=gen)).clamp(0, 1)
    causal = torch.as_tensor(labels, dtype=torch.long)
    return x, causal, torch.as_tensor(shortcut, dtype=torch.long), causal


def make_colored_digits_task(
    n_train: int = 512,
    n_test: int = 256,
    train_corr: float = 0.95,
    id_corr: float = 0.95,
    shift_corr: float = 0.95,
    shifted_mode: str = "in_support_flip",
    seed: int = 0,
    noise: float = 0.02,
) -> DatasetBundle:
    train = tensor_dataset(*_split(n_train, train_corr, seed + 1, "correlated", noise))
    id_test = tensor_dataset(*_split(n_test, id_corr, seed + 2, "correlated", noise))
    shifted = tensor_dataset(*_split(n_test, shift_corr, seed + 3, shifted_mode, noise))
    return DatasetBundle(train, id_test, shifted, input_shape=(3, 8, 8), num_classes=10, task_type="vision")


def dataset_info(n_probe: int = 16, seed: int = 0) -> ColoredDigitsInfo:
    loaded = _try_sklearn_digits()
    if loaded is not None:
        return ColoredDigitsInfo(source=loaded[2])
    _generated_digits(max(n_probe, 1), seed)
    return ColoredDigitsInfo(source="generated seven-segment digit-like fallback")


def expected_shift_failure_rate(labels: torch.Tensor, shortcuts: torch.Tensor) -> float:
    expected = labels % len(PALETTE)
    return float((expected != shortcuts.cpu()).float().mean())


__all__ = ["ColoredDigitsInfo", "PALETTE", "colorize_digits", "dataset_info", "expected_shift_failure_rate", "make_colored_digits_task"]
