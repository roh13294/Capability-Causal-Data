from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from causal_reliability.utils.io import ensure_dir


CLASS_NAMES = ["circle", "square", "triangle", "star", "cross"]
SHORTCUT_COLORS = {
    0: (0.88, 0.18, 0.16),
    1: (0.12, 0.42, 0.86),
    2: (0.16, 0.68, 0.28),
    3: (0.88, 0.74, 0.18),
    4: (0.62, 0.22, 0.78),
}


def _prepare_matplotlib() -> None:
    import os
    import tempfile

    cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    import matplotlib

    matplotlib.use("Agg")


@dataclass(frozen=True)
class RealModelShortcutBundle:
    id_examples: list[dict[str, Any]]
    shifted_examples: list[dict[str, Any]]
    class_names: list[str]
    shortcut_type: str


def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:size, 0:size]
    return xs.astype(float), ys.astype(float)


def _shape_mask(label: int, size: int, jitter: tuple[float, float], scale: float) -> np.ndarray:
    xs, ys = _grid(size)
    cx = size * (0.5 + jitter[0])
    cy = size * (0.5 + jitter[1])
    r = size * scale
    x = xs - cx
    y = ys - cy
    if label == 0:
        return x * x + y * y <= r * r
    if label == 1:
        return (np.abs(x) <= r) & (np.abs(y) <= r)
    if label == 2:
        p1 = np.array([cx, cy - 1.25 * r])
        p2 = np.array([cx - 1.25 * r, cy + 1.1 * r])
        p3 = np.array([cx + 1.25 * r, cy + 1.1 * r])
        den = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
        a = ((p2[1] - p3[1]) * (xs - p3[0]) + (p3[0] - p2[0]) * (ys - p3[1])) / den
        b = ((p3[1] - p1[1]) * (xs - p3[0]) + (p1[0] - p3[0]) * (ys - p3[1])) / den
        c = 1 - a - b
        return (a >= 0) & (b >= 0) & (c >= 0)
    if label == 3:
        angle = np.arctan2(y, x)
        radius = np.sqrt(x * x + y * y)
        boundary = r * (0.78 + 0.28 * np.cos(5 * angle))
        return radius <= boundary
    return ((np.abs(x) <= r * 0.36) & (np.abs(y) <= r)) | ((np.abs(y) <= r * 0.36) & (np.abs(x) <= r))


def _shortcut_label(label: int, n_classes: int, shifted: bool) -> int:
    return int((label + (1 if shifted else 0)) % n_classes)


def _draw_text_blocks(image: np.ndarray, label: int, color: tuple[float, float, float]) -> np.ndarray:
    # Tiny seven-segment-ish block cue. It is intentionally a label shortcut, not readable typography.
    patterns = [
        [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
    ][label % 5]
    out = image.copy()
    s = max(2, image.shape[0] // 14)
    y0 = image.shape[0] - 4 * s
    x0 = 2 * s
    for px, py in patterns:
        out[y0 + py * s : y0 + (py + 1) * s, x0 + px * s : x0 + (px + 1) * s, :] = color
    return out


def render_shortcut_image(
    label: int,
    shortcut_label: int | None = None,
    shortcut_type: str = "background",
    size: int = 64,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    shortcut = label if shortcut_label is None else int(shortcut_label)
    bg = np.array(SHORTCUT_COLORS[shortcut % len(SHORTCUT_COLORS)], dtype=np.float32)
    image = np.ones((size, size, 3), dtype=np.float32) * 0.94
    if shortcut_type in {"background", "mixed"}:
        image[:] = bg
    else:
        image[:] = np.array([0.92, 0.92, 0.88], dtype=np.float32)

    jitter = (float(rng.uniform(-0.05, 0.05)), float(rng.uniform(-0.04, 0.04)))
    mask = _shape_mask(label, size, jitter, float(rng.uniform(0.22, 0.27)))
    object_color = np.array([0.06, 0.06, 0.07], dtype=np.float32)
    image[mask] = object_color

    border_mask = np.zeros((size, size), dtype=bool)
    width = max(3, size // 16)
    border_mask[:width, :] = True
    border_mask[-width:, :] = True
    border_mask[:, :width] = True
    border_mask[:, -width:] = True
    if shortcut_type in {"border", "mixed"}:
        image[border_mask] = bg
    if shortcut_type in {"text", "mixed"}:
        image = _draw_text_blocks(image, shortcut, tuple(bg))

    shortcut_region = np.zeros((size, size), dtype=bool)
    if shortcut_type in {"background", "mixed"}:
        shortcut_region |= ~mask
    if shortcut_type in {"border", "mixed"}:
        shortcut_region |= border_mask
    if shortcut_type in {"text", "mixed"}:
        s = max(2, size // 14)
        shortcut_region[size - 4 * s : size - s, 2 * s : 5 * s] = True

    meta = {
        "object_mask": mask,
        "shortcut_mask": shortcut_region,
        "background_mask": ~mask,
        "border_mask": border_mask,
        "shortcut_label": shortcut,
        "shortcut_color": SHORTCUT_COLORS[shortcut % len(SHORTCUT_COLORS)],
    }
    return np.clip(image, 0.0, 1.0), meta


def make_real_model_shortcut_dataset(
    n_per_class: int = 16,
    size: int = 64,
    shortcut_type: str = "background",
    seed: int = 0,
    classes: list[str] | None = None,
) -> RealModelShortcutBundle:
    class_names = list(classes or CLASS_NAMES)
    n_classes = len(class_names)
    id_examples: list[dict[str, Any]] = []
    shifted_examples: list[dict[str, Any]] = []
    example_id = 0
    for label in range(n_classes):
        for j in range(n_per_class):
            item_seed = seed + label * 1000 + j
            id_shortcut = _shortcut_label(label, n_classes, shifted=False)
            shifted_shortcut = _shortcut_label(label, n_classes, shifted=True)
            image, meta = render_shortcut_image(label, id_shortcut, shortcut_type, size, item_seed)
            shifted_image, shifted_meta = render_shortcut_image(label, shifted_shortcut, shortcut_type, size, item_seed)
            common_id = {
                "example_id": example_id,
                "label": label,
                "class_name": class_names[label],
                "image": image,
                "counterfactual_image": shifted_image,
                "object_mask": meta["object_mask"],
                "shortcut_mask": meta["shortcut_mask"],
                "counterfactual_shortcut_mask": shifted_meta["shortcut_mask"],
            }
            common_shifted = {
                "example_id": example_id,
                "label": label,
                "class_name": class_names[label],
                "image": shifted_image,
                "counterfactual_image": image,
                "object_mask": shifted_meta["object_mask"],
                "shortcut_mask": shifted_meta["shortcut_mask"],
                "counterfactual_shortcut_mask": meta["shortcut_mask"],
            }
            id_examples.append({**common_id, "split": "id", "shortcut_label": id_shortcut})
            shifted_examples.append({**common_shifted, "split": "shifted", "shortcut_label": shifted_shortcut})
            example_id += 1
    return RealModelShortcutBundle(id_examples, shifted_examples, class_names, shortcut_type)


def images_to_tensor(examples: list[dict[str, Any]], key: str = "image") -> torch.Tensor:
    arr = np.stack([ex[key] for ex in examples]).astype(np.float32)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def labels_to_tensor(examples: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([int(ex["label"]) for ex in examples], dtype=torch.long)


def save_example_grid(examples: list[dict[str, Any]], path: str | Path, n: int = 8) -> None:
    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    path = Path(path)
    ensure_dir(path.parent)
    take = examples[: max(1, min(n, len(examples)))]
    fig, axes = plt.subplots(2, len(take), figsize=(1.45 * len(take), 3.0))
    if len(take) == 1:
        axes = np.array(axes).reshape(2, 1)
    for i, ex in enumerate(take):
        axes[0, i].imshow(ex["image"])
        axes[0, i].set_title(ex["class_name"], fontsize=7)
        axes[1, i].imshow(ex["counterfactual_image"])
        axes[1, i].set_title("cf", fontsize=7)
        axes[0, i].set_axis_off()
        axes[1, i].set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
