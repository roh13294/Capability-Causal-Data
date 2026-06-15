"""Second shortcut family: non-text visual decoy patches (pilot only).

This is a *different* shortcut family from the typographic text-overlay headline
experiment. The image contains a large central causal object (a shape) whose
identity is the true label, plus a separate, smaller non-text decoy patch in a
corner that visually depicts a *competing* class. There are no written words
anywhere. The decoy patch location is known at generation time (the oracle decoy
region); it is used only for the oracle upper-bound baseline and for downstream
localization metrics, never by the non-oracle region scorer.

Scope: this is a controlled pilot stimulus. It is not a claim of open-world
discovery, general robustness, or universal shortcut repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from causal_reliability.utils.io import ensure_dir


# Reuse the same shape vocabulary as the text-overlay family so the zero-shot
# CLIP prompts behave comparably; only the shortcut *modality* differs.
VISUAL_DECOY_CLASSES = ["circle", "square", "triangle", "star"]

_BACKGROUND = (238, 240, 235)
_OBJECT_FILL = (32, 34, 36)
_DECOY_FILL = (40, 42, 44)


@dataclass(frozen=True)
class VisualDecoyBundle:
    examples: list[dict[str, Any]]
    class_names: list[str]


def _star_points(box: list[float]) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    points = []
    for k in range(10):
        rad = 1.0 if k % 2 == 0 else 0.45
        ang = -np.pi / 2 + k * np.pi / 5
        points.append((cx + rad * rx * np.cos(ang), cy + rad * ry * np.sin(ang)))
    return points


def _triangle_points(box: list[float]) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0
    return [(cx, y0), (x0, y1), (x1, y1)]


def _draw_shape(draw: ImageDraw.ImageDraw, label: int, box: list[float], fill: tuple[int, int, int]) -> None:
    if label == 0:
        draw.ellipse(box, fill=fill)
    elif label == 1:
        draw.rectangle(box, fill=fill)
    elif label == 2:
        draw.polygon(_triangle_points(box), fill=fill)
    else:
        draw.polygon(_star_points(box), fill=fill)


def _corner_box(corner: int, size: int, frac: float, margin_frac: float = 0.05) -> list[int]:
    side = int(round(size * frac))
    margin = int(round(size * margin_frac))
    if corner == 0:  # top-left
        x0, y0 = margin, margin
    elif corner == 1:  # top-right
        x0, y0 = size - margin - side, margin
    elif corner == 2:  # bottom-left
        x0, y0 = margin, size - margin - side
    else:  # bottom-right
        x0, y0 = size - margin - side, size - margin - side
    return [int(x0), int(y0), int(x0 + side), int(y0 + side)]


def render_decoy_image(
    label: int,
    decoy_label: int | None,
    corner: int,
    size: int = 224,
    object_frac: tuple[float, float] = (0.42, 0.42),
    decoy_frac: float = 0.22,
    draw_decoy: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render a centered causal object plus an optional corner decoy patch.

    No text is ever drawn. The decoy is a filled shape of a competing class.
    """

    img = Image.new("RGB", (size, size), _BACKGROUND)
    draw = ImageDraw.Draw(img)

    ow, oh = object_frac
    obj_box = [
        size * (0.5 - ow / 2),
        size * (0.48 - oh / 2),
        size * (0.5 + ow / 2),
        size * (0.48 + oh / 2),
    ]
    object_bbox = [int(round(v)) for v in obj_box]
    _draw_shape(draw, label, obj_box, _OBJECT_FILL)

    decoy_bbox = _corner_box(corner, size, decoy_frac)
    if draw_decoy and decoy_label is not None:
        _draw_shape(draw, decoy_label, [float(v) for v in decoy_bbox], _DECOY_FILL)

    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr, {
        "class_name": VISUAL_DECOY_CLASSES[label],
        "decoy_label": decoy_label,
        "decoy_class_name": "" if decoy_label is None else VISUAL_DECOY_CLASSES[decoy_label],
        "object_bbox": object_bbox,
        "decoy_bbox": decoy_bbox,
        "corner": corner,
    }


def _decoy_for(label: int, regime: str, index: int, n_classes: int) -> int | None:
    if regime == "no_decoy":
        return None
    if regime == "aligned_decoy":
        return label
    if regime == "misleading_decoy":
        return (label + 1 + index % (n_classes - 1)) % n_classes
    raise ValueError(f"unknown decoy regime: {regime}")


def make_visual_decoy_dataset(
    n_per_condition: int = 64,
    size: int = 224,
    regimes: list[str] | None = None,
    split: str = "test",
    start_id: int = 0,
    decoy_frac: float = 0.22,
    object_frac: tuple[float, float] = (0.42, 0.42),
) -> VisualDecoyBundle:
    """Build the visual-decoy dataset.

    ``n_per_condition`` is the number of images per regime, distributed evenly
    across the class vocabulary. Corners are cycled so the oracle decoy region
    is not confounded with a fixed location.
    """

    regimes = regimes or ["no_decoy", "misleading_decoy", "aligned_decoy"]
    n_classes = len(VISUAL_DECOY_CLASSES)
    per_class = max(1, n_per_condition // n_classes)
    examples: list[dict[str, Any]] = []
    example_id = start_id
    for regime in regimes:
        for label, class_name in enumerate(VISUAL_DECOY_CLASSES):
            for j in range(per_class):
                decoy_label = _decoy_for(label, regime, j, n_classes)
                corner = (label + j) % 4
                image, meta = render_decoy_image(
                    label,
                    decoy_label,
                    corner,
                    size=size,
                    object_frac=object_frac,
                    decoy_frac=decoy_frac,
                    draw_decoy=regime != "no_decoy",
                )
                # Clean reference (object only) for inspection / sanity.
                clean_image, _ = render_decoy_image(
                    label, None, corner, size=size, object_frac=object_frac, decoy_frac=decoy_frac, draw_decoy=False
                )
                if regime == "no_decoy":
                    decoy_relation = "none"
                elif decoy_label == label:
                    decoy_relation = "aligned"
                else:
                    decoy_relation = "misleading"
                examples.append(
                    {
                        "example_id": example_id,
                        "split": split,
                        "regime": regime,
                        "label": label,
                        "true_label": class_name,
                        "class_name": class_name,
                        "decoy_label": -1 if decoy_label is None else int(decoy_label),
                        "decoy_class_name": meta["decoy_class_name"],
                        "decoy_relation": decoy_relation,
                        "decoy_bbox": meta["decoy_bbox"],
                        "object_bbox": meta["object_bbox"],
                        "corner": meta["corner"],
                        "image": image,
                        "clean_image": clean_image,
                    }
                )
                example_id += 1
    return VisualDecoyBundle(examples, VISUAL_DECOY_CLASSES.copy())


def save_example_images(examples: list[dict[str, Any]], out_dir: str | Path, keys: list[str] | None = None) -> None:
    keys = keys or ["image"]
    root = ensure_dir(out_dir)
    for ex in examples:
        ex_dir = ensure_dir(root / str(ex["split"]) / str(ex["regime"]))
        for key in keys:
            arr = ex.get(key)
            if arr is None:
                continue
            path = ex_dir / f"{ex['example_id']}_{key}.png"
            Image.fromarray((np.asarray(arr).clip(0, 1) * 255).astype(np.uint8)).save(path)
            ex[f"{key}_path"] = str(path)


def examples_to_tensor(examples: list[dict[str, Any]], key: str = "image") -> torch.Tensor:
    arr = np.stack([ex[key] for ex in examples]).astype(np.float32)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
