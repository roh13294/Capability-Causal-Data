"""Second shortcut family (final attempt): high-salience non-text semantic decoys.

This is the same shortcut *family* idea as ``clip_visual_decoy_shortcuts`` (a
central causal object plus a spatially separated competing-class corner patch,
with **no written words anywhere**), but the prior simple near-black filled
shapes were not CLIP-salient enough: the corner decoy only flipped ~42% of
misleading predictions, failing the failure-richness gate.

Here both the central causal object and the corner decoy are rendered as
strongly *colored, recognizable pictorial icons* of natural semantic classes
(sun, heart, leaf, moon). Color + silhouette make each class highly salient to
zero-shot CLIP, so a bright competing-class corner icon can actually dominate
the global embedding. The central icon's identity is the true label; the corner
icon depicts a competing class. No text is ever drawn.

The decoy patch location is known at generation time (the oracle decoy region).
It is used only for the oracle upper-bound baseline and for downstream
localization metrics, never by the non-oracle region scorer.

Scope: a controlled pilot stimulus, not a claim of open-world discovery, general
robustness, cross-shortcut transfer, or universal shortcut repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from causal_reliability.utils.io import ensure_dir


# Natural semantic icon classes with strong, distinct colour signatures. These
# are recognisable to zero-shot CLIP as clip-art / icons without any text.
SEMANTIC_DECOY_CLASSES = ["sun", "heart", "leaf", "moon"]

_BACKGROUND = (240, 241, 238)

# Saturated, well-separated colours so a competing-class corner icon is highly
# salient to CLIP. The decoy uses the same colour as its class (no extra cue).
_CLASS_COLORS = {
    "sun": (250, 196, 18),
    "heart": (214, 38, 52),
    "leaf": (46, 158, 64),
    "moon": (96, 124, 214),
}


@dataclass(frozen=True)
class SemanticDecoyBundle:
    examples: list[dict[str, Any]]
    class_names: list[str]


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


def _rotate(points, cx: float, cy: float, deg: float):
    rad = np.deg2rad(deg)
    ca, sa = np.cos(rad), np.sin(rad)
    out = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        out.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    return out


def _draw_sun(draw: ImageDraw.ImageDraw, box, color) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    n = 12
    for k in range(n):
        ang = 2.0 * np.pi * k / n
        spread = 0.22
        tip = (cx + 1.02 * rx * np.cos(ang), cy + 1.02 * ry * np.sin(ang))
        b1 = (cx + 0.6 * rx * np.cos(ang - spread), cy + 0.6 * ry * np.sin(ang - spread))
        b2 = (cx + 0.6 * rx * np.cos(ang + spread), cy + 0.6 * ry * np.sin(ang + spread))
        draw.polygon([tip, b1, b2], fill=color)
    draw.ellipse([cx - 0.62 * rx, cy - 0.62 * ry, cx + 0.62 * rx, cy + 0.62 * ry], fill=color)


def _draw_heart(draw: ImageDraw.ImageDraw, box, color) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2.0
    lobe_r = w * 0.27
    lobe_cy = y0 + h * 0.33
    draw.ellipse([cx - 2 * lobe_r, lobe_cy - lobe_r, cx, lobe_cy + lobe_r], fill=color)
    draw.ellipse([cx, lobe_cy - lobe_r, cx + 2 * lobe_r, lobe_cy + lobe_r], fill=color)
    draw.polygon(
        [(x0 + w * 0.06, lobe_cy), (x1 - w * 0.06, lobe_cy), (cx, y1 - h * 0.04)],
        fill=color,
    )


def _draw_leaf(draw: ImageDraw.ImageDraw, box, color) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    n = 26
    top, bot = [], []
    for i in range(n + 1):
        t = i / n
        x = (x0 + 0.06 * rx) + (2 * rx - 0.12 * rx) * t
        bulge = ry * 0.74 * np.sin(np.pi * t)
        top.append((x, cy - bulge))
        bot.append((x, cy + bulge))
    pts = top + bot[::-1]
    pts = _rotate(pts, cx, cy, -35.0)
    draw.polygon(pts, fill=color)
    # midrib
    a = _rotate([(x0 + 0.1 * rx, cy), (x1 - 0.1 * rx, cy)], cx, cy, -35.0)
    draw.line(a, fill=_BACKGROUND, width=max(1, int(rx * 0.10)))


def _draw_moon(draw: ImageDraw.ImageDraw, box, color) -> None:
    x0, y0, x1, y1 = box
    w = x1 - x0
    draw.ellipse([x0, y0, x1, y1], fill=color)
    # carve a crescent with an offset background-coloured disk
    off = w * 0.34
    draw.ellipse([x0 + off, y0 - 0.02 * w, x1 + off, y1 + 0.02 * w], fill=_BACKGROUND)


_DRAWERS = {"sun": _draw_sun, "heart": _draw_heart, "leaf": _draw_leaf, "moon": _draw_moon}


def _draw_icon(draw: ImageDraw.ImageDraw, label: int, box, badge: bool = False) -> None:
    name = SEMANTIC_DECOY_CLASSES[label]
    color = _CLASS_COLORS[name]
    if badge:
        # A soft tinted circular badge increases the corner icon's visual mass
        # (salience) without adding any text or class-identity cue beyond colour.
        x0, y0, x1, y1 = box
        pad = (x1 - x0) * 0.04
        tint = tuple(int(0.30 * c + 0.70 * b) for c, b in zip(color, _BACKGROUND))
        draw.ellipse([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill=tint)
    _DRAWERS[name](draw, [float(v) for v in box], color)


def render_semantic_decoy_image(
    label: int,
    decoy_label: int | None,
    corner: int,
    size: int = 224,
    object_frac: tuple[float, float] = (0.46, 0.46),
    decoy_frac: float = 0.30,
    draw_decoy: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render a centred causal icon plus an optional corner decoy icon.

    No text is ever drawn. The decoy is a coloured icon of a competing class.
    """

    img = Image.new("RGB", (size, size), _BACKGROUND)
    draw = ImageDraw.Draw(img)

    ow, oh = object_frac
    obj_box = [
        size * (0.5 - ow / 2),
        size * (0.49 - oh / 2),
        size * (0.5 + ow / 2),
        size * (0.49 + oh / 2),
    ]
    object_bbox = [int(round(v)) for v in obj_box]
    _draw_icon(draw, label, obj_box, badge=False)

    decoy_bbox = _corner_box(corner, size, decoy_frac)
    if draw_decoy and decoy_label is not None:
        _draw_icon(draw, decoy_label, decoy_bbox, badge=True)

    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr, {
        "class_name": SEMANTIC_DECOY_CLASSES[label],
        "decoy_label": decoy_label,
        "decoy_class_name": "" if decoy_label is None else SEMANTIC_DECOY_CLASSES[decoy_label],
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


def make_semantic_decoy_dataset(
    n_per_condition: int = 64,
    size: int = 224,
    regimes: list[str] | None = None,
    split: str = "test",
    start_id: int = 0,
    decoy_frac: float = 0.30,
    object_frac: tuple[float, float] = (0.46, 0.46),
) -> SemanticDecoyBundle:
    """Build the semantic-decoy dataset (parallel to the visual-decoy family)."""

    regimes = regimes or ["no_decoy", "misleading_decoy", "aligned_decoy"]
    n_classes = len(SEMANTIC_DECOY_CLASSES)
    per_class = max(1, n_per_condition // n_classes)
    examples: list[dict[str, Any]] = []
    example_id = start_id
    for regime in regimes:
        for label, class_name in enumerate(SEMANTIC_DECOY_CLASSES):
            for j in range(per_class):
                decoy_label = _decoy_for(label, regime, j, n_classes)
                corner = (label + j) % 4
                image, meta = render_semantic_decoy_image(
                    label,
                    decoy_label,
                    corner,
                    size=size,
                    object_frac=object_frac,
                    decoy_frac=decoy_frac,
                    draw_decoy=regime != "no_decoy",
                )
                clean_image, _ = render_semantic_decoy_image(
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
    return SemanticDecoyBundle(examples, SEMANTIC_DECOY_CLASSES.copy())


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
