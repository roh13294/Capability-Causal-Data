from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from causal_reliability.utils.io import ensure_dir


CLIP_OVERLAY_CLASSES = ["circle", "square", "triangle", "star"]


@dataclass(frozen=True)
class ClipOverlayBundle:
    examples: list[dict[str, Any]]
    class_names: list[str]


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", max(10, size // 7))
    except Exception:
        return ImageFont.load_default()


def _shape_points(label: int, size: int) -> list[tuple[float, float]]:
    cx, cy, r = size * 0.5, size * 0.48, size * 0.22
    if label == 2:
        return [(cx, cy - 1.15 * r), (cx - 1.15 * r, cy + r), (cx + 1.15 * r, cy + r)]
    points = []
    for k in range(10):
        rad = r if k % 2 == 0 else r * 0.45
        ang = -np.pi / 2 + k * np.pi / 5
        points.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
    return points


def render_overlay_image(
    label: int,
    overlay_label: int,
    size: int = 224,
    neutral_word: str | None = None,
    draw_overlay: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    name = CLIP_OVERLAY_CLASSES[label]
    overlay_word = CLIP_OVERLAY_CLASSES[overlay_label] if neutral_word is None else neutral_word
    img = Image.new("RGB", (size, size), (238, 240, 235))
    draw = ImageDraw.Draw(img)
    shape_mask_img = Image.new("L", (size, size), 0)
    shape_draw = ImageDraw.Draw(shape_mask_img)
    box = [size * 0.28, size * 0.22, size * 0.72, size * 0.66]
    fill = (32, 34, 36)
    if label == 0:
        draw.ellipse(box, fill=fill)
        shape_draw.ellipse(box, fill=255)
    elif label == 1:
        draw.rectangle(box, fill=fill)
        shape_draw.rectangle(box, fill=255)
    elif label == 2:
        pts = _shape_points(label, size)
        draw.polygon(pts, fill=fill)
        shape_draw.polygon(pts, fill=255)
    else:
        pts = _shape_points(label, size)
        draw.polygon(pts, fill=fill)
        shape_draw.polygon(pts, fill=255)

    text_mask_img = Image.new("L", (size, size), 0)
    text_draw = ImageDraw.Draw(text_mask_img)
    font = _font(size)
    bbox = draw.textbbox((0, 0), overlay_word, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(4, (size - tw) // 2)
    y = int(size * 0.72)
    pad = max(4, size // 45)
    overlay_bbox = [int(x - pad), int(y - pad), int(x + tw + pad), int(y + th + pad)]
    if draw_overlay and overlay_word:
        draw.rounded_rectangle(overlay_bbox, radius=3, fill=(255, 255, 255))
        draw.text((x, y), overlay_word, font=font, fill=(180, 20, 24))
        text_draw.rectangle(overlay_bbox, fill=255)

    arr = np.asarray(img).astype(np.float32) / 255.0
    shape_mask = np.asarray(shape_mask_img) > 0
    text_mask = np.asarray(text_mask_img) > 0
    background_mask = ~(shape_mask | text_mask)
    return arr, {
        "class_name": name,
        "overlay_word": overlay_word,
        "overlay_label": overlay_label,
        "shape_mask": shape_mask,
        "object_mask": shape_mask,
        "text_mask": text_mask,
        "shortcut_mask": text_mask,
        "background_mask": background_mask,
        "overlay_bbox": overlay_bbox,
    }


def neutralize_overlay_array(image: np.ndarray, bbox: list[int], strategy: str, size: int | None = None) -> np.ndarray:
    arr = np.asarray(image).copy()
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if strategy in {"mask_overlay_bbox_background", "replace_overlay_bbox_background"}:
        arr[y0:y1, x0:x1, :] = np.array([238, 240, 235], dtype=np.float32) / 255.0
        return arr
    if strategy == "blur_overlay_bbox":
        pil = Image.fromarray((arr * 255).astype(np.uint8))
        patch = pil.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(radius=max(2, (size or w) // 32)))
        pil.paste(patch, (x0, y0))
        return np.asarray(pil).astype(np.float32) / 255.0
    if strategy == "crop_out_overlay":
        cropped = arr[: max(1, y0), :, :]
        pil = Image.fromarray((cropped * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC)
        return np.asarray(pil).astype(np.float32) / 255.0
    return arr


def _overlay_for(label: int, regime: str, index: int, n_classes: int) -> int:
    if regime == "aligned_overlay":
        return label
    if regime == "misleading_overlay":
        return (label + 1 + index % (n_classes - 1)) % n_classes
    if regime == "mixed_overlay":
        return label if index % 2 == 0 else (label + 1) % n_classes
    raise ValueError(f"unknown overlay regime: {regime}")


def make_clip_overlay_dataset(
    n_per_class: int = 8,
    size: int = 224,
    regimes: list[str] | None = None,
    split: str = "test",
    start_id: int = 0,
) -> ClipOverlayBundle:
    regimes = regimes or ["aligned_overlay", "misleading_overlay", "mixed_overlay", "neutral_overlay", "no_overlay"]
    examples: list[dict[str, Any]] = []
    example_id = start_id
    n_classes = len(CLIP_OVERLAY_CLASSES)
    for regime in regimes:
        for label, class_name in enumerate(CLIP_OVERLAY_CLASSES):
            for j in range(n_per_class):
                if regime == "neutral_overlay":
                    overlay = label
                    image, meta = render_overlay_image(label, overlay, size=size, neutral_word=["object", "shape", "image"][j % 3])
                    overlay_relation = "neutral"
                elif regime == "no_overlay":
                    overlay = label
                    image, meta = render_overlay_image(label, overlay, size=size, neutral_word="", draw_overlay=False)
                    overlay_relation = "none"
                else:
                    overlay = _overlay_for(label, regime, j, n_classes)
                    image, meta = render_overlay_image(label, overlay, size=size)
                    overlay_relation = "aligned" if overlay == label else "misleading"
                no_text, no_text_meta = render_overlay_image(label, overlay, size=size, neutral_word="", draw_overlay=False)
                neutral, neutral_meta = render_overlay_image(label, overlay, size=size, neutral_word="object")
                neutral_shape, neutral_shape_meta = render_overlay_image(label, overlay, size=size, neutral_word="shape")
                correct, correct_meta = render_overlay_image(label, label, size=size)
                other = (label + 2) % n_classes
                other_img, other_meta = render_overlay_image(label, other, size=size)
                masked = neutralize_overlay_array(image, meta["overlay_bbox"], "mask_overlay_bbox_background", size)
                blurred = neutralize_overlay_array(image, meta["overlay_bbox"], "blur_overlay_bbox", size)
                cropped = neutralize_overlay_array(image, meta["overlay_bbox"], "crop_out_overlay", size)
                examples.append(
                    {
                        "example_id": example_id,
                        "split": split,
                        "regime": regime,
                        "label": label,
                        "true_label": class_name,
                        "class_name": class_name,
                        "shortcut": meta["overlay_word"],
                        "overlay_text": meta["overlay_word"] if overlay_relation != "none" else "",
                        "shortcut_label": overlay,
                        "overlay_relation": overlay_relation,
                        "overlay_bbox": meta["overlay_bbox"],
                        "image": image,
                        "counterfactual_image": neutral,
                        "overlay_removed_image": no_text,
                        "neutral_overlay_image": neutral,
                        "neutral_word_overlay_image": neutral_shape,
                        "mask_overlay_bbox_background_image": masked,
                        "replace_overlay_bbox_background_image": masked,
                        "blur_overlay_bbox_image": blurred,
                        "crop_out_overlay_image": cropped,
                        "correct_overlay_image": correct,
                        "other_overlay_image": other_img,
                        "object_mask": meta["object_mask"],
                        "shape_mask": meta["shape_mask"],
                        "shortcut_mask": meta["shortcut_mask"],
                        "text_mask": meta["text_mask"],
                        "background_mask": meta["background_mask"],
                        "counterfactual_text_mask": neutral_meta["text_mask"],
                        "neutral_word_text_mask": neutral_shape_meta["text_mask"],
                        "removed_text_mask": no_text_meta["text_mask"],
                        "correct_text_mask": correct_meta["text_mask"],
                        "other_text_mask": other_meta["text_mask"],
                    }
                )
                example_id += 1
    return ClipOverlayBundle(examples, CLIP_OVERLAY_CLASSES.copy())


def save_example_images(examples: list[dict[str, Any]], out_dir: str | Path, keys: list[str] | None = None) -> None:
    keys = keys or [
        "image",
        "overlay_removed_image",
        "neutral_overlay_image",
        "neutral_word_overlay_image",
        "mask_overlay_bbox_background_image",
        "replace_overlay_bbox_background_image",
        "blur_overlay_bbox_image",
        "crop_out_overlay_image",
    ]
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


def save_overlay_grid(examples: list[dict[str, Any]], path: str | Path, key: str = "image", n: int = 8) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    ensure_dir(path.parent)
    take = examples[: max(1, min(n, len(examples)))]
    fig, axes = plt.subplots(1, len(take), figsize=(1.55 * len(take), 1.8))
    axes = np.atleast_1d(axes)
    for ax, ex in zip(axes, take):
        ax.imshow(ex[key])
        ax.set_title(f"{ex['class_name']} / {ex['shortcut']}", fontsize=7)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_default_example_grids(bundle: ClipOverlayBundle, out_dir: str | Path) -> None:
    out = ensure_dir(out_dir)
    for regime, name in (("aligned_overlay", "aligned_examples.png"), ("misleading_overlay", "misleading_examples.png")):
        save_overlay_grid([ex for ex in bundle.examples if ex["regime"] == regime], out / name)
    misleading = [ex for ex in bundle.examples if ex["regime"] == "misleading_overlay"]
    save_overlay_grid(misleading, out / "counterfactual_examples.png", key="counterfactual_image")
