from __future__ import annotations

"""Deterministic region-neutralization operators for natural-text repair sweeps.

This module supplies a registry of *intervention operators* that neutralize one
or more boxes in a natural image. They are used by
``run_natural_text_intervention_sweep`` to diagnose whether strict natural-text
repair is limited by the **intervention operator / masking strategy** rather than
by CIC proposal *selection*.

Every operator is fully deterministic (no RNG, no time, no global state) and
modifies pixels **only** inside the target box (or, for ``expansion > 1``, inside
the expanded-and-clipped box). The OpenCV Telea inpaint operator requires ``cv2``;
when it is unavailable the operator is reported as ``available=False`` and leaves
the image unchanged rather than raising.

These operators carry no label, OCR-text, or correctness information: they take an
image plus geometry only. Operator *selection* for any headline/gate must remain
label-free (see the sweep runner's non-leakage guard).
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter

try:  # pragma: no cover - exercised indirectly by the availability test
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover - cv2 frequently absent
    cv2 = None  # type: ignore
    _HAS_CV2 = False


BBox = tuple[int, int, int, int]


def cv2_available() -> bool:
    """True iff OpenCV (``cv2``) imported successfully in this environment."""

    return bool(_HAS_CV2)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def clip_box(bbox: BBox, width: int, height: int) -> BBox:
    x0, y0, x1, y1 = (int(round(float(v))) for v in bbox)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    return (x0, y0, x1, y1)


def expand_box(bbox: BBox, factor: float, width: int, height: int) -> BBox:
    """Expand ``bbox`` about its centre by ``factor`` and clip to image bounds.

    ``factor == 1.0`` returns the (clipped) box unchanged. The result always stays
    within ``[0, width] x [0, height]``.
    """

    x0, y0, x1, y1 = clip_box(bbox, width, height)
    if factor == 1.0:
        return (x0, y0, x1, y1)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw, bh = (x1 - x0), (y1 - y0)
    nw, nh = bw * float(factor), bh * float(factor)
    nx0 = int(np.floor(cx - nw / 2.0))
    ny0 = int(np.floor(cy - nh / 2.0))
    nx1 = int(np.ceil(cx + nw / 2.0))
    ny1 = int(np.ceil(cy + nh / 2.0))
    return clip_box((nx0, ny0, nx1, ny1), width, height)


def _is_degenerate(box: BBox) -> bool:
    x0, y0, x1, y1 = box
    return x1 <= x0 or y1 <= y0


# --------------------------------------------------------------------------- #
# Array helpers
# --------------------------------------------------------------------------- #
def _to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB")).astype(np.float32) / 255.0


def _from_array(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))


def _surround_ring(arr: np.ndarray, box: BBox) -> np.ndarray:
    """Pixels in a padded ring *around* ``box`` (excluding the box itself)."""

    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box
    pad = max(4, int(0.06 * max(w, h)))
    sx0, sy0 = max(0, x0 - pad), max(0, y0 - pad)
    sx1, sy1 = min(w, x1 + pad), min(h, y1 + pad)
    region = arr[sy0:sy1, sx0:sx1].astype(np.float64).copy()
    region[y0 - sy0 : y1 - sy0, x0 - sx0 : x1 - sx0] = np.nan
    flat = region.reshape(-1, arr.shape[2])
    flat = flat[~np.isnan(flat).any(axis=1)]
    if len(flat) == 0:
        flat = arr.reshape(-1, arr.shape[2]).astype(np.float64)
    return flat


def _border_band(arr: np.ndarray, box: BBox) -> np.ndarray:
    """Pixels along the inner perimeter band of ``box`` (its own border)."""

    x0, y0, x1, y1 = box
    patch = arr[y0:y1, x0:x1].astype(np.float64)
    bh, bw = patch.shape[:2]
    band = max(1, int(round(0.12 * min(bh, bw))))
    band = min(band, bh, bw)
    mask = np.zeros((bh, bw), dtype=bool)
    mask[:band, :] = True
    mask[-band:, :] = True
    mask[:, :band] = True
    mask[:, -band:] = True
    flat = patch[mask].reshape(-1, arr.shape[2])
    if len(flat) == 0:
        flat = patch.reshape(-1, arr.shape[2])
    return flat


# --------------------------------------------------------------------------- #
# Operator specification + registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Operator:
    name: str
    kind: str  # constant | local_mean | local_median | blur | pixelate | border_fill | inpaint_telea
    value: float | None = None
    expansion: float = 1.0
    requires_cv2: bool = False

    @property
    def is_expanded(self) -> bool:
        return self.expansion != 1.0


def default_operators() -> list[Operator]:
    """The full deterministic operator panel evaluated by the sweep."""

    return [
        Operator("gray_fill", "constant", value=0.5),
        Operator("black_fill", "constant", value=0.0),
        Operator("white_fill", "constant", value=1.0),
        Operator("local_mean_fill", "local_mean"),
        Operator("local_median_fill", "local_median"),
        Operator("gaussian_blur", "blur"),
        Operator("pixelation", "pixelate"),
        Operator("background_border_fill", "border_fill"),
        Operator("telea_inpaint", "inpaint_telea", requires_cv2=True),
        Operator("expanded_gray_fill_1.10", "constant", value=0.5, expansion=1.10),
        Operator("expanded_gray_fill_1.25", "constant", value=0.5, expansion=1.25),
        Operator("expanded_blur_1.10", "blur", expansion=1.10),
        Operator("expanded_blur_1.25", "blur", expansion=1.25),
    ]


# --------------------------------------------------------------------------- #
# Per-box fill primitives (operate in place on a float array)
# --------------------------------------------------------------------------- #
def _blur_radius(width: int, height: int) -> int:
    return max(2, int(0.04 * max(width, height)))


def _apply_blur_box(image: Image.Image, box: BBox, width: int, height: int) -> Image.Image:
    out = image.copy()
    x0, y0, x1, y1 = box
    patch = out.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(radius=_blur_radius(width, height)))
    out.paste(patch, (x0, y0))
    return out


def _apply_pixelate_box(arr: np.ndarray, box: BBox) -> None:
    x0, y0, x1, y1 = box
    patch = arr[y0:y1, x0:x1]
    bh, bw = patch.shape[:2]
    block = max(3, int(round(max(bh, bw) / 8.0)))
    small_h = max(1, bh // block)
    small_w = max(1, bw // block)
    pil = _from_array(patch)
    down = pil.resize((small_w, small_h), Image.Resampling.NEAREST)
    up = down.resize((bw, bh), Image.Resampling.NEAREST)
    arr[y0:y1, x0:x1] = _to_array(up)


def _apply_fill_box(arr: np.ndarray, box: BBox, op: Operator) -> None:
    x0, y0, x1, y1 = box
    if op.kind == "constant":
        arr[y0:y1, x0:x1] = float(op.value if op.value is not None else 0.5)
    elif op.kind == "local_mean":
        arr[y0:y1, x0:x1] = _surround_ring(arr, box).mean(axis=0).astype(np.float32)
    elif op.kind == "local_median":
        arr[y0:y1, x0:x1] = np.median(_surround_ring(arr, box), axis=0).astype(np.float32)
    elif op.kind == "border_fill":
        arr[y0:y1, x0:x1] = np.median(_border_band(arr, box), axis=0).astype(np.float32)
    elif op.kind == "pixelate":
        _apply_pixelate_box(arr, box)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unsupported array-fill kind: {op.kind}")


def _apply_telea(image: Image.Image, boxes: Iterable[BBox], width: int, height: int) -> Image.Image:
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        x0, y0, x1, y1 = box
        if _is_degenerate((x0, y0, x1, y1)):
            continue
        mask[y0:y1, x0:x1] = 255
    radius = max(3, int(0.03 * max(width, height)))
    out = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# --------------------------------------------------------------------------- #
# Public application API
# --------------------------------------------------------------------------- #
def operator_boxes(boxes: Iterable[BBox], op: Operator, width: int, height: int) -> list[BBox]:
    """The (expanded, clipped) non-degenerate boxes this operator will modify."""

    out: list[BBox] = []
    for box in boxes:
        eb = expand_box(tuple(int(v) for v in box), op.expansion, width, height)
        if not _is_degenerate(eb):
            out.append(eb)
    return out


def apply_operator(image: Image.Image, boxes: Iterable[BBox], op: Operator) -> tuple[Image.Image, bool]:
    """Apply ``op`` to ``boxes`` of ``image``.

    Returns ``(neutralized_image, available)``. When the operator requires ``cv2``
    and it is unavailable, returns ``(image.copy(), False)`` (a graceful skip).
    Pixels outside the (expanded) target boxes are never modified.
    """

    pil = image.convert("RGB")
    width, height = pil.size
    target_boxes = operator_boxes(list(boxes), op, width, height)

    if op.requires_cv2 and not _HAS_CV2:
        return pil.copy(), False
    if not target_boxes:
        return pil.copy(), True

    if op.kind == "inpaint_telea":
        return _apply_telea(pil, target_boxes, width, height), True

    if op.kind == "blur":
        out = pil
        for box in target_boxes:
            out = _apply_blur_box(out, box, width, height)
        return out, True

    arr = _to_array(pil)
    for box in target_boxes:
        _apply_fill_box(arr, box, op)
    return _from_array(arr), True
