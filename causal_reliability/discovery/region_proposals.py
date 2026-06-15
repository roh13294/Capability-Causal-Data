from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class RegionProposal:
    candidate_id: str
    bbox: BBox
    proposal_type: str
    area_fraction: float
    edge_density: float
    horizontalness_score: float
    center_overlap_score: float
    border_distance: float
    center_distance: float
    width: int
    height: int
    aspect_ratio: float

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["bbox"] = list(self.bbox)
        return row


def _clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(width - 1, int(x0)))
    y0 = max(0, min(height - 1, int(y0)))
    x1 = max(x0 + 1, min(width, int(x1)))
    y1 = max(y0 + 1, min(height, int(y1)))
    return x0, y0, x1, y1


def _gray_edges(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L")).astype(np.float32) / 255.0
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    return mag


def _center_overlap(bbox: BBox, width: int, height: int) -> float:
    x0, y0, x1, y1 = bbox
    cx0, cy0 = int(width * 0.25), int(height * 0.20)
    cx1, cy1 = int(width * 0.75), int(height * 0.72)
    ix0, iy0 = max(x0, cx0), max(y0, cy0)
    ix1, iy1 = min(x1, cx1), min(y1, cy1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = max(1, (x1 - x0) * (y1 - y0))
    return float(inter / area)


def _features(candidate_id: str, bbox: BBox, proposal_type: str, edges: np.ndarray, width: int, height: int) -> RegionProposal:
    x0, y0, x1, y1 = _clip_bbox(bbox, width, height)
    bw, bh = x1 - x0, y1 - y0
    area = float(bw * bh) / float(max(1, width * height))
    patch_edges = edges[y0:y1, x0:x1]
    edge_density = float((patch_edges > 0.08).mean()) if patch_edges.size else 0.0
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    diag = float(np.hypot(width, height))
    center_distance = float(np.hypot(cx - width / 2.0, cy - height / 2.0) / max(diag / 2.0, 1.0))
    aspect = float(bw / max(1, bh))
    horizontalness = float(np.clip((aspect - 1.0) / 5.0, 0.0, 1.0) * np.clip(1.0 - bh / max(1.0, 0.35 * height), 0.0, 1.0))
    border_distance = float(min(x0, y0, width - x1, height - y1) / max(1.0, min(width, height)))
    center_overlap = _center_overlap((x0, y0, x1, y1), width, height)
    return RegionProposal(
        candidate_id,
        (x0, y0, x1, y1),
        proposal_type,
        area,
        edge_density,
        horizontalness,
        center_overlap,
        border_distance,
        center_distance,
        bw,
        bh,
        aspect,
    )


def _dedupe(proposals: Iterable[RegionProposal]) -> list[RegionProposal]:
    seen: set[tuple[str, BBox]] = set()
    out: list[RegionProposal] = []
    for prop in proposals:
        key = (prop.proposal_type, prop.bbox)
        if key in seen:
            continue
        seen.add(key)
        out.append(prop)
    return out


def sliding_window_proposals(image: Image.Image, edges: np.ndarray) -> list[RegionProposal]:
    width, height = image.size
    specs = [
        ("small", 0.18, 0.14, 0.16),
        ("medium", 0.30, 0.24, 0.22),
        ("wide_horizontal", 0.44, 0.16, 0.20),
    ]
    proposals: list[RegionProposal] = []
    idx = 0
    for name, wf, hf, step_f in specs:
        bw, bh = max(8, int(width * wf)), max(8, int(height * hf))
        step_x, step_y = max(6, int(width * step_f)), max(6, int(height * step_f))
        xs = sorted(set(list(range(0, max(1, width - bw + 1), step_x)) + [max(0, width - bw)]))
        ys = sorted(set(list(range(0, max(1, height - bh + 1), step_y)) + [max(0, height - bh)]))
        for y in ys:
            for x in xs:
                proposals.append(_features(f"slide_{idx:04d}", (x, y, x + bw, y + bh), f"sliding_{name}", edges, width, height))
                idx += 1
    return proposals


def textness_proposals(image: Image.Image, edges: np.ndarray, max_components: int = 16) -> list[RegionProposal]:
    width, height = image.size
    arr = np.asarray(image.convert("L")).astype(np.float32) / 255.0
    local = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES)).astype(np.float32) / 255.0
    mask = (edges > np.quantile(edges, 0.88)) | (local > np.quantile(local, 0.90))
    visited = np.zeros(mask.shape, dtype=bool)
    comps: list[tuple[int, BBox]] = []
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                sx, sy = stack.pop()
                xs.append(sx)
                ys.append(sy)
                for nx in (sx - 1, sx, sx + 1):
                    for ny in (sy - 1, sy, sy + 1):
                        if nx < 0 or ny < 0 or nx >= width or ny >= height or visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if len(xs) < max(8, width * height // 2200):
                continue
            x0, x1 = min(xs), max(xs) + 1
            y0, y1 = min(ys), max(ys) + 1
            pad = max(3, int(0.025 * max(width, height)))
            bbox = _clip_bbox((x0 - pad, y0 - pad, x1 + pad, y1 + pad), width, height)
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if bw * bh > 0.30 * width * height or bw < 6 or bh < 6:
                continue
            contrast = float(arr[bbox[1] : bbox[3], bbox[0] : bbox[2]].std())
            comps.append((int(len(xs) + 1000 * contrast), bbox))
    comps = sorted(comps, key=lambda item: item[0], reverse=True)[:max_components]
    return [_features(f"textness_{i:04d}", bbox, "textness_high_frequency", edges, width, height) for i, (_, bbox) in enumerate(comps)]


def text_box_component_proposals(image: Image.Image, edges: np.ndarray, max_components: int = 24) -> list[RegionProposal]:
    """Recover separated high-contrast text boxes without using overlay metadata."""

    width, height = image.size
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    bg = np.median(arr.reshape(-1, 3), axis=0)
    color_delta = np.linalg.norm(arr - bg[None, None, :], axis=2)
    bright_panel = (gray > 0.90) & (color_delta > 0.035)
    colored_ink = (arr[:, :, 0] > arr[:, :, 1] + 0.18) | (arr[:, :, 2] > arr[:, :, 1] + 0.18)
    edge_mask = edges > max(0.035, float(np.quantile(edges, 0.82)))
    mask = bright_panel | (colored_ink & edge_mask)
    visited = np.zeros(mask.shape, dtype=bool)
    comps: list[tuple[float, BBox]] = []
    min_pixels = max(10, width * height // 3500)
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                sx, sy = stack.pop()
                xs.append(sx)
                ys.append(sy)
                for nx in (sx - 1, sx, sx + 1):
                    for ny in (sy - 1, sy, sy + 1):
                        if nx < 0 or ny < 0 or nx >= width or ny >= height or visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if len(xs) < min_pixels:
                continue
            x0, x1 = min(xs), max(xs) + 1
            y0, y1 = min(ys), max(ys) + 1
            pad_x = max(4, int(0.020 * width))
            pad_y = max(3, int(0.016 * height))
            bbox = _clip_bbox((x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y), width, height)
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            area = bw * bh
            aspect = bw / max(1, bh)
            if area > 0.18 * width * height or bw < 10 or bh < 8 or aspect < 1.1 or aspect > 8.5:
                continue
            center_overlap = _center_overlap(bbox, width, height)
            score = float(len(xs)) + 120.0 * aspect + 40.0 * (1.0 - center_overlap)
            comps.append((score, bbox))
    comps = sorted(comps, key=lambda item: item[0], reverse=True)[:max_components]
    return [_features(f"text_box_{i:04d}", bbox, "text_box_component", edges, width, height) for i, (_, bbox) in enumerate(comps)]


def horizontal_text_band_proposals(image: Image.Image, edges: np.ndarray, n_bands: int = 10) -> list[RegionProposal]:
    width, height = image.size
    row_density = (edges > max(0.06, float(np.quantile(edges, 0.86)))).mean(axis=1)
    smooth = np.convolve(row_density, np.ones(5) / 5.0, mode="same")
    centers = np.argsort(smooth)[::-1][:n_bands]
    proposals: list[RegionProposal] = []
    for i, cy in enumerate(sorted(set(int(c) for c in centers))):
        for hf, wf in [(0.10, 0.78), (0.14, 0.92), (0.08, 0.55)]:
            bh = max(8, int(height * hf))
            bw = max(12, int(width * wf))
            y0 = int(cy - bh // 2)
            x0 = int((width - bw) // 2)
            proposals.append(_features(f"text_band_{i:04d}_{int(hf * 100):02d}", (x0, y0, x0 + bw, y0 + bh), "horizontal_text_band", edges, width, height))
    return proposals


def corner_edge_watermark_proposals(image: Image.Image, edges: np.ndarray) -> list[RegionProposal]:
    width, height = image.size
    boxes: list[BBox] = []
    for wf, hf in [(0.34, 0.13), (0.46, 0.16), (0.24, 0.24)]:
        bw, bh = max(8, int(width * wf)), max(8, int(height * hf))
        boxes.extend(
            [
                (0, 0, bw, bh),
                (width - bw, 0, width, bh),
                (0, height - bh, bw, height),
                (width - bw, height - bh, width, height),
                ((width - bw) // 2, 0, (width + bw) // 2, bh),
                ((width - bw) // 2, height - bh, (width + bw) // 2, height),
            ]
        )
    return [_features(f"watermark_{i:04d}", bbox, "corner_edge_watermark", edges, width, height) for i, bbox in enumerate(boxes)]


def random_patch_proposals(image: Image.Image, edges: np.ndarray, seed: int = 0, n: int = 24) -> list[RegionProposal]:
    width, height = image.size
    rng = np.random.default_rng(seed)
    proposals: list[RegionProposal] = []
    for i in range(n):
        wf, hf = rng.choice([0.18, 0.25, 0.35]), rng.choice([0.12, 0.18, 0.25])
        if rng.random() < 0.35:
            wf, hf = 0.44, 0.16
        bw, bh = max(8, int(width * wf)), max(8, int(height * hf))
        x = int(rng.integers(0, max(1, width - bw + 1)))
        y = int(rng.integers(0, max(1, height - bh + 1)))
        proposals.append(_features(f"random_{i:04d}", (x, y, x + bw, y + bh), "random_patch_control", edges, width, height))
    return proposals


def center_object_control_proposals(image: Image.Image, edges: np.ndarray) -> list[RegionProposal]:
    width, height = image.size
    boxes = [
        (int(width * 0.25), int(height * 0.18), int(width * 0.75), int(height * 0.72)),
        (int(width * 0.32), int(height * 0.25), int(width * 0.68), int(height * 0.62)),
    ]
    return [_features(f"object_control_{i:04d}", bbox, "object_control", edges, width, height) for i, bbox in enumerate(boxes)]


def proposal_from_bbox(
    image: Image.Image | np.ndarray,
    bbox: BBox,
    candidate_id: str,
    proposal_type: str,
) -> RegionProposal:
    """Build a feature-complete proposal for an explicitly supplied bbox.

    Used by controlled experiments that construct a fixed candidate set (e.g. a
    known decoy region plus distractors) instead of relying on the generic
    proposer. Features are derived only from pixels and geometry.
    """

    pil = Image.fromarray((np.asarray(image).clip(0, 1) * 255).astype(np.uint8)).convert("RGB") if not isinstance(image, Image.Image) else image.convert("RGB")
    edges = _gray_edges(pil)
    width, height = pil.size
    return _features(candidate_id, bbox, proposal_type, edges, width, height)


def generate_region_proposals(image: Image.Image | np.ndarray, seed: int = 0, max_candidates: int = 96) -> list[RegionProposal]:
    pil = Image.fromarray((np.asarray(image).clip(0, 1) * 255).astype(np.uint8)).convert("RGB") if not isinstance(image, Image.Image) else image.convert("RGB")
    edges = _gray_edges(pil)
    proposals = []
    proposals.extend(sliding_window_proposals(pil, edges))
    proposals.extend(text_box_component_proposals(pil, edges))
    proposals.extend(textness_proposals(pil, edges))
    proposals.extend(horizontal_text_band_proposals(pil, edges))
    proposals.extend(corner_edge_watermark_proposals(pil, edges))
    proposals.extend(random_patch_proposals(pil, edges, seed=seed))
    proposals.extend(center_object_control_proposals(pil, edges))
    text_like = {"text_box_component", "textness_high_frequency", "horizontal_text_band", "corner_edge_watermark"}
    ranked = sorted(_dedupe(proposals), key=lambda p: (p.proposal_type in text_like, p.edge_density + 0.5 * p.horizontalness_score - 0.25 * p.center_overlap_score), reverse=True)
    return ranked[:max_candidates]
