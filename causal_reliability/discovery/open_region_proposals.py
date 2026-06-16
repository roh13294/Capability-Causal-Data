from __future__ import annotations

"""Shortcut-agnostic, open-candidate region proposals for natural-image CIC.

The goal is *open-candidate intervention search*: given a natural image, propose a
broad set of candidate regions **without telling CIC which region is the
shortcut**. Candidate families are:

* ``grid_patch``          - multi-scale tiled patches (always available)
* ``connected_component`` - high-frequency connected components
* ``high_contrast``       - bright/colored high-contrast text-box-like regions
* ``edge_dense``          - edge-dense horizontal bands and corner regions
* ``ocr_text_box``        - OCR/detector boxes, when supplied (geometry only)
* ``object_box``          - annotation object boxes (optional; off by default)
* ``random_patch``        - random control patches for the matched-random baseline

The generator deliberately takes **no** true label, shortcut identity, OCR text
*content*, correctness, or benchmark condition. Supplied boxes are used only as
candidate *geometry*. Scoring (which region to neutralize) is done downstream by
``cic_region_scoring.score_region_candidates`` using model predictions alone.
"""

from typing import Iterable

import numpy as np
from PIL import Image

from causal_reliability.discovery.region_proposals import (
    BBox,
    RegionProposal,
    _clip_bbox,
    _dedupe,
    _features,
    _gray_edges,
    corner_edge_watermark_proposals,
    horizontal_text_band_proposals,
    random_patch_proposals,
    text_box_component_proposals,
    textness_proposals,
)


OCR_FAMILY = "ocr_text_box"
RANDOM_FAMILY = "random_patch"
DEFAULT_GRID_SCALES: list[float] = [0.18, 0.30, 0.45]

# Map a concrete ``proposal_type`` to its high-level candidate family.
_TYPE_TO_FAMILY = {
    "textness_high_frequency": "connected_component",
    "text_box_component": "high_contrast",
    "horizontal_text_band": "edge_dense",
    "corner_edge_watermark": "edge_dense",
    "random_patch_control": RANDOM_FAMILY,
    "ocr_text_box": OCR_FAMILY,
    "object_box": "object_box",
}

NON_OCR_FAMILIES = {
    "grid_patch",
    "connected_component",
    "high_contrast",
    "edge_dense",
    "object_box",
    "sam_proposal",
}


def proposal_family(proposal_type: str) -> str:
    """Return the high-level candidate family for a ``proposal_type``."""

    if proposal_type.startswith("grid_patch"):
        return "grid_patch"
    if proposal_type.startswith("sam"):
        return "sam_proposal"
    return _TYPE_TO_FAMILY.get(proposal_type, proposal_type)


def families_present(proposals: Iterable[RegionProposal]) -> set[str]:
    return {proposal_family(p.proposal_type) for p in proposals}


def has_non_ocr_family(proposals: Iterable[RegionProposal]) -> bool:
    """True if at least one non-OCR, non-random proposal family is present.

    This guards the open-proposal claim: the method must not reduce to "use the
    OCR box", so we require a genuine non-OCR candidate family. Random control
    patches do not count as evidence of a real proposal family.
    """

    fams = {f for f in families_present(proposals) if f != RANDOM_FAMILY}
    return bool(fams & NON_OCR_FAMILIES)


def _as_pil(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray((np.asarray(image).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")


def grid_patch_proposals(
    image: Image.Image,
    edges: np.ndarray,
    scales: list[float] | None = None,
    overlap: float = 0.5,
) -> list[RegionProposal]:
    """Multi-scale tiled grid patches covering the whole image.

    Always produces proposals regardless of image content, so the method has a
    content-independent candidate family even when no OCR boxes exist.
    """

    width, height = image.size
    scales = scales or DEFAULT_GRID_SCALES
    proposals: list[RegionProposal] = []
    idx = 0
    for scale in scales:
        bw = max(8, int(width * float(scale)))
        bh = max(8, int(height * float(scale)))
        step_x = max(6, int(bw * (1.0 - overlap)))
        step_y = max(6, int(bh * (1.0 - overlap)))
        xs = sorted(set(list(range(0, max(1, width - bw + 1), step_x)) + [max(0, width - bw)]))
        ys = sorted(set(list(range(0, max(1, height - bh + 1), step_y)) + [max(0, height - bh)]))
        tag = int(round(scale * 100))
        for y in ys:
            for x in xs:
                bbox = _clip_bbox((x, y, x + bw, y + bh), width, height)
                proposals.append(_features(f"grid_{tag:02d}_{idx:04d}", bbox, f"grid_patch_{tag:02d}", edges, width, height))
                idx += 1
    return proposals


def _boxes_to_proposals(
    image: Image.Image,
    edges: np.ndarray,
    boxes: list[BBox] | None,
    proposal_type: str,
    prefix: str,
) -> list[RegionProposal]:
    if not boxes:
        return []
    width, height = image.size
    out: list[RegionProposal] = []
    for i, box in enumerate(boxes):
        bbox = _clip_bbox(tuple(int(v) for v in box), width, height)
        out.append(_features(f"{prefix}_{i:04d}", bbox, proposal_type, edges, width, height))
    return out


def generate_open_region_proposals(
    image: Image.Image | np.ndarray,
    *,
    text_boxes: list[BBox] | None = None,
    object_boxes: list[BBox] | None = None,
    seed: int = 0,
    max_candidates: int = 64,
    grid_scales: list[float] | None = None,
    enable_grid: bool = True,
    enable_connected_components: bool = True,
    enable_high_contrast: bool = True,
    enable_edge_dense: bool = True,
    enable_ocr_family: bool = True,
    enable_object_box_family: bool = False,
    enable_random_control: bool = True,
    enable_sam: bool = False,
    sam_proposer=None,
) -> list[RegionProposal]:
    """Generate an open candidate set from multiple families.

    ``text_boxes``/``object_boxes`` are treated as candidate *geometry* only
    (e.g. OCR-detector output and annotation boxes). This function never receives
    the true label, shortcut identity, OCR text content, or correctness, so it
    cannot leak the answer into the candidate set.
    """

    pil = _as_pil(image)
    edges = _gray_edges(pil)
    proposals: list[RegionProposal] = []

    if enable_grid:
        proposals.extend(grid_patch_proposals(pil, edges, grid_scales))
    if enable_connected_components:
        proposals.extend(textness_proposals(pil, edges))
    if enable_high_contrast:
        proposals.extend(text_box_component_proposals(pil, edges))
    if enable_edge_dense:
        proposals.extend(horizontal_text_band_proposals(pil, edges))
        proposals.extend(corner_edge_watermark_proposals(pil, edges))
    if enable_ocr_family and text_boxes:
        proposals.extend(_boxes_to_proposals(pil, edges, text_boxes, "ocr_text_box", "ocr"))
    if enable_object_box_family and object_boxes:
        proposals.extend(_boxes_to_proposals(pil, edges, object_boxes, "object_box", "objbox"))
    if enable_sam and sam_proposer is not None:
        # Optional SAM/object proposer, guarded behind a flag and an injected
        # callable so the dependency is never required. ``sam_proposer`` returns
        # a list of bboxes given a PIL image.
        try:
            sam_boxes = list(sam_proposer(pil))
        except Exception:
            sam_boxes = []
        proposals.extend(_boxes_to_proposals(pil, edges, sam_boxes, "sam_proposal", "sam"))
    if enable_random_control:
        randoms = random_patch_proposals(pil, edges, seed=seed)
        proposals.extend(
            RegionProposal(
                candidate_id=r.candidate_id,
                bbox=r.bbox,
                proposal_type="random_patch_control",
                area_fraction=r.area_fraction,
                edge_density=r.edge_density,
                horizontalness_score=r.horizontalness_score,
                center_overlap_score=r.center_overlap_score,
                border_distance=r.border_distance,
                center_distance=r.center_distance,
                width=r.width,
                height=r.height,
                aspect_ratio=r.aspect_ratio,
            )
            for r in randoms
        )

    deduped = _dedupe(proposals)
    # Rank so that informative (edge-dense, non-central) candidates are kept when
    # truncating, but always retain at least some grid + random controls.
    ranked = sorted(
        deduped,
        key=lambda p: (p.edge_density + 0.5 * p.horizontalness_score - 0.25 * p.center_overlap_score),
        reverse=True,
    )
    if len(ranked) <= max_candidates:
        return ranked
    head = ranked[:max_candidates]
    head_ids = {(p.proposal_type, p.bbox) for p in head}
    # Guarantee a random control survives truncation for the matched-random baseline.
    if not any(p.proposal_type == "random_patch_control" for p in head):
        for p in ranked[max_candidates:]:
            if p.proposal_type == "random_patch_control":
                head[-1] = p
                break
    return head
