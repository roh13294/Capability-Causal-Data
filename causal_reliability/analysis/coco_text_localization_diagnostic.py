from __future__ import annotations

"""Pure (model-free) helpers for the COCO-Text proposal-localization diagnostic.

These functions answer the *geometry* and *ranking* half of the question "why did
the full COCO-Text CIC gates fail only on selected text-box overlap": given the
already-scored open proposals (``coco_text_full_proposal_diagnostics.csv``) and
the evaluation text/object boxes (rescaled to the model input resolution), they
compute proposal recall, text-overlap@k, the CIC rank of the best text-overlapping
proposal, area-normalised score re-orderings, and top-k union construction.

Everything here is deliberately model-free and side-effect-free so it can be unit
tested without loading CLIP. The model-dependent half (actual repair outcomes for
the best text-overlapping proposal, top-k unions, area-normalised top-1 and
inference-time text-dilated proposals) lives in the experiment runner
``run_coco_text_cic_localization_diagnostic``.

IoU semantics match the existing pipeline (``run_natural_text_open_proposal_cic._iou``):
overlap is integer-pixel IoU and the canonical overlap threshold is 0.1.
"""

import math
from dataclasses import dataclass

BBox = tuple[int, int, int, int]

# Canonical overlap threshold used throughout the existing pipeline.
OVERLAP_IOU = 0.1
IOU_THRESHOLDS = (0.1, 0.3, 0.5)
COVERAGE_THRESHOLDS = (0.30, 0.50, 0.80)
TOPK_VALUES = (1, 3, 5, 10)

# Area-normalisation behaviour shared by the experiment runner and the tests.
AREA_FLOOR = 0.02
OBJECT_PENALTY = 0.8
TEXT_REWARD = 2.0

# Modes that consume ground-truth text geometry in the *scoring* rule (not merely
# for evaluation) are leakage / oracle-only and must never be deployed.
LEAKAGE_MODES = frozenset({"reward_text_oracle"})


def _as_box(box) -> BBox:
    x0, y0, x1, y1 = (int(v) for v in box)
    return x0, y0, x1, y1


def iou(a, b) -> float:
    """Integer-pixel IoU, matching the pipeline's ``_iou``."""

    ax0, ay0, ax1, ay1 = _as_box(a)
    bx0, by0, bx1, by1 = _as_box(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    return float(inter / max(1, area_a + area_b - inter))


def coverage_of_box(prop_box, target_box) -> float:
    """Fraction of ``target_box`` covered by ``prop_box`` (intersection / target area)."""

    px0, py0, px1, py1 = _as_box(prop_box)
    tx0, ty0, tx1, ty1 = _as_box(target_box)
    ix0, iy0 = max(px0, tx0), max(py0, ty0)
    ix1, iy1 = min(px1, tx1), min(py1, ty1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    target_area = max(0, tx1 - tx0) * max(0, ty1 - ty0)
    return float(inter / max(1, target_area))


def best_iou_against(prop_box, boxes) -> float:
    """Max IoU of a proposal against any box in ``boxes`` (0.0 if none)."""

    return max((iou(prop_box, b) for b in boxes), default=0.0)


def best_coverage_against(prop_box, boxes) -> float:
    """Max coverage of any box in ``boxes`` by ``prop_box`` (0.0 if none)."""

    return max((coverage_of_box(prop_box, b) for b in boxes), default=0.0)


def overlaps_any(prop_box, boxes, threshold: float = OVERLAP_IOU) -> bool:
    return best_iou_against(prop_box, boxes) >= threshold if boxes else False


def proposal_recall_metrics(
    prop_boxes: list[BBox],
    text_boxes: list[BBox],
    object_boxes: list[BBox],
    *,
    iou_thresholds: tuple[float, ...] = IOU_THRESHOLDS,
    coverage_thresholds: tuple[float, ...] = COVERAGE_THRESHOLDS,
    overlap_iou: float = OVERLAP_IOU,
) -> dict[str, float]:
    """Per-example proposal-recall geometry over an open candidate set.

    Answers task-1: do *any* available proposals reach a text box at increasing
    IoU / coverage, what is the best available text IoU / coverage and object IoU,
    and how many proposals overlap text vs object boxes.
    """

    # Best (max over proposals) IoU / coverage against the text boxes.
    best_text_iou = max((best_iou_against(p, text_boxes) for p in prop_boxes), default=0.0)
    best_text_cov = max((best_coverage_against(p, text_boxes) for p in prop_boxes), default=0.0)
    best_object_iou = max((best_iou_against(p, object_boxes) for p in prop_boxes), default=0.0)

    out: dict[str, float] = {
        "n_proposals": int(len(prop_boxes)),
        "n_text_boxes": int(len(text_boxes)),
        "n_object_boxes": int(len(object_boxes)),
        "best_text_iou": float(best_text_iou),
        "best_text_coverage": float(best_text_cov),
        "best_object_iou": float(best_object_iou),
        "n_text_overlapping_proposals": int(sum(1 for p in prop_boxes if overlaps_any(p, text_boxes, overlap_iou))),
        "n_object_overlapping_proposals": int(sum(1 for p in prop_boxes if overlaps_any(p, object_boxes, overlap_iou))),
    }
    for thr in iou_thresholds:
        out[f"has_text_iou_{_tag(thr)}"] = bool(best_text_iou >= thr)
    for thr in coverage_thresholds:
        out[f"has_text_coverage_{_tag(thr)}"] = bool(best_text_cov >= thr)
    return out


def _tag(thr: float) -> str:
    """Stable column suffix for a threshold (0.1 -> '010', 0.3 -> '030')."""

    return f"{int(round(thr * 100)):03d}"


def text_overlap_at_k(
    overlap_flags: list[bool],
    ks: tuple[int, ...] = TOPK_VALUES,
) -> dict[int, bool]:
    """text-overlap@k: does any of the top-k CIC-ranked proposals overlap text?

    ``overlap_flags`` is ordered by descending CIC score (rank 1 first).
    """

    return {int(k): bool(any(overlap_flags[: int(k)])) for k in ks}


def rank_of_first_true(flags: list[bool]) -> int | None:
    """1-indexed rank of the first True flag, or None if none.

    Used for the CIC rank of the best (highest-CIC-score) text-overlapping or
    object-overlapping proposal, given ``flags`` ordered by descending CIC score.
    """

    for i, f in enumerate(flags, start=1):
        if f:
            return i
    return None


def adjusted_score(
    score: float,
    area_fraction: float,
    *,
    mode: str,
    overlaps_object: bool = False,
    text_iou: float = 0.0,
    area_floor: float = AREA_FLOOR,
    object_penalty: float = OBJECT_PENALTY,
    text_reward: float = TEXT_REWARD,
) -> float:
    """Area-normalised / re-weighted CIC score under a diagnostic ``mode``.

    Modes:
    * ``original``           - unchanged CIC score.
    * ``div_sqrt_area``      - score / sqrt(area fraction): favours smaller regions.
    * ``div_area_clip``      - score / clip(area, area_floor, 1): stronger small-region bias.
    * ``penalize_object``    - downweight proposals overlapping an object box.
    * ``reward_text_oracle`` - reward proposals by ground-truth text IoU. LEAKAGE /
      oracle-only: uses evaluation geometry inside the scoring rule, never deployable.
    """

    s = float(score)
    a = float(area_fraction)
    if mode == "original":
        return s
    if mode == "div_sqrt_area":
        return s / math.sqrt(max(a, 1e-9))
    if mode == "div_area_clip":
        return s / min(max(a, area_floor), 1.0)
    if mode == "penalize_object":
        return s * (1.0 - object_penalty * (1.0 if overlaps_object else 0.0))
    if mode == "reward_text_oracle":
        return s * (1.0 + text_reward * max(0.0, float(text_iou)))
    raise ValueError(f"unknown area-normalisation mode: {mode!r}")


@dataclass(frozen=True)
class ScoredProposal:
    candidate_id: str
    score: float
    area_fraction: float
    overlaps_text: bool
    overlaps_object: bool
    text_iou: float = 0.0


def reorder(items: list[ScoredProposal], mode: str, **kw) -> list[ScoredProposal]:
    """Return ``items`` re-ordered by descending adjusted score under ``mode``.

    Sort is stable so ties keep the incoming (original CIC) order.
    """

    keyed = list(enumerate(items))
    keyed.sort(
        key=lambda pair: -adjusted_score(
            pair[1].score,
            pair[1].area_fraction,
            mode=mode,
            overlaps_object=pair[1].overlaps_object,
            text_iou=pair[1].text_iou,
            **kw,
        )
    )
    return [it for _, it in keyed]


def topk_union_ids(ranked_ids: list[str], k: int) -> list[str]:
    """Construct the top-k union: the first ``k`` candidate ids of a CIC ranking.

    De-duplicates while preserving order so a union never repeats a region.
    """

    out: list[str] = []
    for cid in ranked_ids:
        if cid not in out:
            out.append(cid)
        if len(out) >= int(k):
            break
    return out


def is_leakage_mode(mode: str) -> bool:
    return mode in LEAKAGE_MODES
