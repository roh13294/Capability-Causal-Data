from __future__ import annotations

"""Directional error-analysis metrics for the natural-text verified-failure eval.

These are **secondary diagnostic** metrics. They never replace the strict support
gate in
``causal_reliability.experiments.run_natural_text_verified_failure_eval``; they
only describe *how* and *why* the strict repair gate fails on the curated
human-verified natural-text failures.

The functions here are pure (no model, no I/O) so they can be unit-tested. They
operate on probability/logit vectors over a fixed ``allowed_clip_labels`` set,
plus the alias-aware label structure derived from ``visual_label_aliases`` and
``text_distractor_labels``.

Definitions
-----------
* **alias-aware target set**: the indices of ``allowed_clip_labels`` that are
  *not* text/logo distractors. ``visual_label_aliases`` is the candidate label
  set shown to CLIP; ``text_distractor_labels`` are the text/logo-driven
  distractors. Everything else is treated as a target-family (alias) label, so a
  prediction that lands on a synonym of the visual target still counts as an
  *alias-aware* recovery even when the strict exact-string match fails.
* **strict target**: the single ``visual_target_label`` index.
* **directional repair**: a method moved the prediction *toward* the visual
  target (target probability up) and/or *away* from the strongest text
  distractor (distractor probability down), even if the strict argmax did not
  flip to the exact target string.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


EPS = 1e-6

# Methods analysed per verified failure. ``original`` is the pre-repair baseline
# (after == before); every other method supplies a repaired probability vector.
DIRECTIONAL_METHODS = [
    "original",
    "oracle_text_box_repair",
    "cic_top1",
    "cic_top3",
    "matched_random",
    "largest_region",
    "ocr_text_box_proposal",
]


# --------------------------------------------------------------------------- #
# Label structure
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabelInfo:
    """Alias-aware label structure over a fixed ``allowed_clip_labels`` set."""

    allowed: tuple[str, ...]
    label: int  # strict target index
    alias_indices: frozenset[int]  # target-family (non-distractor) indices
    distractor_indices: frozenset[int]


def build_label_info(
    allowed: Sequence[str],
    target_label: str,
    distractor_labels: Iterable[str],
) -> LabelInfo:
    """Build the alias-aware label structure for one example.

    ``alias_indices`` are the non-distractor labels (the target plus any synonym
    that is not flagged as a text/logo distractor); ``distractor_indices`` are the
    text/logo distractor labels. The strict target is always part of the alias
    set even in the degenerate case where it was also listed as a distractor.
    """

    allowed = tuple(str(a) for a in allowed)
    label = allowed.index(target_label) if target_label in allowed else 0
    distractor_set = {str(d) for d in distractor_labels}
    distractor_indices = frozenset(i for i, name in enumerate(allowed) if name in distractor_set)
    alias_indices = frozenset(i for i in range(len(allowed)) if i not in distractor_indices) | {label}
    return LabelInfo(allowed=allowed, label=label, alias_indices=alias_indices, distractor_indices=distractor_indices)


# --------------------------------------------------------------------------- #
# Rank / membership primitives
# --------------------------------------------------------------------------- #
def label_rank(probs: Sequence[float], idx: int) -> int:
    """1-based rank of label ``idx`` (1 = highest probability).

    Ties are broken pessimistically for the queried label: the rank is
    ``1 + (number of labels with strictly greater probability)``.
    """

    arr = np.asarray(probs, dtype=np.float64)
    return int(1 + np.sum(arr > arr[idx]))


def best_rank(probs: Sequence[float], indices: Iterable[int]) -> int:
    """Best (minimum) rank achieved by any index in ``indices``."""

    idxs = list(indices)
    if not idxs:
        return int(len(np.asarray(probs)) + 1)
    return min(label_rank(probs, i) for i in idxs)


def in_top_k(probs: Sequence[float], idx: int, k: int) -> bool:
    return label_rank(probs, idx) <= int(k)


def alias_in_top_k(probs: Sequence[float], info: LabelInfo, k: int) -> bool:
    return best_rank(probs, info.alias_indices) <= int(k)


def _max_distractor_prob(probs: np.ndarray, info: LabelInfo) -> float:
    if not info.distractor_indices:
        return float("nan")
    return float(max(probs[i] for i in info.distractor_indices))


def _best_distractor_rank(probs: np.ndarray, info: LabelInfo) -> float:
    if not info.distractor_indices:
        return float("nan")
    return float(best_rank(probs, info.distractor_indices))


# --------------------------------------------------------------------------- #
# Per-(method, example) directional row
# --------------------------------------------------------------------------- #
def directional_metrics_row(
    before_probs: Sequence[float],
    after_probs: Sequence[float],
    info: LabelInfo,
    *,
    before_logits: Sequence[float] | None = None,
    after_logits: Sequence[float] | None = None,
    eps: float = EPS,
) -> dict[str, Any]:
    """Compute directional metrics comparing ``after`` (repaired) to ``before``.

    ``before`` is the original CLIP prediction; ``after`` is the method's repaired
    prediction. All probability vectors are over the same ``allowed_clip_labels``.
    Logits are optional; when absent a ``log(prob)`` proxy is reported.
    """

    b = np.asarray(before_probs, dtype=np.float64)
    a = np.asarray(after_probs, dtype=np.float64)
    bl = np.asarray(before_logits, dtype=np.float64) if before_logits is not None else np.log(np.clip(b, 1e-12, 1.0))
    al = np.asarray(after_logits, dtype=np.float64) if after_logits is not None else np.log(np.clip(a, 1e-12, 1.0))
    label = info.label

    tgt_prob_before = float(b[label])
    tgt_prob_after = float(a[label])
    tgt_rank_before = label_rank(b, label)
    tgt_rank_after = label_rank(a, label)
    tgt_logit_before = float(bl[label])
    tgt_logit_after = float(al[label])

    dist_prob_before = _max_distractor_prob(b, info)
    dist_prob_after = _max_distractor_prob(a, info)
    dist_rank_before = _best_distractor_rank(b, info)
    dist_rank_after = _best_distractor_rank(a, info)

    alias_rank_before = best_rank(b, info.alias_indices)
    alias_rank_after = best_rank(a, info.alias_indices)

    target_prob_gain = tgt_prob_after - tgt_prob_before
    target_rank_gain = tgt_rank_before - tgt_rank_after  # positive = improved (rank decreased)
    alias_rank_gain = alias_rank_before - alias_rank_after

    if np.isnan(dist_prob_before) or np.isnan(dist_prob_after):
        dist_prob_decrease = float("nan")
        moved_away_from_text = False
    else:
        dist_prob_decrease = float(dist_prob_before - dist_prob_after)
        moved_away_from_text = bool(dist_prob_after < dist_prob_before - eps)

    return {
        "target_prob_before": tgt_prob_before,
        "target_prob_after": tgt_prob_after,
        "target_rank_before": int(tgt_rank_before),
        "target_rank_after": int(tgt_rank_after),
        "target_logit_before": tgt_logit_before,
        "target_logit_after": tgt_logit_after,
        "text_distractor_prob_before": dist_prob_before,
        "text_distractor_prob_after": dist_prob_after,
        "text_distractor_rank_before": dist_rank_before,
        "text_distractor_rank_after": dist_rank_after,
        "target_prob_gain": float(target_prob_gain),
        "target_rank_gain": int(target_rank_gain),
        "text_distractor_prob_decrease": dist_prob_decrease,
        "moved_away_from_text": moved_away_from_text,
        "moved_toward_target": bool(tgt_prob_after > tgt_prob_before + eps),
        "target_top3_before": bool(in_top_k(b, label, 3)),
        "target_top3_after": bool(in_top_k(a, label, 3)),
        "target_top5_before": bool(in_top_k(b, label, 5)),
        "target_top5_after": bool(in_top_k(a, label, 5)),
        "target_entered_top3": bool(in_top_k(a, label, 3) and not in_top_k(b, label, 3)),
        "target_entered_top5": bool(in_top_k(a, label, 5) and not in_top_k(b, label, 5)),
        "strict_top1_after": bool(int(a.argmax()) == label),
        "strict_top1_before": bool(int(b.argmax()) == label),
        "alias_top1_after": bool(alias_rank_after == 1),
        "alias_top3_after": bool(alias_rank_after <= 3),
        "alias_top1_before": bool(alias_rank_before == 1),
        "alias_rank_before": int(alias_rank_before),
        "alias_rank_after": int(alias_rank_after),
        "alias_rank_improvement": int(alias_rank_gain),
    }


# --------------------------------------------------------------------------- #
# Aggregate directional metrics (per method)
# --------------------------------------------------------------------------- #
def _rate(values: Iterable[Any]) -> float:
    vals = [bool(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _median(values: Iterable[Any]) -> float:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.median(vals)) if vals else float("nan")


def aggregate_directional_metrics(rows: list[dict[str, Any]], eps: float = EPS) -> dict[str, Any]:
    """Aggregate one method's per-failure directional rows into a summary dict."""

    if not rows:
        nan = float("nan")
        return {
            "n": 0,
            "strict_top1_repair_accuracy": nan,
            "alias_top1_repair_accuracy": nan,
            "alias_top3_repair_accuracy": nan,
            "target_prob_improvement_rate": nan,
            "median_target_prob_gain": nan,
            "target_rank_improvement_rate": nan,
            "median_target_rank_gain": nan,
            "text_distractor_prob_decrease_rate": nan,
            "median_text_distractor_prob_decrease": nan,
            "top3_target_recovery_rate": nan,
            "top5_target_recovery_rate": nan,
            "moved_away_from_text_rate": nan,
            "moved_toward_target_rate": nan,
        }
    return {
        "n": len(rows),
        "strict_top1_repair_accuracy": _rate(r["strict_top1_after"] for r in rows),
        "alias_top1_repair_accuracy": _rate(r["alias_top1_after"] for r in rows),
        "alias_top3_repair_accuracy": _rate(r["alias_top3_after"] for r in rows),
        "target_prob_improvement_rate": _rate(r["moved_toward_target"] for r in rows),
        "median_target_prob_gain": _median(r["target_prob_gain"] for r in rows),
        "target_rank_improvement_rate": _rate((r["target_rank_gain"] > 0) for r in rows),
        "median_target_rank_gain": _median(r["target_rank_gain"] for r in rows),
        "text_distractor_prob_decrease_rate": _rate(
            (r["text_distractor_prob_decrease"] > eps)
            for r in rows
            if not np.isnan(r["text_distractor_prob_decrease"])
        ),
        "median_text_distractor_prob_decrease": _median(r["text_distractor_prob_decrease"] for r in rows),
        "top3_target_recovery_rate": _rate(r["target_top3_after"] for r in rows),
        "top5_target_recovery_rate": _rate(r["target_top5_after"] for r in rows),
        "moved_away_from_text_rate": _rate(r["moved_away_from_text"] for r in rows),
        "moved_toward_target_rate": _rate(r["moved_toward_target"] for r in rows),
    }


# --------------------------------------------------------------------------- #
# Proposal-selection geometry diagnostics
# --------------------------------------------------------------------------- #
def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = [int(v) for v in a]
    bx0, by0, bx1, by1 = [int(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    return float(inter / max(1, area_a + area_b - inter))


def _coverage_of_selected(selected, other) -> float:
    """Fraction of the *selected* box area that lies inside ``other``."""

    sx0, sy0, sx1, sy1 = [int(v) for v in selected]
    ox0, oy0, ox1, oy1 = [int(v) for v in other]
    ix0, iy0 = max(sx0, ox0), max(sy0, oy0)
    ix1, iy1 = min(sx1, ox1), min(sy1, oy1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    sel_area = max(1, (sx1 - sx0) * (sy1 - sy0))
    return float(inter / sel_area)


def _center(box) -> tuple[float, float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _min_center_distance(box, boxes) -> float:
    if not boxes:
        return float("inf")
    cx, cy = _center(box)
    return min(float(np.hypot(cx - c[0], cy - c[1])) for c in (_center(b) for b in boxes))


def selection_geometry(selected_box, text_boxes, object_boxes, *, overlap_threshold: float = 0.1) -> dict[str, Any]:
    """Geometry of a CIC-selected region relative to text and object boxes."""

    text_iou = max((_iou(selected_box, b) for b in text_boxes), default=0.0)
    object_iou = max((_iou(selected_box, b) for b in object_boxes), default=0.0)
    text_cov = max((_coverage_of_selected(selected_box, b) for b in text_boxes), default=0.0)
    object_cov = max((_coverage_of_selected(selected_box, b) for b in object_boxes), default=0.0)
    overlaps_text = bool(text_iou >= overlap_threshold)
    overlaps_object = bool(object_iou >= overlap_threshold)
    dist_text = _min_center_distance(selected_box, text_boxes)
    dist_object = _min_center_distance(selected_box, object_boxes)
    if np.isinf(dist_text) and np.isinf(dist_object):
        closer_to = "none"
    elif dist_text <= dist_object:
        closer_to = "text"
    else:
        closer_to = "object"
    return {
        "text_iou": float(text_iou),
        "object_iou": float(object_iou),
        "text_coverage": float(text_cov),
        "object_coverage": float(object_cov),
        "overlaps_text_box": overlaps_text,
        "overlaps_object_box": overlaps_object,
        "overlaps_both": bool(overlaps_text and overlaps_object),
        "closer_to": closer_to,
        "text_overlap_bucket": text_overlap_bucket(text_cov, text_iou),
    }


def text_overlap_bucket(coverage: float, iou: float) -> str:
    """Bucket the selected/text overlap for conditional repair-rate reporting."""

    if coverage <= 0.0 and iou <= 0.0:
        return "no_overlap"
    if coverage >= 0.5:
        return "coverage_ge_0.5"
    if coverage >= 0.3 or iou >= 0.3:
        return "iou_or_coverage_ge_0.3"
    return "partial_overlap"


# --------------------------------------------------------------------------- #
# Example categorization
# --------------------------------------------------------------------------- #
CATEGORIES = [
    "cic_strict_repaired",
    "oracle_strict_repaired",
    "label_alias_ambiguity",
    "cic_directional_only",
    "oracle_directional_only",
    "cic_selected_text_no_repair",
    "cic_selected_object_damaged",
    "hard_no_clear_repair",
]


def categorize_failure(
    *,
    cic_row: dict[str, Any],
    oracle_row: dict[str, Any] | None,
    geometry: dict[str, Any] | None,
) -> tuple[str, dict[str, bool]]:
    """Assign one primary category (priority-ordered) plus per-category flags.

    ``cic_row``/``oracle_row`` are directional rows from
    :func:`directional_metrics_row`; ``geometry`` comes from
    :func:`selection_geometry`.
    """

    oracle_strict = bool(oracle_row and oracle_row["strict_top1_after"])
    oracle_dir = bool(oracle_row and oracle_row["moved_toward_target"])
    oracle_alias = bool(oracle_row and oracle_row["alias_top1_after"])
    cic_strict = bool(cic_row["strict_top1_after"])
    cic_dir = bool(cic_row["moved_toward_target"])
    cic_alias = bool(cic_row["alias_top1_after"])
    sel_text = bool(geometry and geometry["overlaps_text_box"])
    sel_object = bool(geometry and geometry["overlaps_object_box"])
    cic_damaged = bool(cic_row["target_prob_after"] < cic_row["target_prob_before"] - EPS)

    # Alias ambiguity: a perfect oracle text removal (or CIC) recovers an alias
    # at top-1 but the strict exact-string target never reaches top-1 — i.e. the
    # exact-label metric is the limiting factor, not the intervention.
    alias_ambiguity = bool(((oracle_alias and not oracle_strict) or (cic_alias and not cic_strict)))

    flags = {
        "cic_strict_repaired": cic_strict,
        "oracle_strict_repaired": oracle_strict,
        "label_alias_ambiguity": alias_ambiguity,
        "cic_directional_only": bool(cic_dir and not cic_strict),
        "oracle_directional_only": bool(oracle_dir and not oracle_strict),
        "cic_selected_text_no_repair": bool(sel_text and not cic_strict and not cic_dir),
        "cic_selected_object_damaged": bool(sel_object and not sel_text and cic_damaged),
        "hard_no_clear_repair": bool(not oracle_strict and not oracle_dir and not cic_strict and not cic_dir),
    }
    primary = "hard_no_clear_repair"
    for name in CATEGORIES:
        if flags[name]:
            primary = name
            break
    return primary, flags


# --------------------------------------------------------------------------- #
# Directional-evidence flag (diagnostic only; NOT a headline claim)
# --------------------------------------------------------------------------- #
DEFAULT_MIN_VERIFIED_FAILURES = 20
DEFAULT_MIN_ORACLE_PROB_IMPROVEMENT = 0.80
DEFAULT_MIN_CIC_OVER_RANDOM_PROB_GAP = 0.10
DEFAULT_MIN_CIC_TEXT_OVERLAP_RATE = 0.60


def evaluate_directional_evidence(
    *,
    n_verified_failures: int,
    oracle_target_prob_improvement_rate: float,
    cic_target_prob_improvement_rate: float,
    random_target_prob_improvement_rate: float,
    cic_selected_text_overlap_rate: float,
    no_oracle_leakage: bool,
    min_verified_failures: int = DEFAULT_MIN_VERIFIED_FAILURES,
    min_oracle_prob_improvement: float = DEFAULT_MIN_ORACLE_PROB_IMPROVEMENT,
    min_cic_over_random_prob_gap: float = DEFAULT_MIN_CIC_OVER_RANDOM_PROB_GAP,
    min_cic_text_overlap_rate: float = DEFAULT_MIN_CIC_TEXT_OVERLAP_RATE,
) -> tuple[bool, list[str]]:
    """Decide the diagnostic ``natural_text_directional_evidence`` flag.

    This is **not** a support gate and never licenses a positive natural-text
    claim. It flags whether the verified failures show coherent *directional*
    movement toward the visual target under text-box removal.
    """

    reasons: list[str] = []
    if int(n_verified_failures) < int(min_verified_failures):
        reasons.append(
            f"verified failures {int(n_verified_failures)} < minimum {int(min_verified_failures)}"
        )
    if not np.isfinite(oracle_target_prob_improvement_rate) or float(oracle_target_prob_improvement_rate) < float(min_oracle_prob_improvement):
        reasons.append(
            f"oracle target-probability improvement rate {float(oracle_target_prob_improvement_rate):.3f} "
            f"< required {float(min_oracle_prob_improvement):.2f}"
        )
    gap = float(cic_target_prob_improvement_rate) - float(random_target_prob_improvement_rate)
    if not np.isfinite(gap) or gap < float(min_cic_over_random_prob_gap):
        reasons.append(
            f"CIC target-probability improvement does not beat matched random by >= "
            f"{float(min_cic_over_random_prob_gap):.2f} (gap={gap:.3f})"
        )
    if not np.isfinite(cic_selected_text_overlap_rate) or float(cic_selected_text_overlap_rate) < float(min_cic_text_overlap_rate):
        reasons.append(
            f"CIC selected-region text overlap rate {float(cic_selected_text_overlap_rate):.3f} "
            f"< required {float(min_cic_text_overlap_rate):.2f}"
        )
    if not bool(no_oracle_leakage):
        reasons.append("oracle leakage check failed: scoring/proposal rule exposes forbidden parameters")
    return (len(reasons) == 0), reasons
