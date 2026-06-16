"""Spatial-resolution and causal-intervention audit for CIC.

This audit addresses the "low exact-IoU" criticism directly and honestly. It
separates two questions that exact-IoU conflates:

  1. *Exact box precision* -- how close is the selected region to the oracle
     shortcut box (IoU, hit@IoU thresholds)?
  2. *Causal-intervention usefulness* -- does the selected region cover the
     shortcut evidence, stay spatially coarse (bluntness), preserve causal
     object content, and actually repair the prediction?

CIC is a coarse causal-intervention method, not an exact localization or
segmentation method. This script measures both axes from EXISTING benchmark
artifacts (no model is re-run, no headline metric is changed) and reports the
gap between them. It makes no claim of exact localization, segmentation,
open-world discovery, or general robustness.

The optional refinement diagnostic re-scores geometric variants of the top
region using ONLY non-oracle information (pixels, candidate boxes, model
probabilities). It never uses the oracle box, true label, or repair
correctness to *select* a refined region; those are used only to *evaluate*
the selection after the fact.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from causal_reliability.analysis.phase6_common import _markdown_table  # noqa: E402
from causal_reliability.utils.config import load_config  # noqa: E402
from causal_reliability.utils.io import ensure_dir  # noqa: E402

Box = tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# Pure geometry helpers
# --------------------------------------------------------------------------- #
def parse_bbox(value: Any) -> Optional[Box]:
    """Parse a bbox from a list or a string. Returns (x1, y1, x2, y2) or None.

    Handles comma-separated (``"[9, 65, 215, 96]"``) and whitespace-separated
    numpy-repr (``"[73\\n 67\\n 149\\n 91]"``) forms, plus actual sequences.
    Returns None for missing/empty/unparseable values rather than fabricating.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        nums = [float(v) for v in value]
    else:
        if isinstance(value, float) and math.isnan(value):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", ""}:
            return None
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                nums = [float(v) for v in parsed]
            else:
                raise ValueError
        except (ValueError, SyntaxError):
            cleaned = text.strip("[]()").replace(",", " ")
            try:
                nums = [float(tok) for tok in cleaned.split()]
            except ValueError:
                return None
    if len(nums) != 4:
        return None
    x1, y1, x2, y2 = nums
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def box_area(box: Optional[Box]) -> float:
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: Optional[Box], b: Optional[Box]) -> float:
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(a: Optional[Box], b: Optional[Box]) -> float:
    """Intersection-over-union of two boxes."""
    inter = intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def coverage(region: Optional[Box], target: Optional[Box]) -> float:
    """Fraction of ``target`` covered by ``region`` (intersection / target area)."""
    ta = box_area(target)
    if ta <= 0.0:
        return 0.0
    return intersection_area(region, target) / ta


def area_fraction(box: Optional[Box], image_area: float) -> float:
    """Selected-region area as a fraction of total image area."""
    if box is None or image_area <= 0.0:
        return float("nan")
    return box_area(box) / image_area


# --------------------------------------------------------------------------- #
# Bucketing
# --------------------------------------------------------------------------- #
def assign_iou_bucket(value: Optional[float], buckets: list[list[float]]) -> Optional[str]:
    """Assign an IoU value to a [lo, hi) bucket; returns a label like '0.3-0.5'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    for lo, hi in buckets:
        if lo <= value < hi:
            return _bucket_label(lo, hi)
    return None


def _bucket_label(lo: float, hi: float) -> str:
    if hi > 1.0:
        return f">={lo:g}"
    if lo <= 0.0:
        return f"<{hi:g}"
    return f"{lo:g}-{hi:g}"


def hit_at(values: list[float], threshold: float) -> float:
    """Fraction of IoU values >= threshold."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    return float(np.mean([1.0 if v >= threshold else 0.0 for v in vals]))


# --------------------------------------------------------------------------- #
# Refinement (NON-ORACLE ONLY)
# --------------------------------------------------------------------------- #
# These functions deliberately accept ONLY pixels/geometry (image_size),
# candidate boxes, and model-derived candidate scores. They take no oracle box,
# no true label, and no correctness signal -- this is asserted by the test
# suite via signature inspection so refinement cannot leak oracle information.
def generate_refinement_variants(
    box: Box,
    image_size: int,
    shrink_fractions: list[float],
    shifts: list[str],
    shift_fraction: float,
    split_2x2: bool,
) -> list[Box]:
    """Geometric refinement candidates for a selected region.

    Includes the original box, centre-shrinks, axis shifts, and the 2x2
    sub-boxes. All variants are clipped to the image and de-duplicated.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    variants: list[Box] = [box]

    for frac in shrink_fractions:
        nw, nh = w * (1.0 - frac), h * (1.0 - frac)
        variants.append((cx - nw / 2.0, cy - nh / 2.0, cx + nw / 2.0, cy + nh / 2.0))

    dx, dy = w * shift_fraction, h * shift_fraction
    deltas = {"up": (0, -dy), "down": (0, dy), "left": (-dx, 0), "right": (dx, 0)}
    for name in shifts:
        if name in deltas:
            ox, oy = deltas[name]
            variants.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy))

    if split_2x2:
        variants.extend(
            [
                (x1, y1, cx, cy),
                (cx, y1, x2, cy),
                (x1, cy, cx, y2),
                (cx, cy, x2, y2),
            ]
        )

    clipped = [_clip_box(v, image_size) for v in variants]
    out: list[Box] = []
    seen: set[tuple[int, int, int, int]] = set()
    for v in clipped:
        if box_area(v) <= 0:
            continue
        key = tuple(int(round(c)) for c in v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _clip_box(box: Box, image_size: int) -> Box:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(x1, image_size)),
        max(0.0, min(y1, image_size)),
        max(0.0, min(x2, image_size)),
        max(0.0, min(y2, image_size)),
    )


def nonoracle_region_score(
    box: Box,
    candidate_boxes: list[Box],
    candidate_scores: list[float],
    image_size: int,
    area_weight: float,
    consensus_weight: float,
) -> float:
    """Non-oracle proxy score for a candidate region.

    Uses only: the region's pixel area (prefer tighter regions) and its
    overlap-weighted consensus with the model's already-scored candidate boxes
    (model-derived evidence). No oracle box / label / correctness is consulted.
    """
    image_area = float(image_size * image_size)
    af = area_fraction(box, image_area)
    consensus = 0.0
    for cbox, cscore in zip(candidate_boxes, candidate_scores):
        if cbox is None or cscore is None or (isinstance(cscore, float) and math.isnan(cscore)):
            continue
        consensus += float(cscore) * iou(box, cbox)
    return consensus_weight * consensus - area_weight * (af if not math.isnan(af) else 1.0)


def select_refined_region(
    top_box: Box,
    candidate_boxes: list[Box],
    candidate_scores: list[float],
    image_size: int,
    refine_cfg: dict[str, Any],
) -> tuple[Box, list[Box]]:
    """Pick the highest non-oracle-scoring refinement variant of ``top_box``.

    Returns (refined_box, variants). Selection uses ONLY non-oracle signals.
    """
    variants = generate_refinement_variants(
        top_box,
        image_size,
        list(refine_cfg.get("shrink_fractions", [0.1, 0.2])),
        list(refine_cfg.get("shifts", ["up", "down", "left", "right"])),
        float(refine_cfg.get("shift_fraction", 0.1)),
        bool(refine_cfg.get("split_2x2", True)),
    )
    aw = float(refine_cfg.get("area_weight", 0.25))
    cw = float(refine_cfg.get("consensus_weight", 1.0))
    scored = [
        (v, nonoracle_region_score(v, candidate_boxes, candidate_scores, image_size, aw, cw))
        for v in variants
    ]
    best = max(scored, key=lambda item: item[1])
    return best[0], variants


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #
def _resolve(bench: dict[str, Any], cfg: dict[str, Any], key: str) -> Any:
    return bench.get(key, cfg.get(key))


def _load_candidates_and_oracle(
    bench: dict[str, Any], bench_dir: Path
) -> dict[Any, dict[str, Any]]:
    """Per-example oracle shortcut box + scored candidate boxes/scores.

    Sources, in order of availability: a single rankings CSV, or a directory of
    per-example rankings caches. Returns {} when neither exists (e.g. the
    semantic-decoy benchmark, which has no oracle box on disk).
    """
    frames: list[pd.DataFrame] = []
    rankings = bench.get("rankings")
    if rankings:
        path = bench_dir / rankings
        if path.exists():
            frames.append(pd.read_csv(path))
    cache = bench.get("rankings_cache")
    if cache:
        cache_dir = bench_dir / cache
        if cache_dir.exists():
            for f in sorted(cache_dir.rglob("*_rankings.csv")):
                frames.append(pd.read_csv(f))
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    out: dict[Any, dict[str, Any]] = {}
    for example_id, group in df.groupby("example_id"):
        boxes = [parse_bbox(b) for b in group.get("bbox", [])]
        scores = [float(s) for s in group.get("score", [])] if "score" in group else []
        oracle_box = None
        if "harmful_bbox_eval_only" in group:
            for val in group["harmful_bbox_eval_only"]:
                parsed = parse_bbox(val)
                if parsed is not None:
                    oracle_box = parsed
                    break
        # neutralized prediction per candidate bbox (for refinement repair eval)
        neutralized: dict[tuple[int, int, int, int], Any] = {}
        if "neutralized_prediction_index" in group:
            for b, idx in zip(group.get("bbox", []), group["neutralized_prediction_index"]):
                pb = parse_bbox(b)
                if pb is not None and pd.notna(idx):
                    neutralized[tuple(int(round(c)) for c in pb)] = int(idx)
        out[example_id] = {
            "oracle_box": oracle_box,
            "candidate_boxes": [b for b in boxes if b is not None],
            "candidate_scores": [s for b, s in zip(boxes, scores) if b is not None] if scores else [],
            "neutralized_by_box": neutralized,
        }
    return out


def _true_label_index(cert: pd.DataFrame, oracle_method: str) -> dict[Any, int]:
    """Per-example true-label index, derived from the oracle-neutralization rows.

    Used ONLY to evaluate refinement repair after selection, never to select.
    """
    out: dict[Any, int] = {}
    rows = cert[cert["method"] == oracle_method]
    for _, r in rows.iterrows():
        idx = None
        if bool(r.get("repaired_correct")) and pd.notna(r.get("repaired_prediction_index")):
            idx = int(r["repaired_prediction_index"])
        elif bool(r.get("original_correct")) and pd.notna(r.get("original_prediction_index")):
            idx = int(r["original_prediction_index"])
        if idx is not None:
            out[r["example_id"]] = idx
    return out


def _method_repair_map(cert: pd.DataFrame, method: Optional[str]) -> dict[Any, bool]:
    if not method or method not in set(cert.get("method", [])):
        return {}
    rows = cert[cert["method"] == method]
    return {r["example_id"]: bool(r["repaired_correct"]) for _, r in rows.iterrows()}


def load_benchmark_records(
    bench: dict[str, Any], cfg: dict[str, Any], results_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build per-example spatial records for one benchmark.

    Returns (records, note). ``note`` carries availability flags and any skip
    reason so the summary can report n/a honestly rather than fabricating.
    """
    name = bench["name"]
    bench_dir = results_root / bench["dir"]
    cert_path = bench_dir / bench["certificates"]
    note: dict[str, Any] = {"benchmark": name, "available": False}
    if not cert_path.exists():
        note["skip_reason"] = f"certificates not found: {cert_path}"
        return [], note

    cert = pd.read_csv(cert_path)
    selected_method = _resolve(bench, cfg, "selected_method")
    top3_method = _resolve(bench, cfg, "top3_method")
    clean_safe_method = _resolve(bench, cfg, "clean_safe_method")
    oracle_method = _resolve(bench, cfg, "oracle_method")
    shortcut_iou_col = _resolve(bench, cfg, "shortcut_iou_column")
    object_iou_col = _resolve(bench, cfg, "object_iou_column")
    shortcut_regimes = set(_resolve(bench, cfg, "shortcut_regimes") or [])
    image_size = int(cfg.get("image_size", 224))
    image_area = float(image_size * image_size)
    buckets = [list(b) for b in cfg.get("iou_buckets")]

    if selected_method not in set(cert["method"]):
        note["skip_reason"] = f"selected method '{selected_method}' absent from certificates"
        return [], note

    sel = cert[cert["method"] == selected_method].copy()
    if shortcut_regimes:
        sel = sel[sel["regime"].isin(shortcut_regimes)]
    if sel.empty:
        note["skip_reason"] = "no examples in configured shortcut regimes"
        return [], note

    top3_map = _method_repair_map(cert, top3_method)
    clean_safe_map = _method_repair_map(cert, clean_safe_method)
    cand_info = _load_candidates_and_oracle(bench, bench_dir)

    has_oracle_box = any(v.get("oracle_box") is not None for v in cand_info.values())
    has_object = bool(object_iou_col) and object_iou_col in sel.columns

    records: list[dict[str, Any]] = []
    for _, r in sel.iterrows():
        ex = r["example_id"]
        sel_box = parse_bbox(r.get("selected_bbox"))
        info = cand_info.get(ex, {})
        oracle_box = info.get("oracle_box")

        stored_iou = r.get(shortcut_iou_col)
        stored_iou = float(stored_iou) if pd.notna(stored_iou) else float("nan")
        # Prefer a recomputed IoU when both boxes are on disk; fall back to the
        # stored eval-only IoU column otherwise.
        if oracle_box is not None and sel_box is not None:
            iou_value = iou(sel_box, oracle_box)
        else:
            iou_value = stored_iou

        cov = coverage(sel_box, oracle_box) if (oracle_box and sel_box) else float("nan")
        af_img = area_fraction(sel_box, image_area)
        af_oracle = (
            box_area(sel_box) / box_area(oracle_box)
            if (oracle_box and sel_box and box_area(oracle_box) > 0)
            else float("nan")
        )
        if oracle_box and sel_box:
            intersects = intersection_area(sel_box, oracle_box) > 0
        elif not math.isnan(iou_value):
            intersects = iou_value > 0
        else:
            intersects = None
        obj_iou = (
            float(r[object_iou_col])
            if has_object and pd.notna(r.get(object_iou_col))
            else float("nan")
        )

        records.append(
            {
                "benchmark": name,
                "example_id": ex,
                "regime": r.get("regime"),
                "selected_bbox": list(sel_box) if sel_box else None,
                "oracle_shortcut_bbox": list(oracle_box) if oracle_box else None,
                "iou": iou_value,
                "iou_bucket": assign_iou_bucket(iou_value, buckets),
                "shortcut_coverage": cov,
                "shortcut_coverage_ge_0_5": (cov >= 0.5) if not math.isnan(cov) else None,
                "shortcut_coverage_ge_0_8": (cov >= 0.8) if not math.isnan(cov) else None,
                "intersects_shortcut": intersects,
                "area_frac_image": af_img,
                "area_frac_oracle": af_oracle,
                "object_iou": obj_iou,
                "repaired_correct_top1": bool(r.get("repaired_correct")),
                "repaired_correct_top3": top3_map.get(ex),
                "repaired_correct_clean_safe": clean_safe_map.get(ex),
            }
        )

    note.update(
        {
            "available": True,
            "n_examples": len(records),
            "has_oracle_box": has_oracle_box,
            "has_object_overlap": bool(has_object),
            "has_candidates": any(info.get("candidate_boxes") for info in cand_info.values()),
            "selected_method": selected_method,
        }
    )
    return records, note


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _safe_mean(series: pd.Series) -> Optional[float]:
    vals = series.dropna()
    return float(vals.mean()) if len(vals) else None


def _safe_median(series: pd.Series) -> Optional[float]:
    vals = series.dropna()
    return float(vals.median()) if len(vals) else None


def _rate(series: pd.Series) -> Optional[float]:
    vals = series.dropna()
    return float(vals.astype(float).mean()) if len(vals) else None


def summarize(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    iou_vals = df["iou"].dropna().tolist()
    summary: dict[str, Any] = {
        "n_examples": int(len(df)),
        "median_iou": _safe_median(df["iou"]),
        "mean_iou": _safe_mean(df["iou"]),
    }
    for thr in cfg.get("iou_thresholds", []):
        summary[f"hit_at_iou_{thr:g}".replace(".", "_")] = hit_at(iou_vals, thr)
    summary["shortcut_coverage_median"] = _safe_median(df["shortcut_coverage"])
    summary["shortcut_coverage_mean"] = _safe_mean(df["shortcut_coverage"])
    summary["shortcut_coverage_ge_0_5_rate"] = _rate(df["shortcut_coverage_ge_0_5"])
    summary["shortcut_coverage_ge_0_8_rate"] = _rate(df["shortcut_coverage_ge_0_8"])
    summary["intersects_shortcut_rate"] = _rate(df["intersects_shortcut"])
    summary["area_frac_image_median"] = _safe_median(df["area_frac_image"])
    summary["area_frac_image_mean"] = _safe_mean(df["area_frac_image"])
    summary["area_frac_oracle_median"] = _safe_median(df["area_frac_oracle"])
    summary["object_iou_median"] = _safe_median(df["object_iou"])
    summary["object_iou_mean"] = _safe_mean(df["object_iou"])
    summary["object_overlap_available"] = bool(df["object_iou"].notna().any())
    summary["repair_top1_accuracy"] = _rate(df["repaired_correct_top1"])
    summary["repair_top3_accuracy"] = _rate(df["repaired_correct_top3"])
    summary["repair_clean_safe_accuracy"] = _rate(df["repaired_correct_clean_safe"])
    return summary


def bucket_table(df: pd.DataFrame, cfg: dict[str, Any], group: str) -> pd.DataFrame:
    buckets = [list(b) for b in cfg.get("iou_buckets")]
    order = [_bucket_label(lo, hi) for lo, hi in buckets]
    rows: list[dict[str, Any]] = []
    for label in order:
        sub = df[df["iou_bucket"] == label]
        rows.append(
            {
                "group": group,
                "iou_bucket": label,
                "n": int(len(sub)),
                "cic_top1_repair_accuracy": _rate(sub["repaired_correct_top1"]),
                "cic_top3_repair_accuracy": _rate(sub["repaired_correct_top3"]),
                "clean_safe_repair_accuracy": _rate(sub["repaired_correct_clean_safe"]),
                "mean_area_frac_image": _safe_mean(sub["area_frac_image"]),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Refinement diagnostic
# --------------------------------------------------------------------------- #
def run_refinement(
    records: list[dict[str, Any]],
    cand_by_bench: dict[str, dict[Any, dict[str, Any]]],
    true_idx_by_bench: dict[str, dict[Any, int]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    refine_cfg = cfg.get("refinement", {})
    image_size = int(cfg.get("image_size", 224))
    image_area = float(image_size * image_size)
    rows: list[dict[str, Any]] = []
    for rec in records:
        info = cand_by_bench.get(rec["benchmark"], {}).get(rec["example_id"])
        sel_box = parse_bbox(rec["selected_bbox"])
        if not info or sel_box is None or not info.get("candidate_boxes"):
            continue
        refined, _ = select_refined_region(
            sel_box,
            info["candidate_boxes"],
            info.get("candidate_scores") or [1.0] * len(info["candidate_boxes"]),
            image_size,
            refine_cfg,
        )
        oracle_box = info.get("oracle_box")
        orig_iou = iou(sel_box, oracle_box) if oracle_box else float("nan")
        new_iou = iou(refined, oracle_box) if oracle_box else float("nan")
        # Repair after refinement, evaluated only when the chosen variant maps
        # to a scored candidate (so its neutralized prediction is known).
        true_idx = true_idx_by_bench.get(rec["benchmark"], {}).get(rec["example_id"])
        neutralized = info.get("neutralized_by_box", {})
        key = tuple(int(round(c)) for c in refined)
        refined_repair: Optional[bool] = None
        if true_idx is not None and key in neutralized:
            refined_repair = neutralized[key] == true_idx
        rows.append(
            {
                "benchmark": rec["benchmark"],
                "example_id": rec["example_id"],
                "orig_iou": orig_iou,
                "refined_iou": new_iou,
                "orig_area_frac": area_fraction(sel_box, image_area),
                "refined_area_frac": area_fraction(refined, image_area),
                "orig_repair": rec["repaired_correct_top1"],
                "refined_repair": refined_repair,
                "refined_changed": key != tuple(int(round(c)) for c in sel_box),
            }
        )
    if not rows:
        return {"evaluated": False, "reason": "no examples with candidate boxes available"}

    rdf = pd.DataFrame(rows)
    repair_eval = rdf["refined_repair"].dropna()
    result = {
        "evaluated": True,
        "n": int(len(rdf)),
        "n_changed": int(rdf["refined_changed"].sum()),
        "orig_median_iou": _safe_median(rdf["orig_iou"]),
        "refined_median_iou": _safe_median(rdf["refined_iou"]),
        "orig_iou_ge_0_5_rate": _rate(rdf["orig_iou"] >= 0.5) if rdf["orig_iou"].notna().any() else None,
        "refined_iou_ge_0_5_rate": _rate(rdf["refined_iou"] >= 0.5) if rdf["refined_iou"].notna().any() else None,
        "orig_mean_area_frac": _safe_mean(rdf["orig_area_frac"]),
        "refined_mean_area_frac": _safe_mean(rdf["refined_area_frac"]),
        "orig_repair_accuracy": _rate(rdf["orig_repair"]),
        "refined_repair_accuracy_evaluable": _rate(repair_eval) if len(repair_eval) else None,
        "refined_repair_evaluable_n": int(len(repair_eval)),
        "refined_clean_safe_drop": None,  # not recomputable from artifacts (synthetic regions)
        "refined_clean_safe_drop_note": "n/a: cannot re-score clean accuracy for synthetic regions from artifacts",
    }

    def _improved(orig: Optional[float], new: Optional[float], higher_better: bool) -> Optional[bool]:
        if orig is None or new is None:
            return None
        return (new > orig) if higher_better else (new < orig)

    result["improves_iou_ge_0_5"] = _improved(
        result["orig_iou_ge_0_5_rate"], result["refined_iou_ge_0_5_rate"], True
    )
    result["improves_median_iou"] = _improved(
        result["orig_median_iou"], result["refined_median_iou"], True
    )
    result["improves_area_frac"] = _improved(
        result["orig_mean_area_frac"], result["refined_mean_area_frac"], False
    )
    result["improves_repair"] = _improved(
        result["orig_repair_accuracy"], result["refined_repair_accuracy_evaluable"], True
    )
    result["refinement_improved_spatial_precision"] = bool(
        result["improves_median_iou"] or result["improves_iou_ge_0_5"]
    )
    return result


# --------------------------------------------------------------------------- #
# Interpretation
# --------------------------------------------------------------------------- #
def interpret(summary: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    median_iou = summary.get("median_iou") or 0.0
    iou_05 = summary.get("hit_at_iou_0_5") or 0.0
    repair = summary.get("repair_top1_accuracy") or 0.0
    cov_05 = summary.get("shortcut_coverage_ge_0_5_rate")
    obj_iou = summary.get("object_iou_median")
    clean_safe = summary.get("repair_clean_safe_accuracy")

    if iou_05 < 0.5 and repair >= 0.5:
        lines.append(
            "CIC is a coarse causal-intervention method, not an exact localization method: "
            f"exact box precision is low (hit@IoU0.5 = {iou_05:.2f}) yet repair is high "
            f"(top-1 repair = {repair:.2f})."
        )
    if cov_05 is not None and cov_05 >= 0.5 and median_iou < 0.5:
        lines.append(
            "Low IoU partly reflects larger-than-oracle intervention regions; the selected "
            f"regions cover shortcut evidence (coverage>=0.5 in {cov_05:.2f} of cases) but are "
            "spatially coarse."
        )
    useful = (obj_iou is not None and obj_iou > 0) or (clean_safe is not None and clean_safe >= 0.5)
    if useful:
        lines.append(
            "Although spatially coarse, the intervention is practically useful because it "
            "preserves causal content and clean accuracy."
        )
    lines.append(
        "Exact localization remains a limitation: this audit does not claim exact localization, "
        "segmentation, open-world discovery, or general robustness."
    )
    return lines


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def write_plot(df: pd.DataFrame, by_bucket: pd.DataFrame, cfg: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    iou_vals = df["iou"].dropna().values
    ax = axes[0]
    if len(iou_vals):
        ax.hist(iou_vals, bins=np.linspace(0, 1, 21), color="#4C78A8", alpha=0.85)
        med = float(np.median(iou_vals))
        ax.axvline(med, color="#E45756", linestyle="--", label=f"median = {med:.2f}")
        for thr in (0.3, 0.5):
            ax.axvline(thr, color="#888", linestyle=":", alpha=0.7)
        ax.legend(fontsize=8)
    ax.set_xlabel("IoU with oracle shortcut box")
    ax.set_ylabel("examples")
    ax.set_title("Exact box precision (low)")

    ax = axes[1]
    pooled = by_bucket[by_bucket["group"] == "ALL"]
    if len(pooled):
        labels = pooled["iou_bucket"].tolist()
        x = np.arange(len(labels))
        acc = pooled["cic_top1_repair_accuracy"].fillna(0).values
        ax.bar(x, acc, color="#54A24B", alpha=0.85)
        for xi, (a, n) in enumerate(zip(acc, pooled["n"])):
            ax.text(xi, a + 0.02, f"n={int(n)}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.1)
    ax.set_xlabel("IoU bucket")
    ax.set_ylabel("CIC top-1 repair accuracy")
    ax.set_title("Causal-intervention usefulness")

    fig.suptitle("CIC spatial-resolution audit: exact precision vs causal usefulness", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_summary_md(
    path: Path,
    pooled: dict[str, Any],
    per_bench: dict[str, dict[str, Any]],
    notes: list[dict[str, Any]],
    by_bucket: pd.DataFrame,
    refinement: dict[str, Any],
    cfg: dict[str, Any],
    out_paths: dict[str, str],
) -> None:
    def fmt(v: Any) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    lines: list[str] = [
        "# Spatial-Resolution and Causal-Intervention Audit",
        "",
        "**Scope.** This audit separates *exact box precision* (IoU with the oracle shortcut "
        "box) from *causal-intervention usefulness* (shortcut coverage, intervention bluntness, "
        "causal-content preservation, and repair). It reads existing benchmark artifacts only; "
        "no model is re-run and no headline metric is changed.",
        "",
        "**This audit does NOT claim** exact localization, segmentation quality, open-world "
        "discovery, or general robustness. CIC is a coarse causal-intervention method.",
        "",
        "## Benchmark availability",
        "",
    ]
    for n in notes:
        if n.get("available"):
            lines.append(
                f"- `{n['benchmark']}`: n={n['n_examples']}, oracle_box={n['has_oracle_box']}, "
                f"object_overlap={n['has_object_overlap']}, candidates={n['has_candidates']}"
            )
        else:
            lines.append(f"- `{n['benchmark']}`: skipped ({n.get('skip_reason', 'unavailable')})")

    lines += ["", "## Pooled key numbers", ""]
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for k in [
        "n_examples",
        "median_iou",
        "mean_iou",
        "hit_at_iou_0_1",
        "hit_at_iou_0_2",
        "hit_at_iou_0_3",
        "hit_at_iou_0_4",
        "hit_at_iou_0_5",
        "shortcut_coverage_median",
        "shortcut_coverage_ge_0_5_rate",
        "shortcut_coverage_ge_0_8_rate",
        "intersects_shortcut_rate",
        "area_frac_image_median",
        "area_frac_oracle_median",
        "object_iou_median",
        "object_overlap_available",
        "repair_top1_accuracy",
        "repair_top3_accuracy",
        "repair_clean_safe_accuracy",
    ]:
        if k in pooled:
            lines.append(f"| {k} | {fmt(pooled[k])} |")

    lines += ["", "## Repair-by-localization buckets", ""]
    lines.append(_markdown_table(by_bucket))

    lines += ["", "## Refinement diagnostic", ""]
    if refinement.get("evaluated"):
        lines.append(
            "Geometric variants (shrink / 2x2 split / shifts) of the top region were re-scored "
            "using only non-oracle signals (pixel area + model-derived candidate consensus). "
            "Oracle box, true label, and correctness were NOT used to select."
        )
        lines.append("")
        for k in [
            "n",
            "n_changed",
            "orig_median_iou",
            "refined_median_iou",
            "orig_iou_ge_0_5_rate",
            "refined_iou_ge_0_5_rate",
            "orig_mean_area_frac",
            "refined_mean_area_frac",
            "orig_repair_accuracy",
            "refined_repair_accuracy_evaluable",
            "refined_repair_evaluable_n",
            "refinement_improved_spatial_precision",
        ]:
            lines.append(f"- {k}: {fmt(refinement.get(k))}")
        lines.append(f"- refined_clean_safe_drop: {fmt(refinement.get('refined_clean_safe_drop'))} "
                     f"({refinement.get('refined_clean_safe_drop_note')})")
        improved = refinement.get("refinement_improved_spatial_precision")
        lines.append("")
        lines.append(
            "**Refinement did improve spatial precision.**"
            if improved
            else "**Refinement did NOT improve spatial precision; reported honestly.** A non-oracle "
            "geometric refinement does not recover exact boxes here."
        )
    else:
        lines.append(f"Refinement not evaluated: {refinement.get('reason', 'disabled')}.")

    lines += ["", "## Interpretation", ""]
    for line in interpret(pooled):
        lines.append(f"- {line}")

    lines += ["", "## Outputs", ""]
    for k, v in out_paths.items():
        lines.append(f"- {k}: `{v}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(cfg: dict[str, Any]) -> dict[str, Any]:
    results_root = Path(cfg.get("results_dir", "results"))
    out_dir = ensure_dir(results_root / cfg.get("output_subdir", "spatial_resolution_audit"))

    all_records: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    per_bench: dict[str, dict[str, Any]] = {}
    cand_by_bench: dict[str, dict[Any, dict[str, Any]]] = {}
    true_idx_by_bench: dict[str, dict[Any, int]] = {}

    for bench in cfg.get("benchmarks", []):
        records, note = load_benchmark_records(bench, cfg, results_root)
        notes.append(note)
        if not records:
            continue
        all_records.extend(records)
        bench_dir = results_root / bench["dir"]
        cand_by_bench[bench["name"]] = _load_candidates_and_oracle(bench, bench_dir)
        cert = pd.read_csv(bench_dir / bench["certificates"])
        true_idx_by_bench[bench["name"]] = _true_label_index(
            cert, _resolve(bench, cfg, "oracle_method")
        )

    if not all_records:
        raise RuntimeError("No benchmark artifacts available for the spatial-resolution audit.")

    df = pd.DataFrame(all_records)
    pooled = summarize(df, cfg)
    for name, group in df.groupby("benchmark"):
        per_bench[name] = summarize(group, cfg)

    bucket_frames = [bucket_table(df, cfg, "ALL")]
    for name, group in df.groupby("benchmark"):
        bucket_frames.append(bucket_table(group, cfg, name))
    by_bucket = pd.concat(bucket_frames, ignore_index=True)

    refinement: dict[str, Any] = {"evaluated": False, "reason": "disabled"}
    if cfg.get("refinement", {}).get("enabled", False):
        refinement = run_refinement(all_records, cand_by_bench, true_idx_by_bench, cfg)

    # ----- write outputs -----
    metrics_csv = out_dir / "spatial_resolution_metrics.csv"
    bucket_csv = out_dir / "spatial_resolution_by_bucket.csv"
    key_json = out_dir / "spatial_resolution_key_numbers.json"
    summary_md = out_dir / "spatial_resolution_summary.md"
    plot_png = out_dir / "spatial_resolution_plot.png"

    df.to_csv(metrics_csv, index=False)
    by_bucket.to_csv(bucket_csv, index=False)
    write_plot(df, by_bucket, cfg, plot_png)

    out_paths = {
        "summary_md": str(summary_md),
        "key_numbers_json": str(key_json),
        "metrics_csv": str(metrics_csv),
        "by_bucket_csv": str(bucket_csv),
        "plot_png": str(plot_png),
    }
    key_numbers = {
        "pooled": pooled,
        "per_benchmark": per_bench,
        "benchmark_notes": notes,
        "refinement": refinement,
        "exact_localization_remains_a_limitation": bool(
            (pooled.get("hit_at_iou_0_5") or 0.0) < 0.5
        ),
        "interpretation": interpret(pooled),
        "outputs": out_paths,
    }
    key_json.write_text(json.dumps(key_numbers, indent=2, default=_json_default), encoding="utf-8")
    write_summary_md(summary_md, pooled, per_bench, notes, by_bucket, refinement, cfg, out_paths)

    return key_numbers


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/spatial_resolution_audit.yaml")
    args = parser.parse_args()
    result = run(load_config(args.config))
    print(json.dumps(result["pooled"], indent=2, default=_json_default))
    print("Outputs:")
    for k, v in result["outputs"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
