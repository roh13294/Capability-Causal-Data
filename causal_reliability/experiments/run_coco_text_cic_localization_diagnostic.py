from __future__ import annotations

"""COCO-Text proposal-localization diagnostic.

Experiment name: ``coco_text_cic_localization_diagnostic``.

Scientific goal: the full COCO-Text CIC run (``coco_text_cic_full``) shows CIC
beats matched random proposals strongly on target-prob, rank and text-distractor
metrics, yet both support gates fail **only** because the selected (top-1) region
rarely overlaps an annotated text box (strict 0.128 / directional 0.140 vs a 0.60
threshold). This diagnostic decides whether that low text-overlap is:

* **A. a proposal-coverage failure** - text-overlapping proposals are simply not
  in the open candidate set, or
* **B. a scoring/ranking failure** - text-overlapping proposals exist but CIC
  ranks object/background regions above them.

It does so without touching any existing metric, gate or final-report file. It
re-reads the frozen ``coco_text_cic_full`` proposal diagnostics (the exact open
proposals the headline run scored) for the geometry/ranking half, and re-runs the
real CLIP backend only for the repair half (best text-overlapping proposal, top-k
unions, area-normalised top-1, inference-time text-dilated proposals) over the
verified directional/strict subsets.

Writes ONLY under ``results/coco_text_cic_localization_diagnostic/``.

Guardrails (asserted in code):
* never writes under ``results/final_report/`` or ``results/coco_text_cic_full/``;
* ``open_world_claim_allowed`` stays ``False``;
* any selection that consumes ground-truth text geometry inside the *scoring* rule
  (``reward_text_oracle`` mode, the best-text-IoU proposal) is labelled an oracle /
  leakage diagnostic and never feeds a deployable claim.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from causal_reliability.analysis import coco_text_localization_diagnostic as lib
from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.data.natural_text_dataset import load_local_folder_dataset
from causal_reliability.discovery.cic_region_scoring import neutralize_region, score_region_candidates
from causal_reliability.discovery.open_region_proposals import OCR_FAMILY, proposal_family
from causal_reliability.discovery.region_proposals import (
    _clip_bbox,
    _features,
    _gray_edges,
    text_box_component_proposals,
    textness_proposals,
)
from causal_reliability.experiments.run_coco_text_cic_full import _build_chunked_predict_fn
from causal_reliability.experiments.run_coco_text_cic_triage import (
    aliases_for,
    is_target_label,
    label_rank,
    label_set_prob,
    pairwise_margin_toward_target,
)
from causal_reliability.experiments.run_natural_text_open_proposal_cic import (
    _device,
    _downloads_allowed,
    _json_default,
)
from causal_reliability.real_models.clip_zero_shot import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED_TAG,
    DEFAULT_TRANSFORMERS_MODEL,
    ClipStatus,
    check_clip_available,
)
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir

DEFAULT_OUTPUT_SUBDIR = "coco_text_cic_localization_diagnostic"

# Paths this diagnostic must never write to.
FORBIDDEN_WRITE_ROOTS = (
    Path("results/final_report"),
    Path("results/coco_text_cic_full"),
    Path("results/coco_text_cic_triage"),
)

PROB_EPS = 0.01
TOPK = lib.TOPK_VALUES
AREA_MODES = ("original", "div_sqrt_area", "div_area_clip", "penalize_object", "reward_text_oracle")
TEXT_DILATED_FAMILIES = ("text_like_cc", "dilated_text_like", "thin_high_contrast_rect", "edge_dense_small")


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _parse_box_json(value: str) -> lib.BBox:
    return tuple(int(v) for v in json.loads(value))  # type: ignore[return-value]


def _proposals_for_example(diag: pd.DataFrame, exclude_ocr: bool) -> pd.DataFrame:
    df = diag.copy()
    if exclude_ocr:
        df = df[df["proposal_family"] != OCR_FAMILY]
    # CIC ranking is by descending score; ties keep the recorded global rank.
    return df.sort_values(["score", "rank"], ascending=[False, True]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Model-free recall + ranking diagnostics (Tasks 1 & 2, geometry/ranking half)
# --------------------------------------------------------------------------- #
def compute_recall_and_ranking(
    diag_by_example: dict[int, pd.DataFrame],
    boxes_by_example: dict[int, dict[str, list[lib.BBox]]],
) -> pd.DataFrame:
    """One row per example: proposal recall + CIC ranking of text/object proposals.

    Computed over the *open* (OCR-excluded) candidate pool - the exact pool the
    headline ``cic_top1_repair_excl_ocr`` selects from.
    """

    rows: list[dict[str, Any]] = []
    for eid, raw in diag_by_example.items():
        boxes = boxes_by_example.get(eid, {"text": [], "object": []})
        text_boxes, object_boxes = boxes["text"], boxes["object"]
        ranked = _proposals_for_example(raw, exclude_ocr=True)
        prop_boxes = [_parse_box_json(b) for b in ranked["bbox"].tolist()]

        recall = lib.proposal_recall_metrics(prop_boxes, text_boxes, object_boxes)

        text_flags = [lib.overlaps_any(p, text_boxes) for p in prop_boxes]
        object_flags = [lib.overlaps_any(p, object_boxes) for p in prop_boxes]
        overlap_at_k = lib.text_overlap_at_k(text_flags)

        rank_best_text = lib.rank_of_first_true(text_flags)
        rank_best_object = lib.rank_of_first_true(object_flags)

        selected_score = float(ranked["score"].iloc[0]) if len(ranked) else float("nan")
        selected_overlaps_text = bool(text_flags[0]) if text_flags else False

        # CIC score of the best text-overlapping proposal (highest CIC score among
        # text-overlapping proposals == first text flag in the CIC ranking).
        best_text_cic_score = (
            float(ranked["score"].iloc[rank_best_text - 1]) if rank_best_text is not None else float("nan")
        )

        row = {
            "example_id": int(eid),
            **recall,
            "selected_cic_score": selected_score,
            "selected_overlaps_text": selected_overlaps_text,
            "rank_best_text_proposal": rank_best_text if rank_best_text is not None else np.nan,
            "rank_best_object_proposal": rank_best_object if rank_best_object is not None else np.nan,
            "best_text_proposal_cic_score": best_text_cic_score,
            "has_text_overlapping_proposal": bool(rank_best_text is not None),
        }
        for k, v in overlap_at_k.items():
            row[f"text_overlap_at_{k}"] = bool(v)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_by_subset(per_example: pd.DataFrame, subset_ids: dict[str, list[int]], boolcols: list[str], meancols: list[str], mediancols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subset, ids in subset_ids.items():
        sub = per_example[per_example["example_id"].isin(set(int(i) for i in ids))]
        rec: dict[str, Any] = {"subset": subset, "n": int(len(sub))}
        for c in boolcols:
            rec[f"{c}_rate"] = float(sub[c].astype(bool).mean()) if len(sub) else float("nan")
        for c in meancols:
            rec[f"{c}_mean"] = float(sub[c].astype(float).mean()) if len(sub) else float("nan")
        for c in mediancols:
            vals = sub[c].dropna().astype(float)
            rec[f"{c}_median"] = float(vals.median()) if len(vals) else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Model-dependent repair helpers
# --------------------------------------------------------------------------- #
def _repair_outcome(
    probs: np.ndarray | None,
    *,
    allowed: list[str],
    target: str,
    aliases: set[str],
    target_idxs: list[int],
    distractor_idx: int,
    orig: dict[str, float],
) -> dict[str, Any]:
    """Mirror ``run_coco_text_cic_full._derive_method_row`` for one prob vector."""

    if probs is None:
        return {
            "available": False, "strict_correct": np.nan, "alias_correct": np.nan,
            "target_prob_post": np.nan, "target_rank_post": np.nan, "distractor_prob_post": np.nan,
            "target_prob_gain": np.nan, "target_prob_improved": np.nan,
            "distractor_decreased": np.nan, "recovers_top5": np.nan, "pairwise_recovered": np.nan,
        }
    probs = np.asarray(probs, dtype=np.float64)
    pred_label = allowed[int(probs.argmax())]
    t_prob = label_set_prob(probs, target_idxs)
    t_rank = label_rank(probs, target_idxs)
    d_prob = float(probs[distractor_idx])
    margin = pairwise_margin_toward_target(t_prob, d_prob)
    return {
        "available": True,
        "strict_correct": bool(pred_label == target),
        "alias_correct": bool(is_target_label(pred_label, target, aliases)),
        "target_prob_post": t_prob,
        "target_rank_post": t_rank,
        "distractor_prob_post": d_prob,
        "target_prob_gain": float(t_prob - orig["target_prob"]),
        "target_prob_improved": bool(t_prob > orig["target_prob"] + PROB_EPS),
        "distractor_decreased": bool(d_prob < orig["distractor_prob"] - PROB_EPS),
        "recovers_top5": bool(t_rank <= 5),
        "pairwise_recovered": bool(margin > orig["pairwise_margin"] and t_prob > d_prob),
    }


def _build_dilated_proposals(pil: Image.Image, family: str, max_components: int = 12):
    """Inference-time text-dilated proposal families (no ground-truth boxes).

    All families derive purely from edge/contrast/connected-component structure of
    the image pixels - the COCO-Text annotations are never consulted here.
    """

    width, height = pil.size
    edges = _gray_edges(pil)
    if family == "text_like_cc":
        return textness_proposals(pil, edges, max_components=max_components)
    if family == "thin_high_contrast_rect":
        return text_box_component_proposals(pil, edges, max_components=max_components)
    if family == "dilated_text_like":
        return _dilated_text_like_proposals(pil, edges, max_components=max_components)
    if family == "edge_dense_small":
        return _edge_dense_small_proposals(pil, edges, max_components=max_components)
    raise ValueError(f"unknown text-dilated family {family!r}")


def _dilated_text_like_proposals(pil: Image.Image, edges: np.ndarray, max_components: int = 12):
    """High-frequency components, morphologically dilated to merge glyphs into words."""

    from scipy import ndimage

    width, height = pil.size
    mask = edges > max(0.06, float(np.quantile(edges, 0.88)))
    # Dilate horizontally more than vertically so adjacent characters merge into
    # word/line-shaped components (text-like), still without any annotation.
    struct = np.ones((max(2, height // 90), max(3, width // 45)), dtype=bool)
    dilated = ndimage.binary_dilation(mask, structure=struct)
    labels, n = ndimage.label(dilated)
    comps: list[tuple[float, lib.BBox]] = []
    for idx in range(1, n + 1):
        ys, xs = np.where(labels == idx)
        if xs.size < max(12, width * height // 2600):
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bbox = _clip_bbox((x0, y0, x1, y1), width, height)
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if bw * bh > 0.30 * width * height or bw < 8 or bh < 6:
            continue
        comps.append((float(xs.size), bbox))
    comps = sorted(comps, key=lambda it: it[0], reverse=True)[:max_components]
    return [_features(f"dilated_{i:04d}", b, "dilated_text_like", edges, width, height) for i, (_, b) in enumerate(comps)]


def _edge_dense_small_proposals(pil: Image.Image, edges: np.ndarray, max_components: int = 12):
    """Small sliding windows ranked by edge density (text-like high-frequency)."""

    width, height = pil.size
    boxes: list[tuple[float, lib.BBox]] = []
    for wf, hf in [(0.16, 0.10), (0.22, 0.14), (0.12, 0.12)]:
        bw, bh = max(8, int(width * wf)), max(8, int(height * hf))
        step_x, step_y = max(6, bw // 2), max(6, bh // 2)
        for y in range(0, max(1, height - bh + 1), step_y):
            for x in range(0, max(1, width - bw + 1), step_x):
                patch = edges[y : y + bh, x : x + bw]
                dens = float((patch > 0.08).mean()) if patch.size else 0.0
                boxes.append((dens, _clip_bbox((x, y, x + bw, y + bh), width, height)))
    boxes = sorted(boxes, key=lambda it: it[0], reverse=True)[:max_components]
    return [_features(f"edgesmall_{i:04d}", b, "edge_dense_small", edges, width, height) for i, (_, b) in enumerate(boxes)]


def _matched_random_box(ranked_all: pd.DataFrame, top1_box: lib.BBox | None) -> lib.BBox | None:
    randoms = ranked_all[ranked_all["proposal_type"] == "random_patch_control"]
    if randoms.empty:
        return None
    target_area = 0.0
    if top1_box is not None:
        # area fraction of top1 within the 224x224 frame
        x0, y0, x1, y1 = top1_box
        target_area = float((x1 - x0) * (y1 - y0)) / float(224 * 224)
    idx = (randoms["area_fraction"].astype(float) - target_area).abs().idxmin()
    return _parse_box_json(randoms.loc[idx, "bbox"])


# --------------------------------------------------------------------------- #
# Orchestration of the model-dependent diagnostics over the union subset
# --------------------------------------------------------------------------- #
def run_model_diagnostics(
    union_ids: list[int],
    examples_by_id: dict[int, dict[str, Any]],
    diag_by_example: dict[int, pd.DataFrame],
    status: ClipStatus,
    device: str,
    batch_size: int,
    max_dilated_components: int,
) -> dict[str, pd.DataFrame]:
    """Run CLIP-backed repair diagnostics. Returns per-example frames keyed by name."""

    predict_cache: dict[tuple[str, ...], Any] = {}
    ranking_repair_rows: list[dict[str, Any]] = []
    area_rows: list[dict[str, Any]] = []
    dilated_rows: list[dict[str, Any]] = []
    contact_records: list[dict[str, Any]] = []

    for eid in union_ids:
        ex = examples_by_id[eid]
        allowed = list(ex["allowed_clip_labels"])
        key = tuple(allowed)
        predict_fn = predict_cache.get(key)
        if predict_fn is None:
            predict_fn = _build_chunked_predict_fn(status, allowed, device, batch_size)
            predict_cache[key] = predict_fn

        target = str(ex["human_label"])
        aliases = aliases_for(target)
        target_idxs = [i for i, lbl in enumerate(allowed) if is_target_label(lbl, target, aliases)]
        if not target_idxs:
            continue

        pil = Image.fromarray((np.asarray(ex["image"]).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")
        text_boxes = [tuple(int(v) for v in b) for b in ex.get("text_boxes", [])]
        object_boxes = [tuple(int(v) for v in b) for b in ex.get("object_boxes", [])]

        original_probs = np.asarray(predict_fn([pil]), dtype=np.float64)[0]
        non_target_idxs = [i for i in range(len(allowed)) if i not in set(target_idxs)]
        distractor_idx = int(max(non_target_idxs, key=lambda i: original_probs[i])) if non_target_idxs else int(target_idxs[0])
        orig = {
            "target_prob": label_set_prob(original_probs, target_idxs),
            "target_rank": label_rank(original_probs, target_idxs),
            "distractor_prob": float(original_probs[distractor_idx]),
            "pairwise_margin": pairwise_margin_toward_target(
                label_set_prob(original_probs, target_idxs), float(original_probs[distractor_idx])
            ),
        }

        ranked_all = _proposals_for_example(diag_by_example[eid], exclude_ocr=False)
        ranked = _proposals_for_example(diag_by_example[eid], exclude_ocr=True)
        prop_boxes = [_parse_box_json(b) for b in ranked["bbox"].tolist()]
        text_ious = [lib.best_iou_against(p, text_boxes) for p in prop_boxes]
        object_ious = [lib.best_iou_against(p, object_boxes) for p in prop_boxes]

        def probs_for_box(box):
            return np.asarray(predict_fn([neutralize_region(pil, tuple(int(v) for v in box))]), dtype=np.float64)[0]

        def union_probs(boxes):
            if not boxes:
                return None
            imgs = [neutralize_region(pil, tuple(int(v) for v in b)) for b in boxes]
            return np.asarray(predict_fn(imgs), dtype=np.float64).mean(axis=0)

        # --- selected CIC top-1 (excl OCR) ---
        selected_box = prop_boxes[0] if prop_boxes else None
        selected = _repair_outcome(probs_for_box(selected_box) if selected_box else None,
                                   allowed=allowed, target=target, aliases=aliases,
                                   target_idxs=target_idxs, distractor_idx=distractor_idx, orig=orig)

        # --- best text-overlapping proposal (highest text IoU; ORACLE selection) ---
        best_text_i = int(np.argmax(text_ious)) if text_ious and max(text_ious) > 0 else None
        best_text_box = prop_boxes[best_text_i] if best_text_i is not None else None
        best_text = _repair_outcome(probs_for_box(best_text_box) if best_text_box else None,
                                    allowed=allowed, target=target, aliases=aliases,
                                    target_idxs=target_idxs, distractor_idx=distractor_idx, orig=orig)

        # --- best object-overlapping proposal (highest object IoU; ORACLE selection) ---
        best_obj_i = int(np.argmax(object_ious)) if object_ious and max(object_ious) > 0 else None
        best_obj_box = prop_boxes[best_obj_i] if best_obj_i is not None else None
        best_obj = _repair_outcome(probs_for_box(best_obj_box) if best_obj_box else None,
                                   allowed=allowed, target=target, aliases=aliases,
                                   target_idxs=target_idxs, distractor_idx=distractor_idx, orig=orig)

        # --- top-k unions (CIC-ranked, deployable: no ground truth) ---
        topk_repairs: dict[int, dict[str, Any]] = {}
        for k in TOPK:
            union = prop_boxes[:k]
            topk_repairs[k] = _repair_outcome(union_probs(union),
                                              allowed=allowed, target=target, aliases=aliases,
                                              target_idxs=target_idxs, distractor_idx=distractor_idx, orig=orig)

        rr = {
            "example_id": int(eid),
            "human_label": target,
            "selected_alias_correct": selected["alias_correct"],
            "selected_target_prob_gain": selected["target_prob_gain"],
            "selected_overlaps_text": bool(text_ious[0] >= lib.OVERLAP_IOU) if text_ious else False,
            "best_text_iou_available": float(max(text_ious)) if text_ious else 0.0,
            "best_text_alias_correct": best_text["alias_correct"],
            "best_text_target_prob_gain": best_text["target_prob_gain"],
            "best_text_recovers_top5": best_text["recovers_top5"],
            "best_object_alias_correct": best_obj["alias_correct"],
            "best_object_target_prob_gain": best_obj["target_prob_gain"],
        }
        for k in TOPK:
            rr[f"topk{k}_alias_correct"] = topk_repairs[k]["alias_correct"]
            rr[f"topk{k}_target_prob_gain"] = topk_repairs[k]["target_prob_gain"]
            rr[f"topk{k}_distractor_decreased"] = topk_repairs[k]["distractor_decreased"]
            rr[f"topk{k}_overlaps_text"] = bool(any(t >= lib.OVERLAP_IOU for t in text_ious[:k])) if text_ious else False
        ranking_repair_rows.append(rr)

        # --- area-normalised scoring re-orderings ---
        scored = [
            lib.ScoredProposal(
                candidate_id=str(ranked["candidate_id"].iloc[i]),
                score=float(ranked["score"].iloc[i]),
                area_fraction=float(ranked["area_fraction"].iloc[i]),
                overlaps_text=bool(text_ious[i] >= lib.OVERLAP_IOU),
                overlaps_object=bool(object_ious[i] >= lib.OVERLAP_IOU),
                text_iou=float(text_ious[i]),
            )
            for i in range(len(prop_boxes))
        ]
        box_by_cid = {str(ranked["candidate_id"].iloc[i]): prop_boxes[i] for i in range(len(prop_boxes))}
        textiou_by_cid = {str(ranked["candidate_id"].iloc[i]): text_ious[i] for i in range(len(prop_boxes))}
        for mode in AREA_MODES:
            if not scored:
                continue
            new_top = lib.reorder(scored, mode)[0]
            box = box_by_cid[new_top.candidate_id]
            outcome = _repair_outcome(probs_for_box(box), allowed=allowed, target=target, aliases=aliases,
                                      target_idxs=target_idxs, distractor_idx=distractor_idx, orig=orig)
            area_rows.append({
                "example_id": int(eid),
                "mode": mode,
                "is_leakage": lib.is_leakage_mode(mode),
                "new_top1_overlaps_text": bool(textiou_by_cid[new_top.candidate_id] >= lib.OVERLAP_IOU),
                "new_top1_text_iou": float(textiou_by_cid[new_top.candidate_id]),
                "new_top1_area_fraction": float(new_top.area_fraction),
                "alias_correct": outcome["alias_correct"],
                "target_prob_gain": outcome["target_prob_gain"],
                "target_prob_improved": outcome["target_prob_improved"],
            })

        # --- inference-time text-dilated proposal families ---
        for family in TEXT_DILATED_FAMILIES:
            props = _build_dilated_proposals(pil, family, max_components=max_dilated_components)
            if not props:
                dilated_rows.append({
                    "example_id": int(eid), "family": family, "n_proposals": 0,
                    "best_text_iou_available": 0.0, "top1_overlaps_text": False,
                    "alias_correct": np.nan, "target_prob_gain": np.nan, "target_prob_improved": np.nan,
                })
                continue
            scores, _ = score_region_candidates(pil, props, predict_fn)
            fam_boxes = [tuple(int(v) for v in s.bbox) for s in scores]
            fam_text_ious = [lib.best_iou_against(b, text_boxes) for b in fam_boxes]
            top1_box = fam_boxes[0]
            outcome = _repair_outcome(probs_for_box(top1_box), allowed=allowed, target=target, aliases=aliases,
                                      target_idxs=target_idxs, distractor_idx=distractor_idx, orig=orig)
            dilated_rows.append({
                "example_id": int(eid),
                "family": family,
                "n_proposals": int(len(props)),
                "best_text_iou_available": float(max(fam_text_ious)) if fam_text_ious else 0.0,
                "top1_overlaps_text": bool(fam_text_ious[0] >= lib.OVERLAP_IOU),
                "alias_correct": outcome["alias_correct"],
                "target_prob_gain": outcome["target_prob_gain"],
                "target_prob_improved": outcome["target_prob_improved"],
            })

        contact_records.append({
            "example_id": int(eid),
            "human_label": target,
            "pil": pil,
            "text_boxes": text_boxes,
            "selected_box": selected_box,
            "best_text_box": best_text_box,
            "best_object_box": best_obj_box,
            "oracle_text_box": text_boxes[0] if text_boxes else None,
            "matched_random_box": _matched_random_box(ranked_all, selected_box),
            "best_text_iou": float(max(text_ious)) if text_ious else 0.0,
        })

    return {
        "ranking_repair": pd.DataFrame(ranking_repair_rows),
        "area": pd.DataFrame(area_rows),
        "dilated": pd.DataFrame(dilated_rows),
        "contact_records": contact_records,  # type: ignore[dict-item]
    }


# --------------------------------------------------------------------------- #
# Aggregation of model-dependent frames
# --------------------------------------------------------------------------- #
def aggregate_ranking_repair(rr: pd.DataFrame) -> dict[str, Any]:
    if rr.empty:
        return {}

    def rate(col):
        v = rr[col].dropna()
        return float(v.astype(bool).mean()) if len(v) else float("nan")

    def mean(col):
        v = rr[col].dropna()
        return float(v.astype(float).mean()) if len(v) else float("nan")

    out = {
        "n": int(len(rr)),
        "selected_alias_repair": rate("selected_alias_correct"),
        "selected_text_overlap": rate("selected_overlaps_text"),
        "best_text_alias_repair": rate("best_text_alias_correct"),
        "best_text_recovers_top5": rate("best_text_recovers_top5"),
        "best_object_alias_repair": rate("best_object_alias_correct"),
        "best_text_median_target_prob_gain": float(rr["best_text_target_prob_gain"].dropna().median()) if len(rr) else float("nan"),
        "best_object_median_target_prob_gain": float(rr["best_object_target_prob_gain"].dropna().median()) if len(rr) else float("nan"),
        "examples_with_text_overlapping_proposal": rate_positive(rr["best_text_iou_available"], lib.OVERLAP_IOU),
    }
    for k in TOPK:
        out[f"topk{k}_alias_repair"] = rate(f"topk{k}_alias_correct")
        out[f"topk{k}_target_prob_improvement"] = float(
            (rr[f"topk{k}_target_prob_gain"].dropna() > PROB_EPS).mean()
        ) if len(rr) else float("nan")
        out[f"topk{k}_median_target_prob_gain"] = float(rr[f"topk{k}_target_prob_gain"].dropna().median()) if len(rr) else float("nan")
        out[f"topk{k}_distractor_decrease"] = rate(f"topk{k}_distractor_decreased")
        out[f"topk{k}_text_overlap"] = rate(f"topk{k}_overlaps_text")
    return out


def rate_positive(series: pd.Series, thr: float) -> float:
    v = series.dropna().astype(float)
    return float((v >= thr).mean()) if len(v) else float("nan")


def aggregate_area(area: pd.DataFrame) -> pd.DataFrame:
    if area.empty:
        return pd.DataFrame()
    rows = []
    for mode, g in area.groupby("mode"):
        rows.append({
            "mode": mode,
            "is_leakage": bool(g["is_leakage"].any()),
            "n": int(len(g)),
            "new_top1_text_overlap_rate": float(g["new_top1_overlaps_text"].astype(bool).mean()),
            "mean_new_top1_text_iou": float(g["new_top1_text_iou"].astype(float).mean()),
            "mean_new_top1_area_fraction": float(g["new_top1_area_fraction"].astype(float).mean()),
            "alias_repair_rate": float(g["alias_correct"].dropna().astype(bool).mean()) if g["alias_correct"].notna().any() else float("nan"),
            "target_prob_improvement_rate": float(g["target_prob_improved"].dropna().astype(bool).mean()) if g["target_prob_improved"].notna().any() else float("nan"),
            "median_target_prob_gain": float(g["target_prob_gain"].dropna().median()) if g["target_prob_gain"].notna().any() else float("nan"),
        })
    order = {m: i for i, m in enumerate(AREA_MODES)}
    return pd.DataFrame(rows).sort_values("mode", key=lambda s: s.map(order)).reset_index(drop=True)


def aggregate_dilated(dilated: pd.DataFrame) -> pd.DataFrame:
    if dilated.empty:
        return pd.DataFrame()
    rows = []
    for family, g in dilated.groupby("family"):
        rows.append({
            "family": family,
            "n_examples": int(len(g)),
            "mean_n_proposals": float(g["n_proposals"].astype(float).mean()),
            "mean_best_text_iou_available": float(g["best_text_iou_available"].astype(float).mean()),
            "text_recall_iou_010": float((g["best_text_iou_available"].astype(float) >= lib.OVERLAP_IOU).mean()),
            "top1_text_overlap_rate": float(g["top1_overlaps_text"].astype(bool).mean()),
            "alias_repair_rate": float(g["alias_correct"].dropna().astype(bool).mean()) if g["alias_correct"].notna().any() else float("nan"),
            "target_prob_improvement_rate": float(g["target_prob_improved"].dropna().astype(bool).mean()) if g["target_prob_improved"].notna().any() else float("nan"),
            "median_target_prob_gain": float(g["target_prob_gain"].dropna().median()) if g["target_prob_gain"].notna().any() else float("nan"),
        })
    order = {m: i for i, m in enumerate(TEXT_DILATED_FAMILIES)}
    return pd.DataFrame(rows).sort_values("family", key=lambda s: s.map(order)).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Contact sheet
# --------------------------------------------------------------------------- #
def write_contact_sheet(records: list[dict[str, Any]], out_dir: Path, n: int) -> list[str]:
    ensure_dir(out_dir)
    # Prefer examples that actually have a text-overlapping proposal available so the
    # panels are informative; fall back to first records otherwise.
    records = sorted(records, key=lambda r: r["best_text_iou"], reverse=True)
    panels = [
        ("selected_box", "selected CIC region", "#1b9e77"),
        ("best_text_box", "best text-overlapping prop", "#e4572e"),
        ("best_object_box", "best object-overlapping prop", "#4c78a8"),
        ("oracle_text_box", "oracle text box (eval-only)", "#9467bd"),
        ("matched_random_box", "matched random proposal", "#7f7f7f"),
    ]
    paths: list[str] = []
    for r in records[:n]:
        fig, axes = plt.subplots(1, len(panels), figsize=(2.4 * len(panels), 2.8))
        for ax, (k, title, color) in zip(axes, panels):
            ax.imshow(r["pil"])
            box = r.get(k)
            if box is not None:
                x0, y0, x1, y1 = box
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, lw=2))
            for (tx0, ty0, tx1, ty1) in r["text_boxes"]:
                ax.add_patch(Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0, fill=False, edgecolor="#cccccc", lw=0.6, ls=":"))
            ax.set_title(title, fontsize=7)
            ax.set_axis_off()
        fig.suptitle(f"example {r['example_id']} ({r['human_label']}); best text IoU available={r['best_text_iou']:.2f}", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        path = out_dir / f"contact_{r['example_id']}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(str(path))
    return paths


# --------------------------------------------------------------------------- #
# Diagnosis logic
# --------------------------------------------------------------------------- #
GATE_TEXT_OVERLAP_THRESHOLD = 0.60


def diagnose(recall_agg: pd.DataFrame, ranking_agg: pd.DataFrame, repair_summary: dict[str, Any]) -> dict[str, Any]:
    """Decide proposal-coverage failure (A) vs scoring/ranking failure (B).

    The shortfall of the selected text-overlap (text-overlap@1) relative to the
    gate threshold is decomposed into two additive components:

    * ``coverage_gap`` = max(0, threshold - recall_ceiling), where ``recall_ceiling``
      is the fraction of examples with *any* text-overlapping proposal (the best a
      perfect ranker could achieve). This is the part no re-ranking can fix.
    * ``ranking_gap``  = recall_ceiling - text_overlap@1: the text-overlapping
      proposals that exist but are ranked below rank 1.

    Whichever component is larger is the primary diagnosis; both are reported.
    """

    def get(df, subset, col):
        row = df[df["subset"] == subset]
        return float(row[col].iloc[0]) if not row.empty and col in row.columns else float("nan")

    strict_has_text = get(recall_agg, "strict_39", "has_text_iou_010_rate")
    dir_has_text = get(recall_agg, "directional_57", "has_text_iou_010_rate")
    strict_text_at1 = get(ranking_agg, "strict_39", "text_overlap_at_1_rate")
    strict_text_at5 = get(ranking_agg, "strict_39", "text_overlap_at_5_rate")
    strict_text_at10 = get(ranking_agg, "strict_39", "text_overlap_at_10_rate")
    dir_text_at5 = get(ranking_agg, "directional_57", "text_overlap_at_5_rate")
    strict_rank_best_text = get(ranking_agg, "strict_39", "rank_best_text_proposal_median")

    thr = GATE_TEXT_OVERLAP_THRESHOLD
    coverage_gap = float(max(0.0, thr - strict_has_text)) if np.isfinite(strict_has_text) else float("nan")
    ranking_gap = float(strict_has_text - strict_text_at1) if (np.isfinite(strict_has_text) and np.isfinite(strict_text_at1)) else float("nan")

    if np.isfinite(ranking_gap) and np.isfinite(coverage_gap):
        primary = "ranking_failure" if ranking_gap >= coverage_gap else "proposal_coverage_failure"
    else:
        primary = "undetermined"
    # A coverage ceiling exists whenever even a perfect ranker could not clear the gate.
    coverage_caps_gate = bool(np.isfinite(strict_has_text) and strict_has_text < thr)

    best_text_repairs = repair_summary.get("best_text_alias_repair")
    selected_repairs = repair_summary.get("selected_alias_repair")
    return {
        "primary_diagnosis": primary,
        "gate_text_overlap_threshold": thr,
        "ranking_gap": ranking_gap,
        "coverage_gap_vs_threshold": coverage_gap,
        "coverage_caps_gate_even_with_perfect_ranking": coverage_caps_gate,
        "strict_proposal_recall_text_iou_010": strict_has_text,
        "directional_proposal_recall_text_iou_010": dir_has_text,
        "strict_text_overlap_at_1": strict_text_at1,
        "strict_text_overlap_at_5": strict_text_at5,
        "strict_text_overlap_at_10": strict_text_at10,
        "directional_text_overlap_at_5": dir_text_at5,
        "strict_median_rank_best_text_proposal": strict_rank_best_text,
        "best_text_proposal_alias_repair": best_text_repairs,
        "selected_cic_alias_repair": selected_repairs,
        "best_text_beats_selected": (
            bool(best_text_repairs is not None and selected_repairs is not None
                 and np.isfinite(best_text_repairs) and np.isfinite(selected_repairs)
                 and best_text_repairs > selected_repairs)
        ),
    }


# --------------------------------------------------------------------------- #
# Main run
# --------------------------------------------------------------------------- #
def _assert_safe_output(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    for forbidden in FORBIDDEN_WRITE_ROOTS:
        if resolved == forbidden.resolve() or forbidden.resolve() in resolved.parents:
            raise RuntimeError(f"refusing to write under protected path {forbidden}")


def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / cfg.get("output_subdir", DEFAULT_OUTPUT_SUBDIR))
    _assert_safe_output(out_dir)

    full_dir = Path(cfg.get("full_dir", "results/coco_text_cic_full"))
    triage_dir = Path(cfg.get("triage_dir", "results/coco_text_cic_triage"))
    diag = pd.read_csv(full_dir / "coco_text_full_proposal_diagnostics.csv")

    data_cfg = dict(cfg.get("data", {}))
    image_size = int(data_cfg.get("image_size", 224))
    bundle = load_local_folder_dataset(
        root=data_cfg.get("root", "data/coco_text_cic"),
        metadata_csv=data_cfg.get("metadata_csv", "data/coco_text_cic/metadata.csv"),
        image_size=image_size,
        split=str(data_cfg.get("split", "test")),
    )
    examples_by_id = {int(ex["example_id"]): ex for ex in bundle.examples}
    boxes_by_example = {
        int(ex["example_id"]): {
            "text": [tuple(int(v) for v in b) for b in ex.get("text_boxes", [])],
            "object": [tuple(int(v) for v in b) for b in ex.get("object_boxes", [])],
        }
        for ex in bundle.examples
    }
    all_ids = sorted(examples_by_id)

    def read_ids(path: Path) -> list[int]:
        if not path.exists():
            return []
        f = pd.read_csv(path)
        return [int(v) for v in f["example_id"].tolist()] if "example_id" in f.columns else []

    directional = [i for i in read_ids(triage_dir / "coco_text_verified_directional_failures.csv") if i in examples_by_id]
    strict = [i for i in read_ids(triage_dir / "coco_text_verified_oracle_repairable_failures.csv") if i in examples_by_id]
    subset_ids = {"all_500": all_ids, "directional_57": directional, "strict_39": strict}

    diag_by_example = {int(eid): g for eid, g in diag.groupby("example_id")}

    # ---- model-free recall + ranking over all examples ----
    per_example = compute_recall_and_ranking(diag_by_example, boxes_by_example)

    recall_bool = [c for c in per_example.columns if c.startswith("has_text_")]
    recall_mean = ["best_text_iou", "best_text_coverage", "best_object_iou",
                   "n_text_overlapping_proposals", "n_object_overlapping_proposals"]
    recall_agg = aggregate_by_subset(per_example, subset_ids, recall_bool, recall_mean, [])

    ranking_bool = [c for c in per_example.columns if c.startswith("text_overlap_at_")] + [
        "selected_overlaps_text", "has_text_overlapping_proposal"]
    ranking_median = ["rank_best_text_proposal", "rank_best_object_proposal", "selected_cic_score",
                      "best_text_proposal_cic_score"]
    ranking_agg = aggregate_by_subset(per_example, subset_ids, ranking_bool, [], ranking_median)

    # ---- model-dependent repair diagnostics over the union subset ----
    union_ids = sorted(set(directional) | set(strict))
    cap = cfg.get("max_model_examples")
    if cap is not None:
        union_ids = union_ids[: int(cap)]

    model_cfg = dict(cfg.get("model", {}))
    device = _device(model_cfg, cfg)
    fake_backend = str(model_cfg.get("backend", "")).lower() == "fake" or str(model_cfg.get("preferred_backend", "")).lower() == "fake"
    status = None
    model_ran = False
    repair_summary: dict[str, Any] = {}
    area_agg = pd.DataFrame()
    dilated_agg = pd.DataFrame()
    ranking_repair = pd.DataFrame()
    contact_paths: list[str] = []

    if not fake_backend:
        status = check_clip_available(
            device=device,
            allow_download=_downloads_allowed(model_cfg),
            preferred_backend=str(model_cfg.get("preferred_backend", "open_clip")),
            model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
            pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
            transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
        )
        if status.available and status.backend in {"open_clip", "transformers"} and status.pretrained and union_ids:
            frames = run_model_diagnostics(
                union_ids, examples_by_id, diag_by_example, status, device,
                int(cfg.get("predict_batch_size", 64)), int(cfg.get("max_dilated_components", 12)),
            )
            ranking_repair = frames["ranking_repair"]
            area_agg = aggregate_area(frames["area"])
            dilated_agg = aggregate_dilated(frames["dilated"])
            repair_summary = aggregate_ranking_repair(ranking_repair)
            contact_paths = write_contact_sheet(frames["contact_records"], out_dir / "contact_sheet",
                                                int(cfg.get("n_contact_examples", 6)))
            model_ran = True
            # persist per-example model frames too
            frames["area"].to_csv(out_dir / "area_normalized_scoring_per_example.csv", index=False)
            frames["dilated"].to_csv(out_dir / "text_dilated_proposal_per_example.csv", index=False)

    diagnosis = diagnose(recall_agg, ranking_agg, repair_summary)

    # ---- write artifacts ----
    per_example.to_csv(out_dir / "proposal_localization_per_example.csv", index=False)
    recall_agg.to_csv(out_dir / "proposal_recall_by_subset.csv", index=False)
    ranking_agg.to_csv(out_dir / "ranking_diagnostic_by_subset.csv", index=False)
    if not ranking_repair.empty:
        ranking_repair.to_csv(out_dir / "ranking_repair_by_example.csv", index=False)
    area_agg.to_csv(out_dir / "area_normalized_scoring_diagnostic.csv", index=False)
    dilated_agg.to_csv(out_dir / "text_dilated_proposal_diagnostic.csv", index=False)

    key_numbers = {
        "experiment": "coco_text_cic_localization_diagnostic",
        "open_world_claim_allowed": False,
        "model_ran": bool(model_ran),
        "backend": (status.backend if status else "not_loaded"),
        "real_pretrained_model_loaded": bool(status.pretrained) if status else False,
        "n_all": len(all_ids),
        "n_directional": len(directional),
        "n_strict": len(strict),
        "n_model_examples": len(union_ids) if model_ran else 0,
        "diagnosis": diagnosis,
        "repair_summary": repair_summary,
        "recall_by_subset": recall_agg.to_dict(orient="records"),
        "ranking_by_subset": ranking_agg.to_dict(orient="records"),
        "area_normalized": area_agg.to_dict(orient="records") if not area_agg.empty else [],
        "text_dilated": dilated_agg.to_dict(orient="records") if not dilated_agg.empty else [],
        "guardrails": {
            "wrote_only_under_output_subdir": True,
            "did_not_modify_full_or_final_report": True,
            "oracle_diagnostics_labeled": ["best_text_iou_proposal", "best_object_iou_proposal", "reward_text_oracle"],
        },
        "contact_sheet": contact_paths,
    }
    (out_dir / "localization_diagnostic_key_numbers.json").write_text(
        json.dumps(key_numbers, indent=2, default=_json_default), encoding="utf-8"
    )
    (out_dir / "localization_diagnostic_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    _write_summary(out_dir, key_numbers, recall_agg, ranking_agg, area_agg, dilated_agg)

    return {
        "key_numbers": str(out_dir / "localization_diagnostic_key_numbers.json"),
        "summary": str(out_dir / "localization_diagnostic_summary.md"),
        "proposal_recall_by_subset": str(out_dir / "proposal_recall_by_subset.csv"),
        "ranking_diagnostic_by_subset": str(out_dir / "ranking_diagnostic_by_subset.csv"),
        "area_normalized_scoring_diagnostic": str(out_dir / "area_normalized_scoring_diagnostic.csv"),
        "text_dilated_proposal_diagnostic": str(out_dir / "text_dilated_proposal_diagnostic.csv"),
        "per_example": str(out_dir / "proposal_localization_per_example.csv"),
        "contact_sheet": contact_paths,
    }


def _fmt(x) -> str:
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def _write_summary(out_dir: Path, key: dict[str, Any], recall_agg, ranking_agg, area_agg, dilated_agg) -> None:
    d = key["diagnosis"]
    rs = key.get("repair_summary", {})
    lines: list[str] = []
    lines += [
        "# COCO-Text Proposal-Localization Diagnostic",
        "",
        "Diagnoses *why* the full COCO-Text CIC gates fail only on selected text-box overlap: ",
        "is it **(A) a proposal-coverage failure** (text-overlapping proposals are not available) ",
        "or **(B) a scoring/ranking failure** (text-overlapping proposals exist but CIC ranks ",
        "object/background regions higher)? Geometry/ranking uses the frozen `coco_text_cic_full` ",
        "proposals; repair uses the real CLIP backend on the verified directional/strict subsets. ",
        "No existing metric, gate or final-report file is modified; `open_world_claim_allowed=False`.",
        "",
        f"- Backend: `{key['backend']}` (real pretrained loaded: `{key['real_pretrained_model_loaded']}`); model ran: `{key['model_ran']}`.",
        f"- N: all={key['n_all']}, directional={key['n_directional']}, strict={key['n_strict']}, model-evaluated union={key['n_model_examples']}.",
        "",
        "## Headline diagnosis",
        "",
        f"- **Primary diagnosis: `{d['primary_diagnosis']}`**",
        f"- Selected text-overlap@1 shortfall vs the {_fmt(d['gate_text_overlap_threshold'])} gate decomposes into "
        f"**ranking_gap={_fmt(d['ranking_gap'])}** (text proposals exist but rank below 1) and "
        f"**coverage_gap={_fmt(d['coverage_gap_vs_threshold'])}** (no text proposal available at all).",
        f"- Coverage caps the gate even with a perfect ranker: {d['coverage_caps_gate_even_with_perfect_ranking']} "
        f"(recall ceiling {_fmt(d['strict_proposal_recall_text_iou_010'])} < threshold {_fmt(d['gate_text_overlap_threshold'])}).",
        f"- Strict proposal recall (best text IoU >= 0.1): {_fmt(d['strict_proposal_recall_text_iou_010'])}",
        f"- Directional proposal recall (best text IoU >= 0.1): {_fmt(d['directional_proposal_recall_text_iou_010'])}",
        f"- Strict text-overlap@1 / @5 / @10: {_fmt(d['strict_text_overlap_at_1'])} / {_fmt(d['strict_text_overlap_at_5'])} / {_fmt(d['strict_text_overlap_at_10'])}",
        f"- Strict median CIC rank of best text-overlapping proposal: {_fmt(d['strict_median_rank_best_text_proposal'])}",
        f"- Best text-overlapping proposal alias repair (ORACLE selection): {_fmt(d['best_text_proposal_alias_repair'])}",
        f"- Selected CIC alias repair: {_fmt(d['selected_cic_alias_repair'])}",
        f"- Best-text-proposal repair beats selected CIC: {d['best_text_beats_selected']}",
        "",
        "## Interpretation (answers to the diagnostic questions)",
        "",
        f"1. **Do open proposals contain text-overlapping regions?** Strict-subset recall at IoU>=0.1 "
        f"is {_fmt(d['strict_proposal_recall_text_iou_010'])}. See `proposal_recall_by_subset.csv` for IoU 0.1/0.3/0.5 and coverage 30/50/80%.",
        f"2. **Does CIC rank those text-overlapping regions highly?** text-overlap@1={_fmt(d['strict_text_overlap_at_1'])} "
        f"but @10={_fmt(d['strict_text_overlap_at_10'])}; median CIC rank of the best text-overlapping proposal is in `ranking_diagnostic_by_subset.csv`.",
        f"3. **Does repairing with the best text-overlapping proposal beat selected CIC?** "
        f"best-text alias repair {_fmt(rs.get('best_text_alias_repair'))} vs selected {_fmt(rs.get('selected_alias_repair'))} "
        f"(this is an ORACLE selection - it consumes ground-truth text geometry and is not deployable).",
        f"4. **Coverage problem or ranking problem?** -> **{d['primary_diagnosis']}** (see logic below).",
        f"5. **Does area normalization help?** See `area_normalized_scoring_diagnostic.csv`; "
        f"`reward_text_oracle` is leakage/oracle-only.",
        f"6. **Does top-k union help?** See top-k rows in `ranking_diagnostic` / key numbers "
        f"(topk text-overlap and target-prob improvement for k=1,3,5,10).",
        "",
        "Diagnosis logic: a *coverage* failure means text-overlapping proposals are largely absent "
        "(low recall). A *ranking* failure means recall is adequate but the top-1 (and small top-k) "
        "rarely land on text because CIC scores object/background regions higher.",
        "",
        "## Proposal recall by subset",
        "",
        _markdown_table(recall_agg) if not recall_agg.empty else "(none)",
        "",
        "## Ranking diagnostic by subset",
        "",
        _markdown_table(ranking_agg) if not ranking_agg.empty else "(none)",
        "",
        "## Area-normalized scoring diagnostic (diagnostic only; `reward_text_oracle` = leakage)",
        "",
        _markdown_table(area_agg) if not area_agg.empty else "(model did not run)",
        "",
        "## Inference-time text-dilated proposal diagnostic (no ground-truth boxes)",
        "",
        _markdown_table(dilated_agg) if not dilated_agg.empty else "(model did not run)",
        "",
        "## Scientific recommendation",
        "",
        _recommendation(d, rs),
        "",
        "## Guardrails",
        "",
        "- Wrote only under this output subdirectory; did not touch `results/coco_text_cic_full/`, "
        "`results/coco_text_cic_triage/`, or `results/final_report/`.",
        "- Did not change any existing gate, metric, or support flag. `open_world_claim_allowed` stays False.",
        "- Best-text-IoU / best-object-IoU proposal selection and `reward_text_oracle` are **oracle / leakage "
        "diagnostics** (they consume ground-truth geometry) and are reported as upper bounds, not deployable methods.",
    ]
    (out_dir / "localization_diagnostic_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _recommendation(d: dict[str, Any], rs: dict[str, Any]) -> str:
    diag = d["primary_diagnosis"]
    ceiling = d.get("coverage_caps_gate_even_with_perfect_ranking")
    if diag == "ranking_failure":
        base = (
            "The low selected text-overlap is **primarily a scoring/ranking** problem: text-overlapping "
            f"proposals are present in the open candidate set (recall {_fmt(d['strict_proposal_recall_text_iou_010'])}) "
            f"but CIC ranks them at median rank {_fmt(d['strict_median_rank_best_text_proposal'])}, so the top-1 "
            f"lands on text only {_fmt(d['strict_text_overlap_at_1'])} of the time (ranking_gap "
            f"{_fmt(d['ranking_gap'])} > coverage_gap {_fmt(d['coverage_gap_vs_threshold'])})."
        )
        if ceiling:
            base += (
                " A secondary **coverage ceiling** also exists: even a perfect ranker could not clear the "
                f"{_fmt(d['gate_text_overlap_threshold'])} gate because proposal recall is below it, so the "
                "proposal generator is a genuine but smaller limitation."
            )
        base += (
            " Because CIC still produces large, real directional repair (target-prob up, text-distractor down) "
            "without localizing onto the annotated text, the honest framing is: **report COCO-Text as directional "
            "repair and treat the text-overlap shortfall as a documented localization limitation**, not evidence "
            "that CIC failed."
        )
    elif diag == "proposal_coverage_failure":
        base = (
            "The low selected text-overlap is primarily a **proposal-coverage** problem: text-overlapping "
            "proposals are rarely available in the open candidate set, so no re-ranking can recover them. "
            "This bounds the claim - directional repair holds, but localization onto scene text is limited by "
            "the proposal generator, which should be reported as a limitation."
        )
    else:
        base = (
            "The evidence is mixed between coverage and ranking; report directional repair with an explicit "
            "localization caveat and include the per-subset recall/ranking tables."
        )
    venue = (
        " Given that the variants tested (area normalization, top-k union, inference-time text-dilated "
        "proposals) are diagnostic rather than headline methods, this is best presented as an **appendix "
        "table plus a one-paragraph limitation** in the main text, not as a new headline result."
    )
    return base + venue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/coco_text_cic_localization_diagnostic.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
