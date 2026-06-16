from __future__ import annotations

"""Full shortcut-agnostic, proposal-based CIC on the COCO-Text dataset-backed benchmark.

Experiment name: ``coco_text_cic_full``.

Scientific goal: having triaged the curated 500-image COCO-Text x COCO-objects
metadata sample (``run_coco_text_cic_triage``), run the *full* open-candidate
intervention search (grid / connected-component / high-contrast / edge-dense
proposals, plus an explicitly-separated OCR/text-box proposal family) on the
verified COCO-Text subsets, and determine whether proposal-based CIC beats matched
random proposals on **real natural images with real scene text**.

Scope (enforced in the summary):
* This is **dataset-backed natural-image validation** of proposal-based CIC, NOT
  full open-world shortcut discovery. ``open_world_claim_allowed`` is always
  ``False``.
* Candidate *scoring* never sees the true label, correctness, oracle repair
  success, subset membership, or any evaluation label. Text/object boxes enter the
  pipeline **only** as candidate geometry for the OCR/object proposal families.
* OCR/text-box proposals are reported as a **separate inference-time proposal
  family**. We always report both (A) open proposals *excluding* OCR/text boxes
  and (B) proposals *including* OCR/text boxes.
* Oracle text-box repair (global, label-free operators over the annotated text
  boxes) remains an **evaluation-only upper bound**.

This script writes ONLY under ``results/coco_text_cic_full/`` (or
``cfg['output_subdir']``) and therefore cannot disturb any final-report headline
metric, the curated natural-text Round-1 artifacts, or the COCO-Text triage
artifacts.

Three subsets are evaluated separately (all share one pass over the union of
example ids, then are aggregated independently):

1. ``all_500``          - all metadata rows
2. ``directional_57``   - verified directional text-driven failures
3. ``strict_39``        - strict oracle-repairable failures (main support eval)

Two support gates are produced:

* ``coco_text_strict_support``      (on the strict subset)
* ``coco_text_directional_support`` (on the directional subset)
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
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

from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.data.natural_text_dataset import load_local_folder_dataset, parse_label_list
from causal_reliability.discovery.cic_region_scoring import neutralize_region, score_region_candidates
from causal_reliability.discovery.natural_text_operators import Operator, apply_operator, default_operators
from causal_reliability.discovery.open_region_proposals import (
    OCR_FAMILY,
    generate_open_region_proposals,
    proposal_family,
)
from causal_reliability.experiments.run_coco_text_cic_triage import (
    aliases_for,
    is_target_label,
    label_rank,
    label_set_prob,
    pairwise_margin_toward_target,
)
from causal_reliability.experiments.run_natural_text_open_proposal_cic import (
    PROMPT_TEMPLATE,
    _device,
    _downloads_allowed,
    _json_default,
    _overlaps_any,
    _pil_to_tensor,
    scoring_is_leakage_free,
)
from causal_reliability.real_models.clip_zero_shot import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED_TAG,
    DEFAULT_TRANSFORMERS_MODEL,
    ClipStatus,
    ClipZeroShotClassifier,
    check_clip_available,
)
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


DEFAULT_OUTPUT_SUBDIR = "coco_text_cic_full"

# Eval-only oracle text-box operators (global, label-free), matching the triage.
DEFAULT_ORACLE_OPERATORS = (
    "gray_fill",
    "expanded_gray_fill_1.25",
    "gaussian_blur",
    "expanded_blur_1.25",
)
# The best global intervention operator identified by the prior intervention sweep.
DEFAULT_BEST_GLOBAL_OPERATOR = "expanded_gray_fill_1.25"

DEFAULT_PROB_IMPROVE_EPS = 0.01
DEFAULT_MAX_CANDIDATES = 48
DEFAULT_ORACLE_TOP_K = 5

# Methods reported in the metrics table. The "headline" CIC method for the gates
# is ``cic_top1_repair_excl_ocr`` (open proposals excluding OCR/text boxes).
METHOD_ORDER = [
    "original_clip_prediction",
    "oracle_text_box_repair",
    "oracle_best_global_op",
    "cic_top1_repair_excl_ocr",
    "cic_top3_repair_excl_ocr",
    "cic_top1_repair_incl_ocr",
    "cic_top3_repair_incl_ocr",
    "cic_top1_best_global_op_excl_ocr",
    "matched_random_proposal_repair",
    "largest_region_repair",
    "ocr_proposal_repair",
]

HEADLINE_CIC_METHOD = "cic_top1_repair_excl_ocr"
RANDOM_METHOD = "matched_random_proposal_repair"
ORACLE_METHOD = "oracle_text_box_repair"

SUBSET_KEYS = ["all_500", "directional_57", "strict_39"]

# Strict-support gate thresholds (overridable via config).
DEFAULT_STRICT_MIN_N = 30
DEFAULT_STRICT_ORACLE_REPAIR = 0.70
DEFAULT_STRICT_ORACLE_RECOVERY = 0.80
DEFAULT_STRICT_CIC_RANDOM_GAP = 0.15
DEFAULT_STRICT_TEXT_OVERLAP = 0.60
DEFAULT_MAX_CONTENT_DROP = 0.15

# Directional-support gate thresholds.
DEFAULT_DIRECTIONAL_MIN_FAILURES = 50
DEFAULT_DIRECTIONAL_ORACLE_PROB_IMPROVE = 0.80
DEFAULT_DIRECTIONAL_CIC_PROB_GAP = 0.10
DEFAULT_DIRECTIONAL_CIC_DISTRACTOR_GAP = 0.10
DEFAULT_DIRECTIONAL_TEXT_OVERLAP = 0.60

PREFERRED_WORDING = [
    "shortcut-agnostic proposal-based CIC",
    "open-candidate intervention search",
    "dataset-backed natural-image validation",
    "does not require a pre-specified shortcut family, but still depends on candidate region proposals",
]
FORBIDDEN_WORDING = [
    "fully open-world shortcut discovery",
    "solves shortcut discovery",
    "general robustness",
    "universal repair",
    "deployment-ready",
]


# --------------------------------------------------------------------------- #
# Prediction helper (chunked to bound memory on large candidate batches)
# --------------------------------------------------------------------------- #
def _build_chunked_predict_fn(status: ClipStatus, allowed_labels: list[str], device: str, batch_size: int):
    prompts = [PROMPT_TEMPLATE.format(label=name) for name in allowed_labels]
    classifier = ClipZeroShotClassifier(status, allowed_labels, prompts=prompts, device=device)

    def predict_fn(images: list[Image.Image]) -> np.ndarray:
        out: list[np.ndarray] = []
        bs = max(1, int(batch_size))
        for start in range(0, len(images), bs):
            chunk = images[start : start + bs]
            res = classifier.predict(_pil_to_tensor(chunk))
            out.append(np.asarray(res["probabilities"].detach().cpu().numpy(), dtype=np.float64))
        return np.concatenate(out, axis=0) if out else np.zeros((0, len(allowed_labels)), dtype=np.float64)

    return predict_fn


# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #
def proposal_separation_is_reported() -> bool:
    """True iff OCR-included and OCR-excluded methods are both reported.

    Guards the requirement that OCR/text-box proposals are an explicit, separately
    reported inference-time family rather than silently folded into the open
    proposals.
    """

    has_excl = any(m.endswith("excl_ocr") for m in METHOD_ORDER)
    has_incl = any(m.endswith("incl_ocr") for m in METHOD_ORDER)
    return bool(has_excl and has_incl)


# --------------------------------------------------------------------------- #
# Oracle text-box repair (eval-only upper bound)
# --------------------------------------------------------------------------- #
def _operators_by_name(names: list[str]) -> list[Operator]:
    registry = {op.name: op for op in default_operators()}
    return [registry[n] for n in names if n in registry]


def _best_oracle_probs(
    predict_fn,
    pil: Image.Image,
    text_boxes: list,
    operators: list[Operator],
    target_idxs: list[int],
    distractor_idx: int,
) -> tuple[np.ndarray | None, str]:
    """Return ``(best_probs, op_name)`` over oracle operators, or ``(None, "")``.

    Best is chosen to maximise alias-aware target probability (tie-break lower
    rank, then larger target-vs-text margin). This mirrors the triage oracle and
    is an eval-only upper bound.
    """

    if not text_boxes or not operators:
        return None, ""
    images = [apply_operator(pil, text_boxes, op)[0] for op in operators]
    probs = np.asarray(predict_fn(images), dtype=np.float64)
    best = None
    best_op = ""
    for i, op in enumerate(operators):
        row = probs[i]
        t_prob = label_set_prob(row, target_idxs)
        t_rank = label_rank(row, target_idxs)
        d_prob = float(row[distractor_idx])
        key = (t_prob, -t_rank, t_prob - d_prob)
        if best is None or key > best[0]:
            best = (key, row)
            best_op = op.name
    return (best[1] if best is not None else None), best_op


# --------------------------------------------------------------------------- #
# Per-example evaluation
# --------------------------------------------------------------------------- #
def _select_matched_random(scores: list, top1, field: str = "area_fraction"):
    randoms = [s for s in scores if s.proposal_type == "random_patch_control"]
    if not randoms:
        return None
    target = float(getattr(top1, field, 0.0)) if top1 is not None else 0.0
    return min(randoms, key=lambda s: abs(float(getattr(s, field, 0.0)) - target))


def _derive_method_row(
    method: str,
    probs: np.ndarray | None,
    *,
    allowed: list[str],
    target: str,
    aliases: set[str],
    target_idxs: list[int],
    distractor_idx: int,
    orig: dict[str, float],
    oracle: bool = False,
    prob_eps: float = DEFAULT_PROB_IMPROVE_EPS,
) -> dict[str, Any]:
    """Build one (example, method) metric row from a post-intervention prob vector."""

    available = probs is not None
    if not available:
        return {
            "method": method,
            "method_available": False,
            "oracle_upper_bound": bool(oracle),
            "post_pred_label": "",
            "strict_correct": np.nan,
            "alias_correct": np.nan,
            "target_prob_orig": orig["target_prob"],
            "target_prob_post": np.nan,
            "target_rank_orig": orig["target_rank"],
            "target_rank_post": np.nan,
            "distractor_prob_orig": orig["distractor_prob"],
            "distractor_prob_post": np.nan,
            "pairwise_margin_orig": orig["pairwise_margin"],
            "pairwise_margin_post": np.nan,
            "target_prob_improved": np.nan,
            "target_prob_gain": np.nan,
            "target_rank_improved": np.nan,
            "distractor_prob_decreased": np.nan,
            "pairwise_recovered": np.nan,
            "recovers_top3": np.nan,
            "recovers_top5": np.nan,
        }
    probs = np.asarray(probs, dtype=np.float64)
    pred = int(probs.argmax())
    pred_label = allowed[pred]
    t_prob = label_set_prob(probs, target_idxs)
    t_rank = label_rank(probs, target_idxs)
    d_prob = float(probs[distractor_idx])
    margin = pairwise_margin_toward_target(t_prob, d_prob)
    target_prob_improved = bool(t_prob > orig["target_prob"] + prob_eps)
    rank_improved = bool(t_rank < orig["target_rank"])
    distractor_decreased = bool(d_prob < orig["distractor_prob"] - prob_eps)
    pairwise_recovered = bool(margin > orig["pairwise_margin"] and t_prob > d_prob)
    return {
        "method": method,
        "method_available": True,
        "oracle_upper_bound": bool(oracle),
        "post_pred_label": pred_label,
        "strict_correct": bool(pred_label == target),
        "alias_correct": bool(is_target_label(pred_label, target, aliases)),
        "target_prob_orig": orig["target_prob"],
        "target_prob_post": t_prob,
        "target_rank_orig": orig["target_rank"],
        "target_rank_post": t_rank,
        "distractor_prob_orig": orig["distractor_prob"],
        "distractor_prob_post": d_prob,
        "pairwise_margin_orig": orig["pairwise_margin"],
        "pairwise_margin_post": margin,
        "target_prob_improved": target_prob_improved,
        "target_prob_gain": float(t_prob - orig["target_prob"]),
        "target_rank_improved": rank_improved,
        "distractor_prob_decreased": distractor_decreased,
        "pairwise_recovered": pairwise_recovered,
        "recovers_top3": bool(t_rank <= 3),
        "recovers_top5": bool(t_rank <= DEFAULT_ORACLE_TOP_K),
    }


def _evaluate_examples(
    examples: list[dict[str, Any]],
    status: ClipStatus,
    device: str,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    seed = int(cfg.get("seed", 0))
    max_candidates = int(cfg.get("max_candidates", DEFAULT_MAX_CANDIDATES))
    grid_scales = cfg.get("grid_scales")
    prob_eps = float(cfg.get("prob_improve_eps", DEFAULT_PROB_IMPROVE_EPS))
    batch_size = int(cfg.get("predict_batch_size", 64))
    high_conf_threshold = float(cfg.get("high_confidence_threshold", 0.7))
    operators = _operators_by_name(list(cfg.get("oracle_operators", DEFAULT_ORACLE_OPERATORS)))
    best_global_name = str(cfg.get("best_global_operator", DEFAULT_BEST_GLOBAL_OPERATOR))
    best_global_ops = _operators_by_name([best_global_name])
    best_global_op = best_global_ops[0] if best_global_ops else None

    predict_cache: dict[tuple[str, ...], Any] = {}

    method_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    example_records: list[dict[str, Any]] = []

    for ex in examples:
        allowed = list(ex["allowed_clip_labels"])
        key = tuple(allowed)
        predict_fn = predict_cache.get(key)
        if predict_fn is None:
            predict_fn = _build_chunked_predict_fn(status, allowed, device, batch_size)
            predict_cache[key] = predict_fn

        target = str(ex["human_label"])
        aliases = aliases_for(target, extra=set(ex.get("target_aliases", [])))
        target_idxs = [i for i, lbl in enumerate(allowed) if is_target_label(lbl, target, aliases)]
        if not target_idxs:
            continue

        pil = Image.fromarray((np.asarray(ex["image"]).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")
        text_boxes = [tuple(int(v) for v in b) for b in ex.get("text_boxes", [])]
        object_boxes = [tuple(int(v) for v in b) for b in ex.get("object_boxes", [])]

        # Open candidate set (OCR family included as geometry only). Scoring is
        # done once over the full set; OCR-excluded results are derived by
        # filtering, since each candidate's score is computed independently.
        proposals = generate_open_region_proposals(
            pil,
            text_boxes=text_boxes,
            object_boxes=object_boxes,
            seed=seed + int(ex["example_id"]),
            max_candidates=max_candidates,
            grid_scales=grid_scales,
            enable_object_box_family=False,
        )
        scores, original_probs = score_region_candidates(pil, proposals, predict_fn)
        original_probs = np.asarray(original_probs, dtype=np.float64)

        # Original alias-aware target / tracked-distractor stats.
        orig_pred = int(original_probs.argmax())
        orig_pred_label = allowed[orig_pred]
        orig_conf = float(original_probs.max())
        original_strict = bool(orig_pred_label == target)
        original_alias = bool(is_target_label(orig_pred_label, target, aliases))
        high_conf_failure = bool((not original_alias) and orig_conf >= high_conf_threshold)

        non_target_idxs = [i for i in range(len(allowed)) if i not in set(target_idxs)]
        explicit_distractors = [
            d for d in parse_label_list(ex.get("text_distractor_labels", []))
            if d in allowed and not is_target_label(d, target, aliases)
        ]
        if explicit_distractors:
            cand_idxs = [allowed.index(d) for d in explicit_distractors]
        else:
            cand_idxs = non_target_idxs
        distractor_idx = int(max(cand_idxs, key=lambda i: original_probs[i])) if cand_idxs else int(target_idxs[0])

        orig = {
            "target_prob": label_set_prob(original_probs, target_idxs),
            "target_rank": label_rank(original_probs, target_idxs),
            "distractor_prob": float(original_probs[distractor_idx]),
            "pairwise_margin": pairwise_margin_toward_target(
                label_set_prob(original_probs, target_idxs), float(original_probs[distractor_idx])
            ),
        }

        excl_scores = [s for s in scores if proposal_family(s.proposal_type) != OCR_FAMILY]
        incl_scores = list(scores)
        ocr_scores = [s for s in scores if proposal_family(s.proposal_type) == OCR_FAMILY]

        top1_excl = excl_scores[0] if excl_scores else None
        top1_incl = incl_scores[0] if incl_scores else None

        def probs_for_box(box) -> np.ndarray:
            return predict_fn([neutralize_region(pil, tuple(int(v) for v in box))])[0]

        def consensus_probs(sel: list) -> np.ndarray | None:
            if not sel:
                return None
            imgs = [neutralize_region(pil, s.bbox) for s in sel]
            return np.asarray(predict_fn(imgs), dtype=np.float64).mean(axis=0)

        # Method prob vectors.
        method_probs: dict[str, tuple[np.ndarray | None, bool]] = {}
        method_probs["original_clip_prediction"] = (original_probs, False)

        oracle_probs, oracle_op = _best_oracle_probs(predict_fn, pil, text_boxes, operators, target_idxs, distractor_idx)
        method_probs["oracle_text_box_repair"] = (oracle_probs, True)

        if best_global_op is not None and text_boxes:
            bg_probs = predict_fn([apply_operator(pil, text_boxes, best_global_op)[0]])[0]
        else:
            bg_probs = None
        method_probs["oracle_best_global_op"] = (bg_probs, True)

        method_probs["cic_top1_repair_excl_ocr"] = (
            probs_for_box(top1_excl.bbox) if top1_excl else None, False
        )
        method_probs["cic_top3_repair_excl_ocr"] = (consensus_probs(excl_scores[:3]), False)
        method_probs["cic_top1_repair_incl_ocr"] = (
            probs_for_box(top1_incl.bbox) if top1_incl else None, False
        )
        method_probs["cic_top3_repair_incl_ocr"] = (consensus_probs(incl_scores[:3]), False)

        if best_global_op is not None and top1_excl is not None:
            bg_cic = apply_operator(pil, [top1_excl.bbox], best_global_op)[0]
            method_probs["cic_top1_best_global_op_excl_ocr"] = (predict_fn([bg_cic])[0], False)
        else:
            method_probs["cic_top1_best_global_op_excl_ocr"] = (None, False)

        rand = _select_matched_random(scores, top1_excl, "area_fraction")
        method_probs["matched_random_proposal_repair"] = (
            probs_for_box(rand.bbox) if rand else None, False
        )

        largest = max(scores, key=lambda s: s.area_fraction) if scores else None
        method_probs["largest_region_repair"] = (
            probs_for_box(largest.bbox) if largest else None, False
        )

        method_probs["ocr_proposal_repair"] = (
            probs_for_box(ocr_scores[0].bbox) if ocr_scores else None, False
        )

        for method in METHOD_ORDER:
            probs, oracle = method_probs.get(method, (None, False))
            row = _derive_method_row(
                method, probs,
                allowed=allowed, target=target, aliases=aliases,
                target_idxs=target_idxs, distractor_idx=distractor_idx,
                orig=orig, oracle=oracle, prob_eps=prob_eps,
            )
            row.update(
                {
                    "example_id": int(ex["example_id"]),
                    "human_label": target,
                    "original_strict_correct": original_strict,
                    "original_alias_correct": original_alias,
                    "high_confidence_failure": high_conf_failure,
                    "oracle_operator": oracle_op if method == "oracle_text_box_repair" else "",
                }
            )
            method_rows.append(row)

        # Selection diagnostics for the two CIC top-1 variants.
        for label, top1 in (("excl_ocr", top1_excl), ("incl_ocr", top1_incl)):
            if top1 is None:
                continue
            selection_rows.append(
                {
                    "example_id": int(ex["example_id"]),
                    "human_label": target,
                    "variant": label,
                    "selected_proposal_type": top1.proposal_type,
                    "selected_family": proposal_family(top1.proposal_type),
                    "selected_bbox": json.dumps([int(v) for v in top1.bbox]),
                    "selected_area_fraction": float(top1.area_fraction),
                    "overlaps_text_box": _overlaps_any(top1.bbox, text_boxes),
                    "overlaps_object_box": _overlaps_any(top1.bbox, object_boxes),
                }
            )

        for rank, s in enumerate(scores, start=1):
            diag_rows.append(
                {
                    "example_id": int(ex["example_id"]),
                    "rank": rank,
                    "candidate_id": s.candidate_id,
                    "proposal_type": s.proposal_type,
                    "proposal_family": proposal_family(s.proposal_type),
                    "bbox": json.dumps([int(v) for v in s.bbox]),
                    "score": float(s.score),
                    "area_fraction": float(s.area_fraction),
                    "prediction_flip_indicator": float(s.prediction_flip_indicator),
                    "js_divergence": float(s.js_divergence),
                    "overlaps_text_box": _overlaps_any(s.bbox, text_boxes),
                    "overlaps_object_box": _overlaps_any(s.bbox, object_boxes),
                }
            )

        example_records.append(
            {
                "example_id": int(ex["example_id"]),
                "human_label": target,
                "pil": pil,
                "text_boxes": text_boxes,
                "object_boxes": object_boxes,
                "top1_excl_box": top1_excl.bbox if top1_excl else None,
                "rand_box": rand.bbox if rand else None,
                "oracle_operator": oracle_op,
                "original_alias_correct": original_alias,
                "cic_excl_alias_correct": bool(
                    is_target_label(allowed[int(np.asarray(method_probs["cic_top1_repair_excl_ocr"][0]).argmax())], target, aliases)
                ) if method_probs["cic_top1_repair_excl_ocr"][0] is not None else None,
            }
        )

    return (
        pd.DataFrame(method_rows),
        pd.DataFrame(selection_rows),
        pd.DataFrame(diag_rows),
        example_records,
    )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _rate(series: pd.Series) -> float:
    vals = series.dropna()
    return float(vals.astype(bool).mean()) if len(vals) else float("nan")


def _median(series: pd.Series) -> float:
    vals = series.dropna()
    return float(vals.astype(float).median()) if len(vals) else float("nan")


def _aggregate_subset(method_rows: pd.DataFrame, selection_rows: pd.DataFrame, subset: str, example_ids: list[int], status: ClipStatus) -> pd.DataFrame:
    ids = set(int(i) for i in example_ids)
    sub = method_rows[method_rows["example_id"].isin(ids)]
    sel = selection_rows[selection_rows["example_id"].isin(ids)] if not selection_rows.empty else selection_rows
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        df = sub[sub["method"] == method]
        avail = df[df["method_available"].astype(bool)] if not df.empty else df
        # Selection metrics only apply to the CIC top-1 variants.
        if method == "cic_top1_repair_excl_ocr":
            sel_v = sel[sel["variant"] == "excl_ocr"] if not sel.empty else sel
        elif method == "cic_top1_repair_incl_ocr":
            sel_v = sel[sel["variant"] == "incl_ocr"] if not sel.empty else sel
        else:
            sel_v = sel.iloc[0:0] if not sel.empty else sel
        rows.append(
            {
                "subset": subset,
                "method": method,
                "backend": status.backend,
                "model_name": status.model_name,
                "pretrained_loaded": bool(status.pretrained),
                "oracle_upper_bound": bool(df["oracle_upper_bound"].any()) if not df.empty else False,
                "n": int(len(avail)),
                "accuracy_strict": _rate(avail["strict_correct"]) if not avail.empty else float("nan"),
                "accuracy_alias": _rate(avail["alias_correct"]) if not avail.empty else float("nan"),
                "recovers_top3": _rate(avail["recovers_top3"]) if not avail.empty else float("nan"),
                "recovers_top5": _rate(avail["recovers_top5"]) if not avail.empty else float("nan"),
                "pairwise_recovery": _rate(avail["pairwise_recovered"]) if not avail.empty else float("nan"),
                "target_prob_improvement_rate": _rate(avail["target_prob_improved"]) if not avail.empty else float("nan"),
                "median_target_prob_gain": _median(avail["target_prob_gain"]) if not avail.empty else float("nan"),
                "target_rank_improvement_rate": _rate(avail["target_rank_improved"]) if not avail.empty else float("nan"),
                "text_distractor_decrease_rate": _rate(avail["distractor_prob_decreased"]) if not avail.empty else float("nan"),
                "selected_text_overlap_rate": _rate(sel_v["overlaps_text_box"]) if (sel_v is not None and not sel_v.empty) else float("nan"),
                "selected_object_overlap_rate": _rate(sel_v["overlaps_object_box"]) if (sel_v is not None and not sel_v.empty) else float("nan"),
                "selected_area_fraction": (float(sel_v["selected_area_fraction"].astype(float).mean()) if (sel_v is not None and not sel_v.empty) else float("nan")),
            }
        )
    return pd.DataFrame(rows)


def _metric(metrics: pd.DataFrame, subset: str, method: str, column: str) -> float:
    row = metrics[(metrics["subset"] == subset) & (metrics["method"] == method)]
    if row.empty:
        return float("nan")
    return float(row[column].iloc[0])


def _content_preservation(method_rows: pd.DataFrame, clean_ids: list[int]) -> tuple[float, float | None]:
    """Among clean & originally-strict-correct images, fraction still strict-correct after CIC top-1 (excl)."""

    if not clean_ids:
        return float("nan"), None
    ids = set(int(i) for i in clean_ids)
    orig = method_rows[(method_rows["method"] == "original_clip_prediction") & (method_rows["example_id"].isin(ids))]
    cic = method_rows[(method_rows["method"] == HEADLINE_CIC_METHOD) & (method_rows["example_id"].isin(ids))]
    orig = orig.set_index("example_id")
    cic = cic.set_index("example_id")
    correct_ids = [i for i in orig.index if bool(orig.loc[i, "original_strict_correct"])]
    correct_ids = [i for i in correct_ids if i in cic.index and bool(cic.loc[i, "method_available"])]
    if not correct_ids:
        return float("nan"), None
    preserved = float(np.mean([bool(cic.loc[i, "strict_correct"]) for i in correct_ids]))
    return preserved, float(1.0 - preserved)


# --------------------------------------------------------------------------- #
# Support gates
# --------------------------------------------------------------------------- #
def evaluate_strict_support_gate(
    *,
    backend: str,
    pretrained: bool,
    fake_backend: bool,
    n: int,
    oracle_strict_repair: float,
    oracle_alias_repair: float,
    oracle_top3_recovery: float,
    oracle_top5_recovery: float,
    oracle_pairwise_recovery: float,
    cic_strict_repair: float,
    cic_alias_repair: float,
    random_strict_repair: float,
    random_alias_repair: float,
    cic_pairwise_recovery: float,
    random_pairwise_recovery: float,
    cic_text_overlap_rate: float,
    content_preservation_drop: float | None,
    content_preservation_documented: bool,
    no_oracle_leakage: bool,
    open_world_claim_allowed: bool,
    min_n: int = DEFAULT_STRICT_MIN_N,
    min_oracle_repair: float = DEFAULT_STRICT_ORACLE_REPAIR,
    min_oracle_recovery: float = DEFAULT_STRICT_ORACLE_RECOVERY,
    min_cic_random_gap: float = DEFAULT_STRICT_CIC_RANDOM_GAP,
    min_text_overlap: float = DEFAULT_STRICT_TEXT_OVERLAP,
    max_content_drop: float = DEFAULT_MAX_CONTENT_DROP,
) -> tuple[bool, list[str]]:
    """Decide ``coco_text_strict_support`` on the strict oracle-repairable subset."""

    reasons: list[str] = []
    if backend not in {"open_clip", "transformers"} or not pretrained or fake_backend or backend == "fake":
        reasons.append("real pretrained OpenCLIP/transformers backend did not load (fake backend or unavailable)")
    if int(n) < int(min_n):
        reasons.append(f"n {int(n)} < minimum {int(min_n)}")

    def _ge(x, t):
        return bool(np.isfinite(x) and float(x) >= float(t))

    oracle_repair_ok = _ge(oracle_strict_repair, min_oracle_repair) or _ge(oracle_alias_repair, min_oracle_repair)
    oracle_recovery_ok = (
        _ge(oracle_top3_recovery, min_oracle_recovery)
        or _ge(oracle_top5_recovery, min_oracle_recovery)
        or _ge(oracle_pairwise_recovery, min_oracle_recovery)
    )
    if not (oracle_repair_ok or oracle_recovery_ok):
        reasons.append(
            f"oracle repair/recovery insufficient: strict={oracle_strict_repair}, alias={oracle_alias_repair}, "
            f"top3={oracle_top3_recovery}, top5={oracle_top5_recovery}, pairwise={oracle_pairwise_recovery}"
        )

    strict_gap = (float(cic_strict_repair) - float(random_strict_repair)) if (np.isfinite(cic_strict_repair) and np.isfinite(random_strict_repair)) else float("-inf")
    alias_gap = (float(cic_alias_repair) - float(random_alias_repair)) if (np.isfinite(cic_alias_repair) and np.isfinite(random_alias_repair)) else float("-inf")
    pairwise_gap = (float(cic_pairwise_recovery) - float(random_pairwise_recovery)) if (np.isfinite(cic_pairwise_recovery) and np.isfinite(random_pairwise_recovery)) else float("-inf")
    repair_beats = (strict_gap >= float(min_cic_random_gap)) or (alias_gap >= float(min_cic_random_gap))
    pairwise_beats = pairwise_gap >= float(min_cic_random_gap)
    if not (repair_beats or pairwise_beats):
        reasons.append(
            f"CIC does not beat matched random by >= {float(min_cic_random_gap):.2f} "
            f"(strict_gap={strict_gap:.3f}, alias_gap={alias_gap:.3f}, pairwise_gap={pairwise_gap:.3f})"
        )
    if not _ge(cic_text_overlap_rate, min_text_overlap):
        reasons.append(f"CIC selected text-overlap rate {cic_text_overlap_rate} < {float(min_text_overlap):.2f}")
    if content_preservation_drop is not None and np.isfinite(content_preservation_drop):
        if float(content_preservation_drop) > float(max_content_drop) and not content_preservation_documented:
            reasons.append(
                f"content-preservation drop {float(content_preservation_drop):.3f} > {float(max_content_drop):.2f} and not documented"
            )
    if not no_oracle_leakage:
        reasons.append("oracle leakage check failed: scoring/proposal rule exposes forbidden parameters")
    if bool(open_world_claim_allowed):
        reasons.append("open_world_claim_allowed must remain False")
    return (len(reasons) == 0), reasons


def evaluate_directional_support_gate(
    *,
    backend: str,
    pretrained: bool,
    fake_backend: bool,
    n_verified_failures: int,
    oracle_target_prob_improvement: float,
    cic_target_prob_improvement: float,
    random_target_prob_improvement: float,
    cic_text_distractor_decrease: float,
    random_text_distractor_decrease: float,
    cic_text_overlap_rate: float,
    no_oracle_leakage: bool,
    min_failures: int = DEFAULT_DIRECTIONAL_MIN_FAILURES,
    min_oracle_prob_improve: float = DEFAULT_DIRECTIONAL_ORACLE_PROB_IMPROVE,
    min_cic_prob_gap: float = DEFAULT_DIRECTIONAL_CIC_PROB_GAP,
    min_cic_distractor_gap: float = DEFAULT_DIRECTIONAL_CIC_DISTRACTOR_GAP,
    min_text_overlap: float = DEFAULT_DIRECTIONAL_TEXT_OVERLAP,
) -> tuple[bool, list[str]]:
    """Decide ``coco_text_directional_support`` on the directional subset."""

    reasons: list[str] = []
    if backend not in {"open_clip", "transformers"} or not pretrained or fake_backend or backend == "fake":
        reasons.append("real pretrained OpenCLIP/transformers backend did not load (fake backend or unavailable)")
    if int(n_verified_failures) < int(min_failures):
        reasons.append(f"verified failures {int(n_verified_failures)} < minimum {int(min_failures)}")

    def _ge(x, t):
        return bool(np.isfinite(x) and float(x) >= float(t))

    if not _ge(oracle_target_prob_improvement, min_oracle_prob_improve):
        reasons.append(f"oracle target-prob improvement {oracle_target_prob_improvement} < {float(min_oracle_prob_improve):.2f}")
    prob_gap = (float(cic_target_prob_improvement) - float(random_target_prob_improvement)) if (np.isfinite(cic_target_prob_improvement) and np.isfinite(random_target_prob_improvement)) else float("-inf")
    if prob_gap < float(min_cic_prob_gap):
        reasons.append(f"CIC target-prob improvement does not beat random by >= {float(min_cic_prob_gap):.2f} (gap={prob_gap:.3f})")
    distractor_gap = (float(cic_text_distractor_decrease) - float(random_text_distractor_decrease)) if (np.isfinite(cic_text_distractor_decrease) and np.isfinite(random_text_distractor_decrease)) else float("-inf")
    if distractor_gap < float(min_cic_distractor_gap):
        reasons.append(f"CIC text-distractor decrease does not beat random by >= {float(min_cic_distractor_gap):.2f} (gap={distractor_gap:.3f})")
    if not _ge(cic_text_overlap_rate, min_text_overlap):
        reasons.append(f"CIC selected text-overlap rate {cic_text_overlap_rate} < {float(min_text_overlap):.2f}")
    if not no_oracle_leakage:
        reasons.append("oracle leakage check failed: scoring/proposal rule exposes forbidden parameters")
    return (len(reasons) == 0), reasons


# --------------------------------------------------------------------------- #
# Subset id loading
# --------------------------------------------------------------------------- #
def _read_ids(csv_path: Path) -> list[int]:
    if not csv_path.exists():
        return []
    frame = pd.read_csv(csv_path)
    if "example_id" not in frame.columns:
        return []
    return [int(v) for v in frame["example_id"].tolist()]


def load_subset_ids(cfg: dict[str, Any], all_ids: list[int]) -> dict[str, list[int]]:
    """Resolve the three subset id lists from the triage artifacts."""

    triage_cfg = dict(cfg.get("triage", {}))
    triage_dir = Path(triage_cfg.get("dir", "results/coco_text_cic_triage"))
    directional = _read_ids(triage_dir / triage_cfg.get("directional_csv", "coco_text_verified_directional_failures.csv"))
    strict = _read_ids(triage_dir / triage_cfg.get("strict_csv", "coco_text_verified_oracle_repairable_failures.csv"))
    present = set(int(i) for i in all_ids)
    return {
        "all_500": [int(i) for i in all_ids],
        "directional_57": [i for i in directional if i in present],
        "strict_39": [i for i in strict if i in present],
    }


def load_clean_ids(cfg: dict[str, Any], all_ids: list[int]) -> list[int]:
    triage_cfg = dict(cfg.get("triage", {}))
    triage_dir = Path(triage_cfg.get("dir", "results/coco_text_cic_triage"))
    clean = _read_ids(triage_dir / triage_cfg.get("clean_csv", "coco_text_clean_subset.csv"))
    present = set(int(i) for i in all_ids)
    return [i for i in clean if i in present]


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def _plot_summary(metrics: pd.DataFrame, png: Path) -> None:
    needed = {
        "accuracy_strict", "accuracy_alias", "target_prob_improvement_rate", "text_distractor_decrease_rate",
    }
    if metrics.empty or not needed.issubset(set(metrics.columns)):
        fig = plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No eligible COCO-Text CIC metrics to plot", ha="center", va="center")
        plt.axis("off")
        fig.savefig(png, dpi=120)
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    plot_methods = [
        "original_clip_prediction",
        "matched_random_proposal_repair",
        "cic_top1_repair_excl_ocr",
        "cic_top1_repair_incl_ocr",
        "oracle_text_box_repair",
    ]
    strict = metrics[metrics["subset"] == "strict_39"]
    take = strict[strict["method"].isin(plot_methods)].set_index("method").reindex(plot_methods)
    ax = axes[0]
    if not take["accuracy_alias"].dropna().empty:
        x = np.arange(len(plot_methods))
        ax.bar(x - 0.2, take["accuracy_strict"].values, width=0.4, label="strict", color="#4c78a8")
        ax.bar(x + 0.2, take["accuracy_alias"].values, width=0.4, label="alias-aware", color="#f58518")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_repair", "").replace("_", "\n") for m in plot_methods], fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("repair accuracy")
        ax.set_title("Strict subset (n=39): repair accuracy")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no strict-subset metrics", ha="center", va="center")
        ax.axis("off")

    directional = metrics[metrics["subset"] == "directional_57"]
    dir_methods = ["oracle_text_box_repair", "cic_top1_repair_excl_ocr", "matched_random_proposal_repair"]
    dtake = directional[directional["method"].isin(dir_methods)].set_index("method").reindex(dir_methods)
    ax = axes[1]
    if not dtake["target_prob_improvement_rate"].dropna().empty:
        x = np.arange(len(dir_methods))
        ax.bar(x - 0.2, dtake["target_prob_improvement_rate"].values, width=0.4, label="target-prob improve", color="#54a24b")
        ax.bar(x + 0.2, dtake["text_distractor_decrease_rate"].values, width=0.4, label="text-distractor decrease", color="#e45756")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_repair", "").replace("_", "\n") for m in dir_methods], fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("rate")
        ax.set_title("Directional subset (n=57): directional effects")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no directional-subset metrics", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def _write_examples(records: list[dict[str, Any]], out_dir: Path, cfg: dict[str, Any], subset_ids: dict[str, list[int]]) -> list[str]:
    ensure_dir(out_dir)
    n_each = int(cfg.get("n_example_visualizations", 4))
    strict_ids = set(subset_ids.get("strict_39", []))
    by_id = {r["example_id"]: r for r in records}

    # Categorize strict examples.
    successes, oracle_only, cic_failures = [], [], []
    for eid in strict_ids:
        r = by_id.get(eid)
        if r is None or r["top1_excl_box"] is None:
            continue
        cic_ok = bool(r.get("cic_excl_alias_correct"))
        if cic_ok:
            successes.append(r)
        else:
            oracle_only.append(r)  # strict subset is oracle-repairable by construction
            cic_failures.append(r)

    def render(r, tag) -> str | None:
        if r["top1_excl_box"] is None:
            return None
        pil = r["pil"]
        cic_neutral = neutralize_region(pil, r["top1_excl_box"])
        rand_neutral = neutralize_region(pil, r["rand_box"]) if r["rand_box"] else pil
        fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9))
        axes[0].imshow(pil)
        axes[0].set_title(f"original\n{r['human_label']} (correct={r['original_alias_correct']})", fontsize=7)
        for (x0, y0, x1, y1) in r["text_boxes"]:
            axes[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#e4572e", lw=1.0))
        axes[1].imshow(cic_neutral)
        x0, y0, x1, y1 = r["top1_excl_box"]
        axes[1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#1b9e77", lw=2))
        axes[1].set_title(f"CIC top-1 (excl OCR)\ncic_ok={r.get('cic_excl_alias_correct')}", fontsize=7)
        axes[2].imshow(rand_neutral)
        if r["rand_box"]:
            x0, y0, x1, y1 = r["rand_box"]
            axes[2].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#9467bd", lw=2))
        axes[2].set_title("matched random", fontsize=7)
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(f"[{tag}] example {r['example_id']} (oracle op={r['oracle_operator']})", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        path = out_dir / f"{tag}_{r['example_id']}_before_after.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return str(path)

    paths: list[str] = []
    for tag, group in (("success", successes), ("cic_failure_oracle_only", cic_failures)):
        for r in group[:n_each]:
            p = render(r, tag)
            if p:
                paths.append(p)
    return paths


def _write_artifacts(
    out_dir: Path,
    cfg: dict[str, Any],
    status: ClipStatus,
    bundle_notes: str,
    method_rows: pd.DataFrame,
    selection_rows: pd.DataFrame,
    diagnostics: pd.DataFrame,
    metrics: pd.DataFrame,
    key_numbers: dict[str, Any],
    example_paths: list[str],
) -> dict[str, str]:
    metrics_csv = out_dir / "coco_text_full_metrics.csv"
    key_json = out_dir / "coco_text_full_key_numbers.json"
    summary_md = out_dir / "coco_text_full_summary.md"
    per_example_csv = out_dir / "coco_text_full_per_example.csv"
    diag_csv = out_dir / "coco_text_full_proposal_diagnostics.csv"
    directional_csv = out_dir / "coco_text_full_directional_metrics.csv"
    plot_png = out_dir / "coco_text_full_plots.png"

    metrics.to_csv(metrics_csv, index=False)
    method_rows.to_csv(per_example_csv, index=False)
    diagnostics.to_csv(diag_csv, index=False)
    metrics[metrics["subset"] == "directional_57"].to_csv(directional_csv, index=False)
    selection_rows.to_csv(out_dir / "coco_text_full_selection.csv", index=False)
    key_json.write_text(json.dumps(key_numbers, indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "coco_text_full_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    _plot_summary(metrics, plot_png)

    strict_gate = key_numbers.get("coco_text_strict_support")
    dir_gate = key_numbers.get("coco_text_directional_support")
    subsets = key_numbers.get("subsets", {})

    def subset_block(name: str) -> list[str]:
        s = subsets.get(name, {})
        return [
            f"### {name} (n={s.get('n')})",
            "",
            f"- Original strict / alias-aware accuracy: {s.get('original_accuracy_strict')} / {s.get('original_accuracy_alias')}",
            f"- Oracle strict / alias-aware repair: {s.get('oracle_repair_strict')} / {s.get('oracle_repair_alias')}",
            f"- Oracle top-3 / top-5 / pairwise recovery: {s.get('oracle_top3_recovery')} / {s.get('oracle_top5_recovery')} / {s.get('oracle_pairwise_recovery')}",
            f"- CIC (excl OCR) strict / alias top-1: {s.get('cic_excl_strict_top1')} / {s.get('cic_excl_alias_top1')}",
            f"- CIC (excl OCR) strict / alias top-3: {s.get('cic_excl_strict_top3')} / {s.get('cic_excl_alias_top3')}",
            f"- CIC (incl OCR) strict / alias top-1: {s.get('cic_incl_strict_top1')} / {s.get('cic_incl_alias_top1')}",
            f"- CIC top-5 recovery (excl OCR): {s.get('cic_excl_top5_recovery')}",
            f"- CIC pairwise target-vs-text recovery (excl OCR): {s.get('cic_excl_pairwise_recovery')}",
            f"- Matched-random / largest-region / OCR-proposal repair (alias): {s.get('random_repair_alias')} / {s.get('largest_repair_alias')} / {s.get('ocr_repair_alias')}",
            f"- CIC - random gap (alias): {s.get('cic_random_alias_gap')}",
            f"- Target-prob improvement rate (CIC / random / oracle): {s.get('cic_target_prob_improvement')} / {s.get('random_target_prob_improvement')} / {s.get('oracle_target_prob_improvement')}",
            f"- Median target-prob gain (CIC excl OCR): {s.get('cic_median_target_prob_gain')}",
            f"- Target-rank improvement rate (CIC excl OCR): {s.get('cic_target_rank_improvement')}",
            f"- Text-distractor decrease rate (CIC / random): {s.get('cic_text_distractor_decrease')} / {s.get('random_text_distractor_decrease')}",
            f"- Selected text-box / object-box overlap rate (CIC excl OCR): {s.get('cic_selected_text_overlap')} / {s.get('cic_selected_object_overlap')}",
            f"- Selected area fraction (CIC excl OCR): {s.get('cic_selected_area_fraction')}",
            "",
        ]

    summary = [
        "# COCO-Text Full Proposal-Based CIC (dataset-backed natural-image validation)",
        "",
        "Full **shortcut-agnostic proposal-based CIC** / **open-candidate intervention search** over the ",
        "verified COCO-Text subsets. Candidate scoring saw only pixels, proposal geometry, and model ",
        "predictions — never the true label, correctness, oracle repair success, or subset membership. ",
        "OCR/text-box proposals are reported as a **separate inference-time family** (we always report ",
        "both *excluding* and *including* OCR boxes). Oracle text-box repair is an **eval-only upper bound**. ",
        "This is **dataset-backed natural-image validation**, NOT full open-world shortcut discovery.",
        "",
        f"Backend: `{status.backend}`. Model: `{status.model_name or 'n/a'}`. Real pretrained loaded: `{status.pretrained}`. ",
        f"Fake backend: `{key_numbers.get('fake_backend')}`. Data: {bundle_notes}.",
        "",
        "## Support gates",
        "",
        f"- `coco_text_strict_support` (strict subset, n={subsets.get('strict_39', {}).get('n')}): **{strict_gate}**",
        f"  - reasons: {key_numbers.get('strict_support_reasons') or 'none'}",
        f"- `coco_text_directional_support` (directional subset, n={subsets.get('directional_57', {}).get('n')}): **{dir_gate}**",
        f"  - reasons: {key_numbers.get('directional_support_reasons') or 'none'}",
        f"- `open_world_claim_allowed`: {key_numbers.get('open_world_claim_allowed')}",
        f"- OCR-included materially improves over OCR-excluded: {key_numbers.get('ocr_inclusion_materially_helps')}",
        f"- Open proposals excluding OCR are sufficient (strict gate via excl-OCR CIC): {key_numbers.get('excl_ocr_sufficient')}",
        "",
        "## Per-subset metrics",
        "",
        *subset_block("all_500"),
        *subset_block("directional_57"),
        *subset_block("strict_39"),
        "## Content preservation (clean subset)",
        "",
        f"- Clean-subset content-preservation rate: {key_numbers.get('content_preservation_rate')}",
        f"- Clean-subset content-preservation drop: {key_numbers.get('content_preservation_drop')}",
        f"- Documented: {key_numbers.get('content_preservation_documented')}",
        "",
        "## Leakage / scope guard",
        "",
        f"- No-oracle-leakage (scoring/proposal signatures clean): {key_numbers.get('no_oracle_leakage')}",
        f"- OCR-included vs OCR-excluded reported separately: {key_numbers.get('proposal_separation_reported')}",
        "- Oracle operators are global and label-free; text boxes are eval-only geometry for the upper bound.",
        "- Writes only under this output subdirectory; no final-report / Round-1 / triage artifact was touched.",
        "",
        "## Full metrics table",
        "",
        _markdown_table(metrics),
        "",
        "## Best examples to inspect",
        "",
        *([f"- {e}" for e in key_numbers.get("examples_to_inspect", [])] or ["- (none)"]),
    ]
    summary_md.write_text("\n".join(summary), encoding="utf-8")

    return {
        "metrics": str(metrics_csv),
        "key_numbers": str(key_json),
        "summary": str(summary_md),
        "per_example": str(per_example_csv),
        "proposal_diagnostics": str(diag_csv),
        "directional_metrics": str(directional_csv),
        "plots": str(plot_png),
        "examples": [str(p) for p in example_paths],
    }


def _write_unavailable(out_dir: Path, cfg: dict[str, Any], status: ClipStatus, bundle_notes: str, n_rows: int, fake_backend: bool) -> dict[str, str]:
    empty = pd.DataFrame()
    metrics = pd.DataFrame(
        [{"subset": "n/a", "method": "unavailable", "backend": status.backend, "model_name": status.model_name,
          "pretrained_loaded": bool(status.pretrained), "oracle_upper_bound": False, "n": 0,
          "accuracy_strict": float("nan"), "accuracy_alias": float("nan")}]
    )
    key_numbers = {
        "n_metadata_rows": int(n_rows),
        "real_pretrained_model_loaded": bool(status.available and status.pretrained and not fake_backend),
        "fake_backend": bool(fake_backend),
        "backend": status.backend,
        "model_name": status.model_name,
        "subsets": {},
        "coco_text_strict_support": False,
        "strict_support_reasons": [status.error_message or "real pretrained CLIP unavailable or no data"],
        "coco_text_directional_support": False,
        "directional_support_reasons": [status.error_message or "real pretrained CLIP unavailable or no data"],
        "no_oracle_leakage": scoring_is_leakage_free(),
        "proposal_separation_reported": proposal_separation_is_reported(),
        "content_preservation_rate": None,
        "content_preservation_drop": None,
        "content_preservation_documented": bool(cfg.get("content_preservation_documented", False)),
        "ocr_inclusion_materially_helps": None,
        "excl_ocr_sufficient": False,
        "open_world_claim_allowed": False,
        "examples_to_inspect": [],
        "preferred_wording": PREFERRED_WORDING,
        "forbidden_wording": FORBIDDEN_WORDING,
    }
    ensure_dir(out_dir / "examples")
    return _write_artifacts(out_dir, cfg, status, bundle_notes, empty, empty, empty, metrics, key_numbers, [])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _subset_summary(metrics: pd.DataFrame, method_rows: pd.DataFrame, subset: str) -> dict[str, Any]:
    g = lambda method, col: _metric(metrics, subset, method, col)
    n = int(g("original_clip_prediction", "n"))
    cic_alias = g(HEADLINE_CIC_METHOD, "accuracy_alias")
    rand_alias = g(RANDOM_METHOD, "accuracy_alias")
    return {
        "n": n,
        "original_accuracy_strict": g("original_clip_prediction", "accuracy_strict"),
        "original_accuracy_alias": g("original_clip_prediction", "accuracy_alias"),
        "oracle_repair_strict": g(ORACLE_METHOD, "accuracy_strict"),
        "oracle_repair_alias": g(ORACLE_METHOD, "accuracy_alias"),
        "oracle_top3_recovery": g(ORACLE_METHOD, "recovers_top3"),
        "oracle_top5_recovery": g(ORACLE_METHOD, "recovers_top5"),
        "oracle_pairwise_recovery": g(ORACLE_METHOD, "pairwise_recovery"),
        "oracle_target_prob_improvement": g(ORACLE_METHOD, "target_prob_improvement_rate"),
        "cic_excl_strict_top1": g("cic_top1_repair_excl_ocr", "accuracy_strict"),
        "cic_excl_alias_top1": cic_alias,
        "cic_excl_strict_top3": g("cic_top3_repair_excl_ocr", "accuracy_strict"),
        "cic_excl_alias_top3": g("cic_top3_repair_excl_ocr", "accuracy_alias"),
        "cic_incl_strict_top1": g("cic_top1_repair_incl_ocr", "accuracy_strict"),
        "cic_incl_alias_top1": g("cic_top1_repair_incl_ocr", "accuracy_alias"),
        "cic_excl_top5_recovery": g("cic_top1_repair_excl_ocr", "recovers_top5"),
        "cic_excl_pairwise_recovery": g("cic_top1_repair_excl_ocr", "pairwise_recovery"),
        "random_repair_alias": rand_alias,
        "largest_repair_alias": g("largest_region_repair", "accuracy_alias"),
        "ocr_repair_alias": g("ocr_proposal_repair", "accuracy_alias"),
        "cic_random_alias_gap": (float(cic_alias) - float(rand_alias)) if (np.isfinite(cic_alias) and np.isfinite(rand_alias)) else float("nan"),
        "cic_target_prob_improvement": g("cic_top1_repair_excl_ocr", "target_prob_improvement_rate"),
        "random_target_prob_improvement": g(RANDOM_METHOD, "target_prob_improvement_rate"),
        "cic_median_target_prob_gain": g("cic_top1_repair_excl_ocr", "median_target_prob_gain"),
        "cic_target_rank_improvement": g("cic_top1_repair_excl_ocr", "target_rank_improvement_rate"),
        "cic_text_distractor_decrease": g("cic_top1_repair_excl_ocr", "text_distractor_decrease_rate"),
        "random_text_distractor_decrease": g(RANDOM_METHOD, "text_distractor_decrease_rate"),
        "cic_selected_text_overlap": g("cic_top1_repair_excl_ocr", "selected_text_overlap_rate"),
        "cic_selected_object_overlap": g("cic_top1_repair_excl_ocr", "selected_object_overlap_rate"),
        "cic_selected_area_fraction": g("cic_top1_repair_excl_ocr", "selected_area_fraction"),
    }


def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / cfg.get("output_subdir", DEFAULT_OUTPUT_SUBDIR))

    data_cfg = dict(cfg.get("data", {}))
    image_size = data_cfg.get("image_size", 224)
    image_size = int(image_size) if image_size else None
    root = data_cfg.get("root", "data/coco_text_cic")
    metadata_csv = data_cfg.get("metadata_csv") or str(Path(root) / "metadata.csv")
    bundle = load_local_folder_dataset(
        root=root, metadata_csv=metadata_csv, image_size=image_size, split=str(data_cfg.get("split", "test"))
    )
    examples = bundle.examples
    all_ids = [int(ex["example_id"]) for ex in examples]
    subset_ids = load_subset_ids(cfg, all_ids)
    clean_ids = load_clean_ids(cfg, all_ids)

    model_cfg = dict(cfg.get("model", {}))
    device = _device(model_cfg, cfg)
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))
    fake_backend = str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend.lower() == "fake"
    if fake_backend:
        status = ClipStatus(False, "fake", "fake_coco_text_full", pretrained=False, device=device,
                            backend_attempted="fake", error_message="fake backend cannot support the COCO-Text full CIC claim")
        return _write_unavailable(out_dir, cfg, status, bundle.notes, len(examples), fake_backend=True)

    status = check_clip_available(
        device=device,
        allow_download=_downloads_allowed(model_cfg),
        preferred_backend=requested_backend,
        model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
        pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
        transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return _write_unavailable(out_dir, cfg, status, bundle.notes, len(examples), fake_backend=False)
    if not examples:
        return _write_unavailable(out_dir, cfg, status, bundle.notes, 0, fake_backend=False)

    # Evaluation set: the union of every requested subset plus the clean ids
    # (so content-preservation can always be computed). Default: all examples.
    eval_subsets = list(cfg.get("eval_subsets", SUBSET_KEYS))
    eval_ids: set[int] = set()
    for key in eval_subsets:
        eval_ids.update(subset_ids.get(key, []))
    eval_ids.update(clean_ids)
    if "all_500" in eval_subsets:
        eval_ids.update(all_ids)
    eval_examples = [ex for ex in examples if int(ex["example_id"]) in eval_ids]

    method_rows, selection_rows, diagnostics, example_records = _evaluate_examples(eval_examples, status, device, cfg)

    metrics_frames = []
    for key in SUBSET_KEYS:
        ids = subset_ids.get(key, [])
        if key == "all_500":
            ids = all_ids
        metrics_frames.append(_aggregate_subset(method_rows, selection_rows, key, ids, status))
    metrics = pd.concat(metrics_frames, ignore_index=True)

    subsets_summary = {key: _subset_summary(metrics, method_rows, key) for key in SUBSET_KEYS}

    content_rate, content_drop = _content_preservation(method_rows, clean_ids)
    content_documented = bool(cfg.get("content_preservation_documented", False))
    no_leak = scoring_is_leakage_free()
    sep_reported = proposal_separation_is_reported()

    strict = subsets_summary["strict_39"]
    directional = subsets_summary["directional_57"]

    strict_support, strict_reasons = evaluate_strict_support_gate(
        backend=status.backend,
        pretrained=bool(status.pretrained),
        fake_backend=False,
        n=strict["n"],
        oracle_strict_repair=strict["oracle_repair_strict"],
        oracle_alias_repair=strict["oracle_repair_alias"],
        oracle_top3_recovery=strict["oracle_top3_recovery"],
        oracle_top5_recovery=strict["oracle_top5_recovery"],
        oracle_pairwise_recovery=strict["oracle_pairwise_recovery"],
        cic_strict_repair=strict["cic_excl_strict_top1"],
        cic_alias_repair=strict["cic_excl_alias_top1"],
        random_strict_repair=_metric(metrics, "strict_39", RANDOM_METHOD, "accuracy_strict"),
        random_alias_repair=strict["random_repair_alias"],
        cic_pairwise_recovery=strict["cic_excl_pairwise_recovery"],
        random_pairwise_recovery=_metric(metrics, "strict_39", RANDOM_METHOD, "pairwise_recovery"),
        cic_text_overlap_rate=strict["cic_selected_text_overlap"],
        content_preservation_drop=content_drop,
        content_preservation_documented=content_documented,
        no_oracle_leakage=no_leak,
        open_world_claim_allowed=False,
        min_n=int(cfg.get("strict_min_n", DEFAULT_STRICT_MIN_N)),
        min_oracle_repair=float(cfg.get("strict_min_oracle_repair", DEFAULT_STRICT_ORACLE_REPAIR)),
        min_oracle_recovery=float(cfg.get("strict_min_oracle_recovery", DEFAULT_STRICT_ORACLE_RECOVERY)),
        min_cic_random_gap=float(cfg.get("strict_min_cic_random_gap", DEFAULT_STRICT_CIC_RANDOM_GAP)),
        min_text_overlap=float(cfg.get("strict_min_text_overlap", DEFAULT_STRICT_TEXT_OVERLAP)),
        max_content_drop=float(cfg.get("max_content_drop", DEFAULT_MAX_CONTENT_DROP)),
    )

    directional_support, directional_reasons = evaluate_directional_support_gate(
        backend=status.backend,
        pretrained=bool(status.pretrained),
        fake_backend=False,
        n_verified_failures=directional["n"],
        oracle_target_prob_improvement=directional["oracle_target_prob_improvement"],
        cic_target_prob_improvement=directional["cic_target_prob_improvement"],
        random_target_prob_improvement=directional["random_target_prob_improvement"],
        cic_text_distractor_decrease=directional["cic_text_distractor_decrease"],
        random_text_distractor_decrease=directional["random_text_distractor_decrease"],
        cic_text_overlap_rate=directional["cic_selected_text_overlap"],
        no_oracle_leakage=no_leak,
        min_failures=int(cfg.get("directional_min_failures", DEFAULT_DIRECTIONAL_MIN_FAILURES)),
        min_oracle_prob_improve=float(cfg.get("directional_min_oracle_prob_improve", DEFAULT_DIRECTIONAL_ORACLE_PROB_IMPROVE)),
        min_cic_prob_gap=float(cfg.get("directional_min_cic_prob_gap", DEFAULT_DIRECTIONAL_CIC_PROB_GAP)),
        min_cic_distractor_gap=float(cfg.get("directional_min_cic_distractor_gap", DEFAULT_DIRECTIONAL_CIC_DISTRACTOR_GAP)),
        min_text_overlap=float(cfg.get("directional_min_text_overlap", DEFAULT_DIRECTIONAL_TEXT_OVERLAP)),
    )

    # Does including OCR boxes materially help over excluding them (strict subset)?
    excl_alias = strict["cic_excl_alias_top1"]
    incl_alias = strict["cic_incl_alias_top1"]
    ocr_help_delta = (float(incl_alias) - float(excl_alias)) if (np.isfinite(incl_alias) and np.isfinite(excl_alias)) else float("nan")
    ocr_materially_helps = bool(np.isfinite(ocr_help_delta) and ocr_help_delta >= float(cfg.get("ocr_material_help_delta", 0.05)))

    inspect = []
    for r in example_records:
        if int(r["example_id"]) in set(subset_ids.get("strict_39", [])) and r.get("cic_excl_alias_correct"):
            inspect.append(f"{r['example_id']}:{r['human_label']} (cic-success, oracle op={r['oracle_operator']})")
        if len(inspect) >= 8:
            break

    key_numbers = {
        "n_metadata_rows": len(examples),
        "n_evaluated": int(len(eval_examples)),
        "real_pretrained_model_loaded": bool(status.pretrained),
        "fake_backend": False,
        "backend": status.backend,
        "model_name": status.model_name,
        "subsets": subsets_summary,
        "coco_text_strict_support": bool(strict_support),
        "strict_support_reasons": strict_reasons,
        "coco_text_directional_support": bool(directional_support),
        "directional_support_reasons": directional_reasons,
        "content_preservation_rate": content_rate,
        "content_preservation_drop": content_drop,
        "content_preservation_documented": content_documented,
        "no_oracle_leakage": bool(no_leak),
        "proposal_separation_reported": bool(sep_reported),
        "ocr_inclusion_help_delta_strict": ocr_help_delta,
        "ocr_inclusion_materially_helps": ocr_materially_helps,
        "excl_ocr_sufficient": bool(strict_support),
        "open_world_claim_allowed": False,
        "best_global_operator": str(cfg.get("best_global_operator", DEFAULT_BEST_GLOBAL_OPERATOR)),
        "oracle_operators": list(cfg.get("oracle_operators", DEFAULT_ORACLE_OPERATORS)),
        "examples_to_inspect": inspect,
        "preferred_wording": PREFERRED_WORDING,
        "forbidden_wording": FORBIDDEN_WORDING,
    }

    example_paths = _write_examples(example_records, out_dir / "examples", cfg, subset_ids)
    return _write_artifacts(out_dir, cfg, status, bundle.notes, method_rows, selection_rows, diagnostics, metrics, key_numbers, example_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/coco_text_cic_full.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
