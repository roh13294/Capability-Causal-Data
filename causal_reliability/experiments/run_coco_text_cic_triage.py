from __future__ import annotations

"""Lightweight COCO-Text CIC triage pass (pre-flight for the full proposal CIC).

Experiment name: ``coco_text_cic_triage``.

Scientific goal: decide whether the curated 500-image COCO-Text x COCO-objects
metadata sample (``data/coco_text_cic/metadata.csv``) contains *enough* real,
oracle-repairable, **text-driven** CLIP failures to justify a full proposal-CIC
run. This is a triage / pre-flight pass, NOT the CIC experiment.

For each image we compute, with a real pretrained OpenCLIP backend:

1. The original CLIP prediction and confidence over the allowed COCO vocabulary.
2. The target label probability / rank (alias-aware).
3. A tracked text-distractor label probability / rank (the originally-dominant
   non-target label, or an explicit ``text_distractor_labels`` column if present).
4. An oracle text-box intervention using a few **global, label-free** operators
   (gray fill, expanded gray fill 1.25, Gaussian blur, expanded blur 1.25). The
   best operator per image is an eval-only upper bound on simple text-box repair.
5. Post-oracle prediction, target probability/rank, and text-distractor
   probability/rank.
6. Whether oracle masking improves target probability/rank, decreases the
   text-distractor probability, recovers target top-1/top-3/top-5, and flips the
   target-vs-text pairwise margin toward the target.

This pass **does not run open-proposal CIC**. The oracle text boxes are used only
as eval-only geometry for the upper bound, exactly as in the curated natural-text
Round-1 protocol. The script writes ONLY under
``results/coco_text_cic_triage/`` (or ``cfg['output_subdir']``) and therefore
cannot disturb any final-report headline metric or curated Round-1 artifact.

Gate: ``coco_text_ready_for_full_cic`` becomes ``True`` only with a real
pretrained model (``fake_backend`` false), >= ``min_directional_failures``
directional verified failures, >= ``min_strict_failures`` strict
oracle-repairable failures OR >= ``min_oracle_top5_or_pairwise_recovery``
top-5/pairwise recoveries, and enough clean examples remaining after the
ambiguity filters.
"""

import argparse
import json
import math
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
from causal_reliability.discovery.natural_text_operators import (
    Operator,
    apply_operator,
    default_operators,
)
from causal_reliability.experiments.run_natural_text_open_proposal_cic import (
    PROMPT_TEMPLATE,
    _build_predict_fn,
    _device,
    _downloads_allowed,
    _iou,
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


DEFAULT_OUTPUT_SUBDIR = "coco_text_cic_triage"

# Default thresholds (overridable via config).
DEFAULT_HIGH_CONF_THRESHOLD = 0.7
DEFAULT_PROB_IMPROVE_EPS = 0.01
DEFAULT_MAX_TEXT_OBJECT_IOU = 0.50
DEFAULT_ORACLE_TOP_K = 5
DEFAULT_STRONG_PAIRWISE_DELTA = 0.15
DEFAULT_ORACLE_OPERATORS = (
    "gray_fill",
    "expanded_gray_fill_1.25",
    "gaussian_blur",
    "expanded_blur_1.25",
)
DEFAULT_MIN_DIRECTIONAL_FAILURES = 50
DEFAULT_MIN_STRICT_FAILURES = 30
DEFAULT_MIN_ORACLE_TOP5_OR_PAIRWISE = 50
DEFAULT_MIN_CLEAN_EXAMPLES = 30


# Common COCO category synonyms so alias-aware matching does not under-count
# correctness when the predicted label is a synonym of the target. Only synonyms
# that actually appear in the prompt vocabulary affect the prediction; the rest
# are harmless. Plural forms are added automatically.
DEFAULT_LABEL_ALIASES: dict[str, set[str]] = {
    "airplane": {"aeroplane", "plane", "jet", "aircraft"},
    "motorcycle": {"motorbike"},
    "bicycle": {"bike"},
    "cell phone": {"mobile phone", "cellphone", "phone", "smartphone"},
    "couch": {"sofa"},
    "tv": {"television"},
    "car": {"automobile"},
    "truck": {"lorry"},
    "teddy bear": {"teddy"},
    "hot dog": {"hotdog"},
}


# --------------------------------------------------------------------------- #
# Alias helpers
# --------------------------------------------------------------------------- #
def aliases_for(target: str, extra: set[str] | None = None) -> set[str]:
    """Return the alias set for ``target`` (default synonyms + plural + extras)."""

    target = str(target).strip()
    out = set(DEFAULT_LABEL_ALIASES.get(target, set()))
    if extra:
        out |= {str(e).strip() for e in extra if str(e).strip()}
    if target:
        out.add(target + "s")
    out.discard(target)
    return out


def is_target_label(label: str, target: str, aliases: set[str]) -> bool:
    """True iff ``label`` is the target or one of its aliases."""

    return str(label) == str(target) or str(label) in aliases


# --------------------------------------------------------------------------- #
# Probability / rank helpers
# --------------------------------------------------------------------------- #
def label_rank(probs: np.ndarray, idx_set: list[int]) -> int:
    """1-based best (minimum) rank over ``idx_set`` in descending-prob order."""

    order = np.argsort(-np.asarray(probs, dtype=np.float64), kind="stable")
    rank_of = {int(i): r + 1 for r, i in enumerate(order)}
    return min(rank_of[int(i)] for i in idx_set)


def label_set_prob(probs: np.ndarray, idx_set: list[int]) -> float:
    """Maximum probability over ``idx_set`` (alias-aware target probability)."""

    arr = np.asarray(probs, dtype=np.float64)
    return float(max(arr[int(i)] for i in idx_set))


def pairwise_margin_toward_target(target_prob: float, distractor_prob: float) -> float:
    """Signed target-vs-text margin; positive favours the target."""

    return float(target_prob) - float(distractor_prob)


def pairwise_margin_improved(orig_margin: float, post_margin: float, eps: float = 0.0) -> bool:
    """True iff the post margin moved toward the target relative to the original."""

    return float(post_margin) > float(orig_margin) + float(eps)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _box_area(b) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def max_text_object_iou(text_boxes: list, object_boxes: list) -> float:
    """Maximum IoU between any text box and any object box (0 if either empty)."""

    if not text_boxes or not object_boxes:
        return 0.0
    return max(_iou(t, o) for t in text_boxes for o in object_boxes)


# --------------------------------------------------------------------------- #
# Predicates (pure; tested directly)
# --------------------------------------------------------------------------- #
def passes_ambiguity_filters(
    rec: dict[str, Any],
    *,
    min_target_area_frac: float,
    max_target_area_frac: float,
    max_object_boxes: int,
    max_text_boxes: int,
) -> bool:
    """Basic sanity gate so degenerate / extremely cluttered images are excluded."""

    if int(rec["n_text_boxes"]) < 1:
        return False
    if int(rec["n_text_boxes"]) > int(max_text_boxes):
        return False
    if int(rec["n_object_boxes"]) > int(max_object_boxes):
        return False
    area = float(rec["target_area_frac"])
    if not (float(min_target_area_frac) <= area <= float(max_target_area_frac)):
        return False
    return True


def directional_failure_predicate(
    rec: dict[str, Any],
    *,
    prob_improve_eps: float = DEFAULT_PROB_IMPROVE_EPS,
    max_text_object_iou_threshold: float = DEFAULT_MAX_TEXT_OBJECT_IOU,
) -> bool:
    """Subset A: verified *directional* text-driven failure.

    Criteria (all must hold):
      * original prediction is not target/alias OR text-distractor prob > target prob
      * oracle text-box masking improves target probability
      * oracle text-box masking decreases text-distractor prob OR improves target rank
      * object/text overlap not too high
      * image passes basic ambiguity filters
    """

    eps = float(prob_improve_eps)
    c1 = (not bool(rec["original_is_target"])) or (
        float(rec["distractor_prob_orig"]) > float(rec["target_prob_orig"])
    )
    c2 = float(rec["target_prob_post"]) > float(rec["target_prob_orig"]) + eps
    c3 = (float(rec["distractor_prob_post"]) < float(rec["distractor_prob_orig"]) - eps) or (
        int(rec["target_rank_post"]) < int(rec["target_rank_orig"])
    )
    c4 = float(rec["text_object_iou"]) <= float(max_text_object_iou_threshold)
    c5 = bool(rec["passes_ambiguity"])
    return bool(c1 and c2 and c3 and c4 and c5)


def strict_oracle_repairable_predicate(
    rec: dict[str, Any],
    *,
    is_directional: bool,
    oracle_top_k: int = DEFAULT_ORACLE_TOP_K,
    strong_pairwise_delta: float = DEFAULT_STRONG_PAIRWISE_DELTA,
) -> bool:
    """Subset B: strict oracle-repairable failure.

    Requires subset A, plus either:
      * oracle (alias-aware) top-1 recovers the target, OR
      * oracle top-3/top-k recovers the target with a strong target-vs-text
        pairwise improvement.
    """

    if not is_directional:
        return False
    top1 = bool(rec["post_is_target"])
    top3 = int(rec["target_rank_post"]) <= 3
    topk = int(rec["target_rank_post"]) <= int(oracle_top_k)
    strong = (
        float(rec["pairwise_margin_post"]) - float(rec["pairwise_margin_orig"])
    ) >= float(strong_pairwise_delta)
    return bool(top1 or ((top3 or topk) and strong))


def clean_subset_predicate(geom: dict[str, Any], clean_cfg: dict[str, Any]) -> bool:
    """Cleaner-subset filter operating on geometry features only.

    Prefers exactly one dominant target object, few object/text boxes, a
    reasonably-sized target, text neither too tiny nor too huge, and low
    text/object IoU. Extremely cluttered images are excluded.
    """

    max_obj = int(clean_cfg.get("max_object_boxes", 3))
    max_txt = int(clean_cfg.get("max_text_boxes", 5))
    min_area = float(clean_cfg.get("min_target_area_frac", 0.05))
    max_area = float(clean_cfg.get("max_target_area_frac", 0.80))
    min_text_area = float(clean_cfg.get("min_text_area_frac", 0.003))
    max_text_area = float(clean_cfg.get("max_text_area_frac", 0.50))
    max_iou = float(clean_cfg.get("max_text_object_iou", 0.30))
    require_single = bool(clean_cfg.get("require_single_dominant", True))

    if int(geom["n_object_boxes"]) > max_obj or int(geom["n_object_boxes"]) < 1:
        return False
    n_text = int(geom["n_text_boxes"])
    if n_text > max_txt or n_text < 1:
        return False
    if require_single and int(geom["n_dominant_objects"]) != 1:
        return False
    area = float(geom["target_area_frac"])
    if not (min_area <= area <= max_area):
        return False
    text_area = float(geom["max_text_area_frac"])
    if not (min_text_area <= text_area <= max_text_area):
        return False
    if float(geom["text_object_iou"]) > max_iou:
        return False
    return True


# --------------------------------------------------------------------------- #
# Oracle operator application
# --------------------------------------------------------------------------- #
def operators_by_name(names: list[str]) -> list[Operator]:
    """Resolve operator names against the deterministic operator registry."""

    registry = {op.name: op for op in default_operators()}
    out: list[Operator] = []
    for name in names:
        if name in registry:
            out.append(registry[name])
    return out


def oracle_neutralize(pil: Image.Image, text_boxes: list, op: Operator) -> Image.Image:
    """Apply ``op`` to all text boxes (eval-only oracle text-box intervention)."""

    out, _available = apply_operator(pil, text_boxes, op)
    return out


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def evaluate_coco_text_triage_gate(
    *,
    backend: str,
    pretrained: bool,
    fake_backend: bool,
    n_directional: int,
    n_strict: int,
    n_oracle_top5_or_pairwise: int,
    n_clean: int,
    min_directional_failures: int = DEFAULT_MIN_DIRECTIONAL_FAILURES,
    min_strict_failures: int = DEFAULT_MIN_STRICT_FAILURES,
    min_oracle_top5_or_pairwise_recovery: int = DEFAULT_MIN_ORACLE_TOP5_OR_PAIRWISE,
    min_clean_examples: int = DEFAULT_MIN_CLEAN_EXAMPLES,
) -> tuple[bool, list[str]]:
    """Decide whether the dataset is ready for a full proposal-CIC run.

    Returns ``(ready, failed_reasons)``.
    """

    reasons: list[str] = []
    if backend not in {"open_clip", "transformers"} or not pretrained or fake_backend or backend == "fake":
        reasons.append("real pretrained OpenCLIP/transformers backend did not load (fake backend or unavailable)")
    if int(n_directional) < int(min_directional_failures):
        reasons.append(
            f"directional verified failures {int(n_directional)} < minimum {int(min_directional_failures)}"
        )
    strict_ok = int(n_strict) >= int(min_strict_failures)
    recovery_ok = int(n_oracle_top5_or_pairwise) >= int(min_oracle_top5_or_pairwise_recovery)
    if not (strict_ok or recovery_ok):
        reasons.append(
            f"strict oracle-repairable failures {int(n_strict)} < {int(min_strict_failures)} "
            f"AND oracle top-5/pairwise recoveries {int(n_oracle_top5_or_pairwise)} < "
            f"{int(min_oracle_top5_or_pairwise_recovery)}"
        )
    if int(n_clean) < int(min_clean_examples):
        reasons.append(
            f"clean-subset examples {int(n_clean)} < minimum {int(min_clean_examples)} after ambiguity filtering"
        )
    return (len(reasons) == 0), reasons


# --------------------------------------------------------------------------- #
# Per-example evaluation
# --------------------------------------------------------------------------- #
def _pil_from_example(ex: dict[str, Any]) -> Image.Image:
    return Image.fromarray((np.asarray(ex["image"]).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")


def _evaluate_examples(
    examples: list[dict[str, Any]],
    status: ClipStatus,
    device: str,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    high_conf_threshold = float(cfg.get("high_confidence_threshold", DEFAULT_HIGH_CONF_THRESHOLD))
    prob_eps = float(cfg.get("prob_improve_eps", DEFAULT_PROB_IMPROVE_EPS))
    max_iou = float(cfg.get("max_text_object_iou", DEFAULT_MAX_TEXT_OBJECT_IOU))
    oracle_top_k = int(cfg.get("oracle_top_k", DEFAULT_ORACLE_TOP_K))
    strong_delta = float(cfg.get("strong_pairwise_delta", DEFAULT_STRONG_PAIRWISE_DELTA))
    operator_names = list(cfg.get("oracle_operators", DEFAULT_ORACLE_OPERATORS))
    operators = operators_by_name(operator_names)

    amb = dict(
        min_target_area_frac=float(cfg.get("min_target_area_frac", 0.03)),
        max_target_area_frac=float(cfg.get("max_target_area_frac", 0.90)),
        max_object_boxes=int(cfg.get("max_object_boxes", 8)),
        max_text_boxes=int(cfg.get("max_text_boxes", 12)),
    )
    clean_cfg = dict(cfg.get("clean", {}))
    dominant_frac = float(clean_cfg.get("dominant_object_area_frac", 0.10))

    # One predict_fn per allowed-label vocabulary (all rows share the COCO vocab,
    # so text features are encoded once and reused across every image).
    predict_cache: dict[tuple[str, ...], Any] = {}

    rows: list[dict[str, Any]] = []
    example_records: list[dict[str, Any]] = []

    for ex in examples:
        allowed = list(ex["allowed_clip_labels"])
        key = tuple(allowed)
        predict_fn = predict_cache.get(key)
        if predict_fn is None:
            predict_fn = _build_predict_fn(status, allowed, device)
            predict_cache[key] = predict_fn

        target = str(ex["human_label"])
        target_aliases = aliases_for(target, extra=set(ex.get("target_aliases", [])))
        target_idxs = [i for i, lbl in enumerate(allowed) if is_target_label(lbl, target, target_aliases)]
        if not target_idxs:  # loader guarantees target in allowed; defensive only.
            continue

        pil = _pil_from_example(ex)
        width, height = pil.size
        image_area = float(width * height)
        text_boxes = [tuple(int(v) for v in b) for b in ex.get("text_boxes", [])]
        object_boxes = [tuple(int(v) for v in b) for b in ex.get("object_boxes", [])]

        # Geometry features.
        target_box = object_boxes[0] if object_boxes else (0, 0, width, height)
        target_area_frac = _box_area(target_box) / image_area if image_area > 0 else 0.0
        text_area_fracs = [_box_area(b) / image_area for b in text_boxes] if image_area > 0 else []
        max_text_area_frac = max(text_area_fracs) if text_area_fracs else 0.0
        n_dominant = sum(
            1 for b in object_boxes if (_box_area(b) / image_area if image_area > 0 else 0.0) >= dominant_frac
        )
        txt_obj_iou = max_text_object_iou(text_boxes, object_boxes)

        # Optional explicit text-distractor labels (restricted to allowed, non-target).
        explicit_distractors = [
            d for d in parse_label_list(ex.get("text_distractor_labels", []))
            if d in allowed and not is_target_label(d, target, target_aliases)
        ]
        distractor_present = bool(explicit_distractors)

        # Build the batch: original + one image per operator (all text boxes masked).
        batch = [pil] + [oracle_neutralize(pil, text_boxes, op) for op in operators]
        probs = np.asarray(predict_fn(batch), dtype=np.float64)
        orig_probs = probs[0]

        non_target_idxs = [i for i in range(len(allowed)) if i not in set(target_idxs)]
        # Tracked distractor: best explicit distractor by original prob, else the
        # originally-dominant non-target label (the competing text-driven label).
        if explicit_distractors:
            cand_idxs = [allowed.index(d) for d in explicit_distractors]
        else:
            cand_idxs = non_target_idxs
        distractor_idx = int(max(cand_idxs, key=lambda i: orig_probs[i])) if cand_idxs else int(target_idxs[0])

        orig_pred = int(orig_probs.argmax())
        orig_pred_label = allowed[orig_pred]
        orig_conf = float(orig_probs.max())
        original_is_target = is_target_label(orig_pred_label, target, target_aliases)
        high_conf_failure = bool((not original_is_target) and orig_conf >= high_conf_threshold)

        target_prob_orig = label_set_prob(orig_probs, target_idxs)
        target_rank_orig = label_rank(orig_probs, target_idxs)
        distractor_prob_orig = float(orig_probs[distractor_idx])
        distractor_rank_orig = label_rank(orig_probs, [distractor_idx])
        margin_orig = pairwise_margin_toward_target(target_prob_orig, distractor_prob_orig)

        # Choose the best operator (maximise target prob; tie-break lower rank then
        # higher margin) as the eval-only oracle text-box repair upper bound.
        best = None
        for op_i, op in enumerate(operators):
            op_probs = probs[op_i + 1]
            t_prob = label_set_prob(op_probs, target_idxs)
            t_rank = label_rank(op_probs, target_idxs)
            d_prob = float(op_probs[distractor_idx])
            margin = pairwise_margin_toward_target(t_prob, d_prob)
            score_key = (t_prob, -t_rank, margin)
            if best is None or score_key > best["score_key"]:
                best = {
                    "op": op.name,
                    "probs": op_probs,
                    "target_prob": t_prob,
                    "target_rank": t_rank,
                    "distractor_prob": d_prob,
                    "margin": margin,
                    "score_key": score_key,
                }
        if best is None:  # no operators configured; degrade to original.
            best = {
                "op": "", "probs": orig_probs, "target_prob": target_prob_orig,
                "target_rank": target_rank_orig, "distractor_prob": distractor_prob_orig,
                "margin": margin_orig, "score_key": (target_prob_orig, -target_rank_orig, margin_orig),
            }

        post_probs = best["probs"]
        post_pred = int(post_probs.argmax())
        post_pred_label = allowed[post_pred]
        post_is_target = is_target_label(post_pred_label, target, target_aliases)
        target_prob_post = float(best["target_prob"])
        target_rank_post = int(best["target_rank"])
        distractor_prob_post = float(best["distractor_prob"])
        margin_post = float(best["margin"])

        oracle_improves_target_prob = bool(target_prob_post > target_prob_orig + prob_eps)
        oracle_improves_target_rank = bool(target_rank_post < target_rank_orig)
        oracle_decreases_distractor = bool(distractor_prob_post < distractor_prob_orig - prob_eps)
        oracle_recovers_top1 = bool(post_is_target)
        oracle_recovers_top3 = bool(target_rank_post <= 3)
        oracle_recovers_top5 = bool(target_rank_post <= int(oracle_top_k))
        margin_improved = pairwise_margin_improved(margin_orig, margin_post)
        strong_pairwise = bool((margin_post - margin_orig) >= strong_delta)
        oracle_top5_or_pairwise = bool(oracle_recovers_top5 or strong_pairwise)

        passes_amb = passes_ambiguity_filters(
            {
                "n_text_boxes": len(text_boxes),
                "n_object_boxes": len(object_boxes),
                "target_area_frac": target_area_frac,
            },
            **amb,
        )

        rec = {
            "original_is_target": original_is_target,
            "target_prob_orig": target_prob_orig,
            "target_rank_orig": target_rank_orig,
            "distractor_prob_orig": distractor_prob_orig,
            "target_prob_post": target_prob_post,
            "target_rank_post": target_rank_post,
            "distractor_prob_post": distractor_prob_post,
            "post_is_target": post_is_target,
            "pairwise_margin_orig": margin_orig,
            "pairwise_margin_post": margin_post,
            "text_object_iou": txt_obj_iou,
            "passes_ambiguity": passes_amb,
        }
        is_directional = directional_failure_predicate(
            rec, prob_improve_eps=prob_eps, max_text_object_iou_threshold=max_iou
        )
        is_strict = strict_oracle_repairable_predicate(
            rec, is_directional=is_directional, oracle_top_k=oracle_top_k, strong_pairwise_delta=strong_delta
        )
        geom = {
            "n_object_boxes": len(object_boxes),
            "n_text_boxes": len(text_boxes),
            "n_dominant_objects": n_dominant,
            "target_area_frac": target_area_frac,
            "max_text_area_frac": max_text_area_frac,
            "text_object_iou": txt_obj_iou,
        }
        in_clean = clean_subset_predicate(geom, clean_cfg)

        rows.append(
            {
                "example_id": ex["example_id"],
                "image_path": ex.get("input_image_path", ""),
                "human_label": target,
                "target_label": target,
                "allowed_clip_labels": "|".join(allowed),
                "original_prediction_label": orig_pred_label,
                "original_confidence": orig_conf,
                "original_is_target": original_is_target,
                "original_correct": original_is_target,
                "high_confidence_failure": high_conf_failure,
                "target_prob_orig": target_prob_orig,
                "target_rank_orig": target_rank_orig,
                "distractor_label": allowed[distractor_idx],
                "distractor_present": distractor_present,
                "distractor_prob_orig": distractor_prob_orig,
                "distractor_rank_orig": distractor_rank_orig,
                "best_operator": best["op"],
                "post_prediction_label": post_pred_label,
                "post_is_target": post_is_target,
                "target_prob_post": target_prob_post,
                "target_rank_post": target_rank_post,
                "distractor_prob_post": distractor_prob_post,
                "pairwise_margin_orig": margin_orig,
                "pairwise_margin_post": margin_post,
                "margin_improved": margin_improved,
                "oracle_improves_target_prob": oracle_improves_target_prob,
                "oracle_improves_target_rank": oracle_improves_target_rank,
                "oracle_decreases_distractor": oracle_decreases_distractor,
                "oracle_recovers_top1": oracle_recovers_top1,
                "oracle_recovers_top3": oracle_recovers_top3,
                "oracle_recovers_top5": oracle_recovers_top5,
                "oracle_top5_or_pairwise": oracle_top5_or_pairwise,
                "strong_pairwise_improvement": strong_pairwise,
                "text_object_iou": txt_obj_iou,
                "n_text_boxes": len(text_boxes),
                "n_object_boxes": len(object_boxes),
                "n_dominant_objects": n_dominant,
                "target_area_frac": target_area_frac,
                "max_text_area_frac": max_text_area_frac,
                "passes_ambiguity": passes_amb,
                "is_directional_failure": is_directional,
                "is_strict_oracle_repairable": is_strict,
                "in_clean_subset": in_clean,
            }
        )
        example_records.append(
            {
                "example_id": ex["example_id"],
                "human_label": target,
                "pil": pil,
                "text_boxes": text_boxes,
                "object_boxes": object_boxes,
                "is_directional_failure": is_directional,
                "is_strict_oracle_repairable": is_strict,
                "best_operator": best["op"],
            }
        )

    return pd.DataFrame(rows), example_records


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _rate(series: pd.Series) -> float:
    vals = series.dropna()
    return float(vals.astype(bool).mean()) if len(vals) else float("nan")


def _category_summary(per_image: pd.DataFrame) -> pd.DataFrame:
    if per_image.empty:
        return pd.DataFrame(columns=["category", "n_images", "n_directional", "n_strict", "strict_rate"])
    grp = per_image.groupby("human_label")
    out = pd.DataFrame(
        {
            "n_images": grp.size(),
            "n_directional": grp["is_directional_failure"].sum(),
            "n_strict": grp["is_strict_oracle_repairable"].sum(),
        }
    ).reset_index().rename(columns={"human_label": "category"})
    out["strict_rate"] = out["n_strict"] / out["n_images"].clip(lower=1)
    return out.sort_values(["n_directional", "n_strict"], ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def _contact_sheet(records: list[dict[str, Any]], png: Path, n: int, seed: int) -> int:
    failures = [r for r in records if r["is_directional_failure"]]
    rng = np.random.default_rng(seed)
    if len(failures) > n:
        idx = rng.choice(len(failures), size=n, replace=False)
        sample = [failures[int(i)] for i in sorted(idx)]
    else:
        sample = failures
    ensure_dir(png.parent)
    if not sample:
        fig = plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No verified directional failures to display", ha="center", va="center")
        plt.axis("off")
        fig.savefig(png, dpi=120)
        plt.close(fig)
        return 0
    ncols = min(10, len(sample))
    nrows = int(math.ceil(len(sample) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.8))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_axis_off()
    for ax, rec in zip(axes, sample):
        ax.imshow(rec["pil"])
        for (x0, y0, x1, y1) in rec["text_boxes"]:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#e4572e", lw=1.2))
        for (x0, y0, x1, y1) in rec["object_boxes"][:1]:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#1b9e77", lw=1.0))
        tag = "S" if rec["is_strict_oracle_repairable"] else "D"
        ax.set_title(f"{rec['human_label']} [{tag}]", fontsize=6)
    fig.suptitle(f"Verified directional COCO-Text failures (n={len(sample)})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return len(sample)


def _write_artifacts(
    out_dir: Path,
    cfg: dict[str, Any],
    status: ClipStatus,
    bundle_notes: str,
    per_image: pd.DataFrame,
    directional: pd.DataFrame,
    strict: pd.DataFrame,
    clean: pd.DataFrame,
    key_numbers: dict[str, Any],
    category_summary: pd.DataFrame,
) -> dict[str, str]:
    metrics_csv = out_dir / "coco_text_triage_metrics.csv"
    key_json = out_dir / "coco_text_triage_key_numbers.json"
    directional_csv = out_dir / "coco_text_verified_directional_failures.csv"
    strict_csv = out_dir / "coco_text_verified_oracle_repairable_failures.csv"
    clean_csv = out_dir / "coco_text_clean_subset.csv"
    summary_md = out_dir / "coco_text_triage_summary.md"
    contact_png = out_dir / "verified_failure_contact_sheet.png"

    per_image.to_csv(metrics_csv, index=False)
    directional.to_csv(directional_csv, index=False)
    strict.to_csv(strict_csv, index=False)
    clean.to_csv(clean_csv, index=False)
    key_json.write_text(json.dumps(key_numbers, indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "coco_text_triage_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    ready = bool(key_numbers.get("coco_text_ready_for_full_cic", False))
    reasons = key_numbers.get("gate_failed_reasons", [])
    strongest = key_numbers.get("strongest_categories", [])
    weakest = key_numbers.get("weakest_categories", [])
    inspect = key_numbers.get("examples_to_inspect", [])
    headline = (
        "COCO-Text sample is READY for a full proposal-CIC run"
        if ready
        else "COCO-Text sample is NOT ready for a full proposal-CIC run"
    )
    recommend = (
        "Proceed to the full proposal-CIC run on this metadata sample."
        if ready
        else "Do NOT proceed to CIC yet. Apply stricter filtering (cleaner subset) "
        "and/or rebuild metadata with a larger max_images to recruit more verified failures."
    )

    summary = [
        "# COCO-Text CIC Triage (pre-flight)",
        "",
        "Lightweight triage over the curated COCO-Text x COCO-objects metadata sample. ",
        "Oracle text-box masking uses **global, label-free operators only** (gray fill, ",
        "expanded gray fill 1.25, Gaussian blur, expanded blur 1.25). **No open-proposal ",
        "CIC was run.** Text/object boxes are eval-only geometry for the oracle upper bound.",
        "",
        f"Backend: `{status.backend}`. Model: `{status.model_name or 'n/a'}`. ",
        f"Real pretrained loaded: `{status.pretrained}`. Fake backend: `{key_numbers.get('fake_backend')}`.",
        f"Data: {bundle_notes}.",
        "",
        f"**Result: {headline}.**",
        f"`coco_text_ready_for_full_cic = {ready}`.",
        ("All gate conditions met." if ready else f"Failed reasons: {'; '.join(reasons) if reasons else 'see key numbers'}."),
        "",
        f"**Recommendation:** {recommend}",
        "",
        "## Key numbers",
        "",
        f"- Metadata rows loaded: {key_numbers.get('n_metadata_rows')}",
        f"- Real pretrained model loaded: {key_numbers.get('real_pretrained_model_loaded')}",
        f"- Fake backend: {key_numbers.get('fake_backend')}",
        f"- Original CLIP accuracy (alias-aware): {key_numbers.get('original_clip_accuracy')}",
        f"- High-confidence failure rate: {key_numbers.get('high_confidence_failure_rate')}",
        f"- Directional verified failures: {key_numbers.get('n_directional_failures')}",
        f"- Strict oracle-repairable failures: {key_numbers.get('n_strict_oracle_repairable_failures')}",
        f"- Oracle top-5/pairwise recoveries (over directional): {key_numbers.get('n_oracle_top5_or_pairwise_recovery')}",
        f"- Oracle strict top-1 rate (over directional): {key_numbers.get('oracle_strict_top1_rate')}",
        f"- Oracle strict top-3 rate (over directional): {key_numbers.get('oracle_strict_top3_rate')}",
        f"- Oracle strict top-5 rate (over directional): {key_numbers.get('oracle_strict_top5_rate')}",
        f"- Oracle target-probability improvement rate (over failures): {key_numbers.get('oracle_target_prob_improvement_rate')}",
        f"- Oracle text-distractor decrease rate (over failures): {key_numbers.get('oracle_text_distractor_decrease_rate')}",
        f"- Clean-subset examples: {key_numbers.get('n_clean_subset')}",
        "",
        "## Gate status",
        "",
        f"- `coco_text_ready_for_full_cic`: {ready}",
        f"- Thresholds: directional >= {key_numbers.get('min_directional_failures')}, "
        f"strict >= {key_numbers.get('min_strict_failures')} OR top5/pairwise >= "
        f"{key_numbers.get('min_oracle_top5_or_pairwise_recovery')}, clean >= {key_numbers.get('min_clean_examples')}",
        f"- Failed gate reasons: {reasons or 'none'}",
        "",
        "## Categories",
        "",
        f"- Strongest (most strict-repairable): {', '.join(strongest) if strongest else '(none)'}",
        f"- Weakest (images but no directional failures): {', '.join(weakest) if weakest else '(none)'}",
        "",
        "## Examples to inspect",
        "",
        *([f"- {e}" for e in inspect] or ["- (none)"]),
        "",
        "## Scope guard",
        "",
        "- This is a triage pass. Open-proposal CIC was **not** run.",
        "- Oracle operators are global and label-free; text boxes are eval-only geometry.",
        "- Writes only under this output subdirectory; no final-report metric was touched.",
        "",
        "## Per-category summary",
        "",
        _markdown_table(category_summary if not category_summary.empty else pd.DataFrame([{"category": "(none)"}])),
    ]
    summary_md.write_text("\n".join(summary), encoding="utf-8")
    n_sheet = _contact_sheet(
        key_numbers.get("_records", []), contact_png, int(cfg.get("n_contact_sheet", 50)), int(cfg.get("seed", 0))
    )
    key_numbers["n_contact_sheet_images"] = n_sheet
    key_numbers.pop("_records", None)
    key_json.write_text(json.dumps(key_numbers, indent=2, default=_json_default), encoding="utf-8")

    return {
        "metrics": str(metrics_csv),
        "key_numbers": str(key_json),
        "directional_failures": str(directional_csv),
        "oracle_repairable_failures": str(strict_csv),
        "clean_subset": str(clean_csv),
        "summary": str(summary_md),
        "contact_sheet": str(contact_png),
    }


def _write_unavailable(
    out_dir: Path,
    cfg: dict[str, Any],
    status: ClipStatus,
    bundle_notes: str,
    n_rows: int,
    fake_backend: bool,
) -> dict[str, str]:
    empty = pd.DataFrame()
    key_numbers = {
        "n_metadata_rows": int(n_rows),
        "real_pretrained_model_loaded": bool(status.available and status.pretrained and not fake_backend),
        "fake_backend": bool(fake_backend),
        "backend": status.backend,
        "model_name": status.model_name,
        "original_clip_accuracy": None,
        "high_confidence_failure_rate": None,
        "n_directional_failures": 0,
        "n_strict_oracle_repairable_failures": 0,
        "n_oracle_top5_or_pairwise_recovery": 0,
        "oracle_strict_top1_rate": None,
        "oracle_strict_top3_rate": None,
        "oracle_strict_top5_rate": None,
        "oracle_target_prob_improvement_rate": None,
        "oracle_text_distractor_decrease_rate": None,
        "n_clean_subset": 0,
        "min_directional_failures": int(cfg.get("min_directional_failures", DEFAULT_MIN_DIRECTIONAL_FAILURES)),
        "min_strict_failures": int(cfg.get("min_strict_failures", DEFAULT_MIN_STRICT_FAILURES)),
        "min_oracle_top5_or_pairwise_recovery": int(
            cfg.get("min_oracle_top5_or_pairwise_recovery", DEFAULT_MIN_ORACLE_TOP5_OR_PAIRWISE)
        ),
        "min_clean_examples": int(cfg.get("min_clean_examples", DEFAULT_MIN_CLEAN_EXAMPLES)),
        "coco_text_ready_for_full_cic": False,
        "gate_failed_reasons": [status.error_message or "real pretrained CLIP unavailable or no data"],
        "strongest_categories": [],
        "weakest_categories": [],
        "examples_to_inspect": [],
        "_records": [],
    }
    return _write_artifacts(out_dir, cfg, status, bundle_notes, empty, empty, empty, empty, key_numbers, pd.DataFrame())


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
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
    max_images = cfg.get("max_images")
    if max_images is not None:
        examples = examples[: int(max_images)]
    n_rows = len(examples)

    model_cfg = dict(cfg.get("model", {}))
    device = _device(model_cfg, cfg)
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))

    fake_backend = str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend.lower() == "fake"
    if fake_backend:
        status = ClipStatus(
            False, "fake", "fake_coco_text", pretrained=False, device=device,
            backend_attempted="fake", error_message="fake backend cannot support the COCO-Text triage",
        )
        return _write_unavailable(out_dir, cfg, status, bundle.notes, n_rows, fake_backend=True)

    status = check_clip_available(
        device=device,
        allow_download=_downloads_allowed(model_cfg),
        preferred_backend=requested_backend,
        model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
        pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
        transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return _write_unavailable(out_dir, cfg, status, bundle.notes, n_rows, fake_backend=False)
    if not examples:
        return _write_unavailable(out_dir, cfg, status, bundle.notes, 0, fake_backend=False)

    per_image, example_records = _evaluate_examples(examples, status, device, cfg)

    directional = per_image[per_image["is_directional_failure"].astype(bool)].copy()
    strict = per_image[per_image["is_strict_oracle_repairable"].astype(bool)].copy()
    clean = per_image[per_image["in_clean_subset"].astype(bool)].copy()
    failures = per_image[~per_image["original_is_target"].astype(bool)]

    n_directional = int(len(directional))
    n_strict = int(len(strict))
    n_clean = int(len(clean))
    n_top5_or_pairwise = int(directional["oracle_top5_or_pairwise"].sum()) if n_directional else 0

    original_acc = _rate(per_image["original_correct"])
    high_conf_rate = _rate(per_image["high_confidence_failure"])
    strict_top1 = _rate(directional["oracle_recovers_top1"]) if n_directional else float("nan")
    strict_top3 = _rate(directional["oracle_recovers_top3"]) if n_directional else float("nan")
    strict_top5 = _rate(directional["oracle_recovers_top5"]) if n_directional else float("nan")
    oracle_prob_improve = _rate(failures["oracle_improves_target_prob"]) if len(failures) else float("nan")
    oracle_distractor_decrease = _rate(failures["oracle_decreases_distractor"]) if len(failures) else float("nan")

    cat = _category_summary(per_image)
    strongest = [str(r.category) for r in cat[cat["n_strict"] > 0].head(5).itertuples(index=False)]
    weakest = [
        str(r.category) for r in cat[(cat["n_images"] >= 3) & (cat["n_directional"] == 0)].head(5).itertuples(index=False)
    ]
    inspect = [
        f"{int(r.example_id)}:{r.human_label} (op={r.best_operator})"
        for r in strict.head(8).itertuples(index=False)
    ]

    fake = False
    ready, reasons = evaluate_coco_text_triage_gate(
        backend=status.backend,
        pretrained=bool(status.pretrained),
        fake_backend=fake,
        n_directional=n_directional,
        n_strict=n_strict,
        n_oracle_top5_or_pairwise=n_top5_or_pairwise,
        n_clean=n_clean,
        min_directional_failures=int(cfg.get("min_directional_failures", DEFAULT_MIN_DIRECTIONAL_FAILURES)),
        min_strict_failures=int(cfg.get("min_strict_failures", DEFAULT_MIN_STRICT_FAILURES)),
        min_oracle_top5_or_pairwise_recovery=int(
            cfg.get("min_oracle_top5_or_pairwise_recovery", DEFAULT_MIN_ORACLE_TOP5_OR_PAIRWISE)
        ),
        min_clean_examples=int(cfg.get("min_clean_examples", DEFAULT_MIN_CLEAN_EXAMPLES)),
    )

    key_numbers = {
        "n_metadata_rows": n_rows,
        "real_pretrained_model_loaded": bool(status.pretrained),
        "fake_backend": False,
        "backend": status.backend,
        "model_name": status.model_name,
        "pretrained_loaded": bool(status.pretrained),
        "original_clip_accuracy": original_acc,
        "high_confidence_failure_rate": high_conf_rate,
        "n_original_failures": int(len(failures)),
        "n_directional_failures": n_directional,
        "n_strict_oracle_repairable_failures": n_strict,
        "n_oracle_top5_or_pairwise_recovery": n_top5_or_pairwise,
        "oracle_strict_top1_rate": strict_top1,
        "oracle_strict_top3_rate": strict_top3,
        "oracle_strict_top5_rate": strict_top5,
        "oracle_target_prob_improvement_rate": oracle_prob_improve,
        "oracle_text_distractor_decrease_rate": oracle_distractor_decrease,
        "n_clean_subset": n_clean,
        "min_directional_failures": int(cfg.get("min_directional_failures", DEFAULT_MIN_DIRECTIONAL_FAILURES)),
        "min_strict_failures": int(cfg.get("min_strict_failures", DEFAULT_MIN_STRICT_FAILURES)),
        "min_oracle_top5_or_pairwise_recovery": int(
            cfg.get("min_oracle_top5_or_pairwise_recovery", DEFAULT_MIN_ORACLE_TOP5_OR_PAIRWISE)
        ),
        "min_clean_examples": int(cfg.get("min_clean_examples", DEFAULT_MIN_CLEAN_EXAMPLES)),
        "oracle_operators": list(cfg.get("oracle_operators", DEFAULT_ORACLE_OPERATORS)),
        "ran_open_proposal_cic": False,
        "coco_text_ready_for_full_cic": bool(ready),
        "gate_failed_reasons": reasons,
        "strongest_categories": strongest,
        "weakest_categories": weakest,
        "examples_to_inspect": inspect,
        "_records": example_records,
    }

    return _write_artifacts(out_dir, cfg, status, bundle.notes, per_image, directional, strict, clean, key_numbers, cat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/coco_text_cic_triage.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
