from __future__ import annotations

"""Natural-text intervention/operator sweep (diagnostic).

Experiment name: ``natural_text_intervention_sweep``.

Scientific question: is strict natural-text repair limited by the **intervention
operator / masking strategy** rather than by CIC proposal *selection*? We hold the
candidate geometry fixed (the human-annotated text/logo boxes for the oracle
ceiling, and the existing CIC top-1 proposal for the method) and sweep a panel of
deterministic neutralization operators (gray/black/white fill, local mean/median,
Gaussian blur, pixelation, border-colour fill, OpenCV Telea inpaint, and several
box-expansion variants). For each operator we report strict and directional repair
on the **verified text-driven failures**.

This experiment is **purely diagnostic**:

* It writes ONLY to ``results/natural_text_intervention_sweep/``.
* It NEVER updates the headline / final-report metrics.
* ``open_world_claim_allowed`` stays ``False``.
* It never converts directional evidence into a positive strict support claim; the
  strict gate is reported but not flipped.
* Operator *selection* for any reported gate is global / label-free only — no
  per-example operator is chosen using the true label or correctness.

It reuses the exact leakage-free proposal + scoring path of
``run_natural_text_verified_failure_eval`` so the CIC top-1 region per failure is
identical to that of the headline run.
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

from causal_reliability.data.natural_text_dataset import (
    load_verified_natural_text_dataset,
    save_example_images,
)
from causal_reliability.discovery.cic_region_scoring import score_region_candidates
from causal_reliability.discovery.natural_text_operators import (
    Operator,
    apply_operator,
    cv2_available,
    default_operators,
    operator_boxes,
)
from causal_reliability.discovery.open_region_proposals import (
    generate_open_region_proposals,
    proposal_family,
)
from causal_reliability.experiments.run_natural_text_open_proposal_cic import (
    FORBIDDEN_WORDING,
    PREFERRED_WORDING,
    _build_predict_fn,
    _device,
    _downloads_allowed,
    _json_default,
    _overlaps_any,
    _select_matched_random,
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


DEFAULT_OUTPUT_SUBDIR = "natural_text_intervention_sweep"
EPS = 1e-6
ALWAYS_OPEN_WORLD_CLAIM_ALLOWED = False
# Pre-declared, label-free criterion for choosing a single GLOBAL operator (used
# only to answer the "could a strict gate pass" diagnostic question). It uses an
# aggregate directional signal, never per-example correctness.
GLOBAL_OPERATOR_CRITERION = "oracle target-probability improvement rate (global, aggregate)"
RANDOM_REPAIR_BASELINE = 1.0 / 3.0  # ~chance for a typical small candidate label set
STRICT_GATE_CIC_RANDOM_GAP = 0.15
ORACLE_CEILING_MODERATE = 0.50
ORACLE_CEILING_HIGH = 0.70


# --------------------------------------------------------------------------- #
# Probability + image metrics
# --------------------------------------------------------------------------- #
def _rank_of(probs: np.ndarray, idx: int) -> int:
    """0-based rank of class ``idx`` (0 == top)."""

    order = np.argsort(-np.asarray(probs, dtype=np.float64), kind="stable")
    return int(np.where(order == idx)[0][0])


def _prob_metrics(
    p_before: np.ndarray,
    p_after: np.ndarray,
    label: int,
    distractor_idxs: list[int],
    non_distractor_idxs: list[int],
) -> dict[str, Any]:
    pb = np.asarray(p_before, dtype=np.float64)
    pa = np.asarray(p_after, dtype=np.float64)
    pred_after = int(pa.argmax())
    order_after = np.argsort(-pa, kind="stable")
    distr_before = float(pb[distractor_idxs].sum()) if distractor_idxs else 0.0
    distr_after = float(pa[distractor_idxs].sum()) if distractor_idxs else 0.0
    return {
        "strict_repair": bool(pred_after == label),
        "alias_repair": bool(pred_after in set(non_distractor_idxs)),
        "top3_recovery": bool(label in order_after[:3]),
        "top5_recovery": bool(label in order_after[:5]),
        "target_prob_improved": bool(pa[label] > pb[label] + EPS),
        "target_prob_gain": float(pa[label] - pb[label]),
        "target_rank_improved": bool(_rank_of(pa, label) < _rank_of(pb, label)),
        "distractor_prob_decreased": bool(distr_after < distr_before - EPS),
        "distractor_prob_delta": float(distr_after - distr_before),
    }


def _image_damage(
    before: Image.Image,
    after: Image.Image,
    object_boxes: list[tuple[int, int, int, int]],
) -> dict[str, float]:
    a = np.asarray(before.convert("RGB")).astype(np.float64) / 255.0
    b = np.asarray(after.convert("RGB")).astype(np.float64) / 255.0
    diff = np.abs(a - b)
    changed = (diff.max(axis=2) > (1.0 / 255.0))
    content_change_fraction = float(changed.mean())
    if object_boxes:
        h, w = changed.shape
        mask = np.zeros((h, w), dtype=bool)
        for x0, y0, x1, y1 in object_boxes:
            mask[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = True
        if mask.any():
            object_damage = float(diff.mean(axis=2)[mask].mean())
            object_change_fraction = float(changed[mask].mean())
        else:
            object_damage = float("nan")
            object_change_fraction = float("nan")
    else:
        object_damage = float("nan")
        object_change_fraction = float("nan")
    return {
        "content_change_fraction": content_change_fraction,
        "content_preservation_proxy": float(1.0 - content_change_fraction),
        "object_box_damage_proxy": object_damage,
        "object_change_fraction": object_change_fraction,
    }


# --------------------------------------------------------------------------- #
# Failure identification (reuses the leakage-free CIC path)
# --------------------------------------------------------------------------- #
def _build_failures(examples: list[dict[str, Any]], status: ClipStatus, device: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(cfg.get("seed", 0))
    max_candidates = int(cfg.get("max_candidates", 64))
    grid_scales = cfg.get("grid_scales")
    high_conf_threshold = float(cfg.get("high_confidence_threshold", 0.7))
    enable_object_box_family = bool(cfg.get("enable_object_box_family", False))

    failures: list[dict[str, Any]] = []
    for ex in examples:
        allowed = list(ex["allowed_clip_labels"])
        label = int(ex["label"])
        distractors = set(ex.get("text_distractor_labels", []))
        distractor_idxs = [i for i, name in enumerate(allowed) if name in distractors and i != label]
        non_distractor_idxs = [i for i in range(len(allowed)) if i not in set(distractor_idxs)]
        pil = Image.fromarray((np.asarray(ex["image"]).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")
        text_boxes = [tuple(int(v) for v in b) for b in ex.get("text_boxes", [])]
        object_boxes = [tuple(int(v) for v in b) for b in ex.get("object_boxes", [])]

        predict_fn = _build_predict_fn(status, allowed, device)
        proposals = generate_open_region_proposals(
            pil,
            text_boxes=text_boxes,
            object_boxes=object_boxes,
            seed=seed + int(ex["example_id"]),
            max_candidates=max_candidates,
            grid_scales=grid_scales,
            enable_object_box_family=enable_object_box_family,
        )
        scores, original_probs = score_region_candidates(pil, proposals, predict_fn)
        orig = np.asarray(original_probs, dtype=np.float64)
        orig_pred = int(orig.argmax())
        orig_conf = float(orig.max())
        original_correct = bool(orig_pred == label)
        high_conf_failure = bool((not original_correct) and orig_conf >= high_conf_threshold)
        predicted_is_text_distractor = bool((not original_correct) and allowed[orig_pred] in distractors)
        text_driven_candidate = str(ex.get("text_driven_candidate", "")).lower() == "yes"
        if not (high_conf_failure and predicted_is_text_distractor and text_driven_candidate):
            continue
        if not text_boxes or not scores:
            continue

        top1 = scores[0]
        top3 = scores[:3]
        rand = _select_matched_random(scores, top1, "area_fraction")
        failures.append(
            {
                "example_id": int(ex["example_id"]),
                "human_label": ex["human_label"],
                "pil": pil,
                "predict_fn": predict_fn,
                "label": label,
                "allowed": allowed,
                "orig_pred": orig_pred,
                "p_before": orig,
                "text_boxes": text_boxes,
                "object_boxes": object_boxes,
                "top1_box": tuple(int(v) for v in top1.bbox),
                "top3_boxes": [tuple(int(v) for v in s.bbox) for s in top3],
                "rand_box": (tuple(int(v) for v in rand.bbox) if rand is not None else None),
                "distractor_idxs": distractor_idxs,
                "non_distractor_idxs": non_distractor_idxs,
                "selected_overlaps_text_box": _overlaps_any(top1.bbox, text_boxes),
                "selected_overlaps_object_box": _overlaps_any(top1.bbox, object_boxes),
            }
        )
    return failures


# --------------------------------------------------------------------------- #
# Operator sweeps
# --------------------------------------------------------------------------- #
def _label_free_stability(p_before: np.ndarray, p_after: np.ndarray, orig_pred: int) -> float:
    """Drop in the model's *own* clean top-class probability (label-free).

    Deliberately has no ``label`` / correctness parameter: this is the objective
    used to pick among independent top-k neutralizations, and the non-leakage rule
    forbids choosing a per-example operator/box using the true label.
    """

    return float(np.asarray(p_before)[orig_pred] - np.asarray(p_after)[orig_pred])


def select_label_free_best_index(
    after_probs: list[np.ndarray], p_before: np.ndarray, orig_pred: int
) -> int | None:
    """Index of the neutralization maximizing label-free stability.

    Ties break toward the lower index (deterministic). Returns ``None`` for an
    empty list. Takes no label, so per-example selection cannot leak correctness.
    """

    if not after_probs:
        return None
    best_idx = 0
    best_val = _label_free_stability(p_before, after_probs[0], orig_pred)
    for i in range(1, len(after_probs)):
        val = _label_free_stability(p_before, after_probs[i], orig_pred)
        if val > best_val:
            best_idx, best_val = i, val
    return best_idx


def _sweep_failure(fail: dict[str, Any], operators: list[Operator]) -> dict[str, list[dict[str, Any]]]:
    """Run every operator over a single failure, batching CLIP calls.

    Returns ``{"oracle": [...], "cic": [...], "topk": [...]}`` row lists.
    """

    pil = fail["pil"]
    predict_fn = fail["predict_fn"]
    label = fail["label"]
    orig_pred = fail["orig_pred"]
    p_before = fail["p_before"]
    text_boxes = fail["text_boxes"]
    object_boxes = fail["object_boxes"]
    top1_box = fail["top1_box"]
    top3_boxes = fail["top3_boxes"]
    rand_box = fail["rand_box"]
    distractor_idxs = fail["distractor_idxs"]
    non_distractor_idxs = fail["non_distractor_idxs"]
    width, height = pil.size

    # Build all neutralized images for this failure, then predict in one batch.
    batch_imgs: list[Image.Image] = []
    jobs: list[dict[str, Any]] = []  # {sweep, operator, region_boxes, available, idx | indep_idxs, after_img}

    def enqueue(sweep: str, op: Operator, boxes: list, *, extra: dict[str, Any] | None = None) -> None:
        after, avail = apply_operator(pil, boxes, op)
        rec = {"sweep": sweep, "operator": op, "boxes": boxes, "available": bool(avail), "after": after}
        if extra:
            rec.update(extra)
        if avail:
            rec["batch_index"] = len(batch_imgs)
            batch_imgs.append(after)
        jobs.append(rec)

    for op in operators:
        enqueue("oracle", op, text_boxes)
        enqueue("cic", op, [top1_box], extra={"role": "cic_top1"})
        if rand_box is not None:
            enqueue("cic_random", op, [rand_box], extra={"role": "matched_random"})
        enqueue("topk_union", op, list(top3_boxes))
        # top-3 independent: one image per box; selection is label-free downstream.
        for bi, b in enumerate(top3_boxes):
            enqueue("topk_indep", op, [b], extra={"indep_box_index": bi})

    probs = predict_fn(batch_imgs) if batch_imgs else np.zeros((0, len(p_before)))
    for rec in jobs:
        rec["p_after"] = probs[rec["batch_index"]] if rec.get("available") else None

    oracle_rows: list[dict[str, Any]] = []
    cic_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []

    def base_metrics(rec: dict[str, Any], boxes: list) -> dict[str, Any]:
        if not rec["available"] or rec["p_after"] is None:
            return {"available": False}
        pm = _prob_metrics(p_before, rec["p_after"], label, distractor_idxs, non_distractor_idxs)
        dm = _image_damage(pil, rec["after"], object_boxes)
        applied = operator_boxes(boxes, rec["operator"], width, height)
        area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in applied) / float(max(1, width * height))
        pm.update(dm)
        pm["available"] = True
        pm["area_fraction"] = float(area)
        pm["label_free_stability"] = _label_free_stability(p_before, rec["p_after"], orig_pred)
        return pm

    # Index cic_random rows by operator name for gap computation.
    rand_by_op = {
        rec["operator"].name: rec for rec in jobs if rec["sweep"] == "cic_random"
    }
    indep_by_op: dict[str, list[dict[str, Any]]] = {}
    for rec in jobs:
        if rec["sweep"] == "topk_indep":
            indep_by_op.setdefault(rec["operator"].name, []).append(rec)

    for op in operators:
        op_jobs = {rec["sweep"]: rec for rec in jobs if rec["operator"].name == op.name and rec["sweep"] in {"oracle", "cic", "topk_union"}}

        # Oracle ceiling row.
        m = base_metrics(op_jobs["oracle"], text_boxes)
        m.update({"example_id": fail["example_id"], "operator": op.name})
        oracle_rows.append(m)

        # CIC row (+ matched-random gap).
        m = base_metrics(op_jobs["cic"], [top1_box])
        rand_rec = rand_by_op.get(op.name)
        rand_strict = None
        if rand_rec is not None and rand_rec.get("available") and rand_rec["p_after"] is not None:
            rand_strict = bool(int(np.asarray(rand_rec["p_after"]).argmax()) == label)
        m.update(
            {
                "example_id": fail["example_id"],
                "operator": op.name,
                "matched_random_strict_repair": rand_strict,
                "selected_overlaps_text_box": fail["selected_overlaps_text_box"],
                "selected_overlaps_object_box": fail["selected_overlaps_object_box"],
            }
        )
        cic_rows.append(m)

        # Top-k union + independent-best (label-free) row.
        m_union = base_metrics(op_jobs["topk_union"], list(top3_boxes))
        indep = sorted(indep_by_op.get(op.name, []), key=lambda r: int(r["indep_box_index"]))
        indep_avail = [r for r in indep if r.get("available") and r["p_after"] is not None]
        best_indep = None
        if indep_avail:
            # Label-free selection only (no true label / correctness used).
            sel = select_label_free_best_index([r["p_after"] for r in indep_avail], p_before, orig_pred)
            best_indep = indep_avail[sel] if sel is not None else None
        if best_indep is not None:
            bm = _prob_metrics(p_before, best_indep["p_after"], label, distractor_idxs, non_distractor_idxs)
        else:
            bm = {}
        topk_rows.append(
            {
                "example_id": fail["example_id"],
                "operator": op.name,
                "union_available": m_union.get("available", False),
                "union_strict_repair": m_union.get("strict_repair"),
                "union_target_prob_improved": m_union.get("target_prob_improved"),
                "union_target_rank_improved": m_union.get("target_rank_improved"),
                "union_distractor_prob_decreased": m_union.get("distractor_prob_decreased"),
                "union_content_change_fraction": m_union.get("content_change_fraction"),
                "union_object_box_damage_proxy": m_union.get("object_box_damage_proxy"),
                "union_area_fraction": m_union.get("area_fraction"),
                "indep_best_available": bool(best_indep is not None),
                "indep_best_strict_repair": bm.get("strict_repair"),
                "indep_best_target_prob_improved": bm.get("target_prob_improved"),
                "indep_best_target_rank_improved": bm.get("target_rank_improved"),
                "indep_best_box_index": (int(best_indep["indep_box_index"]) if best_indep is not None else None),
            }
        )

    return {"oracle": oracle_rows, "cic": cic_rows, "topk": topk_rows}


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _rate(series: pd.Series) -> float:
    vals = series.dropna()
    return float(vals.astype(bool).mean()) if len(vals) else float("nan")


def _med(series: pd.Series) -> float:
    vals = series.dropna()
    return float(np.median(vals)) if len(vals) else float("nan")


def _mean(series: pd.Series) -> float:
    vals = series.dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def _aggregate_oracle(df: pd.DataFrame, operators: list[Operator]) -> pd.DataFrame:
    rows = []
    for op in operators:
        sub = df[df["operator"] == op.name]
        avail = sub[sub.get("available", False) == True] if "available" in sub else sub
        rows.append(
            {
                "operator": op.name,
                "available": bool(len(avail) > 0),
                "n_failures": int(len(avail)),
                "strict_repair": _rate(avail["strict_repair"]) if len(avail) else float("nan"),
                "alias_repair": _rate(avail["alias_repair"]) if len(avail) else float("nan"),
                "top3_recovery": _rate(avail["top3_recovery"]) if len(avail) else float("nan"),
                "top5_recovery": _rate(avail["top5_recovery"]) if len(avail) else float("nan"),
                "target_prob_improve_rate": _rate(avail["target_prob_improved"]) if len(avail) else float("nan"),
                "median_target_prob_gain": _med(avail["target_prob_gain"]) if len(avail) else float("nan"),
                "target_rank_improve_rate": _rate(avail["target_rank_improved"]) if len(avail) else float("nan"),
                "distractor_prob_decrease_rate": _rate(avail["distractor_prob_decreased"]) if len(avail) else float("nan"),
                "object_box_damage_proxy": _mean(avail["object_box_damage_proxy"]) if len(avail) else float("nan"),
                "content_preservation_proxy": _mean(avail["content_preservation_proxy"]) if len(avail) else float("nan"),
                "mean_area_fraction": _mean(avail["area_fraction"]) if len(avail) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_cic(df: pd.DataFrame, operators: list[Operator]) -> pd.DataFrame:
    rows = []
    for op in operators:
        sub = df[df["operator"] == op.name]
        avail = sub[sub.get("available", False) == True] if "available" in sub else sub
        strict = _rate(avail["strict_repair"]) if len(avail) else float("nan")
        rand = _rate(avail["matched_random_strict_repair"]) if len(avail) else float("nan")
        gap = (strict - rand) if (np.isfinite(strict) and np.isfinite(rand)) else float("nan")
        rows.append(
            {
                "operator": op.name,
                "available": bool(len(avail) > 0),
                "n_failures": int(len(avail)),
                "strict_repair": strict,
                "alias_repair": _rate(avail["alias_repair"]) if len(avail) else float("nan"),
                "target_prob_improve_rate": _rate(avail["target_prob_improved"]) if len(avail) else float("nan"),
                "median_target_prob_gain": _med(avail["target_prob_gain"]) if len(avail) else float("nan"),
                "target_rank_improve_rate": _rate(avail["target_rank_improved"]) if len(avail) else float("nan"),
                "distractor_prob_decrease_rate": _rate(avail["distractor_prob_decreased"]) if len(avail) else float("nan"),
                "matched_random_strict_repair": rand,
                "cic_minus_random_gap": gap,
                "selected_overlaps_text_box_rate": _rate(avail["selected_overlaps_text_box"]) if len(avail) else float("nan"),
                "selected_overlaps_object_box_rate": _rate(avail["selected_overlaps_object_box"]) if len(avail) else float("nan"),
                "object_box_damage_proxy": _mean(avail["object_box_damage_proxy"]) if len(avail) else float("nan"),
                "content_preservation_proxy": _mean(avail["content_preservation_proxy"]) if len(avail) else float("nan"),
                "mean_area_fraction": _mean(avail["area_fraction"]) if len(avail) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_topk(df: pd.DataFrame, operators: list[Operator]) -> pd.DataFrame:
    rows = []
    for op in operators:
        sub = df[df["operator"] == op.name]
        u = sub[sub["union_available"] == True]
        i = sub[sub["indep_best_available"] == True]
        rows.append(
            {
                "operator": op.name,
                "available": bool(len(u) > 0 or len(i) > 0),
                "union_strict_repair": _rate(u["union_strict_repair"]) if len(u) else float("nan"),
                "union_target_prob_improve_rate": _rate(u["union_target_prob_improved"]) if len(u) else float("nan"),
                "union_target_rank_improve_rate": _rate(u["union_target_rank_improved"]) if len(u) else float("nan"),
                "union_distractor_prob_decrease_rate": _rate(u["union_distractor_prob_decreased"]) if len(u) else float("nan"),
                "union_content_change_fraction": _mean(u["union_content_change_fraction"]) if len(u) else float("nan"),
                "union_object_box_damage_proxy": _mean(u["union_object_box_damage_proxy"]) if len(u) else float("nan"),
                "union_mean_area_fraction": _mean(u["union_area_fraction"]) if len(u) else float("nan"),
                "indep_best_strict_repair": _rate(i["indep_best_strict_repair"]) if len(i) else float("nan"),
                "indep_best_target_prob_improve_rate": _rate(i["indep_best_target_prob_improved"]) if len(i) else float("nan"),
                "indep_best_target_rank_improve_rate": _rate(i["indep_best_target_rank_improved"]) if len(i) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _best_operator(agg: pd.DataFrame, column: str) -> tuple[str, float]:
    sub = agg.dropna(subset=[column])
    if sub.empty:
        return ("", float("nan"))
    row = sub.loc[sub[column].idxmax()]
    return (str(row["operator"]), float(row[column]))


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def _read_baseline_reference() -> dict[str, Any]:
    path = Path("results/natural_text_verified_failure_eval/verified_failure_key_numbers.json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    keep = [
        "n_verified_text_driven_failures",
        "oracle_text_box_repair_accuracy",
        "oracle_text_box_repair_or_improve_rate",
        "cic_top1_repair_accuracy",
        "matched_random_proposal_repair_accuracy",
        "selected_overlaps_text_box_rate",
        "natural_text_supported",
        "open_world_claim_allowed",
    ]
    return {k: data.get(k) for k in keep if k in data}


def _save_operator_examples(failures: list[dict[str, Any]], operators_to_show: list[str], out_dir: Path, n: int) -> list[str]:
    ensure_dir(out_dir)
    op_map = {op.name: op for op in default_operators()}
    chosen = [op_map[name] for name in operators_to_show if name in op_map]
    paths: list[str] = []
    for fail in failures[: max(0, n)]:
        pil = fail["pil"]
        text_boxes = fail["text_boxes"]
        fig, axes = plt.subplots(1, 1 + len(chosen), figsize=(2.6 * (1 + len(chosen)), 2.8))
        axes = np.atleast_1d(axes)
        axes[0].imshow(pil)
        axes[0].set_title(f"{fail['human_label']} (orig)", fontsize=8)
        for x0, y0, x1, y1 in text_boxes:
            axes[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#e4572e", lw=1.5))
        for ax, op in zip(axes[1:], chosen):
            after, avail = apply_operator(pil, text_boxes, op)
            pa = fail["predict_fn"]([after])[0]
            correct = bool(int(np.asarray(pa).argmax()) == fail["label"])
            axes_title = f"{op.name}\n(oracle, correct={correct})" if avail else f"{op.name}\n(unavailable)"
            ax.imshow(after)
            ax.set_title(axes_title, fontsize=7)
        for ax in axes:
            ax.set_axis_off()
        fig.tight_layout()
        path = out_dir / f"{fail['example_id']}_{str(fail['human_label']).replace(' ', '_')}_operators.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _plot_oracle_vs_cic(oracle_agg: pd.DataFrame, cic_agg: pd.DataFrame, png: Path) -> None:
    ops = list(oracle_agg["operator"])
    x = np.arange(len(ops))
    plt.figure(figsize=(11.0, 4.8))
    plt.bar(x - 0.2, oracle_agg["strict_repair"].fillna(0.0), width=0.38, label="oracle strict", color="#4c78a8")
    plt.bar(x + 0.2, cic_agg.set_index("operator").reindex(ops)["strict_repair"].fillna(0.0).values, width=0.38, label="CIC top-1 strict", color="#e4572e")
    plt.axhline(ORACLE_CEILING_MODERATE, ls="--", color="#888", lw=1, label="0.50")
    plt.axhline(ORACLE_CEILING_HIGH, ls=":", color="#444", lw=1, label="0.70")
    plt.xticks(x, ops, rotation=35, ha="right", fontsize=8)
    plt.ylim(0, 1.02)
    plt.ylabel("strict repair on verified failures")
    plt.title("Intervention-operator sweep: oracle ceiling vs CIC top-1 (diagnostic)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(png, dpi=160)
    plt.close()


def _ceiling_analysis_md(key: dict[str, Any]) -> str:
    best_oracle_op = key["best_oracle_strict_operator"]
    best_oracle = key["best_oracle_strict_repair"]
    best_cic_op = key["best_cic_strict_operator"]
    best_cic = key["best_cic_strict_repair"]
    above_50 = key["oracle_strict_exceeds_0.50"]
    above_70 = key["oracle_strict_exceeds_0.70"]
    best_dir = key["best_directional_operator"]
    best_dir_rate = key["best_directional_improve_rate"]

    lines = [
        "# Natural-text intervention-operator: oracle-ceiling analysis",
        "",
        "**Diagnostic only.** This analysis isolates whether strict natural-text repair is ",
        "limited by the *intervention operator / masking strategy* or by *CIC proposal ",
        "selection/scoring*. It does **not** update any headline / final-report metric, and ",
        "`open_world_claim_allowed` stays **False**.",
        "",
        "## Oracle ceiling (operator applied to the annotated text/logo boxes)",
        "",
        f"- Best oracle **strict** operator: `{best_oracle_op}` at **{best_oracle:.3f}** strict repair.",
        f"- Best **directional** operator: `{best_dir}` at {best_dir_rate:.3f} target-probability improvement rate.",
        f"- Does any operator raise oracle strict repair above {ORACLE_CEILING_MODERATE:.2f}? **{'YES' if above_50 else 'NO'}**.",
        f"- Does any operator raise oracle strict repair above {ORACLE_CEILING_HIGH:.2f}? **{'YES' if above_70 else 'NO'}**.",
        "",
        "## CIC (operator applied to the existing CIC top-1 proposal)",
        "",
        f"- Best CIC **strict** operator: `{best_cic_op}` at **{best_cic:.3f}** strict repair.",
        f"- Best CIC vs matched-random strict gap: **{key['best_cic_random_gap']:.3f}** "
        f"(operator `{key['best_cic_random_gap_operator']}`).",
        "",
        "## Interpretation",
        "",
        f"- {key['interpretation']}",
        "",
        "## Strict-support candidacy",
        "",
        f"- CIC strict beats the random baseline by >= {STRICT_GATE_CIC_RANDOM_GAP:.2f}: "
        f"**{'YES' if key['cic_strict_candidate'] else 'NO'}**.",
        f"- A pre-declared GLOBAL operator (`{key['global_operator']}`, chosen by "
        f"\"{GLOBAL_OPERATOR_CRITERION}\") would let a strict gate pass: "
        f"**{'YES' if key['strict_gate_could_pass_global'] else 'NO'}**.",
        "",
        "> Even where a candidate is flagged, the final paper is **not** updated here. Any ",
        "> strict natural-text support requires separate review, and natural-text directional ",
        "> evidence is never reported as positive strict support unless the strict gate truly ",
        "> passes. `open_world_claim_allowed = false`.",
        "",
        f"_Operator panel: {key['n_operators']} operators ({key['n_unavailable_operators']} unavailable: "
        f"{', '.join(key['unavailable_operators']) or 'none'}). Verified failures: {key['n_verified_failures']}._",
    ]
    return "\n".join(lines)


def _summary_md(key: dict[str, Any], oracle_agg: pd.DataFrame, cic_agg: pd.DataFrame) -> str:
    def table(df: pd.DataFrame, cols: list[str]) -> str:
        head = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body = []
        for _, r in df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    cells.append("n/a" if not np.isfinite(v) else f"{v:.3f}")
                else:
                    cells.append(str(v))
            body.append("| " + " | ".join(cells) + " |")
        return "\n".join([head, sep, *body])

    oracle_cols = ["operator", "strict_repair", "alias_repair", "top3_recovery", "top5_recovery", "target_prob_improve_rate", "median_target_prob_gain", "target_rank_improve_rate", "distractor_prob_decrease_rate", "object_box_damage_proxy", "content_preservation_proxy"]
    cic_cols = ["operator", "strict_repair", "target_prob_improve_rate", "target_rank_improve_rate", "distractor_prob_decrease_rate", "cic_minus_random_gap", "content_preservation_proxy", "mean_area_fraction"]

    lines = [
        "# Natural-text intervention/operator sweep (diagnostic)",
        "",
        "Diagnoses whether strict natural-text repair is limited by the **intervention ",
        "operator / masking strategy** rather than by CIC proposal selection. Holds the ",
        "candidate geometry fixed (annotated text/logo boxes for the oracle ceiling; the ",
        "existing CIC top-1 proposal for the method) and sweeps deterministic neutralization ",
        "operators on the verified text-driven failures.",
        "",
        f"- Backend: `{key['backend']}`. Model: `{key['model_name']}`. Real pretrained: `{key['real_pretrained_model_loaded']}`.",
        f"- Verified text-driven failures: **{key['n_verified_failures']}**.",
        f"- Operators evaluated: **{key['n_operators']}** ({key['n_unavailable_operators']} unavailable: {', '.join(key['unavailable_operators']) or 'none'}).",
        f"- cv2 / Telea inpaint available: `{key['cv2_available']}`.",
        "",
        "## Headline diagnostic answers",
        "",
        f"- Best **oracle strict** operator: `{key['best_oracle_strict_operator']}` = **{key['best_oracle_strict_repair']:.3f}**.",
        f"- Best **CIC strict** operator: `{key['best_cic_strict_operator']}` = **{key['best_cic_strict_repair']:.3f}**.",
        f"- Best **directional** operator: `{key['best_directional_operator']}` = **{key['best_directional_improve_rate']:.3f}** (target-prob improvement rate).",
        f"- Oracle strict ceiling exceeds 0.50: **{key['oracle_strict_exceeds_0.50']}**; exceeds 0.70: **{key['oracle_strict_exceeds_0.70']}**.",
        f"- Oracle ceiling high enough for strict natural-text support: **{key['oracle_ceiling_supports_strict']}**.",
        f"- CIC bottleneck attribution: **{key['cic_bottleneck']}**.",
        f"- Any strict gate could pass under a pre-declared global operator: **{key['strict_gate_could_pass_global']}**.",
        "",
        "## Non-leakage / scope",
        "",
        "- Operators are reported as a **diagnostic panel**; no operator is selected per-example using the true label or correctness.",
        "- The only global operator choice uses a pre-declared, label-free aggregate criterion.",
        f"- `open_world_claim_allowed`: **{key['open_world_claim_allowed']}** (unchanged).",
        "- Existing headline / final-report metrics are **not** modified by this experiment.",
        "",
        "## Baseline reference (from the existing verified-failure run; unchanged)",
        "",
        "```json",
        json.dumps(key.get("baseline_reference", {}), indent=2),
        "```",
        "",
        "## Oracle text-box ceiling sweep",
        "",
        "Note: for this verified set the allowed labels per image are the visual target plus the ",
        "text/logo distractors, so `alias_repair` (predicting any non-distractor label) collapses ",
        "to strict repair here; it is reported for generality.",
        "",
        table(oracle_agg, oracle_cols),
        "",
        "## CIC-selected-region operator sweep",
        "",
        table(cic_agg, cic_cols),
        "",
        "See `oracle_ceiling_analysis.md` for the ceiling interpretation, and the CSVs for the ",
        "full metric set (including top-k union/independent results).",
    ]
    return "\n".join(lines)


def _write_unavailable(out_dir: Path, cfg: dict[str, Any], status: ClipStatus, reason: str, n_failures: int) -> dict[str, str]:
    ensure_dir(out_dir / "examples")
    key = {
        "diagnostic_only": True,
        "real_pretrained_model_loaded": bool(status.available and status.pretrained and status.backend in {"open_clip", "transformers"}),
        "backend": status.backend,
        "model_name": status.model_name,
        "n_verified_failures": int(n_failures),
        "cv2_available": cv2_available(),
        "open_world_claim_allowed": ALWAYS_OPEN_WORLD_CLAIM_ALLOWED,
        "natural_text_supported_unchanged": False,
        "headline": "intervention sweep unavailable",
        "reason": reason,
        "baseline_reference": _read_baseline_reference(),
        "preferred_wording": PREFERRED_WORDING,
        "forbidden_wording": FORBIDDEN_WORDING,
    }
    (out_dir / "operator_sweep_key_numbers.json").write_text(json.dumps(key, indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "operator_sweep_summary.md").write_text(
        f"# Natural-text intervention/operator sweep (diagnostic)\n\nUnavailable: {reason}\n", encoding="utf-8"
    )
    (out_dir / "oracle_ceiling_analysis.md").write_text(
        f"# Oracle-ceiling analysis\n\nUnavailable: {reason}\n", encoding="utf-8"
    )
    pd.DataFrame().to_csv(out_dir / "oracle_operator_metrics.csv", index=False)
    pd.DataFrame().to_csv(out_dir / "cic_operator_metrics.csv", index=False)
    pd.DataFrame().to_csv(out_dir / "topk_operator_metrics.csv", index=False)
    (out_dir / "intervention_sweep_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return {
        "key_numbers": str(out_dir / "operator_sweep_key_numbers.json"),
        "summary": str(out_dir / "operator_sweep_summary.md"),
        "oracle_ceiling": str(out_dir / "oracle_ceiling_analysis.md"),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / cfg.get("output_subdir", DEFAULT_OUTPUT_SUBDIR))
    examples_dir = ensure_dir(out_dir / "examples")

    data_cfg = dict(cfg.get("data", {}))
    image_size = data_cfg.get("image_size", 224)
    image_size = int(image_size) if image_size else None
    split = str(data_cfg.get("split", "test"))
    local_cfg = dict(data_cfg.get("local", {}))
    verified_cfg = dict(data_cfg.get("verified", {}))
    root = verified_cfg.get("root", local_cfg.get("root", "data/natural_text_images"))
    annotations_csv = verified_cfg.get("annotations_csv") or str(Path(root) / "verified_annotations.csv")

    bundle = load_verified_natural_text_dataset(
        root=root, annotations_csv=annotations_csv, image_size=image_size, split=split, include_only=True
    )
    examples = bundle.examples

    model_cfg = dict(cfg.get("model", {}))
    device = _device(model_cfg, cfg)
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))
    if str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend.lower() == "fake":
        status = ClipStatus(False, "fake", "fake_natural_text", pretrained=False, device=device, backend_attempted="fake", error_message="fake backend")
        return _write_unavailable(out_dir, cfg, status, "fake backend cannot support the intervention sweep", 0)

    status = check_clip_available(
        device=device,
        allow_download=_downloads_allowed(model_cfg),
        preferred_backend=requested_backend,
        model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
        pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
        transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return _write_unavailable(out_dir, cfg, status, status.error_message or "real pretrained CLIP unavailable", 0)
    if not examples:
        return _write_unavailable(out_dir, cfg, status, "no verified annotations loaded", 0)

    save_example_images(examples, examples_dir)
    failures = _build_failures(examples, status, device, cfg)
    n_failures = len(failures)
    if n_failures == 0:
        return _write_unavailable(out_dir, cfg, status, "no verified text-driven failures found", 0)

    operators = default_operators()
    oracle_records: list[dict[str, Any]] = []
    cic_records: list[dict[str, Any]] = []
    topk_records: list[dict[str, Any]] = []
    for fail in failures:
        out = _sweep_failure(fail, operators)
        oracle_records.extend(out["oracle"])
        cic_records.extend(out["cic"])
        topk_records.extend(out["topk"])

    oracle_df = pd.DataFrame(oracle_records)
    cic_df = pd.DataFrame(cic_records)
    topk_df = pd.DataFrame(topk_records)

    oracle_agg = _aggregate_oracle(oracle_df, operators)
    cic_agg = _aggregate_cic(cic_df, operators)
    topk_agg = _aggregate_topk(topk_df, operators)

    unavailable_ops = [op.name for op in operators if not bool(oracle_agg.loc[oracle_agg["operator"] == op.name, "available"].iloc[0])]

    best_oracle_op, best_oracle = _best_operator(oracle_agg, "strict_repair")
    best_cic_op, best_cic = _best_operator(cic_agg, "strict_repair")
    best_dir_op, best_dir = _best_operator(oracle_agg, "target_prob_improve_rate")
    best_gap_op, best_gap = _best_operator(cic_agg, "cic_minus_random_gap")

    above_50 = bool(np.nanmax(oracle_agg["strict_repair"].to_numpy()) > ORACLE_CEILING_MODERATE) if oracle_agg["strict_repair"].notna().any() else False
    above_70 = bool(np.nanmax(oracle_agg["strict_repair"].to_numpy()) > ORACLE_CEILING_HIGH) if oracle_agg["strict_repair"].notna().any() else False

    # Pre-declared GLOBAL operator: maximize oracle directional improvement rate.
    global_op, _ = _best_operator(oracle_agg, "target_prob_improve_rate")
    global_cic_gap = float(cic_agg.loc[cic_agg["operator"] == global_op, "cic_minus_random_gap"].iloc[0]) if global_op else float("nan")
    global_oracle_repair_or_improve = (
        float(oracle_agg.loc[oracle_agg["operator"] == global_op, "target_prob_improve_rate"].iloc[0]) if global_op else float("nan")
    )
    cic_strict_candidate = bool(np.isfinite(best_gap) and best_gap >= STRICT_GATE_CIC_RANDOM_GAP)
    strict_gate_could_pass_global = bool(np.isfinite(global_cic_gap) and global_cic_gap >= STRICT_GATE_CIC_RANDOM_GAP)

    # Bottleneck attribution.
    if above_70 and (not np.isfinite(best_cic) or best_cic < 0.50):
        cic_bottleneck = "proposal/scoring (oracle ceiling is high but CIC strict stays low)"
    elif (not above_50) and np.isfinite(best_dir) and best_dir >= 0.70:
        cic_bottleneck = "residual natural-image ambiguity / label-set difficulty (oracle strict stays low even with a known text box, while directional improvement is high)"
    elif cic_strict_candidate:
        cic_bottleneck = "intervention strength matters (a stronger operator lifts CIC strict to beat random by >= 0.15) — candidate, review required"
    else:
        cic_bottleneck = "mixed: neither oracle ceiling nor a stronger operator alone resolves strict repair"

    oracle_ceiling_supports_strict = bool(above_70)

    if above_70 and (not np.isfinite(best_cic) or best_cic < 0.50):
        interpretation = (
            "Oracle strict repair becomes high while CIC strict stays low: the main bottleneck is "
            "CIC proposal/scoring, not the intervention operator."
        )
    elif not above_50:
        interpretation = (
            "No operator lifts oracle strict repair above 0.50 even with the known text box, while "
            "directional improvement stays high: exact top-1 natural-image recovery is limited by "
            "residual natural-image ambiguity / label-set difficulty, not only by CIC."
        )
    elif cic_strict_candidate:
        interpretation = (
            "A stronger operator lifts CIC strict repair to beat random by >= 0.15: candidate for strict "
            "natural-text support — flagged for review; the final paper is NOT updated here."
        )
    else:
        interpretation = (
            "Operators move directional metrics but exact strict top-1 recovery remains partial; strength "
            "and proposal/scoring both contribute. No strict support claim is made."
        )

    cv2_ok = cv2_available()
    key = {
        "diagnostic_only": True,
        "experiment": "natural_text_intervention_sweep",
        "backend": status.backend,
        "model_name": status.model_name,
        "real_pretrained_model_loaded": bool(status.pretrained),
        "cv2_available": cv2_ok,
        "n_verified_failures": int(n_failures),
        "n_operators": len(operators),
        "n_unavailable_operators": len(unavailable_ops),
        "unavailable_operators": unavailable_ops,
        "best_oracle_strict_operator": best_oracle_op,
        "best_oracle_strict_repair": best_oracle,
        "best_cic_strict_operator": best_cic_op,
        "best_cic_strict_repair": best_cic,
        "best_directional_operator": best_dir_op,
        "best_directional_improve_rate": best_dir,
        "best_cic_random_gap_operator": best_gap_op,
        "best_cic_random_gap": best_gap,
        "oracle_strict_exceeds_0.50": above_50,
        "oracle_strict_exceeds_0.70": above_70,
        "oracle_ceiling_supports_strict": oracle_ceiling_supports_strict,
        "cic_strict_candidate": cic_strict_candidate,
        "cic_bottleneck": cic_bottleneck,
        "interpretation": interpretation,
        "global_operator": global_op,
        "global_operator_criterion": GLOBAL_OPERATOR_CRITERION,
        "global_operator_cic_random_gap": global_cic_gap,
        "global_operator_oracle_directional_rate": global_oracle_repair_or_improve,
        "strict_gate_could_pass_global": strict_gate_could_pass_global,
        "strict_gate_cic_random_gap_required": STRICT_GATE_CIC_RANDOM_GAP,
        "open_world_claim_allowed": ALWAYS_OPEN_WORLD_CLAIM_ALLOWED,
        "natural_text_supported_unchanged": False,
        "headline": "natural-text intervention/operator sweep (diagnostic) — final metrics unchanged",
        "baseline_reference": _read_baseline_reference(),
        "preferred_wording": PREFERRED_WORDING,
        "forbidden_wording": FORBIDDEN_WORDING,
    }

    # Visual examples for the best and worst oracle-strict operators.
    finite = oracle_agg.dropna(subset=["strict_repair"])
    worst_op = str(finite.loc[finite["strict_repair"].idxmin(), "operator"]) if not finite.empty else best_oracle_op
    show_ops = [op for op in [best_oracle_op, best_dir_op, worst_op] if op]
    seen: set[str] = set()
    show_ops = [o for o in show_ops if not (o in seen or seen.add(o))]
    example_paths = _save_operator_examples(failures, show_ops, examples_dir, int(cfg.get("n_example_visualizations", 4)))

    oracle_df.to_csv(out_dir / "oracle_operator_per_example.csv", index=False)
    cic_df.to_csv(out_dir / "cic_operator_per_example.csv", index=False)
    oracle_agg.to_csv(out_dir / "oracle_operator_metrics.csv", index=False)
    cic_agg.to_csv(out_dir / "cic_operator_metrics.csv", index=False)
    topk_agg.to_csv(out_dir / "topk_operator_metrics.csv", index=False)
    topk_df.to_csv(out_dir / "topk_operator_per_example.csv", index=False)
    (out_dir / "operator_sweep_key_numbers.json").write_text(json.dumps(key, indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "oracle_ceiling_analysis.md").write_text(_ceiling_analysis_md(key), encoding="utf-8")
    (out_dir / "operator_sweep_summary.md").write_text(_summary_md(key, oracle_agg, cic_agg), encoding="utf-8")
    (out_dir / "intervention_sweep_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    _plot_oracle_vs_cic(oracle_agg, cic_agg, out_dir / "operator_sweep_plot.png")

    return {
        "key_numbers": str(out_dir / "operator_sweep_key_numbers.json"),
        "oracle_metrics": str(out_dir / "oracle_operator_metrics.csv"),
        "cic_metrics": str(out_dir / "cic_operator_metrics.csv"),
        "topk_metrics": str(out_dir / "topk_operator_metrics.csv"),
        "oracle_ceiling": str(out_dir / "oracle_ceiling_analysis.md"),
        "summary": str(out_dir / "operator_sweep_summary.md"),
        "plot": str(out_dir / "operator_sweep_plot.png"),
        "examples": [str(p) for p in example_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/natural_text_intervention_sweep.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
