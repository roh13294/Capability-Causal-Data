from __future__ import annotations

"""Apples-to-apples COCO-Text automated finite-candidate proposal sweep.

Experiment name: ``coco_text_auto_proposal_sweep``.

Why this exists
---------------
The first pilot (`run_auto_proposal_cic_pilot.py`) compared automated proposal
families against an `existing_cic_baseline` that used a **different** candidate
budget/family from the finalized COCO-Text headline (see
`results/auto_proposal_pilot/coco_reconciliation.md`). Its baseline reported
strict CIC repair = 0.410, whereas the finalized report records 0.538. The two
numbers are not the same estimator, so the pilot's "saliency beats baseline"
statement could not be promoted.

This script fixes that by introducing an **apples-to-apples** baseline,
`existing_cic_baseline_a2a`, that *exactly* reproduces the finalized
`cic_top1_repair_excl_ocr` recipe (max_candidates=48, grid_scales=[0.18,0.3,0.45],
text/object geometry passed in, OCR family generated then excluded at selection,
default neutralization, alias-aware scoring). Every automated proposal family
(grid / edge_component / saliency, optional SAM / DINO) is then evaluated with the
**identical** scoring and metric code and compared *only* against this a2a
baseline.

Scope / honesty
---------------
* This is **automated finite-candidate proposal generation**, NOT open-world
  shortcut discovery, NOT universal repair, NOT deployment validation, and NOT a
  replacement for the finalized STS report.
* Writes ONLY under ``results/auto_proposal_pilot/``. It never modifies the
  finalized report, `results/final_report/`, `results/coco_text_cic_full/`, the
  triage artifacts, or any existing benchmark JSON/CSV. It does not change support
  gates or finalized metrics.
* If real pretrained CLIP or the data is unavailable it records the reason and
  exits 0 (clean skip).
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_reliability.data.natural_text_dataset import load_local_folder_dataset
from causal_reliability.discovery.cic_region_scoring import neutralize_region, score_region_candidates
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
)
from causal_reliability.proposals.auto_proposals import (
    ALL_FAMILIES,
    CLASSICAL_FAMILIES,
    RANDOM_TYPE,
    SamConfig,
    generate_proposal_sets,
    generator_availability,
    proposal_sets_to_region_proposals,
)
from causal_reliability.real_models.clip_zero_shot import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED_TAG,
    ClipStatus,
    ClipZeroShotClassifier,
    check_clip_available,
)
from causal_reliability.utils.io import ensure_dir

RESULTS_SUBDIR = "auto_proposal_pilot"
PROMPT_TEMPLATE = "a photo of a {label}"
DEFAULT_PROB_EPS = 0.01

# The apples-to-apples finalized baseline (reproduces cic_top1_repair_excl_ocr).
BASELINE_FAMILY = "existing_cic_baseline_a2a"

# Finalized full-run candidate-generation settings (must match
# results/coco_text_cic_full/coco_text_full_config_used.yaml).
A2A_MAX_CANDIDATES = 48
A2A_GRID_SCALES = [0.18, 0.3, 0.45]

NON_CLAIMS = [
    "This is automated finite-candidate proposal generation, NOT open-world shortcut discovery.",
    "This is NOT universal repair or general robustness.",
    "This is NOT deployment validation or clinical validation.",
    "This is NOT a replacement for the finalized STS report.",
]

# Promotion thresholds (any automated family beating the a2a baseline by one).
PROMOTE_STRICT_REPAIR_GAIN = 0.05
PROMOTE_STRICT_DIR_MAX_DROP = 0.03
PROMOTE_TEXT_OVERLAP_GAIN = 0.15
PROMOTE_TEXT_OVERLAP_REPAIR_MAX_DROP = 0.05
PROMOTE_COVERAGE_GAIN = 0.20
PROMOTE_COVERAGE_REPAIR_TOL = 0.05  # "repair approximately preserved"

# SAM-specific promotion thresholds (task-specified). sam_promotable=True if ANY:
#   A: SAM strict repair beats a2a by >= +0.05 absolute
#   B: SAM directional repair beats a2a by >= +0.05 absolute
#   C: SAM text overlap improves >= +0.15 while strict repair drops <= 0.05
#   D: SAM coverage ceiling improves >= +0.20 while strict repair drops <= 0.05
SAM_PROMOTE_STRICT_REPAIR_GAIN = 0.05
SAM_PROMOTE_DIR_REPAIR_GAIN = 0.05
SAM_PROMOTE_TEXT_OVERLAP_GAIN = 0.15
SAM_PROMOTE_TEXT_OVERLAP_REPAIR_MAX_DROP = 0.05
SAM_PROMOTE_COVERAGE_GAIN = 0.20
SAM_PROMOTE_COVERAGE_REPAIR_MAX_DROP = 0.05


# --------------------------------------------------------------------------- #
# Model / geometry helpers
# --------------------------------------------------------------------------- #
def _pil_to_tensor(images):
    import torch

    arrays = [np.asarray(img.convert("RGB")).astype(np.float32) / 255.0 for img in images]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


def _build_predict_fn(status: ClipStatus, allowed_labels: list[str], device: str, batch_size: int = 64):
    prompts = [PROMPT_TEMPLATE.format(label=name) for name in allowed_labels]
    classifier = ClipZeroShotClassifier(status, allowed_labels, prompts=prompts, device=device)

    def predict_fn(images):
        out: list[np.ndarray] = []
        bs = max(1, int(batch_size))
        for start in range(0, len(images), bs):
            chunk = images[start : start + bs]
            res = classifier.predict(_pil_to_tensor(chunk))
            out.append(np.asarray(res["probabilities"].detach().cpu().numpy(), dtype=np.float64))
        return np.concatenate(out, axis=0) if out else np.zeros((0, len(allowed_labels)), dtype=np.float64)

    return predict_fn


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = [int(v) for v in a]
    bx0, by0, bx1, by1 = [int(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    return float(inter / max(1, area_a + area_b - inter))


def _max_iou(box, boxes) -> float:
    return max((_iou(box, b) for b in boxes), default=0.0)


def _overlaps_any(box, boxes, threshold: float = 0.1) -> bool:
    return _max_iou(box, boxes) >= threshold if boxes else False


def _select_matched_random(scores: list, top1, field: str = "area_fraction"):
    randoms = [s for s in scores if s.proposal_type == RANDOM_TYPE]
    if not randoms:
        return None
    target = float(getattr(top1, field, 0.0)) if top1 is not None else 0.0
    return min(randoms, key=lambda s: abs(float(getattr(s, field, 0.0)) - target))


# --------------------------------------------------------------------------- #
# Subset loading
# --------------------------------------------------------------------------- #
def _read_ids(csv_path: Path) -> list[int]:
    if not csv_path.exists():
        return []
    frame = pd.read_csv(csv_path)
    if "example_id" not in frame.columns:
        return []
    return [int(v) for v in frame["example_id"].tolist()]


def _all500_ids(full_per_example: Path, by_id: dict[int, Any]) -> list[int]:
    """Return the same example ids the finalized all_500 run used, if available.

    Reads ``results/coco_text_cic_full/coco_text_full_per_example.csv`` READ-ONLY
    to match membership exactly. Falls back to all loaded ids (sorted) otherwise.
    """

    if full_per_example.exists():
        frame = pd.read_csv(full_per_example, usecols=["example_id"])
        ids = sorted({int(v) for v in frame["example_id"].tolist()})
        ids = [i for i in ids if i in by_id]
        if ids:
            return ids
    return sorted(by_id.keys())


def load_subsets(triage_dir: Path, full_per_example: Path, by_id: dict[int, Any], want_all500: bool) -> dict[str, list[int]]:
    strict = _read_ids(triage_dir / "coco_text_verified_oracle_repairable_failures.csv")
    directional = _read_ids(triage_dir / "coco_text_verified_directional_failures.csv")
    subs = {
        "strict": [i for i in strict if i in by_id],
        "directional": [i for i in directional if i in by_id],
    }
    if want_all500:
        subs["all_500"] = _all500_ids(full_per_example, by_id)
    return subs


# --------------------------------------------------------------------------- #
# Candidate generation per family
# --------------------------------------------------------------------------- #
def _baseline_a2a_scores(pil, text_boxes, object_boxes, predict_fn, eid: int, seed: int):
    """Exactly the finalized cic_top1_repair_excl_ocr candidate path.

    OCR/text geometry is generated (enable_ocr_family default True) and competes
    for the 48 slots, then excluded from the *selectable* set downstream.
    """

    proposals = generate_open_region_proposals(
        pil,
        text_boxes=text_boxes,
        object_boxes=object_boxes,
        seed=seed + eid,
        max_candidates=A2A_MAX_CANDIDATES,
        grid_scales=A2A_GRID_SCALES,
        enable_object_box_family=False,
    )
    scores, original_probs = score_region_candidates(pil, proposals, predict_fn)
    return scores, original_probs


def _auto_family_scores(pil, family, predict_fn, eid: int, seed: int, allow_download: bool, max_boxes: int, family_kwargs: dict | None = None):
    extra = dict(family_kwargs or {})
    sets = generate_proposal_sets(
        pil, [family], allow_download=allow_download, max_boxes=max_boxes, seed=seed + eid, **extra
    )
    ps = sets[family]
    rps = proposal_sets_to_region_proposals(pil, [ps], include_random_control=True, seed=seed + eid)
    scores, original_probs = score_region_candidates(pil, rps, predict_fn)
    return scores, original_probs, ps


def _resolve_sam_points_per_side(args) -> int:
    if args.sam_points_per_side is not None:
        return int(args.sam_points_per_side)
    return 8 if args.sam_fast else 16


def _resolve_sam_max_side(args) -> int:
    if args.sam_max_side is not None:
        return int(args.sam_max_side)
    return 512 if args.sam_fast else 0


def _sam_cache_dir(args) -> Path:
    return Path(args.results_dir) / RESULTS_SUBDIR / "cache" / "sam_proposals"


def resolve_subsets_and_gating(args) -> tuple[list[str], list[str]]:
    """Resolve which subsets to evaluate and record any gating notes.

    Default is strict+directional; with ``--include-sam`` the guarded default is
    strict-only (SAM is slow), and the ``all_500`` subset with SAM requires an
    explicit ``--all500 --confirm-slow-sam``. ``--strict-only`` / ``--no-all500``
    override as expected.
    """

    if args.strict_only:
        subset_spec = "strict"
    elif args.subset is not None:
        subset_spec = args.subset
    elif args.include_sam:
        subset_spec = "strict"
    else:
        subset_spec = "strict,directional"

    if subset_spec == "auto":
        chosen = ["strict", "directional", "all_500"]
    else:
        chosen = [s.strip() for s in subset_spec.split(",") if s.strip()]

    gating: list[str] = []
    if args.no_all500 and "all_500" in chosen:
        chosen = [s for s in chosen if s != "all_500"]
        gating.append("all_500 dropped: --no-all500 set")
    if args.include_sam and "all_500" in chosen and not (args.all500 and args.confirm_slow_sam):
        chosen = [s for s in chosen if s != "all_500"]
        gating.append(
            "all_500 SAM not run: requires --all500 --confirm-slow-sam (intentionally slow); "
            "default SAM run is strict-only"
        )
    return chosen, gating


def _sam_kwargs(args) -> dict[str, Any]:
    """Translate the --sam-* CLI knobs into kwargs for the SAM generator.

    Resolves the fast defaults (``--sam-fast`` => points_per_side=8, max_side=512)
    and wires the per-image proposal cache when ``--cache-sam-proposals``/``--resume``
    is set.
    """

    use_cache = bool(args.cache_sam_proposals or args.resume)
    return {
        "enable_real_sam": True,
        "checkpoint_path": args.sam_checkpoint,
        "model_type": args.sam_model_type,
        "points_per_side": _resolve_sam_points_per_side(args),
        "pred_iou_thresh": float(args.sam_pred_iou_thresh),
        "stability_score_thresh": float(args.sam_stability_score_thresh),
        "min_area_frac": float(args.sam_min_area_frac),
        "max_area_frac": float(args.sam_max_area_frac),
        "min_side": int(args.sam_min_side),
        "device": args.sam_device,
        "dedupe_iou": float(args.sam_dedupe_iou),
        "crop_n_layers": int(args.sam_crop_n_layers),
        "max_side": _resolve_sam_max_side(args),
        "cache_dir": str(_sam_cache_dir(args)) if use_cache else None,
        "use_cache": use_cache,
    }


def _sam_config_record(args) -> dict[str, Any]:
    """A JSON-friendly record of the SAM settings used (for the metrics output)."""

    cfg = SamConfig(
        checkpoint_path=args.sam_checkpoint,
        model_type=args.sam_model_type,
        points_per_side=_resolve_sam_points_per_side(args),
        pred_iou_thresh=float(args.sam_pred_iou_thresh),
        stability_score_thresh=float(args.sam_stability_score_thresh),
        max_proposals=int(args.sam_max_proposals),
        min_area_frac=float(args.sam_min_area_frac),
        max_area_frac=float(args.sam_max_area_frac),
        min_side=int(args.sam_min_side),
        device=args.sam_device,
        dedupe_iou=float(args.sam_dedupe_iou),
        crop_n_layers=int(args.sam_crop_n_layers),
        max_side=_resolve_sam_max_side(args),
    )
    rec = cfg.to_dict()
    rec["fast"] = bool(args.sam_fast)
    rec["timeout_seconds"] = float(args.sam_timeout_seconds)
    rec["cache_enabled"] = bool(args.cache_sam_proposals or args.resume)
    rec["resume"] = bool(args.resume)
    return rec


# --------------------------------------------------------------------------- #
# Identical per-example evaluation for every family
# --------------------------------------------------------------------------- #
def _selectable(scores: list) -> list:
    """Candidates eligible for CIC top-1 selection: non-random, non-OCR.

    For the a2a baseline this removes the generated-but-excluded OCR family (so it
    matches `cic_top1_repair_excl_ocr`). For automated families there is no OCR
    family, so this is just "non-random" — i.e. identical treatment."""

    return [s for s in scores if s.proposal_type != RANDOM_TYPE and proposal_family(s.proposal_type) != OCR_FAMILY]


def _evaluate_example(
    pil,
    scores: list,
    original_probs,
    predict_fn,
    *,
    allowed,
    target,
    aliases,
    target_idxs,
    distractor_idx,
    text_boxes,
    object_boxes,
    prob_eps: float,
) -> dict[str, Any]:
    op = np.asarray(original_probs, dtype=np.float64)
    orig_target_prob = label_set_prob(op, target_idxs)
    orig_distractor_prob = float(op[distractor_idx])

    selectable = _selectable(scores)
    top1 = selectable[0] if selectable else None

    def repaired_stats(probs) -> dict[str, Any]:
        p = np.asarray(probs, dtype=np.float64)
        pred = int(p.argmax())
        t_prob = label_set_prob(p, target_idxs)
        d_prob = float(p[distractor_idx])
        return {
            "alias_correct": bool(is_target_label(allowed[pred], target, aliases)),
            "target_prob": t_prob,
            "distractor_prob": d_prob,
            "target_prob_improved": bool(t_prob > orig_target_prob + prob_eps),
            "distractor_decreased": bool(d_prob < orig_distractor_prob - prob_eps),
        }

    if top1 is not None:
        cic = repaired_stats(predict_fn([neutralize_region(pil, top1.bbox)])[0])
        sel_text_overlap = _overlaps_any(top1.bbox, text_boxes)
        sel_obj_overlap = _overlaps_any(top1.bbox, object_boxes)
        sel_area = float(top1.area_fraction)
    else:
        cic = {"alias_correct": False, "target_prob_improved": False, "distractor_decreased": False}
        sel_text_overlap = False
        sel_obj_overlap = False
        sel_area = float("nan")

    rand = _select_matched_random(scores, top1, "area_fraction")
    if rand is not None:
        rnd = repaired_stats(predict_fn([neutralize_region(pil, rand.bbox)])[0])
    else:
        rnd = {"alias_correct": False, "target_prob_improved": False, "distractor_decreased": False}

    cov01 = any(_max_iou(s.bbox, text_boxes) >= 0.1 for s in selectable) if text_boxes else False
    cov03 = any(_max_iou(s.bbox, text_boxes) >= 0.3 for s in selectable) if text_boxes else False

    best_overlap_rank = float("nan")
    if text_boxes:
        for rank, s in enumerate(selectable, start=1):
            if _max_iou(s.bbox, text_boxes) >= 0.1:
                best_overlap_rank = float(rank)
                break

    return {
        "n_candidates": len(selectable),
        "cic_repair_alias_correct": bool(cic["alias_correct"]),
        "random_repair_alias_correct": bool(rnd["alias_correct"]),
        "cic_target_prob_improved": bool(cic["target_prob_improved"]),
        "random_target_prob_improved": bool(rnd.get("target_prob_improved", False)),
        "cic_distractor_decreased": bool(cic["distractor_decreased"]),
        "selected_text_overlap": bool(sel_text_overlap),
        "selected_object_overlap": bool(sel_obj_overlap),
        "selected_area_fraction": sel_area,
        "coverage_iou01": bool(cov01),
        "coverage_iou03": bool(cov03),
        "best_text_overlap_rank": best_overlap_rank,
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _rate(vals) -> float:
    arr = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean([bool(v) for v in arr])) if arr else float("nan")


def _mean(vals) -> float:
    arr = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(arr)) if arr else float("nan")


def _median(vals) -> float:
    arr = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.median(arr)) if arr else float("nan")


def _finite(v) -> float | None:
    return float(v) if (v is not None and np.isfinite(float(v))) else None


def _aggregate_family(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cic = _rate([r["cic_repair_alias_correct"] for r in rows])
    rnd = _rate([r["random_repair_alias_correct"] for r in rows])
    return {
        "n": len(rows),
        "cic_repair_alias_accuracy": cic,
        "matched_random_repair_alias_accuracy": rnd,
        "cic_random_gap": (cic - rnd) if (np.isfinite(cic) and np.isfinite(rnd)) else float("nan"),
        "cic_target_prob_improvement_rate": _rate([r["cic_target_prob_improved"] for r in rows]),
        "cic_text_distractor_decrease_rate": _rate([r["cic_distractor_decreased"] for r in rows]),
        "selected_text_overlap_rate": _rate([r["selected_text_overlap"] for r in rows]),
        "selected_object_overlap_rate": _rate([r["selected_object_overlap"] for r in rows]),
        "coverage_ceiling_iou01": _rate([r["coverage_iou01"] for r in rows]),
        "coverage_ceiling_iou03": _rate([r["coverage_iou03"] for r in rows]),
        "median_rank_best_text_overlap": _median([r["best_text_overlap_rank"] for r in rows]),
        "mean_selected_area_fraction": _mean([r["selected_area_fraction"] for r in rows]),
        "mean_candidates": _mean([r["n_candidates"] for r in rows]),
    }


def _paired_vs_baseline(fam_examples: dict[int, dict], base_examples: dict[int, dict]) -> dict[str, Any]:
    """Per-example wins/losses on CIC alias repair vs the a2a baseline."""

    wins = losses = ties = 0
    for eid, row in fam_examples.items():
        b = base_examples.get(eid)
        if b is None:
            continue
        f_ok = bool(row["cic_repair_alias_correct"])
        b_ok = bool(b["cic_repair_alias_correct"])
        if f_ok and not b_ok:
            wins += 1
        elif b_ok and not f_ok:
            losses += 1
        else:
            ties += 1
    n = wins + losses + ties
    return {"wins": wins, "losses": losses, "ties": ties, "n_paired": n, "net": wins - losses}


# --------------------------------------------------------------------------- #
# Promotion logic
# --------------------------------------------------------------------------- #
def _promotion_for_family(strict_base, strict_fam, dir_base, dir_fam) -> dict[str, Any]:
    """Apply the three promotion criteria for one automated family.

    All comparisons are against the apples-to-apples finalized baseline.
    """

    def g(m, k):
        return _finite(m.get(k)) if m else None

    s_repair_gain = _delta(g(strict_fam, "cic_repair_alias_accuracy"), g(strict_base, "cic_repair_alias_accuracy"))
    d_repair_drop = _delta(g(dir_base, "cic_repair_alias_accuracy"), g(dir_fam, "cic_repair_alias_accuracy"))
    s_overlap_gain = _delta(g(strict_fam, "selected_text_overlap_rate"), g(strict_base, "selected_text_overlap_rate"))
    s_repair_drop = _delta(g(strict_base, "cic_repair_alias_accuracy"), g(strict_fam, "cic_repair_alias_accuracy"))
    s_cov_gain = _delta(g(strict_fam, "coverage_ceiling_iou01"), g(strict_base, "coverage_ceiling_iou01"))

    # Criterion A: strict repair +0.05 without dropping directional repair > 0.03.
    crit_a = bool(
        s_repair_gain is not None
        and s_repair_gain >= PROMOTE_STRICT_REPAIR_GAIN
        and (d_repair_drop is None or d_repair_drop <= PROMOTE_STRICT_DIR_MAX_DROP)
    )
    # Criterion B: selected text overlap +0.15 without strict repair dropping > 0.05.
    crit_b = bool(
        s_overlap_gain is not None
        and s_overlap_gain >= PROMOTE_TEXT_OVERLAP_GAIN
        and (s_repair_drop is None or s_repair_drop <= PROMOTE_TEXT_OVERLAP_REPAIR_MAX_DROP)
    )
    # Criterion C: coverage ceiling +0.20 with repair approximately preserved.
    crit_c = bool(
        s_cov_gain is not None
        and s_cov_gain >= PROMOTE_COVERAGE_GAIN
        and (s_repair_drop is None or s_repair_drop <= PROMOTE_COVERAGE_REPAIR_TOL)
    )
    return {
        "strict_repair_gain_vs_a2a": s_repair_gain,
        "directional_repair_drop_vs_a2a": d_repair_drop,
        "strict_text_overlap_gain_vs_a2a": s_overlap_gain,
        "strict_repair_drop_vs_a2a": s_repair_drop,
        "strict_coverage_iou01_gain_vs_a2a": s_cov_gain,
        "criterion_A_strict_repair_+0.05": crit_a,
        "criterion_B_text_overlap_+0.15": crit_b,
        "criterion_C_coverage_+0.20": crit_c,
        "family_promotable": bool(crit_a or crit_b or crit_c),
    }


def _sam_promotion(strict_base, strict_sam, dir_base, dir_sam) -> dict[str, Any]:
    """Apply the task-specified SAM promotion rule against the a2a baseline.

    ``sam_promotable`` is True iff at least one criterion holds:
      A) SAM strict repair beats a2a by >= +0.05 absolute.
      B) SAM directional repair beats a2a by >= +0.05 absolute.
      C) SAM strict text overlap improves >= +0.15 while strict repair drops <= 0.05.
      D) SAM strict coverage ceiling (IoU>=0.1) improves >= +0.20 while strict
         repair drops <= 0.05.
    """

    def g(m, k):
        return _finite(m.get(k)) if m else None

    s_repair_gain = _delta(g(strict_sam, "cic_repair_alias_accuracy"), g(strict_base, "cic_repair_alias_accuracy"))
    d_repair_gain = _delta(g(dir_sam, "cic_repair_alias_accuracy"), g(dir_base, "cic_repair_alias_accuracy"))
    s_overlap_gain = _delta(g(strict_sam, "selected_text_overlap_rate"), g(strict_base, "selected_text_overlap_rate"))
    s_cov_gain = _delta(g(strict_sam, "coverage_ceiling_iou01"), g(strict_base, "coverage_ceiling_iou01"))
    s_repair_drop = _delta(g(strict_base, "cic_repair_alias_accuracy"), g(strict_sam, "cic_repair_alias_accuracy"))

    crit_a = bool(s_repair_gain is not None and s_repair_gain >= SAM_PROMOTE_STRICT_REPAIR_GAIN)
    crit_b = bool(d_repair_gain is not None and d_repair_gain >= SAM_PROMOTE_DIR_REPAIR_GAIN)
    crit_c = bool(
        s_overlap_gain is not None
        and s_overlap_gain >= SAM_PROMOTE_TEXT_OVERLAP_GAIN
        and (s_repair_drop is None or s_repair_drop <= SAM_PROMOTE_TEXT_OVERLAP_REPAIR_MAX_DROP)
    )
    crit_d = bool(
        s_cov_gain is not None
        and s_cov_gain >= SAM_PROMOTE_COVERAGE_GAIN
        and (s_repair_drop is None or s_repair_drop <= SAM_PROMOTE_COVERAGE_REPAIR_MAX_DROP)
    )
    return {
        "sam_strict_repair_gain_vs_a2a": s_repair_gain,
        "sam_directional_repair_gain_vs_a2a": d_repair_gain,
        "sam_strict_text_overlap_gain_vs_a2a": s_overlap_gain,
        "sam_strict_coverage_iou01_gain_vs_a2a": s_cov_gain,
        "sam_strict_repair_drop_vs_a2a": s_repair_drop,
        "criterion_A_sam_strict_repair_+0.05": crit_a,
        "criterion_B_sam_directional_repair_+0.05": crit_b,
        "criterion_C_sam_text_overlap_+0.15": crit_c,
        "criterion_D_sam_coverage_+0.20": crit_d,
        "sam_promotable": bool(crit_a or crit_b or crit_c or crit_d),
    }


def _delta(a, b) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


# --------------------------------------------------------------------------- #
# JSON helpers / outputs
# --------------------------------------------------------------------------- #
def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(f):
        return "n/a"
    return f"{f:.3f}"


def _resolve_families(spec: str | None) -> list[str]:
    if not spec or spec == "classical":
        return list(CLASSICAL_FAMILIES)
    if spec == "all":
        return list(ALL_FAMILIES)
    return [f.strip() for f in spec.split(",") if f.strip()]


def _write_outputs(out_dir: Path, metrics: dict[str, Any], table_rows: list[dict[str, Any]], per_example_rows: list[dict[str, Any]]) -> dict[str, str]:
    ensure_dir(out_dir)
    metrics_path = out_dir / "coco_text_auto_proposal_sweep_metrics.json"
    table_path = out_dir / "coco_text_auto_proposal_sweep_table.csv"
    per_path = out_dir / "coco_text_auto_proposal_sweep_per_example.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    pd.DataFrame(table_rows or [{"subset": "", "family": "", "note": "no rows"}]).to_csv(table_path, index=False)
    pd.DataFrame(per_example_rows or [{"subset": "", "family": "", "example_id": -1}]).to_csv(per_path, index=False)
    return {"metrics": str(metrics_path), "table": str(table_path), "per_example": str(per_path)}


def _skip(out_dir: Path, reason: str, extra: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "experiment": "coco_text_auto_proposal_sweep",
        "status": "skipped",
        "skip_reason": reason,
        "auto_proposal_promotable": False,
        "non_claims": NON_CLAIMS,
        **extra,
    }
    _write_outputs(out_dir, metrics, [], [])
    write_sweep_summary(out_dir)
    print(json.dumps({"status": "skipped", "reason": reason}, indent=2))
    return metrics


# --------------------------------------------------------------------------- #
# Main run
# --------------------------------------------------------------------------- #
def run(args) -> dict[str, Any]:
    import time as _time

    from PIL import Image

    run_start = _time.monotonic()
    out_dir = ensure_dir(Path(args.results_dir) / RESULTS_SUBDIR)
    seed = int(args.seed)
    np.random.seed(seed)
    families = _resolve_families(args.families)
    if args.include_sam and "sam_boxes" not in families:
        families = families + ["sam_boxes"]
    sam_kwargs = _sam_kwargs(args) if args.include_sam else None

    avail = generator_availability(allow_download=bool(args.allow_download))
    if args.include_sam:
        # Probe the real SAM family explicitly with the --sam-* settings. This loads
        # (and caches) the checkpoint once so per-example generation reuses it, and
        # reports a clean availability/skip_reason if the package/checkpoint is absent.
        # The probe never reads/writes the proposal cache (random tiny image).
        from causal_reliability.proposals.auto_proposals import sam_boxes as _sam_probe

        probe_kwargs = {**sam_kwargs, "cache_dir": None, "use_cache": False}
        probe_img = Image.fromarray((np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8))
        avail["sam_boxes"] = _sam_probe(
            probe_img, allow_download=bool(args.allow_download), max_boxes=int(args.sam_max_proposals), **probe_kwargs
        )
    gens_available = {k: v.available for k, v in avail.items()}
    gens_skipped = {k: v.skip_reason for k, v in avail.items() if not v.available}
    auto_families = [f for f in families if gens_available.get(f, False)]
    eval_families = [BASELINE_FAMILY] + auto_families

    sam_block = None
    if args.include_sam:
        from causal_reliability.proposals.auto_proposals import _resolve_device as _sam_resolve_device

        sam_ps = avail.get("sam_boxes")
        sam_block = {
            "requested": True,
            "loaded": bool(sam_ps.available) if sam_ps is not None else False,
            "skip_reason": (sam_ps.skip_reason if (sam_ps is not None and not sam_ps.available) else ""),
            "settings": _sam_config_record(args),
            "resolved_device": _sam_resolve_device(args.sam_device),
        }

    base_extra = {
        "generators_available": gens_available,
        "generators_skipped": gens_skipped,
        "families_evaluated": eval_families,
        "baseline_family": BASELINE_FAMILY,
        "a2a_max_candidates": A2A_MAX_CANDIDATES,
        "a2a_grid_scales": A2A_GRID_SCALES,
        "max_examples": int(args.max_examples),
        "sam": sam_block,
    }

    metadata_csv = Path(args.metadata_csv)
    if not metadata_csv.exists():
        return _skip(out_dir, f"COCO-Text metadata not found at {metadata_csv}", base_extra)
    bundle = load_local_folder_dataset(Path(args.data_root), metadata_csv, image_size=int(args.image_size))
    if not bundle.examples:
        return _skip(out_dir, "COCO-Text metadata loaded but no examples available", base_extra)
    by_id = {int(ex["example_id"]): ex for ex in bundle.examples}

    # Resolve which subsets to evaluate, with the SAM all_500 confirmation gate.
    chosen, subset_gating = resolve_subsets_and_gating(args)
    base_extra["subsets_chosen"] = chosen
    base_extra["subset_gating"] = subset_gating

    want_all500 = "all_500" in chosen
    subsets = load_subsets(Path(args.triage_dir), Path(args.full_per_example), by_id, want_all500)
    if not any(subsets.get(s) for s in chosen):
        return _skip(out_dir, f"no requested subsets available under {args.triage_dir}", base_extra)

    if args.backend == "fake":
        return _skip(out_dir, "fake backend cannot support a sweep result", base_extra)
    status = check_clip_available(
        device=args.device,
        allow_download=bool(args.allow_download),
        preferred_backend=args.backend,
        model_name=DEFAULT_MODEL_NAME,
        pretrained_tag=DEFAULT_PRETRAINED_TAG,
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return _skip(
            out_dir,
            f"real pretrained CLIP unavailable ({status.error_message or 'no cached weights'}); pass --allow-download to fetch",
            {**base_extra, "backend": status.backend, "pretrained_loaded": bool(status.pretrained)},
        )

    predict_cache: dict[tuple, Any] = {}
    subset_results: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    per_example_rows: list[dict[str, Any]] = []

    # SAM telemetry + runtime guard. ``sam_seconds`` accumulates *generation* time
    # only (cache hits are ~free); when it exceeds the budget we stop cleanly and
    # write partial results rather than running for hours.
    sam_seconds = 0.0
    sam_cache_hits = 0
    sam_cache_misses = 0
    sam_images_completed = 0
    sam_timed_out = False
    sam_active = bool(args.include_sam and "sam_boxes" in auto_families)
    sam_timeout = float(args.sam_timeout_seconds) if sam_active else float("inf")
    subset_completed: dict[str, bool] = {}

    for subset_name in chosen:
        if sam_timed_out:
            break
        ids = [i for i in subsets.get(subset_name, []) if i in by_id]
        if int(args.max_examples) > 0:
            ids = ids[: int(args.max_examples)]
        if not ids:
            continue
        fam_rows: dict[str, list[dict[str, Any]]] = {f: [] for f in eval_families}
        fam_examples: dict[str, dict[int, dict]] = {f: {} for f in eval_families}
        orig_correct: list[bool] = []
        used = 0
        for eid in ids:
            # Runtime guard: stop before starting another SAM-bearing example once
            # the SAM generation budget is spent (overshoot bounded to one example).
            if sam_active and sam_seconds >= sam_timeout:
                sam_timed_out = True
                break
            ex = by_id[eid]
            allowed = list(ex["allowed_clip_labels"])
            target = str(ex["human_label"])
            aliases = aliases_for(target, extra=set(ex.get("target_aliases", [])))
            target_idxs = [i for i, lbl in enumerate(allowed) if is_target_label(lbl, target, aliases)]
            if not target_idxs:
                continue
            key = tuple(allowed)
            predict_fn = predict_cache.get(key)
            if predict_fn is None:
                predict_fn = _build_predict_fn(status, allowed, args.device, int(args.predict_batch_size))
                predict_cache[key] = predict_fn

            pil = Image.fromarray((np.asarray(ex["image"]).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")
            text_boxes = [tuple(int(v) for v in b) for b in ex.get("text_boxes", [])]
            object_boxes = [tuple(int(v) for v in b) for b in ex.get("object_boxes", [])]

            original_probs = predict_fn([pil])[0]
            non_target = [i for i in range(len(allowed)) if i not in set(target_idxs)]
            distractor_idx = int(max(non_target, key=lambda i: original_probs[i])) if non_target else int(target_idxs[0])
            orig_correct.append(bool(is_target_label(allowed[int(np.asarray(original_probs).argmax())], target, aliases)))
            used += 1

            for fam in eval_families:
                if fam == BASELINE_FAMILY:
                    scores, fam_orig = _baseline_a2a_scores(pil, text_boxes, object_boxes, predict_fn, eid, seed)
                else:
                    if fam == "sam_boxes":
                        fam_kwargs = {**sam_kwargs, "image_id": int(eid)}
                        fam_cap = int(args.sam_max_proposals)
                    else:
                        fam_kwargs = None
                        fam_cap = int(args.max_candidates)
                    scores, fam_orig, ps = _auto_family_scores(
                        pil, fam, predict_fn, eid, seed, bool(args.allow_download), fam_cap, family_kwargs=fam_kwargs
                    )
                    if fam == "sam_boxes":
                        if ps.cache_hit:
                            sam_cache_hits += 1
                        else:
                            sam_cache_misses += 1
                            sam_seconds += float(ps.gen_seconds or 0.0)
                        sam_images_completed += 1
                row = _evaluate_example(
                    pil, scores, fam_orig, predict_fn,
                    allowed=allowed, target=target, aliases=aliases,
                    target_idxs=target_idxs, distractor_idx=distractor_idx,
                    text_boxes=text_boxes, object_boxes=object_boxes, prob_eps=DEFAULT_PROB_EPS,
                )
                fam_rows[fam].append(row)
                fam_examples[fam][eid] = row
                per_example_rows.append({"subset": subset_name, "example_id": eid, "family": fam, **row})

        # The subset is "completed" iff we evaluated every requested example
        # without tripping the SAM runtime guard mid-subset.
        subset_completed[subset_name] = (not sam_timed_out) and (used == len([i for i in (
            subsets.get(subset_name, [])[: int(args.max_examples)] if int(args.max_examples) > 0
            else subsets.get(subset_name, [])
        ) if i in by_id]))

        if used == 0:
            continue
        fam_metrics = {fam: _aggregate_family(rows) for fam, rows in fam_rows.items() if rows}
        paired = {
            fam: _paired_vs_baseline(fam_examples[fam], fam_examples[BASELINE_FAMILY])
            for fam in auto_families
        }
        subset_results[subset_name] = {
            "n": used,
            "original_alias_accuracy": _rate(orig_correct),
            "families": fam_metrics,
            "paired_vs_baseline": paired,
        }
        for fam, m in fam_metrics.items():
            table_rows.append({"subset": subset_name, "family": fam, "n": used,
                               "original_alias_accuracy": _rate(orig_correct), **m})

    # Promotion: compare automated families against the a2a baseline using strict
    # and directional subset metrics.
    strict = subset_results.get("strict", {}).get("families", {})
    directional = subset_results.get("directional", {}).get("families", {})
    strict_base = strict.get(BASELINE_FAMILY)
    dir_base = directional.get(BASELINE_FAMILY)
    promotion: dict[str, Any] = {}
    for fam in auto_families:
        promotion[fam] = _promotion_for_family(
            strict_base, strict.get(fam), dir_base, directional.get(fam)
        )
    auto_proposal_promotable = any(v["family_promotable"] for v in promotion.values())

    # Record SAM runtime/cache telemetry on the sam block (in place — base_extra
    # already references it).
    strict_complete = bool(subset_completed.get("strict", False))
    if sam_block is not None:
        sam_block["images_completed"] = int(sam_images_completed)
        sam_block["cache_hits"] = int(sam_cache_hits)
        sam_block["cache_misses"] = int(sam_cache_misses)
        sam_block["generation_seconds"] = round(float(sam_seconds), 3)
        sam_block["timed_out"] = bool(sam_timed_out)
        sam_block["partial"] = bool(sam_timed_out)
        sam_block["timeout_seconds"] = float(args.sam_timeout_seconds)
        sam_block["strict_complete"] = strict_complete
        sam_block["total_runtime_seconds"] = round(_time.monotonic() - run_start, 3)
        sam_block["subset_completed"] = subset_completed

    # Dedicated SAM promotion rule (task-specified, distinct from the generic rule).
    # sam_promotable can only be True if the strict run completed fully (no timeout)
    # AND a pre-registered criterion is cleared.
    sam_promotion: dict[str, Any] | None = None
    sam_promotable = False
    if "sam_boxes" in auto_families:
        sam_promotion = _sam_promotion(
            strict_base, strict.get("sam_boxes"), dir_base, directional.get("sam_boxes")
        )
        sam_promotion["strict_complete"] = strict_complete
        sam_promotion["sam_timed_out"] = bool(sam_timed_out)
        sam_promotable = bool(sam_promotion["sam_promotable"]) and strict_complete and not sam_timed_out

    metrics = {
        "experiment": "coco_text_auto_proposal_sweep",
        "status": "ok",
        "backend": status.backend,
        "model_name": status.model_name,
        "pretrained_loaded": bool(status.pretrained),
        **base_extra,
        "reconciliation": {
            "finalized_strict_cic_repair": 0.5384615384615384,
            "finalized_strict_matched_random_repair": 0.20512820512820512,
            "a2a_reproduces_finalized": _a2a_matches(strict_base),
            "pilot_baseline_strict_cic_repair": 0.41025641025641024,
            "pilot_baseline_directly_comparable": False,
            "reconciliation_doc": "results/auto_proposal_pilot/coco_reconciliation.md",
        },
        "subsets": subset_results,
        "promotion": promotion,
        "promotion_thresholds": {
            "strict_repair_gain": PROMOTE_STRICT_REPAIR_GAIN,
            "strict_directional_max_drop": PROMOTE_STRICT_DIR_MAX_DROP,
            "text_overlap_gain": PROMOTE_TEXT_OVERLAP_GAIN,
            "text_overlap_repair_max_drop": PROMOTE_TEXT_OVERLAP_REPAIR_MAX_DROP,
            "coverage_gain": PROMOTE_COVERAGE_GAIN,
            "coverage_repair_tol": PROMOTE_COVERAGE_REPAIR_TOL,
        },
        "auto_proposal_promotable": bool(auto_proposal_promotable),
        "sam_promotion": sam_promotion,
        "sam_promotion_thresholds": {
            "sam_strict_repair_gain": SAM_PROMOTE_STRICT_REPAIR_GAIN,
            "sam_directional_repair_gain": SAM_PROMOTE_DIR_REPAIR_GAIN,
            "sam_text_overlap_gain": SAM_PROMOTE_TEXT_OVERLAP_GAIN,
            "sam_text_overlap_repair_max_drop": SAM_PROMOTE_TEXT_OVERLAP_REPAIR_MAX_DROP,
            "sam_coverage_gain": SAM_PROMOTE_COVERAGE_GAIN,
            "sam_coverage_repair_max_drop": SAM_PROMOTE_COVERAGE_REPAIR_MAX_DROP,
        },
        "sam_promotable": bool(sam_promotable),
        "non_claims": NON_CLAIMS,
    }
    paths = _write_outputs(out_dir, metrics, table_rows, per_example_rows)
    write_sweep_summary(out_dir)
    print(json.dumps({
        "status": "ok",
        "auto_proposal_promotable": bool(auto_proposal_promotable),
        "sam_promotable": bool(sam_promotable),
        **paths,
    }, indent=2))
    return metrics


def _a2a_matches(strict_base: dict | None) -> bool:
    if not strict_base:
        return False
    v = strict_base.get("cic_repair_alias_accuracy")
    return bool(v is not None and np.isfinite(float(v)) and abs(float(v) - 0.5384615384615384) < 1e-6)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def write_sweep_summary(out_dir: Path) -> str:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    metrics_path = out_dir / "coco_text_auto_proposal_sweep_metrics.json"
    m = json.loads(metrics_path.read_text()) if metrics_path.exists() else None

    lines: list[str] = [
        "# COCO-Text Automated Proposal Sweep — Apples-to-Apples Summary",
        "",
        "This is **automated finite-candidate proposal generation**: CIC still scores a ",
        "finite candidate set; only the *source* of the candidate boxes changes (manual ",
        "open-proposal generator vs. automatic grid / edge-component / saliency generators). ",
        "Every family is scored and evaluated with identical code, and compared **only** ",
        "against the apples-to-apples finalized baseline `existing_cic_baseline_a2a` that ",
        "exactly reproduces the finalized `cic_top1_repair_excl_ocr` recipe.",
        "",
        "## Non-claims (explicit, bounded language)",
        "",
        "- This is **not** open-world shortcut discovery.",
        "- This is **not** universal repair or general robustness.",
        "- This is **not** deployment validation or clinical validation.",
        "- This is **not** a replacement for the finalized STS report.",
        "",
    ]

    if m is None:
        lines += ["_No sweep metrics written yet._", ""]
        path = out_dir / "full_coco_sweep_summary.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    rec = m.get("reconciliation", {})
    lines += [
        "## 1. Reconciliation result",
        "",
        f"- Finalized strict CIC repair (headline): **{_fmt(rec.get('finalized_strict_cic_repair'))}** "
        f"(matched random {_fmt(rec.get('finalized_strict_matched_random_repair'))}).",
        f"- First pilot `existing_cic_baseline` strict CIC repair: **{_fmt(rec.get('pilot_baseline_strict_cic_repair'))}**.",
        f"- Pilot baseline directly comparable to finalized result: **{rec.get('pilot_baseline_directly_comparable')}** "
        "(different candidate cap 14 vs 48, different grid scales, and OCR geometry disabled — "
        "see `coco_reconciliation.md`).",
        f"- Apples-to-apples baseline reproduces the finalized strict number exactly: "
        f"**{rec.get('a2a_reproduces_finalized')}**.",
        "",
    ]

    if m.get("status") != "ok":
        lines += [f"## Status: **{m.get('status')}** — {m.get('skip_reason','')}", ""]
        path = out_dir / "full_coco_sweep_summary.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    lines += [
        f"- Backend: `{m.get('backend')}` ({m.get('model_name')}); a2a candidate cap "
        f"{m.get('a2a_max_candidates')}, grid scales {m.get('a2a_grid_scales')}.",
        f"- Families evaluated: {', '.join(m.get('families_evaluated', []))}.",
        "",
    ]

    sam = m.get("sam")
    if sam:
        st = sam.get("settings", {})
        if sam.get("loaded"):
            timed_out = bool(sam.get("timed_out"))
            run_state = (
                "**timed out (partial run)**" if timed_out
                else ("**complete**" if sam.get("strict_complete") else "**partial**")
            )
            lines += [
                "## 1b. SAM (Segment Anything) status",
                "",
                f"- SAM loaded successfully: **True** (device `{sam.get('resolved_device')}`); run state: {run_state}.",
                f"- Settings: model_type=`{st.get('model_type')}`, checkpoint=`{st.get('checkpoint_path')}`, "
                f"fast={st.get('fast')}, points_per_side={st.get('points_per_side')}, crop_n_layers={st.get('crop_n_layers')}, "
                f"max_side={st.get('max_side')}, pred_iou_thresh={st.get('pred_iou_thresh')}, "
                f"stability_score_thresh={st.get('stability_score_thresh')}, max_proposals={st.get('max_proposals')}, "
                f"area_frac∈[{st.get('min_area_frac')}, {st.get('max_area_frac')}], min_side={st.get('min_side')}, "
                f"dedupe_iou={st.get('dedupe_iou')}.",
                f"- Runtime: {_fmt(sam.get('generation_seconds'))} s SAM generation "
                f"(total run {_fmt(sam.get('total_runtime_seconds'))} s); images completed: "
                f"{sam.get('images_completed')}; cache hits/misses: {sam.get('cache_hits')}/{sam.get('cache_misses')}; "
                f"timeout budget {_fmt(sam.get('timeout_seconds'))} s.",
                (
                    "- **SAM timed out** before finishing: results below are a bounded **partial** run "
                    "(not a silent failure). `sam_promotable` is forced False until a full strict run completes."
                    if timed_out
                    else "- SAM masks → XYWH→XYXY boxes, downscaled (max_side) for speed, filtered by side/area, "
                    "IoU-deduplicated, top-K by predicted_iou / stability_score / area, and cached per image. "
                    "This stays **automated finite-candidate proposal generation**, not open-world shortcut discovery."
                ),
                "",
            ]
        else:
            lines += [
                "## 1b. SAM (Segment Anything) status",
                "",
                f"- SAM loaded successfully: **False** — {sam.get('skip_reason','(no reason)')}. SAM was "
                "requested but skipped cleanly; the classical families and the a2a baseline are unaffected.",
                "",
            ]

    gating = m.get("subset_gating") or []
    if gating:
        lines += ["## 1c. Subset gating", ""] + [f"- {g}" for g in gating] + [""]

    lines += [
        "## 2. Full strict / directional / all-500 table",
        "",
    ]
    for sname in ("strict", "directional", "all_500"):
        s = m.get("subsets", {}).get(sname)
        if not s:
            continue
        lines.append(f"### Subset `{sname}` (n={s.get('n')}, original alias accuracy={_fmt(s.get('original_alias_accuracy'))})")
        lines.append("")
        lines.append("| family | CIC repair | random repair | gap | tgt-prob↑ | distr↓ | sel text-ovl | sel obj-ovl | cov@.1 | cov@.3 | med rank | sel area | wins/losses vs a2a |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        paired = s.get("paired_vs_baseline", {})
        for fam, mm in s.get("families", {}).items():
            if fam == BASELINE_FAMILY:
                wl = "— (baseline)"
            else:
                p = paired.get(fam, {})
                wl = f"{p.get('wins','?')}/{p.get('losses','?')} (net {p.get('net','?')})"
            lines.append(
                f"| {fam} | {_fmt(mm.get('cic_repair_alias_accuracy'))} | {_fmt(mm.get('matched_random_repair_alias_accuracy'))} | "
                f"{_fmt(mm.get('cic_random_gap'))} | {_fmt(mm.get('cic_target_prob_improvement_rate'))} | "
                f"{_fmt(mm.get('cic_text_distractor_decrease_rate'))} | {_fmt(mm.get('selected_text_overlap_rate'))} | "
                f"{_fmt(mm.get('selected_object_overlap_rate'))} | {_fmt(mm.get('coverage_ceiling_iou01'))} | "
                f"{_fmt(mm.get('coverage_ceiling_iou03'))} | {_fmt(mm.get('median_rank_best_text_overlap'))} | "
                f"{_fmt(mm.get('mean_selected_area_fraction'))} | {wl} |"
            )
        lines.append("")

    lines += ["## 3. Do automated proposals genuinely improve over the finalized baseline?", ""]
    promotion = m.get("promotion", {})
    for fam, v in promotion.items():
        lines.append(
            f"- `{fam}`: strict repair gain {_fmt(v.get('strict_repair_gain_vs_a2a'))}, "
            f"directional repair drop {_fmt(v.get('directional_repair_drop_vs_a2a'))}, "
            f"text-overlap gain {_fmt(v.get('strict_text_overlap_gain_vs_a2a'))}, "
            f"coverage gain {_fmt(v.get('strict_coverage_iou01_gain_vs_a2a'))} → "
            f"**promotable={v.get('family_promotable')}** "
            f"(A={v.get('criterion_A_strict_repair_+0.05')}, B={v.get('criterion_B_text_overlap_+0.15')}, "
            f"C={v.get('criterion_C_coverage_+0.20')})"
        )
    lines.append("")

    sam_prom = m.get("sam_promotion")
    if sam_prom is not None:
        lines += ["### SAM-specific promotion rule", ""]
        lines.append(
            f"- `sam_boxes`: strict repair gain {_fmt(sam_prom.get('sam_strict_repair_gain_vs_a2a'))}, "
            f"directional repair gain {_fmt(sam_prom.get('sam_directional_repair_gain_vs_a2a'))}, "
            f"text-overlap gain {_fmt(sam_prom.get('sam_strict_text_overlap_gain_vs_a2a'))}, "
            f"coverage gain {_fmt(sam_prom.get('sam_strict_coverage_iou01_gain_vs_a2a'))} → "
            f"**sam_promotable={m.get('sam_promotable')}** "
            f"(A={sam_prom.get('criterion_A_sam_strict_repair_+0.05')}, "
            f"B={sam_prom.get('criterion_B_sam_directional_repair_+0.05')}, "
            f"C={sam_prom.get('criterion_C_sam_text_overlap_+0.15')}, "
            f"D={sam_prom.get('criterion_D_sam_coverage_+0.20')})"
        )
        lines.append("")

    promotable = bool(m.get("auto_proposal_promotable"))
    lines += [
        "## 4. Promotion verdict",
        "",
        f"- **auto_proposal_promotable = {promotable}**",
        f"- **sam_promotable = {bool(m.get('sam_promotable'))}**" if m.get("sam") else "",
        (
            "- At least one automated proposal family beats the apples-to-apples finalized "
            "baseline by a pre-registered margin. This is promotable as a *finite-candidate "
            "automated-proposal* result for a conference version, **with** the explicit "
            "non-claims below."
            if promotable
            else
            "- No automated proposal family beats the apples-to-apples finalized baseline by "
            "a pre-registered margin. Preserved honestly as a negative/diagnostic result: "
            "automated proposals are competitive but **not** a promotable improvement over "
            "the finalized hand-designed candidate set."
        ),
        "",
        "## 5. Explicit non-claims",
        "",
        "- This is **not** open-world discovery — it is automated finite-candidate proposal generation.",
        "- This is **not** universal repair.",
        "- This is **not** deployment validation.",
        "- This is **not** a replacement for the finalized STS report; the finalized COCO-Text "
        "and STS numbers and support gates are unchanged.",
        "",
    ]
    path = out_dir / "full_coco_sweep_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", default=None, help="comma list of {strict,directional,all_500} or 'auto' (default: strict,directional; strict only when --include-sam)")
    p.add_argument("--families", default="classical", help="'classical', 'all', or comma list of generator families")
    p.add_argument("--include-sam", action="store_true", help="add the real SAM (sam_boxes) family alongside the classical families")
    p.add_argument("--strict-only", action="store_true", help="evaluate only the strict subset")
    p.add_argument("--no-all500", action="store_true", help="never evaluate the all_500 subset")
    p.add_argument("--all500", action="store_true", help="request the all_500 subset (with SAM also needs --confirm-slow-sam)")
    p.add_argument("--confirm-slow-sam", action="store_true", help="explicitly confirm an intentionally slow all_500 SAM run")
    p.add_argument("--max-examples", type=int, default=0, help="cap per subset (0 = no cap / use all)")
    p.add_argument("--max-candidates", type=int, default=48, help="max auto candidates per family (excl. random controls)")
    p.add_argument("--allow-download", action="store_true")
    # SAM (Segment Anything) knobs — only used when --include-sam is set. The
    # checkpoint is NOT committed (gitignored); place it at the default path.
    p.add_argument("--sam-fast", action="store_true", help="aggressive fast SAM defaults (points_per_side=8, crop_n_layers=0, max_side=512)")
    p.add_argument("--sam-checkpoint", default="models/sam/sam_vit_b_01ec64.pth")
    p.add_argument("--sam-model-type", default="vit_b")
    p.add_argument("--sam-points-per-side", type=int, default=None, help="default 16 (8 with --sam-fast)")
    p.add_argument("--sam-pred-iou-thresh", type=float, default=0.86)
    p.add_argument("--sam-stability-score-thresh", type=float, default=0.90)
    p.add_argument("--sam-max-proposals", type=int, default=48)
    p.add_argument("--sam-min-area-frac", type=float, default=0.002)
    p.add_argument("--sam-max-area-frac", type=float, default=0.80)
    p.add_argument("--sam-min-side", type=int, default=8)
    p.add_argument("--sam-dedupe-iou", type=float, default=0.7)
    p.add_argument("--sam-crop-n-layers", type=int, default=0)
    p.add_argument("--sam-max-side", type=int, default=None, help="downscale longest image side before SAM; 0 disables (default 0; 512 with --sam-fast)")
    p.add_argument("--sam-device", default="cpu", help="auto/cpu/cuda/mps (cpu default: empirically faster than mps for SAM AMG here)")
    p.add_argument("--sam-timeout-seconds", type=float, default=1800.0, help="SAM generation wall-clock budget; on exceed, stop cleanly + write partial results")
    p.add_argument("--cache-sam-proposals", action="store_true", help="cache per-image SAM proposals under results/auto_proposal_pilot/cache/sam_proposals/")
    p.add_argument("--resume", action="store_true", help="reuse cached SAM proposals (implies reading the SAM proposal cache)")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--data-root", default="data/coco_text_cic")
    p.add_argument("--metadata-csv", default="data/coco_text_cic/metadata.csv")
    p.add_argument("--triage-dir", default="results/coco_text_cic_triage")
    p.add_argument("--full-per-example", default="results/coco_text_cic_full/coco_text_full_per_example.csv")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--backend", default="open_clip")
    p.add_argument("--device", default="cpu")
    p.add_argument("--predict-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
