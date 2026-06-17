from __future__ import annotations

"""Automated finite-candidate proposal CIC pilot on COCO-Text natural images.

Experiment name: ``auto_proposal_cic_pilot``.

Scientific question
-------------------
Can *automatic* proposal generation reduce CIC's proposal bottleneck on natural
images? CIC still scores a **finite** candidate set; the only change versus the
existing COCO-Text pipeline is that the candidate set is generated automatically
from pixels (grid / edge-component / saliency / optional SAM / optional DINO)
rather than from the manually-designed open-proposal generator tuned around the
text-shortcut family.

This is **automated finite-candidate proposal generation**, NOT guaranteed
open-world shortcut discovery. We compare each automatic proposal family against
the existing CIC proposal baseline (``generate_open_region_proposals``) on the
verified COCO-Text strict / directional subsets, capped to a small pilot.

Honesty / scope
---------------
* Writes ONLY under ``results/auto_proposal_pilot/``. It cannot touch the
  finalized report, the headline COCO-Text artifacts, or any existing result JSON/CSV.
* Candidate scoring sees only pixels, proposal geometry, and model predictions.
* If results are weak this script preserves them honestly: ``pilot_promising``
  becomes ``False`` and the summary records it as a negative/diagnostic result.
* No model downloads unless ``--allow-download`` is passed.
* In a clean skip (no real CLIP, no data) it records the reason and exits 0.
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

# Make the repo root importable when run as ``python3 experiments/<file>.py``.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_reliability.data.natural_text_dataset import load_local_folder_dataset, parse_label_list
from causal_reliability.discovery.cic_region_scoring import neutralize_region, score_region_candidates
from causal_reliability.discovery.open_region_proposals import generate_open_region_proposals
from causal_reliability.experiments.run_coco_text_cic_triage import (
    aliases_for,
    is_target_label,
    label_rank,
    label_set_prob,
    pairwise_margin_toward_target,
)
from causal_reliability.proposals.auto_proposals import (
    ALL_FAMILIES,
    CLASSICAL_FAMILIES,
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

# The existing manually-designed CIC proposal generator is the baseline family.
BASELINE_FAMILY = "existing_cic_baseline"

NON_CLAIMS = [
    "This is automated finite-candidate proposal generation, NOT open-world shortcut discovery.",
    "This is NOT universal repair or general robustness.",
    "This is NOT deployment validation or clinical validation.",
    "This is NOT a replacement for the finalized STS report.",
]

# Go/no-go thresholds (COCO-Text).
PROMISE_REPAIR_GAIN = 0.10
PROMISE_TEXT_OVERLAP_GAIN = 0.15
PROMISE_REPAIR_MAX_DROP = 0.05
PROMISE_COVERAGE_GAIN = 0.20


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _pil_to_tensor(images):
    import torch
    from PIL import Image

    arrays = [np.asarray(img.convert("RGB")).astype(np.float32) / 255.0 for img in images]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


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


def _select_matched_random(scores: list, top1, field: str = "area_fraction"):
    randoms = [s for s in scores if s.proposal_type == "random_patch_control"]
    if not randoms:
        return None
    target = float(getattr(top1, field, 0.0)) if top1 is not None else 0.0
    return min(randoms, key=lambda s: abs(float(getattr(s, field, 0.0)) - target))


# --------------------------------------------------------------------------- #
# Subset id loading from triage artifacts
# --------------------------------------------------------------------------- #
def _read_ids(csv_path: Path) -> list[int]:
    if not csv_path.exists():
        return []
    frame = pd.read_csv(csv_path)
    if "example_id" not in frame.columns:
        return []
    return [int(v) for v in frame["example_id"].tolist()]


def _distractor_map(csv_path: Path) -> dict[int, str]:
    if not csv_path.exists():
        return {}
    frame = pd.read_csv(csv_path)
    if "example_id" not in frame.columns or "distractor_label" not in frame.columns:
        return {}
    out: dict[int, str] = {}
    for _, row in frame.iterrows():
        d = row.get("distractor_label")
        if isinstance(d, str) and d:
            out[int(row["example_id"])] = d
    return out


def load_subsets(triage_dir: Path) -> dict[str, dict[str, Any]]:
    strict_csv = triage_dir / "coco_text_verified_oracle_repairable_failures.csv"
    dir_csv = triage_dir / "coco_text_verified_directional_failures.csv"
    return {
        "strict": {"ids": _read_ids(strict_csv), "distractors": _distractor_map(strict_csv)},
        "directional": {"ids": _read_ids(dir_csv), "distractors": _distractor_map(dir_csv)},
    }


# --------------------------------------------------------------------------- #
# Candidate generation per family
# --------------------------------------------------------------------------- #
def _baseline_region_proposals(pil, text_boxes, seed: int, max_candidates: int):
    """The existing manually-designed CIC open proposals (excluding OCR/object).

    We deliberately exclude the OCR/text-box geometry family so the baseline is a
    genuine pixel-driven proposal set, matching what the automatic families do.
    """

    return generate_open_region_proposals(
        pil,
        text_boxes=None,
        object_boxes=None,
        seed=seed,
        max_candidates=max_candidates,
        enable_ocr_family=False,
        enable_object_box_family=False,
        enable_random_control=True,
    )


def _auto_region_proposals(pil, family: str, seed: int, max_candidates: int, allow_download: bool):
    sets = generate_proposal_sets(
        pil, [family], allow_download=allow_download, max_boxes=max_candidates, seed=seed
    )
    ps = sets[family]
    rps = proposal_sets_to_region_proposals(pil, [ps], include_random_control=True, seed=seed)
    return rps, ps


# --------------------------------------------------------------------------- #
# Per-example evaluation for one family
# --------------------------------------------------------------------------- #
def _orig_stats(original_probs, allowed, target, aliases, target_idxs, distractor_idx) -> dict[str, Any]:
    op = np.asarray(original_probs, dtype=np.float64)
    pred = int(op.argmax())
    return {
        "pred_label": allowed[pred],
        "alias_correct": bool(is_target_label(allowed[pred], target, aliases)),
        "target_prob": label_set_prob(op, target_idxs),
        "target_rank": label_rank(op, target_idxs),
        "distractor_prob": float(op[distractor_idx]),
        "pairwise_margin": pairwise_margin_toward_target(label_set_prob(op, target_idxs), float(op[distractor_idx])),
    }


def _evaluate_family_example(
    pil,
    scores: list,
    proposal_set,
    predict_fn,
    *,
    allowed,
    target,
    aliases,
    target_idxs,
    distractor_idx,
    orig,
    text_boxes,
    object_boxes,
    prob_eps: float,
) -> dict[str, Any]:
    """Compute one example's metrics for one proposal family."""

    # Non-random candidates only count toward coverage / localization.
    non_random = [s for s in scores if s.proposal_type != "random_patch_control"]
    top1 = non_random[0] if non_random else (scores[0] if scores else None)

    def repaired_stats(probs) -> dict[str, Any]:
        p = np.asarray(probs, dtype=np.float64)
        pred = int(p.argmax())
        t_prob = label_set_prob(p, target_idxs)
        d_prob = float(p[distractor_idx])
        return {
            "alias_correct": bool(is_target_label(allowed[pred], target, aliases)),
            "target_prob": t_prob,
            "distractor_prob": d_prob,
            "target_prob_improved": bool(t_prob > orig["target_prob"] + prob_eps),
            "distractor_decreased": bool(d_prob < orig["distractor_prob"] - prob_eps),
        }

    # CIC top-1 repair.
    if top1 is not None:
        cic = repaired_stats(predict_fn([neutralize_region(pil, top1.bbox)])[0])
        sel_text_overlap = _overlaps_any(top1.bbox, text_boxes)
        sel_obj_overlap = _overlaps_any(top1.bbox, object_boxes)
        sel_area = float(top1.area_fraction)
    else:
        cic = repaired_stats(orig and predict_fn([pil])[0])
        sel_text_overlap = False
        sel_obj_overlap = False
        sel_area = float("nan")

    # Matched-random repair (area-matched control).
    rand = _select_matched_random(scores, top1, "area_fraction")
    if rand is not None:
        rnd = repaired_stats(predict_fn([neutralize_region(pil, rand.bbox)])[0])
    else:
        rnd = {"alias_correct": False, "target_prob_improved": False, "distractor_decreased": False}

    # Coverage ceiling: does ANY candidate overlap a text box at the IoU threshold?
    cov01 = any(_max_iou(s.bbox, text_boxes) >= 0.1 for s in non_random) if text_boxes else False
    cov03 = any(_max_iou(s.bbox, text_boxes) >= 0.3 for s in non_random) if text_boxes else False

    # Median rank diagnostic: rank (by CIC score) of best text-overlapping candidate.
    best_overlap_rank = float("nan")
    if text_boxes:
        for rank, s in enumerate(non_random, start=1):
            if _max_iou(s.bbox, text_boxes) >= 0.1:
                best_overlap_rank = float(rank)
                break

    # Repair-localization conflict: repair succeeds but the selected region does
    # not overlap a human text box. This is a localization MISMATCH, not a bug:
    # CIC optimizes causal effect on prediction stability, which need not coincide
    # with the human-annotated text region (see docs/auto_proposal_pilot.md).
    conflict = bool(cic["alias_correct"] and (top1 is not None) and (not sel_text_overlap) and bool(text_boxes))

    return {
        "n_candidates": len(non_random),
        "cic_repair_alias_correct": bool(cic["alias_correct"]),
        "random_repair_alias_correct": bool(rnd["alias_correct"]),
        "cic_target_prob_improved": bool(cic["target_prob_improved"]),
        "random_target_prob_improved": bool(rnd.get("target_prob_improved", False)),
        "cic_distractor_decreased": bool(cic["distractor_decreased"]),
        "random_distractor_decreased": bool(rnd.get("distractor_decreased", False)),
        "selected_text_overlap": bool(sel_text_overlap),
        "selected_object_overlap": bool(sel_obj_overlap),
        "selected_area_fraction": sel_area,
        "coverage_iou01": bool(cov01),
        "coverage_iou03": bool(cov03),
        "best_text_overlap_rank": best_overlap_rank,
        "repair_localization_conflict": conflict,
        "proposal_available": bool(getattr(proposal_set, "available", True)),
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


def _aggregate_family(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cic = _rate([r["cic_repair_alias_correct"] for r in rows])
    rnd = _rate([r["random_repair_alias_correct"] for r in rows])
    return {
        "n": len(rows),
        "cic_repair_alias_accuracy": cic,
        "matched_random_repair_alias_accuracy": rnd,
        "cic_random_gap": (cic - rnd) if (np.isfinite(cic) and np.isfinite(rnd)) else float("nan"),
        "cic_target_prob_improvement_rate": _rate([r["cic_target_prob_improved"] for r in rows]),
        "random_target_prob_improvement_rate": _rate([r["random_target_prob_improved"] for r in rows]),
        "cic_text_distractor_decrease_rate": _rate([r["cic_distractor_decreased"] for r in rows]),
        "selected_text_overlap_rate": _rate([r["selected_text_overlap"] for r in rows]),
        "selected_object_overlap_rate": _rate([r["selected_object_overlap"] for r in rows]),
        "coverage_ceiling_iou01": _rate([r["coverage_iou01"] for r in rows]),
        "coverage_ceiling_iou03": _rate([r["coverage_iou03"] for r in rows]),
        "median_rank_best_text_overlap": _median([r["best_text_overlap_rank"] for r in rows]),
        "repair_localization_conflict_rate": _rate([r["repair_localization_conflict"] for r in rows]),
        "mean_selected_area_fraction": _mean([r["selected_area_fraction"] for r in rows]),
        "mean_candidates": _mean([r["n_candidates"] for r in rows]),
    }


def _go_no_go_for_subset(families: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = families.get(BASELINE_FAMILY)
    verdicts: dict[str, Any] = {}
    promising = False
    if base is None:
        return {"promising": False, "per_family": verdicts, "note": "no baseline family available"}
    for fam, m in families.items():
        if fam == BASELINE_FAMILY:
            continue
        repair_gain = m["cic_repair_alias_accuracy"] - base["cic_repair_alias_accuracy"]
        overlap_gain = m["selected_text_overlap_rate"] - base["selected_text_overlap_rate"]
        repair_drop = base["cic_repair_alias_accuracy"] - m["cic_repair_alias_accuracy"]
        coverage_gain = m["coverage_ceiling_iou01"] - base["coverage_ceiling_iou01"]
        c_repair = bool(np.isfinite(repair_gain) and repair_gain >= PROMISE_REPAIR_GAIN)
        c_overlap = bool(
            np.isfinite(overlap_gain)
            and overlap_gain >= PROMISE_TEXT_OVERLAP_GAIN
            and (not np.isfinite(repair_drop) or repair_drop <= PROMISE_REPAIR_MAX_DROP)
        )
        c_coverage = bool(np.isfinite(coverage_gain) and coverage_gain >= PROMISE_COVERAGE_GAIN)
        fam_promising = c_repair or c_overlap or c_coverage
        promising = promising or fam_promising
        verdicts[fam] = {
            "repair_gain_vs_baseline": _finite(repair_gain),
            "selected_text_overlap_gain_vs_baseline": _finite(overlap_gain),
            "coverage_ceiling_iou01_gain_vs_baseline": _finite(coverage_gain),
            "criterion_repair_+0.10": c_repair,
            "criterion_text_overlap_+0.15_no_big_repair_drop": c_overlap,
            "criterion_coverage_+0.20": c_coverage,
            "family_promising": fam_promising,
        }
    return {"promising": promising, "per_family": verdicts}


def _finite(v) -> float | None:
    return float(v) if (v is not None and np.isfinite(v)) else None


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #
def dry_run(args) -> dict[str, Any]:
    avail = generator_availability(allow_download=bool(args.allow_download))
    families = _resolve_families(args.families)
    info = {
        "pilot": "coco_text_auto_proposal",
        "status": "dry_run",
        "max_examples": int(args.max_examples),
        "max_candidates_per_family": int(args.max_candidates),
        "families_requested": families,
        "generators_available": {k: v.available for k, v in avail.items()},
        "generators_skipped": {k: v.skip_reason for k, v in avail.items() if not v.available},
        "results_dir": str(Path(args.results_dir) / RESULTS_SUBDIR),
        "non_claims": NON_CLAIMS,
        "note": "dry-run validates wiring and generator availability; no CLIP scoring, no canonical artifacts written.",
    }
    print(json.dumps(info, indent=2))
    return info


def _resolve_families(spec: str | None) -> list[str]:
    if not spec or spec == "classical":
        return list(CLASSICAL_FAMILIES)
    if spec == "all":
        return list(ALL_FAMILIES)
    return [f.strip() for f in spec.split(",") if f.strip()]


# --------------------------------------------------------------------------- #
# Skip writer
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


def _write_outputs(out_dir: Path, metrics: dict[str, Any], table_rows: list[dict[str, Any]]) -> dict[str, str]:
    ensure_dir(out_dir)
    metrics_path = out_dir / "coco_text_auto_proposal_metrics.json"
    table_path = out_dir / "coco_text_auto_proposal_table.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    pd.DataFrame(table_rows).to_csv(table_path, index=False)
    return {"metrics": str(metrics_path), "table": str(table_path)}


def _skip(out_dir: Path, reason: str, extra: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "pilot": "coco_text_auto_proposal",
        "status": "skipped",
        "skip_reason": reason,
        "pilot_promising": False,
        "non_claims": NON_CLAIMS,
        **extra,
    }
    _write_outputs(out_dir, metrics, [{"subset": "", "family": "", "skip_reason": reason}])
    write_combined_summary(out_dir.parent if out_dir.name == RESULTS_SUBDIR else out_dir)
    print(json.dumps({"status": "skipped", "reason": reason}, indent=2))
    return metrics


# --------------------------------------------------------------------------- #
# Main run
# --------------------------------------------------------------------------- #
def run(args) -> dict[str, Any]:
    from PIL import Image

    out_dir = ensure_dir(Path(args.results_dir) / RESULTS_SUBDIR)
    seed = int(args.seed)
    np.random.seed(seed)
    families = _resolve_families(args.families)
    avail = generator_availability(allow_download=bool(args.allow_download))
    gens_available = {k: v.available for k, v in avail.items()}
    gens_skipped = {k: v.skip_reason for k, v in avail.items() if not v.available}
    eval_families = [BASELINE_FAMILY] + [f for f in families if gens_available.get(f, False)]

    base_extra = {
        "generators_available": gens_available,
        "generators_skipped": gens_skipped,
        "families_evaluated": eval_families,
        "max_examples": int(args.max_examples),
        "max_candidates_per_family": int(args.max_candidates),
    }

    # Data.
    root = Path(args.data_root)
    metadata_csv = Path(args.metadata_csv)
    if not metadata_csv.exists():
        return _skip(out_dir, f"COCO-Text metadata not found at {metadata_csv}", base_extra)
    bundle = load_local_folder_dataset(root, metadata_csv, image_size=int(args.image_size))
    if not bundle.examples:
        return _skip(out_dir, "COCO-Text metadata loaded but no examples available", base_extra)
    by_id = {int(ex["example_id"]): ex for ex in bundle.examples}

    subsets_spec = load_subsets(Path(args.triage_dir))
    chosen_subsets = [s.strip() for s in args.subset.split(",")] if args.subset != "auto" else ["strict", "directional"]
    available_subsets = {k: v for k, v in subsets_spec.items() if v["ids"]}
    if not available_subsets:
        return _skip(out_dir, f"no triage subsets found under {args.triage_dir}", base_extra)

    # Model.
    if args.backend == "fake":
        return _skip(out_dir, "fake backend cannot support a pilot result", base_extra)
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
    per_example_diag: list[dict[str, Any]] = []

    for subset_name in chosen_subsets:
        spec = subsets_spec.get(subset_name)
        if not spec or not spec["ids"]:
            continue
        ids = [i for i in spec["ids"] if i in by_id][: int(args.max_examples)]
        distractors = spec["distractors"]
        fam_rows: dict[str, list[dict[str, Any]]] = {f: [] for f in eval_families}
        orig_correct: list[bool] = []
        used = 0
        for eid in ids:
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
                predict_fn = _build_predict_fn(status, allowed, args.device)
                predict_cache[key] = predict_fn

            pil = Image.fromarray((np.asarray(ex["image"]).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")
            text_boxes = [tuple(int(v) for v in b) for b in ex.get("text_boxes", [])]
            object_boxes = [tuple(int(v) for v in b) for b in ex.get("object_boxes", [])]

            original_probs = predict_fn([pil])[0]
            non_target = [i for i in range(len(allowed)) if i not in set(target_idxs)]
            d_label = distractors.get(eid)
            if d_label and d_label in allowed and not is_target_label(d_label, target, aliases):
                distractor_idx = allowed.index(d_label)
            elif non_target:
                distractor_idx = int(max(non_target, key=lambda i: original_probs[i]))
            else:
                distractor_idx = int(target_idxs[0])
            orig = _orig_stats(original_probs, allowed, target, aliases, target_idxs, distractor_idx)
            orig_correct.append(orig["alias_correct"])
            used += 1

            for fam in eval_families:
                if fam == BASELINE_FAMILY:
                    rps = _baseline_region_proposals(pil, text_boxes, seed + eid, int(args.max_candidates))
                    proposal_set = None
                else:
                    rps, proposal_set = _auto_region_proposals(pil, fam, seed + eid, int(args.max_candidates), bool(args.allow_download))
                scores, _ = score_region_candidates(pil, rps, predict_fn)
                row = _evaluate_family_example(
                    pil, scores, proposal_set, predict_fn,
                    allowed=allowed, target=target, aliases=aliases,
                    target_idxs=target_idxs, distractor_idx=distractor_idx, orig=orig,
                    text_boxes=text_boxes, object_boxes=object_boxes, prob_eps=DEFAULT_PROB_EPS,
                )
                fam_rows[fam].append(row)
                per_example_diag.append({"subset": subset_name, "example_id": eid, "family": fam, **row})

        if used == 0:
            continue
        fam_metrics = {fam: _aggregate_family(rows) for fam, rows in fam_rows.items() if rows}
        go = _go_no_go_for_subset(fam_metrics)
        subset_results[subset_name] = {
            "n": used,
            "original_alias_accuracy": _rate(orig_correct),
            "families": fam_metrics,
            "go_no_go": go,
        }
        for fam, m in fam_metrics.items():
            table_rows.append({"subset": subset_name, "family": fam, "n": used,
                               "original_alias_accuracy": _rate(orig_correct), **m})

    pilot_promising = any(s["go_no_go"]["promising"] for s in subset_results.values())
    metrics = {
        "pilot": "coco_text_auto_proposal",
        "status": "ok",
        "backend": status.backend,
        "model_name": status.model_name,
        "pretrained_loaded": bool(status.pretrained),
        **base_extra,
        "subsets": subset_results,
        "pilot_promising": bool(pilot_promising),
        "non_claims": NON_CLAIMS,
    }
    paths = _write_outputs(out_dir, metrics, table_rows or [{"subset": "", "family": "", "note": "no examples evaluated"}])
    pd.DataFrame(per_example_diag).to_csv(out_dir / "coco_text_auto_proposal_per_example.csv", index=False)
    write_combined_summary(out_dir)
    print(json.dumps({"status": "ok", "pilot_promising": bool(pilot_promising), **paths}, indent=2))
    return metrics


# --------------------------------------------------------------------------- #
# Combined summary (COCO-Text + Waterbirds), regenerated from whatever exists
# --------------------------------------------------------------------------- #
def write_combined_summary(out_dir: Path) -> str:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    coco_path = out_dir / "coco_text_auto_proposal_metrics.json"
    water_path = out_dir / "waterbirds_auto_proposal_metrics.json"
    coco = json.loads(coco_path.read_text()) if coco_path.exists() else None
    water = json.loads(water_path.read_text()) if water_path.exists() else None

    lines: list[str] = [
        "# Automated Finite-Candidate Proposal CIC — Pilot Summary",
        "",
        "This pilot tests whether **automatic** proposal generation can reduce CIC's ",
        "proposal bottleneck on natural images. It is **automated finite-candidate ",
        "proposal generation**: CIC still scores a finite candidate set, but the set is ",
        "generated automatically from pixels instead of being manually designed around the ",
        "text-shortcut family.",
        "",
        "## Non-claims (explicit)",
        "",
        "- This is **not** open-world shortcut discovery.",
        "- This is **not** universal repair or general robustness.",
        "- This is **not** deployment validation or clinical validation.",
        "- This is **not** a replacement for the finalized STS report.",
        "",
    ]

    # Generator availability (prefer COCO record, fall back to Waterbirds).
    gens = (coco or {}).get("generators_available") or (water or {}).get("generators_available") or {}
    skipped = (coco or {}).get("generators_skipped") or (water or {}).get("generators_skipped") or {}
    lines += ["## Proposal generators", ""]
    if gens:
        for name, ok in gens.items():
            note = "" if ok else f" — skipped: {skipped.get(name, 'unavailable')}"
            lines.append(f"- `{name}`: {'available' if ok else 'skipped'}{note}")
    else:
        lines.append("- (no generator availability recorded yet)")
    lines.append("")

    # COCO-Text section.
    lines += ["## COCO-Text automated proposal pilot", ""]
    if coco is None:
        lines += ["- Not run yet.", ""]
    elif coco.get("status") != "ok":
        lines += [f"- Status: **{coco.get('status')}** — {coco.get('skip_reason', '')}", ""]
    else:
        lines.append(f"- Backend: `{coco.get('backend')}` ({coco.get('model_name')}), capped sample per subset: {coco.get('max_examples')}")
        lines.append(f"- Families evaluated: {', '.join(coco.get('families_evaluated', []))}")
        lines.append("")
        for sname, s in coco.get("subsets", {}).items():
            lines.append(f"### Subset `{sname}` (n={s.get('n')}, original alias accuracy={_fmt(s.get('original_alias_accuracy'))})")
            lines.append("")
            base = s.get("families", {}).get(BASELINE_FAMILY, {})
            lines.append("| family | CIC repair | random repair | gap | sel text-overlap | cov@.1 | cov@.3 | med rank | conflict | sel area |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for fam, m in s.get("families", {}).items():
                lines.append(
                    f"| {fam} | {_fmt(m.get('cic_repair_alias_accuracy'))} | {_fmt(m.get('matched_random_repair_alias_accuracy'))} | "
                    f"{_fmt(m.get('cic_random_gap'))} | {_fmt(m.get('selected_text_overlap_rate'))} | {_fmt(m.get('coverage_ceiling_iou01'))} | "
                    f"{_fmt(m.get('coverage_ceiling_iou03'))} | {_fmt(m.get('median_rank_best_text_overlap'))} | "
                    f"{_fmt(m.get('repair_localization_conflict_rate'))} | {_fmt(m.get('mean_selected_area_fraction'))} |"
                )
            lines.append("")
            go = s.get("go_no_go", {})
            lines.append(f"- Improves over existing proposal baseline (any auto family meets a go-criterion): **{go.get('promising')}**")
            for fam, v in go.get("per_family", {}).items():
                lines.append(
                    f"  - `{fam}`: repair gain {_fmt(v.get('repair_gain_vs_baseline'))}, "
                    f"text-overlap gain {_fmt(v.get('selected_text_overlap_gain_vs_baseline'))}, "
                    f"coverage gain {_fmt(v.get('coverage_ceiling_iou01_gain_vs_baseline'))} → promising={v.get('family_promising')}"
                )
            lines.append("")
        lines.append(f"- **COCO-Text pilot_promising: {coco.get('pilot_promising')}**")
        lines.append("")

    # Waterbirds section.
    lines += ["## Waterbirds automated proposal diagnostic", ""]
    if water is None:
        lines += ["- Not run yet.", ""]
    elif water.get("status") != "ok":
        lines += [f"- Status: **{water.get('status')}** — {water.get('skip_reason', '')}", ""]
    else:
        lines.append(f"- Backend: `{water.get('backend')}`, capped sample: {water.get('n_examples')}")
        lines.append(f"- Overall accuracy before/after repair: {_fmt(water.get('overall_accuracy_before'))} / {_fmt(water.get('overall_accuracy_after'))}")
        lines.append(f"- Worst-group accuracy before/after: {_fmt(water.get('worst_group_accuracy_before'))} / {_fmt(water.get('worst_group_accuracy_after'))}")
        lines.append(f"- Background-sensitivity proxy before/after: {_fmt(water.get('background_sensitivity_before'))} / {_fmt(water.get('background_sensitivity_after'))}")
        lines.append(f"- Mean confidence change: {_fmt(water.get('mean_confidence_change'))}, mean selected area: {_fmt(water.get('mean_selected_area_fraction'))}")
        lines.append("")
        lines.append("| group | n | acc before | acc after |")
        lines.append("|---|---|---|---|")
        for g, gm in water.get("groups", {}).items():
            lines.append(f"| {g} | {gm.get('n')} | {_fmt(gm.get('accuracy_before'))} | {_fmt(gm.get('accuracy_after'))} |")
        lines.append("")
        lines.append("- **Caveat:** no oracle bird/background masks are available, so this is a ")
        lines.append("  **diagnostic**, not full validation.")
        lines.append(f"- **Waterbirds pilot_promising: {water.get('pilot_promising')}**")
        lines.append("")

    # Overall verdict.
    coco_ok = bool((coco or {}).get("pilot_promising")) if coco and coco.get("status") == "ok" else False
    water_ok = bool((water or {}).get("pilot_promising")) if water and water.get("status") == "ok" else False
    lines += [
        "## Is the pilot promising enough for a full run?",
        "",
        f"- COCO-Text promising: **{coco_ok}**",
        f"- Waterbirds promising: **{water_ok}**",
        f"- Overall: **{coco_ok or water_ok}** "
        + ("(at least one setting cleared a pre-registered go threshold)" if (coco_ok or water_ok)
           else "(no setting cleared a pre-registered go threshold — preserved as a negative/diagnostic result)"),
        "",
        "See `docs/auto_proposal_pilot.md` for the repair-vs-localization discussion, the ",
        "human-validation note, and the global-additivity motivation.",
    ]
    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(summary_path)


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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-examples", type=int, default=50)
    p.add_argument("--max-candidates", type=int, default=14, help="max auto candidates per family (excl. random controls)")
    p.add_argument("--subset", default="auto", help="comma list of {strict,directional,all_500} or 'auto'")
    p.add_argument("--families", default="classical", help="'classical', 'all', or comma list of generator families")
    p.add_argument("--allow-download", action="store_true", help="permit model downloads (off by default)")
    p.add_argument("--dry-run", action="store_true", help="validate wiring + generator availability only")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--data-root", default="data/coco_text_cic")
    p.add_argument("--metadata-csv", default="data/coco_text_cic/metadata.csv")
    p.add_argument("--triage-dir", default="results/coco_text_cic_triage")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--backend", default="open_clip")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        dry_run(args)
        return
    run(args)


if __name__ == "__main__":
    main()
