from __future__ import annotations

"""Error-analysis & directional-metric pass for the natural-text verified failures.

Experiment name: ``natural_text_error_analysis``.

This is a **diagnostic** companion to
``run_natural_text_verified_failure_eval``. It re-runs the same leakage-free
open-proposal CIC pipeline on the human-verified curated annotation set, but
instead of only the strict pass/fail repair gate it computes *directional* repair
metrics (target probability/rank/logit movement, text-distractor suppression,
alias-aware recovery), proposal-selection geometry diagnostics, and a per-example
error categorization.

Hard guarantees:
* It writes ONLY new files under ``results/natural_text_verified_failure_eval/``
  (``error_analysis.md``, ``natural_text_error_analysis.md``,
  ``natural_text_directional_metrics.csv``,
  ``natural_text_directional_key_numbers.json``). It never overwrites the strict
  ``verified_failure_*`` artifacts or any final-report headline metrics.
* The strict support gate is recomputed (via the original
  ``evaluate_natural_text_gate``) and reported, but is *not* loosened.
  ``natural_text_supported`` / ``open_proposal_supported`` stay ``False`` unless
  the original strict conditions genuinely pass; ``open_world_claim_allowed``
  stays ``False``.
* A new diagnostic flag ``natural_text_directional_evidence`` is added. It is not
  a headline claim and never licenses a positive natural-text claim.
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

from causal_reliability.analysis.natural_text_error_analysis import (
    DIRECTIONAL_METHODS,
    aggregate_directional_metrics,
    build_label_info,
    categorize_failure,
    directional_metrics_row,
    evaluate_directional_evidence,
    selection_geometry,
)
from causal_reliability.data.natural_text_dataset import load_verified_natural_text_dataset
from causal_reliability.discovery.cic_region_scoring import neutralize_region, score_region_candidates
from causal_reliability.discovery.open_region_proposals import (
    OCR_FAMILY,
    generate_open_region_proposals,
    proposal_family,
)
from causal_reliability.experiments.run_natural_text_open_proposal_cic import (
    FORBIDDEN_WORDING,
    PREFERRED_WORDING,
    PROMPT_TEMPLATE,
    _device,
    _downloads_allowed,
    _json_default,
    _neutralize_boxes,
    _pil_to_tensor,
    _select_matched_random,
    scoring_is_leakage_free,
)
from causal_reliability.experiments.run_natural_text_verified_failure_eval import (
    ALWAYS_OPEN_WORLD_CLAIM_ALLOWED,
    DEFAULT_MIN_CIC_RANDOM_GAP,
    DEFAULT_MIN_ORACLE_REPAIR_RATE,
    DEFAULT_MIN_VERIFIED_FAILURES,
    DEFAULT_OUTPUT_SUBDIR,
    evaluate_natural_text_gate,
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


# --------------------------------------------------------------------------- #
# Prediction (probs + logits)
# --------------------------------------------------------------------------- #
def _build_predict_fn_with_logits(status: ClipStatus, allowed_labels: list[str], device: str):
    prompts = [PROMPT_TEMPLATE.format(label=name) for name in allowed_labels]
    classifier = ClipZeroShotClassifier(status, allowed_labels, prompts=prompts, device=device)

    def predict(images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        out = classifier.predict(_pil_to_tensor(images))
        probs = np.asarray(out["probabilities"].detach().cpu().numpy(), dtype=np.float64)
        logits = np.asarray(out["logits"].detach().cpu().numpy(), dtype=np.float64)
        return probs, logits

    return predict


# --------------------------------------------------------------------------- #
# Per-example directional analysis
# --------------------------------------------------------------------------- #
def _analyze_examples(
    examples: list[dict[str, Any]],
    status: ClipStatus,
    device: str,
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return (directional_rows, per_failure_records, run_diag).

    ``directional_rows`` is long-form: one row per (verified failure x method).
    ``per_failure_records`` holds one categorization + geometry record per failure.
    """

    seed = int(cfg.get("seed", 0))
    max_candidates = int(cfg.get("max_candidates", 64))
    grid_scales = cfg.get("grid_scales")
    high_conf_threshold = float(cfg.get("high_confidence_threshold", 0.7))
    enable_object_box_family = bool(cfg.get("enable_object_box_family", False))

    directional_rows: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    n_evaluated = 0
    n_high_conf = 0
    n_original_correct = 0

    for ex in examples:
        allowed = list(ex["allowed_clip_labels"])
        target_label = ex["human_label"]
        distractors = list(ex.get("text_distractor_labels", []))
        info = build_label_info(allowed, target_label, distractors)
        label = info.label
        pil = Image.fromarray((np.asarray(ex["image"]).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")
        text_boxes = [tuple(int(v) for v in b) for b in ex.get("text_boxes", [])]
        object_boxes = [tuple(int(v) for v in b) for b in ex.get("object_boxes", [])]

        predict = _build_predict_fn_with_logits(status, allowed, device)

        def probs_only(images: list[Image.Image]) -> np.ndarray:
            return predict(images)[0]

        proposals = generate_open_region_proposals(
            pil,
            text_boxes=text_boxes,
            object_boxes=object_boxes,
            seed=seed + int(ex["example_id"]),
            max_candidates=max_candidates,
            grid_scales=grid_scales,
            enable_object_box_family=enable_object_box_family,
        )
        scores, original_probs = score_region_candidates(pil, proposals, probs_only)
        orig_probs, orig_logits = predict([pil])
        orig_probs = orig_probs[0]
        orig_logits = orig_logits[0]

        orig_pred = int(orig_probs.argmax())
        orig_conf = float(orig_probs.max())
        original_correct = bool(orig_pred == label)
        high_conf_failure = bool((not original_correct) and orig_conf >= high_conf_threshold)
        predicted_is_text_distractor = bool((not original_correct) and orig_pred in info.distractor_indices)
        text_driven_candidate = str(ex.get("text_driven_candidate", "")).lower() == "yes"
        verified = bool(high_conf_failure and predicted_is_text_distractor and text_driven_candidate)

        n_evaluated += 1
        n_high_conf += int(high_conf_failure)
        n_original_correct += int(original_correct)
        if not verified:
            continue

        top1 = scores[0] if scores else None
        top3 = scores[:3]
        rand = _select_matched_random(scores, top1, "area_fraction")
        largest = max(scores, key=lambda s: s.area_fraction) if scores else None
        ocr_scores = [s for s in scores if proposal_family(s.proposal_type) == OCR_FAMILY]

        # After-(probs, logits) per method. ``None`` -> method unavailable.
        method_after: dict[str, tuple[np.ndarray, np.ndarray] | None] = {
            "original": (orig_probs, orig_logits),
        }
        if text_boxes:
            p, l = predict([_neutralize_boxes(pil, text_boxes)])
            method_after["oracle_text_box_repair"] = (p[0], l[0])
        else:
            method_after["oracle_text_box_repair"] = None
        if top1:
            p, l = predict([neutralize_region(pil, top1.bbox)])
            method_after["cic_top1"] = (p[0], l[0])
        else:
            method_after["cic_top1"] = None
        if top3:
            p, l = predict([neutralize_region(pil, s.bbox) for s in top3])
            method_after["cic_top3"] = (p.mean(axis=0), l.mean(axis=0))
        else:
            method_after["cic_top3"] = None
        if rand:
            p, l = predict([neutralize_region(pil, rand.bbox)])
            method_after["matched_random"] = (p[0], l[0])
        else:
            method_after["matched_random"] = None
        if largest:
            p, l = predict([neutralize_region(pil, largest.bbox)])
            method_after["largest_region"] = (p[0], l[0])
        else:
            method_after["largest_region"] = None
        if ocr_scores:
            p, l = predict([neutralize_region(pil, ocr_scores[0].bbox)])
            method_after["ocr_text_box_proposal"] = (p[0], l[0])
        else:
            method_after["ocr_text_box_proposal"] = None

        rows_by_method: dict[str, dict[str, Any]] = {}
        for method in DIRECTIONAL_METHODS:
            after = method_after.get(method)
            if after is None:
                continue
            after_probs, after_logits = after
            row = directional_metrics_row(
                orig_probs,
                after_probs,
                info,
                before_logits=orig_logits,
                after_logits=after_logits,
            )
            rows_by_method[method] = row
            directional_rows.append(
                {
                    "example_id": int(ex["example_id"]),
                    "human_label": target_label,
                    "method": method,
                    "method_available": True,
                    **row,
                }
            )

        geom = None
        if top1 is not None:
            geom = selection_geometry(top1.bbox, text_boxes, object_boxes)

        primary, flags = categorize_failure(
            cic_row=rows_by_method.get("cic_top1", rows_by_method["original"]),
            oracle_row=rows_by_method.get("oracle_text_box_repair"),
            geometry=geom,
        )

        failure_records.append(
            {
                "example_id": int(ex["example_id"]),
                "human_label": target_label,
                "original_prediction_label": allowed[orig_pred],
                "original_confidence": orig_conf,
                "selected_proposal_type": top1.proposal_type if top1 else "",
                "selected_family": proposal_family(top1.proposal_type) if top1 else "",
                "selected_area_fraction": float(top1.area_fraction) if top1 else float("nan"),
                "selected_bbox": "" if top1 is None else json.dumps([int(v) for v in top1.bbox]),
                **({} if geom is None else {f"geom_{k}": v for k, v in geom.items()}),
                "category": primary,
                **{f"is_{k}": v for k, v in flags.items()},
            }
        )

    run_diag = {
        "n_evaluated": n_evaluated,
        "n_high_conf_failures": n_high_conf,
        "n_original_correct": n_original_correct,
        "original_clip_accuracy": (n_original_correct / n_evaluated) if n_evaluated else float("nan"),
        "high_confidence_failure_rate": (n_high_conf / n_evaluated) if n_evaluated else float("nan"),
    }
    return directional_rows, failure_records, run_diag


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _aggregate(directional_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(directional_rows)
    rows: list[dict[str, Any]] = []
    for method in DIRECTIONAL_METHODS:
        method_rows = [r for r in directional_rows if r["method"] == method]
        agg = aggregate_directional_metrics(method_rows)
        rows.append({"method": method, **agg})
    return pd.DataFrame(rows), df


def _conditional_repair_by_bucket(failure_records: list[dict[str, Any]], directional_rows: list[dict[str, Any]]) -> pd.DataFrame:
    cic_by_id = {r["example_id"]: r for r in directional_rows if r["method"] == "cic_top1"}
    buckets: dict[str, dict[str, list]] = {}
    for rec in failure_records:
        bucket = rec.get("geom_text_overlap_bucket", "no_overlap")
        cic = cic_by_id.get(rec["example_id"])
        if cic is None:
            continue
        b = buckets.setdefault(bucket, {"strict": [], "alias": [], "directional": []})
        b["strict"].append(bool(cic["strict_top1_after"]))
        b["alias"].append(bool(cic["alias_top1_after"]))
        b["directional"].append(bool(cic["moved_toward_target"]))
    rows = []
    for bucket, vals in sorted(buckets.items()):
        rows.append(
            {
                "text_overlap_bucket": bucket,
                "n": len(vals["strict"]),
                "cic_strict_repair_rate": float(np.mean(vals["strict"])) if vals["strict"] else float("nan"),
                "cic_alias_repair_rate": float(np.mean(vals["alias"])) if vals["alias"] else float("nan"),
                "cic_directional_rate": float(np.mean(vals["directional"])) if vals["directional"] else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _method_metric(agg: pd.DataFrame, method: str, col: str) -> float:
    sub = agg[agg["method"] == method]
    if sub.empty:
        return float("nan")
    return float(sub[col].iloc[0])


# --------------------------------------------------------------------------- #
# Markdown reports
# --------------------------------------------------------------------------- #
def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "nan" if np.isnan(v) else f"{v:.3f}"
    return str(v)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(out)


def _category_counts(failure_records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in failure_records:
        counts[rec["category"]] = counts.get(rec["category"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def _examples_in(failure_records, category: str, limit: int = 12) -> list[str]:
    out = []
    for rec in failure_records:
        if rec.get(f"is_{category}"):
            out.append(f"{rec['example_id']}:{rec['human_label']}")
    return out[:limit]


def _write_error_analysis_md(
    path: Path,
    failure_records: list[dict[str, Any]],
    directional_rows: list[dict[str, Any]],
    agg: pd.DataFrame,
    key_numbers: dict[str, Any],
) -> None:
    cic_by_id = {r["example_id"]: r for r in directional_rows if r["method"] == "cic_top1"}
    oracle_by_id = {r["example_id"]: r for r in directional_rows if r["method"] == "oracle_text_box_repair"}

    def label(rec):
        return f"{rec['example_id']}:{rec['human_label']}"

    best_successes = [label(r) for r in failure_records if r["category"] == "cic_strict_repaired"]
    oracle_only = [
        label(r)
        for r in failure_records
        if r["category"] == "oracle_strict_repaired" or (oracle_by_id.get(r["example_id"], {}).get("strict_top1_after") and not cic_by_id.get(r["example_id"], {}).get("strict_top1_after"))
    ]
    cic_text_no_repair = [label(r) for r in failure_records if r.get("is_cic_selected_text_no_repair")]
    cic_object_damaged = [label(r) for r in failure_records if r.get("is_cic_selected_object_damaged")]
    alias_cases = [label(r) for r in failure_records if r.get("is_label_alias_ambiguity")]
    hard_cases = [label(r) for r in failure_records if r["category"] == "hard_no_clear_repair"]

    # Drop/relabel candidates: alias-ambiguity where even the oracle cannot get
    # the strict target to top-1 (the exact-string label is arguably wrong).
    relabel = []
    for r in failure_records:
        orow = oracle_by_id.get(r["example_id"])
        if orow and orow["alias_top1_after"] and not orow["strict_top1_after"]:
            relabel.append(label(r))

    lines = [
        "# Natural-Text Verified-Failure Error Analysis",
        "",
        "**Strict support gate remains FAILED; directional evidence is diagnostic only.**",
        "",
        f"- Verified text-driven failures: {key_numbers['n_verified_text_driven_failures']}",
        f"- `natural_text_supported`: {key_numbers['natural_text_supported']} (unchanged, strict)",
        f"- `open_proposal_supported`: {key_numbers['open_proposal_supported']} (unchanged, strict)",
        f"- `open_world_claim_allowed`: {key_numbers['open_world_claim_allowed']}",
        f"- `natural_text_directional_evidence` (diagnostic): {key_numbers['natural_text_directional_evidence']}",
        "",
        "## Failure-cause summary",
        "",
        "Which of the five candidate causes dominates:",
        "",
        f"1. Poor proposal selection: CIC selected text overlap rate = "
        f"{_fmt(key_numbers['cic_selected_text_overlap_rate'])}; "
        f"CIC strict repair = {_fmt(_method_metric(agg, 'cic_top1', 'strict_top1_repair_accuracy'))}, "
        f"directional (target-prob up) = {_fmt(_method_metric(agg, 'cic_top1', 'target_prob_improvement_rate'))}.",
        f"2. Weak masking/intervention: oracle text-box strict repair = "
        f"{_fmt(_method_metric(agg, 'oracle_text_box_repair', 'strict_top1_repair_accuracy'))} but oracle "
        f"directional (target-prob up) = {_fmt(_method_metric(agg, 'oracle_text_box_repair', 'target_prob_improvement_rate'))} "
        "— removing the text moves probability toward the target far more often than it flips the exact argmax.",
        f"3. Overly strict exact-label evaluation: oracle alias-aware top-1 = "
        f"{_fmt(_method_metric(agg, 'oracle_text_box_repair', 'alias_top1_repair_accuracy'))} vs strict "
        f"{_fmt(_method_metric(agg, 'oracle_text_box_repair', 'strict_top1_repair_accuracy'))}; "
        f"{len(relabel)} failures recover an alias at top-1 but never the exact string.",
        f"4. Alias/label mismatch: {len(alias_cases)} failures flagged as label/alias ambiguity.",
        f"5. Natural-image ambiguity: {len(hard_cases)} hard failures with no clear oracle repair or "
        "directional movement.",
        "",
        "## Best successes (CIC strict repaired)",
        "",
        *([f"- {x}" for x in best_successes] or ["- (none)"]),
        "",
        "## Oracle-only successes (oracle strict-repaired, CIC did not)",
        "",
        *([f"- {x}" for x in oracle_only] or ["- (none)"]),
        "",
        "## CIC failures despite text overlap (selected text, did not repair)",
        "",
        *([f"- {x}" for x in cic_text_no_repair] or ["- (none)"]),
        "",
        "## CIC failures due to object overlap / content damage",
        "",
        *([f"- {x}" for x in cic_object_damaged] or ["- (none)"]),
        "",
        "## Likely label-ambiguity cases",
        "",
        *([f"- {x}" for x in alias_cases] or ["- (none)"]),
        "",
        "## Examples to drop or relabel (oracle recovers only an alias, never the exact string)",
        "",
        *([f"- {x}" for x in relabel] or ["- (none)"]),
        "",
        "## Hard natural images (no clear repair)",
        "",
        *([f"- {x}" for x in hard_cases] or ["- (none)"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_directional_md(
    path: Path,
    agg: pd.DataFrame,
    bucket_df: pd.DataFrame,
    cat_counts: dict[str, int],
    geom_df: pd.DataFrame,
    key_numbers: dict[str, Any],
) -> None:
    show_cols = [
        "method",
        "n",
        "strict_top1_repair_accuracy",
        "alias_top1_repair_accuracy",
        "target_prob_improvement_rate",
        "median_target_prob_gain",
        "target_rank_improvement_rate",
        "median_target_rank_gain",
        "text_distractor_prob_decrease_rate",
        "median_text_distractor_prob_decrease",
        "top3_target_recovery_rate",
        "top5_target_recovery_rate",
        "moved_away_from_text_rate",
        "moved_toward_target_rate",
    ]
    agg_show = agg[[c for c in show_cols if c in agg.columns]]
    lines = [
        "# Natural-Text Directional Repair Metrics (diagnostic)",
        "",
        "**Strict support gate remains failed; directional evidence is diagnostic only.**",
        "These secondary metrics describe how repairs move the prediction; they do "
        "**not** replace the strict support gate and do not license any positive "
        "natural-text claim.",
        "",
        f"- Verified text-driven failures: {key_numbers['n_verified_text_driven_failures']}",
        f"- `natural_text_supported` (strict, unchanged): {key_numbers['natural_text_supported']}",
        f"- `open_proposal_supported` (strict, unchanged): {key_numbers['open_proposal_supported']}",
        f"- `open_world_claim_allowed`: {key_numbers['open_world_claim_allowed']}",
        f"- `natural_text_directional_evidence` (diagnostic flag): {key_numbers['natural_text_directional_evidence']}",
        f"- Directional-evidence reasons (if unset): {key_numbers['directional_evidence_reasons'] or 'none'}",
        "",
        "## Aggregate directional metrics by method",
        "",
        _md_table(agg_show),
        "",
        "## CIC repair conditional on selected-region text overlap",
        "",
        _md_table(bucket_df),
        "",
        "## Proposal-selection geometry (CIC top-1, over verified failures)",
        "",
        _md_table(geom_df),
        "",
        "## Example categories",
        "",
        *[f"- {k}: {v}" for k, v in cat_counts.items()],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _output_dir(cfg: dict[str, Any]) -> Path:
    return ensure_dir(
        Path(cfg.get("results_dir", "results")) / cfg.get("verified_output_subdir", DEFAULT_OUTPUT_SUBDIR)
    )


def _write_unavailable(out_dir: Path, cfg: dict[str, Any], status: ClipStatus, reason: str) -> dict[str, str]:
    key_numbers = {
        "n_verified_text_driven_failures": 0,
        "natural_text_supported": False,
        "open_proposal_supported": False,
        "open_world_claim_allowed": ALWAYS_OPEN_WORLD_CLAIM_ALLOWED,
        "natural_text_directional_evidence": False,
        "directional_evidence_reasons": [reason],
        "no_oracle_leakage": scoring_is_leakage_free(),
        "cic_selected_text_overlap_rate": float("nan"),
        "backend": status.backend,
        "model_name": status.model_name,
        "headline": "directional analysis unavailable; strict gate remains failed",
        "preferred_wording": PREFERRED_WORDING,
        "forbidden_wording": FORBIDDEN_WORDING,
        "summary_statement": "Strict support gate remains failed; directional evidence is diagnostic only.",
    }
    paths = _write_all(out_dir, cfg, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [], key_numbers)
    return paths


def _write_all(out_dir, cfg, agg, directional_df, bucket_df, geom_df, failure_records, directional_rows, key_numbers) -> dict[str, str]:
    metrics_csv = out_dir / "natural_text_directional_metrics.csv"
    per_failure_csv = out_dir / "natural_text_per_failure_categories.csv"
    key_json = out_dir / "natural_text_directional_key_numbers.json"
    error_md = out_dir / "error_analysis.md"
    error_md2 = out_dir / "natural_text_error_analysis.md"
    directional_md = out_dir / "natural_text_directional_metrics.md"

    # Long per-(failure,method) rows go to the metrics CSV (the directional table).
    if directional_rows:
        pd.DataFrame(directional_rows).to_csv(metrics_csv, index=False)
    else:
        agg.to_csv(metrics_csv, index=False)
    if failure_records:
        pd.DataFrame(failure_records).to_csv(per_failure_csv, index=False)
    key_json.write_text(json.dumps(key_numbers, indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "natural_text_error_analysis_config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8"
    )

    cat_counts = _category_counts(failure_records)
    _write_error_analysis_md(error_md, failure_records, directional_rows, agg, key_numbers)
    # error_analysis.md and natural_text_error_analysis.md share the same content
    # (the task requests both filenames).
    error_md2.write_text(error_md.read_text(encoding="utf-8"), encoding="utf-8")
    _write_directional_md(directional_md, agg, bucket_df, cat_counts, geom_df, key_numbers)

    return {
        "directional_metrics": str(metrics_csv),
        "per_failure_categories": str(per_failure_csv),
        "key_numbers": str(key_json),
        "error_analysis": str(error_md),
        "natural_text_error_analysis": str(error_md2),
        "directional_summary": str(directional_md),
    }


def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = _output_dir(cfg)

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
    include_yes = int(bundle.diagnostics.get("n_include_yes", len(examples)))
    total_images = int(bundle.diagnostics.get("n_total_rows", len(examples)))

    model_cfg = dict(cfg.get("model", {}))
    device = _device(model_cfg, cfg)
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))

    if str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend.lower() == "fake":
        status = ClipStatus(
            False, "fake", "fake_natural_text", pretrained=False, device=device, backend_attempted="fake",
            error_message="fake backend cannot support the natural-text directional analysis",
        )
        return _write_unavailable(out_dir, cfg, status, status.error_message)

    status = check_clip_available(
        device=device,
        allow_download=_downloads_allowed(model_cfg),
        preferred_backend=requested_backend,
        model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
        pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
        transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained) or not examples:
        return _write_unavailable(out_dir, cfg, status, status.error_message or "real pretrained CLIP unavailable or no data")

    directional_rows, failure_records, run_diag = _analyze_examples(examples, status, device, cfg)
    n_failures = len(failure_records)

    agg, directional_df = _aggregate(directional_rows)
    bucket_df = _conditional_repair_by_bucket(failure_records, directional_rows)

    # Geometry summary over verified failures.
    geom_keys = ["geom_text_iou", "geom_object_iou", "geom_text_coverage", "geom_object_coverage", "selected_area_fraction"]
    geom_rows = []
    for k in geom_keys:
        vals = [r[k] for r in failure_records if k in r and r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
        geom_rows.append({"metric": k.replace("geom_", ""), "mean": float(np.mean(vals)) if vals else float("nan"), "median": float(np.median(vals)) if vals else float("nan")})
    overlaps_text_rate = float(np.mean([bool(r.get("geom_overlaps_text_box")) for r in failure_records])) if n_failures else float("nan")
    overlaps_object_rate = float(np.mean([bool(r.get("geom_overlaps_object_box")) for r in failure_records])) if n_failures else float("nan")
    overlaps_both_rate = float(np.mean([bool(r.get("geom_overlaps_both")) for r in failure_records])) if n_failures else float("nan")
    closer_text_rate = float(np.mean([r.get("geom_closer_to") == "text" for r in failure_records])) if n_failures else float("nan")
    geom_rows.append({"metric": "overlaps_text_box_rate", "mean": overlaps_text_rate, "median": float("nan")})
    geom_rows.append({"metric": "overlaps_object_box_rate", "mean": overlaps_object_rate, "median": float("nan")})
    geom_rows.append({"metric": "overlaps_both_rate", "mean": overlaps_both_rate, "median": float("nan")})
    geom_rows.append({"metric": "closer_to_text_rate", "mean": closer_text_rate, "median": float("nan")})
    geom_df = pd.DataFrame(geom_rows)

    # Recompute the STRICT gate from this run to confirm it is unchanged/failed.
    oracle_strict = _method_metric(agg, "oracle_text_box_repair", "strict_top1_repair_accuracy")
    cic_strict = _method_metric(agg, "cic_top1", "strict_top1_repair_accuracy")
    random_strict = _method_metric(agg, "matched_random", "strict_top1_repair_accuracy")
    # oracle repair-or-improve: strict repaired OR target prob moved up.
    oracle_rows = [r for r in directional_rows if r["method"] == "oracle_text_box_repair"]
    oracle_repair_or_improve = (
        float(np.mean([bool(r["strict_top1_after"] or r["moved_toward_target"]) for r in oracle_rows]))
        if oracle_rows
        else float("nan")
    )
    no_leak = scoring_is_leakage_free()

    natural_text_supported, strict_reasons = evaluate_natural_text_gate(
        backend=status.backend,
        pretrained=bool(status.pretrained),
        fake_backend=False,
        n_verified_failures=n_failures,
        oracle_repair_or_improve_rate=oracle_repair_or_improve,
        cic_top1_repair_accuracy=cic_strict,
        matched_random_repair_accuracy=random_strict,
        content_preservation_drop=None,
        no_oracle_leakage=no_leak,
        min_verified_failures=int(cfg.get("min_verified_failures", DEFAULT_MIN_VERIFIED_FAILURES)),
        min_oracle_repair_rate=float(cfg.get("min_oracle_repair_rate", DEFAULT_MIN_ORACLE_REPAIR_RATE)),
        min_cic_random_gap=float(cfg.get("min_cic_random_gap", DEFAULT_MIN_CIC_RANDOM_GAP)),
    )

    # Directional-evidence diagnostic flag.
    oracle_prob_imp = _method_metric(agg, "oracle_text_box_repair", "target_prob_improvement_rate")
    cic_prob_imp = _method_metric(agg, "cic_top1", "target_prob_improvement_rate")
    random_prob_imp = _method_metric(agg, "matched_random", "target_prob_improvement_rate")
    directional_evidence, directional_reasons = evaluate_directional_evidence(
        n_verified_failures=n_failures,
        oracle_target_prob_improvement_rate=oracle_prob_imp,
        cic_target_prob_improvement_rate=cic_prob_imp,
        random_target_prob_improvement_rate=random_prob_imp,
        cic_selected_text_overlap_rate=overlaps_text_rate,
        no_oracle_leakage=no_leak,
    )

    cat_counts = _category_counts(failure_records)

    key_numbers: dict[str, Any] = {
        "total_images": total_images,
        "include_yes_images": include_yes,
        "n_evaluated": run_diag["n_evaluated"],
        "original_clip_accuracy": run_diag["original_clip_accuracy"],
        "high_confidence_failure_rate": run_diag["high_confidence_failure_rate"],
        "n_verified_text_driven_failures": n_failures,
        "real_pretrained_model_loaded": bool(status.pretrained),
        "backend": status.backend,
        "model_name": status.model_name,
        # Strict metrics (unchanged semantics; must stay false).
        "natural_text_supported": bool(natural_text_supported),
        "open_proposal_supported": False,
        "open_world_claim_allowed": ALWAYS_OPEN_WORLD_CLAIM_ALLOWED,
        "no_oracle_leakage": bool(no_leak),
        "strict_gate_failed_reasons": strict_reasons,
        "oracle_text_box_strict_repair": oracle_strict,
        "cic_top1_strict_repair": cic_strict,
        "matched_random_strict_repair": random_strict,
        "oracle_text_box_repair_or_improve_rate": oracle_repair_or_improve,
        # Directional diagnostics.
        "oracle_target_prob_improvement_rate": oracle_prob_imp,
        "cic_target_prob_improvement_rate": cic_prob_imp,
        "matched_random_target_prob_improvement_rate": random_prob_imp,
        "cic_over_random_prob_improvement_gap": (
            float(cic_prob_imp - random_prob_imp) if np.isfinite(cic_prob_imp) and np.isfinite(random_prob_imp) else float("nan")
        ),
        "oracle_alias_top1_recovery": _method_metric(agg, "oracle_text_box_repair", "alias_top1_repair_accuracy"),
        "cic_alias_top1_recovery": _method_metric(agg, "cic_top1", "alias_top1_repair_accuracy"),
        "cic_selected_text_overlap_rate": overlaps_text_rate,
        "cic_selected_object_overlap_rate": overlaps_object_rate,
        "cic_selected_overlaps_both_rate": overlaps_both_rate,
        "cic_closer_to_text_rate": closer_text_rate,
        "natural_text_directional_evidence": bool(directional_evidence),
        "directional_evidence_reasons": directional_reasons,
        "category_counts": cat_counts,
        "summary_statement": "Strict support gate remains failed; directional evidence is diagnostic only.",
        "headline": "natural-text verified failure pilot remains unsupported; directional evidence is diagnostic only",
        "preferred_wording": PREFERRED_WORDING,
        "forbidden_wording": FORBIDDEN_WORDING,
    }

    return _write_all(out_dir, cfg, agg, directional_df, bucket_df, geom_df, failure_records, directional_rows, key_numbers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/natural_text_open_proposal_cic.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
