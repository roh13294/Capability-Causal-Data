from __future__ import annotations

"""Verified natural-text failure evaluation for shortcut-agnostic proposal CIC.

Experiment name: ``natural_text_verified_failure_eval``.

Scientific goal: using a *human-verified* curated annotation set
(``data/natural_text_images/verified_annotations.csv``), determine whether the
benchmark actually contains **verified text-driven CLIP failures** and whether
shortcut-agnostic open-proposal CIC repairs those failures better than a matched
random-patch control.

A *verified text-driven failure* is an ``include_in_verified_failure_eval == yes``
image on which the real pretrained CLIP, at high confidence, predicts a
text/logo distractor label instead of the visual target. Repair accuracies are
reported **restricted to these verified failures**.

Scope (enforced in the summary):
* This is **natural-image validation of shortcut-agnostic, proposal-based CIC**,
  not full open-world shortcut discovery. ``open_world_claim_allowed`` is always
  ``False``.
* Candidate *scoring* never sees the true label, OCR text content, the shortcut
  box, or correctness (same leakage-free path as
  ``run_natural_text_open_proposal_cic``). Text/logo boxes are used only as
  candidate *geometry* (OCR-detector-like proposals) and for the eval-only oracle
  upper bound.

This experiment writes ONLY to ``results/natural_text_verified_failure_eval/``
(or ``cfg['verified_output_subdir']``). It does not modify or overwrite the
text-overlay, semantic-decoy, spatial-audit, open-proposal, or final-report
artifacts.
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
from causal_reliability.data.natural_text_dataset import (
    load_verified_natural_text_dataset,
    save_example_images,
)
from causal_reliability.discovery.cic_region_scoring import neutralize_region, score_region_candidates
from causal_reliability.discovery.open_region_proposals import (
    OCR_FAMILY,
    families_present,
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
    _neutralize_boxes,
    _overlaps_any,
    _safe_mean,
    _select_matched_random,
    evaluate_open_proposal_gate,
    families_present_from_diag,
    scoring_is_leakage_free,
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


DEFAULT_OUTPUT_SUBDIR = "natural_text_verified_failure_eval"
DEFAULT_MIN_VERIFIED_FAILURES = 20
DEFAULT_MIN_ORACLE_REPAIR_RATE = 0.70
DEFAULT_MIN_CIC_RANDOM_GAP = 0.15
DEFAULT_MAX_CONTENT_PRESERVATION_DROP = 0.10
ALWAYS_OPEN_WORLD_CLAIM_ALLOWED = False


# --------------------------------------------------------------------------- #
# Natural-text verified-failure gate
# --------------------------------------------------------------------------- #
def evaluate_natural_text_gate(
    *,
    backend: str,
    pretrained: bool,
    fake_backend: bool,
    n_verified_failures: int,
    oracle_repair_or_improve_rate: float,
    cic_top1_repair_accuracy: float,
    matched_random_repair_accuracy: float,
    content_preservation_drop: float | None,
    no_oracle_leakage: bool,
    open_world_claim_allowed: bool = ALWAYS_OPEN_WORLD_CLAIM_ALLOWED,
    min_verified_failures: int = DEFAULT_MIN_VERIFIED_FAILURES,
    min_oracle_repair_rate: float = DEFAULT_MIN_ORACLE_REPAIR_RATE,
    min_cic_random_gap: float = DEFAULT_MIN_CIC_RANDOM_GAP,
    max_content_preservation_drop: float = DEFAULT_MAX_CONTENT_PRESERVATION_DROP,
) -> tuple[bool, list[str]]:
    """Decide whether the curated natural-text verified-failure claim is supported.

    Returns ``(natural_text_supported, failed_reasons)``. All conditions must hold
    for support; ``open_world_claim_allowed`` must remain ``False``.
    """

    reasons: list[str] = []
    if backend not in {"open_clip", "transformers"} or not pretrained or fake_backend or backend == "fake":
        reasons.append("real pretrained OpenCLIP/transformers backend did not load (fake backend or unavailable)")
    if int(n_verified_failures) < int(min_verified_failures):
        reasons.append(
            f"verified text-driven failures {int(n_verified_failures)} < minimum {int(min_verified_failures)}"
        )
    if not np.isfinite(oracle_repair_or_improve_rate) or float(oracle_repair_or_improve_rate) < float(min_oracle_repair_rate):
        reasons.append(
            f"oracle text-box masking repairs/improves {float(oracle_repair_or_improve_rate):.3f} "
            f"< required {float(min_oracle_repair_rate):.2f} of verified failures"
        )
    gap = float(cic_top1_repair_accuracy) - float(matched_random_repair_accuracy)
    if not np.isfinite(gap) or gap < float(min_cic_random_gap):
        reasons.append(
            f"CIC top-1 does not beat matched random by >= {float(min_cic_random_gap):.2f} on verified failures (gap={gap:.3f})"
        )
    if content_preservation_drop is not None and np.isfinite(content_preservation_drop):
        if float(content_preservation_drop) > float(max_content_preservation_drop):
            reasons.append(
                f"content-preservation drop {float(content_preservation_drop):.3f} > "
                f"{float(max_content_preservation_drop):.2f} (and not clearly explained)"
            )
    if not no_oracle_leakage:
        reasons.append("oracle leakage check failed: scoring/proposal rule exposes forbidden parameters")
    if bool(open_world_claim_allowed):
        reasons.append("open_world_claim_allowed must remain False for this validation")
    return (len(reasons) == 0), reasons


# --------------------------------------------------------------------------- #
# Per-example evaluation
# --------------------------------------------------------------------------- #
def _evaluate_examples(
    examples: list[dict[str, Any]],
    status: ClipStatus,
    device: str,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    seed = int(cfg.get("seed", 0))
    max_candidates = int(cfg.get("max_candidates", 64))
    grid_scales = cfg.get("grid_scales")
    high_conf_threshold = float(cfg.get("high_confidence_threshold", 0.7))
    enable_object_box_family = bool(cfg.get("enable_object_box_family", False))

    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    example_records: list[dict[str, Any]] = []

    for ex in examples:
        allowed = list(ex["allowed_clip_labels"])
        label = int(ex["label"])
        target_label = allowed[label]
        distractors = set(ex.get("text_distractor_labels", []))
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
        # Candidate scoring sees pixels, proposal geometry, and model predictions
        # only — no labels, boxes, OCR text, or correctness.
        scores, original_probs = score_region_candidates(pil, proposals, predict_fn)

        orig_probs = np.asarray(original_probs, dtype=np.float64)
        orig_pred = int(orig_probs.argmax())
        orig_conf = float(orig_probs.max())
        orig_target_prob = float(orig_probs[label])
        orig_pred_label = allowed[orig_pred]
        original_correct = bool(orig_pred == label)
        high_conf_failure = bool((not original_correct) and orig_conf >= high_conf_threshold)
        predicted_is_text_distractor = bool((not original_correct) and orig_pred_label in distractors)
        text_driven_candidate = str(ex.get("text_driven_candidate", "")).lower() == "yes"
        verified_text_driven_failure = bool(
            high_conf_failure and predicted_is_text_distractor and text_driven_candidate
        )

        top1 = scores[0] if scores else None
        top3 = scores[:3]

        def repaired(probs) -> tuple[bool, float]:
            arr = np.asarray(probs, dtype=np.float64)
            return bool(int(arr.argmax()) == label), float(arr[label])

        # CIC top-1 repair.
        if top1:
            top1_correct, _ = repaired(predict_fn([neutralize_region(pil, top1.bbox)])[0])
        else:
            top1_correct, _ = original_correct, orig_target_prob

        # CIC top-3 consensus repair (mean over neutralized top-3).
        if top3:
            top3_probs = predict_fn([neutralize_region(pil, s.bbox) for s in top3]).mean(axis=0)
        else:
            top3_probs = orig_probs
        top3_correct, _ = repaired(top3_probs)

        # Matched random proposal repair (area-matched control).
        rand = _select_matched_random(scores, top1, "area_fraction")
        rand_correct, _ = repaired(predict_fn([neutralize_region(pil, rand.bbox)])[0]) if rand else (original_correct, orig_target_prob)

        # Largest-region repair (geometry-only control).
        largest = max(scores, key=lambda s: s.area_fraction) if scores else None
        largest_correct, _ = repaired(predict_fn([neutralize_region(pil, largest.bbox)])[0]) if largest else (original_correct, orig_target_prob)

        # OCR/text-box proposal repair (best-scoring OCR candidate as an
        # inference-time proposal), when present.
        ocr_scores = [s for s in scores if proposal_family(s.proposal_type) == OCR_FAMILY]
        ocr_correct = None
        if ocr_scores:
            ocr_correct, _ = repaired(predict_fn([neutralize_region(pil, ocr_scores[0].bbox)])[0])

        # Oracle text/logo-box repair (eval-only upper bound): neutralize all text boxes.
        oracle_correct = None
        oracle_repair_or_improve = None
        if text_boxes:
            oracle_correct, oracle_target_prob = repaired(predict_fn([_neutralize_boxes(pil, text_boxes)])[0])
            oracle_repair_or_improve = bool(oracle_correct or (oracle_target_prob > orig_target_prob + 1e-6))

        selected_box = top1.bbox if top1 else None
        selected_area = float(top1.area_fraction) if top1 else float("nan")
        overlaps_text = _overlaps_any(selected_box, text_boxes) if selected_box else None
        overlaps_object = _overlaps_any(selected_box, object_boxes) if selected_box else None

        example_records.append(
            {
                "example_id": ex["example_id"],
                "human_label": ex["human_label"],
                "pil": pil,
                "selected_box": selected_box,
                "top1_correct": top1_correct,
                "original_correct": original_correct,
                "verified_text_driven_failure": verified_text_driven_failure,
                "selected_overlaps_text_box": overlaps_text,
                "text_boxes": text_boxes,
                "object_boxes": object_boxes,
            }
        )

        rows.append(
            {
                "example_id": ex["example_id"],
                "human_label": ex["human_label"],
                "target_label": target_label,
                "label": label,
                "allowed_clip_labels": "|".join(allowed),
                "n_text_boxes": len(text_boxes),
                "n_object_boxes": len(object_boxes),
                "original_prediction_label": orig_pred_label,
                "original_confidence": orig_conf,
                "original_target_prob": orig_target_prob,
                "original_correct": original_correct,
                "high_confidence_failure": high_conf_failure,
                "predicted_is_text_distractor": predicted_is_text_distractor,
                "text_driven_candidate": text_driven_candidate,
                "verified_text_driven_failure": verified_text_driven_failure,
                "oracle_text_box_repair_correct": oracle_correct,
                "oracle_text_box_repair_or_improve": oracle_repair_or_improve,
                "cic_top1_repair_correct": top1_correct,
                "cic_top3_repair_correct": top3_correct,
                "matched_random_repair_correct": rand_correct,
                "largest_region_repair_correct": largest_correct,
                "ocr_text_box_proposal_repair_correct": ocr_correct,
                "selected_bbox": "" if selected_box is None else json.dumps([int(v) for v in selected_box]),
                "selected_proposal_type": top1.proposal_type if top1 else "",
                "selected_family": proposal_family(top1.proposal_type) if top1 else "",
                "selected_area_fraction": selected_area,
                "selected_overlaps_text_box": overlaps_text,
                "selected_overlaps_object_box": overlaps_object,
            }
        )

        for rank, s in enumerate(scores, start=1):
            diag_rows.append(
                {
                    "example_id": ex["example_id"],
                    "rank": rank,
                    "candidate_id": s.candidate_id,
                    "proposal_type": s.proposal_type,
                    "proposal_family": proposal_family(s.proposal_type),
                    "bbox": json.dumps([int(v) for v in s.bbox]),
                    "score": float(s.score),
                    "area_fraction": float(s.area_fraction),
                    "overlaps_text_box": _overlaps_any(s.bbox, text_boxes),
                    "overlaps_object_box": _overlaps_any(s.bbox, object_boxes),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(diag_rows), example_records


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _bool_rate(series: pd.Series) -> float:
    vals = series.dropna()
    if not len(vals):
        return float("nan")
    return float(vals.astype(bool).mean())


def _repair_accuracy_on(subset: pd.DataFrame, column: str) -> float:
    if subset.empty or column not in subset:
        return float("nan")
    return _bool_rate(subset[column])


def _build_metrics_table(per_example: pd.DataFrame, failures: pd.DataFrame, status: ClipStatus) -> pd.DataFrame:
    methods = [
        ("original_clip_prediction", "original_correct", False),
        ("oracle_text_box_repair", "oracle_text_box_repair_correct", True),
        ("cic_top1_repair", "cic_top1_repair_correct", False),
        ("cic_top3_repair", "cic_top3_repair_correct", False),
        ("matched_random_proposal_repair", "matched_random_repair_correct", False),
        ("largest_region_repair", "largest_region_repair_correct", False),
        ("ocr_text_box_proposal_repair", "ocr_text_box_proposal_repair_correct", False),
    ]
    rows = []
    for name, col, oracle in methods:
        col_present = col in failures
        avail = failures[col].notna().sum() if col_present else 0
        rows.append(
            {
                "method": name,
                "backend": status.backend,
                "model_name": status.model_name,
                "pretrained_loaded": bool(status.pretrained),
                "oracle_upper_bound": oracle,
                "scope": "verified_failures",
                "n_examples": int(avail),
                "accuracy_on_verified_failures": _repair_accuracy_on(failures, col),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def _write_examples(records: list[dict[str, Any]], out_dir: Path, n: int) -> list[str]:
    ensure_dir(out_dir)
    # Prefer verified failures that CIC repairs, then any verified failure.
    ordered = sorted(
        [r for r in records if r["selected_box"] is not None],
        key=lambda r: (not (r["verified_text_driven_failure"] and r["top1_correct"]), not r["verified_text_driven_failure"]),
    )
    paths: list[str] = []
    for rec in ordered[: max(0, n)]:
        pil = rec["pil"]
        neutral = neutralize_region(pil, rec["selected_box"])
        fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.8))
        axes[0].imshow(pil)
        axes[0].set_title(f"{rec['human_label']} (orig correct={rec['original_correct']})", fontsize=8)
        x0, y0, x1, y1 = rec["selected_box"]
        axes[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#e4572e", lw=2))
        axes[1].imshow(neutral)
        axes[1].set_title(f"top-1 neutralized (correct={rec['top1_correct']})", fontsize=8)
        for ax in axes:
            ax.set_axis_off()
        fig.tight_layout()
        path = out_dir / f"{rec['example_id']}_{rec['human_label'].replace(' ', '_')}_before_after.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _plot_summary(metrics: pd.DataFrame, png: Path) -> None:
    order = [
        "original_clip_prediction",
        "matched_random_proposal_repair",
        "largest_region_repair",
        "ocr_text_box_proposal_repair",
        "cic_top1_repair",
        "cic_top3_repair",
        "oracle_text_box_repair",
    ]
    take = (
        metrics[metrics["method"].isin(order)]
        .set_index("method")
        .reindex(order)
        .dropna(subset=["accuracy_on_verified_failures"])
    )
    plt.figure(figsize=(9.0, 4.6))
    if len(take):
        x = np.arange(len(take))
        plt.bar(x, take["accuracy_on_verified_failures"], color="#4c78a8")
        plt.xticks(x, take.index, rotation=25, ha="right")
        plt.ylim(0, 1.02)
        plt.ylabel("repair accuracy on verified failures")
        plt.title("Verified natural-text text-driven failures: repair accuracy")
    else:
        plt.text(0.5, 0.5, "No verified text-driven failures to plot", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(png, dpi=160)
    plt.close()


def _write_artifacts(
    out_dir: Path,
    cfg: dict[str, Any],
    status: ClipStatus,
    bundle_notes: str,
    per_example: pd.DataFrame,
    diagnostics: pd.DataFrame,
    metrics: pd.DataFrame,
    key_numbers: dict[str, Any],
    example_paths: list[str],
) -> dict[str, str]:
    metrics_csv = out_dir / "verified_failure_metrics.csv"
    per_example_csv = out_dir / "verified_failure_per_example.csv"
    key_json = out_dir / "verified_failure_key_numbers.json"
    summary_md = out_dir / "verified_failure_summary.md"
    diag_csv = out_dir / "verified_failure_proposal_diagnostics.csv"
    plot_png = out_dir / "verified_failure_plot.png"

    metrics.to_csv(metrics_csv, index=False)
    per_example.to_csv(per_example_csv, index=False)
    diagnostics.to_csv(diag_csv, index=False)
    key_json.write_text(json.dumps(key_numbers, indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "verified_failure_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    _plot_summary(metrics, plot_png)

    supported = bool(key_numbers.get("natural_text_supported", False))
    reasons = key_numbers.get("failed_gate_reasons", [])
    inspect_lines = [f"- {p}" for p in key_numbers.get("examples_to_inspect", [])] or ["- (none)"]
    headline = (
        "natural-text verified failure pilot is SUPPORTED"
        if supported
        else "natural-text verified failure pilot remains unsupported"
    )
    summary = [
        "# Verified Natural-Text Text-Driven Failure Evaluation",
        "",
        "**Natural-image validation of shortcut-agnostic, proposal-based CIC** on a ",
        "human-verified curated annotation set. This is an **open-candidate intervention ",
        "search**; it **does not require a pre-specified shortcut family, but still depends ",
        "on candidate region proposals**. It is **not** full open-world shortcut discovery.",
        "",
        f"Backend: `{status.backend}`. Model: `{status.model_name or 'n/a'}`. ",
        f"Real pretrained loaded: `{status.pretrained}`. Fake backend: `{key_numbers.get('fake_backend')}`.",
        f"Data: {bundle_notes}.",
        "",
        f"**Result: {headline}.**",
        ("All gate conditions met." if supported else f"Failed reasons: {'; '.join(reasons) if reasons else 'see key numbers'}."),
        "",
        "## Key numbers",
        "",
        f"- Total annotated images: {key_numbers.get('total_images')}",
        f"- include=yes images evaluated: {key_numbers.get('include_yes_images')}",
        f"- Real pretrained model loaded: {key_numbers.get('real_pretrained_model_loaded')}",
        f"- Fake backend: {key_numbers.get('fake_backend')}",
        f"- Original CLIP accuracy (include=yes): {key_numbers.get('original_clip_accuracy')}",
        f"- High-confidence failure rate (include=yes): {key_numbers.get('high_confidence_failure_rate')}",
        f"- Verified text-driven failures: {key_numbers.get('n_verified_text_driven_failures')}",
        f"- Oracle text-box repair accuracy (verified failures): {key_numbers.get('oracle_text_box_repair_accuracy')}",
        f"- Oracle text-box repair-or-improve rate (verified failures): {key_numbers.get('oracle_text_box_repair_or_improve_rate')}",
        f"- CIC top-1 repair accuracy (verified failures): {key_numbers.get('cic_top1_repair_accuracy')}",
        f"- CIC top-3 repair accuracy (verified failures): {key_numbers.get('cic_top3_repair_accuracy')}",
        f"- Matched-random proposal repair accuracy (verified failures): {key_numbers.get('matched_random_proposal_repair_accuracy')}",
        f"- Largest-region repair accuracy (verified failures): {key_numbers.get('largest_region_repair_accuracy')}",
        f"- OCR/text-box proposal repair accuracy (verified failures): {key_numbers.get('ocr_text_box_proposal_repair_accuracy')}",
        f"- CIC vs matched-random gap (verified failures): {key_numbers.get('cic_random_gap')}",
        f"- Content-preservation rate: {key_numbers.get('content_preservation_rate')}",
        f"- Content-preservation drop: {key_numbers.get('content_preservation_drop')}",
        f"- Selected-region overlap with text boxes (verified failures): {key_numbers.get('selected_overlaps_text_box_rate')}",
        f"- Selected-region overlap with object boxes (verified failures): {key_numbers.get('selected_overlaps_object_box_rate')}",
        f"- Mean selected-area fraction (verified failures): {key_numbers.get('mean_selected_area_fraction')}",
        f"- Candidate families present: {', '.join(key_numbers.get('candidate_families_present', []))}",
        "",
        "## Gate status",
        "",
        f"- `natural_text_supported`: {key_numbers.get('natural_text_supported')}",
        f"- `open_proposal_supported`: {key_numbers.get('open_proposal_supported')}",
        f"- `open_world_claim_allowed`: {key_numbers.get('open_world_claim_allowed')}",
        f"- `no_oracle_leakage`: {key_numbers.get('no_oracle_leakage')}",
        f"- Failed gate reasons: {key_numbers.get('failed_gate_reasons') or 'none'}",
        "",
        "## Examples worth inspecting",
        "",
        *inspect_lines,
        "",
        "## Scope guard",
        "",
        "- Candidate scoring received only pixels, proposal geometry, and model predictions.",
        "- True labels, text/logo boxes, and object boxes are used only for evaluation and the ",
        "  oracle upper bound.",
        "- Proposal-based shortcut discovery, **not** full open-world discovery. ",
        f"  `open_world_claim_allowed = {key_numbers.get('open_world_claim_allowed')}`.",
        "",
        "## Metrics (accuracy restricted to verified text-driven failures)",
        "",
        _markdown_table(metrics),
    ]
    summary_md.write_text("\n".join(summary), encoding="utf-8")

    return {
        "metrics": str(metrics_csv),
        "per_example": str(per_example_csv),
        "key_numbers": str(key_json),
        "summary": str(summary_md),
        "proposal_diagnostics": str(diag_csv),
        "plot": str(plot_png),
        "examples": [str(p) for p in example_paths],
    }


def _write_unavailable(
    out_dir: Path,
    cfg: dict[str, Any],
    status: ClipStatus,
    bundle_notes: str,
    total_images: int,
    include_yes: int,
    fake_backend: bool,
) -> dict[str, str]:
    metrics = pd.DataFrame(
        [
            {
                "method": "unavailable",
                "backend": status.backend,
                "model_name": status.model_name,
                "pretrained_loaded": bool(status.pretrained),
                "oracle_upper_bound": False,
                "scope": "verified_failures",
                "n_examples": 0,
                "accuracy_on_verified_failures": float("nan"),
            }
        ]
    )
    reasons = [status.error_message or "real pretrained CLIP unavailable or no verified data"]
    key_numbers = {
        "total_images": int(total_images),
        "include_yes_images": int(include_yes),
        "real_pretrained_model_loaded": bool(status.available and status.pretrained and not fake_backend),
        "fake_backend": bool(fake_backend),
        "n_verified_text_driven_failures": 0,
        "natural_text_supported": False,
        "open_proposal_supported": False,
        "open_world_claim_allowed": ALWAYS_OPEN_WORLD_CLAIM_ALLOWED,
        "no_oracle_leakage": scoring_is_leakage_free(),
        "failed_gate_reasons": reasons,
        "candidate_families_present": [],
        "examples_to_inspect": [],
        "headline": "natural-text verified failure pilot remains unsupported",
        "preferred_wording": PREFERRED_WORDING,
        "forbidden_wording": FORBIDDEN_WORDING,
    }
    ensure_dir(out_dir / "examples")
    return _write_artifacts(out_dir, cfg, status, bundle_notes, pd.DataFrame(), pd.DataFrame(), metrics, key_numbers, [])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = ensure_dir(
        Path(cfg.get("results_dir", "results")) / cfg.get("verified_output_subdir", DEFAULT_OUTPUT_SUBDIR)
    )
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
        root=root,
        annotations_csv=annotations_csv,
        image_size=image_size,
        split=split,
        include_only=True,
    )
    examples = bundle.examples
    total_images = int(bundle.diagnostics.get("n_total_rows", len(examples)))
    include_yes = int(bundle.diagnostics.get("n_include_yes", len(examples)))

    model_cfg = dict(cfg.get("model", {}))
    device = _device(model_cfg, cfg)
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))

    fake_backend = str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend.lower() == "fake"
    if fake_backend:
        status = ClipStatus(
            False,
            "fake",
            "fake_natural_text",
            pretrained=False,
            device=device,
            backend_attempted="fake",
            error_message="fake backend cannot support the verified natural-text failure claim",
        )
        return _write_unavailable(out_dir, cfg, status, bundle.notes, total_images, include_yes, fake_backend=True)

    status = check_clip_available(
        device=device,
        allow_download=_downloads_allowed(model_cfg),
        preferred_backend=requested_backend,
        model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
        pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
        transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return _write_unavailable(out_dir, cfg, status, bundle.notes, total_images, include_yes, fake_backend=False)
    if not examples:
        return _write_unavailable(out_dir, cfg, status, bundle.notes, total_images, include_yes, fake_backend=False)

    save_example_images(examples, examples_dir)
    per_example, diagnostics, example_records = _evaluate_examples(examples, status, device, cfg)

    failures = per_example[per_example["verified_text_driven_failure"].astype(bool)].copy()
    n_failures = int(len(failures))

    metrics = _build_metrics_table(per_example, failures, status)

    def acc(col: str) -> float:
        return _repair_accuracy_on(failures, col)

    original_acc = _bool_rate(per_example["original_correct"])
    high_conf_failure_rate = _bool_rate(per_example["high_confidence_failure"])
    oracle_acc = acc("oracle_text_box_repair_correct")
    oracle_repair_improve = acc("oracle_text_box_repair_or_improve")
    top1_acc = acc("cic_top1_repair_correct")
    top3_acc = acc("cic_top3_repair_correct")
    random_acc = acc("matched_random_repair_correct")
    largest_acc = acc("largest_region_repair_correct")
    ocr_acc = acc("ocr_text_box_proposal_repair_correct")
    cic_random_gap = (top1_acc - random_acc) if (np.isfinite(top1_acc) and np.isfinite(random_acc)) else float("nan")

    # Content preservation: among include=yes originally-correct images, fraction
    # still correct after CIC top-1 repair (a repair should not break clean cases).
    clean = per_example[per_example["original_correct"].astype(bool)]
    if len(clean):
        content_preservation_rate = _bool_rate(clean["cic_top1_repair_correct"])
        content_preservation_drop = float(1.0 - content_preservation_rate)
    else:
        content_preservation_rate = float("nan")
        content_preservation_drop = None

    # Selected-region geometry, reported over verified failures.
    text_overlap_rate = _bool_rate(failures["selected_overlaps_text_box"]) if n_failures else float("nan")
    object_overlap_rate = _bool_rate(failures["selected_overlaps_object_box"]) if n_failures else float("nan")
    mean_area = _safe_mean(failures["selected_area_fraction"]) if n_failures else float("nan")

    fams = sorted(families_present_from_diag(diagnostics))
    non_ocr_present = any(
        f in {"grid_patch", "connected_component", "high_contrast", "edge_dense", "object_box", "sam_proposal"}
        for f in fams
    )
    no_leak = scoring_is_leakage_free()

    min_failures = int(cfg.get("min_verified_failures", DEFAULT_MIN_VERIFIED_FAILURES))
    min_oracle = float(cfg.get("min_oracle_repair_rate", DEFAULT_MIN_ORACLE_REPAIR_RATE))
    min_gap = float(cfg.get("min_cic_random_gap", DEFAULT_MIN_CIC_RANDOM_GAP))
    max_drop = float(cfg.get("max_content_preservation_drop", DEFAULT_MAX_CONTENT_PRESERVATION_DROP))

    natural_text_supported, failed_reasons = evaluate_natural_text_gate(
        backend=status.backend,
        pretrained=bool(status.pretrained),
        fake_backend=False,
        n_verified_failures=n_failures,
        oracle_repair_or_improve_rate=oracle_repair_improve,
        cic_top1_repair_accuracy=top1_acc,
        matched_random_repair_accuracy=random_acc,
        content_preservation_drop=content_preservation_drop,
        no_oracle_leakage=no_leak,
        open_world_claim_allowed=ALWAYS_OPEN_WORLD_CLAIM_ALLOWED,
        min_verified_failures=min_failures,
        min_oracle_repair_rate=min_oracle,
        min_cic_random_gap=min_gap,
        max_content_preservation_drop=max_drop,
    )

    # Also report the existing open-proposal gate computed on the verified-failure
    # subset for cross-reference.
    open_proposal_supported, _ = evaluate_open_proposal_gate(
        backend=status.backend,
        pretrained=bool(status.pretrained),
        n_images=n_failures,
        cic_top1_repair_accuracy=top1_acc,
        matched_random_repair_accuracy=random_acc,
        content_preservation_drop=content_preservation_drop,
        non_ocr_family_present=non_ocr_present,
        no_oracle_leakage=no_leak,
        min_images=min_failures,
        min_cic_random_gap=min_gap,
        max_content_preservation_drop=max_drop,
    )

    # Examples to inspect: verified failures CIC repairs while selecting a text box.
    inspect = failures[
        failures["cic_top1_repair_correct"].astype("boolean").fillna(False)
        & failures["selected_overlaps_text_box"].astype("boolean").fillna(False)
    ]
    examples_to_inspect = [
        f"{int(r.example_id)}:{r.human_label}" for r in inspect.itertuples(index=False)
    ][:8]

    n_examples_to_save = int(cfg.get("n_example_visualizations", 6))
    example_paths = _write_examples(example_records, examples_dir, n_examples_to_save)

    key_numbers = {
        "total_images": total_images,
        "include_yes_images": include_yes,
        "n_evaluated": int(len(per_example)),
        "real_pretrained_model_loaded": bool(status.pretrained),
        "fake_backend": False,
        "original_clip_accuracy": original_acc,
        "high_confidence_failure_rate": high_conf_failure_rate,
        "n_verified_text_driven_failures": n_failures,
        "oracle_text_box_repair_accuracy": oracle_acc,
        "oracle_text_box_repair_or_improve_rate": oracle_repair_improve,
        "cic_top1_repair_accuracy": top1_acc,
        "cic_top3_repair_accuracy": top3_acc,
        "matched_random_proposal_repair_accuracy": random_acc,
        "largest_region_repair_accuracy": largest_acc,
        "ocr_text_box_proposal_repair_accuracy": (None if not np.isfinite(ocr_acc) else ocr_acc),
        "cic_random_gap": cic_random_gap,
        "content_preservation_rate": content_preservation_rate,
        "content_preservation_drop": content_preservation_drop,
        "selected_overlaps_text_box_rate": (None if not np.isfinite(text_overlap_rate) else text_overlap_rate),
        "selected_overlaps_object_box_rate": (None if not np.isfinite(object_overlap_rate) else object_overlap_rate),
        "mean_selected_area_fraction": (None if not np.isfinite(mean_area) else mean_area),
        "backend": status.backend,
        "model_name": status.model_name,
        "pretrained_loaded": bool(status.pretrained),
        "candidate_families_present": fams,
        "non_ocr_family_present": bool(non_ocr_present),
        "no_oracle_leakage": bool(no_leak),
        "min_verified_failures": min_failures,
        "min_oracle_repair_rate": min_oracle,
        "min_cic_random_gap": min_gap,
        "max_content_preservation_drop": max_drop,
        "natural_text_supported": bool(natural_text_supported),
        "open_proposal_supported": bool(open_proposal_supported),
        "failed_gate_reasons": failed_reasons,
        "open_world_claim_allowed": ALWAYS_OPEN_WORLD_CLAIM_ALLOWED,
        "examples_to_inspect": examples_to_inspect,
        "headline": (
            "natural-text verified failure pilot is SUPPORTED"
            if natural_text_supported
            else "natural-text verified failure pilot remains unsupported"
        ),
        "preferred_wording": PREFERRED_WORDING,
        "forbidden_wording": FORBIDDEN_WORDING,
    }

    return _write_artifacts(out_dir, cfg, status, bundle.notes, per_example, diagnostics, metrics, key_numbers, example_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/natural_text_open_proposal_cic.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
