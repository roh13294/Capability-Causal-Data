from __future__ import annotations

"""Natural-image validation of shortcut-agnostic, proposal-based CIC.

Experiment name: ``natural_text_open_proposal_cic``.

Scientific goal: test whether CIC can detect and repair shortcut-dependent
predictions on *natural* images containing real scene text / signage / memes,
without being told which region is the shortcut. We generate a broad open set of
candidate regions (grid patches, connected components, high-contrast regions,
edge-dense bands, optional OCR/object boxes) and score them with the same
model-agnostic CIC region-scoring logic used elsewhere
(``cic_region_scoring.score_region_candidates``).

Scope and wording (enforced in the summary):
* "shortcut-agnostic proposal-based CIC" / "open-candidate intervention search".
* "natural-image validation".
* "does not require a pre-specified shortcut family, but still depends on
  candidate region proposals".
This is proposal-based shortcut discovery, *not* full open-world discovery:
``open_world_claim_allowed`` is always ``False``.
"""

import argparse
import inspect
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
from causal_reliability.data.natural_text_dataset import load_natural_text_dataset, save_example_images
from causal_reliability.discovery.cic_region_scoring import neutralize_region, score_region_candidates
from causal_reliability.discovery.open_region_proposals import (
    OCR_FAMILY,
    RANDOM_FAMILY,
    families_present,
    generate_open_region_proposals,
    has_non_ocr_family,
    proposal_family,
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


PROMPT_TEMPLATE = "a photo of a {label}"
DEFAULT_MIN_CIC_RANDOM_GAP = 0.15

# Wording compliance, surfaced in the summary so downstream report builders reuse it.
PREFERRED_WORDING = [
    "shortcut-agnostic proposal-based CIC",
    "open-candidate intervention search",
    "natural-image validation",
    "does not require a pre-specified shortcut family, but still depends on candidate region proposals",
]
FORBIDDEN_WORDING = [
    "fully open-world shortcut discovery",
    "solves shortcut discovery",
    "general robustness",
    "universal repair",
    "deployment-ready",
]

# The candidate *scoring* rule must never see the answer. Note: text/object boxes
# are allowed as candidate *geometry* inputs to the proposal generator (OCR and
# object proposal families), so they are excluded from the proposal-side check.
FORBIDDEN_SCORING_PARAMS = {
    "true_label",
    "label",
    "human_label",
    "allowed_clip_labels",
    "overlay_bbox",
    "text_boxes",
    "object_boxes",
    "shortcut_bbox",
    "shortcut_identity",
    "ocr_text",
    "correctness",
    "test_correctness",
    "repaired_correct",
    "benchmark_condition",
    "regime",
}
FORBIDDEN_PROPOSAL_PARAMS = {
    "true_label",
    "label",
    "human_label",
    "allowed_clip_labels",
    "overlay_bbox",
    "shortcut_bbox",
    "shortcut_identity",
    "ocr_text",
    "correctness",
    "test_correctness",
    "repaired_correct",
    "benchmark_condition",
    "regime",
}


# --------------------------------------------------------------------------- #
# Geometry / prediction helpers
# --------------------------------------------------------------------------- #
def _device(model_cfg: dict[str, Any], cfg: dict[str, Any]) -> str:
    requested = str(model_cfg.get("device", "auto"))
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() and bool(cfg.get("prefer_gpu", False)) else "cpu"
    return requested


def _downloads_allowed(model_cfg: dict[str, Any]) -> bool:
    return bool(model_cfg.get("allow_pretrained_download", model_cfg.get("allow_download", False)))


def _pil_to_tensor(images: list[Image.Image]) -> torch.Tensor:
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


def _overlaps_any(box, boxes, threshold: float = 0.1) -> bool:
    return any(_iou(box, b) >= threshold for b in boxes) if boxes else False


def _build_predict_fn(status: ClipStatus, allowed_labels: list[str], device: str):
    prompts = [PROMPT_TEMPLATE.format(label=name) for name in allowed_labels]
    classifier = ClipZeroShotClassifier(status, allowed_labels, prompts=prompts, device=device)

    def predict_fn(images: list[Image.Image]) -> np.ndarray:
        out = classifier.predict(_pil_to_tensor(images))
        return np.asarray(out["probabilities"].detach().cpu().numpy(), dtype=np.float64)

    return predict_fn


def scoring_is_leakage_free() -> bool:
    """True iff the candidate scoring/proposal rule exposes no oracle parameters."""

    score_params = set(inspect.signature(score_region_candidates).parameters)
    proposal_params = set(inspect.signature(generate_open_region_proposals).parameters)
    leaking = (score_params & FORBIDDEN_SCORING_PARAMS) | (proposal_params & FORBIDDEN_PROPOSAL_PARAMS)
    return not leaking


def _safe_mean(series: pd.Series) -> float:
    vals = series.dropna()
    return float(vals.mean()) if len(vals) else float("nan")


# --------------------------------------------------------------------------- #
# Open-proposal claim gate
# --------------------------------------------------------------------------- #
def evaluate_open_proposal_gate(
    *,
    backend: str,
    pretrained: bool,
    n_images: int,
    cic_top1_repair_accuracy: float,
    matched_random_repair_accuracy: float,
    content_preservation_drop: float | None,
    non_ocr_family_present: bool,
    no_oracle_leakage: bool,
    min_images: int,
    min_cic_random_gap: float = DEFAULT_MIN_CIC_RANDOM_GAP,
    max_content_preservation_drop: float = 0.10,
) -> tuple[bool, list[str]]:
    """Decide whether the open-proposal natural-image claim is supported.

    The claim is supported only if a real pretrained OpenCLIP/transformers model
    loaded, there are enough images, CIC top-1 beats matched random proposals by
    the required gap, content preservation is acceptable, no oracle leakage is
    detected, and at least one non-OCR proposal family contributed (so the method
    is not merely "use the OCR box").
    """

    reasons: list[str] = []
    if backend not in {"open_clip", "transformers"} or not pretrained or backend == "fake":
        reasons.append("real pretrained OpenCLIP/transformers backend did not load (fake backend or unavailable)")
    if int(n_images) < int(min_images):
        reasons.append(f"n_images {int(n_images)} < minimum {int(min_images)}")
    gap = float(cic_top1_repair_accuracy) - float(matched_random_repair_accuracy)
    if not np.isfinite(gap) or gap < float(min_cic_random_gap):
        reasons.append(f"CIC top-1 does not beat matched random by >= {float(min_cic_random_gap):.2f} (gap={gap:.3f})")
    if content_preservation_drop is not None and np.isfinite(content_preservation_drop):
        if float(content_preservation_drop) > float(max_content_preservation_drop):
            reasons.append(
                f"content-preservation drop {float(content_preservation_drop):.3f} > {float(max_content_preservation_drop):.2f}"
            )
    if not non_ocr_family_present:
        reasons.append("no non-OCR proposal family contributed candidates (method reduces to OCR box)")
    if not no_oracle_leakage:
        reasons.append("oracle leakage check failed: scoring/proposal rule exposes forbidden parameters")
    return (len(reasons) == 0), reasons


# --------------------------------------------------------------------------- #
# Per-example evaluation
# --------------------------------------------------------------------------- #
def _neutralize_boxes(pil: Image.Image, boxes: list) -> Image.Image:
    out = pil
    for box in boxes:
        out = neutralize_region(out, tuple(int(v) for v in box))
    return out


def _select_matched_random(scores: list, top1, field: str = "area_fraction"):
    randoms = [s for s in scores if s.proposal_type == "random_patch_control"]
    if not randoms:
        return None
    target = float(getattr(top1, field, 0.0)) if top1 is not None else 0.0
    return min(randoms, key=lambda s: abs(float(getattr(s, field, 0.0)) - target))


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

    cert_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    example_records: list[dict[str, Any]] = []

    for ex in examples:
        allowed = list(ex["allowed_clip_labels"])
        label = int(ex["label"])
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
        # Candidate scoring uses pixels, proposal geometry, and model predictions
        # only — no labels, boxes, OCR text, or correctness.
        scores, original_probs = score_region_candidates(pil, proposals, predict_fn)

        orig_pred = int(original_probs.argmax())
        orig_conf = float(original_probs.max())
        original_correct = bool(orig_pred == label)
        high_conf_failure = bool((not original_correct) and orig_conf >= high_conf_threshold)

        top1 = scores[0] if scores else None
        top3 = scores[:3]

        def repaired(probs) -> tuple[int, bool, float]:
            pred = int(np.asarray(probs).argmax())
            return pred, bool(pred == label), float(np.asarray(probs).max())

        # CIC top-1 repair.
        top1_probs = predict_fn([neutralize_region(pil, top1.bbox)])[0] if top1 else original_probs
        top1_pred, top1_correct, top1_conf = repaired(top1_probs)

        # CIC top-3 consensus repair (mean over neutralized top-3).
        if top3:
            top3_neutral = predict_fn([neutralize_region(pil, s.bbox) for s in top3])
            top3_probs = top3_neutral.mean(axis=0)
        else:
            top3_probs = original_probs
        top3_pred, top3_correct, _ = repaired(top3_probs)

        # Matched random proposal repair (area-matched control).
        rand = _select_matched_random(scores, top1, "area_fraction")
        rand_probs = predict_fn([neutralize_region(pil, rand.bbox)])[0] if rand else original_probs
        rand_pred, rand_correct, _ = repaired(rand_probs)

        # Largest-region repair (geometry-only control).
        largest = max(scores, key=lambda s: s.area_fraction) if scores else None
        largest_probs = predict_fn([neutralize_region(pil, largest.bbox)])[0] if largest else original_probs
        _, largest_correct, _ = repaired(largest_probs)

        # OCR-only proposal repair (best-scoring OCR candidate), when present.
        ocr_scores = [s for s in scores if proposal_family(s.proposal_type) == OCR_FAMILY]
        ocr_correct = None
        if ocr_scores:
            ocr_probs = predict_fn([neutralize_region(pil, ocr_scores[0].bbox)])[0]
            _, ocr_correct, _ = repaired(ocr_probs)

        # Oracle text-box repair (eval-only upper bound): neutralize all text boxes.
        oracle_correct = None
        if text_boxes:
            oracle_probs = predict_fn([_neutralize_boxes(pil, text_boxes)])[0]
            _, oracle_correct, _ = repaired(oracle_probs)

        selected_box = top1.bbox if top1 else None
        selected_area = float(top1.area_fraction) if top1 else float("nan")
        overlaps_text = _overlaps_any(selected_box, text_boxes) if selected_box else None
        overlaps_object = _overlaps_any(selected_box, object_boxes) if selected_box else None

        example_records.append(
            {
                "example_id": ex["example_id"],
                "pil": pil,
                "selected_box": selected_box,
                "top1_correct": top1_correct,
                "original_correct": original_correct,
                "text_boxes": text_boxes,
                "object_boxes": object_boxes,
            }
        )

        base = {
            "example_id": ex["example_id"],
            "human_label": ex["human_label"],
            "label": label,
            "source": ex.get("source", ""),
            "allowed_clip_labels": "|".join(allowed),
            "original_prediction_index": orig_pred,
            "original_confidence": orig_conf,
            "original_correct": original_correct,
            "high_confidence_failure": high_conf_failure,
            "selected_bbox": "" if selected_box is None else json.dumps([int(v) for v in selected_box]),
            "selected_proposal_type": top1.proposal_type if top1 else "",
            "selected_family": proposal_family(top1.proposal_type) if top1 else "",
            "selected_area_fraction": selected_area,
            "selected_overlaps_text_box": overlaps_text,
            "selected_overlaps_object_box": overlaps_object,
        }

        def cert(method: str, correct, *, oracle: bool = False, available: bool = True) -> dict[str, Any]:
            row = dict(base)
            row.update(
                {
                    "method": method,
                    "repaired_correct": (np.nan if (correct is None or not available) else bool(correct)),
                    "oracle_upper_bound": oracle,
                    "method_available": bool(available and correct is not None),
                }
            )
            return row

        cert_rows.extend(
            [
                cert("original_clip_prediction", original_correct),
                cert("cic_top1_repair", top1_correct),
                cert("cic_top3_repair", top3_correct),
                cert("matched_random_proposal_repair", rand_correct),
                cert("largest_region_repair", largest_correct),
                cert("ocr_only_proposal_repair", ocr_correct, available=ocr_correct is not None),
                cert("oracle_text_box_repair", oracle_correct, oracle=True, available=oracle_correct is not None),
            ]
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
                    "prediction_flip_indicator": float(s.prediction_flip_indicator),
                    "js_divergence": float(s.js_divergence),
                    "overlaps_text_box": _overlaps_any(s.bbox, text_boxes),
                    "overlaps_object_box": _overlaps_any(s.bbox, object_boxes),
                }
            )

    return pd.DataFrame(cert_rows), pd.DataFrame(diag_rows), example_records


# --------------------------------------------------------------------------- #
# Aggregation & artifacts
# --------------------------------------------------------------------------- #
def _aggregate_metrics(certs: pd.DataFrame, status: ClipStatus) -> pd.DataFrame:
    rows = []
    for method, df in certs.groupby("method", sort=False):
        avail = df["method_available"].astype(bool)
        repaired = df["repaired_correct"]
        rows.append(
            {
                "method": method,
                "backend": status.backend,
                "model_name": status.model_name,
                "pretrained_loaded": bool(status.pretrained),
                "oracle_upper_bound": bool(df["oracle_upper_bound"].any()),
                "n_examples": int(avail.sum()),
                "accuracy": _safe_mean(repaired[avail]) if bool(avail.sum()) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _method_accuracy(metrics: pd.DataFrame, method: str) -> float:
    sub = metrics[metrics["method"] == method]
    if sub.empty:
        return float("nan")
    return float(sub["accuracy"].iloc[0])


def _write_examples(records: list[dict[str, Any]], out_dir: Path, n: int) -> list[str]:
    ensure_dir(out_dir)
    paths: list[str] = []
    for rec in records[: max(0, n)]:
        if rec["selected_box"] is None:
            continue
        pil = rec["pil"]
        neutral = neutralize_region(pil, rec["selected_box"])
        fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.8))
        axes[0].imshow(pil)
        axes[0].set_title(f"original (correct={rec['original_correct']})", fontsize=8)
        x0, y0, x1, y1 = rec["selected_box"]
        axes[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#e4572e", lw=2))
        axes[1].imshow(neutral)
        axes[1].set_title(f"top-1 neutralized (correct={rec['top1_correct']})", fontsize=8)
        for ax in axes:
            ax.set_axis_off()
        fig.tight_layout()
        path = out_dir / f"{rec['example_id']}_before_after.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _plot_summary(metrics: pd.DataFrame, png: Path) -> None:
    order = [
        "original_clip_prediction",
        "matched_random_proposal_repair",
        "largest_region_repair",
        "ocr_only_proposal_repair",
        "cic_top1_repair",
        "cic_top3_repair",
        "oracle_text_box_repair",
    ]
    take = metrics[metrics["method"].isin(order)].set_index("method").reindex(order).dropna(subset=["accuracy"])
    plt.figure(figsize=(9.0, 4.6))
    if len(take):
        x = np.arange(len(take))
        plt.bar(x, take["accuracy"], color="#4c78a8")
        plt.xticks(x, take.index, rotation=25, ha="right")
        plt.ylim(0, 1.02)
        plt.ylabel("accuracy")
        plt.title("Natural-image open-candidate CIC repair accuracy")
    else:
        plt.text(0.5, 0.5, "No eligible natural-image repair metrics", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(png, dpi=160)
    plt.close()


def _write_artifacts(
    out_dir: Path,
    cfg: dict[str, Any],
    status: ClipStatus,
    bundle_notes: str,
    certs: pd.DataFrame,
    diagnostics: pd.DataFrame,
    metrics: pd.DataFrame,
    key_numbers: dict[str, Any],
    example_paths: list[str],
) -> dict[str, str]:
    metrics_csv = out_dir / "natural_text_metrics.csv"
    key_json = out_dir / "natural_text_key_numbers.json"
    summary_md = out_dir / "natural_text_summary.md"
    diag_csv = out_dir / "proposal_diagnostics.csv"
    certs_csv = out_dir / "natural_text_certificates.csv"
    plot_png = out_dir / "natural_text_plot.png"

    metrics.to_csv(metrics_csv, index=False)
    certs.to_csv(certs_csv, index=False)
    diagnostics.to_csv(diag_csv, index=False)
    key_json.write_text(json.dumps(key_numbers, indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "natural_text_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    _plot_summary(metrics, plot_png)

    supported = bool(key_numbers.get("open_proposal_supported", False))
    reasons = key_numbers.get("open_proposal_support_reasons", [])
    summary = [
        "# Natural-Image Open-Candidate (Shortcut-Agnostic) CIC",
        "",
        "This experiment is **natural-image validation** of **shortcut-agnostic proposal-based CIC**: ",
        "an **open-candidate intervention search** over candidate regions generated without a ",
        "pre-specified shortcut family. The method **does not require a pre-specified shortcut family, ",
        "but still depends on candidate region proposals**.",
        "",
        f"Backend: `{status.backend}`. Model: `{status.model_name or 'n/a'}`. Pretrained loaded: `{status.pretrained}`.",
        f"Data: {bundle_notes}.",
        f"Open-proposal claim supported: `{supported}`.",
        ("All gate conditions met." if supported else f"Not supported: {'; '.join(reasons) if reasons else 'see key numbers'}."),
        "",
        "## Key numbers",
        "",
        f"- n images: {key_numbers.get('n_images')}",
        f"- Original CLIP accuracy: {key_numbers.get('original_clip_accuracy')}",
        f"- High-confidence failure rate: {key_numbers.get('high_confidence_failure_rate')}",
        f"- CIC top-1 repair accuracy: {key_numbers.get('cic_top1_repair_accuracy')}",
        f"- CIC top-3 repair accuracy: {key_numbers.get('cic_top3_repair_accuracy')}",
        f"- Matched-random proposal repair accuracy: {key_numbers.get('matched_random_proposal_repair_accuracy')}",
        f"- Largest-region repair accuracy: {key_numbers.get('largest_region_repair_accuracy')}",
        f"- OCR-only proposal repair accuracy: {key_numbers.get('ocr_only_proposal_repair_accuracy')}",
        f"- Oracle text-box repair (eval-only upper bound): {key_numbers.get('oracle_text_box_repair_accuracy')}",
        f"- CIC vs matched-random gap: {key_numbers.get('cic_random_gap')}",
        f"- Content-preservation rate: {key_numbers.get('content_preservation_rate')}",
        f"- Mean selected-region area fraction: {key_numbers.get('mean_selected_region_area_fraction')}",
        f"- Selected overlaps OCR/text box rate: {key_numbers.get('selected_overlaps_text_box_rate')}",
        f"- Selected overlaps object box rate: {key_numbers.get('selected_overlaps_object_box_rate')}",
        f"- Candidate families present: {', '.join(key_numbers.get('candidate_families_present', []))}",
        "",
        "## Scope",
        "",
        "- Candidate scoring received only pixels, proposal geometry, and model predictions. It did not ",
        "  receive true labels, OCR text content, the shortcut box, correctness, or the benchmark condition.",
        "- True labels, text boxes, and object boxes are used only for evaluation and oracle upper bounds.",
        "- This is proposal-based shortcut discovery, **not** full open-world discovery. ",
        f"  `open_world_claim_allowed = {key_numbers.get('open_world_claim_allowed')}`.",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
    ]
    summary_md.write_text("\n".join(summary), encoding="utf-8")

    out = {
        "metrics": str(metrics_csv),
        "key_numbers": str(key_json),
        "summary": str(summary_md),
        "proposal_diagnostics": str(diag_csv),
        "certificates": str(certs_csv),
        "plot": str(plot_png),
        "examples": [str(p) for p in example_paths],
    }
    return out


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


def _write_unavailable(out_dir: Path, cfg: dict[str, Any], status: ClipStatus, bundle_notes: str, n_images: int) -> dict[str, str]:
    metrics = pd.DataFrame(
        [
            {
                "method": "unavailable",
                "backend": status.backend,
                "model_name": status.model_name,
                "pretrained_loaded": bool(status.pretrained),
                "oracle_upper_bound": False,
                "n_examples": 0,
                "accuracy": float("nan"),
            }
        ]
    )
    key_numbers = {
        "n_images": int(n_images),
        "backend": status.backend,
        "model_name": status.model_name,
        "pretrained_loaded": bool(status.pretrained),
        "open_proposal_supported": False,
        "open_proposal_support_reasons": [status.error_message or "real pretrained CLIP unavailable or no data"],
        "open_world_claim_allowed": False,
        "no_oracle_leakage": scoring_is_leakage_free(),
        "candidate_families_present": [],
        "preferred_wording": PREFERRED_WORDING,
        "forbidden_wording": FORBIDDEN_WORDING,
    }
    ensure_dir(out_dir / "examples")
    return _write_artifacts(
        out_dir,
        cfg,
        status,
        bundle_notes,
        pd.DataFrame(),
        pd.DataFrame(),
        metrics,
        key_numbers,
        [],
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / cfg.get("output_subdir", "natural_text_open_proposal_cic"))
    examples_dir = ensure_dir(out_dir / "examples")

    data_cfg = dict(cfg.get("data", {}))
    bundle = load_natural_text_dataset(data_cfg)
    examples = bundle.examples

    model_cfg = dict(cfg.get("model", {}))
    device = _device(model_cfg, cfg)
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))

    # Fake backend is never allowed to support the claim.
    if str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend.lower() == "fake":
        status = ClipStatus(False, "fake", "fake_natural_text", pretrained=False, device=device, backend_attempted="fake", error_message="fake backend cannot support the open-proposal claim")
        return _write_unavailable(out_dir, cfg, status, bundle.notes, len(examples))

    status = check_clip_available(
        device=device,
        allow_download=_downloads_allowed(model_cfg),
        preferred_backend=requested_backend,
        model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
        pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
        transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return _write_unavailable(out_dir, cfg, status, bundle.notes, len(examples))
    if not examples:
        return _write_unavailable(out_dir, cfg, status, bundle.notes, 0)

    save_example_images(examples, examples_dir)
    certs, diagnostics, example_records = _evaluate_examples(examples, status, device, cfg)
    metrics = _aggregate_metrics(certs, status)

    # Headline example-level numbers.
    per_example = certs[certs["method"] == "original_clip_prediction"]
    n_images = int(len(per_example))
    original_acc = _safe_mean(per_example["repaired_correct"])
    high_conf_failure_rate = _safe_mean(per_example["high_confidence_failure"].astype(float))

    top1_acc = _method_accuracy(metrics, "cic_top1_repair")
    top3_acc = _method_accuracy(metrics, "cic_top3_repair")
    random_acc = _method_accuracy(metrics, "matched_random_proposal_repair")
    largest_acc = _method_accuracy(metrics, "largest_region_repair")
    ocr_acc = _method_accuracy(metrics, "ocr_only_proposal_repair")
    oracle_acc = _method_accuracy(metrics, "oracle_text_box_repair")
    cic_random_gap = (top1_acc - random_acc) if (np.isfinite(top1_acc) and np.isfinite(random_acc)) else float("nan")

    # Content preservation: among originally-correct images, fraction still
    # correct after top-1 repair (a repair should not break clean predictions).
    top1_rows = certs[certs["method"] == "cic_top1_repair"].set_index("example_id")
    orig_rows = per_example.set_index("example_id")
    clean_ids = orig_rows.index[orig_rows["repaired_correct"].astype(bool)]
    if len(clean_ids):
        preserved = top1_rows.loc[clean_ids, "repaired_correct"].astype(bool)
        content_preservation_rate = float(preserved.mean())
        content_preservation_drop = float(1.0 - content_preservation_rate)
    else:
        content_preservation_rate = float("nan")
        content_preservation_drop = None

    fams = sorted(families_present_from_diag(diagnostics))
    non_ocr_present = any(f in {"grid_patch", "connected_component", "high_contrast", "edge_dense", "object_box", "sam_proposal"} for f in fams)
    text_overlap_rate = _safe_mean(per_example["selected_overlaps_text_box"].astype(float)) if "selected_overlaps_text_box" in per_example else float("nan")
    object_overlap_rate = _safe_mean(per_example["selected_overlaps_object_box"].astype(float)) if "selected_overlaps_object_box" in per_example else float("nan")
    mean_area = _safe_mean(per_example["selected_area_fraction"])

    no_leak = scoring_is_leakage_free()
    min_images = int(cfg.get("min_images", 30))
    min_gap = float(cfg.get("min_cic_random_gap", DEFAULT_MIN_CIC_RANDOM_GAP))
    max_drop = float(cfg.get("max_content_preservation_drop", 0.10))
    supported, reasons = evaluate_open_proposal_gate(
        backend=status.backend,
        pretrained=bool(status.pretrained),
        n_images=n_images,
        cic_top1_repair_accuracy=top1_acc,
        matched_random_repair_accuracy=random_acc,
        content_preservation_drop=content_preservation_drop,
        non_ocr_family_present=non_ocr_present,
        no_oracle_leakage=no_leak,
        min_images=min_images,
        min_cic_random_gap=min_gap,
        max_content_preservation_drop=max_drop,
    )

    key_numbers = {
        "n_images": n_images,
        "original_clip_accuracy": original_acc,
        "high_confidence_failure_rate": high_conf_failure_rate,
        "cic_top1_repair_accuracy": top1_acc,
        "cic_top3_repair_accuracy": top3_acc,
        "matched_random_proposal_repair_accuracy": random_acc,
        "largest_region_repair_accuracy": largest_acc,
        "ocr_only_proposal_repair_accuracy": (None if not np.isfinite(ocr_acc) else ocr_acc),
        "oracle_text_box_repair_accuracy": (None if not np.isfinite(oracle_acc) else oracle_acc),
        "cic_random_gap": cic_random_gap,
        "content_preservation_rate": content_preservation_rate,
        "content_preservation_drop": content_preservation_drop,
        "mean_selected_region_area_fraction": mean_area,
        "selected_overlaps_text_box_rate": (None if not np.isfinite(text_overlap_rate) else text_overlap_rate),
        "selected_overlaps_object_box_rate": (None if not np.isfinite(object_overlap_rate) else object_overlap_rate),
        "backend": status.backend,
        "model_name": status.model_name,
        "pretrained_loaded": bool(status.pretrained),
        "candidate_families_present": fams,
        "non_ocr_family_present": bool(non_ocr_present),
        "no_oracle_leakage": bool(no_leak),
        "min_images": min_images,
        "min_cic_random_gap": min_gap,
        "max_content_preservation_drop": max_drop,
        "open_proposal_supported": bool(supported),
        "open_proposal_support_reasons": reasons,
        "open_world_claim_allowed": False,
        "preferred_wording": PREFERRED_WORDING,
        "forbidden_wording": FORBIDDEN_WORDING,
    }

    n_examples_to_save = int(cfg.get("n_example_visualizations", 6))
    example_paths = _write_examples(example_records, examples_dir, n_examples_to_save)
    return _write_artifacts(out_dir, cfg, status, bundle.notes, certs, diagnostics, metrics, key_numbers, example_paths)


def families_present_from_diag(diagnostics: pd.DataFrame) -> set[str]:
    if diagnostics.empty or "proposal_family" not in diagnostics:
        return set()
    return set(str(v) for v in diagnostics["proposal_family"].unique())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/natural_text_open_proposal_cic.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
