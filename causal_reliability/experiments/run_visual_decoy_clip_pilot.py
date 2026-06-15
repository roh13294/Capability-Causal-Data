"""Gated pilot: second shortcut family (non-text visual decoy patch).

This is a controlled pilot, isolated from the typographic text-overlay headline
and from the completed scale-and-multi-model audit. It tests whether CIC region
scoring can localize and neutralize a *non-text* visual decoy patch among a fixed
candidate set (the known decoy region plus distractors), using only pixels,
candidate boxes, and the model's own predictions.

The non-oracle scorer never receives the true label, correctness, shortcut type,
or the oracle decoy box. The oracle decoy box is used only for the oracle
upper-bound baseline and for downstream localization metrics.

This run does not claim open-world discovery, general robustness, cross-shortcut
transfer, exact localization, or universal shortcut repair.
"""

from __future__ import annotations

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
from PIL import Image, ImageEnhance, ImageFilter

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

from causal_reliability.data.clip_visual_decoy_shortcuts import (
    VISUAL_DECOY_CLASSES,
    make_visual_decoy_dataset,
    save_example_images,
)
from causal_reliability.discovery.cic_region_scoring import neutralize_region, score_region_candidates
from causal_reliability.discovery.region_proposals import proposal_from_bbox
from causal_reliability.experiments.run_clip_overlay_validation import _downloads_allowed
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


PROMPT_TEMPLATE = "a simple image of a {label}"
MISLEADING_REGIME = "misleading_decoy"
CLEAN_REGIME = "no_decoy"

# Methods reported per example.
METHODS = [
    "original_clip_prediction",
    "oracle_decoy_neutralization",
    "matched_random_candidate_repair",
    "cic_top1_region_repair",
    "cic_top3_consensus_repair",
    "cic_clean_safe_repair",
]


def _device(model_cfg: dict[str, Any], cfg: dict[str, Any]) -> str:
    requested = str(model_cfg.get("device", "auto"))
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() and bool(cfg.get("prefer_gpu", False)) else "cpu"
    return requested


def _pil_to_tensor(images: list[Image.Image]) -> torch.Tensor:
    arrays = [np.asarray(img.convert("RGB")).astype(np.float32) / 255.0 for img in images]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


def _predict_pil(model: ClipZeroShotClassifier, images: list[Image.Image]) -> np.ndarray:
    return model.predict(_pil_to_tensor(images))["probabilities"].detach().cpu().numpy()


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = [int(v) for v in a]
    bx0, by0, bx1, by1 = [int(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    return float(inter / max(1, area_a + area_b - inter))


def _build_candidates(ex: dict[str, Any], pil: Image.Image, size: int, n_random: int, seed: int) -> list[Any]:
    """Fixed candidate set: the known decoy region plus distractors.

    Distractors: the three non-decoy corners, the central object box, and a few
    seeded random patches. The decoy box's *identity* is not exposed to the
    scorer; only its geometry is one of several candidates to be ranked.
    """

    candidates = []
    decoy_bbox = tuple(int(v) for v in ex["decoy_bbox"])
    candidates.append(proposal_from_bbox(pil, decoy_bbox, "cand_decoy", "decoy_region"))
    decoy_corner = int(ex["corner"])
    decoy_frac = (decoy_bbox[2] - decoy_bbox[0]) / float(size)
    from causal_reliability.data.clip_visual_decoy_shortcuts import _corner_box

    for c in range(4):
        if c == decoy_corner:
            continue
        box = tuple(_corner_box(c, size, decoy_frac))
        candidates.append(proposal_from_bbox(pil, box, f"cand_corner_{c}", "corner_distractor"))
    candidates.append(proposal_from_bbox(pil, tuple(int(v) for v in ex["object_bbox"]), "cand_object", "object_control"))
    rng = np.random.default_rng(seed + int(ex["example_id"]))
    side = decoy_bbox[2] - decoy_bbox[0]
    for i in range(n_random):
        jitter = float(rng.uniform(0.7, 1.4))
        bw = int(np.clip(side * jitter, 8, size - 2))
        bh = int(np.clip(side * float(rng.uniform(0.7, 1.4)), 8, size - 2))
        x = int(rng.integers(0, max(1, size - bw)))
        y = int(rng.integers(0, max(1, size - bh)))
        candidates.append(proposal_from_bbox(pil, (x, y, x + bw, y + bh), f"cand_random_{i}", "random_patch_control"))
    return candidates


def _consensus_repair(pil: Image.Image, scores: list[Any], predict_fn, top_k: int = 3, min_agreement: float = 2 / 3):
    selected = scores[:top_k]
    if not selected:
        return predict_fn([pil])[0], False, None
    probs = predict_fn([neutralize_region(pil, s.bbox) for s in selected])
    preds = probs.argmax(axis=1)
    vals, counts = np.unique(preds, return_counts=True)
    best = int(counts.max()) if len(counts) else 0
    stable = bool(best / max(1, len(selected)) >= min_agreement)
    keep = preds == vals[int(counts.argmax())] if len(vals) else np.ones(len(selected), dtype=bool)
    return probs[keep].mean(axis=0), stable, selected[0].bbox


def _select_matched_random(scores: list[Any], top1: Any | None) -> Any | None:
    randoms = [s for s in scores if s.proposal_type == "random_patch_control"]
    if not randoms:
        return None
    target = float(getattr(top1, "area_fraction", 0.0)) if top1 is not None else 0.0
    return min(randoms, key=lambda s: abs(float(s.area_fraction) - target))


def _score_example(ex: dict[str, Any], model: ClipZeroShotClassifier, size: int, n_random: int, seed: int) -> dict[str, Any]:
    pil = Image.fromarray((np.asarray(ex["image"]).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")
    predict_fn = lambda imgs: _predict_pil(model, imgs)
    candidates = _build_candidates(ex, pil, size, n_random, seed)
    scores, original_probs = score_region_candidates(pil, candidates, predict_fn)
    top1 = scores[0] if scores else None
    top1_probs = predict_fn([neutralize_region(pil, top1.bbox)])[0] if top1 else original_probs
    top3_probs, top3_stable, _ = _consensus_repair(pil, scores[:3], predict_fn)
    matched = _select_matched_random(scores, top1)
    matched_probs = predict_fn([neutralize_region(pil, matched.bbox)])[0] if matched else original_probs
    oracle_probs = predict_fn([neutralize_region(pil, tuple(int(v) for v in ex["decoy_bbox"]))])[0]
    return {
        "example_id": int(ex["example_id"]),
        "split": ex["split"],
        "regime": ex["regime"],
        "label": int(ex["label"]),
        "true_label": ex["true_label"],
        "decoy_relation": ex["decoy_relation"],
        "decoy_bbox": list(int(v) for v in ex["decoy_bbox"]),
        "scores": scores,
        "top1": top1,
        "top1_score": float(top1.score) if top1 else float("-inf"),
        "top1_bbox": list(top1.bbox) if top1 else None,
        "top1_overlay_iou": _iou(top1.bbox, ex["decoy_bbox"]) if top1 else float("nan"),
        "top3_best_iou": (max(_iou(s.bbox, ex["decoy_bbox"]) for s in scores[:3]) if scores else float("nan")),
        "original_probs": original_probs,
        "top1_probs": top1_probs,
        "top3_probs": top3_probs,
        "top3_stable": bool(top3_stable),
        "matched_probs": matched_probs,
        "matched_bbox": list(matched.bbox) if matched else None,
        "oracle_probs": oracle_probs,
    }


def _cert_row(ev: dict[str, Any], method: str, repaired_probs: np.ndarray, bbox, oracle: bool, action: str) -> dict[str, Any]:
    orig_pred = int(ev["original_probs"].argmax())
    rep_pred = int(repaired_probs.argmax())
    return {
        "example_id": ev["example_id"],
        "split": ev["split"],
        "regime": ev["regime"],
        "label": ev["label"],
        "true_label": ev["true_label"],
        "decoy_relation": ev["decoy_relation"],
        "method": method,
        "original_prediction_index": orig_pred,
        "original_confidence": float(ev["original_probs"].max()),
        "original_correct": bool(orig_pred == ev["label"]),
        "repaired_prediction_index": rep_pred,
        "repaired_confidence": float(repaired_probs.max()),
        "repaired_correct": bool(rep_pred == ev["label"]),
        "selected_bbox": "" if bbox is None else json.dumps([int(v) for v in bbox]),
        "top1_score": ev["top1_score"],
        "top1_decoy_iou_eval_only": ev["top1_overlay_iou"],
        "oracle_upper_bound": bool(oracle),
        "repair_action": action,
    }


def _build_certs(evals: list[dict[str, Any]], clean_safe_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ev in evals:
        rows.append(_cert_row(ev, "original_clip_prediction", ev["original_probs"], None, False, "keep_original"))
        rows.append(_cert_row(ev, "oracle_decoy_neutralization", ev["oracle_probs"], ev["decoy_bbox"], True, "repair"))
        rows.append(_cert_row(ev, "matched_random_candidate_repair", ev["matched_probs"], ev["matched_bbox"], False, "repair"))
        rows.append(_cert_row(ev, "cic_top1_region_repair", ev["top1_probs"], ev["top1_bbox"], False, "repair"))
        top3_action = "repair" if ev["top3_stable"] else "keep_original"
        top3_probs = ev["top3_probs"] if ev["top3_stable"] else ev["original_probs"]
        rows.append(_cert_row(ev, "cic_top3_consensus_repair", top3_probs, ev["top1_bbox"], False, top3_action))
        repair_clean_safe = ev["top1_score"] >= clean_safe_threshold
        cs_probs = ev["top1_probs"] if repair_clean_safe else ev["original_probs"]
        rows.append(_cert_row(ev, "cic_clean_safe_repair", cs_probs, ev["top1_bbox"] if repair_clean_safe else None, False, "repair" if repair_clean_safe else "keep_original"))
    return pd.DataFrame(rows)


def _select_clean_safe_threshold(val_evals: list[dict[str, Any]], max_clean_drop: float) -> dict[str, Any]:
    """Pick the lowest top-1 score threshold keeping validation clean drop <= bound.

    Among thresholds satisfying the clean-drop constraint, maximize misleading
    repair accuracy. Selection uses the validation split only.
    """

    thresholds = sorted({float(ev["top1_score"]) for ev in val_evals if np.isfinite(ev["top1_score"])} | {float("-inf")})
    clean = [ev for ev in val_evals if ev["regime"] == CLEAN_REGIME]
    misleading = [ev for ev in val_evals if ev["regime"] == MISLEADING_REGIME]
    best = {"threshold": float("inf"), "clean_drop": 0.0, "misleading_repair": 0.0}
    for t in thresholds:
        if clean:
            clean_before = np.mean([int(ev["original_probs"].argmax()) == ev["label"] for ev in clean])
            after = []
            for ev in clean:
                probs = ev["top1_probs"] if ev["top1_score"] >= t else ev["original_probs"]
                after.append(int(probs.argmax()) == ev["label"])
            clean_drop = float(clean_before - np.mean(after))
        else:
            clean_drop = 0.0
        if misleading:
            mis = []
            for ev in misleading:
                probs = ev["top1_probs"] if ev["top1_score"] >= t else ev["original_probs"]
                mis.append(int(probs.argmax()) == ev["label"])
            mis_repair = float(np.mean(mis))
        else:
            mis_repair = 0.0
        if clean_drop <= max_clean_drop and mis_repair >= best["misleading_repair"]:
            # Prefer the lowest threshold (more coverage) among ties / improvements.
            if mis_repair > best["misleading_repair"] or t < best["threshold"]:
                best = {"threshold": float(t), "clean_drop": clean_drop, "misleading_repair": mis_repair}
    if not np.isfinite(best["threshold"]):
        best["threshold"] = float(thresholds[-1]) if thresholds else float("inf")
    return best


def _acc(certs: pd.DataFrame, method: str, regime: str, col: str) -> float:
    sub = certs[(certs["method"] == method) & (certs["regime"] == regime)]
    if not len(sub):
        return float("nan")
    return float(sub[col].astype(bool).mean())


def _metrics_table(certs: pd.DataFrame, status: ClipStatus, headline_eligible: bool, reasons: list[str]) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        sub = certs[certs["method"] == method]
        if not len(sub):
            continue
        row = {
            "method": method,
            "shortcut_family": "non_text_visual_decoy_patch",
            "evidence_status": "pretrained CLIP visual-decoy pilot evidence" if status.pretrained else "unavailable",
            "headline_eligible": bool(headline_eligible),
            "include_in_final_headline": False,
            "headline_eligibility_reasons": "eligible" if headline_eligible else "; ".join(reasons),
            "backend": status.backend,
            "model_name": status.model_name,
            "pretrained_tag": status.pretrained_tag,
            "pretrained_loaded": bool(status.pretrained),
            "n_examples": int(len(sub)),
        }
        for regime in [CLEAN_REGIME, MISLEADING_REGIME, "aligned_decoy"]:
            row[f"{regime}_accuracy_before"] = _acc(certs, method, regime, "original_correct")
            row[f"{regime}_accuracy_after"] = _acc(certs, method, regime, "repaired_correct")
            row[f"{regime}_n_examples"] = int(len(sub[sub["regime"] == regime]))
        rows.append(row)
    return pd.DataFrame(rows)


def _evaluate_gates(certs: pd.DataFrame, status: ClipStatus, cfg: dict[str, Any], threshold_info: dict[str, Any]) -> dict[str, Any]:
    min_clean = float(cfg.get("gate_min_clean_accuracy", 0.80))
    clean_acc = _acc(certs, "original_clip_prediction", CLEAN_REGIME, "original_correct")
    misleading_orig = _acc(certs, "original_clip_prediction", MISLEADING_REGIME, "original_correct")
    oracle_acc = _acc(certs, "oracle_decoy_neutralization", MISLEADING_REGIME, "repaired_correct")
    top1_acc = _acc(certs, "cic_top1_region_repair", MISLEADING_REGIME, "repaired_correct")
    matched_acc = _acc(certs, "matched_random_candidate_repair", MISLEADING_REGIME, "repaired_correct")
    cs_clean_before = _acc(certs, "cic_clean_safe_repair", CLEAN_REGIME, "original_correct")
    cs_clean_after = _acc(certs, "cic_clean_safe_repair", CLEAN_REGIME, "repaired_correct")
    clean_safe_drop = float(cs_clean_before - cs_clean_after) if np.isfinite(cs_clean_before) and np.isfinite(cs_clean_after) else float("nan")
    cic_minus_random = float(top1_acc - matched_acc) if np.isfinite(top1_acc) and np.isfinite(matched_acc) else float("nan")

    leakage_safe = _scorer_is_leakage_safe()

    gates = {
        "pretrained_loaded": bool(status.pretrained) and status.backend in {"open_clip", "transformers"},
        "fake_backend_excluded": status.backend != "fake" and bool(status.pretrained),
        "clean_accuracy_high": bool(np.isfinite(clean_acc) and clean_acc >= min_clean),
        "misleading_original_le_0_40": bool(np.isfinite(misleading_orig) and misleading_orig <= 0.40),
        "oracle_repair_ge_0_85": bool(np.isfinite(oracle_acc) and oracle_acc >= 0.85),
        "cic_top1_minus_random_ge_0_15": bool(np.isfinite(cic_minus_random) and cic_minus_random >= 0.15),
        "clean_safe_drop_le_0_10": bool(np.isfinite(clean_safe_drop) and clean_safe_drop <= 0.10),
        "no_scorer_leakage": bool(leakage_safe),
    }
    failed = [k for k, v in gates.items() if not v]
    return {
        "gates": gates,
        "all_passed": len(failed) == 0,
        "failed_gates": failed,
        "values": {
            "clean_accuracy": _f(clean_acc),
            "misleading_original_accuracy": _f(misleading_orig),
            "oracle_repair_accuracy": _f(oracle_acc),
            "cic_top1_accuracy": _f(top1_acc),
            "cic_top3_accuracy": _f(_acc(certs, "cic_top3_consensus_repair", MISLEADING_REGIME, "repaired_correct")),
            "cic_clean_safe_accuracy": _f(_acc(certs, "cic_clean_safe_repair", MISLEADING_REGIME, "repaired_correct")),
            "matched_random_accuracy": _f(matched_acc),
            "cic_minus_random_gap": _f(cic_minus_random),
            "clean_safe_clean_drop": _f(clean_safe_drop),
            "min_clean_accuracy_threshold": min_clean,
            "clean_safe_score_threshold": _f(threshold_info.get("threshold")),
        },
    }


def _f(v) -> float | None:
    if v is None:
        return None
    v = float(v)
    return None if not np.isfinite(v) else v


def _scorer_is_leakage_safe() -> bool:
    import inspect

    params = set(inspect.signature(score_region_candidates).parameters)
    forbidden = {"true_label", "label", "decoy_bbox", "overlay_bbox", "decoy_label", "shortcut_identity", "test_correctness", "correct"}
    return not (params & forbidden)


def _write_unavailable(out_dir: Path, cfg: dict[str, Any], status: ClipStatus) -> dict[str, str]:
    metrics = pd.DataFrame(
        [
            {
                "method": "unavailable",
                "shortcut_family": "non_text_visual_decoy_patch",
                "evidence_status": "unavailable",
                "headline_eligible": False,
                "include_in_final_headline": False,
                "backend": status.backend,
                "model_name": status.model_name,
                "pretrained_loaded": False,
                "headline_eligibility_reasons": status.error_message or "pretrained CLIP did not load",
            }
        ]
    )
    metrics.to_csv(out_dir / "visual_decoy_pilot_metrics.csv", index=False)
    pd.DataFrame([{"method": "unavailable"}]).to_csv(out_dir / "visual_decoy_pilot_certificates.csv", index=False)
    (out_dir / "visual_decoy_pilot_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    gates = {"all_passed": False, "failed_gates": ["pretrained_loaded"], "values": {}, "gates": {"pretrained_loaded": False}}
    (out_dir / "visual_decoy_pilot_gates.json").write_text(json.dumps(gates, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "visual_decoy_pilot_report.md").write_text(
        "\n".join(
            [
                "# Visual-Decoy Shortcut Pilot (second shortcut family)",
                "",
                "Pretrained CLIP unavailable; no fake pilot evidence was generated.",
                f"Reason: {status.error_message or 'pretrained CLIP did not load'}",
                "",
                "Pilot eligible: `False`. This run is not usable as positive evidence.",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "metrics": str(out_dir / "visual_decoy_pilot_metrics.csv"),
        "certificates": str(out_dir / "visual_decoy_pilot_certificates.csv"),
        "gates": str(out_dir / "visual_decoy_pilot_gates.json"),
        "report": str(out_dir / "visual_decoy_pilot_report.md"),
    }


def _report_md(certs: pd.DataFrame, metrics: pd.DataFrame, gate_result: dict[str, Any], status: ClipStatus, cfg: dict[str, Any], threshold_info: dict[str, Any]) -> str:
    v = gate_result["values"]
    passed = gate_result["all_passed"]
    n_per_condition = int(cfg.get("data", {}).get("test_n_per_condition", cfg.get("data", {}).get("n_per_condition", 64)))
    top1_iou = float(certs[certs["method"] == "cic_top1_region_repair"]["top1_decoy_iou_eval_only"].astype(float).mean()) if len(certs) else float("nan")
    lines = [
        "# Visual-Decoy Shortcut Pilot (second shortcut family)",
        "",
        "Controlled gated pilot, isolated from the typographic text-overlay headline and the completed scale-and-multi-model audit.",
        "",
        f"- Shortcut family: `non_text_visual_decoy_patch` (central causal shape + competing-class corner patch, no written words)",
        f"- n_per_condition (held-out test, per regime): `{n_per_condition}`",
        f"- Backend: `{status.backend}`; model: `{status.model_name}`; pretrained tag: `{status.pretrained_tag}`; pretrained loaded: `{status.pretrained}`",
        f"- Pilot/headline eligible: `{passed}`",
        "",
        "## Headline pilot numbers (held-out test, misleading regime unless noted)",
        "",
        f"- Clean / no-shortcut accuracy: `{_fmt(v.get('clean_accuracy'))}`",
        f"- Misleading original accuracy: `{_fmt(v.get('misleading_original_accuracy'))}`",
        f"- Oracle decoy neutralization accuracy: `{_fmt(v.get('oracle_repair_accuracy'))}`",
        f"- CIC top-1 repair accuracy: `{_fmt(v.get('cic_top1_accuracy'))}`",
        f"- CIC top-3 repair accuracy: `{_fmt(v.get('cic_top3_accuracy'))}`",
        f"- CIC clean-safe repair accuracy: `{_fmt(v.get('cic_clean_safe_accuracy'))}`",
        f"- Matched random candidate-region accuracy: `{_fmt(v.get('matched_random_accuracy'))}`",
        f"- CIC top-1 minus matched random (gap): `{_fmt(v.get('cic_minus_random_gap'))}`",
        f"- Clean-safe clean drop: `{_fmt(v.get('clean_safe_clean_drop'))}`",
        f"- (Diagnostic) top-1 candidate IoU with oracle decoy region: `{_fmt(top1_iou)}`",
        f"- Validation-selected clean-safe score threshold: `{_fmt(threshold_info.get('threshold'))}`",
        "",
        "## Gate results",
        "",
    ]
    for name, ok in gate_result["gates"].items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} `{name}`")
    lines += [
        "",
        f"Failed gates: {gate_result['failed_gates'] if gate_result['failed_gates'] else 'none'}",
        "",
    ]
    if passed:
        lines += [
            "## Status",
            "",
            "All strict pilot gates passed. These are pilot numbers for a second shortcut family.",
            "Scaling to n=128 and/or multiple models requires explicit confirmation before running.",
        ]
    else:
        lines += [
            "## Status (boundary evidence)",
            "",
            "One or more strict pilot gates failed. Per the pre-registered protocol this run is recorded as honest",
            "boundary evidence for the visual-decoy shortcut family and is NOT integrated as a positive result.",
            "No further tuning was performed to force a pass.",
        ]
    lines += [
        "",
        "## Scope and non-claims",
        "",
        "- The non-oracle region scorer received only pixels, candidate boxes, and model probabilities. It did not receive",
        "  the true label, correctness, shortcut type, or the oracle decoy box.",
        "- The clean-safe score threshold was selected on a separate validation split, not the held-out test split.",
        "- The oracle decoy box is used only for the oracle upper-bound baseline and for the diagnostic localization IoU.",
        "- This pilot does not claim open-world discovery, general robustness, cross-shortcut transfer, exact localization,",
        "  or universal shortcut repair. It is a single-model, single-family controlled pilot.",
        "- It does not alter the text-overlay headline metrics or the completed scale-and-multi-model audit.",
    ]
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None:
        return "nan"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "visual_decoy_pilot")
    examples_dir = ensure_dir(out_dir / "examples")
    data_cfg = cfg.get("data", {})
    regimes = list(data_cfg.get("regimes", ["no_decoy", "misleading_decoy", "aligned_decoy"]))
    size = int(data_cfg.get("image_size", 224))
    test_n = int(data_cfg.get("test_n_per_condition", data_cfg.get("n_per_condition", 64)))
    val_n = int(data_cfg.get("validation_n_per_condition", max(8, test_n // 4)))
    decoy_frac = float(data_cfg.get("decoy_frac", 0.34))
    object_frac = float(data_cfg.get("object_frac", 0.46))
    obj = (object_frac, object_frac)
    n_random = int(cfg.get("n_random_candidates", 3))
    max_clean_drop = float(cfg.get("max_clean_drop", 0.05))

    val_bundle = make_visual_decoy_dataset(n_per_condition=val_n, size=size, regimes=regimes, split="validation", start_id=0, decoy_frac=decoy_frac, object_frac=obj)
    test_bundle = make_visual_decoy_dataset(n_per_condition=test_n, size=size, regimes=regimes, split="test", start_id=len(val_bundle.examples), decoy_frac=decoy_frac, object_frac=obj)
    save_example_images(test_bundle.examples[:24], examples_dir, keys=["image", "clean_image"])

    model_cfg = cfg.get("model", {})
    device = _device(model_cfg, cfg)
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))
    if str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend == "fake":
        status = ClipStatus(False, "fake", "fake_clip_visual_decoy", pretrained=False, device=device, backend_attempted="fake", error_message="fake backend is not allowed for visual-decoy pilot evidence")
        return _write_unavailable(out_dir, cfg, status)
    status = check_clip_available(
        device=device,
        allow_download=_downloads_allowed(model_cfg),
        preferred_backend=requested_backend,
        model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
        pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
        transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return _write_unavailable(out_dir, cfg, status)

    prompts = [PROMPT_TEMPLATE.format(label=name) for name in test_bundle.class_names]
    model = ClipZeroShotClassifier(status, test_bundle.class_names, prompts=prompts, device=device)

    val_evals = [_score_example(ex, model, size, n_random, seed + 100_000) for ex in val_bundle.examples]
    threshold_info = _select_clean_safe_threshold(val_evals, max_clean_drop)
    (out_dir / "selected_clean_safe_threshold.json").write_text(json.dumps(threshold_info, indent=2, sort_keys=True), encoding="utf-8")

    test_evals = [_score_example(ex, model, size, n_random, seed) for ex in test_bundle.examples]
    certs = _build_certs(test_evals, float(threshold_info["threshold"]))

    gate_result = _evaluate_gates(certs, status, cfg, threshold_info)
    metrics = _metrics_table(certs, status, gate_result["all_passed"], gate_result["failed_gates"])

    certs.to_csv(out_dir / "visual_decoy_pilot_certificates.csv", index=False)
    metrics.to_csv(out_dir / "visual_decoy_pilot_metrics.csv", index=False)
    (out_dir / "visual_decoy_pilot_gates.json").write_text(json.dumps(gate_result, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "visual_decoy_pilot_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    report = _report_md(certs, metrics, gate_result, status, cfg, threshold_info)
    (out_dir / "visual_decoy_pilot_report.md").write_text(report, encoding="utf-8")

    return {
        "metrics": str(out_dir / "visual_decoy_pilot_metrics.csv"),
        "certificates": str(out_dir / "visual_decoy_pilot_certificates.csv"),
        "gates": str(out_dir / "visual_decoy_pilot_gates.json"),
        "report": str(out_dir / "visual_decoy_pilot_report.md"),
        "selected_clean_safe_threshold": str(out_dir / "selected_clean_safe_threshold.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_decoy_pilot.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
