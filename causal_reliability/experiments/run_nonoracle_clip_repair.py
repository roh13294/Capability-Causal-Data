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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.data.clip_overlay_shortcuts import (
    CLIP_OVERLAY_CLASSES,
    make_clip_overlay_dataset,
    neutralize_overlay_array,
    save_example_images,
)
from causal_reliability.discovery.cic_region_scoring import neutralize_region
from causal_reliability.discovery.nonoracle_clip_discovery import discover_clip_shortcut_regions
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


def _predict_arrays(model: ClipZeroShotClassifier, arrays: list[np.ndarray]) -> np.ndarray:
    images = [Image.fromarray((np.asarray(arr).clip(0, 1) * 255).astype(np.uint8)) for arr in arrays]
    return _predict_pil(model, images)


def _iou(a: list[int] | tuple[int, int, int, int], b: list[int] | tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = [int(v) for v in a]
    bx0, by0, bx1, by1 = [int(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    return float(inter / max(1, area_a + area_b - inter))


def _augment_prediction(model: ClipZeroShotClassifier, ex: dict[str, Any], rng: np.random.Generator, n_views: int) -> np.ndarray:
    base = Image.fromarray((np.asarray(ex["image"]).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")
    w, h = base.size
    views = []
    for _ in range(n_views):
        img = ImageEnhance.Brightness(base).enhance(float(rng.uniform(0.82, 1.18)))
        img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.82, 1.18)))
        if rng.random() < 0.35:
            img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.2))))
        crop = int(rng.integers(0, max(1, w // 18)))
        if crop:
            img = img.crop((crop, crop, w - crop, h - crop)).resize((w, h), Image.Resampling.BICUBIC)
        views.append(img)
    return _predict_pil(model, views).mean(axis=0)


def _bbox_compatible(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return _iou(a, b) >= 0.12


def _consensus_repair(
    pil: Image.Image,
    scores: list[Any],
    predict_fn: Any,
    *,
    top_k: int = 3,
    min_agreement: float = 2 / 3,
) -> tuple[np.ndarray, bool, tuple[int, int, int, int] | None, str, str]:
    selected = scores[:top_k]
    if not selected:
        original = predict_fn([pil])[0]
        return original, False, None, "", ""
    probs = predict_fn([neutralize_region(pil, s.bbox) for s in selected])
    preds = probs.argmax(axis=1)
    vals, counts = np.unique(preds, return_counts=True)
    best_count = int(counts.max()) if len(counts) else 0
    prediction_agreement = best_count / max(1, len(selected))
    spatial_agreement = any(_bbox_compatible(selected[0].bbox, s.bbox) for s in selected[1:]) if len(selected) > 1 else True
    stable = bool(prediction_agreement >= min_agreement and (spatial_agreement or prediction_agreement >= 1.0))
    keep = preds == vals[int(counts.argmax())] if len(vals) else np.ones(len(selected), dtype=bool)
    return probs[keep].mean(axis=0), stable, selected[0].bbox, "top3_consensus", selected[0].proposal_type


def _policy_action(
    original_probs: np.ndarray,
    top1_probs: np.ndarray,
    top1: Any | None,
    consensus_probs: np.ndarray,
    consensus_stable: bool,
    policy: dict[str, Any],
    *,
    allow_abstain: bool,
) -> tuple[np.ndarray, bool, tuple[int, int, int, int] | None, str, str, str]:
    threshold = float(policy.get("score_threshold", np.inf))
    min_consensus = float(policy.get("min_consensus_stability", 2 / 3))
    if top1 is None or float(top1.score) < threshold:
        return original_probs, False, None, "none", "none", "keep_original"
    stable = bool(consensus_stable and float(top1.consensus_stability) >= min_consensus)
    if not stable:
        if allow_abstain:
            return original_probs, True, top1.bbox, top1.candidate_id, top1.proposal_type, "abstain"
        return original_probs, False, None, "none", "none", "keep_original"
    repaired = consensus_probs if str(policy.get("repair_source", "consensus")) == "consensus" else top1_probs
    return repaired, False, top1.bbox, top1.candidate_id, top1.proposal_type, "repair"


def _select_matched_random(scores: list[Any], top1: Any | None, field: str) -> Any | None:
    randoms = [s for s in scores if s.proposal_type == "random_patch_control"]
    if not randoms:
        return top1
    target = getattr(top1, field, 0.0) if top1 is not None else 0.0
    return min(randoms, key=lambda s: abs(float(getattr(s, field, 0.0)) - float(target)))


def _row(
    ex: dict[str, Any],
    method: str,
    class_names: list[str],
    original_probs: np.ndarray,
    repaired_probs: np.ndarray,
    selected_bbox: tuple[int, int, int, int] | None,
    selected_candidate_id: str,
    selected_proposal_type: str,
    abstained: bool = False,
    oracle: bool = False,
    repair_action: str | None = None,
) -> dict[str, Any]:
    orig_pred = int(original_probs.argmax())
    rep_pred = None if abstained else int(repaired_probs.argmax())
    return {
        "example_id": ex["example_id"],
        "split": ex["split"],
        "regime": ex["regime"],
        "true_label": ex["true_label"],
        "label": int(ex["label"]),
        "method": method,
        "original_prediction": class_names[orig_pred],
        "original_prediction_index": orig_pred,
        "original_confidence": float(original_probs.max()),
        "original_correct": bool(orig_pred == int(ex["label"])),
        "repaired_prediction": "" if rep_pred is None else class_names[rep_pred],
        "repaired_prediction_index": np.nan if rep_pred is None else rep_pred,
        "repaired_confidence": 0.0 if rep_pred is None else float(repaired_probs.max()),
        "repaired_correct": False if rep_pred is None else bool(rep_pred == int(ex["label"])),
        "selected_candidate_id": selected_candidate_id,
        "selected_proposal_type": selected_proposal_type,
        "selected_bbox": "" if selected_bbox is None else json.dumps([int(v) for v in selected_bbox]),
        "oracle_upper_bound": bool(oracle),
        "repair_action": repair_action or ("abstain" if abstained else ("keep_original" if method == "original_clip_prediction" else "repair")),
        "abstained": bool(abstained),
    }


def _metrics(certs: pd.DataFrame, rankings: pd.DataFrame, status: ClipStatus, headline: bool, reasons: list[str]) -> pd.DataFrame:
    rows = []
    for method, df in certs.groupby("method", sort=False):
        non_abs = ~df["abstained"].astype(bool)
        repaired = df["repaired_correct"].astype(bool)
        original = df["original_correct"].astype(bool)
        row: dict[str, Any] = {
            "method": method,
            "evidence_status": "pretrained CLIP non-oracle repair evidence" if status.pretrained else "unavailable",
            "headline_eligible": bool(headline),
            "include_in_final_headline": bool(headline),
            "headline_eligibility_reasons": "eligible" if headline else "; ".join(reasons),
            "backend": status.backend,
            "model_name": status.model_name,
            "pretrained_loaded": bool(status.pretrained),
            "n_examples": int(len(df)),
            "coverage": float(non_abs.mean()),
            "abstention_rate": float((~non_abs).mean()),
            "selective_accuracy": float(repaired[non_abs].mean()) if bool(non_abs.sum()) else np.nan,
            "failure_capture_rate": float(((~non_abs) & (~original)).sum() / max(1, int((~original).sum()))),
            "false_abstention_rate": float(((~non_abs) & original).sum() / max(1, int(original.sum()))),
            "accuracy_before": float(original.mean()),
            "accuracy_after_non_abstained": float(repaired[non_abs].mean()) if bool(non_abs.sum()) else np.nan,
            "automatic_repaired_accuracy": float(repaired[non_abs].mean()) if bool(non_abs.sum()) else np.nan,
            "no_op_unchanged_accuracy": float(df.loc[df["repair_action"] == "keep_original", "repaired_correct"].astype(bool).mean()) if "repair_action" in df and bool((df["repair_action"] == "keep_original").sum()) else np.nan,
            "false_repair_rate": float((df.loc[df["regime"].isin(["aligned_overlay", "no_overlay"]), "repair_action"] == "repair").mean()) if "repair_action" in df and bool(df["regime"].isin(["aligned_overlay", "no_overlay"]).sum()) else np.nan,
        }
        for regime in ["aligned_overlay", "misleading_overlay", "neutral_overlay", "no_overlay"]:
            sub = df[df["regime"] == regime]
            sub_non_abs = ~sub["abstained"].astype(bool) if len(sub) else pd.Series(dtype=bool)
            row[f"{regime}_n_examples"] = int(len(sub))
            row[f"{regime}_accuracy_before"] = float(sub["original_correct"].astype(bool).mean()) if len(sub) else np.nan
            row[f"{regime}_accuracy_after"] = float(sub.loc[sub_non_abs, "repaired_correct"].astype(bool).mean()) if len(sub) and bool(sub_non_abs.sum()) else np.nan
        clean_before = row["no_overlay_accuracy_before"]
        clean_after = row["no_overlay_accuracy_after"]
        row["clean_accuracy_drop"] = clean_before - clean_after if np.isfinite(clean_before) and np.isfinite(clean_after) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(rankings):
        misleading = rankings[rankings["regime"] == "misleading_overlay"]
        top1 = misleading[misleading["rank"] == 1]
        groups = [group for _, group in misleading.groupby("example_id")]
        loc = {
            "top1_overlay_iou": float(top1["overlay_iou"].mean()) if len(top1) else np.nan,
            "top3_best_overlay_iou": float(np.mean([g.nsmallest(3, "rank")["overlay_iou"].max() for g in groups])) if groups else np.nan,
            "top1_localization_success_iou_0_3": float((top1["overlay_iou"] >= 0.3).mean()) if len(top1) else np.nan,
            "top1_localization_success_iou_0_5": float((top1["overlay_iou"] >= 0.5).mean()) if len(top1) else np.nan,
            "top3_localization_success_iou_0_3": float(np.mean([(g.nsmallest(3, "rank")["overlay_iou"] >= 0.3).any() for g in groups])) if groups else np.nan,
            "top3_localization_success_iou_0_5": float(np.mean([(g.nsmallest(3, "rank")["overlay_iou"] >= 0.5).any() for g in groups])) if groups else np.nan,
            "median_overlay_rank": float(np.median([g.sort_values("overlay_iou", ascending=False)["rank"].iloc[0] for g in groups])) if groups else np.nan,
            "mean_true_overlay_candidate_rank": float(np.mean([g.sort_values("overlay_iou", ascending=False)["rank"].iloc[0] for g in groups if g["overlay_iou"].max() > 0])) if groups else np.nan,
            "fraction_top_candidate_random_control_object": float(top1["proposal_type"].isin(["random_patch_control", "center_object_control", "object_control"]).mean()) if len(top1) else np.nan,
        }
        for key, value in loc.items():
            out[key] = value
    return out


def _headline_eligibility(metrics: pd.DataFrame, rankings: pd.DataFrame, status: ClipStatus) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if status.backend not in {"open_clip", "transformers"} or not status.pretrained:
        reasons.append("pretrained CLIP backend did not load")
    if status.backend == "fake":
        reasons.append("fake backend cannot be headline eligible")
    if metrics.empty:
        reasons.append("no metrics")
        return False, reasons
    lookup = metrics.set_index("method").to_dict("index") if "method" in metrics else {}
    nonoracle = lookup.get("nonoracle_cic_top1_region_repair", {})
    top3 = lookup.get("nonoracle_cic_top3_consensus_repair", {})
    clean_safe = lookup.get("nonoracle_cic_clean_safe_repair", {})
    selective = lookup.get("nonoracle_cic_selective_repair_or_abstain", lookup.get("nonoracle_cic_repair_or_abstain", {}))
    original = lookup.get("original_clip_prediction", {})
    misleading_n = int(original.get("misleading_overlay_n_examples", 0))
    if misleading_n < 30:
        reasons.append("misleading_overlay n_examples < 30")
    top1_loc = float(nonoracle.get("top1_localization_success_iou_0_3", np.nan))
    top3_loc = float(nonoracle.get("top3_localization_success_iou_0_3", np.nan))
    random_control = float(nonoracle.get("fraction_top_candidate_random_control_object", np.nan))
    if not ((np.isfinite(top1_loc) and top1_loc > 0.20) or (np.isfinite(top3_loc) and top3_loc > 0.35)):
        reasons.append("localization success is not clearly above random/control")
    if np.isfinite(random_control) and random_control >= 0.50:
        reasons.append("top candidate too often random/control/object proposal")
    top1_acc = float(nonoracle.get("misleading_overlay_accuracy_after", np.nan))
    top3_acc = float(top3.get("misleading_overlay_accuracy_after", np.nan))
    selective_acc = float(selective.get("selective_accuracy", np.nan))
    failure_capture = float(selective.get("failure_capture_rate", np.nan))
    clean_safe_acc = float(clean_safe.get("misleading_overlay_accuracy_after", np.nan))
    if not ((np.isfinite(top1_acc) and top1_acc >= 0.80) or (np.isfinite(top3_acc) and top3_acc >= 0.80) or (np.isfinite(clean_safe_acc) and clean_safe_acc >= 0.80) or (np.isfinite(selective_acc) and selective_acc >= 0.90 and np.isfinite(failure_capture) and failure_capture >= 0.60)):
        reasons.append("non-oracle repair/selective-abstention criterion not met")
    clean_drop = min(
        [float(v) for v in [nonoracle.get("clean_accuracy_drop", np.nan), top3.get("clean_accuracy_drop", np.nan), clean_safe.get("clean_accuracy_drop", np.nan), selective.get("clean_accuracy_drop", np.nan)] if np.isfinite(float(v))]
        or [np.nan]
    )
    if not np.isfinite(clean_drop) or clean_drop > 0.10:
        reasons.append("aligned/no-overlay clean accuracy drop > 0.10 or unavailable")
    return len(reasons) == 0, reasons


def _write_unavailable(out_dir: Path, cfg: dict[str, Any], status: ClipStatus) -> dict[str, str]:
    rankings = pd.DataFrame()
    certs = pd.DataFrame(
        [
            {
                "method": "unavailable",
                "oracle_upper_bound": False,
                "pretrained_loaded": False,
                "headline_eligible": False,
            }
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "method": "unavailable",
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
    rankings.to_csv(out_dir / "nonoracle_clip_candidate_rankings.csv", index=False)
    certs.to_csv(out_dir / "nonoracle_clip_repair_certificates.csv", index=False)
    metrics.to_csv(out_dir / "nonoracle_clip_repair_metrics.csv", index=False)
    (out_dir / "nonoracle_clip_repair_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    (out_dir / "nonoracle_clip_repair_summary.md").write_text(
        "\n".join(
            [
                "# Non-Oracle CLIP Shortcut Localization and Repair",
                "",
                "Pretrained CLIP unavailable; no fake headline evidence was generated.",
                "Headline eligible: `False`.",
                f"Reason: {status.error_message or 'pretrained CLIP did not load'}",
                "",
                "Oracle overlay repair should not be treated as evidence of automatic shortcut discovery.",
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "nonoracle_clip_repair_examples.md").write_text("# Non-Oracle CLIP Repair Examples\n\nUnavailable.\n", encoding="utf-8")
    (out_dir / "nonoracle_clip_repair_caption.md").write_text("# Caption\n\nUnavailable.\n", encoding="utf-8")
    _plot(metrics, out_dir / "nonoracle_clip_repair_plot.png", out_dir / "nonoracle_clip_repair_plot.pdf")
    return {"metrics": str(out_dir / "nonoracle_clip_repair_metrics.csv"), "certificates": str(out_dir / "nonoracle_clip_repair_certificates.csv"), "rankings": str(out_dir / "nonoracle_clip_candidate_rankings.csv"), "summary": str(out_dir / "nonoracle_clip_repair_summary.md")}


def _plot(metrics: pd.DataFrame, png: Path, pdf: Path) -> None:
    take = metrics[metrics.get("method", pd.Series(dtype=str)).isin(["original_clip_prediction", "random_patch_same_area", "random_patch_same_textness", "oracle_overlay_neutralization", "nonoracle_cic_top1_region_repair", "nonoracle_cic_top3_consensus_repair", "nonoracle_cic_clean_safe_repair"])] if len(metrics) else pd.DataFrame()
    plt.figure(figsize=(9.2, 4.8))
    if len(take) and "misleading_overlay_accuracy_after" in take:
        x = np.arange(len(take))
        plt.bar(x, take["misleading_overlay_accuracy_after"], color="#4c78a8")
        plt.xticks(x, take["method"], rotation=25, ha="right")
        plt.ylim(0, 1.02)
        plt.ylabel("Misleading-overlay accuracy after")
    else:
        plt.text(0.5, 0.5, "No eligible non-oracle CLIP repair metrics", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(png, dpi=170)
    plt.savefig(pdf)
    plt.close()


def _evaluate_examples(
    *,
    examples: list[dict[str, Any]],
    class_names: list[str],
    prompts: list[str],
    model: ClipZeroShotClassifier,
    seed: int,
    size: int,
    max_candidates: int,
    n_views: int,
    rng: np.random.Generator,
    policy: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_rows: list[dict[str, Any]] = []
    cert_rows: list[dict[str, Any]] = []
    default_policy = {"score_threshold": float("inf"), "min_consensus_stability": 2 / 3, "repair_source": "consensus"}

    for ex in examples:
        pil = Image.fromarray((np.asarray(ex["image"]).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")
        predict_fn = lambda imgs: _predict_pil(model, imgs)
        proposals, scores, original_probs = discover_clip_shortcut_regions(pil, predict_fn, prompts, seed=seed + int(ex["example_id"]), max_candidates=max_candidates)
        for rank, score in enumerate(scores, start=1):
            row = score.to_dict()
            row.update(
                {
                    "example_id": ex["example_id"],
                    "split": ex["split"],
                    "regime": ex["regime"],
                    "rank": rank,
                    "overlay_iou": _iou(score.bbox, ex["overlay_bbox"]),
                    "overlay_bbox_eval_only": json.dumps([int(v) for v in ex["overlay_bbox"]]),
                }
            )
            ranking_rows.append(row)

        top1 = scores[0] if scores else None
        top3 = scores[:3]
        top1_probs = predict_fn([neutralize_region(pil, top1.bbox)])[0] if top1 else original_probs
        top3_probs, top3_stable, top3_bbox, top3_id, top3_type = _consensus_repair(pil, top3, predict_fn)
        random_area_score = _select_matched_random(scores, top1, "area_fraction")
        random_text_score = _select_matched_random(scores, top1, "textness_score")
        random_area_probs = predict_fn([neutralize_region(pil, random_area_score.bbox)])[0] if random_area_score else original_probs
        random_text_probs = predict_fn([neutralize_region(pil, random_text_score.bbox)])[0] if random_text_score else original_probs
        random_topk = [s for s in scores if s.proposal_type == "random_patch_control"][:3]
        random_topk_probs = predict_fn([neutralize_region(pil, s.bbox) for s in random_topk]).mean(axis=0) if random_topk else original_probs
        oracle_arr = neutralize_overlay_array(ex["image"], ex["overlay_bbox"], "mask_overlay_bbox_background", size)
        oracle_probs = _predict_arrays(model, [oracle_arr])[0]
        aug_probs = _augment_prediction(model, ex, rng, n_views)

        selected_policy = policy or default_policy
        clean_probs, clean_abs, clean_bbox, clean_id, clean_type, clean_action = _policy_action(original_probs, top1_probs, top1, top3_probs, top3_stable, selected_policy, allow_abstain=False)
        sel_probs, sel_abs, sel_bbox, sel_id, sel_type, sel_action = _policy_action(original_probs, top1_probs, top1, top3_probs, top3_stable, selected_policy, allow_abstain=True)

        cert_rows.extend(
            [
                _row(ex, "original_clip_prediction", class_names, original_probs, original_probs, None, "none", "none", repair_action="keep_original"),
                _row(ex, "random_augmentation_consensus", class_names, original_probs, aug_probs, None, "random_augmentation", "random_augmentation"),
                _row(ex, "random_patch_same_area", class_names, original_probs, random_area_probs, random_area_score.bbox if random_area_score else None, random_area_score.candidate_id if random_area_score else "", "random_patch_control"),
                _row(ex, "random_patch_same_textness", class_names, original_probs, random_text_probs, random_text_score.bbox if random_text_score else None, random_text_score.candidate_id if random_text_score else "", "random_patch_control"),
                _row(ex, "random_topk_patch_consensus", class_names, original_probs, random_topk_probs, random_topk[0].bbox if random_topk else None, "random_topk_consensus", "random_patch_control"),
                _row(ex, "random_patch_neutralization", class_names, original_probs, random_area_probs, random_area_score.bbox if random_area_score else None, random_area_score.candidate_id if random_area_score else "", "random_patch_control"),
                _row(ex, "oracle_overlay_neutralization", class_names, original_probs, oracle_probs, tuple(ex["overlay_bbox"]), "oracle_overlay_bbox", "oracle upper bound", oracle=True),
                _row(ex, "nonoracle_cic_top1_region_repair", class_names, original_probs, top1_probs, top1.bbox if top1 else None, top1.candidate_id if top1 else "", top1.proposal_type if top1 else ""),
                _row(ex, "nonoracle_cic_top3_consensus_repair", class_names, original_probs, top3_probs, top3_bbox, top3_id, top3_type, abstained=not top3_stable, repair_action="repair" if top3_stable else "abstain"),
                _row(ex, "nonoracle_cic_clean_safe_repair", class_names, original_probs, clean_probs, clean_bbox, clean_id, clean_type, abstained=clean_abs, repair_action=clean_action),
                _row(ex, "nonoracle_cic_selective_repair_or_abstain", class_names, original_probs, sel_probs, sel_bbox, sel_id, sel_type, abstained=sel_abs, repair_action=sel_action),
                _row(ex, "nonoracle_cic_repair_or_abstain", class_names, original_probs, sel_probs, sel_bbox, sel_id, sel_type, abstained=sel_abs, repair_action=sel_action),
            ]
        )
        _ = proposals
    return pd.DataFrame(cert_rows), pd.DataFrame(ranking_rows)


def _policy_metrics_for_threshold(certs: pd.DataFrame, rankings: pd.DataFrame, threshold: float, min_consensus: float, *, allow_abstain: bool) -> dict[str, Any]:
    top1 = rankings[rankings["rank"] == 1].set_index("example_id")
    rows = []
    for _, base in certs[certs["method"] == "nonoracle_cic_top3_consensus_repair"].iterrows():
        rid = base["example_id"]
        orig = certs[(certs["example_id"] == rid) & (certs["method"] == "original_clip_prediction")].iloc[0]
        score_row = top1.loc[rid] if rid in top1.index else None
        score = float(score_row["score"]) if score_row is not None else -np.inf
        stable = bool(not base["abstained"]) and (score_row is not None and float(score_row["consensus_stability"]) >= min_consensus)
        action = "keep_original"
        repaired_correct = bool(orig["original_correct"])
        abstained = False
        if score >= threshold:
            if stable:
                action = "repair"
                repaired_correct = bool(base["repaired_correct"])
            elif allow_abstain:
                action = "abstain"
                repaired_correct = False
                abstained = True
        rows.append({**orig.to_dict(), "method": "policy_eval", "repaired_correct": repaired_correct, "repair_action": action, "abstained": abstained})
    df = pd.DataFrame(rows)
    non_abs = ~df["abstained"].astype(bool)
    misleading = df["regime"] == "misleading_overlay"
    clean = df["regime"] == "no_overlay"
    control = df["regime"].isin(["aligned_overlay", "no_overlay"])
    clean_before = float(df.loc[clean, "original_correct"].astype(bool).mean()) if bool(clean.sum()) else np.nan
    clean_after = float(df.loc[clean & non_abs, "repaired_correct"].astype(bool).mean()) if bool((clean & non_abs).sum()) else np.nan
    return {
        "score_threshold": float(threshold),
        "min_consensus_stability": float(min_consensus),
        "coverage": float(non_abs.mean()) if len(df) else np.nan,
        "repair_rate": float((df["repair_action"] == "repair").mean()) if len(df) else np.nan,
        "misleading_repair_accuracy": float(df.loc[misleading & non_abs, "repaired_correct"].astype(bool).mean()) if bool((misleading & non_abs).sum()) else np.nan,
        "clean_accuracy_before": clean_before,
        "clean_accuracy_after": clean_after,
        "clean_accuracy_drop": clean_before - clean_after if np.isfinite(clean_before) and np.isfinite(clean_after) else np.nan,
        "false_repair_rate": float((df.loc[control, "repair_action"] == "repair").mean()) if bool(control.sum()) else np.nan,
        "abstention_rate": float((~non_abs).mean()) if len(df) else np.nan,
    }


def _select_policy(validation_certs: pd.DataFrame, validation_rankings: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    policy_cfg = cfg.get("policy", {})
    max_clean_drop = float(policy_cfg.get("max_clean_drop", cfg.get("max_clean_drop", 0.05)))
    min_coverage = float(policy_cfg.get("min_coverage", 0.50))
    thresholds = sorted(set([0.0, *validation_rankings.loc[validation_rankings["rank"] == 1, "score"].astype(float).tolist()]))
    consensus_grid = [float(v) for v in policy_cfg.get("min_consensus_grid", [2 / 3, 1.0])]
    rows = []
    for min_consensus in consensus_grid:
        for threshold in thresholds:
            for objective in ["clean_safe_repair", "balanced_repair"]:
                row = _policy_metrics_for_threshold(validation_certs, validation_rankings, threshold, min_consensus, allow_abstain=(objective == "balanced_repair"))
                row["objective"] = objective
                row["balanced_score"] = row["misleading_repair_accuracy"] - 2.0 * row["clean_accuracy_drop"] - 0.5 * row["false_repair_rate"]
                rows.append(row)
    sweep = pd.DataFrame(rows)
    safe = sweep[(sweep["objective"] == "clean_safe_repair") & (sweep["clean_accuracy_drop"] <= max_clean_drop) & (sweep["coverage"] >= min_coverage)]
    relaxed = False
    if safe.empty:
        safe = sweep[(sweep["objective"] == "clean_safe_repair") & (sweep["clean_accuracy_drop"] <= 0.10) & (sweep["coverage"] >= min_coverage)]
        relaxed = True
    if safe.empty:
        safe = sweep[sweep["objective"] == "balanced_repair"]
        chosen_objective = "balanced_repair"
    else:
        chosen_objective = "clean_safe_repair"
    chosen = safe.sort_values(["misleading_repair_accuracy", "clean_accuracy_drop", "false_repair_rate", "coverage"], ascending=[False, True, True, False]).iloc[0]
    if chosen_objective == "balanced_repair":
        chosen = safe.sort_values(["balanced_score", "misleading_repair_accuracy", "coverage"], ascending=[False, False, False]).iloc[0]
    policy = {
        "objective": chosen_objective,
        "score_threshold": float(chosen["score_threshold"]),
        "min_consensus_stability": float(chosen["min_consensus_stability"]),
        "repair_source": "consensus",
        "max_clean_drop": max_clean_drop,
        "relaxed_clean_drop_constraint": bool(relaxed and chosen_objective == "clean_safe_repair"),
        "validation_metrics": {k: (None if pd.isna(v) else float(v)) for k, v in chosen.items() if isinstance(v, (int, float, np.floating))},
    }
    return policy, sweep


def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "nonoracle_clip_repair")
    examples_dir = ensure_dir(out_dir / "examples")
    data_cfg = cfg.get("data", {})
    regimes = list(data_cfg.get("regimes", ["aligned_overlay", "misleading_overlay", "neutral_overlay", "no_overlay"]))
    size = int(data_cfg.get("image_size", 224))
    test_n = int(data_cfg.get("test_n_per_class", data_cfg.get("n_per_class", 8)))
    val_n = int(data_cfg.get("validation_n_per_class", data_cfg.get("val_n_per_class", max(2, test_n))))
    val_bundle = make_clip_overlay_dataset(n_per_class=val_n, size=size, regimes=regimes, split="validation", start_id=0)
    bundle = make_clip_overlay_dataset(n_per_class=test_n, size=size, regimes=regimes, split="test", start_id=len(val_bundle.examples))
    save_example_images(bundle.examples, examples_dir, keys=["image"])

    model_cfg = cfg.get("model", {})
    device = _device(model_cfg, cfg)
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))
    if str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend == "fake":
        status = ClipStatus(False, "fake", "fake_clip_overlay", pretrained=False, device=device, backend_attempted="fake", error_message="fake backend is not allowed for CLIP repair evidence")
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

    prompts = [PROMPT_TEMPLATE.format(label=name) for name in bundle.class_names]
    model = ClipZeroShotClassifier(status, bundle.class_names, prompts=prompts, device=device)
    max_candidates = int(cfg.get("max_candidates", 96))
    n_views = int(cfg.get("augmentation_views", 5))

    validation_certs, validation_rankings = _evaluate_examples(
        examples=val_bundle.examples,
        class_names=val_bundle.class_names,
        prompts=prompts,
        model=model,
        seed=seed + 100_000,
        size=size,
        max_candidates=max_candidates,
        n_views=n_views,
        rng=rng,
        policy=None,
    )
    selected_policy, policy_sweep = _select_policy(validation_certs, validation_rankings, cfg)
    (out_dir / "selected_nonoracle_repair_policy.json").write_text(json.dumps(selected_policy, indent=2, sort_keys=True), encoding="utf-8")
    policy_sweep.to_csv(out_dir / "validation_policy_sweep.csv", index=False)
    validation_rankings.to_csv(out_dir / "validation_nonoracle_clip_candidate_rankings.csv", index=False)
    validation_certs.to_csv(out_dir / "validation_nonoracle_clip_repair_certificates.csv", index=False)

    certs, rankings = _evaluate_examples(
        examples=bundle.examples,
        class_names=bundle.class_names,
        prompts=prompts,
        model=model,
        seed=seed,
        size=size,
        max_candidates=max_candidates,
        n_views=n_views,
        rng=rng,
        policy=selected_policy,
    )
    preliminary = _metrics(certs, rankings, status, False, [])
    eligible, reasons = _headline_eligibility(preliminary, rankings, status)
    metrics = _metrics(certs, rankings, status, eligible, reasons)
    rankings.to_csv(out_dir / "nonoracle_clip_candidate_rankings.csv", index=False)
    certs.to_csv(out_dir / "nonoracle_clip_repair_certificates.csv", index=False)
    metrics.to_csv(out_dir / "nonoracle_clip_repair_metrics.csv", index=False)
    (out_dir / "nonoracle_clip_repair_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    _plot(metrics, out_dir / "nonoracle_clip_repair_plot.png", out_dir / "nonoracle_clip_repair_plot.pdf")

    lookup = metrics.set_index("method").to_dict("index")
    top1_row = lookup.get("nonoracle_cic_top1_region_repair", {})
    top3_row = lookup.get("nonoracle_cic_top3_consensus_repair", {})
    oracle_row = lookup.get("oracle_overlay_neutralization", {})
    original_row = lookup.get("original_clip_prediction", {})
    random_row = lookup.get("random_patch_same_area", lookup.get("random_patch_neutralization", {}))
    random_text_row = lookup.get("random_patch_same_textness", {})
    clean_safe_row = lookup.get("nonoracle_cic_clean_safe_repair", {})
    summary = [
        "# Non-Oracle CLIP Shortcut Localization and Repair",
        "",
        "Oracle overlay repair should not be treated as evidence of automatic shortcut discovery. It shows that removing the known shortcut restores performance. The non-oracle experiment tests whether CIC can discover a repair region from candidate interventions.",
        "",
        f"Evidence status: pretrained CLIP non-oracle repair evidence. Backend: {status.backend}. Model: {status.model_name}. Pretrained loaded: `{status.pretrained}`.",
        f"Headline eligible: `{eligible}`.",
        "Eligible." if eligible else f"Not eligible: {'; '.join(reasons)}.",
        f"Validation-selected policy: objective `{selected_policy.get('objective')}`, threshold {selected_policy.get('score_threshold'):.6f}, min consensus {selected_policy.get('min_consensus_stability'):.3f}, relaxed clean constraint `{selected_policy.get('relaxed_clean_drop_constraint')}`.",
        "",
        "## Results",
        "",
        f"- Original misleading accuracy: {original_row.get('misleading_overlay_accuracy_before', np.nan):.3f}",
        f"- Oracle upper-bound misleading accuracy: {oracle_row.get('misleading_overlay_accuracy_after', np.nan):.3f}",
        f"- Non-oracle top-1 repair misleading accuracy: {top1_row.get('misleading_overlay_accuracy_after', np.nan):.3f}",
        f"- Non-oracle top-3 repair misleading accuracy: {top3_row.get('misleading_overlay_accuracy_after', np.nan):.3f}",
        f"- Validation-gated clean-safe repair misleading accuracy: {clean_safe_row.get('misleading_overlay_accuracy_after', np.nan):.3f}",
        f"- Random same-area / same-textness misleading accuracy: {random_row.get('misleading_overlay_accuracy_after', np.nan):.3f} / {random_text_row.get('misleading_overlay_accuracy_after', np.nan):.3f}",
        f"- Top-1/top-3 localization success at IoU >= 0.3: {top1_row.get('top1_localization_success_iou_0_3', np.nan):.3f} / {top1_row.get('top3_localization_success_iou_0_3', np.nan):.3f}",
        f"- Clean accuracy drop, clean-safe policy: {clean_safe_row.get('clean_accuracy_drop', np.nan):.3f}",
        "",
        "## Scope",
        "",
        "- Discovery scoring did not receive true labels, overlay text, overlay relation, test correctness, or overlay bbox.",
        "- Repair thresholds were selected on the validation split only; the held-out test used the saved policy without retuning.",
        "- Overlay bbox is used only after ranking for localization metrics and the oracle upper-bound baseline.",
        "- This searches a finite candidate region class. It does not solve open-world causal discovery or general robustness.",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
    ]
    (out_dir / "nonoracle_clip_repair_summary.md").write_text("\n".join(summary), encoding="utf-8")
    (out_dir / "nonoracle_clip_repair_examples.md").write_text("# Non-Oracle CLIP Repair Examples\n\n" + _markdown_table(certs.head(12)), encoding="utf-8")
    (out_dir / "nonoracle_clip_repair_caption.md").write_text("# Non-Oracle CLIP Repair Figure Caption\n\nMisleading-overlay repair accuracy for original CLIP, random controls, oracle upper bound, and CIC-discovered region neutralization.\n", encoding="utf-8")
    return {
        "metrics": str(out_dir / "nonoracle_clip_repair_metrics.csv"),
        "certificates": str(out_dir / "nonoracle_clip_repair_certificates.csv"),
        "rankings": str(out_dir / "nonoracle_clip_candidate_rankings.csv"),
        "summary": str(out_dir / "nonoracle_clip_repair_summary.md"),
        "selected_policy": str(out_dir / "selected_nonoracle_repair_policy.json"),
        "validation_policy_sweep": str(out_dir / "validation_policy_sweep.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nonoracle_clip_repair.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
