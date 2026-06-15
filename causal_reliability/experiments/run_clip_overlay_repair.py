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
    examples_to_tensor,
    make_clip_overlay_dataset,
    save_default_example_grids,
    save_example_images,
)
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


PROMPT_TEMPLATES = [
    "a photo of a {label}",
    "a simple image of a {label}",
    "a black {label} shape",
    "a diagram of a {label}",
    "a centered {label}",
    "a geometric {label}",
]

NEUTRALIZATION_STRATEGIES = [
    "mask_overlay_bbox_background",
    "replace_overlay_bbox_background",
    "blur_overlay_bbox",
    "crop_out_overlay",
    "neutral_word_shape",
    "no_overlay_raw",
]

CERT_COLUMNS = [
    "example_id",
    "split",
    "regime",
    "true_label",
    "label",
    "overlay_text",
    "overlay_relation",
    "overlay_bbox",
    "image_path",
    "method",
    "original_prediction",
    "original_confidence",
    "original_correct",
    "cic_score",
    "stability_score",
    "quadrant",
    "selected_intervention",
    "repaired_prediction",
    "repaired_confidence",
    "repaired_correct",
    "repair_action",
    "abstained",
    "model_backend",
    "model_name",
    "pretrained_loaded",
]


def _device(model_cfg: dict[str, Any], cfg: dict[str, Any]) -> str:
    requested = str(model_cfg.get("device", "auto"))
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() and bool(cfg.get("prefer_gpu", False)) else "cpu"
    return requested


def _load_image_tensor(paths: list[str]) -> torch.Tensor:
    arrays = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        arrays.append(np.asarray(img).astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


def _predict_probs(model: ClipZeroShotClassifier, examples: list[dict[str, Any]], key: str) -> np.ndarray:
    path_key = f"{key}_path"
    paths = [str(ex[path_key]) for ex in examples]
    pred = model.predict(_load_image_tensor(paths))
    return pred["probabilities"].detach().cpu().numpy()


def _predict_tensor_probs(model: ClipZeroShotClassifier, images: torch.Tensor) -> np.ndarray:
    pred = model.predict(images)
    return pred["probabilities"].detach().cpu().numpy()


def _class_prompts(template: str, class_names: list[str]) -> list[str]:
    return [template.format(label=name) for name in class_names]


def _strategy_key(strategy: str) -> str:
    return {
        "mask_overlay_bbox_background": "mask_overlay_bbox_background_image",
        "replace_overlay_bbox_background": "replace_overlay_bbox_background_image",
        "blur_overlay_bbox": "blur_overlay_bbox_image",
        "crop_out_overlay": "crop_out_overlay_image",
        "neutral_word_shape": "neutral_word_overlay_image",
        "no_overlay_raw": "overlay_removed_image",
    }[strategy]


def _acc(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    if not bool(mask.sum()):
        return float("nan")
    return float((probs[mask].argmax(axis=1) == labels[mask]).mean())


def _validation_sweep(
    status: ClipStatus,
    examples: list[dict[str, Any]],
    class_names: list[str],
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    labels = np.array([ex["label"] for ex in examples], dtype=int)
    aligned = np.array([ex["regime"] == "aligned_overlay" for ex in examples], dtype=bool)
    misleading = np.array([ex["regime"] == "misleading_overlay" for ex in examples], dtype=bool)
    no_overlay = np.array([ex["regime"] == "no_overlay" for ex in examples], dtype=bool)
    rows = []
    for template in PROMPT_TEMPLATES:
        model = ClipZeroShotClassifier(status, class_names, prompts=_class_prompts(template, class_names), device=device)
        original = _predict_probs(model, examples, "image")
        no_overlay_probs = _predict_probs(model, examples, "overlay_removed_image")
        aligned_before = _acc(original, labels, aligned)
        no_overlay_acc = _acc(no_overlay_probs, labels, no_overlay)
        for strategy in NEUTRALIZATION_STRATEGIES:
            neutralized = _predict_probs(model, examples, _strategy_key(strategy))
            misleading_neut = _acc(neutralized, labels, misleading)
            aligned_after = _acc(neutralized, labels, aligned)
            aligned_drop = aligned_before - aligned_after if np.isfinite(aligned_before) and np.isfinite(aligned_after) else np.nan
            score = np.nan_to_num(no_overlay_acc, nan=0.0) + np.nan_to_num(misleading_neut, nan=0.0) - max(0.0, np.nan_to_num(aligned_drop, nan=1.0))
            rows.append(
                {
                    "split": "validation",
                    "prompt_template": template,
                    "neutralization_strategy": strategy,
                    "no_overlay_accuracy": no_overlay_acc,
                    "neutralized_misleading_accuracy": misleading_neut,
                    "aligned_accuracy_before": aligned_before,
                    "aligned_accuracy_after": aligned_after,
                    "aligned_accuracy_drop": aligned_drop,
                    "selection_score": score,
                }
            )
    sweep = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    sweep.to_csv(out_dir / "clip_repair_validation_sweep.csv", index=False)
    best = sweep.iloc[0].to_dict()
    selected = {
        "source_split": "validation",
        "prompt_template": best["prompt_template"],
        "neutralization_strategy": best["neutralization_strategy"],
        "selection_score": float(best["selection_score"]),
        "no_overlay_accuracy": float(best["no_overlay_accuracy"]),
        "neutralized_misleading_accuracy": float(best["neutralized_misleading_accuracy"]),
        "aligned_accuracy_drop": float(best["aligned_accuracy_drop"]),
    }
    (out_dir / "selected_clip_repair_config.json").write_text(json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")
    return selected


def _augment_tensor(examples: list[dict[str, Any]], rng: np.random.Generator, n_views: int) -> torch.Tensor:
    images = []
    for ex in examples:
        base = Image.open(ex["image_path"]).convert("RGB")
        w, h = base.size
        for _ in range(n_views):
            img = ImageEnhance.Brightness(base).enhance(float(rng.uniform(0.82, 1.18)))
            img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.82, 1.18)))
            if rng.random() < 0.35:
                img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.2))))
            crop = int(rng.integers(0, max(1, w // 18)))
            if crop:
                img = img.crop((crop, crop, w - crop, h - crop)).resize((w, h), Image.Resampling.BICUBIC)
            arr = np.asarray(img).astype(np.float32) / 255.0
            noise = rng.normal(0, 0.015, size=arr.shape).astype(np.float32)
            images.append(np.clip(arr + noise, 0, 1))
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()


def _generic_occlusion_tensor(examples: list[dict[str, Any]], rng: np.random.Generator, n_views: int) -> torch.Tensor:
    images = []
    for ex in examples:
        base = np.asarray(Image.open(ex["image_path"]).convert("RGB")).astype(np.float32) / 255.0
        h, w = base.shape[:2]
        box = max(16, w // 6)
        ox0, oy0, ox1, oy1 = ex["overlay_bbox"]
        for _ in range(n_views):
            arr = base.copy()
            for _attempt in range(20):
                x = int(rng.integers(0, max(1, w - box)))
                y = int(rng.integers(0, max(1, h - box)))
                overlap = not (x + box < ox0 or x > ox1 or y + box < oy0 or y > oy1)
                if not overlap:
                    break
            arr[y : y + box, x : x + box, :] = np.array([238, 240, 235], dtype=np.float32) / 255.0
            images.append(arr)
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()


def _batched_view_average(model: ClipZeroShotClassifier, tensor: torch.Tensor, n_examples: int, n_views: int) -> np.ndarray:
    probs = _predict_tensor_probs(model, tensor)
    return probs.reshape(n_examples, n_views, -1).mean(axis=1)


def _stability(original: np.ndarray, variants: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    dists = [0.5 * np.abs(original - variant).sum(axis=1) for variant in variants]
    cic = np.max(np.stack(dists, axis=1), axis=1) if dists else np.zeros(len(original))
    return cic, np.clip(1.0 - cic, 0.0, 1.0)


def _row(
    ex: dict[str, Any],
    method: str,
    class_names: list[str],
    original_probs: np.ndarray,
    repaired_probs: np.ndarray,
    cic_score: float,
    stability_score: float,
    selected_intervention: str,
    repair_action: str,
    abstained: bool,
    status: ClipStatus,
    high_confidence_threshold: float,
    low_stability_threshold: float,
) -> dict[str, Any]:
    orig_pred = int(original_probs.argmax())
    orig_conf = float(original_probs.max())
    if abstained:
        rep_pred: int | None = None
        rep_conf = 0.0
        rep_correct = False
    else:
        rep_pred = int(repaired_probs.argmax())
        rep_conf = float(repaired_probs.max())
        rep_correct = rep_pred == int(ex["label"])
    quadrant = "Dangerous shortcut reliance" if orig_conf >= high_confidence_threshold and stability_score < low_stability_threshold else "Confident and stable"
    if orig_conf < high_confidence_threshold and stability_score < low_stability_threshold:
        quadrant = "Uncertain and unstable"
    elif orig_conf < high_confidence_threshold:
        quadrant = "Low confidence but stable"
    return {
        "example_id": ex["example_id"],
        "split": ex["split"],
        "regime": ex["regime"],
        "true_label": ex["true_label"],
        "label": int(ex["label"]),
        "overlay_text": ex["overlay_text"],
        "overlay_relation": ex["overlay_relation"],
        "overlay_bbox": json.dumps([int(v) for v in ex["overlay_bbox"]]),
        "image_path": ex["image_path"],
        "method": method,
        "original_prediction": class_names[orig_pred],
        "original_prediction_index": orig_pred,
        "original_confidence": orig_conf,
        "original_correct": bool(orig_pred == int(ex["label"])),
        "cic_score": float(cic_score),
        "stability_score": float(stability_score),
        "quadrant": quadrant,
        "selected_intervention": selected_intervention,
        "repaired_prediction": "" if rep_pred is None else class_names[rep_pred],
        "repaired_prediction_index": np.nan if rep_pred is None else rep_pred,
        "repaired_confidence": rep_conf,
        "repaired_correct": bool(rep_correct),
        "repair_action": repair_action,
        "abstained": bool(abstained),
        "model_backend": status.backend,
        "model_name": status.model_name,
        "pretrained_loaded": bool(status.pretrained),
    }


def _certificate_rows(
    examples: list[dict[str, Any]],
    class_names: list[str],
    status: ClipStatus,
    probs: dict[str, np.ndarray],
    selected: dict[str, Any],
    thresholds: tuple[float, float],
) -> pd.DataFrame:
    high_thr, low_stability_thr = thresholds
    original = probs["original"]
    selected_key = _strategy_key(str(selected["neutralization_strategy"]))
    selected_probs = probs[selected_key]
    consensus_probs = np.mean(
        np.stack(
            [
                original,
                probs["mask_overlay_bbox_background_image"],
                probs["blur_overlay_bbox_image"],
                probs["neutral_word_overlay_image"],
                probs["overlay_removed_image"],
            ],
            axis=0,
        ),
        axis=0,
    )
    cic, stability = _stability(
        original,
        [
            probs["mask_overlay_bbox_background_image"],
            probs["blur_overlay_bbox_image"],
            probs["neutral_word_overlay_image"],
            probs["overlay_removed_image"],
        ],
    )
    methods = {
        "original_clip_prediction": ("none", original),
        "random_augmentation_consensus": ("random_augmentation", probs["random_augmentation"]),
        "generic_occlusion_consensus": ("generic_occlusion", probs["generic_occlusion"]),
        "cic_overlay_neutralized_prediction": (str(selected["neutralization_strategy"]), selected_probs),
        "cic_counterfactual_consensus": ("overlay_counterfactual_consensus", consensus_probs),
    }
    rows: list[dict[str, Any]] = []
    for i, ex in enumerate(examples):
        for method, (intervention, repaired) in methods.items():
            rows.append(
                _row(
                    ex,
                    method,
                    class_names,
                    original[i],
                    repaired[i],
                    cic[i],
                    stability[i],
                    intervention,
                    "keep_original" if method == "original_clip_prediction" else "repair",
                    False,
                    status,
                    high_thr,
                    low_stability_thr,
                )
            )
        conf_abstain = float(original[i].max()) < high_thr
        rows.append(
            _row(
                ex,
                "confidence_thresholding",
                class_names,
                original[i],
                original[i],
                cic[i],
                stability[i],
                "none",
                "abstain" if conf_abstain else "keep_original",
                conf_abstain,
                status,
                high_thr,
                low_stability_thr,
            )
        )
        needs_repair = float(original[i].max()) >= high_thr and stability[i] < low_stability_thr
        stable_consensus = int(selected_probs[i].argmax()) == int(consensus_probs[i].argmax())
        abstain = bool(needs_repair and not stable_consensus)
        repaired = selected_probs[i] if needs_repair else original[i]
        rows.append(
            _row(
                ex,
                "cic_repair_or_abstain",
                class_names,
                original[i],
                repaired,
                cic[i],
                stability[i],
                str(selected["neutralization_strategy"]) if needs_repair else "none",
                "abstain" if abstain else ("repair" if needs_repair else "keep_original"),
                abstain,
                status,
                high_thr,
                low_stability_thr,
            )
        )
    return pd.DataFrame(rows)


def _metric_rows(certs: pd.DataFrame, status: ClipStatus, high_confidence_threshold: float, headline_eligible: bool, reasons: list[str]) -> pd.DataFrame:
    rows = []
    regimes = ["aligned_overlay", "misleading_overlay", "neutral_overlay", "no_overlay"]
    for method, method_df in certs.groupby("method", sort=False):
        method_df = method_df.copy()
        original_correct = method_df["original_correct"].astype(bool)
        repaired_correct = method_df["repaired_correct"].astype(bool)
        abstained = method_df["abstained"].astype(bool)
        original_failed = ~original_correct
        repaired_failed = (~abstained) & (~repaired_correct)
        high_before = pd.to_numeric(method_df["original_confidence"], errors="coerce") >= high_confidence_threshold
        high_after = pd.to_numeric(method_df["repaired_confidence"], errors="coerce") >= high_confidence_threshold
        non_abstained = ~abstained
        base: dict[str, Any] = {
            "method": method,
            "regime": "held_out_test",
            "accuracy_before": float(original_correct.mean()),
            "accuracy_after_non_abstained": float(repaired_correct[non_abstained].mean()) if bool(non_abstained.sum()) else np.nan,
            "accuracy_after_counting_abstain_wrong": float((repaired_correct & non_abstained).mean()),
            "high_confidence_failure_rate_before": float((high_before & original_failed).sum() / max(1, int(high_before.sum()))),
            "high_confidence_failure_rate_after": float((high_after & repaired_failed).sum() / max(1, int(high_after.sum()))),
            "coverage": float(non_abstained.mean()),
            "abstention_rate": float(abstained.mean()),
            "selective_accuracy": float(repaired_correct[non_abstained].mean()) if bool(non_abstained.sum()) else np.nan,
            "failure_capture_rate": float((abstained & original_failed).sum() / max(1, int(original_failed.sum()))),
            "false_abstention_rate": float((abstained & original_correct).sum() / max(1, int(original_correct.sum()))),
            "repair_success_rate": float((non_abstained & original_failed & repaired_correct).sum() / max(1, int(original_failed.sum()))),
            "dangerous_quadrant_repair_success_rate": np.nan,
            "n_examples": int(len(method_df)),
            "n_non_abstained": int(non_abstained.sum()),
        }
        dangerous = method_df[method_df["quadrant"] == "Dangerous shortcut reliance"]
        if len(dangerous):
            d_orig_failed = ~dangerous["original_correct"].astype(bool)
            d_non_abs = ~dangerous["abstained"].astype(bool)
            base["dangerous_quadrant_repair_success_rate"] = float((d_non_abs & d_orig_failed & dangerous["repaired_correct"].astype(bool)).sum() / max(1, int(d_orig_failed.sum())))
        for regime in regimes:
            sub = method_df[method_df["regime"] == regime]
            before = float(sub["original_correct"].astype(bool).mean()) if len(sub) else np.nan
            after = float(sub.loc[~sub["abstained"].astype(bool), "repaired_correct"].astype(bool).mean()) if bool((~sub["abstained"].astype(bool)).sum()) else np.nan
            base[f"{regime}_accuracy_before"] = before
            base[f"{regime}_accuracy_after"] = after
            base[f"{regime}_n_examples"] = int(len(sub))
        clean_before = base.get("no_overlay_accuracy_before")
        clean_after = base.get("no_overlay_accuracy_after")
        base["clean_accuracy_drop"] = clean_before - clean_after if np.isfinite(clean_before) and np.isfinite(clean_after) else np.nan
        base["evidence_status"] = "pretrained CLIP repair evidence"
        base["headline_eligible"] = bool(headline_eligible)
        base["include_in_final_headline"] = bool(headline_eligible)
        base["headline_eligibility_reasons"] = "; ".join(reasons) if reasons else "eligible"
        base["backend"] = status.backend
        base["model_name"] = status.model_name
        base["pretrained_tag"] = status.pretrained_tag
        base["pretrained"] = bool(status.pretrained)
        rows.append(base)
    return pd.DataFrame(rows)


def _headline_eligibility(metrics: pd.DataFrame, status: ClipStatus) -> tuple[bool, list[str]]:
    reasons = []
    if status.backend not in {"open_clip", "transformers"} or not status.pretrained:
        reasons.append("pretrained CLIP backend did not load")
    if not len(metrics):
        reasons.append("no held-out test metrics")
    else:
        first = metrics.iloc[0]
        if int(first.get("misleading_overlay_n_examples", 0)) < 30:
            reasons.append("misleading_overlay n_examples < 30")
        if int(first.get("aligned_overlay_n_examples", 0)) < 30:
            reasons.append("aligned_overlay n_examples < 30")
        clean = max(float(first.get("no_overlay_accuracy_before", np.nan)), float(first.get("neutral_overlay_accuracy_before", np.nan)))
        if not np.isfinite(clean) or clean < 0.70:
            reasons.append("no_overlay or neutral_overlay accuracy is not reasonably high")
    return len(reasons) == 0, reasons


def _plot(metrics: pd.DataFrame, png: Path, pdf: Path) -> None:
    methods = ["original_clip_prediction", "random_augmentation_consensus", "cic_overlay_neutralized_prediction", "cic_counterfactual_consensus"]
    take = metrics[metrics["method"].isin(methods)].copy() if "method" in metrics.columns else pd.DataFrame()
    plt.figure(figsize=(9.2, 4.8))
    if len(take):
        x = np.arange(len(take))
        width = 0.35
        plt.bar(x - width / 2, take["misleading_overlay_accuracy_before"], width, label="Misleading before", color="#8c564b")
        plt.bar(x + width / 2, take["misleading_overlay_accuracy_after"], width, label="Misleading after", color="#2ca02c")
        plt.xticks(x, take["method"], rotation=25, ha="right")
        plt.ylim(0, 1.02)
        plt.ylabel("Held-out test accuracy")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No pretrained CLIP repair metrics", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(png, dpi=170)
    plt.savefig(pdf)
    plt.close()


def _write_unavailable(out_dir: Path, cfg: dict[str, Any], status: ClipStatus) -> dict[str, str]:
    metrics = pd.DataFrame(
        [
            {
                "method": "unavailable",
                "regime": "unavailable",
                "evidence_status": "unavailable",
                "headline_eligible": False,
                "include_in_final_headline": False,
                "backend": status.backend,
                "model_name": status.model_name,
                "pretrained_tag": status.pretrained_tag,
                "pretrained": False,
                "pretrained_loaded": False,
                "headline_eligibility_reasons": status.error_message or "pretrained CLIP did not load",
            }
        ]
    )
    certs = pd.DataFrame(columns=CERT_COLUMNS)
    metrics.to_csv(out_dir / "clip_overlay_repair_metrics.csv", index=False)
    certs.to_csv(out_dir / "clip_overlay_repair_certificates.csv", index=False)
    (out_dir / "clip_overlay_repair_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    (out_dir / "selected_clip_repair_config.json").write_text(json.dumps({"source_split": "validation", "status": "unavailable"}, indent=2), encoding="utf-8")
    pd.DataFrame(columns=["split", "prompt_template", "neutralization_strategy", "selection_score"]).to_csv(out_dir / "clip_repair_validation_sweep.csv", index=False)
    _plot(pd.DataFrame(), out_dir / "clip_overlay_repair_plot.png", out_dir / "clip_overlay_repair_plot.pdf")
    summary = [
        "# CLIP Overlay Repair",
        "",
        "CLIP unavailable.",
        "",
        "Evidence status: unavailable.",
        "Headline eligible: `False`.",
        "Include in final headline: `False`.",
        f"Backend attempted: {status.backend_attempted}.",
        f"Backend used: {status.backend}.",
        f"Model name: {status.model_name or 'unavailable'}.",
        f"Pretrained tag: {status.pretrained_tag or 'none'}.",
        f"Downloads allowed: {status.downloads_allowed}.",
        f"Error: {status.error_message}",
        "",
        "No fake repair metrics were generated. This run must not be used as pretrained CLIP repair evidence.",
    ]
    (out_dir / "clip_overlay_repair_summary.md").write_text("\n".join(summary), encoding="utf-8")
    (out_dir / "clip_overlay_repair_examples.md").write_text("# CLIP Overlay Repair Examples\n\nUnavailable: pretrained CLIP did not load.\n", encoding="utf-8")
    (out_dir / "clip_overlay_repair_caption.md").write_text("# CLIP Overlay Repair Figure Caption\n\nUnavailable: pretrained CLIP did not load.\n", encoding="utf-8")
    return {
        "metrics": str(out_dir / "clip_overlay_repair_metrics.csv"),
        "certificates": str(out_dir / "clip_overlay_repair_certificates.csv"),
        "summary": str(out_dir / "clip_overlay_repair_summary.md"),
    }


def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "clip_overlay_repair")
    examples_dir = ensure_dir(out_dir / "examples")
    data_cfg = cfg.get("data", {})
    regimes = list(data_cfg.get("regimes", ["aligned_overlay", "misleading_overlay", "neutral_overlay", "no_overlay"]))
    size = int(data_cfg.get("image_size", 224))
    val_n = int(data_cfg.get("validation_n_per_class", data_cfg.get("n_per_class_validation", 2)))
    test_n = int(data_cfg.get("test_n_per_class", data_cfg.get("n_per_class", 8)))
    val_bundle = make_clip_overlay_dataset(n_per_class=val_n, size=size, regimes=regimes, split="validation", start_id=0)
    test_bundle = make_clip_overlay_dataset(n_per_class=test_n, size=size, regimes=regimes, split="test", start_id=len(val_bundle.examples))
    all_examples = val_bundle.examples + test_bundle.examples
    save_example_images(all_examples, examples_dir)
    save_default_example_grids(test_bundle, examples_dir)

    model_cfg = cfg.get("model", {})
    device = _device(model_cfg, cfg)
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))
    if str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend == "fake":
        status = ClipStatus(
            available=False,
            backend="fake",
            model_name="fake_clip_overlay",
            pretrained=False,
            downloaded_or_cached="not_allowed_for_repair",
            device=device,
            downloads_allowed=False,
            backend_attempted="fake",
            error_message="fake backend is not allowed for CLIP overlay repair evidence",
        )
        return _write_unavailable(out_dir, cfg, status)

    status = check_clip_available(
        device=device,
        allow_download=_downloads_allowed(model_cfg),
        preferred_backend=requested_backend,
        model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
        pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
        transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
    )
    real_pretrained_clip = status.available and status.backend in {"open_clip", "transformers"} and bool(status.pretrained)
    if not real_pretrained_clip:
        return _write_unavailable(out_dir, cfg, status)

    selected = _validation_sweep(status, val_bundle.examples, test_bundle.class_names, device, out_dir)
    model = ClipZeroShotClassifier(status, test_bundle.class_names, prompts=_class_prompts(str(selected["prompt_template"]), test_bundle.class_names), device=device)
    test_examples = test_bundle.examples
    probs: dict[str, np.ndarray] = {
        "original": _predict_probs(model, test_examples, "image"),
        "overlay_removed_image": _predict_probs(model, test_examples, "overlay_removed_image"),
        "neutral_overlay_image": _predict_probs(model, test_examples, "neutral_overlay_image"),
        "neutral_word_overlay_image": _predict_probs(model, test_examples, "neutral_word_overlay_image"),
        "mask_overlay_bbox_background_image": _predict_probs(model, test_examples, "mask_overlay_bbox_background_image"),
        "replace_overlay_bbox_background_image": _predict_probs(model, test_examples, "replace_overlay_bbox_background_image"),
        "blur_overlay_bbox_image": _predict_probs(model, test_examples, "blur_overlay_bbox_image"),
        "crop_out_overlay_image": _predict_probs(model, test_examples, "crop_out_overlay_image"),
    }
    n_views = int(cfg.get("augmentation_views", 5))
    probs["random_augmentation"] = _batched_view_average(model, _augment_tensor(test_examples, rng, n_views), len(test_examples), n_views)
    probs["generic_occlusion"] = _batched_view_average(model, _generic_occlusion_tensor(test_examples, rng, n_views), len(test_examples), n_views)

    certs = _certificate_rows(
        test_examples,
        test_bundle.class_names,
        status,
        probs,
        selected,
        (float(cfg.get("confidence_threshold", 0.8)), float(cfg.get("low_stability_threshold", 0.5))),
    )
    preliminary = _metric_rows(certs, status, float(cfg.get("confidence_threshold", 0.8)), False, [])
    eligible, reasons = _headline_eligibility(preliminary, status)
    metrics = _metric_rows(certs, status, float(cfg.get("confidence_threshold", 0.8)), eligible, reasons)
    certs.to_csv(out_dir / "clip_overlay_repair_certificates.csv", index=False)
    metrics.to_csv(out_dir / "clip_overlay_repair_metrics.csv", index=False)
    (out_dir / "clip_overlay_repair_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    _plot(metrics, out_dir / "clip_overlay_repair_plot.png", out_dir / "clip_overlay_repair_plot.pdf")

    examples_md = ["# CLIP Overlay Repair Examples", ""]
    interesting = certs[(certs["method"].str.startswith("cic_")) & (certs["regime"] == "misleading_overlay")].head(8)
    for _, row in interesting.iterrows():
        examples_md.extend(
            [
                f"## Example {row['example_id']}",
                "",
                f"- True label: `{row['true_label']}`; overlay text: `{row['overlay_text']}`.",
                f"- Method: `{row['method']}`; action: `{row['repair_action']}`; intervention: `{row['selected_intervention']}`.",
                f"- Original prediction: `{row['original_prediction']}`; repaired prediction: `{row['repaired_prediction']}`.",
                f"- Image path: `{row['image_path']}`.",
                "",
            ]
        )
    (out_dir / "clip_overlay_repair_examples.md").write_text("\n".join(examples_md), encoding="utf-8")
    (out_dir / "clip_overlay_repair_caption.md").write_text(
        "\n".join(
            [
                "# CLIP Overlay Repair Figure Caption",
                "",
                "Held-out test misleading-overlay accuracy before and after generic baselines and CIC-guided overlay neutralization. Rows are headline eligible only when pretrained CLIP loaded and split/size guards pass.",
            ]
        ),
        encoding="utf-8",
    )
    lookup = metrics.set_index("method").to_dict("index")
    original_row = lookup.get("original_clip_prediction", {})
    cic_row = lookup.get("cic_overlay_neutralized_prediction", {})
    random_row = lookup.get("random_augmentation_consensus", {})
    summary = [
        "# CLIP Overlay Repair",
        "",
        "This is pretrained CLIP repair evidence: the backend is open_clip or transformers, pretrained weights loaded, no fake backend was used, and inference was performed on generated PNG image files.",
        "",
        f"Evidence status: pretrained CLIP repair evidence. Backend: {status.backend}. Model: {status.model_name}. Pretrained tag: {status.pretrained_tag or 'none'}. Pretrained loaded: `{status.pretrained}`.",
        f"Headline eligible: `{eligible}`.",
        "",
        "## Headline Eligibility",
        "",
        "Eligible." if eligible else f"Not eligible: {'; '.join(reasons)}.",
        "",
        "## Headline Candidate Numbers",
        "",
        f"- Original misleading accuracy: {original_row.get('misleading_overlay_accuracy_before', np.nan):.3f}",
        f"- CIC repaired misleading accuracy: {cic_row.get('misleading_overlay_accuracy_after', np.nan):.3f}",
        f"- Aligned accuracy before/after: {original_row.get('aligned_overlay_accuracy_before', np.nan):.3f} / {cic_row.get('aligned_overlay_accuracy_after', np.nan):.3f}",
        f"- No-overlay accuracy before/after: {original_row.get('no_overlay_accuracy_before', np.nan):.3f} / {cic_row.get('no_overlay_accuracy_after', np.nan):.3f}",
        f"- Clean accuracy drop: {cic_row.get('clean_accuracy_drop', np.nan):.3f}",
        f"- Random augmentation misleading accuracy after: {random_row.get('misleading_overlay_accuracy_after', np.nan):.3f}",
        "",
        "## Validation-Only Selection",
        "",
        f"Selected prompt `{selected['prompt_template']}` and neutralization `{selected['neutralization_strategy']}` using validation split only.",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
        "",
        "## Limitations",
        "",
        "- This is a targeted typographic-overlay shortcut repair attempt, not a claim of general robustness.",
        "- The held-out test split was not used for prompt or neutralization selection.",
        "- If headline eligibility is false, do not claim CIC improves pretrained CLIP repair.",
    ]
    (out_dir / "clip_overlay_repair_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {
        "metrics": str(out_dir / "clip_overlay_repair_metrics.csv"),
        "certificates": str(out_dir / "clip_overlay_repair_certificates.csv"),
        "summary": str(out_dir / "clip_overlay_repair_summary.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clip_overlay_repair.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
