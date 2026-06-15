"""Embedding-additivity validation for the finite-candidate CIC recovery theorem.

The finite-candidate recovery theorem (``docs/theory.md``) assumes an additive
logit decomposition::

    logit_y(g(C,S,N)) = phi_y(C,N) + psi_y(S) + xi_y(C,S,N)

For CLIP, logits are inner products ``logit_y(X) = <u(X), v_y>`` where ``u(X)``
is the (normalized) image embedding and ``v_y`` is the class text embedding.
Therefore additive logits are plausible *iff* the embedding shift caused by a
shortcut is approximately input-independent::

    u(object + shortcut) - u(object only) ~= shortcut_direction

This experiment tests that empirically on two finite-candidate shortcut families
using real pretrained OpenCLIP (no fake backend for headline/theory evidence):

    1. the hard multi-decoy text-overlay shortcut       (expected: supports additivity)
    2. the cross-shortcut colored-symbol watermark       (negative result; expected weak)

It measures direction clustering, shortcut-vs-object separability, logit-channel
consistency, neutralization quality, and a recovery-margin diagnostic, and it
emits a-priori headline-eligibility booleans for whether embedding additivity is
empirically supported for each family. Nothing is tuned on the observed metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

_mpl_cache = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_reliability.data.clip_overlay_shortcuts import CLIP_OVERLAY_CLASSES
from causal_reliability.discovery.cic_region_scoring import neutralize_region
from causal_reliability.experiments.run_clip_overlay_validation import _downloads_allowed
from causal_reliability.experiments.run_cross_shortcut_generalization import (
    ALIGNED_REGIME as WM_ALIGNED_REGIME,
    MIS_REGIME as WM_MIS_REGIME,
    NEUTRAL_REGIME as WM_NEUTRAL_REGIME,
    NO_OVERLAY_REGIME as WM_NO_OVERLAY_REGIME,
    SHORTCUT_TYPE as WATERMARK_SHORTCUT_TYPE,
    render_cross_shortcut_image,
)
from causal_reliability.experiments.run_hard_multidecoy_clip_repair import render_hard_multidecoy_image
from causal_reliability.experiments.run_nonoracle_clip_repair import PROMPT_TEMPLATE, _device
from causal_reliability.real_models.clip_zero_shot import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED_TAG,
    DEFAULT_TRANSFORMERS_MODEL,
    ClipStatus,
    check_clip_available,
    encode_images,
    encode_text_prompts,
)
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


TEXT_MIS_REGIME = "hard_multi_decoy_misleading"
TEXT_ALIGNED_REGIME = "hard_multi_decoy_aligned"
TEXT_NEUTRAL_REGIME = "hard_multi_decoy_neutral"
TEXT_NO_OVERLAY_REGIME = "no_overlay"
TEXT_SHORTCUT_TYPE = "text_overlay_misleading_word"

DEFAULT_TEXT_GENERATION_POLICY = {
    "class_set_size": 4,
    "decoy_near_object": False,
    "decoy_placement": "away",
    "decoy_text_size": 224,
    "harmful_near_object": True,
    "harmful_placement": "over_object",
    "harmful_text_size": 224,
    "n_decoys": 4,
    "policy_id": "default_embedding_additivity_policy",
    "repeat_harmful_word": True,
    "text_box_background": True,
    "text_color": [30, 30, 30],
    "text_opacity": 1.0,
}


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _to_array(image: Any, size: int) -> np.ndarray:
    """Coerce a PIL image or HxWx3 array to a float32 [0,1] RGB array."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    arr = np.asarray(image).astype(np.float32)
    return arr.clip(0.0, 1.0)


def _encode_arrays(status: ClipStatus, arrays: list[np.ndarray], device: str) -> np.ndarray:
    if not arrays:
        return np.zeros((0, 0), dtype=np.float32)
    tensor = torch.from_numpy(np.stack(arrays).astype(np.float32)).permute(0, 3, 1, 2).contiguous()
    feats = encode_images(status, tensor, device)
    return np.asarray(feats.detach().cpu().numpy(), dtype=np.float64)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Example construction (regenerated from frozen policies; nothing tuned)
# ---------------------------------------------------------------------------
def build_text_examples(policy: dict[str, Any], n_per_class: int, size: int) -> list[dict[str, Any]]:
    """Paired text-overlay examples: clean / shortcut / neutralized / aligned / neutral."""
    examples: list[dict[str, Any]] = []
    eid = 0
    for label in range(min(int(policy.get("class_set_size", len(CLIP_OVERLAY_CLASSES))), len(CLIP_OVERLAY_CLASSES))):
        for index in range(n_per_class):
            clean, _ = render_hard_multidecoy_image(label, TEXT_NO_OVERLAY_REGIME, index, policy, size=size)
            shortcut, mis_meta = render_hard_multidecoy_image(label, TEXT_MIS_REGIME, index, policy, size=size)
            aligned, _ = render_hard_multidecoy_image(label, TEXT_ALIGNED_REGIME, index, policy, size=size)
            neutral, _ = render_hard_multidecoy_image(label, TEXT_NEUTRAL_REGIME, index, policy, size=size)
            harmful_bbox = mis_meta.get("harmful_bbox") or []
            neutralized = (
                _to_array(neutralize_region(Image.fromarray((shortcut.clip(0, 1) * 255).astype(np.uint8)), tuple(int(v) for v in harmful_bbox)), size)
                if harmful_bbox
                else None
            )
            harmful_text = str(mis_meta.get("harmful_text", "")).split()[0] if mis_meta.get("harmful_text") else ""
            examples.append(
                {
                    "family": "text_overlay",
                    "example_id": eid,
                    "label": int(label),
                    "true_label": CLIP_OVERLAY_CLASSES[label],
                    "shortcut_value": harmful_text,
                    "harmful_bbox": list(harmful_bbox),
                    "x_clean": clean,
                    "x_shortcut": shortcut,
                    "x_neutralized": neutralized,
                    "x_aligned": aligned,
                    "x_neutral": neutral,
                }
            )
            eid += 1
    return examples


def build_watermark_examples(n_per_class: int, size: int, benchmark_seed: int) -> list[dict[str, Any]]:
    """Paired colored-symbol watermark examples mirroring the text construction."""
    examples: list[dict[str, Any]] = []
    eid = 0
    for label in range(len(CLIP_OVERLAY_CLASSES)):
        for index in range(n_per_class):
            clean, _ = render_cross_shortcut_image(label, WM_NO_OVERLAY_REGIME, index, size=size, benchmark_seed=benchmark_seed)
            shortcut, mis_meta = render_cross_shortcut_image(label, WM_MIS_REGIME, index, size=size, benchmark_seed=benchmark_seed)
            aligned, _ = render_cross_shortcut_image(label, WM_ALIGNED_REGIME, index, size=size, benchmark_seed=benchmark_seed)
            neutral, _ = render_cross_shortcut_image(label, WM_NEUTRAL_REGIME, index, size=size, benchmark_seed=benchmark_seed)
            harmful_bbox = mis_meta.get("harmful_bbox") or []
            neutralized = (
                _to_array(neutralize_region(Image.fromarray((shortcut.clip(0, 1) * 255).astype(np.uint8)), tuple(int(v) for v in harmful_bbox)), size)
                if harmful_bbox
                else None
            )
            examples.append(
                {
                    "family": "watermark",
                    "example_id": eid,
                    "label": int(label),
                    "true_label": CLIP_OVERLAY_CLASSES[label],
                    "shortcut_value": str(mis_meta.get("shortcut_label_association", "")),
                    "harmful_bbox": list(harmful_bbox),
                    "x_clean": clean,
                    "x_shortcut": shortcut,
                    "x_neutralized": neutralized,
                    "x_aligned": aligned,
                    "x_neutral": neutral,
                }
            )
            eid += 1
    return examples


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def _pairwise_within_group_cosines(deltas: np.ndarray, labels: np.ndarray) -> list[float]:
    cosines: list[float] = []
    for value in np.unique(labels):
        idx = np.where(labels == value)[0]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                cosines.append(_cosine(deltas[idx[i]], deltas[idx[j]]))
    return cosines


def _centroids(deltas: np.ndarray, labels: np.ndarray) -> dict[Any, np.ndarray]:
    return {value: deltas[labels == value].mean(axis=0) for value in np.unique(labels)}


def _nearest_centroid_loo_accuracy(deltas: np.ndarray, labels: np.ndarray) -> float:
    """Leave-one-out nearest-centroid accuracy (centroid excludes the held-out point)."""
    if len(deltas) == 0:
        return float("nan")
    unique = list(np.unique(labels))
    correct = 0
    total = 0
    for i in range(len(deltas)):
        best_value = None
        best_sim = -np.inf
        for value in unique:
            idx = np.where(labels == value)[0]
            idx = idx[idx != i]
            if len(idx) == 0:
                continue
            centroid = deltas[idx].mean(axis=0)
            sim = _cosine(deltas[i], centroid)
            if sim > best_sim:
                best_sim = sim
                best_value = value
        if best_value is None:
            continue
        total += 1
        correct += int(best_value == labels[i])
    return float(correct / total) if total else float("nan")


def _centroid_distance_mean(centroids: dict[Any, np.ndarray]) -> float:
    keys = list(centroids)
    dists = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            dists.append(float(np.linalg.norm(centroids[keys[i]] - centroids[keys[j]])))
    return float(np.mean(dists)) if dists else float("nan")


def compute_family_metrics(
    family: str,
    shortcut_type: str,
    examples: list[dict[str, Any]],
    status: ClipStatus,
    device: str,
    class_text_feats: np.ndarray,
    thresholds: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (metrics, per_example_df, delta_shortcut, shortcut_values, object_values)."""
    size = int(examples[0]["x_clean"].shape[0]) if examples else 0
    # Batch-encode all images for the family in one pass per view.
    clean = _encode_arrays(status, [ex["x_clean"] for ex in examples], device)
    shortcut = _encode_arrays(status, [ex["x_shortcut"] for ex in examples], device)
    aligned = _encode_arrays(status, [ex["x_aligned"] for ex in examples], device)
    neutral = _encode_arrays(status, [ex["x_neutral"] for ex in examples], device)
    has_neutralized = [ex["x_neutralized"] is not None for ex in examples]
    neutralized = np.full_like(shortcut, np.nan)
    neut_idx = [i for i, ok in enumerate(has_neutralized) if ok]
    if neut_idx:
        neut_feats = _encode_arrays(status, [examples[i]["x_neutralized"] for i in neut_idx], device)
        for k, i in enumerate(neut_idx):
            neutralized[i] = neut_feats[k]

    delta_shortcut = shortcut - clean  # u(x_shortcut) - u(x_clean)
    object_values = np.asarray([ex["true_label"] for ex in examples])
    shortcut_values = np.asarray([ex["shortcut_value"] for ex in examples])

    # Per-example records.
    rows: list[dict[str, Any]] = []
    logit_pred_errors: list[float] = []
    logit_pred_actual_pairs: list[tuple[float, float]] = []
    for i, ex in enumerate(examples):
        d_sc = delta_shortcut[i]
        shortcut_effect_norm = float(np.linalg.norm(d_sc))
        if has_neutralized[i]:
            d_repair = shortcut[i] - neutralized[i]
            d_neut_err = neutralized[i] - clean[i]
            clean_damage = float(np.linalg.norm(d_neut_err))
        else:
            d_repair = np.full_like(d_sc, np.nan)
            clean_damage = float("nan")
        ratio = float(clean_damage / shortcut_effect_norm) if shortcut_effect_norm > 1e-12 and np.isfinite(clean_damage) else float("nan")

        clean_logits = 100.0 * clean[i] @ class_text_feats.T
        shortcut_logits = 100.0 * shortcut[i] @ class_text_feats.T
        label = ex["label"]
        # causal margin on the no-overlay (clean) image.
        order = np.argsort(clean_logits)[::-1]
        runner_up = clean_logits[order[1]] if order[0] == label else clean_logits[order[0]]
        causal_margin = float(clean_logits[label] - runner_up)
        if has_neutralized[i]:
            neutralized_logits = 100.0 * neutralized[i] @ class_text_feats.T
            residual_error = float(np.max(np.abs(neutralized_logits - clean_logits)))
            repair_success = bool(int(np.argmax(neutralized_logits)) == label)
        else:
            residual_error = float("nan")
            repair_success = False
        margin_condition = bool(np.isfinite(residual_error) and causal_margin > 2.0 * residual_error)

        # Logit-channel consistency: predicted shift <delta, v_y> vs actual logit diff.
        predicted_shift = 100.0 * (d_sc @ class_text_feats.T)
        actual_shift = shortcut_logits - clean_logits
        logit_pred_errors.append(float(np.mean(np.abs(predicted_shift - actual_shift))))
        for y in range(class_text_feats.shape[0]):
            logit_pred_actual_pairs.append((float(predicted_shift[y]), float(actual_shift[y])))

        rows.append(
            {
                "family": family,
                "shortcut_type": shortcut_type,
                "example_id": ex["example_id"],
                "label": label,
                "true_label": ex["true_label"],
                "shortcut_value": ex["shortcut_value"],
                "shortcut_effect_norm": shortcut_effect_norm,
                "clean_damage_proxy": clean_damage,
                "neutralization_ratio": ratio,
                "logit_shift_true_class": float(actual_shift[label]),
                "repaired_logit_shift_true_class": float(100.0 * (d_repair @ class_text_feats.T)[label]) if has_neutralized[i] else float("nan"),
                "causal_margin_proxy": causal_margin,
                "residual_error_proxy": residual_error,
                "margin_condition_satisfied": margin_condition,
                "original_shortcut_correct": bool(int(np.argmax(shortcut_logits)) == label),
                "repair_success": repair_success,
                "has_neutralized": has_neutralized[i],
            }
        )
    per_example = pd.DataFrame(rows)

    # A. Direction clustering.
    within_cosines = _pairwise_within_group_cosines(delta_shortcut, shortcut_values)
    within_object_cosines = _pairwise_within_group_cosines(delta_shortcut, object_values)
    centroids = _centroids(delta_shortcut, shortcut_values)
    cosine_to_centroid = [
        _cosine(delta_shortcut[i], centroids[shortcut_values[i]]) for i in range(len(examples))
    ]
    n_shuffles = int(thresholds.get("n_label_shuffles", 25))
    shuffled_means = []
    for _ in range(max(1, n_shuffles)):
        perm = rng.permutation(len(shortcut_values))
        shuffled_means.append(float(np.mean(_pairwise_within_group_cosines(delta_shortcut, shortcut_values[perm]))) if len(within_cosines) else float("nan"))
    shuffled_mean = float(np.nanmean(shuffled_means)) if shuffled_means else float("nan")

    # B/C. Separability by shortcut vs object.
    nc_shortcut = _nearest_centroid_loo_accuracy(delta_shortcut, shortcut_values)
    nc_object = _nearest_centroid_loo_accuracy(delta_shortcut, object_values)
    shuffled_nc = []
    for _ in range(max(1, n_shuffles)):
        perm = rng.permutation(len(shortcut_values))
        shuffled_nc.append(_nearest_centroid_loo_accuracy(delta_shortcut, shortcut_values[perm]))
    shuffled_nc_mean = float(np.nanmean(shuffled_nc)) if shuffled_nc else float("nan")

    # D. Logit-channel consistency.
    pairs = np.asarray(logit_pred_actual_pairs)
    logit_mae = float(np.mean(np.abs(pairs[:, 0] - pairs[:, 1]))) if len(pairs) else float("nan")
    if len(pairs) > 1 and np.std(pairs[:, 0]) > 1e-12 and np.std(pairs[:, 1]) > 1e-12:
        logit_corr = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])
    else:
        logit_corr = float("nan")

    # E. Neutralization quality.
    ratios = per_example["neutralization_ratio"].to_numpy(dtype=float)
    mean_ratio = float(np.nanmean(ratios)) if np.isfinite(ratios).any() else float("nan")
    mean_clean_damage = float(np.nanmean(per_example["clean_damage_proxy"].to_numpy(dtype=float)))
    mean_shortcut_effect = float(np.nanmean(per_example["shortcut_effect_norm"].to_numpy(dtype=float)))

    # F. Recovery-margin diagnostic.
    neut_df = per_example[per_example["has_neutralized"]]
    satisfied = neut_df[neut_df["margin_condition_satisfied"]]
    violated = neut_df[~neut_df["margin_condition_satisfied"]]
    success_satisfied = float(satisfied["repair_success"].mean()) if len(satisfied) else float("nan")
    success_violated = float(violated["repair_success"].mean()) if len(violated) else float("nan")
    repair_success_rate = float(neut_df["repair_success"].mean()) if len(neut_df) else float("nan")

    within_mean = float(np.mean(within_cosines)) if within_cosines else float("nan")
    metrics = {
        "family": family,
        "shortcut_type": shortcut_type,
        "backend": status.backend,
        "model_name": status.model_name,
        "pretrained_loaded": bool(status.pretrained),
        "n_examples": int(len(examples)),
        "n_shortcut_values": int(len(np.unique(shortcut_values))),
        "n_object_values": int(len(np.unique(object_values))),
        # A
        "within_shortcut_mean_pairwise_cosine": within_mean,
        "within_shortcut_std_pairwise_cosine": float(np.std(within_cosines)) if within_cosines else float("nan"),
        "within_object_mean_pairwise_cosine": float(np.mean(within_object_cosines)) if within_object_cosines else float("nan"),
        "cosine_to_shortcut_centroid_mean": float(np.mean(cosine_to_centroid)) if cosine_to_centroid else float("nan"),
        "shuffled_mean_pairwise_cosine": shuffled_mean,
        "delta_cosine_minus_shuffled": within_mean - shuffled_mean if np.isfinite(within_mean) and np.isfinite(shuffled_mean) else float("nan"),
        "between_shortcut_centroid_distance_mean": _centroid_distance_mean(centroids),
        # B/C
        "nearest_centroid_accuracy_shortcut": nc_shortcut,
        "nearest_centroid_accuracy_object": nc_object,
        "nearest_centroid_accuracy_shuffled": shuffled_nc_mean,
        "nearest_centroid_chance": float(1.0 / max(1, len(np.unique(shortcut_values)))),
        "shortcut_label_clustering_score": nc_shortcut,
        "object_label_clustering_score": nc_object,
        # D
        "logit_consistency_mae": logit_mae,
        "logit_consistency_corr": logit_corr,
        # E
        "mean_neutralization_ratio": mean_ratio,
        "mean_clean_damage_proxy": mean_clean_damage,
        "mean_shortcut_effect_norm": mean_shortcut_effect,
        # F
        "repair_success_rate": repair_success_rate,
        "frac_margin_condition_satisfied": float(neut_df["margin_condition_satisfied"].mean()) if len(neut_df) else float("nan"),
        "repair_success_when_margin_satisfied": success_satisfied,
        "repair_success_when_margin_violated": success_violated,
        "n_margin_satisfied": int(len(satisfied)),
        "n_margin_violated": int(len(violated)),
    }
    return metrics, per_example, delta_shortcut, shortcut_values, object_values


# ---------------------------------------------------------------------------
# Headline eligibility (PART 6) -- a-priori thresholds, not tuned on results
# ---------------------------------------------------------------------------
def _additivity_supported(metrics: dict[str, Any], status: ClipStatus, thresholds: dict[str, Any]) -> tuple[bool, dict[str, bool], list[str]]:
    cluster_margin = float(thresholds.get("cluster_cosine_margin", 0.10))
    ratio_max = float(thresholds.get("neutralization_ratio_max", 0.60))
    margin_success_min = float(thresholds.get("margin_success_min", 0.50))

    within = metrics["within_shortcut_mean_pairwise_cosine"]
    shuffled = metrics["shuffled_mean_pairwise_cosine"]
    delta_clusters = bool(np.isfinite(within) and np.isfinite(shuffled) and (within - shuffled) >= cluster_margin and within > shuffled)
    # Object independence (PART 3C): the shortcut delta must cluster MORE tightly by
    # shortcut value than by object class. We use the (non-saturating) within-group
    # cohesion comparison rather than nearest-centroid accuracy, which can saturate at
    # 1.0 for both groupings and become uninformative. Nearest-centroid accuracies are
    # still reported as diagnostics.
    within_object = metrics["within_object_mean_pairwise_cosine"]
    shortcut_exceeds_object = bool(np.isfinite(within) and np.isfinite(within_object) and within > within_object)
    neutralization_small = bool(np.isfinite(metrics["mean_neutralization_ratio"]) and metrics["mean_neutralization_ratio"] <= ratio_max)
    sat = metrics["repair_success_when_margin_satisfied"]
    vio = metrics["repair_success_when_margin_violated"]
    if not np.isfinite(sat):
        margin_predicts = False
    elif not np.isfinite(vio):
        # No violated examples: directional iff satisfied examples repair well.
        margin_predicts = bool(sat >= margin_success_min)
    else:
        margin_predicts = bool(sat >= vio and sat >= margin_success_min)

    checks = {
        "pretrained_loaded": bool(status.pretrained),
        "not_fake_backend": status.backend != "fake",
        "delta_clusters_above_shuffled": delta_clusters,
        "shortcut_clustering_exceeds_object": shortcut_exceeds_object,
        "neutralization_damage_small": neutralization_small,
        "margin_condition_predicts_repair": margin_predicts,
    }
    supported = all(checks.values())
    reasons = [name for name, ok in checks.items() if not ok]
    return supported, checks, reasons


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _pca_2d(deltas: np.ndarray) -> np.ndarray:
    if len(deltas) < 2:
        return np.zeros((len(deltas), 2))
    centered = np.ascontiguousarray(np.asarray(deltas, dtype=np.float64))
    centered = centered - centered.mean(axis=0, keepdims=True)
    with np.errstate(all="ignore"):
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            coords = centered @ vt[: min(2, vt.shape[0])].T
        except np.linalg.LinAlgError:
            coords = centered[:, :2]
    coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    return coords


def _plot_family(deltas: np.ndarray, shortcut_values: np.ndarray, object_values: np.ndarray, title: str, png: Path, pdf: Path) -> None:
    coords = _pca_2d(deltas)
    plt.figure(figsize=(6.6, 5.2))
    uniq = list(dict.fromkeys(shortcut_values.tolist()))
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "*", "P", "X", "D", "v"]
    obj_uniq = list(dict.fromkeys(object_values.tolist()))
    for si, sval in enumerate(uniq):
        for oi, oval in enumerate(obj_uniq):
            mask = (shortcut_values == sval) & (object_values == oval)
            if not mask.any():
                continue
            plt.scatter(
                coords[mask, 0],
                coords[mask, 1],
                color=cmap(si % 10),
                marker=markers[oi % len(markers)],
                s=46,
                edgecolors="k",
                linewidths=0.4,
                label=f"shortcut={sval or 'none'}" if oi == 0 else None,
            )
    plt.xlabel("delta_shortcut PC1")
    plt.ylabel("delta_shortcut PC2")
    plt.title(title, fontsize=10)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles, labels, fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(png, dpi=170)
    plt.savefig(pdf)
    plt.close()


# ---------------------------------------------------------------------------
# Summary / caption
# ---------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


WATERMARK_WEAK_STATEMENT = (
    "The watermark transfer failure is consistent with the theory: the shortcut channel was weak or flat, "
    "so there was no strong shortcut contribution for CIC to neutralize."
)


def _write_summary(out_dir: Path, key_numbers: dict[str, Any], text_metrics: dict[str, Any], wm_metrics: dict[str, Any], status: ClipStatus) -> None:
    k = key_numbers
    text_supported = bool(k["embedding_additivity_supported_for_text"])
    wm_supported = bool(k["embedding_additivity_supported_for_watermark"])
    if text_supported:
        verdict = (
            "empirically supported for text overlays. The additive-channel condition the theorem depends on is "
            "supported by the embedding-additivity validation, so the finite-candidate recovery theorem plausibly "
            "explains the OpenCLIP text-repair result."
        )
    else:
        verdict = (
            "NOT empirically supported for the current OpenCLIP text benchmark by this validation. The theorem "
            "should be presented as a conditional / theoretical explanation only, and must not be claimed to apply "
            "directly to CLIP text-overlay repair."
        )
    lines = [
        "# Embedding-Additivity Validation",
        "",
        "This experiment gates the theory claim in `docs/theory.md`. CLIP logits are inner products "
        "`logit_y(X) = <u(X), v_y>`, so the additive-logit decomposition the recovery theorem assumes holds iff the "
        "embedding shift caused by a shortcut is approximately input-independent: "
        "`u(object + shortcut) - u(object only) ~= shortcut_direction`. We test that on two finite-candidate shortcut "
        "families with real pretrained OpenCLIP.",
        "",
        f"Backend: {status.backend}. Model: {status.model_name}. Pretrained loaded: `{status.pretrained}`. "
        f"Fake backend: `{status.backend == 'fake'}`.",
        "",
        f"**Embedding additivity is {verdict}**",
        "",
        "## 1. Does the text-overlay shortcut satisfy approximate embedding additivity?",
        "",
        f"- Same-shortcut delta mean pairwise cosine: {_fmt(text_metrics.get('within_shortcut_mean_pairwise_cosine'))} "
        f"(shuffled-label baseline {_fmt(text_metrics.get('shuffled_mean_pairwise_cosine'))}; "
        f"margin {_fmt(text_metrics.get('delta_cosine_minus_shuffled'))})",
        f"- Cosine to shortcut-value centroid (mean): {_fmt(text_metrics.get('cosine_to_shortcut_centroid_mean'))}",
        f"- Logit-channel consistency MAE / corr: {_fmt(text_metrics.get('logit_consistency_mae'))} / {_fmt(text_metrics.get('logit_consistency_corr'))}",
        f"- Verdict: delta vectors cluster above shuffled baseline = `{k['text_delta_clusters_above_shuffled']}`",
        "",
        "## 2. Does the shortcut delta cluster by shortcut value more than by object class?",
        "",
        f"- Within-shortcut delta cohesion (mean pairwise cosine): {_fmt(text_metrics.get('within_shortcut_mean_pairwise_cosine'))}",
        f"- Within-object delta cohesion (mean pairwise cosine): {_fmt(text_metrics.get('within_object_mean_pairwise_cosine'))}",
        f"- Nearest-centroid accuracy by shortcut value: {_fmt(text_metrics.get('nearest_centroid_accuracy_shortcut'))} "
        f"(chance {_fmt(text_metrics.get('nearest_centroid_chance'))}, shuffled {_fmt(text_metrics.get('nearest_centroid_accuracy_shuffled'))})",
        f"- Nearest-centroid accuracy by object class: {_fmt(text_metrics.get('nearest_centroid_accuracy_object'))} "
        "(note: nearest-centroid can saturate at 1.0 for both groupings; the cohesion comparison above is the decisive, non-saturating test)",
        f"- Verdict: shortcut clustering exceeds object clustering = `{k['text_shortcut_clustering_exceeds_object']}`",
        "",
        "## 3. Is neutralization close to the clean embedding?",
        "",
        f"- Mean clean-damage proxy ||u(neutralized) - u(clean)||: {_fmt(text_metrics.get('mean_clean_damage_proxy'))}",
        f"- Mean shortcut-effect norm ||u(shortcut) - u(clean)||: {_fmt(text_metrics.get('mean_shortcut_effect_norm'))}",
        f"- Mean ratio (clean damage / shortcut effect): {_fmt(text_metrics.get('mean_neutralization_ratio'))}",
        f"- Verdict: neutralization damage small relative to shortcut effect = `{k['text_neutralization_damage_small']}`",
        "",
        "## 4. Does the margin diagnostic predict repair success?",
        "",
        f"- Repair-success rate overall: {_fmt(text_metrics.get('repair_success_rate'))}",
        f"- Fraction satisfying margin condition (m > 2*residual): {_fmt(text_metrics.get('frac_margin_condition_satisfied'))}",
        f"- Repair success | margin satisfied: {_fmt(text_metrics.get('repair_success_when_margin_satisfied'))} "
        f"(n={text_metrics.get('n_margin_satisfied')})",
        f"- Repair success | margin violated: {_fmt(text_metrics.get('repair_success_when_margin_violated'))} "
        f"(n={text_metrics.get('n_margin_violated')})",
        f"- Verdict: margin condition predicts repair = `{k['text_margin_condition_predicts_repair']}`",
        "",
        "## 5. Does the watermark negative result show a weak / flat shortcut channel?",
        "",
        f"- Watermark mean shortcut-effect norm: {_fmt(wm_metrics.get('mean_shortcut_effect_norm'))} "
        f"(text {_fmt(text_metrics.get('mean_shortcut_effect_norm'))})",
        f"- Watermark same-shortcut delta cosine vs shuffled: {_fmt(wm_metrics.get('within_shortcut_mean_pairwise_cosine'))} "
        f"vs {_fmt(wm_metrics.get('shuffled_mean_pairwise_cosine'))}",
        f"- Watermark nearest-centroid accuracy (shortcut vs object): {_fmt(wm_metrics.get('nearest_centroid_accuracy_shortcut'))} "
        f"vs {_fmt(wm_metrics.get('nearest_centroid_accuracy_object'))}",
        f"- Watermark shortcut channel weak/flat = `{k['watermark_shortcut_channel_weak']}`",
        "",
        (WATERMARK_WEAK_STATEMENT if k["watermark_shortcut_channel_weak"] else
         "The watermark shortcut channel was not clearly weak by these metrics; see metrics CSV for detail."),
        "",
        "## 6. How should the theorem be presented?",
        "",
        f"- embedding_additivity_supported_for_text = `{text_supported}`",
        f"- embedding_additivity_supported_for_watermark = `{wm_supported}`",
        f"- Recommended framing: {k['theorem_framing']}",
        "",
        "## Scope and caveats",
        "",
        "This is a finite-candidate, controlled validation. It does not establish open-world shortcut discovery, "
        "exact bounding-box localization, or general robustness. It only tests whether the additive-channel "
        "assumption behind the conditional recovery theorem is supported for these specific shortcut families on "
        "this pretrained CLIP model.",
        "",
        "Headline-eligibility checks (text):",
        "",
        *[f"- {name} = `{ok}`" for name, ok in k["text_eligibility_checks"].items()],
    ]
    (out_dir / "embedding_additivity_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "embedding_additivity_caption.md").write_text(
        "# Embedding-Additivity Figure Captions\n\n"
        "**Text delta plot.** PCA projection of per-example `delta_shortcut = u(x_shortcut) - u(x_clean)` for the hard "
        "multi-decoy text-overlay shortcut, colored by misleading-word shortcut value and marked by true object class. "
        "Clustering by color (shortcut value) rather than by marker (object class) supports approximate input-independent "
        "additivity.\n\n"
        "**Watermark delta plot.** The same projection for the non-text colored-symbol watermark shortcut. Weak or absent "
        "clustering by shortcut value is consistent with a weak/flat shortcut channel and the cross-shortcut negative result.\n",
        encoding="utf-8",
    )


def _write_unavailable(out_dir: Path, status: ClipStatus, thresholds: dict[str, Any]) -> dict[str, str]:
    key_numbers = {
        "embedding_additivity_supported_for_text": False,
        "embedding_additivity_supported_for_watermark": False,
        "pretrained_clip_loaded": bool(status.pretrained),
        "fake_backend": status.backend == "fake",
        "backend": status.backend,
        "model_name": status.model_name,
        "evidence_status": status.error_message or "pretrained CLIP unavailable; no theory-supporting evidence generated",
        "theorem_framing": "conditional / theoretical only (validation could not run on pretrained CLIP)",
        "watermark_shortcut_channel_weak": False,
        "text_eligibility_checks": {"pretrained_loaded": bool(status.pretrained), "not_fake_backend": status.backend != "fake"},
        "text_delta_clusters_above_shuffled": False,
        "text_shortcut_clustering_exceeds_object": False,
        "text_neutralization_damage_small": False,
        "text_margin_condition_predicts_repair": False,
    }
    (out_dir / "embedding_additivity_key_numbers.json").write_text(json.dumps(key_numbers, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame([{"family": "unavailable", "backend": status.backend, "pretrained_loaded": False}]).to_csv(out_dir / "embedding_additivity_metrics.csv", index=False)
    pd.DataFrame().to_csv(out_dir / "embedding_additivity_examples.csv", index=False)
    (out_dir / "embedding_additivity_summary.md").write_text(
        "# Embedding-Additivity Validation\n\nPretrained CLIP was unavailable (or a fake backend was requested), so no "
        "theory-supporting evidence was generated. embedding_additivity_supported_for_text = False; "
        "embedding_additivity_supported_for_watermark = False. The theorem remains conditional / theoretical only.\n",
        encoding="utf-8",
    )
    (out_dir / "embedding_additivity_caption.md").write_text("# Captions\n\nUnavailable: pretrained CLIP did not load.\n", encoding="utf-8")
    for stem in ("embedding_additivity_text_delta_plot", "embedding_additivity_watermark_delta_plot"):
        plt.figure(figsize=(5, 3))
        plt.text(0.5, 0.5, "pretrained CLIP unavailable", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_dir / f"{stem}.png", dpi=120)
        plt.savefig(out_dir / f"{stem}.pdf")
        plt.close()
    return {
        "summary": str(out_dir / "embedding_additivity_summary.md"),
        "metrics": str(out_dir / "embedding_additivity_metrics.csv"),
        "examples": str(out_dir / "embedding_additivity_examples.csv"),
        "key_numbers": str(out_dir / "embedding_additivity_key_numbers.json"),
        "caption": str(out_dir / "embedding_additivity_caption.md"),
        "text_delta_plot_png": str(out_dir / "embedding_additivity_text_delta_plot.png"),
        "watermark_delta_plot_png": str(out_dir / "embedding_additivity_watermark_delta_plot.png"),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(cfg: dict[str, Any]) -> dict[str, str]:
    total_start = time.perf_counter()
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "embedding_additivity")
    thresholds = dict(cfg.get("thresholds", {}))
    model_cfg = cfg.get("model", {})
    device = _device(model_cfg, cfg)
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))
    if str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend == "fake":
        status = ClipStatus(False, "fake", "fake_clip_embedding_additivity", pretrained=False, device=device, backend_attempted="fake", error_message="fake backend is not allowed for theory-supporting embedding-additivity evidence")
        return _write_unavailable(out_dir, status, thresholds)
    status = check_clip_available(
        device=device,
        allow_download=_downloads_allowed(model_cfg),
        preferred_backend=requested_backend,
        model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
        pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
        transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return _write_unavailable(out_dir, status, thresholds)

    data_cfg = cfg.get("data", {})
    size = int(data_cfg.get("image_size", 224))
    text_n = int(data_cfg.get("text_n_per_class", 12))
    wm_n = int(data_cfg.get("watermark_n_per_class", 12))
    wm_seed = int(data_cfg.get("watermark_benchmark_seed", 5151))

    # Frozen generation policy (reused verbatim; not reselected/tuned here).
    frozen_policy_path = Path(cfg.get("frozen_policy_dir", "results/hard_multidecoy_clip_repair")) / "selected_generation_policy.json"
    if frozen_policy_path.exists():
        loaded = json.loads(frozen_policy_path.read_text(encoding="utf-8"))
        text_policy = dict(DEFAULT_TEXT_GENERATION_POLICY) if loaded.get("unavailable") else loaded
        text_source = "regenerated_from_frozen_generation_policy" if not loaded.get("unavailable") else "regenerated_from_default_policy"
    else:
        text_policy = dict(DEFAULT_TEXT_GENERATION_POLICY)
        text_source = "regenerated_from_default_policy"

    # Class text embeddings v_y for the class prompts used everywhere.
    prompts = [PROMPT_TEMPLATE.format(label=name) for name in CLIP_OVERLAY_CLASSES]
    class_text_feats = np.asarray(encode_text_prompts(status, prompts, device).detach().cpu().numpy(), dtype=np.float64)

    text_examples = build_text_examples(text_policy, text_n, size)
    wm_examples = build_watermark_examples(wm_n, size, wm_seed)

    text_metrics, text_df, text_delta, text_sc_vals, text_obj_vals = compute_family_metrics(
        "text_overlay", TEXT_SHORTCUT_TYPE, text_examples, status, device, class_text_feats, thresholds, np.random.default_rng(seed + 11)
    )
    wm_metrics, wm_df, wm_delta, wm_sc_vals, wm_obj_vals = compute_family_metrics(
        "watermark", WATERMARK_SHORTCUT_TYPE, wm_examples, status, device, class_text_feats, thresholds, np.random.default_rng(seed + 23)
    )

    text_supported, text_checks, text_reasons = _additivity_supported(text_metrics, status, thresholds)
    wm_supported, wm_checks, wm_reasons = _additivity_supported(wm_metrics, status, thresholds)

    # Watermark weak/flat shortcut channel diagnostic.
    weak_effect_fraction = float(thresholds.get("watermark_weak_effect_fraction", 0.50))
    wm_within = wm_metrics["within_shortcut_mean_pairwise_cosine"]
    wm_shuffled = wm_metrics["shuffled_mean_pairwise_cosine"]
    cluster_margin = float(thresholds.get("cluster_cosine_margin", 0.10))
    wm_no_cluster = not (np.isfinite(wm_within) and np.isfinite(wm_shuffled) and (wm_within - wm_shuffled) >= cluster_margin)
    wm_small_effect = bool(
        np.isfinite(wm_metrics["mean_shortcut_effect_norm"])
        and np.isfinite(text_metrics["mean_shortcut_effect_norm"])
        and wm_metrics["mean_shortcut_effect_norm"] < weak_effect_fraction * text_metrics["mean_shortcut_effect_norm"]
    )
    watermark_channel_weak = bool(wm_no_cluster or wm_small_effect)

    if text_supported:
        framing = "empirically_supported_for_text_overlays"
    else:
        framing = "conditional_theory_only"

    key_numbers: dict[str, Any] = {
        "embedding_additivity_supported_for_text": bool(text_supported),
        "embedding_additivity_supported_for_watermark": bool(wm_supported),
        "pretrained_clip_loaded": bool(status.pretrained),
        "fake_backend": status.backend == "fake",
        "backend": status.backend,
        "model_name": status.model_name,
        "pretrained_tag": status.pretrained_tag,
        "evidence_status": "pretrained CLIP embedding-additivity validation evidence",
        "theorem_framing": framing,
        "text_source": text_source,
        "text_generation_policy_id": str(text_policy.get("policy_id", "")),
        "watermark_source": "regenerated_from_render_cross_shortcut_image",
        "watermark_shortcut_type": WATERMARK_SHORTCUT_TYPE,
        "n_text_examples": int(len(text_examples)),
        "n_watermark_examples": int(len(wm_examples)),
        "thresholds": thresholds,
        # Text metrics (headline)
        "text_within_shortcut_cosine": text_metrics["within_shortcut_mean_pairwise_cosine"],
        "text_within_object_cosine": text_metrics["within_object_mean_pairwise_cosine"],
        "text_shuffled_cosine": text_metrics["shuffled_mean_pairwise_cosine"],
        "text_delta_cosine_minus_shuffled": text_metrics["delta_cosine_minus_shuffled"],
        "text_nearest_centroid_accuracy_shortcut": text_metrics["nearest_centroid_accuracy_shortcut"],
        "text_nearest_centroid_accuracy_object": text_metrics["nearest_centroid_accuracy_object"],
        "text_logit_consistency_mae": text_metrics["logit_consistency_mae"],
        "text_logit_consistency_corr": text_metrics["logit_consistency_corr"],
        "text_mean_neutralization_ratio": text_metrics["mean_neutralization_ratio"],
        "text_mean_shortcut_effect_norm": text_metrics["mean_shortcut_effect_norm"],
        "text_mean_clean_damage_proxy": text_metrics["mean_clean_damage_proxy"],
        "text_repair_success_rate": text_metrics["repair_success_rate"],
        "text_repair_success_when_margin_satisfied": text_metrics["repair_success_when_margin_satisfied"],
        "text_repair_success_when_margin_violated": text_metrics["repair_success_when_margin_violated"],
        "text_frac_margin_satisfied": text_metrics["frac_margin_condition_satisfied"],
        "text_delta_clusters_above_shuffled": text_checks["delta_clusters_above_shuffled"],
        "text_shortcut_clustering_exceeds_object": text_checks["shortcut_clustering_exceeds_object"],
        "text_neutralization_damage_small": text_checks["neutralization_damage_small"],
        "text_margin_condition_predicts_repair": text_checks["margin_condition_predicts_repair"],
        "text_eligibility_checks": text_checks,
        "text_failed_reasons": text_reasons,
        # Watermark metrics (negative-result diagnostic)
        "watermark_within_shortcut_cosine": wm_metrics["within_shortcut_mean_pairwise_cosine"],
        "watermark_within_object_cosine": wm_metrics["within_object_mean_pairwise_cosine"],
        "watermark_shuffled_cosine": wm_metrics["shuffled_mean_pairwise_cosine"],
        "watermark_delta_cosine_minus_shuffled": wm_metrics["delta_cosine_minus_shuffled"],
        "watermark_nearest_centroid_accuracy_shortcut": wm_metrics["nearest_centroid_accuracy_shortcut"],
        "watermark_nearest_centroid_accuracy_object": wm_metrics["nearest_centroid_accuracy_object"],
        "watermark_mean_shortcut_effect_norm": wm_metrics["mean_shortcut_effect_norm"],
        "watermark_mean_neutralization_ratio": wm_metrics["mean_neutralization_ratio"],
        "watermark_repair_success_rate": wm_metrics["repair_success_rate"],
        "watermark_shortcut_channel_weak": watermark_channel_weak,
        "watermark_eligibility_checks": wm_checks,
        "watermark_failed_reasons": wm_reasons,
        "watermark_weak_channel_statement": WATERMARK_WEAK_STATEMENT if watermark_channel_weak else "",
    }

    pd.DataFrame([text_metrics, wm_metrics]).to_csv(out_dir / "embedding_additivity_metrics.csv", index=False)
    pd.concat([text_df, wm_df], ignore_index=True).to_csv(out_dir / "embedding_additivity_examples.csv", index=False)
    (out_dir / "embedding_additivity_key_numbers.json").write_text(json.dumps(key_numbers, indent=2, sort_keys=True, default=str), encoding="utf-8")

    _plot_family(text_delta, text_sc_vals, text_obj_vals, "Text-overlay shortcut delta (clustering by shortcut value)", out_dir / "embedding_additivity_text_delta_plot.png", out_dir / "embedding_additivity_text_delta_plot.pdf")
    _plot_family(wm_delta, wm_sc_vals, wm_obj_vals, "Colored-symbol watermark shortcut delta", out_dir / "embedding_additivity_watermark_delta_plot.png", out_dir / "embedding_additivity_watermark_delta_plot.pdf")
    _write_summary(out_dir, key_numbers, text_metrics, wm_metrics, status)

    pd.DataFrame([{"total_time_sec": time.perf_counter() - total_start, "seed": seed}]).to_csv(out_dir / "embedding_additivity_timing_profile.csv", index=False)
    return {
        "summary": str(out_dir / "embedding_additivity_summary.md"),
        "metrics": str(out_dir / "embedding_additivity_metrics.csv"),
        "examples": str(out_dir / "embedding_additivity_examples.csv"),
        "key_numbers": str(out_dir / "embedding_additivity_key_numbers.json"),
        "caption": str(out_dir / "embedding_additivity_caption.md"),
        "text_delta_plot_png": str(out_dir / "embedding_additivity_text_delta_plot.png"),
        "text_delta_plot_pdf": str(out_dir / "embedding_additivity_text_delta_plot.pdf"),
        "watermark_delta_plot_png": str(out_dir / "embedding_additivity_watermark_delta_plot.png"),
        "watermark_delta_plot_pdf": str(out_dir / "embedding_additivity_watermark_delta_plot.pdf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/embedding_additivity_validation.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
