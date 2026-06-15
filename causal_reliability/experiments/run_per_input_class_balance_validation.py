"""Per-input class-balance validation for the finite-candidate CIC recovery theorem.

This is the **final theory-gating experiment**. The earlier embedding-additivity
validation (`run_embedding_additivity_validation.py`) tested a *sufficient* but
*stronger-than-necessary* condition: a single global, input-independent shortcut
direction (``u(object + shortcut) - u(object) ~= delta_S`` for all objects). That
strong condition was **not** supported for OpenCLIP text overlays -- the per-image
overlay delta clusters more tightly by object than by shortcut value
(within-object cosine ~0.86 > within-shortcut cosine ~0.76), and the multi-decoy
neutralization L2 damage is not small (ratio ~0.92).

The recovery theorem, however, only needs a **weaker per-input condition**: after
neutralization, the *residual* shortcut contribution to the logits should be
approximately **class-independent for each individual image**. A class-independent
logit offset cannot change the argmax, so if the residual class-dependent part is
smaller than half the clean causal margin, the repaired prediction equals the
clean causal argmax (see the per-input corollary in ``docs/theory.md``).

This experiment tests that weaker premise directly on the hard multi-decoy
OpenCLIP text-overlay repair result, using real pretrained OpenCLIP
(``open_clip`` / ``ViT-B-32`` / ``laion2b_s34b_b79k``; no fake backend for theory
evidence). For each misleading-overlay image it compares the per-input
logit-shift balance of:

* **A. oracle** harmful-text neutralization (true shortcut removal),
* **B. CIC top-1** non-oracle region neutralization,
* **C. CIC top-3 consensus** neutralization,
* **D. matched random** text-region neutralization (control), and
* **E. watermark** oracle neutralization (negative shortcut family, for contrast).

Nothing is tuned on the observed metrics; the repair policy and generation policy
are reused frozen. The decision thresholds in PART 6 are fixed a-priori in the
config. The booleans ``per_input_class_balance_supported_for_text`` and
``clip_theory_support_status`` record the outcome and gate the theory framing.
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
from causal_reliability.discovery.nonoracle_clip_discovery import discover_clip_shortcut_regions
from causal_reliability.experiments.run_clip_overlay_validation import _downloads_allowed
from causal_reliability.experiments.run_cross_shortcut_generalization import (
    MIS_REGIME as WM_MIS_REGIME,
    NO_OVERLAY_REGIME as WM_NO_OVERLAY_REGIME,
    SHORTCUT_TYPE as WATERMARK_SHORTCUT_TYPE,
    render_cross_shortcut_image,
)
from causal_reliability.experiments.run_embedding_additivity_validation import (
    DEFAULT_TEXT_GENERATION_POLICY,
    TEXT_MIS_REGIME,
    TEXT_NO_OVERLAY_REGIME,
    TEXT_SHORTCUT_TYPE,
)
from causal_reliability.experiments.run_hard_multidecoy_clip_repair import render_hard_multidecoy_image
from causal_reliability.experiments.run_nonoracle_clip_repair import (
    PROMPT_TEMPLATE,
    _device,
    _pil_to_tensor,
    _predict_pil,
    _select_matched_random,
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


EPSILON = 1e-9

# Neutralization condition keys (PART 4).
COND_ORACLE = "oracle"
COND_CIC_TOP1 = "cic_top1"
COND_CIC_TOP3 = "cic_top3_consensus"
COND_RANDOM = "random_matched_text_region"
COND_WATERMARK = "watermark_oracle"
TEXT_CONDITIONS = [COND_ORACLE, COND_CIC_TOP1, COND_CIC_TOP3, COND_RANDOM]


# ---------------------------------------------------------------------------
# Logit helpers
# ---------------------------------------------------------------------------
def _logits(model: ClipZeroShotClassifier, images: list[Image.Image]) -> np.ndarray:
    """Per-class CLIP logits ``logit_y(X) = 100 * <u(X), v_y>`` for a list of PIL images."""
    if not images:
        return np.zeros((0, len(model.class_names)), dtype=np.float64)
    out = model.predict(_pil_to_tensor([im.convert("RGB") for im in images]))
    return np.asarray(out["logits"].detach().cpu().numpy(), dtype=np.float64)


def _as_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.asarray(arr).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")


# ---------------------------------------------------------------------------
# Per-input class-balance metrics (PART 2 + PART 3)
# ---------------------------------------------------------------------------
def per_input_balance(
    clean_logits: np.ndarray,
    shortcut_logits: np.ndarray,
    neutralized_logits: np.ndarray,
    label: int,
    epsilon_b: float,
) -> dict[str, Any]:
    """Per-input logit-shift balance and margin diagnostic for one neutralization.

    ``delta_neutralize_y = logit_y(neutralized) - logit_y(shortcut)``  (PART 2)
    ``delta_to_clean_y   = logit_y(neutralized) - logit_y(clean)``     (PART 3)

    The recovery-relevant balance is the *residual perturbation relative to the
    clean image* (``delta_to_clean``): if it is class-independent the neutralized
    argmax equals the clean argmax. The shift relative to the shortcut image
    (``delta_neutralize``) is reported as the PART 2 diagnostic.
    """
    delta_neutralize = neutralized_logits - shortcut_logits
    shift_mean = float(np.mean(delta_neutralize))
    shift_std = float(np.std(delta_neutralize))
    shift_range = float(np.max(delta_neutralize) - np.min(delta_neutralize))
    shift_cv = float(shift_std / (abs(shift_mean) + epsilon_b * 0.0 + EPSILON))
    max_centered_shift = float(np.max(np.abs(delta_neutralize - shift_mean)))

    delta_to_clean = neutralized_logits - clean_logits
    residual_mean = float(np.mean(delta_to_clean))
    # Residual class-dependent perturbation relative to the clean image: the
    # quantity that must be small for per-input recovery (this is the per-input
    # class-balance error epsilon_B of the corollary in docs/theory.md).
    residual = float(np.max(np.abs(delta_to_clean - residual_mean)))
    residual_std = float(np.std(delta_to_clean))

    order = np.argsort(clean_logits)[::-1]
    margin_clean = float(clean_logits[order[0]] - clean_logits[order[1]]) if len(clean_logits) >= 2 else float("nan")
    margin_condition_satisfied = bool(np.isfinite(margin_clean) and margin_clean > 2.0 * residual)
    # Per-input class-balance flag: residual class-dependent shift within epsilon_B.
    class_balance_satisfied = bool(residual <= epsilon_b)

    repair_success = bool(int(np.argmax(neutralized_logits)) == int(label))
    original_shortcut_correct = bool(int(np.argmax(shortcut_logits)) == int(label))
    clean_correct = bool(int(np.argmax(clean_logits)) == int(label))
    return {
        "shift_mean": shift_mean,
        "shift_std": shift_std,
        "shift_range": shift_range,
        "shift_cv": shift_cv,
        "max_centered_shift": max_centered_shift,
        "residual_to_clean": residual,
        "residual_to_clean_std": residual_std,
        "margin_clean": margin_clean,
        "margin_condition_satisfied": margin_condition_satisfied,
        "class_balance_satisfied": class_balance_satisfied,
        "repair_success": repair_success,
        "original_shortcut_correct": original_shortcut_correct,
        "clean_correct": clean_correct,
    }


# ---------------------------------------------------------------------------
# Example construction (regenerated from frozen policies; nothing tuned)
# ---------------------------------------------------------------------------
def _load_text_policy(cfg: dict[str, Any]) -> tuple[dict[str, Any], str]:
    frozen_policy_path = Path(cfg.get("frozen_policy_dir", "results/hard_multidecoy_clip_repair")) / "selected_generation_policy.json"
    if frozen_policy_path.exists():
        loaded = json.loads(frozen_policy_path.read_text(encoding="utf-8"))
        if loaded.get("unavailable"):
            return dict(DEFAULT_TEXT_GENERATION_POLICY), "regenerated_from_default_policy"
        return loaded, "regenerated_from_frozen_generation_policy"
    return dict(DEFAULT_TEXT_GENERATION_POLICY), "regenerated_from_default_policy"


def build_text_misleading_examples(policy: dict[str, Any], n_per_class: int, size: int) -> list[dict[str, Any]]:
    """Misleading text-overlay examples paired with their clean (no-overlay) image."""
    examples: list[dict[str, Any]] = []
    eid = 0
    n_classes = min(int(policy.get("class_set_size", len(CLIP_OVERLAY_CLASSES))), len(CLIP_OVERLAY_CLASSES))
    for label in range(n_classes):
        for index in range(n_per_class):
            clean, _ = render_hard_multidecoy_image(label, TEXT_NO_OVERLAY_REGIME, index, policy, size=size)
            shortcut, mis_meta = render_hard_multidecoy_image(label, TEXT_MIS_REGIME, index, policy, size=size)
            harmful_bbox = mis_meta.get("harmful_bbox") or []
            examples.append(
                {
                    "family": "text_overlay",
                    "example_id": eid,
                    "label": int(label),
                    "true_label": CLIP_OVERLAY_CLASSES[label],
                    "harmful_text": str(mis_meta.get("harmful_text", "")),
                    "harmful_bbox": list(harmful_bbox),
                    "x_clean": clean,
                    "x_shortcut": shortcut,
                }
            )
            eid += 1
    return examples


def build_watermark_misleading_examples(n_per_class: int, size: int, benchmark_seed: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    eid = 0
    for label in range(len(CLIP_OVERLAY_CLASSES)):
        for index in range(n_per_class):
            clean, _ = render_cross_shortcut_image(label, WM_NO_OVERLAY_REGIME, index, size=size, benchmark_seed=benchmark_seed)
            shortcut, mis_meta = render_cross_shortcut_image(label, WM_MIS_REGIME, index, size=size, benchmark_seed=benchmark_seed)
            harmful_bbox = mis_meta.get("harmful_bbox") or []
            examples.append(
                {
                    "family": "watermark",
                    "example_id": eid,
                    "label": int(label),
                    "true_label": CLIP_OVERLAY_CLASSES[label],
                    "harmful_bbox": list(harmful_bbox),
                    "x_clean": clean,
                    "x_shortcut": shortcut,
                }
            )
            eid += 1
    return examples


# ---------------------------------------------------------------------------
# Per-example evaluation: build neutralized views and compute balance metrics
# ---------------------------------------------------------------------------
def evaluate_text_example(
    ex: dict[str, Any],
    model: ClipZeroShotClassifier,
    prompts: list[str],
    *,
    seed: int,
    max_candidates: int,
    epsilon_b: float,
) -> list[dict[str, Any]]:
    """Return one record per (example, condition) for a misleading text example."""
    label = int(ex["label"])
    clean_pil = _as_pil(ex["x_clean"])
    shortcut_pil = _as_pil(ex["x_shortcut"])
    predict_fn = lambda imgs: _predict_pil(model, imgs)

    clean_logits = _logits(model, [clean_pil])[0]
    shortcut_logits = _logits(model, [shortcut_pil])[0]

    neutralized_pils: dict[str, Image.Image | None] = {}
    selected_bbox: dict[str, list[int] | None] = {}
    consensus_logits: np.ndarray | None = None

    # A. oracle harmful-text neutralization.
    harmful_bbox = ex.get("harmful_bbox") or []
    if harmful_bbox:
        neutralized_pils[COND_ORACLE] = neutralize_region(shortcut_pil, tuple(int(v) for v in harmful_bbox))
        selected_bbox[COND_ORACLE] = [int(v) for v in harmful_bbox]
    else:
        neutralized_pils[COND_ORACLE] = None
        selected_bbox[COND_ORACLE] = None

    # B/C/D. CIC discovery on the misleading image (non-oracle; no labels/metadata).
    _, scores, _ = discover_clip_shortcut_regions(
        shortcut_pil, predict_fn, prompts, seed=seed + int(ex["example_id"]), max_candidates=max_candidates
    )
    top1 = scores[0] if scores else None
    top3 = scores[:3]
    if top1 is not None:
        neutralized_pils[COND_CIC_TOP1] = neutralize_region(shortcut_pil, top1.bbox)
        selected_bbox[COND_CIC_TOP1] = [int(v) for v in top1.bbox]
    else:
        neutralized_pils[COND_CIC_TOP1] = None
        selected_bbox[COND_CIC_TOP1] = None

    # C. CIC top-3 consensus -> mean logits over the top-3 neutralized images.
    if top3:
        top3_pils = [neutralize_region(shortcut_pil, s.bbox) for s in top3]
        consensus_logits = _logits(model, top3_pils).mean(axis=0)
        selected_bbox[COND_CIC_TOP3] = [int(v) for v in top3[0].bbox]
    else:
        selected_bbox[COND_CIC_TOP3] = None

    # D. matched random text-region neutralization control (matched on textness).
    random_score = _select_matched_random(scores, top1, "textness_score") if scores else None
    if random_score is not None and getattr(random_score, "proposal_type", "") == "random_patch_control":
        neutralized_pils[COND_RANDOM] = neutralize_region(shortcut_pil, random_score.bbox)
        selected_bbox[COND_RANDOM] = [int(v) for v in random_score.bbox]
    else:
        neutralized_pils[COND_RANDOM] = None
        selected_bbox[COND_RANDOM] = None

    records: list[dict[str, Any]] = []
    for cond in TEXT_CONDITIONS:
        if cond == COND_CIC_TOP3:
            neut_logits = consensus_logits
        else:
            pil = neutralized_pils.get(cond)
            neut_logits = _logits(model, [pil])[0] if pil is not None else None
        if neut_logits is None:
            continue
        metrics = per_input_balance(clean_logits, shortcut_logits, neut_logits, label, epsilon_b)
        records.append(
            {
                "family": "text_overlay",
                "condition": cond,
                "example_id": int(ex["example_id"]),
                "label": label,
                "true_label": ex["true_label"],
                "harmful_text": ex.get("harmful_text", ""),
                "selected_bbox": json.dumps(selected_bbox.get(cond)),
                **metrics,
            }
        )
    return records


def evaluate_watermark_example(
    ex: dict[str, Any],
    model: ClipZeroShotClassifier,
    *,
    epsilon_b: float,
) -> list[dict[str, Any]]:
    label = int(ex["label"])
    clean_logits = _logits(model, [_as_pil(ex["x_clean"])])[0]
    shortcut_pil = _as_pil(ex["x_shortcut"])
    shortcut_logits = _logits(model, [shortcut_pil])[0]
    harmful_bbox = ex.get("harmful_bbox") or []
    if not harmful_bbox:
        return []
    neut_pil = neutralize_region(shortcut_pil, tuple(int(v) for v in harmful_bbox))
    neut_logits = _logits(model, [neut_pil])[0]
    metrics = per_input_balance(clean_logits, shortcut_logits, neut_logits, label, epsilon_b)
    return [
        {
            "family": "watermark",
            "condition": COND_WATERMARK,
            "example_id": int(ex["example_id"]),
            "label": label,
            "true_label": ex["true_label"],
            "harmful_text": "",
            "selected_bbox": json.dumps([int(v) for v in harmful_bbox]),
            **metrics,
        }
    ]


# ---------------------------------------------------------------------------
# Aggregation (PART 5)
# ---------------------------------------------------------------------------
def _safe(series: pd.Series, fn) -> float:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(fn(vals)) if len(vals) else float("nan")


def aggregate_condition(df: pd.DataFrame, condition: str) -> dict[str, Any]:
    sub = df[df["condition"] == condition]
    n = int(len(sub))
    repaired = sub[sub["repair_success"].astype(bool)] if n else sub
    failed = sub[~sub["repair_success"].astype(bool)] if n else sub
    sat = sub[sub["margin_condition_satisfied"].astype(bool)] if n else sub
    vio = sub[~sub["margin_condition_satisfied"].astype(bool)] if n else sub
    row: dict[str, Any] = {
        "condition": condition,
        "n_examples": n,
        # PART 2: shift (neutralized - shortcut) balance diagnostics.
        "mean_shift_std": _safe(sub["shift_std"], np.mean) if n else float("nan"),
        "median_shift_std": _safe(sub["shift_std"], np.median) if n else float("nan"),
        "mean_shift_range": _safe(sub["shift_range"], np.mean) if n else float("nan"),
        "median_shift_range": _safe(sub["shift_range"], np.median) if n else float("nan"),
        "mean_max_centered_shift": _safe(sub["max_centered_shift"], np.mean) if n else float("nan"),
        "median_max_centered_shift": _safe(sub["max_centered_shift"], np.median) if n else float("nan"),
        "mean_shift_cv": _safe(sub["shift_cv"], np.mean) if n else float("nan"),
        # PART 3: residual-to-clean balance (the recovery-relevant per-input class-balance error).
        "mean_residual_to_clean": _safe(sub["residual_to_clean"], np.mean) if n else float("nan"),
        "median_residual_to_clean": _safe(sub["residual_to_clean"], np.median) if n else float("nan"),
        "mean_margin_clean": _safe(sub["margin_clean"], np.mean) if n else float("nan"),
        # Class-balance / margin / repair rates.
        "pct_class_balance_satisfied": float(sub["class_balance_satisfied"].astype(bool).mean()) if n else float("nan"),
        "margin_condition_satisfaction_rate": float(sub["margin_condition_satisfied"].astype(bool).mean()) if n else float("nan"),
        "repair_accuracy": float(sub["repair_success"].astype(bool).mean()) if n else float("nan"),
        "n_margin_satisfied": int(len(sat)),
        "n_margin_violated": int(len(vio)),
        "repair_success_when_margin_satisfied": float(sat["repair_success"].astype(bool).mean()) if len(sat) else float("nan"),
        "repair_success_when_margin_violated": float(vio["repair_success"].astype(bool).mean()) if len(vio) else float("nan"),
        # Failure-case diagnostics (PART 6: failures should be worse-balanced).
        "n_repaired": int(len(repaired)),
        "n_failed": int(len(failed)),
        "mean_residual_repaired": _safe(repaired["residual_to_clean"], np.mean) if len(repaired) else float("nan"),
        "mean_residual_failed": _safe(failed["residual_to_clean"], np.mean) if len(failed) else float("nan"),
        "margin_satisfaction_repaired": float(repaired["margin_condition_satisfied"].astype(bool).mean()) if len(repaired) else float("nan"),
        "margin_satisfaction_failed": float(failed["margin_condition_satisfied"].astype(bool).mean()) if len(failed) else float("nan"),
    }
    return row


# ---------------------------------------------------------------------------
# Decision logic (PART 6) -- a-priori thresholds, not tuned on the metrics
# ---------------------------------------------------------------------------
def decide_support(
    agg: dict[str, dict[str, Any]],
    status: ClipStatus,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    balance_advantage_fraction = float(thresholds.get("balance_advantage_fraction", 0.75))
    margin_success_min = float(thresholds.get("margin_success_min", 0.50))
    repaired_balance_fraction_min = float(thresholds.get("repaired_balance_fraction_min", 0.50))

    random_residual = agg.get(COND_RANDOM, {}).get("median_residual_to_clean", float("nan"))

    def _margin_predicts(row: dict[str, Any]) -> bool:
        sat = row.get("repair_success_when_margin_satisfied", float("nan"))
        vio = row.get("repair_success_when_margin_violated", float("nan"))
        if not np.isfinite(sat):
            return False
        if not np.isfinite(vio):
            return bool(sat >= margin_success_min)
        return bool(sat >= vio and sat >= margin_success_min)

    def _more_balanced(row: dict[str, Any]) -> bool:
        targeted = row.get("median_residual_to_clean", float("nan"))
        if not (np.isfinite(targeted) and np.isfinite(random_residual)):
            return False
        return bool(targeted <= balance_advantage_fraction * random_residual)

    def _repaired_balanced(row: dict[str, Any]) -> bool:
        frac = row.get("margin_satisfaction_repaired", float("nan"))
        return bool(np.isfinite(frac) and frac >= repaired_balance_fraction_min)

    def _failures_worse(row: dict[str, Any]) -> bool:
        # Vacuously satisfied when there are no failures.
        if int(row.get("n_failed", 0)) == 0:
            return True
        rep_res = row.get("mean_residual_repaired", float("nan"))
        fail_res = row.get("mean_residual_failed", float("nan"))
        rep_marg = row.get("margin_satisfaction_repaired", float("nan"))
        fail_marg = row.get("margin_satisfaction_failed", float("nan"))
        worse_residual = bool(np.isfinite(rep_res) and np.isfinite(fail_res) and fail_res >= rep_res)
        worse_margin = bool(np.isfinite(rep_marg) and np.isfinite(fail_marg) and fail_marg <= rep_marg)
        return bool(worse_residual or worse_margin)

    per_condition: dict[str, dict[str, bool]] = {}
    for cond in (COND_ORACLE, COND_CIC_TOP1, COND_CIC_TOP3):
        row = agg.get(cond, {})
        if not row or int(row.get("n_examples", 0)) == 0:
            continue
        per_condition[cond] = {
            "more_balanced_than_random": _more_balanced(row),
            "margin_predicts_repair": _margin_predicts(row),
            "repaired_fraction_balanced": _repaired_balanced(row),
            "failures_worse_balanced": _failures_worse(row),
        }

    pretrained_ok = bool(status.pretrained and status.backend in {"open_clip", "transformers"})
    any_more_balanced = any(c["more_balanced_than_random"] for c in per_condition.values())
    full_support = any(
        c["more_balanced_than_random"]
        and c["margin_predicts_repair"]
        and c["repaired_fraction_balanced"]
        and c["failures_worse_balanced"]
        for c in per_condition.values()
    )
    partial_support = any(
        c["more_balanced_than_random"] and (c["margin_predicts_repair"] or c["repaired_fraction_balanced"])
        for c in per_condition.values()
    )

    if not pretrained_ok:
        supported: Any = False
        status_str = "conditional only; per-input class-balance not supported"
    elif full_support:
        supported = True
        status_str = "CLIP-supported via per-input class-balance"
    elif any_more_balanced and partial_support:
        supported = "mixed"
        status_str = "mixed; per-input class-balance partially supported"
    else:
        supported = False
        status_str = "conditional only; per-input class-balance not supported"

    return {
        "per_input_class_balance_supported_for_text": supported,
        "clip_theory_support_status": status_str,
        "per_condition_checks": per_condition,
        "pretrained_ok": pretrained_ok,
        "any_more_balanced_than_random": bool(any_more_balanced),
        "random_median_residual_to_clean": random_residual,
    }


# ---------------------------------------------------------------------------
# Plot (PART 5)
# ---------------------------------------------------------------------------
def _plot(agg: dict[str, dict[str, Any]], png: Path, pdf: Path) -> None:
    conditions = [c for c in (COND_ORACLE, COND_CIC_TOP1, COND_CIC_TOP3, COND_RANDOM, COND_WATERMARK) if c in agg and int(agg[c].get("n_examples", 0)) > 0]
    plt.figure(figsize=(8.8, 4.8))
    if conditions:
        x = np.arange(len(conditions))
        residual = [agg[c]["median_residual_to_clean"] for c in conditions]
        margin = [agg[c]["margin_condition_satisfaction_rate"] for c in conditions]
        repair = [agg[c]["repair_accuracy"] for c in conditions]
        ax1 = plt.gca()
        ax1.bar(x - 0.25, residual, width=0.25, color="#d62728", label="median residual-to-clean (logits)")
        ax1.set_ylabel("median residual-to-clean (logit units)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(conditions, rotation=20, ha="right", fontsize=8)
        ax2 = ax1.twinx()
        ax2.bar(x, margin, width=0.25, color="#2ca02c", label="margin-condition rate")
        ax2.bar(x + 0.25, repair, width=0.25, color="#1f77b4", label="repair accuracy")
        ax2.set_ylabel("rate")
        ax2.set_ylim(0, 1.05)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper center")
        plt.title("Per-input class-balance by neutralization condition", fontsize=10)
    else:
        plt.text(0.5, 0.5, "pretrained CLIP unavailable", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(png, dpi=170)
    plt.savefig(pdf)
    plt.close()


# ---------------------------------------------------------------------------
# Summary / caption (PART 5)
# ---------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


OBJECT_ENTANGLEMENT_STATEMENT = (
    "OpenCLIP's typographic shortcut effect is not a single global additive bias direction. The shift induced by "
    "overlay text is object-entangled: it contains a real shortcut component, but its direction varies substantially "
    "with the underlying object. This helps explain why generic global debiasing is unlikely to suffice, and why "
    "targeted per-input counterfactual region scoring can still repair failures."
)

THEOREM_CAVEAT = (
    "This is a finite-candidate, controlled validation of a weaker per-input premise. It does not establish open-world "
    "shortcut discovery, exact bounding-box localization, or general robustness."
)


def _condition_block(agg: dict[str, dict[str, Any]], cond: str, title: str) -> list[str]:
    row = agg.get(cond, {})
    if not row or int(row.get("n_examples", 0)) == 0:
        return [f"### {title}", "", "Not available (no examples for this condition).", ""]
    return [
        f"### {title}",
        "",
        f"- n examples: {int(row['n_examples'])}",
        f"- mean / median shift_std: {_fmt(row['mean_shift_std'])} / {_fmt(row['median_shift_std'])}",
        f"- mean / median shift_range: {_fmt(row['mean_shift_range'])} / {_fmt(row['median_shift_range'])}",
        f"- mean / median max_centered_shift: {_fmt(row['mean_max_centered_shift'])} / {_fmt(row['median_max_centered_shift'])}",
        f"- mean / median residual-to-clean (recovery-relevant class-dependent residual): {_fmt(row['mean_residual_to_clean'])} / {_fmt(row['median_residual_to_clean'])}",
        f"- % examples satisfying class-balance threshold (residual <= epsilon_B): {_fmt(row['pct_class_balance_satisfied'])}",
        f"- margin-condition satisfaction rate (m_clean > 2*residual): {_fmt(row['margin_condition_satisfaction_rate'])}",
        f"- repair accuracy: {_fmt(row['repair_accuracy'])}",
        f"- repair success | margin satisfied vs violated: {_fmt(row['repair_success_when_margin_satisfied'])} "
        f"(n={int(row['n_margin_satisfied'])}) vs {_fmt(row['repair_success_when_margin_violated'])} (n={int(row['n_margin_violated'])})",
        f"- mean residual repaired vs failed: {_fmt(row['mean_residual_repaired'])} vs {_fmt(row['mean_residual_failed'])}",
        "",
    ]


def _write_summary(out_dir: Path, key_numbers: dict[str, Any], agg: dict[str, dict[str, Any]], status: ClipStatus, epsilon_b: float) -> None:
    supported = key_numbers["per_input_class_balance_supported_for_text"]
    status_str = key_numbers["clip_theory_support_status"]
    lines = [
        "# Per-Input Class-Balance Validation (final theory gate)",
        "",
        "The embedding-additivity validation tested a *stronger-than-necessary* condition (a single global, "
        "input-independent shortcut direction) and did not support it for OpenCLIP text overlays. The recovery theorem "
        "only requires a **weaker per-input condition**: after neutralization, the residual shortcut contribution to the "
        "logits should be approximately **class-independent for each individual image**. A class-independent logit "
        "offset cannot change the argmax, so if the residual class-dependent part is below half the clean causal "
        "margin, the repaired prediction equals the clean causal argmax. This experiment tests that weaker premise "
        "directly on the hard multi-decoy OpenCLIP text-overlay repair result.",
        "",
        f"Backend: {status.backend}. Model: {status.model_name}. Pretrained loaded: `{status.pretrained}`. "
        f"Fake backend: `{status.backend == 'fake'}`. Class-balance threshold epsilon_B = {epsilon_b} logits.",
        "",
        f"**per_input_class_balance_supported_for_text = `{supported}`**",
        "",
        f"**clip_theory_support_status = `{status_str}`**",
        "",
        "Per-input metrics, for each example and class y:",
        "",
        "- `delta_neutralize_y = logit_y(x_neutralized) - logit_y(x_shortcut)` (PART 2 shift diagnostic), summarized by "
        "`shift_std`, `shift_range`, `max_centered_shift = max_y |delta - mean_y delta|`.",
        "- `delta_to_clean_y = logit_y(x_neutralized) - logit_y(x_clean)`; the recovery-relevant per-input class-balance "
        "error is `residual = max_y |delta_to_clean_y - mean_y delta_to_clean|` (PART 3). Small residual relative to the "
        "clean margin means the neutralized shift is approximately class-independent for that input.",
        "- `margin_condition_satisfied = margin_clean > 2 * residual`, compared against repair success.",
        "",
        "## Class-balance by neutralization condition (PART 4/5)",
        "",
        *_condition_block(agg, COND_ORACLE, "A. Oracle harmful-text neutralization"),
        *_condition_block(agg, COND_CIC_TOP1, "B. CIC top-1 neutralization"),
        *_condition_block(agg, COND_CIC_TOP3, "C. CIC top-3 consensus neutralization"),
        *_condition_block(agg, COND_RANDOM, "D. Matched random text-region neutralization (control)"),
        *_condition_block(agg, COND_WATERMARK, "E. Watermark oracle neutralization (negative family)"),
        "## Decision (PART 6)",
        "",
        f"- pretrained CLIP loaded: `{key_numbers['pretrained_clip_loaded']}`; fake backend: `{key_numbers['fake_backend']}`",
        f"- oracle/CIC more class-balanced than random: `{key_numbers['any_more_balanced_than_random']}` "
        f"(random median residual-to-clean = {_fmt(key_numbers['random_median_residual_to_clean'])})",
        f"- per_input_class_balance_supported_for_text = `{supported}`",
        f"- clip_theory_support_status = `{status_str}`",
        "",
        "Per-condition a-priori checks (more_balanced_than_random / margin_predicts_repair / repaired_fraction_balanced "
        "/ failures_worse_balanced):",
        "",
        *[
            f"- {cond}: " + ", ".join(f"{k}=`{v}`" for k, v in checks.items())
            for cond, checks in key_numbers.get("per_condition_checks", {}).items()
        ],
        "",
        "## Object-entangled typographic shortcut effects",
        "",
        OBJECT_ENTANGLEMENT_STATEMENT,
        "",
        "## Scope and caveats",
        "",
        THEOREM_CAVEAT,
        "",
        "This experiment does not change the repair policy, does not tune any threshold on the observed metrics, and "
        "does not claim open-world shortcut discovery, exact localization, or general robustness.",
    ]
    (out_dir / "per_input_class_balance_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "per_input_class_balance_caption.md").write_text(
        "# Per-Input Class-Balance Figure Caption\n\n"
        "Per-input logit-shift class-balance for the hard multi-decoy OpenCLIP text-overlay repair, by neutralization "
        "condition (oracle, CIC top-1, CIC top-3 consensus, matched random text region; watermark oracle shown for "
        "contrast). Red bars: median per-input residual-to-clean class-dependent perturbation "
        "`max_y |delta_to_clean_y - mean_y delta_to_clean|` (lower = more class-balanced). Green/blue bars: "
        "margin-condition satisfaction rate and repair accuracy. Oracle/CIC neutralization is more class-balanced than "
        "matched random neutralization when the recovery theorem's weaker per-input premise holds.\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Unavailable path
# ---------------------------------------------------------------------------
def _write_unavailable(out_dir: Path, status: ClipStatus, epsilon_b: float) -> dict[str, str]:
    key_numbers = {
        "per_input_class_balance_supported_for_text": False,
        "clip_theory_support_status": "conditional only; per-input class-balance not supported",
        "pretrained_clip_loaded": bool(status.pretrained),
        "fake_backend": status.backend == "fake",
        "backend": status.backend,
        "model_name": status.model_name,
        "evidence_status": status.error_message or "pretrained CLIP unavailable; no theory-supporting evidence generated",
        "any_more_balanced_than_random": False,
        "random_median_residual_to_clean": None,
        "per_condition_checks": {},
        "object_entanglement_statement": OBJECT_ENTANGLEMENT_STATEMENT,
        "theorem_caveat": THEOREM_CAVEAT,
    }
    (out_dir / "per_input_class_balance_key_numbers.json").write_text(json.dumps(key_numbers, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame([{"condition": "unavailable", "backend": status.backend, "n_examples": 0}]).to_csv(out_dir / "per_input_class_balance_metrics.csv", index=False)
    pd.DataFrame().to_csv(out_dir / "per_input_class_balance_examples.csv", index=False)
    _write_summary(out_dir, key_numbers, {}, status, epsilon_b)
    _plot({}, out_dir / "per_input_class_balance_plot.png", out_dir / "per_input_class_balance_plot.pdf")
    return {
        "summary": str(out_dir / "per_input_class_balance_summary.md"),
        "metrics": str(out_dir / "per_input_class_balance_metrics.csv"),
        "examples": str(out_dir / "per_input_class_balance_examples.csv"),
        "key_numbers": str(out_dir / "per_input_class_balance_key_numbers.json"),
        "caption": str(out_dir / "per_input_class_balance_caption.md"),
        "plot_png": str(out_dir / "per_input_class_balance_plot.png"),
        "plot_pdf": str(out_dir / "per_input_class_balance_plot.pdf"),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(cfg: dict[str, Any]) -> dict[str, str]:
    total_start = time.perf_counter()
    seed = int(cfg.get("seed", 0))
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "per_input_class_balance")
    thresholds = dict(cfg.get("thresholds", {}))
    epsilon_b = float(thresholds.get("class_balance_epsilon_b", 3.0))
    model_cfg = cfg.get("model", {})
    device = _device(model_cfg, cfg)

    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))
    if str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend == "fake":
        status = ClipStatus(False, "fake", "fake_clip_per_input_class_balance", pretrained=False, device=device, backend_attempted="fake", error_message="fake backend is not allowed for per-input class-balance theory evidence")
        return _write_unavailable(out_dir, status, epsilon_b)
    status = check_clip_available(
        device=device,
        allow_download=_downloads_allowed(model_cfg),
        preferred_backend=requested_backend,
        model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
        pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
        transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return _write_unavailable(out_dir, status, epsilon_b)

    data_cfg = cfg.get("data", {})
    size = int(data_cfg.get("image_size", 224))
    text_n = int(data_cfg.get("text_n_per_class", 8))
    wm_n = int(data_cfg.get("watermark_n_per_class", 6))
    wm_seed = int(data_cfg.get("watermark_benchmark_seed", 5151))
    max_candidates = int(cfg.get("max_candidates", 64))

    text_policy, text_source = _load_text_policy(cfg)
    prompts = [PROMPT_TEMPLATE.format(label=name) for name in CLIP_OVERLAY_CLASSES]
    model = ClipZeroShotClassifier(status, CLIP_OVERLAY_CLASSES.copy(), prompts=prompts, device=device)

    text_examples = build_text_misleading_examples(text_policy, text_n, size)
    records: list[dict[str, Any]] = []
    for ex in text_examples:
        records.extend(
            evaluate_text_example(ex, model, prompts, seed=seed + 7_000, max_candidates=max_candidates, epsilon_b=epsilon_b)
        )

    include_watermark = bool(cfg.get("include_watermark", True))
    if include_watermark:
        wm_examples = build_watermark_misleading_examples(wm_n, size, wm_seed)
        for ex in wm_examples:
            records.extend(evaluate_watermark_example(ex, model, epsilon_b=epsilon_b))

    examples_df = pd.DataFrame(records)
    examples_df.to_csv(out_dir / "per_input_class_balance_examples.csv", index=False)

    conditions = list(TEXT_CONDITIONS) + ([COND_WATERMARK] if include_watermark else [])
    agg = {cond: aggregate_condition(examples_df, cond) for cond in conditions}
    pd.DataFrame([agg[c] for c in conditions]).to_csv(out_dir / "per_input_class_balance_metrics.csv", index=False)

    decision = decide_support(agg, status, thresholds)

    key_numbers: dict[str, Any] = {
        "per_input_class_balance_supported_for_text": decision["per_input_class_balance_supported_for_text"],
        "clip_theory_support_status": decision["clip_theory_support_status"],
        "pretrained_clip_loaded": bool(status.pretrained),
        "fake_backend": status.backend == "fake",
        "backend": status.backend,
        "model_name": status.model_name,
        "pretrained_tag": status.pretrained_tag,
        "evidence_status": "pretrained CLIP per-input class-balance validation evidence",
        "text_source": text_source,
        "text_generation_policy_id": str(text_policy.get("policy_id", "")),
        "class_balance_epsilon_b": epsilon_b,
        "thresholds": thresholds,
        "n_text_examples": int(len(text_examples)),
        "n_text_condition_records": int((examples_df["family"] == "text_overlay").sum()) if len(examples_df) else 0,
        "include_watermark": include_watermark,
        "any_more_balanced_than_random": decision["any_more_balanced_than_random"],
        "random_median_residual_to_clean": decision["random_median_residual_to_clean"],
        "per_condition_checks": decision["per_condition_checks"],
        "object_entanglement_statement": OBJECT_ENTANGLEMENT_STATEMENT,
        "theorem_caveat": THEOREM_CAVEAT,
    }
    # Flatten per-condition aggregate metrics into key numbers.
    for cond in conditions:
        row = agg[cond]
        for metric in (
            "n_examples",
            "mean_shift_std",
            "median_shift_std",
            "mean_shift_range",
            "median_shift_range",
            "mean_max_centered_shift",
            "median_max_centered_shift",
            "mean_residual_to_clean",
            "median_residual_to_clean",
            "pct_class_balance_satisfied",
            "margin_condition_satisfaction_rate",
            "repair_accuracy",
            "repair_success_when_margin_satisfied",
            "repair_success_when_margin_violated",
            "mean_residual_repaired",
            "mean_residual_failed",
        ):
            key_numbers[f"{cond}_{metric}"] = row.get(metric)

    (out_dir / "per_input_class_balance_key_numbers.json").write_text(
        json.dumps(key_numbers, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    _plot(agg, out_dir / "per_input_class_balance_plot.png", out_dir / "per_input_class_balance_plot.pdf")
    _write_summary(out_dir, key_numbers, agg, status, epsilon_b)
    pd.DataFrame([{"total_time_sec": time.perf_counter() - total_start, "seed": seed, "n_text_examples": len(text_examples)}]).to_csv(
        out_dir / "per_input_class_balance_timing_profile.csv", index=False
    )
    return {
        "summary": str(out_dir / "per_input_class_balance_summary.md"),
        "metrics": str(out_dir / "per_input_class_balance_metrics.csv"),
        "examples": str(out_dir / "per_input_class_balance_examples.csv"),
        "key_numbers": str(out_dir / "per_input_class_balance_key_numbers.json"),
        "caption": str(out_dir / "per_input_class_balance_caption.md"),
        "plot_png": str(out_dir / "per_input_class_balance_plot.png"),
        "plot_pdf": str(out_dir / "per_input_class_balance_plot.pdf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/per_input_class_balance_validation.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
