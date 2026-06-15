"""Failure-conditioned hard multi-decoy CLIP repair benchmark.

Instead of asking whether every generated multi-decoy image becomes hard, this
benchmark explicitly evaluates repair on a held-out subset of examples where
pretrained CLIP actually fails because of misleading overlay text.

A candidate is admitted to the failure-conditioned test set only if, using the
frozen selected generation policy:

* the no-overlay image is classified correctly,
* the aligned-overlay image is classified correctly,
* the misleading multi-decoy image is classified incorrectly,
* the original (misleading) confidence is at least a configured threshold, and
* oracle harmful-text neutralization restores the correct prediction.

True labels and oracle repair are used ONLY to define this held-out failure
subset. Non-oracle CIC scoring and repair still receive image pixels, CLIP
predictions, class prompts, and candidate proposals only -- never the true label,
harmful bounding box, harmful text, or correctness signal.

This is framed honestly as a failure-conditioned repair evaluation, not a general
accuracy evaluation and not open-world shortcut discovery.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.data.clip_overlay_shortcuts import CLIP_OVERLAY_CLASSES
from causal_reliability.discovery.cic_region_scoring import neutralize_region
from causal_reliability.experiments.run_clip_overlay_validation import _downloads_allowed
from causal_reliability.experiments.run_hard_multidecoy_clip_repair import (
    HARD_REGIMES,
    _ci_text,
    _wilson_ci,
    render_hard_multidecoy_image,
)
from causal_reliability.experiments.run_hard_multidecoy_clip_repair_audit import _crosstab
from causal_reliability.experiments.run_multidecoy_clip_repair import (
    _PredictionCache,
    _evaluate_examples,
)
from causal_reliability.experiments.run_nonoracle_clip_repair import PROMPT_TEMPLATE, _device
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


HARD_REGIME = "hard_multi_decoy_misleading"
ALIGNED_REGIME = "hard_multi_decoy_aligned"
NO_OVERLAY_REGIME = "no_overlay"
METHOD_RENAME = {
    "nonoracle_cic_top1_region_repair": "nonoracle_cic_top1_repair",
    "nonoracle_cic_top3_consensus_repair": "nonoracle_cic_top3_repair",
}


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.asarray(arr).clip(0, 1) * 255).astype(np.uint8)).convert("RGB")


def _passes_failure_conditions(
    *,
    no_overlay_correct: bool,
    aligned_correct: bool,
    misleading_wrong: bool,
    confidence_ok: bool,
    oracle_restored: bool,
) -> bool:
    """A candidate joins the held-out failure subset only if pretrained CLIP gets
    the clean (no-overlay) and aligned-overlay images right, gets the misleading
    multi-decoy image wrong with high enough confidence, and oracle harmful-text
    neutralization restores the correct prediction."""

    return bool(no_overlay_correct and aligned_correct and misleading_wrong and confidence_ok and oracle_restored)


def _frozen_policy(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return None if data.get("unavailable") else data


def build_failure_conditioned_set(
    cfg: dict[str, Any],
    generation_policy: dict[str, Any],
    predict_fn: _PredictionCache,
    *,
    size: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """Screen a candidate pool and return included misleading examples.

    Returns (failure_examples, inclusion_log, stats). Each failure example is the
    *misleading* rendered image plus its eval-only metadata; its paired aligned
    and no-overlay variants are attached so downstream repair can also measure
    clean preservation. True labels and oracle repair are used here only to
    decide inclusion in the held-out failure subset.
    """

    n_classes = min(int(generation_policy.get("class_set_size", len(CLIP_OVERLAY_CLASSES))), len(CLIP_OVERLAY_CLASSES))
    pool_per_class = int(cfg.get("pool_per_class", 64))
    n_target = int(cfg.get("n_failure_target", 50))
    conf_threshold = float(cfg.get("original_confidence_threshold", 0.50))
    benchmark_seed = int(cfg.get("pool_benchmark_seed", 4242))

    inclusion_rows: list[dict[str, Any]] = []
    failure_examples: list[dict[str, Any]] = []
    n_candidates = 0
    n_included = 0
    eid = benchmark_seed * 1_000_000

    for idx in range(pool_per_class):
        if n_included >= n_target:
            break
        # Batch all four regime variants for every class at this pool index.
        batch_specs: list[tuple[int, int, np.ndarray, dict[str, Any], np.ndarray, dict[str, Any], np.ndarray, dict[str, Any]]] = []
        images: list[Image.Image] = []
        for label in range(n_classes):
            mis_img, mis_meta = render_hard_multidecoy_image(label, HARD_REGIME, idx, generation_policy, size=size, benchmark_seed=benchmark_seed)
            ali_img, ali_meta = render_hard_multidecoy_image(label, ALIGNED_REGIME, idx, generation_policy, size=size, benchmark_seed=benchmark_seed)
            nov_img, nov_meta = render_hard_multidecoy_image(label, NO_OVERLAY_REGIME, idx, generation_policy, size=size, benchmark_seed=benchmark_seed)
            batch_specs.append((idx, label, mis_img, mis_meta, ali_img, ali_meta, nov_img, nov_meta))
            images.extend([_to_pil(nov_img), _to_pil(ali_img), _to_pil(mis_img)])
        base_probs = predict_fn(images)
        # Oracle neutralization batch (only misleading variants with a harmful bbox).
        oracle_images = []
        oracle_index: list[int] = []
        for spec_i, (_, _, mis_img, mis_meta, *_rest) in enumerate(batch_specs):
            bbox = mis_meta.get("harmful_bbox") or []
            if bbox:
                oracle_images.append(neutralize_region(_to_pil(mis_img), tuple(bbox)))
                oracle_index.append(spec_i)
        oracle_probs = predict_fn(oracle_images) if oracle_images else np.zeros((0, n_classes))
        oracle_lookup = {spec_i: oracle_probs[k] for k, spec_i in enumerate(oracle_index)}

        for spec_i, (pool_idx, label, mis_img, mis_meta, ali_img, ali_meta, nov_img, nov_meta) in enumerate(batch_specs):
            n_candidates += 1
            nov_probs = base_probs[spec_i * 3 + 0]
            ali_probs = base_probs[spec_i * 3 + 1]
            mis_probs = base_probs[spec_i * 3 + 2]
            nov_correct = bool(int(nov_probs.argmax()) == label)
            ali_correct = bool(int(ali_probs.argmax()) == label)
            mis_wrong = bool(int(mis_probs.argmax()) != label)
            mis_conf = float(mis_probs.max())
            conf_ok = bool(mis_conf >= conf_threshold)
            oracle_correct = bool(spec_i in oracle_lookup and int(oracle_lookup[spec_i].argmax()) == label)
            passes = _passes_failure_conditions(
                no_overlay_correct=nov_correct,
                aligned_correct=ali_correct,
                misleading_wrong=mis_wrong,
                confidence_ok=conf_ok,
                oracle_restored=oracle_correct,
            )
            included = bool(passes and n_included < n_target)

            reasons = []
            if not nov_correct:
                reasons.append("no_overlay_incorrect")
            if not ali_correct:
                reasons.append("aligned_incorrect")
            if not mis_wrong:
                reasons.append("misleading_not_wrong")
            if not conf_ok:
                reasons.append(f"confidence_below_{conf_threshold:g}")
            if not oracle_correct:
                reasons.append("oracle_did_not_restore")
            if included:
                reasons = []
            elif not reasons and n_included >= n_target:
                reasons.append("target_reached")

            inclusion_rows.append(
                {
                    "pool_index": pool_idx,
                    "label": label,
                    "true_label": CLIP_OVERLAY_CLASSES[label],
                    "no_overlay_correct": nov_correct,
                    "aligned_correct": ali_correct,
                    "misleading_wrong": mis_wrong,
                    "misleading_confidence": mis_conf,
                    "confidence_threshold": conf_threshold,
                    "oracle_restored": oracle_correct,
                    "included": included,
                    "exclusion_reasons": ";".join(reasons),
                    "harmful_text": mis_meta.get("harmful_text", ""),
                }
            )
            if not included:
                continue
            common = {"split": "failure_conditioned_test", "label": label, "true_label": CLIP_OVERLAY_CLASSES[label]}
            failure_examples.append({"example_id": eid, "regime": HARD_REGIME, "image": mis_img, **common, **mis_meta})
            eid += 1
            failure_examples.append({"example_id": eid, "regime": ALIGNED_REGIME, "image": ali_img, **common, **ali_meta})
            eid += 1
            failure_examples.append({"example_id": eid, "regime": NO_OVERLAY_REGIME, "image": nov_img, **common, **nov_meta})
            eid += 1
            n_included += 1

    n_included = sum(1 for ex in failure_examples if ex["regime"] == HARD_REGIME)
    stats = {
        "n_candidates": int(n_candidates),
        "n_failure_examples": int(n_included),
        "inclusion_rate": float(n_included / n_candidates) if n_candidates else 0.0,
        "n_target": int(n_target),
        "original_confidence_threshold": conf_threshold,
        "pool_benchmark_seed": benchmark_seed,
        "pool_per_class": pool_per_class,
        "target_reached": bool(n_included >= n_target),
    }
    return failure_examples, pd.DataFrame(inclusion_rows), stats


def _method_after(lookup: dict[str, dict[str, Any]], method: str, regime: str = HARD_REGIME) -> float:
    return float(lookup.get(method, {}).get(f"{regime}_accuracy_after", np.nan))


def _write_unavailable(out_dir: Path, status: ClipStatus, reason: str, *, write_inclusion_log: bool = True) -> dict[str, str]:
    pd.DataFrame([{"method": "unavailable", "headline_eligible": False, "pretrained_loaded": bool(status.pretrained), "backend": status.backend, "model_name": status.model_name, "reason": reason}]).to_csv(out_dir / "failure_conditioned_metrics.csv", index=False)
    pd.DataFrame([{"unavailable": True, "reason": reason}]).to_csv(out_dir / "failure_conditioned_certificates.csv", index=False)
    if write_inclusion_log:
        pd.DataFrame([{"unavailable": True, "reason": reason}]).to_csv(out_dir / "failure_conditioned_inclusion_log.csv", index=False)
    pd.DataFrame([{"unavailable": True, "reason": reason}]).to_csv(out_dir / "failure_conditioned_repair_vs_localization.csv", index=False)
    key_numbers = {"failure_conditioned_headline_eligible": False, "pretrained_loaded": False, "fake_backend": status.backend == "fake", "reason": reason}
    (out_dir / "failure_conditioned_key_numbers.json").write_text(json.dumps(key_numbers, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "failure_conditioned_summary.md").write_text(
        "# Failure-Conditioned Hard Multi-Decoy CLIP Repair\n\n"
        "This is a failure-conditioned repair evaluation, not open-world shortcut discovery.\n\n"
        f"Pretrained CLIP unavailable or frozen policies missing ({reason}); no fake headline evidence was generated.\n",
        encoding="utf-8",
    )
    return {
        "metrics": str(out_dir / "failure_conditioned_metrics.csv"),
        "certificates": str(out_dir / "failure_conditioned_certificates.csv"),
        "inclusion_log": str(out_dir / "failure_conditioned_inclusion_log.csv"),
        "repair_vs_localization": str(out_dir / "failure_conditioned_repair_vs_localization.csv"),
        "key_numbers": str(out_dir / "failure_conditioned_key_numbers.json"),
        "summary": str(out_dir / "failure_conditioned_summary.md"),
    }


def run(cfg: dict[str, Any]) -> dict[str, str]:
    total_start = time.perf_counter()
    timing: dict[str, float] = {}
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "hard_multidecoy_failure_conditioned")
    size = int(cfg.get("data", {}).get("image_size", 224))

    model_cfg = cfg.get("model", {})
    device = _device(model_cfg, cfg)
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))
    if str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend == "fake":
        status = ClipStatus(False, "fake", "fake_clip_failure_conditioned", pretrained=False, device=device, backend_attempted="fake", error_message="fake backend is not allowed for CLIP repair evidence")
        return _write_unavailable(out_dir, status, "fake backend is not allowed")
    status = check_clip_available(
        device=device,
        allow_download=_downloads_allowed(model_cfg),
        preferred_backend=requested_backend,
        model_name=str(model_cfg.get("model_name", DEFAULT_MODEL_NAME)),
        pretrained_tag=str(model_cfg.get("pretrained_tag", DEFAULT_PRETRAINED_TAG)),
        transformers_model_name=str(model_cfg.get("transformers_model_name", DEFAULT_TRANSFORMERS_MODEL)),
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return _write_unavailable(out_dir, status, status.error_message or "pretrained CLIP did not load")

    frozen_dir = Path(cfg.get("frozen_policy_dir", "results/hard_multidecoy_clip_repair"))
    generation_policy = cfg.get("frozen_generation_policy") or _frozen_policy(frozen_dir / "selected_generation_policy.json")
    repair_policy = cfg.get("frozen_repair_policy") or _frozen_policy(frozen_dir / "selected_repair_policy.json")
    if generation_policy is None or repair_policy is None:
        return _write_unavailable(out_dir, status, f"missing frozen policies in {frozen_dir}")

    prompts = [PROMPT_TEMPLATE.format(label=name) for name in CLIP_OVERLAY_CLASSES]
    model = ClipZeroShotClassifier(status, CLIP_OVERLAY_CLASSES.copy(), prompts=prompts, device=device)
    screen_cache = _PredictionCache(model)

    gen_start = time.perf_counter()
    failure_examples, inclusion_log, stats = build_failure_conditioned_set(cfg, generation_policy, screen_cache, size=size)
    timing["generation_and_screening_time_sec"] = time.perf_counter() - gen_start
    inclusion_log.to_csv(out_dir / "failure_conditioned_inclusion_log.csv", index=False)
    print(
        f"[failure-conditioned] candidates={stats['n_candidates']} included={stats['n_failure_examples']} "
        f"inclusion_rate={stats['inclusion_rate']:.3f} (target {stats['n_target']})",
        flush=True,
    )

    if stats["n_failure_examples"] == 0:
        # Preserve the real screening log written above; only stub the repair artifacts.
        return _write_unavailable(out_dir, status, "no failure-conditioned examples were found", write_inclusion_log=False)

    max_candidates = int(cfg.get("max_candidates", 96))
    n_views = int(cfg.get("augmentation_views", 5))
    random_draws = int(cfg.get("random_draws", 25))
    resume = bool(cfg.get("resume", False))
    cache_root = ensure_dir(out_dir / "example_eval_cache")

    eval_start = time.perf_counter()
    certs, rankings = _evaluate_examples(
        examples=failure_examples,
        class_names=CLIP_OVERLAY_CLASSES.copy(),
        prompts=prompts,
        model=model,
        seed=seed,
        max_candidates=max_candidates,
        n_views=n_views,
        rng=rng,
        policy=repair_policy,
        random_draws=random_draws,
        cache_dir=cache_root / "failure",
        resume=resume,
        progress_label="failure-conditioned repair",
    )
    timing["repair_eval_time_sec"] = time.perf_counter() - eval_start
    certs["method"] = certs["method"].replace(METHOD_RENAME)
    certs.to_csv(out_dir / "failure_conditioned_certificates.csv", index=False)

    metrics = _failure_metrics(certs, rankings, status, stats)
    metrics.to_csv(out_dir / "failure_conditioned_metrics.csv", index=False)
    pd.DataFrame([{k: v for k, v in stats.items()}]).to_csv(out_dir / "failure_conditioned_certificates_stats.csv", index=False)

    crosstab = _crosstab(certs, rankings, "failure_conditioned", "failure_conditioned")
    crosstab.to_csv(out_dir / "failure_conditioned_repair_vs_localization.csv", index=False)

    key_numbers, headline_reasons = _key_numbers(metrics, stats, status, cfg)
    timing["total_time_sec"] = time.perf_counter() - total_start
    key_numbers["timing_sec"] = {k: round(float(v), 3) for k, v in timing.items()}
    (out_dir / "failure_conditioned_key_numbers.json").write_text(json.dumps(key_numbers, indent=2, sort_keys=True), encoding="utf-8")

    _write_summary(out_dir, metrics, crosstab, stats, key_numbers, headline_reasons, status, timing)
    print(
        f"[failure-conditioned] timing generation+screening={timing['generation_and_screening_time_sec']:.2f}s "
        f"repair_eval={timing['repair_eval_time_sec']:.2f}s total={timing['total_time_sec']:.2f}s",
        flush=True,
    )
    return {
        "metrics": str(out_dir / "failure_conditioned_metrics.csv"),
        "certificates": str(out_dir / "failure_conditioned_certificates.csv"),
        "inclusion_log": str(out_dir / "failure_conditioned_inclusion_log.csv"),
        "repair_vs_localization": str(out_dir / "failure_conditioned_repair_vs_localization.csv"),
        "key_numbers": str(out_dir / "failure_conditioned_key_numbers.json"),
        "summary": str(out_dir / "failure_conditioned_summary.md"),
    }


def _failure_metrics(certs: pd.DataFrame, rankings: pd.DataFrame, status: ClipStatus, stats: dict[str, Any]) -> pd.DataFrame:
    rows = []
    misleading = rankings[rankings["regime"] == HARD_REGIME] if len(rankings) else pd.DataFrame()
    loc: dict[str, float] = {}
    if len(misleading):
        groups = [g for _, g in misleading.groupby("example_id")]
        top1 = misleading[misleading["rank"] == 1]
        n_loc = len(groups)
        top1_03 = int((top1["harmful_iou"] >= 0.3).sum())
        top1_05 = int((top1["harmful_iou"] >= 0.5).sum())
        top3_03 = int(sum((g.nsmallest(3, "rank")["harmful_iou"] >= 0.3).any() for g in groups))
        top3_05 = int(sum((g.nsmallest(3, "rank")["harmful_iou"] >= 0.5).any() for g in groups))
        loc = {
            "harmful_top1_iou_0_3": float(top1_03 / max(1, len(top1))),
            "harmful_top1_iou_0_5": float(top1_05 / max(1, len(top1))),
            "harmful_top3_iou_0_3": float(top3_03 / max(1, n_loc)),
            "harmful_top3_iou_0_5": float(top3_05 / max(1, n_loc)),
            "harmful_top1_iou_0_3_ci95": _ci_text(top1_03, len(top1)),
            "harmful_top3_iou_0_3_ci95": _ci_text(top3_03, n_loc),
            "median_harmful_rank": float(np.median([g.sort_values("harmful_iou", ascending=False)["rank"].iloc[0] for g in groups])),
        }
    for method, df in certs.groupby("method", sort=False):
        mis = df[df["regime"] == HARD_REGIME]
        aligned = df[df["regime"] == ALIGNED_REGIME]
        nov = df[df["regime"] == NO_OVERLAY_REGIME]
        mis_non_abs = ~mis["abstained"].astype(bool) if len(mis) else pd.Series(dtype=bool)
        repaired = mis.loc[mis_non_abs, "repaired_correct"].astype(bool) if len(mis) else pd.Series(dtype=bool)
        successes = int(repaired.sum())
        n_rep = int(len(repaired))
        row: dict[str, Any] = {
            "method": method,
            "backend": status.backend,
            "model_name": status.model_name,
            "pretrained_loaded": bool(status.pretrained),
            "n_failure_examples": int(len(mis)),
            "failure_subset_original_accuracy": float(mis["original_correct"].astype(bool).mean()) if len(mis) else np.nan,
            "failure_subset_repaired_accuracy": float(repaired.mean()) if n_rep else np.nan,
            "failure_subset_repaired_accuracy_ci95": _ci_text(successes, n_rep),
            "coverage": float(mis_non_abs.mean()) if len(mis) else np.nan,
            "abstention_rate": float((~mis_non_abs).mean()) if len(mis) else np.nan,
            "selective_accuracy": float(repaired.mean()) if n_rep else np.nan,
            "no_overlay_preservation_after": float(nov.loc[~nov["abstained"].astype(bool), "repaired_correct"].astype(bool).mean()) if len(nov) and bool((~nov["abstained"].astype(bool)).sum()) else np.nan,
            "no_overlay_accuracy_before": float(nov["original_correct"].astype(bool).mean()) if len(nov) else np.nan,
            "aligned_preservation_after": float(aligned.loc[~aligned["abstained"].astype(bool), "repaired_correct"].astype(bool).mean()) if len(aligned) and bool((~aligned["abstained"].astype(bool)).sum()) else np.nan,
            "aligned_accuracy_before": float(aligned["original_correct"].astype(bool).mean()) if len(aligned) else np.nan,
        }
        # naming consistency with hard-multidecoy metrics so downstream readers can reuse keys
        row[f"{HARD_REGIME}_accuracy_before"] = row["failure_subset_original_accuracy"]
        row[f"{HARD_REGIME}_accuracy_after"] = row["failure_subset_repaired_accuracy"]
        if method in {"random_matched_text_region_repair", "random_nontext_patch_repair"} and len(mis):
            draw = mis["selected_candidate_id"].astype(str).str.extract(r"(?:draw|random_nontext_)(\d+)")[0].fillna("0")
            vals = [float(g["repaired_correct"].astype(bool).mean()) for _, g in mis.assign(_draw=draw).groupby("_draw")]
            if vals:
                arr = np.asarray(vals)
                row["random_draw_failure_accuracy_mean"] = float(arr.mean())
                row["random_draw_failure_accuracy_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                row["random_draw_failure_accuracy_ci95"] = float(1.96 * row["random_draw_failure_accuracy_std"] / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    for key, value in loc.items():
        out[key] = value
    out["n_candidates"] = stats["n_candidates"]
    out["inclusion_rate"] = stats["inclusion_rate"]
    return out


def _key_numbers(metrics: pd.DataFrame, stats: dict[str, Any], status: ClipStatus, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    lookup = metrics.set_index("method").to_dict("index") if "method" in metrics else {}

    def after(method: str) -> float | None:
        v = lookup.get(method, {}).get("failure_subset_repaired_accuracy", np.nan)
        return float(v) if pd.notna(v) and np.isfinite(v) else None

    original = lookup.get("original_clip_prediction", {})
    oracle = after("oracle_harmful_text_neutralization")
    top1 = after("nonoracle_cic_top1_repair")
    top3 = after("nonoracle_cic_top3_repair")
    clean_safe = after("nonoracle_cic_clean_safe_repair")
    selective_row = lookup.get("nonoracle_cic_selective_repair_or_abstain", {})
    random_row = lookup.get("random_matched_text_region_repair", {})
    random_mean = float(random_row.get("random_draw_failure_accuracy_mean", np.nan)) if pd.notna(random_row.get("random_draw_failure_accuracy_mean", np.nan)) else None
    random_ci = float(random_row.get("random_draw_failure_accuracy_ci95", np.nan)) if pd.notna(random_row.get("random_draw_failure_accuracy_ci95", np.nan)) else None
    n_failure = int(stats["n_failure_examples"])

    best_cic = max([v for v in [top1, top3] if v is not None], default=None)
    gap = (best_cic - random_mean) if (best_cic is not None and random_mean is not None) else None
    # CIC lower Wilson bound vs random mean + draw CI for the non-overlapping-CI test.
    cic_method = "nonoracle_cic_top1_repair" if (top1 is not None and (top3 is None or top1 >= top3)) else "nonoracle_cic_top3_repair"
    cic_n = int(lookup.get(cic_method, {}).get("n_failure_examples", n_failure) or n_failure)
    cic_succ = int(round((best_cic or 0.0) * cic_n))
    cic_lo, _cic_hi = _wilson_ci(cic_succ, cic_n)
    non_overlapping = bool(best_cic is not None and random_mean is not None and cic_lo > (random_mean + (random_ci or 0.0)))
    beats_random = bool((gap is not None and gap >= 0.15) or non_overlapping)

    clean_min = float(cfg.get("clean_preservation_min", 0.85))
    nov_pres = float(lookup.get("nonoracle_cic_clean_safe_repair", {}).get("no_overlay_preservation_after", np.nan))
    aligned_pres = float(lookup.get("nonoracle_cic_clean_safe_repair", {}).get("aligned_preservation_after", np.nan))
    clean_preserved = bool(np.isfinite(nov_pres) and np.isfinite(aligned_pres) and nov_pres >= 0.90 and aligned_pres >= clean_min)

    reasons: list[str] = []
    if not bool(status.pretrained):
        reasons.append("pretrained CLIP not loaded")
    if status.backend == "fake":
        reasons.append("fake backend")
    if n_failure < 30:
        reasons.append(f"n_failure_examples {n_failure} < 30")
    if oracle is None or oracle < 0.85:
        reasons.append("oracle repair < 0.85")
    if best_cic is None or best_cic < 0.75:
        reasons.append("CIC top-1/top-3 repair < 0.75")
    if not beats_random:
        reasons.append("CIC does not beat matched random text repair by >= 0.15 or non-overlapping CI")
    if not clean_preserved:
        reasons.append("clean / no-overlay preservation not high")

    headline_eligible = len(reasons) == 0

    key_numbers = {
        "failure_conditioned_headline_eligible": headline_eligible,
        "failure_conditioned_headline_failed_reasons": reasons,
        "failure_conditioned_evaluation_label": "failure-conditioned repair evaluation, not open-world shortcut discovery",
        "pretrained_loaded": bool(status.pretrained),
        "fake_backend": status.backend == "fake",
        "backend": status.backend,
        "model_name": status.model_name,
        "nonoracle_scorer_excludes_labels_bboxes_correctness": True,
        "n_candidates": stats["n_candidates"],
        "n_failure_examples": n_failure,
        "inclusion_rate": stats["inclusion_rate"],
        "original_confidence_threshold": stats["original_confidence_threshold"],
        "failure_subset_original_accuracy": float(original.get("failure_subset_original_accuracy", np.nan)) if pd.notna(original.get("failure_subset_original_accuracy", np.nan)) else None,
        "oracle_repair_accuracy": oracle,
        "cic_top1_repair_accuracy": top1,
        "cic_top3_repair_accuracy": top3,
        "cic_clean_safe_repair_accuracy": clean_safe,
        "cic_selective_accuracy": float(selective_row.get("selective_accuracy", np.nan)) if pd.notna(selective_row.get("selective_accuracy", np.nan)) else None,
        "cic_selective_coverage": float(selective_row.get("coverage", np.nan)) if pd.notna(selective_row.get("coverage", np.nan)) else None,
        "cic_selective_abstention": float(selective_row.get("abstention_rate", np.nan)) if pd.notna(selective_row.get("abstention_rate", np.nan)) else None,
        "random_matched_text_repair_mean": random_mean,
        "random_matched_text_repair_std": float(random_row.get("random_draw_failure_accuracy_std", np.nan)) if pd.notna(random_row.get("random_draw_failure_accuracy_std", np.nan)) else None,
        "random_matched_text_repair_95ci": random_ci,
        "highest_textness_repair_accuracy": after("highest_textness_region_repair"),
        "largest_text_repair_accuracy": after("largest_text_region_repair"),
        "random_augmentation_accuracy": after("random_augmentation_consensus"),
        "random_nontext_patch_accuracy": after("random_nontext_patch_repair"),
        "cic_minus_random_gap": gap,
        "cic_beats_random": beats_random,
        "cic_non_overlapping_ci_vs_random": non_overlapping,
        "no_overlay_preservation_after": float(nov_pres) if np.isfinite(nov_pres) else None,
        "aligned_preservation_after": float(aligned_pres) if np.isfinite(aligned_pres) else None,
        "clean_preservation_high": clean_preserved,
        "harmful_top1_iou_0_3": float(metrics["harmful_top1_iou_0_3"].iloc[0]) if "harmful_top1_iou_0_3" in metrics and len(metrics) else None,
        "harmful_top1_iou_0_5": float(metrics["harmful_top1_iou_0_5"].iloc[0]) if "harmful_top1_iou_0_5" in metrics and len(metrics) else None,
        "harmful_top3_iou_0_3": float(metrics["harmful_top3_iou_0_3"].iloc[0]) if "harmful_top3_iou_0_3" in metrics and len(metrics) else None,
        "harmful_top3_iou_0_5": float(metrics["harmful_top3_iou_0_5"].iloc[0]) if "harmful_top3_iou_0_5" in metrics and len(metrics) else None,
    }
    return key_numbers, reasons


def _write_summary(out_dir: Path, metrics: pd.DataFrame, crosstab: pd.DataFrame, stats: dict[str, Any], key_numbers: dict[str, Any], reasons: list[str], status: ClipStatus, timing: dict[str, float]) -> None:
    def fmt(value: Any) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "NA"
        return f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)

    lines = [
        "# Failure-Conditioned Hard Multi-Decoy CLIP Repair",
        "",
        "This is a **failure-conditioned repair evaluation**, not a general accuracy evaluation and not open-world shortcut discovery. "
        "Examples are admitted only when pretrained CLIP classifies the no-overlay and aligned-overlay images correctly, classifies the "
        "misleading multi-decoy image incorrectly with confidence at least the configured threshold, and oracle harmful-text neutralization "
        "restores the correct prediction. True labels and oracle repair are used only to define this held-out failure subset; non-oracle CIC "
        "scoring and repair receive image pixels, CLIP predictions, class prompts, and candidate proposals only.",
        "",
        "Because the test set is finite and conditioned on observed failures, the original failure-subset accuracy is ~0 by construction and "
        "is not a natural benchmark accuracy. This evaluation does not establish general robustness or open-world discovery.",
        "",
        f"Backend: {status.backend}. Model: {status.model_name}. Pretrained loaded: `{status.pretrained}`.",
        "",
        "## Failure-Conditioned Benchmark Construction",
        "",
        f"- Candidate images generated: {stats['n_candidates']}",
        f"- Failure-conditioned examples included: {stats['n_failure_examples']} (target {stats['n_target']})",
        f"- Inclusion rate: {fmt(stats['inclusion_rate'])}",
        f"- Original confidence threshold: {fmt(stats['original_confidence_threshold'])}",
        "",
        "## Repair Results on the Failure Subset",
        "",
        f"- Original failure-subset accuracy (expected ~0 by construction): {fmt(key_numbers['failure_subset_original_accuracy'])}",
        f"- Oracle harmful-text repair (upper bound): {fmt(key_numbers['oracle_repair_accuracy'])}",
        f"- CIC top-1 repair: {fmt(key_numbers['cic_top1_repair_accuracy'])}",
        f"- CIC top-3 repair: {fmt(key_numbers['cic_top3_repair_accuracy'])}",
        f"- CIC clean-safe repair: {fmt(key_numbers['cic_clean_safe_repair_accuracy'])}",
        f"- CIC selective accuracy / coverage / abstention: {fmt(key_numbers['cic_selective_accuracy'])} / {fmt(key_numbers['cic_selective_coverage'])} / {fmt(key_numbers['cic_selective_abstention'])}",
        f"- Random matched text-region repair mean/std/95% CI: {fmt(key_numbers['random_matched_text_repair_mean'])} / {fmt(key_numbers['random_matched_text_repair_std'])} / +/- {fmt(key_numbers['random_matched_text_repair_95ci'])}",
        f"- Highest-textness repair: {fmt(key_numbers['highest_textness_repair_accuracy'])}",
        f"- Largest-text repair: {fmt(key_numbers['largest_text_repair_accuracy'])}",
        f"- Random augmentation: {fmt(key_numbers['random_augmentation_accuracy'])}",
        f"- Random non-text patch: {fmt(key_numbers['random_nontext_patch_accuracy'])}",
        f"- CIC minus random matched text gap: {fmt(key_numbers['cic_minus_random_gap'])} (beats random: `{key_numbers['cic_beats_random']}`, non-overlapping CI: `{key_numbers['cic_non_overlapping_ci_vs_random']}`)",
        "",
        "## Clean Preservation on Included Examples",
        "",
        f"- No-overlay preservation after clean-safe repair: {fmt(key_numbers['no_overlay_preservation_after'])}",
        f"- Aligned-overlay preservation after clean-safe repair: {fmt(key_numbers['aligned_preservation_after'])}",
        "",
        "## Localization on the Failure Subset",
        "",
        f"- CIC top-1 harmful localization IoU >= 0.3 / 0.5: {fmt(key_numbers['harmful_top1_iou_0_3'])} / {fmt(key_numbers['harmful_top1_iou_0_5'])}",
        f"- CIC top-3 harmful localization IoU >= 0.3 / 0.5: {fmt(key_numbers['harmful_top3_iou_0_3'])} / {fmt(key_numbers['harmful_top3_iou_0_5'])}",
        "",
        _localization_interpretation(key_numbers),
        "",
        "## Repair vs Localization Crosstab",
        "",
        "Splits use coarse IoU >= 0.3. If IoU >= 0.5 remains weak, interpret this as coarse localization rather than exact box recovery.",
        "",
        _markdown_table(crosstab) if len(crosstab) else "Crosstab unavailable.",
        "",
        "## Headline Eligibility",
        "",
        f"- failure_conditioned_headline_eligible = `{key_numbers['failure_conditioned_headline_eligible']}`",
        f"- pretrained CLIP loaded: `{key_numbers['pretrained_loaded']}`; fake backend: `{key_numbers['fake_backend']}`",
        f"- non-oracle scorer excludes labels/bboxes/correctness: `{key_numbers['nonoracle_scorer_excludes_labels_bboxes_correctness']}`",
        ("- failed reasons: " + "; ".join(reasons)) if reasons else "- all headline-eligibility checks passed",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
    ]
    (out_dir / "failure_conditioned_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _localization_interpretation(key_numbers: dict[str, Any]) -> str:
    top1_03 = key_numbers.get("harmful_top1_iou_0_3")
    top3_03 = key_numbers.get("harmful_top3_iou_0_3")
    top1_05 = key_numbers.get("harmful_top1_iou_0_5")
    parts = []
    coarse = max([v for v in [top1_03, top3_03] if v is not None], default=0.0)
    if coarse >= 0.5:
        parts.append("Repair works when coarse localization (IoU >= 0.3) succeeds; repair depends substantially on coarse localization.")
    else:
        parts.append("CIC identifies useful text-region interventions even when coarse harmful-box localization is imperfect; exact harmful-box localization is not the sole repair mechanism.")
    if top1_05 is not None and top1_05 < 0.5:
        parts.append("IoU >= 0.5 remains weak, so this is coarse causal-region localization, not precise bounding-box recovery.")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hard_multidecoy_failure_conditioned_repair.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
