"""Scale-and-multi-model replication audit for the hard multi-decoy text-overlay benchmark.

This is a SUPPORTING experiment. It does not change and does not overwrite the
frozen primary headline (ViT-B-32 / laion2b_s34b_b79k at n=32 per condition),
which lives under ``results/hard_multidecoy_clip_repair/``.

It addresses two reviewer concerns:

A. Larger-n benchmark. Re-run the hard multi-decoy benchmark at a larger
   ``n_per_condition`` (128 by default; 256 optional) on a distinct resampled
   held-out benchmark instance using real pretrained OpenCLIP only.

B. Multi-model replication. Re-run the same benchmark on multiple available
   OpenCLIP model/pretrained pairs detected from ``open_clip.list_pretrained()``
   (or attempted directly). Unavailable / unloadable models are skipped (not
   fatal). Models that do not exhibit a strong shortcut failure are reported as
   "not failure-rich / not repair-eligible" rather than as a negative CIC result.

The fake CLIP backend is never headline eligible.

Interpretation is deliberately conservative: this audit does not claim open-world
discovery, general robustness, cross-shortcut generalization, or exact
localization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.experiments.run_hard_multidecoy_clip_repair import run as run_hard_repair
from causal_reliability.experiments.run_hard_multidecoy_clip_repair_audit import _lookup
from causal_reliability.experiments.run_nonoracle_clip_repair import _device
from causal_reliability.real_models.clip_zero_shot import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED_TAG,
)
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


HEADLINE_MODEL_NAME = "ViT-B-32"
HEADLINE_PRETRAINED_TAG = "laion2b_s34b_b79k"

DEFAULT_CANDIDATE_MODELS = [
    {"model_name": HEADLINE_MODEL_NAME, "pretrained_tag": HEADLINE_PRETRAINED_TAG},
    {"model_name": "ViT-B-32", "pretrained_tag": "openai"},
    {"model_name": "ViT-B-16", "pretrained_tag": "laion2b_s34b_b88k"},
    {"model_name": "RN50", "pretrained_tag": "openai"},
]

DEFAULT_GATES = {
    "max_misleading_accuracy": 0.40,
    "min_cic_random_gap": 0.15,
    "max_clean_safe_drop": 0.10,
    "min_hard_misleading_n": 64,
}

DO_NOT_CLAIM = (
    "This audit does not claim open-world shortcut discovery, general robustness, "
    "cross-shortcut generalization, or exact localization. The method searches a "
    "finite candidate class of text-region proposals on a controlled synthetic "
    "text-overlay benchmark."
)


def _model_slug(model_name: str, pretrained_tag: str) -> str:
    raw = f"{model_name}__{pretrained_tag}"
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in raw)


def _is_headline(model_name: str, pretrained_tag: str) -> bool:
    return model_name == HEADLINE_MODEL_NAME and pretrained_tag == HEADLINE_PRETRAINED_TAG


def _candidate_pairs(cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Configured candidate model/checkpoint pairs, headline first and de-duplicated."""
    configured = cfg.get("models") or DEFAULT_CANDIDATE_MODELS
    pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    headline = {"model_name": HEADLINE_MODEL_NAME, "pretrained_tag": HEADLINE_PRETRAINED_TAG}
    for entry in [headline, *configured]:
        model_name = str(entry.get("model_name", DEFAULT_MODEL_NAME))
        pretrained_tag = str(entry.get("pretrained_tag", DEFAULT_PRETRAINED_TAG))
        key = (model_name, pretrained_tag)
        if key in seen:
            continue
        seen.add(key)
        pair = {"model_name": model_name, "pretrained_tag": pretrained_tag}
        if entry.get("transformers_model_name"):
            pair["transformers_model_name"] = str(entry["transformers_model_name"])
        pairs.append(pair)
    return pairs


def _registry_pairs() -> set[tuple[str, str]] | None:
    """Available (model, pretrained) pairs from open_clip, or None if unavailable."""
    try:
        import open_clip  # type: ignore
    except Exception:
        return None
    try:
        return {(str(m), str(t)) for m, t in open_clip.list_pretrained()}
    except Exception:
        return None


def _benchmark_hash(certs_path: Path) -> str:
    """Identity hash of the generated benchmark, derived from the run certificates."""
    if not certs_path.exists():
        return ""
    df = pd.read_csv(certs_path)
    cols = [c for c in ["example_id", "regime", "true_label", "harmful_text"] if c in df.columns]
    if not cols:
        return ""
    sub = df[cols].drop_duplicates().sort_values(cols)
    return hashlib.sha256(sub.to_csv(index=False).encode("utf-8")).hexdigest()


def _num(d: dict[str, Any], key: str) -> float:
    value = d.get(key)
    if value is None or value == "":
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _ci(d: dict[str, Any], key: str) -> str:
    value = d.get(key)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return str(value)


def _extract_metrics(metrics: pd.DataFrame) -> dict[str, Any]:
    """Pull the per-model numbers and CIs from a base hard-repair metrics frame."""
    lk = _lookup(metrics)
    original = lk.get("original_clip_prediction", {})
    oracle = lk.get("oracle_harmful_text_neutralization", {})
    top1 = lk.get("nonoracle_cic_top1_repair", {})
    top3 = lk.get("nonoracle_cic_top3_repair", {})
    clean_safe = lk.get("nonoracle_cic_clean_safe_repair", {})
    random_text = lk.get("random_matched_text_region_repair", {})

    cic_top1 = _num(top1, "hard_multi_decoy_misleading_accuracy_after")
    random_mean = _num(random_text, "random_draw_hard_misleading_accuracy_mean")
    if not np.isfinite(random_mean):
        random_mean = _num(random_text, "hard_multi_decoy_misleading_accuracy_after")
    gap = cic_top1 - random_mean if np.isfinite(cic_top1) and np.isfinite(random_mean) else np.nan

    pretrained = bool(str(top1.get("pretrained_loaded", original.get("pretrained_loaded", False))).lower() in {"true", "1"})
    backend = str(top1.get("backend", original.get("backend", "")))

    return {
        "pretrained_loaded": pretrained,
        "backend": backend,
        "n_hard_misleading_examples": int(_num(original, "n_hard_misleading_examples")) if np.isfinite(_num(original, "n_hard_misleading_examples")) else 0,
        "n_aligned_overlay_examples": int(_num(original, "n_aligned_overlay_examples")) if np.isfinite(_num(original, "n_aligned_overlay_examples")) else 0,
        "n_neutral_overlay_examples": int(_num(original, "n_neutral_overlay_examples")) if np.isfinite(_num(original, "n_neutral_overlay_examples")) else 0,
        "n_no_overlay_examples": int(_num(original, "n_no_overlay_examples")) if np.isfinite(_num(original, "n_no_overlay_examples")) else 0,
        "no_overlay_accuracy": _num(original, "no_overlay_accuracy_before"),
        "no_overlay_accuracy_ci95": _ci(original, "no_overlay_accuracy_before_ci95"),
        "aligned_accuracy": _num(original, "hard_multi_decoy_aligned_accuracy_before"),
        "aligned_accuracy_ci95": _ci(original, "hard_multi_decoy_aligned_accuracy_before_ci95"),
        "original_misleading_accuracy": _num(original, "hard_multi_decoy_misleading_accuracy_before"),
        "original_misleading_accuracy_ci95": _ci(original, "hard_multi_decoy_misleading_accuracy_before_ci95"),
        "oracle_misleading_accuracy": _num(oracle, "hard_multi_decoy_misleading_accuracy_after"),
        "cic_top1_repair_accuracy": cic_top1,
        "cic_top1_repair_accuracy_ci95": _ci(top1, "hard_multi_decoy_misleading_accuracy_after_ci95"),
        "cic_top3_repair_accuracy": _num(top3, "hard_multi_decoy_misleading_accuracy_after"),
        "cic_top3_repair_accuracy_ci95": _ci(top3, "hard_multi_decoy_misleading_accuracy_after_ci95"),
        "cic_clean_safe_repair_accuracy": _num(clean_safe, "hard_multi_decoy_misleading_accuracy_after"),
        "cic_clean_safe_repair_accuracy_ci95": _ci(clean_safe, "hard_multi_decoy_misleading_accuracy_after_ci95"),
        "matched_random_repair_accuracy": random_mean,
        "matched_random_repair_std": _num(random_text, "random_draw_hard_misleading_accuracy_std"),
        "matched_random_repair_ci95_halfwidth": _num(random_text, "random_draw_hard_misleading_accuracy_ci95"),
        "cic_top1_minus_matched_random_gap": gap,
        "clean_safe_clean_drop": _num(clean_safe, "clean_accuracy_drop"),
        "cic_top1_clean_drop": _num(top1, "clean_accuracy_drop"),
    }


def _eligibility(extracted: dict[str, Any], *, fake_backend: bool, gates: dict[str, Any]) -> dict[str, Any]:
    """Apply the headline-eligibility gates for one model and explain the verdict."""
    loaded = bool(extracted.get("pretrained_loaded", False))
    orig = extracted.get("original_misleading_accuracy", np.nan)
    gap = extracted.get("cic_top1_minus_matched_random_gap", np.nan)
    clean_drop = extracted.get("clean_safe_clean_drop", np.nan)
    n_misleading = int(extracted.get("n_hard_misleading_examples", 0) or 0)

    max_mis = float(gates.get("max_misleading_accuracy", DEFAULT_GATES["max_misleading_accuracy"]))
    min_gap = float(gates.get("min_cic_random_gap", DEFAULT_GATES["min_cic_random_gap"]))
    max_drop = float(gates.get("max_clean_safe_drop", DEFAULT_GATES["max_clean_safe_drop"]))
    min_n = int(gates.get("min_hard_misleading_n", DEFAULT_GATES["min_hard_misleading_n"]))

    pretrained_loaded = loaded
    not_fake = not fake_backend
    shortcut_vulnerable = bool(np.isfinite(orig) and orig <= max_mis)
    cic_beats_random = bool(np.isfinite(gap) and gap >= min_gap)
    clean_drop_ok = bool(np.isfinite(clean_drop) and clean_drop <= max_drop)
    sufficient_n = n_misleading >= min_n

    headline_eligible = bool(pretrained_loaded and not_fake and shortcut_vulnerable and cic_beats_random and clean_drop_ok and sufficient_n)

    if headline_eligible:
        status = "repair_eligible"
    elif pretrained_loaded and not_fake and not shortcut_vulnerable:
        status = "not_failure_rich"  # not a negative CIC result
    elif not pretrained_loaded or fake_backend:
        status = "not_real_pretrained"
    else:
        status = "repair_not_demonstrated"

    reasons: list[str] = []
    if not pretrained_loaded:
        reasons.append("pretrained weights not loaded")
    if fake_backend:
        reasons.append("fake backend is never headline eligible")
    if pretrained_loaded and not_fake and not shortcut_vulnerable:
        reasons.append(f"not failure-rich: original misleading accuracy {orig:.3f} > {max_mis:.2f}")
    if pretrained_loaded and not_fake and shortcut_vulnerable and not cic_beats_random:
        reasons.append(f"CIC-random gap {gap:.3f} < {min_gap:.2f}")
    if pretrained_loaded and not_fake and shortcut_vulnerable and not clean_drop_ok:
        reasons.append(f"clean-safe clean drop {clean_drop:.3f} > {max_drop:.2f}")
    if pretrained_loaded and not_fake and not sufficient_n:
        reasons.append(f"hard misleading n {n_misleading} < {min_n}")

    return {
        "pretrained_loaded": pretrained_loaded,
        "fake_backend": fake_backend,
        "shortcut_vulnerable": shortcut_vulnerable,
        "cic_beats_random": cic_beats_random,
        "clean_drop_ok": clean_drop_ok,
        "sufficient_n": sufficient_n,
        "headline_eligible": headline_eligible,
        "eligibility_status": status,
        "eligibility_reasons": "; ".join(reasons) or "all gates passed",
    }


def _run_one_model(
    cfg: dict[str, Any],
    pair: dict[str, str],
    *,
    test_n_per_class: int,
    benchmark_seed: int,
    runs_dir: Path,
    frozen_generation: dict[str, Any] | None,
    frozen_repair: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run the frozen hard multi-decoy benchmark for one model and extract metrics."""
    model_name = pair["model_name"]
    pretrained_tag = pair["pretrained_tag"]
    slug = _model_slug(model_name, pretrained_tag)

    audit_cfg = copy.deepcopy(cfg)
    audit_cfg["results_dir"] = str(ensure_dir(runs_dir / slug))
    audit_cfg["seed"] = int(cfg.get("seed", 0))
    audit_cfg["benchmark_seed"] = int(benchmark_seed)
    audit_cfg["resample_benchmark"] = True
    audit_cfg["max_candidates"] = int(cfg.get("max_candidates", 96))
    audit_cfg["augmentation_views"] = int(cfg.get("augmentation_views", 5))
    audit_cfg["random_draws"] = int(cfg.get("random_draws", 25))
    audit_cfg.setdefault("data", {})
    audit_cfg["data"]["test_n_per_class"] = int(test_n_per_class)
    audit_cfg["data"]["validation_n_per_class"] = int(cfg.get("data", {}).get("validation_n_per_class", 8))
    model_cfg = dict(audit_cfg.get("model", {}))
    model_cfg["preferred_backend"] = str(model_cfg.get("preferred_backend", "open_clip"))
    model_cfg["model_name"] = model_name
    model_cfg["pretrained_tag"] = pretrained_tag
    if pair.get("transformers_model_name"):
        model_cfg["transformers_model_name"] = pair["transformers_model_name"]
    audit_cfg["model"] = model_cfg
    if frozen_generation is not None:
        audit_cfg["frozen_generation_policy"] = frozen_generation
        audit_cfg.pop("frozen_generation_policy_path", None)
    if frozen_repair is not None:
        audit_cfg["frozen_repair_policy"] = frozen_repair
        audit_cfg.pop("frozen_repair_policy_path", None)

    start = time.perf_counter()
    outputs = run_hard_repair(audit_cfg)
    runtime = time.perf_counter() - start

    metrics = pd.read_csv(outputs["metrics"])
    extracted = _extract_metrics(metrics)
    extracted["benchmark_hash"] = _benchmark_hash(Path(outputs["certificates"]))
    extracted["runtime_sec"] = runtime
    extracted["run_dir"] = str(Path(outputs["metrics"]).parent)
    return extracted


def _frozen_policy(cfg: dict[str, Any], inline_key: str, path_key: str) -> dict[str, Any] | None:
    inline = cfg.get(inline_key)
    if inline is not None:
        return dict(inline)
    path = cfg.get(path_key)
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return None


def _plot(rows: list[dict[str, Any]], png_path: Path) -> None:
    loaded = [r for r in rows if r.get("pretrained_loaded")]
    plt.figure(figsize=(max(7.0, 1.9 * max(1, len(loaded))), 4.8))
    if loaded:
        labels = [f"{r['model_name']}\n{r['pretrained_tag']}" for r in loaded]
        x = np.arange(len(loaded))
        width = 0.27

        def col(key: str) -> list[float]:
            return [float(r.get(key)) if np.isfinite(_safe(r.get(key))) else np.nan for r in loaded]

        plt.bar(x - width, col("original_misleading_accuracy"), width, label="original misleading", color="#bab0ac")
        plt.bar(x, col("cic_top1_repair_accuracy"), width, label="CIC top-1 repair", color="#4c78a8")
        plt.bar(x + width, col("matched_random_repair_accuracy"), width, label="matched random text", color="#e45756")
        plt.xticks(x, labels, rotation=20, ha="right", fontsize=8)
        plt.ylim(0, 1.02)
        plt.ylabel("Hard misleading accuracy")
        plt.title("Hard multi-decoy scale + multi-model audit")
        plt.legend(fontsize=8)
    else:
        plt.text(0.5, 0.5, "No real pretrained OpenCLIP model loaded", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=170)
    plt.close()


def _safe(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _interpretation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [r for r in rows if r.get("headline_eligible")]
    loaded = [r for r in rows if r.get("pretrained_loaded")]
    headline_row = next((r for r in rows if _is_headline(r["model_name"], r["pretrained_tag"])), None)
    headline_eligible = bool(headline_row and headline_row.get("headline_eligible"))
    n_eligible = len(eligible)

    statements: list[str] = []
    if headline_eligible:
        statements.append("The main text-overlay result is stable at larger n.")
    if n_eligible >= 2:
        statements.append(
            "The text-overlay CIC effect replicates across multiple pretrained OpenCLIP backbones/checkpoints."
        )
    elif n_eligible <= 1 and len(loaded) >= 2:
        statements.append(
            "Multi-model audit was run, but only models showing a sufficiently strong shortcut failure are repair-eligible."
        )
    return {
        "larger_n_stable": headline_eligible,
        "multi_model_replicates": n_eligible >= 2,
        "n_eligible_models": n_eligible,
        "n_loaded_models": len(loaded),
        "headline_model_eligible": headline_eligible,
        "statements": statements,
    }


def _write_outputs(
    out_dir: Path,
    *,
    rows: list[dict[str, Any]],
    availability: list[dict[str, Any]],
    n_per_condition: int,
    class_set_size: int,
    fake_backend: bool,
) -> dict[str, str]:
    availability_df = pd.DataFrame(availability)
    availability_path = out_dir / "model_availability.csv"
    availability_df.to_csv(availability_path, index=False)

    metric_cols = [
        "model_name", "pretrained_tag", "headline_model", "backend", "n_per_condition",
        "n_hard_misleading_examples", "n_aligned_overlay_examples", "n_neutral_overlay_examples", "n_no_overlay_examples",
        "no_overlay_accuracy", "no_overlay_accuracy_ci95",
        "aligned_accuracy", "aligned_accuracy_ci95",
        "original_misleading_accuracy", "original_misleading_accuracy_ci95",
        "oracle_misleading_accuracy",
        "cic_top1_repair_accuracy", "cic_top1_repair_accuracy_ci95",
        "cic_top3_repair_accuracy", "cic_top3_repair_accuracy_ci95",
        "cic_clean_safe_repair_accuracy", "cic_clean_safe_repair_accuracy_ci95",
        "matched_random_repair_accuracy", "matched_random_repair_std", "matched_random_repair_ci95_halfwidth",
        "cic_top1_minus_matched_random_gap",
        "clean_safe_clean_drop", "cic_top1_clean_drop",
        "pretrained_loaded", "fake_backend", "shortcut_vulnerable", "cic_beats_random", "clean_drop_ok",
        "sufficient_n", "headline_eligible", "eligibility_status", "eligibility_reasons",
        "benchmark_hash", "runtime_sec",
    ]
    metrics_df = pd.DataFrame(rows)
    for col in metric_cols:
        if col not in metrics_df.columns:
            metrics_df[col] = np.nan
    metrics_df = metrics_df[metric_cols] if len(metrics_df) else pd.DataFrame(columns=metric_cols)
    metrics_path = out_dir / "scale_model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    plot_path = out_dir / "scale_model_plot.png"
    _plot(rows, plot_path)

    interp = _interpretation(rows)
    headline_row = next((r for r in rows if _is_headline(r["model_name"], r["pretrained_tag"])), None)

    key_numbers = {
        "fake_backend_allowed": False,
        "fake_backend_requested": bool(fake_backend),
        "n_per_condition": int(n_per_condition),
        "class_set_size": int(class_set_size),
        "models_attempted": [{"model_name": a["model_name"], "pretrained_tag": a["pretrained_tag"]} for a in availability],
        "models_loaded": [
            {"model_name": r["model_name"], "pretrained_tag": r["pretrained_tag"]}
            for r in rows if r.get("pretrained_loaded")
        ],
        "models_skipped": [
            {"model_name": a["model_name"], "pretrained_tag": a["pretrained_tag"], "skip_reason": a.get("skip_reason", "")}
            for a in availability if a.get("skipped")
        ],
        "per_model": [
            {
                "model_name": r["model_name"],
                "pretrained_tag": r["pretrained_tag"],
                "headline_model": bool(r.get("headline_model")),
                "pretrained_loaded": bool(r.get("pretrained_loaded")),
                "n_hard_misleading_examples": int(r.get("n_hard_misleading_examples", 0) or 0),
                "no_overlay_accuracy": _round(r.get("no_overlay_accuracy")),
                "aligned_accuracy": _round(r.get("aligned_accuracy")),
                "original_misleading_accuracy": _round(r.get("original_misleading_accuracy")),
                "cic_top1_repair_accuracy": _round(r.get("cic_top1_repair_accuracy")),
                "cic_top3_repair_accuracy": _round(r.get("cic_top3_repair_accuracy")),
                "cic_clean_safe_repair_accuracy": _round(r.get("cic_clean_safe_repair_accuracy")),
                "matched_random_repair_accuracy": _round(r.get("matched_random_repair_accuracy")),
                "cic_top1_minus_matched_random_gap": _round(r.get("cic_top1_minus_matched_random_gap")),
                "clean_safe_clean_drop": _round(r.get("clean_safe_clean_drop")),
                "headline_eligible": bool(r.get("headline_eligible")),
                "eligibility_status": r.get("eligibility_status"),
                "eligibility_reasons": r.get("eligibility_reasons"),
                "benchmark_hash": r.get("benchmark_hash"),
            }
            for r in rows
        ],
        "headline_model_eligible": interp["headline_model_eligible"],
        "n_eligible_models": interp["n_eligible_models"],
        "n_loaded_models": interp["n_loaded_models"],
        "larger_n_stable": interp["larger_n_stable"],
        "multi_model_replicates": interp["multi_model_replicates"],
        "interpretation_statements": interp["statements"],
        "do_not_claim": DO_NOT_CLAIM,
        "headline_model_protected": True,
        "headline_model_protected_note": (
            "The frozen primary headline (ViT-B-32 / laion2b_s34b_b79k at n=32) under "
            "results/hard_multidecoy_clip_repair/ is not modified by this audit."
        ),
        "outputs": {
            "summary": str(out_dir / "scale_model_summary.md"),
            "key_numbers": str(out_dir / "scale_model_key_numbers.json"),
            "metrics": str(metrics_path),
            "plot": str(plot_path),
            "model_availability": str(availability_path),
        },
    }
    key_numbers_path = out_dir / "scale_model_key_numbers.json"
    key_numbers_path.write_text(json.dumps(key_numbers, indent=2, sort_keys=True), encoding="utf-8")

    summary = _build_summary(
        rows=rows,
        availability=availability,
        interp=interp,
        headline_row=headline_row,
        n_per_condition=n_per_condition,
        class_set_size=class_set_size,
        fake_backend=fake_backend,
        metrics_df=metrics_df,
        availability_df=availability_df,
    )
    summary_path = out_dir / "scale_model_summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    return {
        "summary": str(summary_path),
        "key_numbers": str(key_numbers_path),
        "metrics": str(metrics_path),
        "plot": str(plot_path),
        "model_availability": str(availability_path),
    }


def _round(value: Any, ndigits: int = 4) -> Any:
    v = _safe(value)
    return round(v, ndigits) if np.isfinite(v) else None


def _build_summary(
    *,
    rows: list[dict[str, Any]],
    availability: list[dict[str, Any]],
    interp: dict[str, Any],
    headline_row: dict[str, Any] | None,
    n_per_condition: int,
    class_set_size: int,
    fake_backend: bool,
    metrics_df: pd.DataFrame,
    availability_df: pd.DataFrame,
) -> str:
    lines = [
        "# Hard Multi-Decoy Scale + Multi-Model Replication Audit",
        "",
        "Supporting audit for the hard multi-decoy text-overlay benchmark. This does NOT "
        "replace the frozen primary headline (ViT-B-32 / laion2b_s34b_b79k at n=32 per "
        "condition) under `results/hard_multidecoy_clip_repair/`, which is left unchanged.",
        "",
        f"- n_per_condition: {int(n_per_condition)} (test_n_per_class = {int(n_per_condition) // max(1, int(class_set_size))} x {int(class_set_size)} classes)",
        f"- Real pretrained OpenCLIP only; fake backend is never headline eligible (fake requested: {bool(fake_backend)}).",
        f"- Models attempted: {len(availability)}; loaded: {interp['n_loaded_models']}; headline-eligible: {interp['n_eligible_models']}.",
        "",
        "## Interpretation",
        "",
    ]
    if interp["statements"]:
        for statement in interp["statements"]:
            lines.append(f"- {statement}")
    else:
        lines.append("- No model met the headline-eligibility gates on this audit.")
    lines.extend(
        [
            "",
            DO_NOT_CLAIM,
            "",
            "## Model Availability",
            "",
            _markdown_table(availability_df) if len(availability_df) else "No candidate models.",
            "",
            "## Per-Model Metrics",
            "",
            _markdown_table(metrics_df) if len(metrics_df) else "No model produced metrics.",
            "",
            "## Headline (frozen primary result, n=32) — protected",
            "",
            "This audit never writes to `results/hard_multidecoy_clip_repair/`. The frozen "
            "primary headline remains the ViT-B-32 / laion2b_s34b_b79k n=32 result.",
        ]
    )
    if headline_row is not None and headline_row.get("pretrained_loaded"):
        lines.extend(
            [
                "",
                f"At n_per_condition={int(n_per_condition)} the headline backbone "
                f"(ViT-B-32 / laion2b_s34b_b79k) gave original misleading accuracy "
                f"{_safe(headline_row.get('original_misleading_accuracy')):.3f}, CIC top-1 repair "
                f"{_safe(headline_row.get('cic_top1_repair_accuracy')):.3f}, matched random "
                f"{_safe(headline_row.get('matched_random_repair_accuracy')):.3f} "
                f"(gap {_safe(headline_row.get('cic_top1_minus_matched_random_gap')):.3f}), "
                f"clean-safe clean drop {_safe(headline_row.get('clean_safe_clean_drop')):.3f}.",
            ]
        )
    return "\n".join(lines)


def run(cfg: dict[str, Any]) -> dict[str, str]:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "hard_multidecoy_scale_model_audit")
    runs_dir = ensure_dir(out_dir / "runs")

    model_cfg = cfg.get("model", {})
    device = _device(model_cfg, cfg)
    requested_backend = str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip")))
    fake_backend = str(model_cfg.get("backend", "")).lower() == "fake" or requested_backend == "fake"

    gates = {**DEFAULT_GATES, **(cfg.get("gates") or {})}
    n_per_condition = int(cfg.get("n_per_condition", 128))
    benchmark_seed = int(cfg.get("benchmark_seed", 73019))

    frozen_generation = _frozen_policy(cfg, "frozen_generation_policy", "frozen_generation_policy_path")
    frozen_repair = _frozen_policy(cfg, "frozen_repair_policy", "frozen_repair_policy_path")
    class_set_size = int((frozen_generation or {}).get("class_set_size", 4))
    test_n_per_class = max(1, n_per_condition // max(1, class_set_size))

    pairs = _candidate_pairs(cfg)
    registry = _registry_pairs()

    availability: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for pair in pairs:
        model_name = pair["model_name"]
        pretrained_tag = pair["pretrained_tag"]
        headline_model = _is_headline(model_name, pretrained_tag)
        registry_known = None if registry is None else ((model_name, pretrained_tag) in registry)

        avail: dict[str, Any] = {
            "model_name": model_name,
            "pretrained_tag": pretrained_tag,
            "headline_model": headline_model,
            "registry_known": registry_known,
            "attempted": False,
            "loaded": False,
            "backend": "",
            "fake_backend": bool(fake_backend),
            "skipped": False,
            "skip_reason": "",
        }

        if fake_backend:
            avail["skipped"] = True
            avail["skip_reason"] = "fake backend is not allowed for CLIP repair evidence"
            availability.append(avail)
            continue

        if registry is not None and registry_known is False:
            avail["skipped"] = True
            avail["skip_reason"] = "not present in open_clip.list_pretrained()"
            availability.append(avail)
            continue

        avail["attempted"] = True
        try:
            extracted = _run_one_model(
                cfg,
                pair,
                test_n_per_class=test_n_per_class,
                benchmark_seed=benchmark_seed,
                runs_dir=runs_dir,
                frozen_generation=frozen_generation,
                frozen_repair=frozen_repair,
            )
        except Exception as exc:  # loading or eval failure must not be fatal
            avail["skipped"] = True
            avail["skip_reason"] = f"run failed: {type(exc).__name__}: {exc}"
            availability.append(avail)
            continue

        avail["backend"] = extracted.get("backend", "")
        if not extracted.get("pretrained_loaded"):
            avail["skipped"] = True
            avail["skip_reason"] = "pretrained weights did not load"
            availability.append(avail)
            continue

        avail["loaded"] = True
        availability.append(avail)

        elig = _eligibility(extracted, fake_backend=False, gates=gates)
        rows.append({"model_name": model_name, "pretrained_tag": pretrained_tag, "headline_model": headline_model, "n_per_condition": n_per_condition, **extracted, **elig})

    return _write_outputs(
        out_dir,
        rows=rows,
        availability=availability,
        n_per_condition=n_per_condition,
        class_set_size=class_set_size,
        fake_backend=fake_backend,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hard_multidecoy_scale_model_audit.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
