from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.experiments.run_hard_multidecoy_clip_repair import make_hard_dataset, run
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


HARD_REGIME = "hard_multi_decoy_misleading"


def _lookup(metrics: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return metrics.set_index("method").to_dict("index") if "method" in metrics else {}


def _row(lookup: dict[str, dict[str, Any]], seed_id: int) -> dict[str, Any]:
    original = lookup.get("original_clip_prediction", {})
    oracle = lookup.get("oracle_harmful_text_neutralization", {})
    top1 = lookup.get("nonoracle_cic_top1_repair", {})
    top3 = lookup.get("nonoracle_cic_top3_repair", {})
    clean_safe = lookup.get("nonoracle_cic_clean_safe_repair", {})
    selective = lookup.get("nonoracle_cic_selective_repair_or_abstain", {})
    random_text = lookup.get("random_matched_text_region_repair", {})
    highest_textness = lookup.get("highest_textness_region_repair", {})
    largest_text = lookup.get("largest_text_region_repair", {})
    random_aug = lookup.get("random_augmentation_consensus", {})

    headline = str(top1.get("headline_eligible", original.get("headline_eligible", False))).lower() in {"true", "1"}
    failed_reasons = ""
    if not headline:
        reasons = []
        if not bool(top1.get("pretrained_loaded", original.get("pretrained_loaded", False))):
            reasons.append("pretrained CLIP unavailable")
        if not top1:
            reasons.append("missing CIC top-1 metrics")
        failed_reasons = "; ".join(reasons) or "headline_eligible false"

    return {
        "seed_id": seed_id,
        "n_examples": original.get("n_examples"),
        "n_hard_misleading_examples": original.get("n_hard_misleading_examples"),
        "no_overlay_accuracy": original.get("no_overlay_accuracy_before"),
        "aligned_overlay_accuracy": original.get("hard_multi_decoy_aligned_accuracy_before"),
        "original_hard_misleading_accuracy": original.get("hard_multi_decoy_misleading_accuracy_before"),
        "oracle_repair_accuracy": oracle.get("hard_multi_decoy_misleading_accuracy_after"),
        "cic_top1_repair_accuracy": top1.get("hard_multi_decoy_misleading_accuracy_after"),
        "cic_top3_repair_accuracy": top3.get("hard_multi_decoy_misleading_accuracy_after"),
        "cic_clean_safe_repair_accuracy": clean_safe.get("hard_multi_decoy_misleading_accuracy_after"),
        "cic_selective_accuracy": selective.get("selective_accuracy"),
        "cic_selective_coverage": selective.get("coverage"),
        "cic_selective_abstention": selective.get("abstention_rate"),
        "random_matched_text_repair_mean": random_text.get("random_draw_hard_misleading_accuracy_mean"),
        "random_matched_text_repair_std": random_text.get("random_draw_hard_misleading_accuracy_std"),
        "random_matched_text_repair_95ci": random_text.get("random_draw_hard_misleading_accuracy_ci95"),
        "highest_textness_repair_accuracy": highest_textness.get("hard_multi_decoy_misleading_accuracy_after"),
        "largest_text_repair_accuracy": largest_text.get("hard_multi_decoy_misleading_accuracy_after"),
        "random_augmentation_accuracy": random_aug.get("hard_multi_decoy_misleading_accuracy_after"),
        "clean_safe_clean_drop": clean_safe.get("clean_accuracy_drop"),
        "top1_localization_iou_ge_0_3": top1.get("harmful_top1_iou_0_3"),
        "top1_localization_iou_ge_0_5": top1.get("harmful_top1_iou_0_5"),
        "top3_localization_iou_ge_0_3": top1.get("harmful_top3_iou_0_3"),
        "top3_localization_iou_ge_0_5": top1.get("harmful_top3_iou_0_5"),
        "random_matched_localization_mean": random_text.get("random_draw_localization_iou_0_3_mean"),
        "random_matched_localization_std": random_text.get("random_draw_localization_iou_0_3_std"),
        "random_matched_localization_95ci": random_text.get("random_draw_localization_iou_0_3_ci95"),
        "median_harmful_rank": top1.get("median_harmful_rank"),
        "fixed_benchmark_determinism_check": False,
        "benchmark_resampled": False,
        "headline_eligible": headline,
        "failed_reasons": failed_reasons,
    }


def _seed_root(out_dir: Path, seed_id: int, *, lite: bool, resample_benchmark: bool, full_resample: bool = False) -> Path:
    if full_resample:
        parent = out_dir / "full_benchmark_resampling_runs"
    elif resample_benchmark:
        parent = out_dir / "benchmark_resampling_runs"
    else:
        parent = out_dir / ("fixed_benchmark_determinism_runs" if lite else "seed_stability_runs")
    return parent / f"seed_{seed_id}"


def _row_path(out_dir: Path, seed_id: int, *, lite: bool, resample_benchmark: bool, full_resample: bool = False) -> Path:
    if full_resample:
        name = "full_seed_resampling_row.csv"
    elif resample_benchmark:
        name = "seed_resampling_row.csv"
    else:
        name = "seed_stability_row.csv"
    return _seed_root(out_dir, seed_id, lite=lite, resample_benchmark=resample_benchmark, full_resample=full_resample) / name


def _image_set_hash(examples: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for ex in examples:
        arr = (np.asarray(ex["image"]).clip(0, 1) * 255).astype(np.uint8)
        h.update(str(ex["example_id"]).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _metadata_hash(examples: list[dict[str, Any]]) -> str:
    rows = []
    for ex in examples:
        rows.append(
            {
                "example_id": ex["example_id"],
                "regime": ex["regime"],
                "label": ex["label"],
                "true_label": ex["true_label"],
                "harmful_text": ex.get("harmful_text", ""),
                "harmful_bbox": ex.get("harmful_bbox", []),
                "decoy_bboxes": ex.get("decoy_bboxes", []),
                "all_text_boxes": ex.get("all_text_boxes", []),
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _frozen_policy(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _expected_test_examples(cfg: dict[str, Any], generation_policy: dict[str, Any], seed_id: int, *, resample_benchmark: bool) -> list[dict[str, Any]]:
    data_cfg = cfg.get("data", {})
    size = int(data_cfg.get("image_size", 224))
    val_n = int(data_cfg.get("validation_n_per_class", 8))
    test_n = int(data_cfg.get("test_n_per_class", 8))
    n_classes = int(generation_policy.get("class_set_size", 4))
    val_count = 4 * n_classes * val_n
    return make_hard_dataset(
        test_n,
        generation_policy,
        size=size,
        split="test",
        start_id=val_count,
        benchmark_seed=seed_id,
        resample=resample_benchmark,
    )


def _cache_summary(seed_root: Path) -> dict[str, Any]:
    cache_dir = seed_root / "hard_multidecoy_clip_repair" / "example_eval_cache" / "test"
    certs = list(cache_dir.glob("*_certificates.csv")) if cache_dir.exists() else []
    return {"cache_files": len(certs), "cache_dir": str(cache_dir)}


def _candidate_signature(rankings: pd.DataFrame) -> str:
    if rankings.empty:
        return ""
    top = rankings[rankings["rank"] == 1].sort_values("example_id")
    fields = [c for c in ["example_id", "candidate_id", "bbox", "harmful_iou"] if c in top]
    return hashlib.sha256(top[fields].to_csv(index=False).encode("utf-8")).hexdigest()


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else np.nan


def _crosstab(certs: pd.DataFrame, rankings: pd.DataFrame, seed_id: int | str, run_label: str) -> pd.DataFrame:
    if certs.empty or rankings.empty:
        return pd.DataFrame()
    hard_ids = certs.loc[(certs["method"] == "original_clip_prediction") & (certs["regime"] == HARD_REGIME), "example_id"].drop_duplicates()
    top1 = rankings[(rankings["regime"] == HARD_REGIME) & (rankings["rank"] == 1)].set_index("example_id")
    top3 = rankings[rankings["regime"] == HARD_REGIME].groupby("example_id")["harmful_iou"].apply(lambda s: bool((s.head(3) >= 0.3).any()))
    groups: list[dict[str, Any]] = []
    for split_name, mask_lookup in [
        ("top1_iou_ge_0_3", lambda eid: bool(eid in top1.index and float(top1.loc[eid, "harmful_iou"]) >= 0.3)),
        ("top1_iou_lt_0_3", lambda eid: not bool(eid in top1.index and float(top1.loc[eid, "harmful_iou"]) >= 0.3)),
        ("top3_iou_ge_0_3", lambda eid: bool(top3.get(eid, False))),
        ("top3_iou_lt_0_3", lambda eid: not bool(top3.get(eid, False))),
    ]:
        ids = set(eid for eid in hard_ids if mask_lookup(eid))
        base = certs[certs["example_id"].isin(ids)]
        row: dict[str, Any] = {"run_label": run_label, "seed_id": seed_id, "group": split_name, "n_examples": len(ids)}
        for method, out_key in [
            ("original_clip_prediction", "original_accuracy"),
            ("nonoracle_cic_top1_repair", "cic_top1_repair_accuracy"),
            ("nonoracle_cic_top3_repair", "cic_top3_repair_accuracy"),
            ("nonoracle_cic_clean_safe_repair", "clean_safe_repair_accuracy"),
            ("random_matched_text_region_repair", "random_matched_text_repair_accuracy"),
        ]:
            sub = base[base["method"] == method]
            if method == "original_clip_prediction":
                row[out_key] = _safe_mean(sub["original_correct"].astype(bool)) if len(sub) else np.nan
            else:
                non_abs = ~sub["abstained"].astype(bool) if len(sub) and "abstained" in sub else pd.Series(dtype=bool)
                row[out_key] = _safe_mean(sub.loc[non_abs, "repaired_correct"].astype(bool)) if len(sub) and bool(non_abs.sum()) else np.nan
        diag = base[base["method"] == "nonoracle_cic_top1_repair"]
        row["avg_confidence_before"] = _safe_mean(diag["original_confidence"].astype(float)) if "original_confidence" in diag else np.nan
        row["avg_confidence_after"] = _safe_mean(diag["repaired_confidence"].astype(float)) if "repaired_confidence" in diag else np.nan
        row["avg_drop_in_original_top_class_probability"] = _safe_mean(diag["drop_in_original_top_class_probability"].astype(float)) if "drop_in_original_top_class_probability" in diag else np.nan
        row["avg_prediction_flip_rate"] = _safe_mean(diag["prediction_flipped"].astype(bool)) if "prediction_flipped" in diag else np.nan
        row["avg_js_shift"] = _safe_mean(diag["js_shift"].astype(float)) if "js_shift" in diag else np.nan
        row["avg_kl_shift"] = _safe_mean(diag["kl_shift"].astype(float)) if "kl_shift" in diag else np.nan
        groups.append(row)
    return pd.DataFrame(groups)


def write_repair_vs_localization_crosstab(run_dir: Path, out_dir: Path, *, seed_id: int | str = "main", run_label: str = "main") -> pd.DataFrame:
    cert_path = run_dir / "hard_multidecoy_repair_certificates.csv"
    rank_path = run_dir / "hard_multidecoy_candidate_rankings.csv"
    if not cert_path.exists() or not rank_path.exists():
        return pd.DataFrame()
    certs = pd.read_csv(cert_path)
    certs["method"] = certs["method"].replace({"nonoracle_cic_top1_region_repair": "nonoracle_cic_top1_repair", "nonoracle_cic_top3_consensus_repair": "nonoracle_cic_top3_repair"})
    rankings = pd.read_csv(rank_path)
    df = _crosstab(certs, rankings, seed_id, run_label)
    if not df.empty:
        csv_path = out_dir / "repair_vs_localization_crosstab.csv"
        prior = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        combined = pd.concat([prior[prior.get("run_label", pd.Series(dtype=str)) != run_label], df], ignore_index=True) if not prior.empty else df
        combined.to_csv(csv_path, index=False)
        lines = [
            "# Repair vs Localization Crosstab",
            "",
            "Splits use coarse IoU >= 0.3. If IoU >= 0.5 remains weak, interpret this as coarse localization rather than exact box recovery.",
            "",
            _markdown_table(combined),
        ]
        (out_dir / "repair_vs_localization_crosstab.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def _run_one(cfg: dict[str, Any], seed_id: int, out_dir: Path, *, lite: bool, resume: bool, resample_benchmark: bool, full_resample: bool = False) -> dict[str, Any]:
    row_file = _row_path(out_dir, seed_id, lite=lite, resample_benchmark=resample_benchmark, full_resample=full_resample)
    if resume and row_file.exists():
        print(f"[benchmark-audit] reusing completed seed {seed_id}: {row_file}", flush=True)
        return pd.read_csv(row_file).iloc[0].to_dict()

    seed_root = ensure_dir(_seed_root(out_dir, seed_id, lite=lite, resample_benchmark=resample_benchmark, full_resample=full_resample))
    main_dir = out_dir
    generation_policy = _frozen_policy(main_dir / "selected_generation_policy.json")
    repair_policy = _frozen_policy(main_dir / "selected_repair_policy.json")
    if generation_policy is None:
        raise FileNotFoundError(f"Missing frozen generation policy: {main_dir / 'selected_generation_policy.json'}")
    if repair_policy is None:
        raise FileNotFoundError(f"Missing frozen repair policy: {main_dir / 'selected_repair_policy.json'}")

    expected = _expected_test_examples(cfg, generation_policy, seed_id, resample_benchmark=resample_benchmark)
    image_hash = _image_set_hash(expected)
    metadata_hash = _metadata_hash(expected)

    audit_cfg = copy.deepcopy(cfg)
    audit_cfg["seed"] = seed_id
    audit_cfg["benchmark_seed"] = seed_id
    audit_cfg["resample_benchmark"] = bool(resample_benchmark)
    audit_cfg["resume"] = bool(resume)
    audit_cfg["frozen_generation_policy"] = generation_policy
    audit_cfg["frozen_repair_policy"] = repair_policy
    audit_cfg["results_dir"] = str(seed_root)
    if lite:
        audit_cfg.setdefault("data", {})
        audit_cfg["data"]["test_n_per_class"] = min(int(audit_cfg["data"].get("test_n_per_class", 8)), 1)
        audit_cfg["max_candidates"] = min(int(audit_cfg.get("max_candidates", 96)), 24)
        audit_cfg["augmentation_views"] = min(int(audit_cfg.get("augmentation_views", 5)), 2)
        audit_cfg["random_draws"] = min(int(audit_cfg.get("random_draws", 100)), 5)
    elif full_resample:
        # Full resampling audit: keep candidate search and a real per-seed held-out set,
        # but reduce the random matched-text baseline repeats to keep runtime bounded.
        audit_cfg.setdefault("data", {})
        full_test_n = int(cfg.get("full_resample_test_n_per_class", cfg.get("data", {}).get("test_n_per_class", 8)))
        audit_cfg["data"]["test_n_per_class"] = max(8, full_test_n)
        audit_cfg["random_draws"] = int(cfg.get("full_resample_random_draws", 25))

    before_cache = _cache_summary(seed_root)["cache_files"]
    start = time.perf_counter()
    print(f"[benchmark-audit] running seed {seed_id} (resample={resample_benchmark}, lite={lite}, full={full_resample})", flush=True)
    outputs = run(audit_cfg)
    total_time = time.perf_counter() - start
    metrics = pd.read_csv(outputs["metrics"])
    run_dir = Path(outputs["metrics"]).parent
    rankings = pd.read_csv(outputs["rankings"]) if Path(outputs["rankings"]).exists() else pd.DataFrame()
    row = _row(_lookup(metrics), seed_id)
    row["image_set_hash"] = image_hash
    row["metadata_hash"] = metadata_hash
    row["benchmark_resampled"] = bool(resample_benchmark)
    row["fixed_benchmark_determinism_check"] = bool(lite and not resample_benchmark)
    row["lite_mode"] = bool(lite)
    row["full_resample"] = bool(full_resample)
    row["candidate_signature_hash"] = _candidate_signature(rankings)
    after_cache = _cache_summary(seed_root)["cache_files"]
    row["cache_hits_estimated"] = int(before_cache if resume else 0)
    row["cache_misses_estimated"] = int(max(0, after_cache - before_cache))
    row["total_time_sec"] = total_time
    timing_path = run_dir / "hard_multidecoy_timing_profile.csv"
    if timing_path.exists():
        timing = pd.read_csv(timing_path).iloc[0].to_dict()
        row.update({k: v for k, v in timing.items() if k.endswith("_sec")})
        print(
            f"[benchmark-audit] timing seed {seed_id}: "
            f"generation={timing.get('generation_time_sec', float('nan')):.2f}s "
            f"scoring+random+report={timing.get('clip_prediction_candidate_cic_random_time_sec', float('nan')):.2f}s "
            f"total={total_time:.2f}s",
            flush=True,
        )
    pd.DataFrame([row]).to_csv(row_file, index=False)
    run_label = f"full_seed_{seed_id}" if full_resample else f"seed_{seed_id}"
    write_repair_vs_localization_crosstab(run_dir, out_dir, seed_id=seed_id, run_label=run_label)
    print(f"[benchmark-audit] finished seed {seed_id}: {row_file}", flush=True)
    return row


def run_seed_stability(
    cfg: dict[str, Any],
    seeds: list[int],
    out_dir: Path,
    *,
    lite: bool = False,
    resume: bool = False,
    resample_benchmark: bool = False,
    full_resample: bool = False,
) -> pd.DataFrame:
    rows = [_run_one(cfg, seed_id, out_dir, lite=lite, resume=resume, resample_benchmark=resample_benchmark, full_resample=full_resample) for seed_id in seeds]
    df = pd.DataFrame(rows)
    if resample_benchmark and len(df):
        duplicated_image = df["image_set_hash"].duplicated(keep=False)
        duplicated_meta = df["metadata_hash"].duplicated(keep=False)
        df["benchmark_resampled"] = ~(duplicated_image | duplicated_meta)
        if "candidate_signature_hash" in df and not duplicated_image.any():
            all_same_candidates = df["candidate_signature_hash"].nunique(dropna=True) <= 1 and len(df) > 1
            if all_same_candidates:
                df["failed_reasons"] = df["failed_reasons"].fillna("").astype(str) + "; CIC candidate selections identical across resampled seeds"
    return df


def run_enlarged_test(cfg: dict[str, Any], seed_id: int, test_n_per_class: int) -> pd.DataFrame:
    enlarged = copy.deepcopy(cfg)
    enlarged.setdefault("data", {})["test_n_per_class"] = int(test_n_per_class)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "hard_multidecoy_clip_repair")
    row = _run_one(enlarged, seed_id, out_dir, lite=False, resume=False, resample_benchmark=False)
    return pd.DataFrame([row])


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    def _num(col: str) -> pd.Series:
        if col in df:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)

    original = _num("original_hard_misleading_accuracy")
    top1 = _num("cic_top1_repair_accuracy")
    random = _num("random_matched_text_repair_mean")
    clean_drop = _num("clean_safe_clean_drop")
    gap = top1 - random

    def _std(series: pd.Series) -> float:
        return float(series.std(ddof=1)) if len(series.dropna()) > 1 else 0.0

    n_seeds = int(len(df))
    row = {
        "n_seeds": n_seeds,
        "original_accuracy_mean": float(original.mean()),
        "original_accuracy_std": _std(original),
        "original_accuracy_min": float(original.min()),
        "original_accuracy_max": float(original.max()),
        "cic_top1_mean": float(top1.mean()),
        "cic_top1_std": _std(top1),
        "cic_top1_min": float(top1.min()),
        "cic_top1_max": float(top1.max()),
        "random_matched_mean": float(random.mean()),
        "random_matched_std": _std(random),
        "random_matched_min": float(random.min()),
        "random_matched_max": float(random.max()),
        "cic_minus_random_gap_mean": float(gap.mean()),
        "cic_minus_random_gap_std": _std(gap),
        "cic_minus_random_gap_min": float(gap.min()),
        "cic_minus_random_gap_max": float(gap.max()),
        "mean_cic_minus_random_gap": float(gap.mean()),
        "clean_safe_drop_mean": float(clean_drop.mean()),
        "clean_safe_drop_std": _std(clean_drop),
        "n_seeds_original_accuracy_le_0_40": int((original <= 0.40).sum()),
        "n_seeds_cic_top1_ge_0_80": int((top1 >= 0.80).sum()),
        "n_seeds_cic_beats_random_ge_0_15": int((gap >= 0.15).sum()),
        "n_eligible_seeds": int(df["headline_eligible"].astype(bool).sum()) if "headline_eligible" in df else 0,
        "all_seeds_benchmark_resampled": bool(df["benchmark_resampled"].astype(bool).all()) if "benchmark_resampled" in df else False,
        "lite_mode": bool(df["lite_mode"].astype(bool).any()) if "lite_mode" in df else False,
        "full_resample": bool(df["full_resample"].astype(bool).all()) if "full_resample" in df else False,
    }
    row["headline_survives_benchmark_resampling"] = bool(
        (not row["lite_mode"])
        and row["all_seeds_benchmark_resampled"]
        and row["n_eligible_seeds"] == n_seeds
        and row["mean_cic_minus_random_gap"] > 0
    )
    # Full benchmark-resampling stability requires a genuinely resampled, non-lite,
    # multi-seed audit in which CIC beats matched random text repair on every seed.
    row["full_benchmark_resampling_stability_supported"] = bool(
        row["full_resample"]
        and (not row["lite_mode"])
        and row["all_seeds_benchmark_resampled"]
        and n_seeds >= 2
        and row["n_seeds_cic_beats_random_ge_0_15"] == n_seeds
        and row["n_seeds_original_accuracy_le_0_40"] == n_seeds
    )
    return pd.DataFrame([row])


def _write_summary(df: pd.DataFrame, out_dir: Path, stem: str, title: str, *, resample_benchmark: bool = False, full_resample: bool = False) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    aggregate = _aggregate(df) if resample_benchmark else pd.DataFrame()
    if resample_benchmark and not aggregate.empty:
        aggregate_name = "full_benchmark_resampling_aggregate.csv" if full_resample else "benchmark_resampling_aggregate.csv"
        aggregate.to_csv(out_dir / aggregate_name, index=False)
    lines = [
        f"# {title}",
        "",
        "Frozen hard multi-decoy CLIP repair audit. The generation policy and repair policy are fixed before audit seeds are evaluated.",
        "",
    ]
    if resample_benchmark:
        supported = bool(aggregate["full_benchmark_resampling_stability_supported"].iloc[0]) if not aggregate.empty and "full_benchmark_resampling_stability_supported" in aggregate else False
        lines.extend(
            [
                f"all_seeds_benchmark_resampled = `{bool(df['benchmark_resampled'].astype(bool).all()) if 'benchmark_resampled' in df else False}`",
                "",
                "Benchmark-resampling hashes are reported per seed. Identical image-set hashes invalidate any benchmark-resampling stability claim.",
                "",
            ]
        )
        if full_resample:
            lines.extend(
                [
                    "This is the full benchmark-resampling audit: each seed draws an independently resampled held-out hard "
                    "multi-decoy benchmark instance (new object sequence, harmful wrong class, decoy words, placements, image IDs, "
                    "and image hashes) using the frozen generation and repair policies. The random matched-text baseline repeats "
                    f"were reduced to keep runtime bounded; see `random_matched_text_repair_std`/`_95ci` per seed.",
                    "",
                    f"full_benchmark_resampling_stability_supported = `{supported}`",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "When `lite_mode` is true, this is a small-n, volatile benchmark-resampling check, not full stability evidence.",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "fixed_benchmark_determinism_check = `True`",
                "",
                "The previous two-seed lite pass produced identical core metrics, indicating deterministic evaluation on a fixed benchmark instance. It does not establish robustness to benchmark resampling.",
                "",
            ]
        )
    lines.append(_markdown_table(df))
    if resample_benchmark and not aggregate.empty:
        lines.extend(["", "## Aggregate", "", _markdown_table(aggregate)])
    (out_dir / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hard_multidecoy_clip_repair.yaml")
    parser.add_argument("--mode", choices=["seed_stability", "enlarged"], default="seed_stability")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--resample-benchmark", action="store_true")
    parser.add_argument("--resample-benchmark-full", action="store_true")
    parser.add_argument("--enlarged-seed", type=int, default=0)
    parser.add_argument("--test-n-per-class", type=int, default=24)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "hard_multidecoy_clip_repair")
    write_repair_vs_localization_crosstab(out_dir, out_dir, seed_id="main", run_label="main")
    if args.mode == "seed_stability":
        full_resample = bool(args.resample_benchmark_full)
        resample_benchmark = bool(args.resample_benchmark) or full_resample
        seeds = list(args.seeds)
        if args.max_seeds is not None:
            seeds = seeds[: max(0, int(args.max_seeds))]
        df = run_seed_stability(cfg, seeds, out_dir, lite=bool(args.lite), resume=bool(args.resume), resample_benchmark=resample_benchmark, full_resample=full_resample)
        if full_resample:
            stem = "full_benchmark_resampling_audit"
            title = "Hard Multi-Decoy CLIP Full Benchmark-Resampling Audit"
        elif resample_benchmark:
            stem = "benchmark_resampling_audit"
            title = "Hard Multi-Decoy CLIP Benchmark-Resampling Audit"
        else:
            stem = "fixed_benchmark_determinism_check_summary" if args.lite else "seed_stability_summary"
            title = "Hard Multi-Decoy CLIP Fixed-Benchmark Determinism Check" if args.lite else "Hard Multi-Decoy CLIP Seed Stability Summary"
        _write_summary(df, out_dir, stem, title, resample_benchmark=resample_benchmark, full_resample=full_resample)
        timing_rows = []
        for seed_id in seeds:
            row_path = _row_path(out_dir, seed_id, lite=bool(args.lite), resample_benchmark=resample_benchmark, full_resample=full_resample)
            if row_path.exists():
                timing_rows.append(pd.read_csv(row_path).iloc[0].to_dict())
        if timing_rows:
            timing_cols = [c for c in pd.DataFrame(timing_rows).columns if c.endswith("_sec") or c in {"seed_id"}]
            pd.DataFrame(timing_rows)[timing_cols].to_csv(out_dir / "audit_timing_profile.csv", index=False)
        print(out_dir / f"{stem}.csv")
    else:
        df = run_enlarged_test(cfg, args.enlarged_seed, args.test_n_per_class)
        _write_summary(df, out_dir, "enlarged_test_summary", "Hard Multi-Decoy CLIP Enlarged Held-Out Test Summary")
        print(out_dir / "enlarged_test_summary.csv")


if __name__ == "__main__":
    main()
