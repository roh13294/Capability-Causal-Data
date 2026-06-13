from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causal_reliability.analysis.metrics import auroc
from causal_reliability.analysis.statistics import (
    bootstrap_ci,
    bootstrap_metric_ci,
    format_ci,
    format_mean_std,
    mean_std_summary,
    paired_bootstrap_auc_diff,
    risk_ratio_ci,
)
from causal_reliability.experiments.stress_utils import score_map
from causal_reliability.utils.io import ensure_dir


METHOD_COLUMNS = {
    "confidence": "Confidence AUROC mean +/- std",
    "entropy": "Entropy AUROC mean +/- std",
    "margin": "Margin AUROC mean +/- std",
    "ShiftRisk": "ShiftRisk AUROC mean +/- std",
}

FINAL_TABLE_COLUMNS = [
    "Task",
    "Regime",
    "Shift Type",
    "Seeds",
    "ID Accuracy",
    "Shifted Accuracy",
    "Failure Count",
    "Correct Count",
    "Mean Confidence",
    "Mean Counterfactual Stability",
    "Mean Failed Confidence",
    "Failed Conf >= 0.8",
    "Confidence AUROC",
    "Confidence AUROC mean +/- std",
    "Confidence AUROC 95% CI",
    "Entropy AUROC",
    "Margin AUROC",
    "Old ShiftRisk AUROC",
    "Label Flip AUROC",
    "CIC AUROC",
    "CIC AUROC mean +/- std",
    "CIC AUROC 95% CI",
    "CIC - Confidence AUROC",
    "CIC - Confidence AUROC mean +/- std",
    "CIC - Confidence AUROC 95% CI",
    "High-Confidence CIC AUROC",
    "Dangerous Quadrant Failure Rate",
    "Dangerous Quadrant Count",
    "Shortcut Discovery Top-1 Hit",
    "Shortcut Discovery Top-3 Hit",
    "Model Type",
    "Backend",
    "Pretrained?",
    "Zero-shot?",
    "Aligned Accuracy",
    "Misleading Accuracy",
    "Mixed Accuracy",
    "Linear Probe?",
    "Real Model Validation?",
    "Shortcut Attention Ratio",
    "Evidence Status",
    "Interpretation",
]


def _read_csvs(results_dir: Path, pattern: str) -> list[tuple[Path, pd.DataFrame]]:
    out = []
    for path in sorted(results_dir.rglob(pattern)):
        try:
            out.append((path, pd.read_csv(path)))
        except pd.errors.EmptyDataError:
            continue
    return out


def _infer_task(path: Path, results_dir: Path) -> str:
    rel = path.relative_to(results_dir)
    return rel.parts[0] if len(rel.parts) > 1 else "unknown"


def _infer_model(path: Path, df: pd.DataFrame | None = None) -> str:
    if df is not None and "model" in df.columns and df["model"].nunique() == 1:
        return str(df["model"].iloc[0])
    name = path.stem.lower()
    if "stability" in name or "stab" in name:
        return "stability"
    return "erm"


def _metrics_from_certificates(path: Path, df: pd.DataFrame, results_dir: Path) -> dict[str, Any]:
    if "failure" not in df.columns and {"pred", "label"}.issubset(df.columns):
        df = df.copy()
        df["failure"] = (df["pred"] != df["label"]).astype(int)
    task = str(df["task"].iloc[0]) if "task" in df.columns and df["task"].nunique() == 1 else _infer_task(path, results_dir)
    scores = score_map(df)
    failures = df["failure"].to_numpy(dtype=int)
    rows: dict[str, Any] = {
        "Task": task,
        "Dataset": task,
        "Model": _infer_model(path),
        "Seed": _seed_from_name(path),
        "Shift Type": str(df["shift_type"].iloc[0]) if "shift_type" in df.columns and df["shift_type"].nunique() == 1 else "",
        "Flip Fraction": float(df["partial_flip_fraction"].iloc[0]) if "partial_flip_fraction" in df.columns and df["partial_flip_fraction"].nunique() == 1 else np.nan,
        "Negative Control Type": str(df["control"].iloc[0]) if "control" in df.columns and df["control"].nunique() == 1 else "",
        "Mean ShiftRisk": float(df["shift_risk"].mean()) if "shift_risk" in df.columns else float("nan"),
        "Mean CR": float(df["causal_reliability"].mean()) if "causal_reliability" in df.columns else float("nan"),
        "Failure Count": int(failures.sum()),
        "Correct Count": int((1 - failures).sum()),
        "Mean Failed Confidence": float(df.loc[df["failure"] == 1, "confidence"].mean()) if (failures == 1).any() else float("nan"),
        "Mean Confidence on Failures": float(df.loc[df["failure"] == 1, "confidence"].mean()) if (failures == 1).any() else float("nan"),
        "Mean Confidence on Correct": float(df.loc[df["failure"] == 0, "confidence"].mean()) if (failures == 0).any() else float("nan"),
        "Failed Conf >= 0.8": float((df.loc[df["failure"] == 1, "confidence"] >= 0.8).mean()) if (failures == 1).any() else float("nan"),
        "High-Confidence Failure Fraction": float(df.loc[df["confidence"] >= 0.8, "failure"].mean()) if (df["confidence"] >= 0.8).any() else float("nan"),
    }
    for method, values in scores.items():
        rows[f"{method}_auroc"] = auroc(values, failures)
        low, high = bootstrap_metric_ci(failures, values, auroc, n_boot=300)
        rows[f"{method}_auroc_ci_low"] = low
        rows[f"{method}_auroc_ci_high"] = high
    if "ShiftRisk" in scores and "confidence" in scores:
        diff, low, high = paired_bootstrap_auc_diff(failures, scores["ShiftRisk"], scores["confidence"], n_boot=300)
        rows["ShiftRisk Advantage"] = diff
        rows["ShiftRisk - Confidence AUROC difference"] = diff
        rows["AUROC difference CI low"] = low
        rows["AUROC difference CI high"] = high
        rows["Confidence Beats ShiftRisk?"] = bool(rows.get("confidence_auroc", np.nan) > rows.get("ShiftRisk_auroc", np.nan))
        rows["ShiftRisk Beats Confidence?"] = bool(rows.get("ShiftRisk_auroc", np.nan) > rows.get("confidence_auroc", np.nan))
    high_conf = df[df["confidence"] >= 0.8]
    if len(high_conf) and high_conf["failure"].nunique() > 1 and "ShiftRisk" in score_map(high_conf):
        rows["High-Confidence ShiftRisk AUROC"] = auroc(score_map(high_conf)["ShiftRisk"], high_conf["failure"])
        rows["ShiftRisk AUROC High-Confidence Subset"] = rows["High-Confidence ShiftRisk AUROC"]
    else:
        rows["High-Confidence ShiftRisk AUROC"] = float("nan")
        rows["ShiftRisk AUROC High-Confidence Subset"] = float("nan")
    if "ShiftRisk" in scores:
        order = np.argsort(scores["ShiftRisk"])
        n_decile = max(1, len(order) // 10)
        bottom = failures[order[:n_decile]]
        top = failures[order[-n_decile:]]
        ratio, low, high = risk_ratio_ci(top, bottom, n_boot=300)
        rows["Top-Risk Failure Rate"] = float(top.mean())
        rows["Bottom-Risk Failure Rate"] = float(bottom.mean())
        rows["Risk Ratio"] = ratio
        rows["Risk Ratio CI low"] = low
        rows["Risk Ratio CI high"] = high
    return rows


def _seed_from_name(path: Path) -> int | None:
    parts = path.stem.replace("-", "_").split("_")
    for i, part in enumerate(parts[:-1]):
        if part == "seed":
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def _rows_from_metric_files(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path, df in _read_csvs(results_dir, "*metrics.csv"):
        if df.empty:
            continue
        task = _infer_task(path, results_dir)
        if {"method", "failure_auroc"}.issubset(df.columns):
            pivot = df.pivot_table(index=[c for c in ("shortcut_correlation",) if c in df.columns], columns="method", values="failure_auroc")
            for _, source in df.groupby([c for c in ("shortcut_correlation",) if c in df.columns] or (lambda x: 0)):
                first = source.iloc[0]
                row = {"Task": str(first.get("task", task)), "Dataset": task, "Model": _infer_model(path), "Seed": None}
                for col in ("id_accuracy", "shifted_accuracy", "top_risk_decile_failure_rate", "bottom_risk_decile_failure_rate", "risk_ratio", "shift_risk_mean"):
                    if col in first:
                        row[col] = first[col]
                for col in ("model_type", "pretrained", "zero_shot", "linear_probe"):
                    if col in first:
                        row[col] = first[col]
                if task == "real_model_validation":
                    row["Evidence Status"] = "pretrained CLIP evidence" if bool(first.get("pretrained", False)) and bool(first.get("zero_shot", False)) else "fallback smoke test"
                for _, mrow in source.iterrows():
                    row[f"{mrow['method']}_auroc"] = mrow["failure_auroc"]
                rows.append(row)
            _ = pivot
        else:
            for _, rec in df.iterrows():
                row = {"Task": str(rec.get("task", task)), "Dataset": task, "Model": str(rec.get("model", _infer_model(path))), "Seed": rec.get("seed", None)}
                for src, dst in (
                    ("shift_type", "Shift Type"),
                    ("partial_flip_fraction", "Flip Fraction"),
                    ("negative_control_type", "Negative Control Type"),
                    ("control", "Negative Control Type"),
                    ("id_accuracy", "id_accuracy"),
                    ("shifted_accuracy", "shifted_accuracy"),
                    ("confidence_auroc", "confidence_auroc"),
                    ("entropy_auroc", "entropy_auroc"),
                    ("margin_auroc", "margin_auroc"),
                    ("shift_risk_auroc", "ShiftRisk_auroc"),
                    ("failure_auroc", "ShiftRisk_auroc"),
                    ("mean_shift_risk", "Mean ShiftRisk"),
                    ("n_failures", "Failure Count"),
                    ("failure_count", "Failure Count"),
                    ("n_correct", "Correct Count"),
                    ("correct_count", "Correct Count"),
                    ("mean_confidence_failed", "Mean Confidence on Failures"),
                    ("mean_failed_confidence", "Mean Failed Confidence"),
                    ("mean_confidence_correct", "Mean Confidence on Correct"),
                    ("failed_conf_ge_0.8", "High-Confidence Failure Fraction"),
                    ("failed_conf_ge_0.8", "Failed Conf >= 0.8"),
                    ("high_conf_shift_risk_auroc", "High-Confidence ShiftRisk AUROC"),
                ):
                    if src in rec:
                        row[dst] = rec[src]
                rows.append(row)
    return rows


def _aggregate_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    group_cols = ["Task", "Dataset", "Model"]
    out_rows = []
    for keys, group in raw.groupby(group_cols, dropna=False):
        row: dict[str, Any] = dict(zip(group_cols, keys))
        seeds = group["Seed"].dropna().nunique() if "Seed" in group.columns else 0
        row["Seed Count"] = int(seeds)
        for src, dst in (
            ("id_accuracy", "ID Accuracy mean +/- std"),
            ("shifted_accuracy", "Shifted Accuracy mean +/- std"),
            ("confidence_auroc", METHOD_COLUMNS["confidence"]),
            ("entropy_auroc", METHOD_COLUMNS["entropy"]),
            ("margin_auroc", METHOD_COLUMNS["margin"]),
            ("ShiftRisk_auroc", METHOD_COLUMNS["ShiftRisk"]),
        ):
            mean, std = mean_std_summary(group[src]) if src in group.columns else (float("nan"), float("nan"))
            row[dst] = format_mean_std(mean, std)
            row[f"{src}_mean"] = mean
            row[f"{src}_std"] = std
        diff_values = group["ShiftRisk - Confidence AUROC difference"] if "ShiftRisk - Confidence AUROC difference" in group.columns else group.get("ShiftRisk_auroc", pd.Series(dtype=float)) - group.get("confidence_auroc", pd.Series(dtype=float))
        diff_mean, _ = mean_std_summary(diff_values)
        low, high = bootstrap_ci(diff_values, n_boot=300)
        row["ShiftRisk Advantage"] = diff_mean
        row["ShiftRisk - Confidence AUROC difference"] = diff_mean
        row["95% CI for AUROC difference"] = format_ci(low, high)
        for src, dst in (
            ("Shift Type", "Shift Type"),
            ("Flip Fraction", "Flip Fraction"),
            ("Negative Control Type", "Negative Control Type"),
            ("Top-Risk Failure Rate", "Top-Risk Failure Rate"),
            ("top_risk_decile_failure_rate", "Top-Risk Failure Rate"),
            ("Bottom-Risk Failure Rate", "Bottom-Risk Failure Rate"),
            ("bottom_risk_decile_failure_rate", "Bottom-Risk Failure Rate"),
            ("Risk Ratio", "Risk Ratio"),
            ("risk_ratio", "Risk Ratio"),
            ("Mean ShiftRisk", "Mean ShiftRisk"),
            ("shift_risk_mean", "Mean ShiftRisk"),
            ("Mean CR", "Mean CR"),
            ("Failure Count", "Failure Count"),
            ("Correct Count", "Correct Count"),
            ("Mean Confidence on Failures", "Mean Confidence on Failures"),
            ("Mean Failed Confidence", "Mean Failed Confidence"),
            ("Mean Confidence on Correct", "Mean Confidence on Correct"),
            ("High-Confidence Failure Fraction", "High-Confidence Failure Fraction"),
            ("Failed Conf >= 0.8", "Failed Conf >= 0.8"),
            ("High-Confidence ShiftRisk AUROC", "High-Confidence ShiftRisk AUROC"),
            ("ShiftRisk AUROC High-Confidence Subset", "ShiftRisk AUROC High-Confidence Subset"),
            ("Evidence Status", "Evidence Status"),
        ):
            if dst in row or src not in group.columns:
                continue
            if group[src].dtype == object:
                row[dst] = ", ".join(sorted(str(x) for x in group[src].dropna().unique() if str(x)))
            else:
                row[dst] = mean_std_summary(group[src])[0]
        for src in ("Confidence Beats ShiftRisk?", "ShiftRisk Beats Confidence?"):
            if src in group.columns:
                row[src] = bool(group[src].fillna(False).any())
        if row["Model"] == "stability" and pd.isna(row.get("Mean CR", np.nan)):
            row["Certificate Status"] = "not computed"
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values(["Task", "Model"]).reset_index(drop=True)


def build_main_table(results_dir: str | Path = "results") -> pd.DataFrame:
    results_path = Path(results_dir)
    final_path = results_path / "final_validation" / "final_validation_summary.csv"
    if final_path.exists():
        final = pd.read_csv(final_path)
        plane_points = _read_csvs(results_path / "reliability_plane", "reliability_plane_points.csv")
        plane_quadrants = _read_csvs(results_path / "reliability_plane", "reliability_plane_quadrants.csv")
        shortcut_metrics = _read_csvs(results_path / "shortcut_discovery", "shortcut_discovery_metrics.csv")
        real_metrics = _read_csvs(results_path / "real_model_validation", "real_model_metrics.csv")
        real_attr = _read_csvs(results_path / "real_model_validation" / "attribution", "attribution_metrics.csv")
        clip_metrics = _read_csvs(results_path / "clip_overlay_validation", "clip_overlay_metrics.csv")
        clip_attr = _read_csvs(results_path / "clip_overlay_validation" / "attribution", "clip_overlay_occlusion_metrics.csv")
        points = plane_points[0][1] if plane_points else pd.DataFrame()
        quadrants = plane_quadrants[0][1] if plane_quadrants else pd.DataFrame()
        shortcut = shortcut_metrics[0][1] if shortcut_metrics else pd.DataFrame()
        stability_lookup = {}
        if not points.empty and {"task", "regime", "confidence", "counterfactual_stability"}.issubset(points.columns):
            for keys, group in points.groupby(["task", "regime"], dropna=False):
                stability_lookup[keys] = {
                    "Mean Confidence": float(group["confidence"].mean()),
                    "Mean Counterfactual Stability": float(group["counterfactual_stability"].mean()),
                }
        dangerous_lookup = {}
        if not quadrants.empty and {"task", "regime", "quadrant"}.issubset(quadrants.columns):
            dangerous = quadrants[quadrants["quadrant"] == "Dangerous shortcut reliance"]
            for _, row in dangerous.iterrows():
                dangerous_lookup[(row["task"], row["regime"])] = {
                    "Dangerous Quadrant Failure Rate": row.get("failure_rate", np.nan),
                    "Dangerous Quadrant Count": row.get("count", np.nan),
                }
        shortcut_top1 = shortcut["shortcut_top1_hit"].iloc[0] if "shortcut_top1_hit" in shortcut and len(shortcut) else np.nan
        shortcut_top3 = shortcut["shortcut_top3_hit"].iloc[0] if "shortcut_top3_hit" in shortcut and len(shortcut) else np.nan
        rows = []
        for _, rec in final.iterrows():
            diff = rec.get("cic_minus_confidence_auroc", np.nan)
            if pd.isna(rec.get("cis_auroc", np.nan)) or pd.isna(rec.get("confidence_risk_auroc", np.nan)):
                interp = "Undefined: all examples failed or all examples correct."
            elif str(rec.get("regime", "")).startswith("confidence-solvable"):
                interp = "Confidence-solvable: confidence already detects failures."
            elif diff > 0:
                interp = "Confident-wrong: CIC adds value over confidence."
            else:
                interp = "Mixed: confidence and CIC both contain partial signal."
            task_regime = (rec.get("task"), rec.get("regime"))
            extra = {}
            extra.update(stability_lookup.get(task_regime, {}))
            extra.update(dangerous_lookup.get(task_regime, {}))
            row = {
                    "Task": rec.get("task"),
                    "Regime": rec.get("regime"),
                    "Shift Type": rec.get("shift_type"),
                    "Seeds": rec.get("seeds"),
                    "ID Accuracy": rec.get("id_accuracy"),
                    "Shifted Accuracy": rec.get("shifted_accuracy"),
                    "Failure Count": rec.get("failure_count"),
                    "Correct Count": rec.get("correct_count"),
                    "Mean Confidence": extra.get("Mean Confidence", rec.get("mean_confidence", np.nan)),
                    "Mean Counterfactual Stability": extra.get("Mean Counterfactual Stability", np.nan),
                    "Mean Failed Confidence": rec.get("mean_failed_confidence"),
                    "Failed Conf >= 0.8": rec.get("failed_conf_ge_0.8"),
                    "Confidence AUROC": rec.get("confidence_risk_auroc"),
                    "Confidence AUROC mean +/- std": rec.get("confidence_risk_auroc_mean_std"),
                    "Confidence AUROC 95% CI": rec.get("confidence_risk_auroc_95_ci"),
                    "Entropy AUROC": rec.get("entropy_auroc"),
                    "Margin AUROC": rec.get("negative_margin_auroc"),
                    "Old ShiftRisk AUROC": rec.get("shift_risk_auroc"),
                    "Label Flip AUROC": rec.get("label_flip_only_auroc"),
                    "CIC AUROC": rec.get("cis_auroc"),
                    "CIC AUROC mean +/- std": rec.get("cis_auroc_mean_std"),
                    "CIC AUROC 95% CI": rec.get("cis_auroc_95_ci"),
                    "CIC - Confidence AUROC": diff,
                    "CIC - Confidence AUROC mean +/- std": rec.get("cic_minus_confidence_auroc_mean_std"),
                    "CIC - Confidence AUROC 95% CI": rec.get("cic_minus_confidence_auroc_95_ci"),
                    "High-Confidence CIC AUROC": rec.get("high_conf_0.8_cic_auroc"),
                    "Dangerous Quadrant Failure Rate": extra.get("Dangerous Quadrant Failure Rate", np.nan),
                    "Dangerous Quadrant Count": extra.get("Dangerous Quadrant Count", np.nan),
                    "Shortcut Discovery Top-1 Hit": shortcut_top1,
                    "Shortcut Discovery Top-3 Hit": shortcut_top3,
                    "Interpretation": rec.get("interpretation", interp),
                }
            rows.append(row)
        if real_metrics:
            real = real_metrics[0][1]
            attr = real_attr[0][1] if real_attr else pd.DataFrame()
            lookup = real.set_index("method")["failure_auroc"].to_dict() if {"method", "failure_auroc"}.issubset(real.columns) else {}
            first = real.iloc[0]
            rows.append(
                {
                    "Task": "real_model_validation",
                    "Regime": "controlled shortcut flip",
                    "Shift Type": "in-support shortcut flip",
                    "Seeds": 1,
                    "ID Accuracy": first.get("id_accuracy", np.nan),
                    "Shifted Accuracy": first.get("shifted_accuracy", np.nan),
                    "Confidence AUROC": lookup.get("confidence_risk", np.nan),
                    "Entropy AUROC": lookup.get("entropy", np.nan),
                    "Margin AUROC": lookup.get("negative_margin", np.nan),
                    "Old ShiftRisk AUROC": lookup.get("old_ShiftRisk", np.nan),
                    "Label Flip AUROC": lookup.get("label_flip_only", np.nan),
                    "CIC AUROC": lookup.get("CIC", np.nan),
                    "Model Type": first.get("model_type", ""),
                    "Backend": first.get("model", ""),
                    "Pretrained?": first.get("pretrained", np.nan),
                    "Zero-shot?": first.get("zero_shot", np.nan),
                    "Linear Probe?": first.get("linear_probe", np.nan),
                    "Real Model Validation?": bool(first.get("pretrained", False)),
                    "Shortcut Attention Ratio": float(attr["shortcut_attention_ratio"].mean()) if len(attr) and "shortcut_attention_ratio" in attr else np.nan,
                    "Evidence Status": "pretrained CLIP evidence" if bool(first.get("pretrained", False)) and bool(first.get("zero_shot", False)) else "fallback smoke test",
                    "Interpretation": "Fallback smoke test only; do not use as pretrained evidence." if not bool(first.get("pretrained", False)) else "Controlled real-model shortcut validation; supports but does not replace the main claim.",
                }
            )
        if clip_metrics:
            clip = clip_metrics[0][1]
            attr = clip_attr[0][1] if clip_attr else pd.DataFrame()
            if "method" in clip and "failure_auroc" in clip:
                lookup = clip.set_index("method")["failure_auroc"].to_dict()
                first = clip.iloc[0]
            else:
                lookup = {}
                first = clip.iloc[0]
            rows.append(
                {
                    "Task": "clip_overlay_validation",
                    "Regime": "CLIP text-overlay shortcut",
                    "Shift Type": "misleading text overlay",
                    "Seeds": 1,
                    "Aligned Accuracy": first.get("aligned_accuracy", np.nan),
                    "Misleading Accuracy": first.get("misleading_accuracy", np.nan),
                    "Mixed Accuracy": first.get("mixed_accuracy", np.nan),
                    "Confidence AUROC": lookup.get("confidence_risk", np.nan),
                    "Entropy AUROC": lookup.get("entropy", np.nan),
                    "Margin AUROC": lookup.get("negative_margin", np.nan),
                    "Label Flip AUROC": lookup.get("label_flip_only", np.nan),
                    "CIC AUROC": lookup.get("CIC", np.nan),
                    "High-Confidence CIC AUROC": first.get("high_confidence_cic_auroc", np.nan),
                    "Dangerous Quadrant Failure Rate": first.get("dangerous_quadrant_failure_rate", np.nan),
                    "Model Type": "CLIP",
                    "Backend": first.get("backend", ""),
                    "Pretrained?": first.get("pretrained", np.nan),
                    "Zero-shot?": first.get("zero_shot", np.nan),
                    "Real Model Validation?": first.get("evidence_status") == "pretrained CLIP evidence",
                    "Shortcut Attention Ratio": float(attr["shortcut_attention_ratio"].mean()) if len(attr) and "shortcut_attention_ratio" in attr else np.nan,
                    "Evidence Status": first.get("evidence_status", "unavailable"),
                    "Interpretation": "External pretrained CLIP evidence." if first.get("evidence_status") == "pretrained CLIP evidence" else "Unavailable or test-only; do not use as pretrained evidence.",
                }
            )
        return pd.DataFrame(rows, columns=FINAL_TABLE_COLUMNS)
    rows = _rows_from_metric_files(results_path)
    for path, df in _read_csvs(results_path, "certificates*.csv"):
        required = {"confidence", "margin", "shift_risk", "causal_reliability"}
        if required.issubset(df.columns):
            rows.append(_metrics_from_certificates(path, df, results_path))
    return _aggregate_rows(rows)


def save_outputs(table: pd.DataFrame, results_dir: str | Path = "results") -> None:
    out_dir = ensure_dir(results_dir)
    csv_path = out_dir / "main_results_table.csv"
    md_path = out_dir / "main_results_table.md"
    json_path = out_dir / "main_results_summary.json"
    table.to_csv(csv_path, index=False)
    if set(FINAL_TABLE_COLUMNS).issubset(table.columns):
        display_cols = FINAL_TABLE_COLUMNS
    else:
        display_cols = [
        "Task",
        "Dataset",
        "Model",
        "Seed Count",
        "Shift Type",
        "Flip Fraction",
        "ID Accuracy mean +/- std",
        "Shifted Accuracy mean +/- std",
        "Failure Count",
        "Correct Count",
        "Mean Failed Confidence",
        "Failed Conf >= 0.8",
        "Confidence AUROC mean +/- std",
        "Entropy AUROC mean +/- std",
        "Margin AUROC mean +/- std",
        "ShiftRisk AUROC mean +/- std",
        "ShiftRisk Advantage",
        "High-Confidence ShiftRisk AUROC",
        "Negative Control Type",
        "ShiftRisk - Confidence AUROC difference",
        "95% CI for AUROC difference",
        "Top-Risk Failure Rate",
        "Bottom-Risk Failure Rate",
        "Risk Ratio",
        "Mean ShiftRisk",
        "Mean CR",
        "Mean Confidence on Failures",
        "Mean Confidence on Correct",
        "High-Confidence Failure Fraction",
        "ShiftRisk AUROC High-Confidence Subset",
        "Confidence Beats ShiftRisk?",
        "ShiftRisk Beats Confidence?",
        "Certificate Status",
        ]
    existing = list(dict.fromkeys(c for c in display_cols if c in table.columns))
    md_path.write_text(_markdown_table(table[existing]) if not table.empty else "No results found.\n", encoding="utf-8")
    summary = {
        "n_rows": int(len(table)),
        "tasks": sorted(str(x) for x in table["Task"].dropna().unique()) if "Task" in table else [],
        "best_shift_risk_minus_confidence": float(table["ShiftRisk - Confidence AUROC difference"].max()) if "ShiftRisk - Confidence AUROC difference" in table and len(table) else None,
        "best_cic_minus_confidence": float(table["CIC - Confidence AUROC"].max()) if "CIC - Confidence AUROC" in table and len(table) else None,
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return "NA" if not np.isfinite(value) else f"{value:.3f}"
        if pd.isna(value):
            return "NA"
        return str(value)

    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in headers) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    table = build_main_table(args.results_dir)
    save_outputs(table, args.results_dir)
    print(table)


if __name__ == "__main__":
    main()
