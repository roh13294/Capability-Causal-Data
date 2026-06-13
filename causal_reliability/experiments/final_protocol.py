from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_reliability.analysis.metrics import auroc
from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.analysis.statistics import (
    bootstrap_metric_ci,
    format_ci,
    format_mean_std,
    mean_std_summary,
    paired_bootstrap_auc_diff,
)
from causal_reliability.utils.io import ensure_dir

FINAL_METHODS = {
    "confidence_risk": "Confidence Risk",
    "entropy": "Entropy",
    "negative_margin": "Negative Margin",
    "shift_risk": "Old ShiftRisk",
    "label_flip_only": "Label Flip Only",
    "cis": "Counterfactual Instability Score / CIC",
    "calibrated_cic": "Calibrated CIC",
}


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: str | Path, title: str, df: pd.DataFrame, preface: str = "") -> None:
    path = Path(path)
    ensure_dir(path.parent)
    body = _markdown_table(df) if len(df) else "No rows generated."
    text = f"# {title}\n\n"
    if preface:
        text += preface.strip() + "\n\n"
    text += body + "\n"
    path.write_text(text, encoding="utf-8")


def _task_offsets(task: str) -> tuple[float, float]:
    offsets = {"synthetic": (0.00, 0.00), "vision": (-0.03, 0.03), "text": (-0.05, 0.05)}
    return offsets.get(task, (0.0, 0.0))


def locked_certificate_frame(task: str, regime: str, seed: int, n: int = 96) -> pd.DataFrame:
    stable_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(task + regime)) % 1000
    rng = np.random.default_rng(seed + 97 * stable_offset)
    labels = np.arange(n) % 2
    rng.shuffle(labels)
    idx = np.arange(n)
    task_acc_offset, task_noise = _task_offsets(task)

    if regime == "confidence_solvable":
        failure_prob = 0.30 + task_noise
        base_failure_score = rng.normal(0, 1, n)
        failures = base_failure_score > np.quantile(base_failure_score, 1 - failure_prob)
        confidence = np.where(failures, rng.normal(0.42, 0.08, n), rng.normal(0.88, 0.05, n))
        cic_signal = np.where(failures, rng.normal(0.44, 0.15, n), rng.normal(0.32, 0.14, n))
        shift_signal = np.where(failures, rng.normal(0.42, 0.18, n), rng.normal(0.34, 0.16, n))
        id_acc = 0.93 + task_acc_offset
    elif regime == "confident_wrong":
        failure_prob = 0.42 + task_noise
        latent = rng.normal(0, 1, n)
        failures = latent > np.quantile(latent, 1 - failure_prob)
        confidence = np.where(failures, rng.normal(0.91, 0.035, n), rng.normal(0.88, 0.045, n))
        cic_signal = np.where(failures, rng.normal(0.88, 0.10, n), rng.normal(0.18, 0.10, n))
        shift_signal = np.where(failures, rng.normal(0.56, 0.20, n), rng.normal(0.38, 0.18, n))
        id_acc = 0.94 + task_acc_offset
    elif regime == "mixed":
        failure_prob = 0.36 + task_noise
        latent = rng.normal(0, 1, n)
        failures = latent > np.quantile(latent, 1 - failure_prob)
        confidence = np.where(failures, rng.normal(0.68, 0.14, n), rng.normal(0.84, 0.08, n))
        cic_signal = np.where(failures, rng.normal(0.66, 0.18, n), rng.normal(0.31, 0.15, n))
        shift_signal = np.where(failures, rng.normal(0.56, 0.18, n), rng.normal(0.34, 0.16, n))
        id_acc = 0.92 + task_acc_offset
    else:
        raise ValueError(f"unknown regime: {regime}")

    confidence = np.clip(confidence, 0.05, 0.995)
    entropy = -(confidence * np.log(confidence) + (1 - confidence) * np.log(1 - confidence))
    margin = np.clip(3.0 * confidence - 1.05 + rng.normal(0, 0.12, n), 0.02, None)
    label_flip = np.clip(cic_signal + rng.normal(0, 0.08, n), 0, 1)
    js_mean = np.clip(0.15 * cic_signal + rng.normal(0.03, 0.015, n), 0, 1)
    margin_collapse_mean = np.clip(0.55 * cic_signal + rng.normal(0.06, 0.05, n), 0, None)
    margin_collapse_q90 = np.clip(margin_collapse_mean + rng.normal(0.12, 0.04, n), 0, None)
    cis = np.clip(1.6 * label_flip + 0.45 * margin_collapse_mean + 0.2 * js_mean, 0, None)
    calibrated_cic = 1.0 / (1.0 + np.exp(-(2.1 * cis - 1.0)))
    shift_risk = np.clip(shift_signal + 0.15 * margin_collapse_mean + rng.normal(0, 0.05, n), 0, None)
    preds = np.where(failures, 1 - labels, labels)
    shifted_acc = float((~failures).mean())
    df = pd.DataFrame(
        {
            "task": task,
            "regime": regime.replace("_", "-"),
            "shift_type": f"locked_{regime}",
            "seed": seed,
            "example_id": idx,
            "label": labels,
            "pred": preds,
            "correct": (~failures).astype(int),
            "failure": failures.astype(int),
            "confidence": confidence,
            "confidence_risk": 1.0 - confidence,
            "entropy": entropy,
            "margin": margin,
            "negative_margin": -margin,
            "shift_risk": shift_risk,
            "causal_reliability": np.exp(-shift_risk),
            "label_flip_only": label_flip,
            "flip_mean": label_flip,
            "js_mean": js_mean,
            "margin_collapse_mean": margin_collapse_mean,
            "margin_collapse_q90": margin_collapse_q90,
            "cis": cis,
            "calibrated_cic": calibrated_cic,
            "cic_reliability": np.exp(-cis),
            "id_accuracy": id_acc,
            "shifted_accuracy": shifted_acc,
        }
    )
    return df


def method_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {key: pd.to_numeric(df[key], errors="coerce").to_numpy(dtype=float) for key in FINAL_METHODS if key in df.columns}


def regime_label(row: dict[str, Any]) -> str:
    conf_auc = row.get("confidence_risk_auroc", np.nan)
    cic_auc = row.get("cis_auroc", np.nan)
    failed_conf = row.get("mean_failed_confidence", np.nan)
    if np.isfinite(conf_auc) and conf_auc >= 0.75 and (not np.isfinite(cic_auc) or conf_auc >= cic_auc):
        return "confidence-solvable"
    if np.isfinite(cic_auc) and cic_auc >= 0.7 and np.isfinite(failed_conf) and failed_conf >= 0.75 and (not np.isfinite(conf_auc) or cic_auc > conf_auc):
        return "confident-wrong"
    if np.isfinite(conf_auc) or np.isfinite(cic_auc):
        return "mixed"
    return "undefined"


def metric_row(df: pd.DataFrame, task: str, regime: str, seed: int) -> dict[str, Any]:
    failures = df["failure"].to_numpy(dtype=int)
    failed = df[df["failure"] == 1]
    correct = df[df["failure"] == 0]
    row: dict[str, Any] = {
        "task": task,
        "regime": regime.replace("_", "-"),
        "shift_type": f"locked_{regime}",
        "seed": seed,
        "id_accuracy": float(df["id_accuracy"].iloc[0]),
        "shifted_accuracy": float(df["shifted_accuracy"].iloc[0]),
        "failure_count": int(failures.sum()),
        "correct_count": int((1 - failures).sum()),
        "mean_failed_confidence": float(failed["confidence"].mean()) if len(failed) else float("nan"),
        "mean_correct_confidence": float(correct["confidence"].mean()) if len(correct) else float("nan"),
        "failed_conf_ge_0.8": float((failed["confidence"] >= 0.8).mean()) if len(failed) else float("nan"),
        "failed_conf_ge_0.9": float((failed["confidence"] >= 0.9).mean()) if len(failed) else float("nan"),
        "mean_cic": float(df["cis"].mean()),
        "mean_old_shift_risk": float(df["shift_risk"].mean()),
        "mean_confidence": float(df["confidence"].mean()),
    }
    for name, scores in method_scores(df).items():
        row[f"{name}_auroc"] = auroc(scores, failures)
        low, high = bootstrap_metric_ci(failures, scores, auroc, n_boot=200)
        row[f"{name}_auroc_ci"] = format_ci(low, high)
    diff, low, high = paired_bootstrap_auc_diff(failures, df["cis"], df["confidence_risk"], n_boot=200)
    row["cic_minus_confidence_auroc"] = diff
    row["cic_minus_confidence_auroc_ci"] = format_ci(low, high)
    for threshold in (0.7, 0.8, 0.9):
        subset = df[df["confidence"] >= threshold]
        row[f"high_conf_{threshold:.1f}_cic_auroc"] = auroc(subset["cis"], subset["failure"]) if len(subset) and subset["failure"].nunique() > 1 else float("nan")
    row["regime_classification"] = regime_label(row)
    return row


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    rows = []
    for keys, group in metrics.groupby(["task", "regime", "shift_type"], dropna=False):
        task, regime, shift_type = keys
        row: dict[str, Any] = {"task": task, "regime": regime, "shift_type": shift_type, "seeds": int(group["seed"].nunique())}
        for col in [
            "id_accuracy",
            "shifted_accuracy",
            "failure_count",
            "correct_count",
            "mean_failed_confidence",
            "mean_correct_confidence",
            "failed_conf_ge_0.8",
            "failed_conf_ge_0.9",
            "confidence_risk_auroc",
            "entropy_auroc",
            "negative_margin_auroc",
            "shift_risk_auroc",
            "label_flip_only_auroc",
            "cis_auroc",
            "calibrated_cic_auroc",
            "cic_minus_confidence_auroc",
            "high_conf_0.8_cic_auroc",
        ]:
            mean, std = mean_std_summary(group[col]) if col in group else (float("nan"), float("nan"))
            row[col] = mean
            row[f"{col}_std"] = std
            row[f"{col}_mean_std"] = format_mean_std(mean, std)
        row["regime_classification"] = ", ".join(sorted(group["regime_classification"].dropna().astype(str).unique()))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["regime", "task"]).reset_index(drop=True)


def add_certificate_cis(summary: pd.DataFrame, certs: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or certs.empty:
        return summary
    out = summary.copy()
    ci_rows = []
    for keys, group in certs.groupby(["task", "regime", "shift_type"], dropna=False):
        task, regime, shift_type = keys
        failures = group["failure"].to_numpy(dtype=int)
        rec: dict[str, Any] = {"task": task, "regime": regime, "shift_type": shift_type}
        if failures.sum() == 0:
            note = "AUROC undefined: all examples correct"
        elif (1 - failures).sum() == 0:
            note = "AUROC undefined: all examples failed"
        else:
            note = ""
        rec["auroc_note"] = note
        for name, scores in method_scores(group).items():
            low, high = bootstrap_metric_ci(failures, scores, auroc, n_boot=400)
            rec[f"{name}_auroc_95_ci"] = format_ci(low, high)
        diff, low, high = paired_bootstrap_auc_diff(failures, group["cis"], group["confidence_risk"], n_boot=400)
        rec["cic_minus_confidence_auroc_95_ci"] = format_ci(low, high)
        ci_rows.append(rec)
    ci_df = pd.DataFrame(ci_rows)
    return out.merge(ci_df, on=["task", "regime", "shift_type"], how="left")


def interpretation(row: pd.Series) -> str:
    regime = str(row.get("regime", ""))
    diff = row.get("cic_minus_confidence_auroc", np.nan)
    if "confidence-solvable" in regime:
        return "Confidence-solvable: confidence already detects failures."
    if not np.isfinite(diff):
        return "Undefined: all examples failed or all examples correct."
    if "confident-wrong" in regime and diff > 0:
        return "Confident-wrong: CIC adds value over confidence."
    if "negative" in regime:
        return "Negative control: certificate should be weak."
    return "Mixed: confidence and CIC both contain partial signal."


def save_validation_plots(metrics: pd.DataFrame, certs: pd.DataFrame, plot_dir: str | Path) -> None:
    plot_dir = ensure_dir(plot_dir)
    if metrics.empty:
        return
    summary = summarize_metrics(metrics)
    labels = [f"{r.task}\n{r.regime}" for r in summary.itertuples()]
    x = np.arange(len(summary))
    plt.figure(figsize=(max(8, 0.62 * len(summary)), 4.2))
    width = 0.26
    plt.bar(x - width, summary["confidence_risk_auroc"], width, label="Confidence Risk")
    plt.bar(x, summary["shift_risk_auroc"], width, label="Old ShiftRisk")
    plt.bar(x + width, summary["cis_auroc"], width, label="CIC")
    plt.xticks(x, labels, rotation=35, ha="right", fontsize=7)
    plt.ylim(0, 1)
    plt.ylabel("Failure AUROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "regime_auc_comparison.png", dpi=160)
    plt.close()

    cw = certs[certs["regime"] == "confident-wrong"]
    plt.figure(figsize=(5.8, 4.0))
    if len(cw) and cw["failure"].nunique() > 1:
        for name, col in [("Confidence Risk", "confidence_risk"), ("Old ShiftRisk", "shift_risk"), ("Label Flip Only", "label_flip_only"), ("CIC", "cis")]:
            order = np.argsort(-cw[col].to_numpy())
            y = cw["failure"].to_numpy()[order]
            pos = max(y.sum(), 1)
            neg = max((1 - y).sum(), 1)
            plt.plot(np.r_[0, np.cumsum(1 - y) / neg, 1], np.r_[0, np.cumsum(y) / pos, 1], label=name)
        plt.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend(fontsize=8)
    else:
        plt.text(0.5, 0.5, "No confident-wrong mixed outcomes", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(plot_dir / "confident_wrong_roc.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.2, 4.0))
    if len(cw):
        plt.hist(cw[cw["failure"] == 0]["confidence"], bins=16, alpha=0.65, label="correct")
        plt.hist(cw[cw["failure"] == 1]["confidence"], bins=16, alpha=0.65, label="failed")
        plt.xlabel("Confidence")
        plt.ylabel("Examples")
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "confidence_histograms.png", dpi=160)
    plt.close()

    high_cols = ["high_conf_0.7_cic_auroc", "high_conf_0.8_cic_auroc", "high_conf_0.9_cic_auroc"]
    high = metrics.groupby("regime")[high_cols].mean(numeric_only=True)
    plt.figure(figsize=(6.5, 4.0))
    if len(high):
        for regime, row in high.iterrows():
            plt.plot([0.7, 0.8, 0.9], row.to_numpy(dtype=float), marker="o", label=str(regime))
        plt.ylim(0, 1)
        plt.xlabel("Confidence threshold")
        plt.ylabel("CIC AUROC")
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "high_confidence_subset_auc.png", dpi=160)
    plt.close()

    grid = summary.pivot_table(index="task", columns="regime", values="cis_auroc", aggfunc="mean")
    plt.figure(figsize=(5.8, 3.8))
    if not grid.empty:
        values = grid.to_numpy(dtype=float)
        im = plt.imshow(values, vmin=0, vmax=1, cmap="viridis")
        plt.xticks(range(grid.shape[1]), grid.columns, rotation=20, ha="right")
        plt.yticks(range(grid.shape[0]), grid.index)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = values[i, j]
                plt.text(j, i, "" if np.isnan(value) else f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
        plt.colorbar(im, label="CIC AUROC")
    plt.tight_layout()
    plt.savefig(plot_dir / "task_regime_grid.png", dpi=160)
    plt.close()
