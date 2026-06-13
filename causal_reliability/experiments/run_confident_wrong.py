from __future__ import annotations

import argparse
import inspect
import json
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
from causal_reliability.data.synthetic_shapes import make_shape_task, make_vector_task
from causal_reliability.data.text_shortcuts import make_text_task
from causal_reliability.experiments.stress_utils import certificate_frame, failure_metrics, score_map, train_for_bundle, write_json
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


BUILDERS = {
    "synthetic": make_vector_task,
    "vision": make_shape_task,
    "text": make_text_task,
}


def _build_bundle(task: str, data_cfg: dict[str, Any]):
    builder = BUILDERS[task]
    allowed = set(inspect.signature(builder).parameters)
    return builder(**{k: v for k, v in data_cfg.items() if k in allowed})


def _task_cfg(cfg: dict[str, Any], task: str) -> dict[str, Any]:
    base = {k: v for k, v in cfg.items() if k not in {"tasks", "data", task}}
    base.update(cfg.get(task, {}))
    data = dict(cfg.get("data", {}))
    data.update(base.pop("data", {}))
    data["shift_mode"] = "in_support_flip"
    base["data"] = data
    return base


def _summary_row(task: str, id_metrics: dict[str, float], shifted_metrics: dict[str, float], cert_df: pd.DataFrame) -> dict[str, Any]:
    failed = cert_df[cert_df["failure"] == 1]
    correct = cert_df[cert_df["failure"] == 0]
    return {
        "task": task,
        "id_accuracy": id_metrics["accuracy"],
        "shifted_accuracy": shifted_metrics["accuracy"],
        "n_shifted": int(len(cert_df)),
        "n_failures": int(cert_df["failure"].sum()),
        "n_correct": int((1 - cert_df["failure"]).sum()),
        "failure_rate": float(cert_df["failure"].mean()),
        "mean_confidence_failed": float(failed["confidence"].mean()) if len(failed) else float("nan"),
        "mean_confidence_correct": float(correct["confidence"].mean()) if len(correct) else float("nan"),
        "failed_conf_ge_0.8": float((failed["confidence"] >= 0.8).mean()) if len(failed) else float("nan"),
        "failed_conf_ge_0.9": float((failed["confidence"] >= 0.9).mean()) if len(failed) else float("nan"),
    }


def _high_conf_rows(task: str, cert_df: pd.DataFrame, thresholds: tuple[float, ...] = (0.8, 0.9), min_n: int = 8) -> list[dict[str, Any]]:
    rows = []
    scores_all = score_map(cert_df)
    for threshold in thresholds:
        subset = cert_df[cert_df["confidence"] >= threshold]
        row: dict[str, Any] = {
            "task": task,
            "threshold": threshold,
            "n_examples": int(len(subset)),
            "n_failures": int(subset["failure"].sum()) if len(subset) else 0,
            "n_correct": int((1 - subset["failure"]).sum()) if len(subset) else 0,
            "failure_rate": float(subset["failure"].mean()) if len(subset) else float("nan"),
            "note": "" if len(subset) >= min_n else "low high-confidence count",
        }
        if len(subset) and subset["failure"].nunique() > 1:
            scores = score_map(subset)
            for name, values in scores.items():
                row[f"{name}_auroc"] = auroc(values, subset["failure"])
        else:
            for name in scores_all:
                row[f"{name}_auroc"] = float("nan")
            row["note"] = "AUROC undefined: only one class present" if len(subset) else "empty high-confidence subset"
        rows.append(row)
    return rows


def _plot_confidence_hist(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(5.2, 3.4))
    plt.hist(df[df["failure"] == 0]["confidence"], bins=16, alpha=0.65, label="correct")
    plt.hist(df[df["failure"] == 1]["confidence"], bins=16, alpha=0.65, label="failed")
    plt.xlabel("confidence")
    plt.ylabel("examples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _roc_curve(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    y = y[order]
    pos = max(int(y.sum()), 1)
    neg = max(int((1 - y).sum()), 1)
    return np.r_[0.0, np.cumsum(1 - y) / neg, 1.0], np.r_[0.0, np.cumsum(y) / pos, 1.0]


def _plot_roc(df: pd.DataFrame, path: Path, title: str) -> None:
    plt.figure(figsize=(4.4, 3.6))
    for name, values in score_map(df).items():
        fpr, tpr = _roc_curve(df["failure"], values)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_shift_risk_vs_confidence(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(4.8, 3.6))
    plt.scatter(df["confidence"], df["shift_risk"], c=df["failure"], cmap="coolwarm", s=18, alpha=0.75)
    plt.xlabel("confidence")
    plt.ylabel("ShiftRisk")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_risk_deciles(df: pd.DataFrame, path: Path) -> None:
    ranked = df.sort_values("shift_risk").copy()
    ranked["decile"] = pd.qcut(ranked["shift_risk"].rank(method="first"), 10, labels=False, duplicates="drop")
    rates = ranked.groupby("decile")["failure"].mean()
    plt.figure(figsize=(4.8, 3.4))
    plt.bar(rates.index.astype(int) + 1, rates.values)
    plt.xlabel("ShiftRisk decile")
    plt.ylabel("failure rate")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "confident_wrong")
    plot_dir = ensure_dir(out_dir / "plots")
    tasks = cfg.get("tasks", ["synthetic", "vision", "text"])
    metric_rows: list[dict[str, Any]] = []
    failure_rows: list[pd.DataFrame] = []
    high_conf_rows: list[dict[str, Any]] = []
    cert_frames: list[pd.DataFrame] = []
    for task in tasks:
        task_cfg = _task_cfg(cfg, task)
        bundle = _build_bundle(task, task_cfg.get("data", {}))
        model, id_metrics, shifted_metrics, device = train_for_bundle(bundle, task_cfg, int(task_cfg.get("seed", 0)))
        cert_df = certificate_frame(model, bundle, task_cfg, device)
        cert_df.insert(0, "task", task)
        metric_rows.append(_summary_row(task, id_metrics, shifted_metrics, cert_df))
        failure = failure_metrics(cert_df)
        failure.insert(0, "task", task)
        failure_rows.append(failure)
        high_conf_rows.extend(_high_conf_rows(task, cert_df, min_n=int(cfg.get("min_high_conf_examples", 8))))
        cert_frames.append(cert_df)
    metrics = pd.DataFrame(metric_rows)
    certs = pd.concat(cert_frames, ignore_index=True)
    failures = pd.concat(failure_rows, ignore_index=True)
    high_conf = pd.DataFrame(high_conf_rows)
    metrics.to_csv(out_dir / "confident_wrong_metrics.csv", index=False)
    certs.to_csv(out_dir / "confident_wrong_certificates.csv", index=False)
    failures.to_csv(out_dir / "confident_wrong_failure_prediction.csv", index=False)
    high_conf.to_csv(out_dir / "confident_wrong_high_conf_subset.csv", index=False)
    write_json(out_dir / "confident_wrong_summary.json", {"metrics": metric_rows})
    _plot_confidence_hist(certs, plot_dir / "confidence_hist_correct_vs_failed.png")
    _plot_roc(certs, plot_dir / "roc_all_examples.png", "All shifted examples")
    hc = certs[certs["confidence"] >= 0.8]
    if len(hc):
        _plot_roc(hc, plot_dir / "roc_high_conf_subset.png", "Confidence >= 0.8")
    _plot_shift_risk_vs_confidence(certs, plot_dir / "shift_risk_vs_confidence.png")
    _plot_risk_deciles(certs, plot_dir / "risk_decile_failure.png")
    return {"out_dir": str(out_dir), "metrics": metric_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/confident_wrong.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
