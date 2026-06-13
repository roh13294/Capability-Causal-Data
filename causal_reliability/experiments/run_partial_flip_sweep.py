from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import pandas as pd

from causal_reliability.analysis.metrics import auroc
from causal_reliability.data.synthetic_shapes import make_shape_task, make_vector_task
from causal_reliability.data.text_shortcuts import make_text_task
from causal_reliability.experiments.stress_utils import (
    certificate_frame,
    failure_metrics,
    plot_metric_by_x,
    score_map,
    train_for_bundle,
    write_json,
)
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir

BUILDERS = {"synthetic": make_vector_task, "vision": make_shape_task, "text": make_text_task}


def _build_bundle(task: str, data_cfg: dict[str, Any]):
    builder = BUILDERS[task]
    allowed = set(inspect.signature(builder).parameters)
    return builder(**{k: v for k, v in data_cfg.items() if k in allowed})


def _task_cfg(cfg: dict[str, Any], task: str, fraction: float) -> dict[str, Any]:
    out = {k: v for k, v in cfg.items() if k not in {"tasks", "data", task}}
    out.update(cfg.get(task, {}))
    data = dict(cfg.get("data", {}))
    data.update(out.pop("data", {}))
    data.update({"shift_mode": "partial_in_support_flip", "partial_flip_fraction": fraction})
    data.setdefault("partial_flip_strength", 1.0)
    out["data"] = data
    return out


def _high_conf_shift_risk_auc(cert_df: pd.DataFrame, threshold: float = 0.8) -> float:
    subset = cert_df[cert_df["confidence"] >= threshold]
    if len(subset) == 0 or subset["failure"].nunique() < 2:
        return float("nan")
    return auroc(score_map(subset)["ShiftRisk"], subset["failure"])


def _summary_row(task: str, fraction: float, id_metrics: dict[str, float], shifted_metrics: dict[str, float], cert_df: pd.DataFrame) -> dict[str, Any]:
    failed = cert_df[cert_df["failure"] == 1]
    row: dict[str, Any] = {
        "task": task,
        "shift_type": "partial_in_support_flip",
        "partial_flip_fraction": fraction,
        "id_accuracy": id_metrics["accuracy"],
        "shifted_accuracy": shifted_metrics["accuracy"],
        "failure_count": int(cert_df["failure"].sum()),
        "correct_count": int((1 - cert_df["failure"]).sum()),
        "mean_failed_confidence": float(failed["confidence"].mean()) if len(failed) else float("nan"),
        "failed_conf_ge_0.8": float((failed["confidence"] >= 0.8).mean()) if len(failed) else float("nan"),
        "high_conf_shift_risk_auroc": _high_conf_shift_risk_auc(cert_df),
    }
    for _, rec in failure_metrics(cert_df).iterrows():
        row[f"{rec['method']}_auroc"] = rec["failure_auroc"]
    return row


def _plot_summary(metrics: pd.DataFrame, plot_dir: Path) -> None:
    plot_metric_by_x(metrics, "partial_flip_fraction", "shifted_accuracy", plot_dir / "flip_fraction_vs_shifted_accuracy.png", hue="task")
    long_rows = []
    for _, row in metrics.iterrows():
        long_rows.extend(
            [
                {"task": row["task"], "partial_flip_fraction": row["partial_flip_fraction"], "method": "confidence", "auroc": row.get("confidence_auroc")},
                {"task": row["task"], "partial_flip_fraction": row["partial_flip_fraction"], "method": "ShiftRisk", "auroc": row.get("ShiftRisk_auroc")},
            ]
        )
    long = pd.DataFrame(long_rows)
    plot_metric_by_x(long, "partial_flip_fraction", "auroc", plot_dir / "flip_fraction_vs_failure_auroc.png", hue="method")
    plot_metric_by_x(metrics, "partial_flip_fraction", "failed_conf_ge_0.8", plot_dir / "flip_fraction_vs_high_conf_failure_rate.png", hue="task")
    plot_metric_by_x(long, "partial_flip_fraction", "auroc", plot_dir / "shift_risk_vs_confidence_by_fraction.png", hue="method")


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "partial_flip_sweep")
    plot_dir = ensure_dir(out_dir / "plots")
    tasks = cfg.get("tasks", ["synthetic", "vision", "text"])
    fractions = [float(x) for x in cfg.get("partial_flip_fraction", [0.2, 0.4, 0.6, 0.8, 1.0])]
    metric_rows: list[dict[str, Any]] = []
    cert_frames = []
    failure_frames = []
    run_idx = 0
    for task in tasks:
        for fraction in fractions:
            task_cfg = _task_cfg(cfg, task, fraction)
            task_cfg["seed"] = int(cfg.get("seed", 0)) + run_idx
            bundle = _build_bundle(task, task_cfg["data"])
            model, id_metrics, shifted_metrics, device = train_for_bundle(bundle, task_cfg, int(task_cfg["seed"]))
            cert_df = certificate_frame(model, bundle, task_cfg, device)
            cert_df.insert(0, "partial_flip_fraction", fraction)
            cert_df.insert(0, "task", task)
            cert_frames.append(cert_df)
            failures = failure_metrics(cert_df)
            failures.insert(0, "partial_flip_fraction", fraction)
            failures.insert(0, "task", task)
            failure_frames.append(failures)
            metric_rows.append(_summary_row(task, fraction, id_metrics, shifted_metrics, cert_df))
            run_idx += 1
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "partial_flip_metrics.csv", index=False)
    pd.concat(cert_frames, ignore_index=True).to_csv(out_dir / "partial_flip_certificates.csv", index=False)
    pd.concat(failure_frames, ignore_index=True).to_csv(out_dir / "partial_flip_failure_prediction.csv", index=False)
    write_json(out_dir / "partial_flip_summary.json", {"metrics": metric_rows})
    _plot_summary(metrics, plot_dir)
    return {"out_dir": str(out_dir), "metrics": metric_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/partial_flip_sweep.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
