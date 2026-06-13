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
from causal_reliability.experiments.stress_utils import certificate_frame, plot_metric_by_x, train_for_bundle, write_json
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir

BUILDERS = {"synthetic": make_vector_task, "vision": make_shape_task, "text": make_text_task}


def _build_bundle(task: str, data_cfg: dict[str, Any]):
    builder = BUILDERS[task]
    allowed = set(inspect.signature(builder).parameters)
    return builder(**{k: v for k, v in data_cfg.items() if k in allowed})


def _task_cfg(cfg: dict[str, Any], task: str) -> dict[str, Any]:
    out = {k: v for k, v in cfg.items() if k not in {"tasks", "data", task}}
    out.update(cfg.get(task, {}))
    data = dict(cfg.get("data", {}))
    data.update(out.pop("data", {}))
    data.setdefault("shift_mode", "partial_in_support_flip")
    data.setdefault("partial_flip_fraction", 0.6)
    out["data"] = data
    return out


def _component_scores(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "mean_margin_collapse_only": df["margin_collapse_mean"],
        "tail_margin_collapse_only": df["margin_collapse_q90"],
        "js_only": df["js_mean"],
        "label_flip_only": df["flip_mean"],
        "margin_plus_js": df["margin_collapse_mean"] + df["margin_collapse_q90"] + 0.5 * df["js_mean"],
        "full_shift_risk": df["shift_risk"],
    }


def _markdown_table(df: pd.DataFrame) -> str:
    cols = ["task", "component", "failure_auroc", "failure_count", "correct_count"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        lines.append(
            "| "
            + " | ".join(f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in cols)
            + " |"
        )
    return "\n".join(lines) + "\n"


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "certificate_ablation")
    plot_dir = ensure_dir(out_dir / "plots")
    rows = []
    cert_frames = []
    for i, task in enumerate(cfg.get("tasks", ["synthetic", "vision", "text"])):
        task_cfg = _task_cfg(cfg, task)
        task_cfg["seed"] = int(cfg.get("seed", 0)) + i
        bundle = _build_bundle(task, task_cfg["data"])
        model, id_metrics, shifted_metrics, device = train_for_bundle(bundle, task_cfg, int(task_cfg["seed"]))
        cert_df = certificate_frame(model, bundle, task_cfg, device)
        cert_df.insert(0, "task", task)
        cert_frames.append(cert_df)
        failure = cert_df["failure"].to_numpy()
        for name, scores in _component_scores(cert_df).items():
            rows.append(
                {
                    "task": task,
                    "component": name,
                    "id_accuracy": id_metrics["accuracy"],
                    "shifted_accuracy": shifted_metrics["accuracy"],
                    "failure_count": int(cert_df["failure"].sum()),
                    "correct_count": int((1 - cert_df["failure"]).sum()),
                    "failure_auroc": auroc(scores, failure),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "certificate_ablation_metrics.csv", index=False)
    pd.concat(cert_frames, ignore_index=True).to_csv(out_dir / "certificate_ablation_certificates.csv", index=False)
    (out_dir / "component_auc_table.md").write_text(_markdown_table(metrics), encoding="utf-8")
    plot_metric_by_x(metrics, "component", "failure_auroc", plot_dir / "component_auc_barplot.png", hue="task")
    write_json(out_dir / "certificate_ablation_summary.json", {"metrics": rows})
    return {"out_dir": str(out_dir), "metrics": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/certificate_ablation.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
