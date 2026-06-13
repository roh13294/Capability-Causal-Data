from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_reliability.analysis.metrics import failure_prediction_table
from causal_reliability.analysis.phase6_common import _markdown_table, binary_entropy
from causal_reliability.certificates.reliability import batch_compute_certificates
from causal_reliability.data.synthetic_shapes import make_shape_task, make_vector_task
from causal_reliability.data.text_shortcuts import make_text_task
from causal_reliability.discovery.candidate_factors import generate_candidate_factors
from causal_reliability.discovery.candidate_interventions import CandidateIntervention, make_intervention
from causal_reliability.discovery.discovery_runner import run_discovery_for_task
from causal_reliability.experiments.run_unknown_shortcut_discovery import _text_metadata, _vector_metadata, _vision_metadata
from causal_reliability.models import build_model
from causal_reliability.training.loops import train_model
from causal_reliability.utils.config import load_config
from causal_reliability.utils.device import get_device
from causal_reliability.utils.io import ensure_dir
from causal_reliability.utils.seed import set_seed


def _loader(ds, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _train_erm(bundle, cfg: dict[str, Any]) -> torch.nn.Module:
    device = get_device(bool(cfg.get("prefer_gpu", True)))
    model = build_model(bundle.task_type, bundle.input_shape, bundle.num_classes).to(device)
    train_model(
        model,
        _loader(bundle.train, int(cfg.get("batch_size", 64)), shuffle=True),
        device,
        epochs=int(cfg.get("epochs", 3)),
        lr=float(cfg.get("lr", 1e-3)),
        mode="erm",
    )
    return model


def _bundle_and_metadata(task: str, cfg: dict[str, Any]):
    if task == "synthetic":
        data = cfg.get("synthetic", {})
        bundle = make_vector_task(
            n_train=int(data.get("n_train", 256)),
            n_test=int(data.get("n_test", 128)),
            train_corr=float(data.get("train_corr", 0.97)),
            id_corr=float(data.get("id_corr", 0.97)),
            shift_corr=float(data.get("shift_corr", 0.1)),
            noise=float(data.get("noise", 0.25)),
            shortcut_strength=float(data.get("shortcut_strength", 1.4)),
        )
        return bundle, _vector_metadata()
    if task == "vision":
        data = cfg.get("vision", {})
        shortcut_type = str(data.get("shortcut_type", "color"))
        bundle = make_shape_task(
            n_train=int(data.get("n_train", 192)),
            n_test=int(data.get("n_test", 96)),
            train_corr=float(data.get("train_corr", 0.97)),
            id_corr=float(data.get("id_corr", 0.97)),
            shift_corr=float(data.get("shift_corr", 0.1)),
            image_size=int(data.get("image_size", 16)),
            shortcut_type=shortcut_type,
        )
        return bundle, _vision_metadata(shortcut_type)
    if task == "text":
        data = cfg.get("text", {})
        bundle = make_text_task(
            n_train=int(data.get("n_train", 256)),
            n_test=int(data.get("n_test", 128)),
            train_corr=float(data.get("train_corr", 0.97)),
            id_corr=float(data.get("id_corr", 0.97)),
            shift_corr=float(data.get("shift_corr", 0.1)),
        )
        return bundle, {**_text_metadata(), "vocab": bundle.vocab}
    raise ValueError(f"unknown task: {task}")


def _all_candidates(bundle, seed: int, metadata: dict[str, Any]) -> dict[str, CandidateIntervention]:
    factors = generate_candidate_factors(
        bundle.task_type,
        n_features=bundle.input_shape[0],
        seed=seed,
        vocab=bundle.vocab,
        seq_len=bundle.input_shape[0],
    )
    return {factor.candidate_id: make_intervention(bundle.task_type, factor) for factor in factors}


def _make_cf(candidates: list[CandidateIntervention], metadata: dict[str, Any]):
    def make_cf(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([candidate.apply(x, metadata=metadata) for candidate in candidates], dim=1)

    return make_cf


def _cert_frame(
    model: torch.nn.Module,
    bundle,
    candidates: list[CandidateIntervention],
    metadata: dict[str, Any],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    device = next(model.parameters()).device
    loader = _loader(bundle.shifted_test, int(cfg.get("batch_size", 64)))
    certs = batch_compute_certificates(model, loader, _make_cf(candidates, metadata), device)
    df = pd.DataFrame({key: value.numpy() for key, value in certs.items()})
    df["correct"] = (df["pred"] == df["label"]).astype(int)
    df["failure"] = 1 - df["correct"]
    df["entropy"] = binary_entropy(df["confidence"])
    df["confidence_risk"] = 1.0 - df["confidence"]
    df["negative_margin"] = -df["margin"]
    return df


def _method_auc(df: pd.DataFrame, score_col: str) -> float:
    table = failure_prediction_table({score_col: df[score_col].to_numpy()}, df["failure"].to_numpy())
    return float(table.loc[0, "failure_auroc"])


def _plot_metric_bars(metrics: pd.DataFrame, path: Path, columns: list[str], ylabel: str) -> None:
    if metrics.empty:
        return
    x = np.arange(len(metrics))
    width = 0.8 / max(len(columns), 1)
    plt.figure(figsize=(8.4, 4.2))
    for i, col in enumerate(columns):
        plt.bar(x + (i - (len(columns) - 1) / 2) * width, metrics[col], width=width, label=col.replace("_auroc", ""))
    plt.xticks(x, metrics["task"])
    plt.ylim(0.0, 1.02)
    plt.ylabel(ylabel)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _plot_top_scores(rankings: pd.DataFrame, path: Path) -> None:
    top = rankings.sort_values(["task", "rank"]).groupby("task").head(5).copy()
    if top.empty:
        return
    labels = top["task"] + ":" + top["candidate_id"]
    colors = top["known_shortcut_candidate"].map({True: "#d55e00", False: "#4c78a8"})
    plt.figure(figsize=(9.2, 4.6))
    plt.bar(range(len(top)), top["full_unknown_shortcut_score"], color=colors)
    plt.xticks(range(len(top)), labels, rotation=70, ha="right", fontsize=8)
    plt.ylabel("Discovery score")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def run(cfg: dict[str, Any]) -> dict[str, Path]:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "discovered_cic")
    plot_dir = ensure_dir(out_dir / "plots")
    tasks = list(cfg.get("tasks", ["synthetic", "vision", "text"]))
    top_ks = [int(k) for k in cfg.get("top_k", [1, 3])]
    metrics_rows: list[dict[str, Any]] = []
    by_task: dict[str, Any] = {}
    all_rankings: list[pd.DataFrame] = []

    for task in tasks:
        bundle, metadata = _bundle_and_metadata(task, cfg)
        model = _train_erm(bundle, cfg)
        rankings, discovery_metrics = run_discovery_for_task(task, model, bundle, cfg, metadata)
        all_rankings.append(rankings)
        candidates = _all_candidates(bundle, seed, metadata)
        true_ids = [cid for cid in metadata.get("true_shortcut_candidate_ids", []) if cid in candidates]
        if not true_ids:
            continue
        oracle = [candidates[true_ids[0]]]
        ranked_ids = [str(cid) for cid in rankings["candidate_id"] if str(cid) in candidates]
        discovered_top1 = [candidates[ranked_ids[0]]]
        discovered_top3 = [candidates[cid] for cid in ranked_ids[: max(top_ks)]]
        non_shortcut_ids = [cid for cid in ranked_ids if cid not in set(true_ids)]
        rng = np.random.default_rng(seed)
        random_id = str(rng.choice(non_shortcut_ids or ranked_ids))
        random_candidate = [candidates[random_id]]

        oracle_df = _cert_frame(model, bundle, oracle, metadata, cfg)
        top1_df = _cert_frame(model, bundle, discovered_top1, metadata, cfg)
        top3_df = _cert_frame(model, bundle, discovered_top3[:3], metadata, cfg)
        random_df = _cert_frame(model, bundle, random_candidate, metadata, cfg)

        row = {
            "task": task,
            "confidence_auroc": _method_auc(oracle_df, "confidence_risk"),
            "entropy_auroc": _method_auc(oracle_df, "entropy"),
            "negative_margin_auroc": _method_auc(oracle_df, "negative_margin"),
            "old_shift_risk_auroc": _method_auc(oracle_df, "shift_risk"),
            "oracle_cic_auroc": _method_auc(oracle_df, "cis"),
            "discovered_cic_top1_auroc": _method_auc(top1_df, "cis"),
            "discovered_cic_top3_auroc": _method_auc(top3_df, "cis"),
            "random_candidate_cic_auroc": _method_auc(random_df, "cis"),
            "top1_candidate_name": ranked_ids[0],
            "top3_candidate_names": ", ".join(ranked_ids[:3]),
            "random_candidate_name": random_id,
            "true_shortcut_rank": int(discovery_metrics.loc[0, "true_shortcut_rank"]),
        }
        row["discovered_oracle_gap_top1"] = row["oracle_cic_auroc"] - row["discovered_cic_top1_auroc"]
        row["discovered_oracle_gap_top3"] = row["oracle_cic_auroc"] - row["discovered_cic_top3_auroc"]
        row["discovered_confidence_advantage"] = row["discovered_cic_top1_auroc"] - row["confidence_auroc"]
        metrics_rows.append(row)
        by_task[task] = {
            "metrics": row,
            "discovery": discovery_metrics.iloc[0].to_dict(),
            "top_candidates": rankings.head(5).to_dict(orient="records"),
        }

    metrics = pd.DataFrame(metrics_rows)
    rankings_df = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    metrics.to_csv(out_dir / "discovered_cic_metrics.csv", index=False)
    rankings_df.to_csv(out_dir / "discovered_cic_rankings.csv", index=False)
    (out_dir / "discovered_cic_by_task.json").write_text(json.dumps(by_task, indent=2), encoding="utf-8")
    _plot_metric_bars(metrics, plot_dir / "oracle_vs_discovered_auc.png", ["oracle_cic_auroc", "discovered_cic_top1_auroc", "discovered_cic_top3_auroc"], "Failure prediction AUROC")
    _plot_metric_bars(metrics, plot_dir / "discovered_vs_confidence_auc.png", ["confidence_auroc", "old_shift_risk_auroc", "discovered_cic_top1_auroc"], "Failure prediction AUROC")
    _plot_top_scores(rankings_df, plot_dir / "top_candidate_scores.png")

    mean_gap = float(metrics["discovered_oracle_gap_top1"].mean()) if not metrics.empty else float("nan")
    statement = "Discovered CIC approaches oracle CIC when the top-ranked candidate is the true shortcut." if np.isfinite(mean_gap) and mean_gap <= 0.08 else "Discovered CIC remains meaningfully below oracle CIC on at least some tasks."
    summary = [
        "# Discovered CIC",
        "",
        "This experiment asks whether automatically discovered interventions can replace oracle shortcut interventions when computing CIC. The claim is limited to a finite candidate intervention class in controlled settings.",
        "",
        statement,
        "",
        _markdown_table(metrics),
        "",
    ]
    (out_dir / "discovered_cic_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(metrics.to_string(index=False))
    return {
        "metrics": out_dir / "discovered_cic_metrics.csv",
        "summary": out_dir / "discovered_cic_summary.md",
        "by_task": out_dir / "discovered_cic_by_task.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/discovered_cic.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
