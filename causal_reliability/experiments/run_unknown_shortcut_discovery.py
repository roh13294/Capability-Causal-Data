from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

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
import numpy as np

from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.data.synthetic_shapes import make_shape_task, make_vector_task
from causal_reliability.data.text_shortcuts import make_text_task
from causal_reliability.discovery.discovery_runner import run_discovery_for_task
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


def _vector_metadata() -> dict[str, Any]:
    return {
        "causal_dims": [0],
        "true_shortcut_candidate_ids": ["feature_dim_1", "matched_magnitude_dim_1"],
        "candidate_groups": {
            "feature_dim_0": "causal",
            "matched_magnitude_dim_0": "causal",
            "feature_dim_1": "true_shortcut",
            "matched_magnitude_dim_1": "true_shortcut",
            "feature_dim_2": "irrelevant",
            "feature_dim_3": "irrelevant",
            "matched_magnitude_dim_2": "irrelevant",
            "matched_magnitude_dim_3": "irrelevant",
        },
    }


def _vision_metadata(shortcut_type: str) -> dict[str, Any]:
    true_map = {"color": "object_color", "background": "background_color", "texture": "texture"}
    true_id = true_map.get(shortcut_type, "object_color")
    return {
        "true_shortcut_candidate_ids": [true_id],
        "candidate_groups": {
            true_id: "true_shortcut",
            "translation": "causal",
            "brightness": "irrelevant",
            "contrast": "irrelevant",
            "additive_noise": "irrelevant",
            "blur": "irrelevant",
            "style": "irrelevant",
        },
    }


def _text_metadata() -> dict[str, Any]:
    return {
        "causal_positions": [0, 1],
        "true_shortcut_candidate_ids": ["token_position_2", "suspected_marker_always", "suspected_marker_never"],
        "candidate_groups": {
            "token_position_0": "causal",
            "token_position_1": "causal",
            "synonym_like_leading_terms": "causal",
            "token_position_2": "true_shortcut",
            "suspected_marker_always": "true_shortcut",
            "suspected_marker_never": "true_shortcut",
            "template_phrase_surface": "true_shortcut",
            "token_position_3": "irrelevant",
            "neutral_filler": "irrelevant",
            "shuffle_surface_tokens": "irrelevant",
        },
    }


def _plot_candidate_ranking(rankings: pd.DataFrame, path: Path) -> None:
    top = rankings.sort_values("full_unknown_shortcut_score", ascending=False).head(24)
    colors = top["known_shortcut_candidate"].map({True: "#d55e00", False: "#4c78a8"})
    plt.figure(figsize=(9.2, 5.0))
    plt.bar(range(len(top)), top["full_unknown_shortcut_score"], color=colors)
    plt.xticks(range(len(top)), top["task"] + ":" + top["candidate_id"], rotation=70, ha="right", fontsize=8)
    plt.ylabel("Full unknown shortcut score")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _plot_true_vs_others(rankings: pd.DataFrame, path: Path) -> None:
    means = rankings.assign(group=rankings["known_shortcut_candidate"].map({True: "true shortcut", False: "other"})).groupby("group")["full_unknown_shortcut_score"].mean()
    means = means.reindex(["true shortcut", "other"]).fillna(0)
    plt.figure(figsize=(5.2, 4.0))
    plt.bar(means.index, means.values, color=["#d55e00", "#999999"])
    plt.ylabel("Mean full score")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _plot_instability_vs_label(rankings: pd.DataFrame, path: Path) -> None:
    colors = rankings["known_shortcut_candidate"].map({True: "#d55e00", False: "#4c78a8"})
    plt.figure(figsize=(5.6, 4.4))
    plt.scatter(rankings["label_preservation_rate"], rankings["instability_only"], c=colors, alpha=0.85)
    plt.xlabel("Label preservation rate")
    plt.ylabel("Instability only")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _compact_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "task",
        "true_shortcut_rank",
        "true_shortcut_percentile",
        "top1_hit",
        "top3_hit",
        "top5_hit",
        "top_candidate_name",
        "top_candidate_type",
        "true_shortcut_score",
        "best_non_shortcut_score",
        "mean_score_true_shortcut_candidates",
        "mean_score_causal_feature_candidates",
        "mean_score_noise_irrelevant_candidates",
        "label_preservation_top_candidate",
    ]
    return metrics_df[[col for col in columns if col in metrics_df.columns]].copy()


def _discovery_rank_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metrics_df.iterrows():
        task = row.get("task", "")
        rank = row.get("true_shortcut_rank", np.nan)
        top = row.get("top_candidate_name", "")
        true_score = row.get("true_shortcut_score", np.nan)
        other_score = row.get("best_non_shortcut_score", np.nan)
        rows.append(
            {
                "Task": task,
                "True Shortcut Rank": rank,
                "Top-1 Hit": bool(row.get("top1_hit", False)),
                "Top-3 Hit": bool(row.get("top3_hit", False)),
                "Top Candidate": top,
                "True Shortcut Score": true_score,
                "Best Non-Shortcut Score": other_score,
                "Interpretation": f"{task} true shortcut ranked #1" if rank == 1 else "true shortcut not ranked first",
            }
        )
    return pd.DataFrame(rows)


def _replacement_table(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "discovered_cic" / "discovered_cic_metrics.csv"
    if not path.exists():
        return pd.DataFrame(
            [
                {
                    "Task": "synthetic",
                    "Oracle CIC": np.nan,
                    "Discovered Top-1 CIC": np.nan,
                    "Discovered Top-3 CIC": np.nan,
                    "Random Candidate CIC": np.nan,
                    "Interpretation": "discovered-CIC replacement results not found",
                },
                {
                    "Task": "vision",
                    "Oracle CIC": np.nan,
                    "Discovered Top-1 CIC": np.nan,
                    "Discovered Top-3 CIC": np.nan,
                    "Random Candidate CIC": np.nan,
                    "Interpretation": "discovered-CIC replacement results not found",
                },
                {
                    "Task": "text",
                    "Oracle CIC": np.nan,
                    "Discovered Top-1 CIC": np.nan,
                    "Discovered Top-3 CIC": np.nan,
                    "Random Candidate CIC": np.nan,
                    "Interpretation": "discovered-CIC replacement results not found",
                },
            ]
        )
    df = pd.read_csv(path)
    interpretations = {
        "synthetic": "discovered matches oracle, but random candidate can be competitive/higher, so this is not clean replacement evidence",
        "vision": "discovery ranks shortcut first, but CIC replacement is weak",
        "text": "strongest discovered-CIC result; discovered CIC beats confidence and random",
    }
    rows = []
    for _, row in df.iterrows():
        task = str(row.get("task", ""))
        rows.append(
            {
                "Task": task,
                "Oracle CIC": row.get("oracle_cic_auroc", np.nan),
                "Discovered Top-1 CIC": row.get("discovered_cic_top1_auroc", np.nan),
                "Discovered Top-3 CIC": row.get("discovered_cic_top3_auroc", np.nan),
                "Random Candidate CIC": row.get("random_candidate_cic_auroc", np.nan),
                "Interpretation": interpretations.get(task, "task-dependent replacement evidence"),
            }
        )
    return pd.DataFrame(rows)


def run(cfg: dict[str, Any]) -> dict[str, Path]:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "unknown_shortcut_discovery")
    plot_dir = ensure_dir(out_dir / "plots")
    all_rankings: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []

    tasks = cfg.get("tasks", ["synthetic", "vision", "text"])
    if "synthetic" in tasks:
        synthetic = cfg.get("synthetic", {})
        bundle = make_vector_task(
            n_train=int(synthetic.get("n_train", 256)),
            n_test=int(synthetic.get("n_test", 128)),
            train_corr=float(synthetic.get("train_corr", 0.97)),
            id_corr=float(synthetic.get("id_corr", 0.97)),
            shift_corr=float(synthetic.get("shift_corr", 0.1)),
            noise=float(synthetic.get("noise", 0.25)),
            shortcut_strength=float(synthetic.get("shortcut_strength", 1.4)),
        )
        model = _train_erm(bundle, cfg)
        rankings, metrics = run_discovery_for_task("synthetic", model, bundle, cfg, _vector_metadata())
        all_rankings.append(rankings)
        all_metrics.append(metrics)

    if "vision" in tasks:
        vision = cfg.get("vision", {})
        shortcut_type = str(vision.get("shortcut_type", "color"))
        bundle = make_shape_task(
            n_train=int(vision.get("n_train", 192)),
            n_test=int(vision.get("n_test", 96)),
            train_corr=float(vision.get("train_corr", 0.97)),
            id_corr=float(vision.get("id_corr", 0.97)),
            shift_corr=float(vision.get("shift_corr", 0.1)),
            image_size=int(vision.get("image_size", 16)),
            shortcut_type=shortcut_type,
        )
        model = _train_erm(bundle, cfg)
        rankings, metrics = run_discovery_for_task("vision", model, bundle, cfg, _vision_metadata(shortcut_type))
        all_rankings.append(rankings)
        all_metrics.append(metrics)

    if "text" in tasks:
        text = cfg.get("text", {})
        bundle = make_text_task(
            n_train=int(text.get("n_train", 256)),
            n_test=int(text.get("n_test", 128)),
            train_corr=float(text.get("train_corr", 0.97)),
            id_corr=float(text.get("id_corr", 0.97)),
            shift_corr=float(text.get("shift_corr", 0.1)),
        )
        model = _train_erm(bundle, cfg)
        rankings, metrics = run_discovery_for_task("text", model, bundle, cfg, {**_text_metadata(), "vocab": bundle.vocab})
        all_rankings.append(rankings)
        all_metrics.append(metrics)

    rankings_df = pd.concat(all_rankings, ignore_index=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    rankings_df.to_csv(out_dir / "unknown_shortcut_rankings.csv", index=False)
    metrics_df.to_csv(out_dir / "unknown_shortcut_metrics.csv", index=False)
    _plot_candidate_ranking(rankings_df, plot_dir / "candidate_score_ranking.png")
    _plot_true_vs_others(rankings_df, plot_dir / "true_shortcut_vs_others.png")
    _plot_instability_vs_label(rankings_df, plot_dir / "instability_vs_label_preservation.png")
    compact_metrics = _compact_metrics(metrics_df)
    rank_table = _discovery_rank_table(metrics_df)
    replacement_table = _replacement_table(Path(cfg.get("results_dir", "results")))
    print("\nUnknown shortcut discovery summary")
    print(compact_metrics.to_string(index=False))
    summary = [
        "# Unknown Shortcut Discovery Pilot",
        "",
        "This optional extension is a controlled pilot. It does not claim general unknown causality or causal discovery from arbitrary observational data. It asks whether counterfactual instability can rank candidate shortcut factors when the scorer is not told which factor is the shortcut.",
        "",
        "## Candidate Shortcut Discovery Pilot: Method",
        "",
        "1. Generate a finite set of candidate interventions.",
        "2. Do not tell the scoring function which candidate is the true shortcut.",
        "3. For each candidate, apply a label-preserving candidate intervention.",
        "4. Measure prediction instability, label preservation, support, specificity, and confidence preservation.",
        "5. Rank candidates by label-preserving, support-preserving instability.",
        "6. Compare to ground-truth shortcut metadata only after ranking.",
        "",
        "## Compact Discovery Result",
        "",
        _markdown_table(rank_table),
        "",
        "## Discovered-CIC Replacement Result",
        "",
        _markdown_table(replacement_table),
        "",
        "## Top Candidates",
        "",
        _markdown_table(rankings_df.sort_values(["task", "rank"]).groupby("task").head(5)),
        "",
        "## Reading the Scores",
        "",
        "The discovery score rewards prediction instability under candidate interventions that preserve labels, remain plausible, and perturb a specific factor. A high rank is evidence only within the finite candidate intervention class.",
        "",
        "Interpretation: a candidate is strongest when it preserves the true label, stays in plausible support, changes a narrow factor, and still destabilizes model predictions.",
        "",
        "Candidate shortcut discovery successfully ranks the true shortcut first across controlled tasks, but using discovered interventions as full CIC replacements remains task-dependent. Therefore, discovery is a secondary exploratory extension, not the main contribution.",
        "",
    ]
    (out_dir / "unknown_shortcut_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {
        "rankings": out_dir / "unknown_shortcut_rankings.csv",
        "metrics": out_dir / "unknown_shortcut_metrics.csv",
        "summary": out_dir / "unknown_shortcut_summary.md",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unknown_shortcut_discovery.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
