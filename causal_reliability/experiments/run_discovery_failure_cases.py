from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.data.synthetic_shapes import make_shape_task, make_vector_task
from causal_reliability.discovery.discovery_runner import run_discovery_for_task
from causal_reliability.experiments.run_unknown_shortcut_discovery import _train_erm, _vector_metadata, _vision_metadata
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir
from causal_reliability.utils.seed import set_seed


def _case_row(case: str, expectation: str, rankings: pd.DataFrame, metrics: pd.DataFrame, audit_candidate: str | None = None) -> dict[str, Any]:
    top = rankings.iloc[0]
    true_top3 = rankings.head(3)["known_shortcut_candidate"].astype(bool).any()
    top_score = float(top["full_unknown_shortcut_score"])
    second_score = float(rankings.iloc[1]["full_unknown_shortcut_score"]) if len(rankings) > 1 else 0.0
    row = {
        "case": case,
        "expectation": expectation,
        "top_candidate_name": str(top["candidate_id"]),
        "top_candidate_type": str(top["candidate_type"]),
        "top_candidate_group": str(top["ground_truth_factor_group"]),
        "top_score": top_score,
        "second_score": second_score,
        "dominance_gap": top_score - second_score,
        "top_label_preservation": float(top["label_preservation_rate"]),
        "true_shortcut_rank": int(metrics.loc[0, "true_shortcut_rank"]),
        "top3_true_shortcut_hit": bool(true_top3),
    }
    if audit_candidate:
        audit_rows = rankings[rankings["candidate_id"] == audit_candidate]
        if len(audit_rows):
            audit = audit_rows.iloc[0]
            row.update(
                {
                    "audit_candidate_name": audit_candidate,
                    "audit_candidate_rank": int(audit["rank"]),
                    "audit_candidate_score": float(audit["full_unknown_shortcut_score"]),
                    "audit_candidate_label_preservation": float(audit["label_preservation_rate"]),
                    "audit_candidate_support": float(audit["support_score"]),
                    "audit_candidate_specificity": float(audit["specificity_score"]),
                }
            )
    return row


def _plot_scores(rankings: pd.DataFrame, path: Path) -> None:
    top = rankings.sort_values(["case", "rank"]).groupby("case").head(4)
    if top.empty:
        return
    labels = top["case"] + ":" + top["candidate_id"]
    colors = top["ground_truth_factor_group"].map(
        {
            "true_shortcut": "#d55e00",
            "causal": "#cc79a7",
            "irrelevant": "#4c78a8",
            "corruption": "#999999",
            "missing": "#777777",
        }
    ).fillna("#4c78a8")
    plt.figure(figsize=(10.2, 4.8))
    plt.bar(range(len(top)), top["full_unknown_shortcut_score"], color=colors)
    plt.xticks(range(len(top)), labels, rotation=70, ha="right", fontsize=8)
    plt.ylabel("Discovery score")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _plot_topk(metrics: pd.DataFrame, path: Path) -> None:
    if metrics.empty:
        return
    values = metrics["top3_true_shortcut_hit"].astype(int)
    plt.figure(figsize=(7.8, 3.8))
    plt.bar(metrics["case"], values, color="#4c78a8")
    plt.xticks(rotation=35, ha="right")
    plt.yticks([0, 1], ["no", "yes"])
    plt.ylabel("True shortcut in top-3")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def run(cfg: dict[str, Any]) -> dict[str, Path]:
    set_seed(int(cfg.get("seed", 0)))
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "unknown_shortcut_discovery" / "failure_cases")
    plot_dir = ensure_dir(out_dir / "plots")
    rows: list[dict[str, Any]] = []
    ranking_frames: list[pd.DataFrame] = []

    base = cfg.get("synthetic", {})
    base_kwargs = {
        "n_train": int(base.get("n_train", 256)),
        "n_test": int(base.get("n_test", 128)),
        "train_corr": float(base.get("train_corr", 0.97)),
        "id_corr": float(base.get("id_corr", 0.97)),
        "shift_corr": float(base.get("shift_corr", 0.1)),
        "noise": float(base.get("noise", 0.25)),
        "shortcut_strength": float(base.get("shortcut_strength", 1.4)),
    }

    no_shortcut = make_vector_task(**{**base_kwargs, "train_corr": 0.5, "id_corr": 0.5, "shift_corr": 0.5, "shortcut_strength": 0.0})
    no_meta = {**_vector_metadata(), "true_shortcut_candidate_ids": [], "candidate_groups": {cid: "irrelevant" for cid in _vector_metadata()["candidate_groups"]}}
    model = _train_erm(no_shortcut, cfg)
    rankings, metrics = run_discovery_for_task("no_shortcut", model, no_shortcut, cfg, no_meta)
    rankings["case"] = "no_shortcut"
    rows.append(_case_row("no_shortcut", "No candidate should dominate strongly; no true shortcut exists.", rankings, metrics))
    ranking_frames.append(rankings)

    multiple = make_vector_task(**base_kwargs)
    multi_meta = _vector_metadata()
    model = _train_erm(multiple, cfg)
    rankings, metrics = run_discovery_for_task("multiple_shortcuts", model, multiple, cfg, multi_meta)
    rankings["case"] = "multiple_shortcuts"
    rows.append(_case_row("multiple_shortcuts", "At least one true shortcut should rank in the top-3.", rankings, metrics))
    ranking_frames.append(rankings)

    causal_meta = {
        **_vector_metadata(),
        "true_shortcut_candidate_ids": ["feature_dim_0"],
        "candidate_groups": {**_vector_metadata()["candidate_groups"], "feature_dim_0": "causal", "feature_dim_1": "irrelevant", "matched_magnitude_dim_1": "irrelevant"},
    }
    model = _train_erm(multiple, cfg)
    rankings, metrics = run_discovery_for_task("causal_feature_intervention", model, multiple, cfg, causal_meta)
    rankings["case"] = "causal_feature_intervention"
    rows.append(_case_row("causal_feature_intervention", "Causal interventions can destabilize predictions but should be penalized by label preservation.", rankings, metrics, "feature_dim_0"))
    ranking_frames.append(rankings)

    vision_cfg = cfg.get("vision", {})
    shortcut_type = str(vision_cfg.get("shortcut_type", "color"))
    vision = make_shape_task(
        n_train=int(vision_cfg.get("n_train", 192)),
        n_test=int(vision_cfg.get("n_test", 96)),
        train_corr=float(vision_cfg.get("train_corr", 0.97)),
        id_corr=float(vision_cfg.get("id_corr", 0.97)),
        shift_corr=float(vision_cfg.get("shift_corr", 0.1)),
        image_size=int(vision_cfg.get("image_size", 16)),
        shortcut_type=shortcut_type,
    )
    corruption_meta = _vision_metadata(shortcut_type)
    corruption_meta["candidate_groups"] = {**corruption_meta["candidate_groups"], "additive_noise": "corruption", "blur": "corruption", "brightness": "corruption", "contrast": "corruption"}
    model = _train_erm(vision, cfg)
    rankings, metrics = run_discovery_for_task("corruption_intervention", model, vision, cfg, corruption_meta)
    rankings["case"] = "corruption_intervention"
    rows.append(_case_row("corruption_intervention", "Global corruption can destabilize predictions but should be penalized by support and specificity.", rankings, metrics, "additive_noise"))
    ranking_frames.append(rankings)

    missing_meta = {**_vector_metadata(), "true_shortcut_candidate_ids": ["unrepresented_shortcut"], "candidate_groups": {**_vector_metadata()["candidate_groups"], "feature_dim_1": "missing", "matched_magnitude_dim_1": "missing"}}
    model = _train_erm(multiple, cfg)
    rankings, metrics = run_discovery_for_task("missing_true_shortcut", model, multiple, cfg, missing_meta)
    rankings["case"] = "missing_true_shortcut"
    rows.append(_case_row("missing_true_shortcut", "Discovery should fail honestly when the true shortcut is absent from the candidate class.", rankings, metrics))
    ranking_frames.append(rankings)

    metrics_df = pd.DataFrame(rows)
    rankings_df = pd.concat(ranking_frames, ignore_index=True)
    metrics_df.to_csv(out_dir / "failure_case_metrics.csv", index=False)
    rankings_df.to_csv(out_dir / "failure_case_rankings.csv", index=False)
    _plot_scores(rankings_df, plot_dir / "failure_case_scores.png")
    _plot_topk(metrics_df, plot_dir / "failure_case_topk.png")

    works = metrics_df[metrics_df["top3_true_shortcut_hit"]]["case"].tolist()
    fails = metrics_df[~metrics_df["top3_true_shortcut_hit"]]["case"].tolist()
    summary = [
        "# Discovery Failure Cases",
        "",
        "These controls bound the moonshot claim. They do not test general causal discovery; they test a finite intervention class.",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics_df),
        "",
        "## Where Discovery Works",
        "",
        ", ".join(works) if works else "No case met the top-3 true-shortcut criterion.",
        "",
        "## Where Discovery Fails",
        "",
        ", ".join(fails) if fails else "No expected failure case was triggered.",
        "",
        "These failures are expected when no shortcut exists, when the shortcut is not represented in the candidate class, or when destabilizing interventions are causal-label-changing or broad corruptions that the audit penalizes.",
        "",
    ]
    (out_dir / "failure_case_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(metrics_df.to_string(index=False))
    return {"metrics": out_dir / "failure_case_metrics.csv", "summary": out_dir / "failure_case_summary.md"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/discovery_failure_cases.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
