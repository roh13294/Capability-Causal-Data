from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from causal_reliability.discovery.candidate_factors import generate_candidate_factors
from causal_reliability.discovery.candidate_interventions import make_intervention
from causal_reliability.discovery.scoring import score_candidate


def run_discovery_for_task(
    task_name: str,
    model: torch.nn.Module,
    bundle,
    cfg: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    batch_size = int(cfg.get("score_batch_size", cfg.get("batch_size", 64)))
    loader = DataLoader(bundle.id_test, batch_size=batch_size, shuffle=False)
    x, y, _shortcut, _causal = next(iter(loader))
    seed = int(cfg.get("seed", 0))
    factors = generate_candidate_factors(
        bundle.task_type,
        n_features=bundle.input_shape[0],
        seed=seed,
        vocab=bundle.vocab,
        seq_len=bundle.input_shape[0],
    )
    rows = []
    for factor in factors:
        candidate = make_intervention(bundle.task_type, factor)
        row = score_candidate(model, x, y, candidate, bundle.task_type, bundle.input_shape, metadata)
        row["task"] = task_name
        row["known_shortcut_candidate"] = bool(factor.candidate_id in set(metadata.get("true_shortcut_candidate_ids", [])))
        row["ground_truth_factor_group"] = metadata.get("candidate_groups", {}).get(factor.candidate_id, "irrelevant")
        rows.append(row)
    rankings = pd.DataFrame(rows).sort_values("full_unknown_shortcut_score", ascending=False).reset_index(drop=True)
    rankings["rank"] = range(1, len(rankings) + 1)
    if len(rankings) > 1:
        rankings["percentile"] = 1.0 - (rankings["rank"] - 1) / (len(rankings) - 1)
    else:
        rankings["percentile"] = 1.0
    true_rows = rankings[rankings["known_shortcut_candidate"]]
    non_shortcut_rows = rankings[~rankings["known_shortcut_candidate"]]
    top = rankings.iloc[0]
    true_rank = int(true_rows["rank"].min()) if len(true_rows) else -1
    true_percentile = float(true_rows.loc[true_rows["rank"].idxmin(), "percentile"]) if len(true_rows) else float("nan")
    best_true = true_rows.loc[true_rows["rank"].idxmin()] if len(true_rows) else None
    metrics = pd.DataFrame(
        [
            {
                "task": task_name,
                "true_shortcut_rank": true_rank,
                "true_shortcut_percentile": true_percentile,
                "top1_hit": bool(true_rank == 1),
                "top3_hit": bool(0 < true_rank <= 3),
                "top5_hit": bool(0 < true_rank <= 5),
                "top_candidate_name": str(top["candidate_id"]),
                "top_candidate_type": str(top["candidate_type"]),
                "true_shortcut_score": float(best_true["full_unknown_shortcut_score"]) if best_true is not None else float("nan"),
                "best_non_shortcut_score": float(non_shortcut_rows["full_unknown_shortcut_score"].max()) if len(non_shortcut_rows) else float("nan"),
                "mean_score_true_shortcut_candidates": float(true_rows["full_unknown_shortcut_score"].mean()) if len(true_rows) else float("nan"),
                "mean_score_causal_feature_candidates": _mean_group(rankings, "causal"),
                "mean_score_noise_irrelevant_candidates": _mean_group(rankings, "irrelevant"),
                "label_preservation_top_candidate": float(top["label_preservation_rate"]),
                "label_preservation_true_shortcut": float(true_rows["label_preservation_rate"].mean()) if len(true_rows) else float("nan"),
                "support_true_shortcut": float(true_rows["support_score"].mean()) if len(true_rows) else float("nan"),
                "instability_true_shortcut": float(true_rows["instability_only"].mean()) if len(true_rows) else float("nan"),
            }
        ]
    )
    return rankings, metrics


def _mean_group(rankings: pd.DataFrame, group: str) -> float:
    values = rankings.loc[rankings["ground_truth_factor_group"] == group, "full_unknown_shortcut_score"]
    return float(values.mean()) if len(values) else float("nan")
