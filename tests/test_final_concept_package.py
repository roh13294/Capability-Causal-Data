from pathlib import Path

import pandas as pd

from causal_reliability.analysis.concept_figure import make_figure as make_concept_figure
from causal_reliability.analysis.reliability_plane import build_reliability_plane, quadrant_label, run as run_reliability_plane
from causal_reliability.experiments.run_final_validation import run as run_final_validation
from causal_reliability.experiments.run_shortcut_discovery import (
    feature_instability_ranking,
    fit_linear_erm_proxy,
    make_synthetic_shortcut_data,
    run as run_shortcut_discovery,
)


def test_reliability_plane_quadrant_labels():
    assert quadrant_label(0.9, 0.9) == "Reliable prediction"
    assert quadrant_label(0.4, 0.9) == "Uncertain but causally stable"
    assert quadrant_label(0.4, 0.2) == "Generally fragile"
    assert quadrant_label(0.9, 0.2) == "Dangerous shortcut reliance"


def test_reliability_plane_builds_dangerous_quadrant():
    df = pd.DataFrame(
        {
            "task": ["synthetic"] * 4,
            "regime": ["confident-wrong"] * 4,
            "confidence": [0.91, 0.92, 0.45, 0.46],
            "cis": [1.0, 0.9, 0.1, 0.2],
            "failure": [1, 1, 0, 0],
        }
    )
    points, quadrants, _ = build_reliability_plane(df, confidence_threshold=0.8, stability_threshold=0.5)
    assert "Dangerous shortcut reliance" in set(points["quadrant"])
    dangerous = quadrants[quadrants["quadrant"] == "Dangerous shortcut reliance"].iloc[0]
    assert dangerous["failure_rate"] == 1.0


def test_shortcut_discovery_ranks_shortcut_above_noise():
    x, y, feature_types = make_synthetic_shortcut_data(
        n=256,
        n_features=6,
        causal_dims=[0, 1],
        shortcut_dims=[2],
        seed=3,
        shortcut_strength=1.8,
    )
    weights = fit_linear_erm_proxy(x, y)
    rankings, metrics = feature_instability_ranking(x, weights, feature_types, [2], seed=3)
    shortcut_rank = int(metrics["shortcut_rank"].iloc[0])
    best_noise_rank = int(rankings[rankings["feature_type"] == "noise"]["rank"].min())
    assert shortcut_rank < best_noise_rank
    assert bool(metrics["shortcut_top3_hit"].iloc[0])


def test_concept_figure_writes_outputs(tmp_path: Path):
    run_final_validation({"results_dir": str(tmp_path), "seeds": [0], "tasks": ["synthetic"], "regimes": ["confidence_solvable", "confident_wrong"], "n_examples": 48})
    run_reliability_plane(tmp_path)
    run_shortcut_discovery({"results_dir": str(tmp_path), "seed": 0, "synthetic": {"n_examples": 128, "n_features": 6, "causal_dims": [0, 1], "shortcut_dims": [2]}})
    make_concept_figure(tmp_path)
    assert (tmp_path / "concept_figure.png").exists()
    assert (tmp_path / "concept_figure.pdf").exists()
    assert (tmp_path / "concept_figure_caption.md").exists()


def test_formal_separation_guardrail_text():
    path = Path("docs/formal_separation.md")
    text = path.read_text(encoding="utf-8").lower()
    assert "confidence and counterfactual stability are separable reliability signals" in text
    assert "always beats confidence" not in text


def test_readme_contains_two_axis_reliability_framing():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "Reliability should be evaluated on two axes" in text
    assert "Dangerous shortcut reliance" in text
