from pathlib import Path

from causal_reliability.data.synthetic_shapes import make_vector_task
from causal_reliability.experiments.common import run_task
from causal_reliability.experiments.run_colored_digits import run as run_colored_digits
from causal_reliability.experiments.run_unknown_shortcut_discovery import run as run_unknown_shortcut_discovery


def test_experiment_smoke_writes_expected_files(tmp_path: Path):
    cfg = {
        "seed": 11,
        "prefer_gpu": False,
        "results_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "n_counterfactuals": 2,
        "lr": 0.003,
    }
    run_task("synthetic_smoke", make_vector_task(n_train=32, n_test=16), cfg)
    out = tmp_path / "synthetic_smoke"
    assert (out / "train_metrics.csv").exists()
    assert (out / "test_metrics.csv").exists()
    assert (out / "certificates.csv").exists()
    assert (out / "failure_prediction.csv").exists()


def test_unknown_shortcut_discovery_smoke(tmp_path: Path):
    cfg = {
        "seed": 13,
        "prefer_gpu": False,
        "results_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "score_batch_size": 16,
        "lr": 0.003,
        "tasks": ["synthetic"],
        "synthetic": {"n_train": 32, "n_test": 16, "noise": 0.25, "shortcut_strength": 1.4},
    }
    outputs = run_unknown_shortcut_discovery(cfg)
    out = tmp_path / "unknown_shortcut_discovery"
    assert outputs["rankings"] == out / "unknown_shortcut_rankings.csv"
    assert outputs["metrics"] == out / "unknown_shortcut_metrics.csv"
    assert outputs["summary"] == out / "unknown_shortcut_summary.md"
    assert (out / "plots" / "candidate_score_ranking.png").exists()


def test_colored_digits_smoke(tmp_path: Path):
    cfg = {
        "seed": 17,
        "prefer_gpu": False,
        "results_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "n_counterfactuals": 2,
        "lr": 0.003,
        "data": {"n_train": 40, "n_test": 20, "noise": 0.02},
    }
    outputs = run_colored_digits(cfg)
    out = tmp_path / "colored_digits"
    assert outputs["metrics"] == str(out / "colored_digits_metrics.csv")
    assert (out / "colored_digits_certificates.csv").exists()
    assert (out / "colored_digits_summary.md").exists()
    assert (out / "plots" / "colored_digits_examples.png").exists()
