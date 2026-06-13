from pathlib import Path

from causal_reliability.experiments.run_counterfactual_mismatch import run as run_mismatch
from causal_reliability.experiments.run_lambda_sweep import run as run_lambda
from causal_reliability.experiments.run_seed_variance import run as run_seed
from causal_reliability.experiments.run_shortcut_sweep import run as run_shortcut


def _cfg(tmp_path: Path):
    return {
        "seed": 5,
        "prefer_gpu": False,
        "results_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "lr": 0.003,
        "n_counterfactuals": 2,
        "data": {"n_train": 32, "n_test": 16, "noise": 0.35, "shift_corr": 0.1},
    }


def test_shortcut_sweep_smoke_test(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg["shortcut_correlations"] = [0.8]
    df = run_shortcut(cfg)
    assert not df.empty
    assert (tmp_path / "synthetic" / "shortcut_sweep" / "shortcut_sweep_metrics.csv").exists()


def test_lambda_sweep_smoke_test(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg["lambda_values"] = [0.0]
    df = run_lambda(cfg)
    assert len(df) == 1
    assert (tmp_path / "synthetic" / "lambda_sweep" / "lambda_sweep_metrics.csv").exists()


def test_seed_variance_smoke_test_with_2_seeds(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg["seeds"] = [0, 1]
    cfg["stability_lambda"] = 0.2
    df = run_seed(cfg)
    assert set(df["model"]) == {"erm", "stability"}
    assert (tmp_path / "synthetic" / "seed_variance" / "seed_metrics.csv").exists()


def test_counterfactual_mismatch_smoke_test(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg["intervention_types"] = ["color"]
    cfg["shift_types"] = ["color"]
    cfg["data"] = {"n_train": 24, "n_test": 12, "image_size": 10, "train_corr": 0.9, "id_corr": 0.9, "shift_corr": 0.1}
    df = run_mismatch(cfg)
    assert len(df) == 1
    assert (tmp_path / "vision" / "counterfactual_mismatch" / "mismatch_matrix.csv").exists()
