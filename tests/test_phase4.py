from pathlib import Path

import numpy as np

from causal_reliability.analysis.metric_audit import build_audit, save_outputs
from causal_reliability.analysis.metrics import auroc, failure_prediction_table
from causal_reliability.data.synthetic_shapes import make_shape_task, make_vector_task
from causal_reliability.data.text_shortcuts import SHORTCUTS, make_text_task
from causal_reliability.experiments.run_confident_wrong import run as run_confident_wrong


def test_in_support_flip_uses_seen_shortcuts_and_changes_mapping():
    bundle = make_vector_task(n_train=80, n_test=40, train_corr=1.0, shift_mode="in_support_flip")
    train_shortcuts = set(bundle.train.tensors[2].tolist())
    shifted_shortcuts = set(bundle.shifted_test.tensors[2].tolist())
    y = bundle.shifted_test.tensors[1]
    s = bundle.shifted_test.tensors[2]
    assert shifted_shortcuts <= train_shortcuts
    assert (s == 1 - y).all()


def test_partial_in_support_flip_mixes_normal_and_flipped_seen_shortcuts():
    bundle = make_vector_task(n_train=80, n_test=200, train_corr=1.0, shift_corr=1.0, shift_mode="partial_in_support_flip", partial_flip_fraction=0.5)
    train_shortcuts = set(bundle.train.tensors[2].tolist())
    y = bundle.shifted_test.tensors[1]
    s = bundle.shifted_test.tensors[2]
    assert set(s.tolist()) <= train_shortcuts
    assert (s == y).any()
    assert (s == 1 - y).any()


def test_shape_and_text_in_support_flip_are_familiar():
    shape = make_shape_task(n_train=20, n_test=12, train_corr=1.0, shift_mode="in_support_flip")
    assert set(shape.shifted_test.tensors[2].tolist()) <= set(shape.train.tensors[2].tolist())
    text = make_text_task(n_train=20, n_test=12, train_corr=1.0, shift_mode="in_support_flip")
    shifted_tokens = set(text.shifted_test.tensors[0][:, 2].tolist())
    assert shifted_tokens <= set(SHORTCUTS)


def test_partial_flip_shape_and_text_use_seen_shortcut_values():
    shape = make_shape_task(n_train=30, n_test=40, train_corr=1.0, shift_corr=1.0, shift_mode="partial_in_support_flip", partial_flip_fraction=0.5)
    assert set(shape.shifted_test.tensors[2].tolist()) <= set(shape.train.tensors[2].tolist())
    text = make_text_task(n_train=30, n_test=40, train_corr=1.0, shift_corr=1.0, shift_mode="partial_in_support_flip", partial_flip_fraction=0.5)
    assert set(text.shifted_test.tensors[0][:, 2].tolist()) <= set(SHORTCUTS)


def test_auroc_nan_and_risk_ratio_smoothing():
    assert np.isnan(auroc([0.1, 0.2], [1, 1]))
    table = failure_prediction_table({"risk": np.array([0.1, 0.2, 0.8, 0.9])}, np.array([0, 0, 1, 1]))
    row = table.iloc[0]
    assert np.isfinite(row["risk_ratio"])
    assert row["risk_ratio"] < 1e8
    assert row["auroc_note"] == "low failure count"


def test_high_conf_subset_low_count_smoke(tmp_path: Path):
    cfg = {
        "seed": 2,
        "prefer_gpu": False,
        "results_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "lr": 0.003,
        "n_counterfactuals": 2,
        "tasks": ["synthetic"],
        "data": {"n_train": 32, "n_test": 16, "train_corr": 0.95, "id_corr": 0.95, "shift_corr": 0.0, "noise": 0.35},
    }
    run_confident_wrong(cfg)
    assert (tmp_path / "confident_wrong" / "confident_wrong_high_conf_subset.csv").exists()


def test_metric_audit_writes_outputs(tmp_path: Path):
    cfg = {
        "seed": 3,
        "prefer_gpu": False,
        "results_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "lr": 0.003,
        "n_counterfactuals": 2,
        "tasks": ["synthetic"],
        "data": {"n_train": 32, "n_test": 16, "train_corr": 0.95, "id_corr": 0.95, "shift_corr": 0.0, "noise": 0.35},
    }
    run_confident_wrong(cfg)
    df = build_audit(tmp_path)
    save_outputs(df, tmp_path)
    assert not df.empty
    assert (tmp_path / "metric_audit" / "metric_audit_summary.csv").exists()
    assert (tmp_path / "metric_audit" / "metric_audit_summary.md").exists()
