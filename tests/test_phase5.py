from pathlib import Path

import pandas as pd

from causal_reliability.analysis.negative_control_diagnosis import build_diagnosis, save_outputs
from causal_reliability.data.synthetic_shapes import make_vector_task
from causal_reliability.experiments.run_certificate_ablation import run as run_certificate_ablation
from causal_reliability.experiments.run_negative_controls import shuffled_index_by_label, shuffled_index_by_shortcut
from causal_reliability.experiments.run_partial_flip_sweep import run as run_partial_flip_sweep


def test_partial_flip_can_produce_mixed_failure_like_shortcut_mappings():
    bundle = make_vector_task(n_train=80, n_test=120, train_corr=1.0, shift_corr=1.0, shift_mode="partial_in_support_flip", partial_flip_fraction=0.5)
    y = bundle.shifted_test.tensors[1]
    s = bundle.shifted_test.tensors[2]
    assert (s == y).any()
    assert (s != y).any()


def test_shuffled_controls_preserve_requested_group():
    bundle = make_vector_task(n_train=40, n_test=80, train_corr=1.0, shift_corr=1.0, shift_mode="partial_in_support_flip", partial_flip_fraction=0.5)
    y = bundle.shifted_test.tensors[1]
    shortcut = bundle.shifted_test.tensors[2]
    within = shuffled_index_by_label(y)
    same_shortcut = shuffled_index_by_shortcut(shortcut)
    assert (y[within] == y).all()
    assert (shortcut[same_shortcut] == shortcut).all()


def test_certificate_component_ablation_writes_outputs(tmp_path: Path):
    cfg = {
        "seed": 4,
        "prefer_gpu": False,
        "results_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "lr": 0.003,
        "n_counterfactuals": 2,
        "tasks": ["synthetic"],
        "data": {"n_train": 32, "n_test": 20, "train_corr": 0.95, "id_corr": 0.95, "shift_corr": 0.95, "shift_mode": "partial_in_support_flip", "partial_flip_fraction": 0.5},
    }
    run_certificate_ablation(cfg)
    assert (tmp_path / "certificate_ablation" / "certificate_ablation_metrics.csv").exists()
    assert (tmp_path / "certificate_ablation" / "component_auc_table.md").exists()


def test_partial_flip_sweep_writes_outputs(tmp_path: Path):
    cfg = {
        "seed": 5,
        "prefer_gpu": False,
        "results_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 16,
        "lr": 0.003,
        "n_counterfactuals": 2,
        "tasks": ["synthetic"],
        "partial_flip_fraction": [0.4],
        "data": {"n_train": 32, "n_test": 20, "train_corr": 0.95, "id_corr": 0.95, "shift_corr": 0.95},
    }
    run_partial_flip_sweep(cfg)
    assert (tmp_path / "partial_flip_sweep" / "partial_flip_metrics.csv").exists()
    assert (tmp_path / "partial_flip_sweep" / "plots" / "flip_fraction_vs_failure_auroc.png").exists()


def test_negative_control_diagnosis_writes_outputs(tmp_path: Path):
    out = tmp_path / "negative_controls"
    out.mkdir()
    pd.DataFrame(
        {
            "control": ["true_counterfactual", "true_counterfactual", "shuffled_any", "shuffled_any"],
            "pred": [0, 1, 0, 1],
            "label": [0, 0, 0, 0],
            "failure": [0, 1, 0, 1],
            "confidence": [0.9, 0.8, 0.88, 0.82],
            "margin": [2.0, 1.0, 1.8, 1.1],
            "shift_risk": [0.1, 1.2, 0.2, 0.9],
            "causal_reliability": [0.9, 0.3, 0.8, 0.4],
            "margin_collapse_mean": [0.1, 0.8, 0.1, 0.6],
            "margin_collapse_q90": [0.2, 0.9, 0.2, 0.7],
            "js_mean": [0.01, 0.1, 0.02, 0.08],
            "flip_mean": [0.0, 1.0, 0.0, 0.5],
        }
    ).to_csv(out / "true_counterfactual_certificates.csv", index=False)
    pd.read_csv(out / "true_counterfactual_certificates.csv").replace({"true_counterfactual": "shuffled_any"}).to_csv(out / "shuffled_any_certificates.csv", index=False)
    summary = build_diagnosis(tmp_path)
    save_outputs(summary, tmp_path)
    assert (tmp_path / "negative_control_diagnosis" / "diagnosis_summary.csv").exists()
    assert (tmp_path / "negative_control_diagnosis" / "plots" / "true_vs_shuffled_shift_risk.png").exists()
