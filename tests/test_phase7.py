from pathlib import Path

from causal_reliability.analysis.final_report import build_report
from causal_reliability.analysis.main_table import build_main_table, save_outputs
from causal_reliability.analysis.sts_figure import make_figure
from causal_reliability.experiments.run_final_negative_controls import run as run_final_negative_controls
from causal_reliability.experiments.run_final_validation import run as run_final_validation


def test_final_validation_runner_writes_outputs(tmp_path: Path):
    cfg = {"results_dir": str(tmp_path), "seeds": [0], "tasks": ["synthetic"], "regimes": ["confidence_solvable", "confident_wrong"], "n_examples": 48}
    run_final_validation(cfg)
    assert (tmp_path / "final_validation" / "final_validation_metrics.csv").exists()
    assert (tmp_path / "final_validation" / "final_validation_summary.md").exists()
    assert (tmp_path / "final_validation" / "plots" / "regime_auc_comparison.png").exists()


def test_final_negative_controls_runner_writes_outputs(tmp_path: Path):
    cfg = {"results_dir": str(tmp_path), "seeds": [0], "n_examples": 48, "controls": ["true_counterfactual", "irrelevant_counterfactuals"]}
    run_final_negative_controls(cfg)
    assert (tmp_path / "final_negative_controls" / "final_negative_control_metrics.csv").exists()
    assert (tmp_path / "final_negative_controls" / "final_negative_control_summary.md").exists()


def test_final_report_table_and_caption(tmp_path: Path):
    run_final_validation({"results_dir": str(tmp_path), "seeds": [0], "tasks": ["synthetic"], "regimes": ["confident_wrong"], "n_examples": 48})
    run_final_negative_controls({"results_dir": str(tmp_path), "seeds": [0], "n_examples": 48, "controls": ["true_counterfactual", "irrelevant_counterfactuals"]})
    build_report(tmp_path)
    table = build_main_table(tmp_path)
    save_outputs(table, tmp_path)
    make_figure(tmp_path)
    assert (tmp_path / "final_report" / "final_report.md").exists()
    assert "CIC AUROC" in (tmp_path / "main_results_table.md").read_text(encoding="utf-8")
    assert (tmp_path / "sts_main_figure_caption.md").exists()


def test_readme_contains_refined_claim():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "Counterfactual Instability Certificates are not universal replacements for confidence" in text
    assert "always beats confidence" not in text.lower()


def test_phase7_scripts_use_python3():
    for path in [Path("scripts/run_phase7_quick.sh"), Path("scripts/run_phase7.sh")]:
        text = path.read_text(encoding="utf-8")
        assert "python3 -m" in text
        assert "python -m" not in text
