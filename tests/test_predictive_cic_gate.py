from __future__ import annotations

"""Tests for the label-free predictive CIC reliability gate.

Covered:
* feature extraction is label-free,
* no-oracle-leakage guard,
* threshold-gate logic,
* leave-one-benchmark-out splitting,
* conservative support-flag logic,
* output confinement (writes only under results/predictive_cic_gate/),
* final headline metrics unchanged by a run,
* a fake backend can never produce a supported gate,
* the conditional predictive-certificate theory encoding.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_reliability.analysis import predictive_cic_gate as gate
from causal_reliability.experiments import run_predictive_cic_gate as runner
from causal_reliability.theory import predictive_certificate as cert

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Synthetic certificate fixture (no real model, deterministic, with signal)
# --------------------------------------------------------------------------- #
def _synthetic_certificates(benchmark: str, n: int = 60, signal: float = 1.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        # ground-truth success is driven by a high repaired confidence (observable)
        rep_conf = float(rng.uniform(0.2, 0.99))
        success = int(rep_conf > 0.6) if rng.uniform() < signal else int(rng.uniform() < 0.5)
        orig_idx = 0
        rep_idx = 0 if rng.uniform() < 0.5 else 1
        rows.append({
            "example_id": i,
            "regime": "misleading",
            "method": "cic_top1",
            "original_confidence": float(rng.uniform(0.3, 0.9)),
            "repaired_confidence": rep_conf,
            "original_prediction_index": orig_idx,
            "repaired_prediction_index": rep_idx,
            "original_correct": bool(rng.uniform() < 0.5),
            "repaired_correct": bool(success),
            "selected_bbox": json.dumps([10, 10, 60, 40]),
            "selected_proposal_type": "horizontal_text_band" if i % 2 else "edge_dense",
            "js_shift": float(rng.uniform(0, 0.5)),
        })
        # a matching consensus row for top-k agreement
        rows.append({**rows[-1], "method": "cic_top3", "repaired_prediction_index": rep_idx})
    return pd.DataFrame(rows)


def _extract(benchmark: str, **kw) -> pd.DataFrame:
    df = _synthetic_certificates(benchmark, **kw)
    return gate.extract_certificate_benchmark(
        benchmark=benchmark, group="controlled", certificates=df,
        cic_method="cic_top1", consensus_method="cic_top3",
    )


# --------------------------------------------------------------------------- #
# 1. Feature extraction is label-free
# --------------------------------------------------------------------------- #
def test_feature_columns_are_label_free():
    gate.assert_label_free(gate.NUMERIC_FEATURES)  # must not raise


def test_extracted_features_contain_no_label_columns():
    frame = _extract("benchA")
    feature_cols = [c for c in frame.columns if c.startswith("feat_")]
    assert feature_cols == gate.NUMERIC_FEATURES
    # no raw correctness/label column survives into the feature namespace
    for forbidden in ("repaired_correct", "original_correct", "true_label", "label", "oracle"):
        assert not any(forbidden in c for c in feature_cols)
    # labels exist but are separate, and the primary label is populated
    assert frame["label_repair_success"].notna().all()
    # a feature is never identical to the label
    assert not np.array_equal(
        frame["feat_repaired_margin"].to_numpy(), frame["label_repair_success"].to_numpy()
    )


def test_assert_label_free_rejects_leaky_name():
    with pytest.raises(ValueError):
        gate.assert_label_free(["feat_orig_confidence", "feat_repaired_correct"])
    with pytest.raises(ValueError):
        gate.assert_label_free(["feat_target_overlap"])


# --------------------------------------------------------------------------- #
# 2. No-oracle-leakage guard
# --------------------------------------------------------------------------- #
def test_no_oracle_leakage_passes_on_clean_features():
    frame = _extract("benchA", n=80)
    ok, reasons = gate.check_no_oracle_leakage(frame, gate.NUMERIC_FEATURES)
    assert ok is True
    # any value-based hits are advisory only
    assert all(r.startswith("[advisory]") for r in reasons)


def test_no_oracle_leakage_catches_label_as_feature():
    frame = _extract("benchA")
    ok, reasons = gate.check_no_oracle_leakage(frame, gate.NUMERIC_FEATURES + ["label_repair_success"])
    assert ok is False
    assert any("label_repair_success" in r for r in reasons)


def test_no_oracle_leakage_catches_forbidden_name():
    frame = _extract("benchA")
    frame["feat_repaired_correct"] = frame["label_repair_success"]
    ok, _ = gate.check_no_oracle_leakage(frame, gate.NUMERIC_FEATURES + ["feat_repaired_correct"])
    assert ok is False


# --------------------------------------------------------------------------- #
# 3. Threshold-gate logic
# --------------------------------------------------------------------------- #
def test_threshold_gate_picks_separating_feature_and_direction():
    n = 200
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=n) < 0.5).astype(int)
    X = pd.DataFrame({
        "feat_signal": y + rng.normal(0, 0.05, n),       # positively separates
        "feat_noise": rng.normal(0, 1, n),
    })
    tg = gate.ThresholdGate().fit(X, y)
    assert tg.feature == "feat_signal"
    assert tg.direction > 0  # accept when high
    probs = tg.predict_proba(X)
    assert gate.auroc(probs, y) > 0.95
    # a negatively-separating feature flips the direction
    X2 = pd.DataFrame({"feat_signal": -(y + rng.normal(0, 0.05, n))})
    tg2 = gate.ThresholdGate().fit(X2, y)
    assert tg2.direction < 0


def test_decision_tree_depth_constraint():
    with pytest.raises(AssertionError):
        gate.DecisionTreeGate(max_depth=4).fit(pd.DataFrame({"feat_a": [0.0, 1.0]}), np.array([0, 1]))


# --------------------------------------------------------------------------- #
# 4. Leave-one-benchmark-out splitting
# --------------------------------------------------------------------------- #
def test_leave_one_benchmark_out_splits_hold_out_each_benchmark():
    frames = [_extract("benchA", seed=1), _extract("benchB", seed=2), _extract("benchC", seed=3)]
    table = pd.concat(frames, ignore_index=True)
    cols = gate.META_COLUMNS + gate.NUMERIC_FEATURES + gate.LABEL_COLUMNS
    table = table[[c for c in cols if c in table.columns]]
    lobo, pooled_scores, pooled_y = gate.leave_one_benchmark_out(table, gate.NUMERIC_FEATURES)
    assert set(lobo["held_out_benchmark"]) == {"benchA", "benchB", "benchC"}
    # each fold's test count equals that benchmark's labelled row count
    for _, r in lobo.iterrows():
        b = r["held_out_benchmark"]
        n_expected = int(table[table["benchmark"] == b]["label_repair_success"].notna().sum())
        assert r["n_test"] == n_expected
    # pooled out-of-fold predictions cover every labelled example exactly once
    assert len(pooled_scores) == int(table["label_repair_success"].notna().sum())


def test_aligned_lobo_scores_never_trains_on_held_benchmark():
    # if a benchmark is the ONLY one, its rows cannot be scored out-of-fold
    table = _extract("solo", n=40)
    table = table.assign(benchmark="solo")
    scores = runner._aligned_lobo_scores(table, gate.NUMERIC_FEATURES, "label_repair_success", "logistic_regression")
    assert np.isnan(scores).all()


# --------------------------------------------------------------------------- #
# 5. Support-flag logic
# --------------------------------------------------------------------------- #
def _good_cov_rows():
    # precision well above 0.8 across coverages >= 0.25
    return [{"coverage": c, "accepted_precision": 0.9, "n_accepted": 100, "abstention": 1 - c}
            for c in (0.25, 0.3, 0.4, 0.5, 0.95)]


def test_support_flag_true_when_all_criteria_met():
    supported, reasons, ev = gate.evaluate_support_flag(
        label_free_ok=True, no_leakage_ok=True, lobo_auroc=0.80,
        coverage_accuracy_rows=_good_cov_rows(), real_evidence=True,
        coco_reported_separately=True, final_metrics_unchanged=True,
    )
    assert supported is True and reasons == []
    assert ev["accepted_coverage"] is not None


def test_support_flag_false_on_low_auroc():
    supported, reasons, _ = gate.evaluate_support_flag(
        label_free_ok=True, no_leakage_ok=True, lobo_auroc=0.60,
        coverage_accuracy_rows=_good_cov_rows(), real_evidence=True,
        coco_reported_separately=True, final_metrics_unchanged=True,
    )
    assert supported is False
    assert any("AUROC" in r for r in reasons)


def test_support_flag_false_when_precision_floor_unreachable():
    cov = [{"coverage": c, "accepted_precision": 0.5, "n_accepted": 100, "abstention": 1 - c}
           for c in (0.25, 0.5, 0.95)]
    supported, reasons, _ = gate.evaluate_support_flag(
        label_free_ok=True, no_leakage_ok=True, lobo_auroc=0.9,
        coverage_accuracy_rows=cov, real_evidence=True,
        coco_reported_separately=True, final_metrics_unchanged=True,
    )
    assert supported is False
    assert any("precision" in r for r in reasons)


def test_support_flag_false_when_final_metrics_changed():
    supported, reasons, _ = gate.evaluate_support_flag(
        label_free_ok=True, no_leakage_ok=True, lobo_auroc=0.9,
        coverage_accuracy_rows=_good_cov_rows(), real_evidence=True,
        coco_reported_separately=True, final_metrics_unchanged=False,
    )
    assert supported is False
    assert any("final" in r.lower() for r in reasons)


# --------------------------------------------------------------------------- #
# 6 & 8. Fake backend can never produce a supported gate
# --------------------------------------------------------------------------- #
def test_fake_backend_blocks_real_evidence(tmp_path):
    j = tmp_path / "fake_key_numbers.json"
    j.write_text(json.dumps({"fake_backend": True, "real_pretrained_model_loaded": False}))
    sources = [{"name": "src", "provenance": {"json": "fake_key_numbers.json", "fake_field": "fake_backend", "real_field": "real_pretrained_model_loaded"}}]
    real, details = runner.assess_real_evidence(sources, tmp_path)
    assert real is False
    assert details["src"]["real"] is False


def test_fake_backend_forces_unsupported_even_with_perfect_metrics():
    supported, reasons, _ = gate.evaluate_support_flag(
        label_free_ok=True, no_leakage_ok=True, lobo_auroc=0.99,
        coverage_accuracy_rows=_good_cov_rows(), real_evidence=False,
        coco_reported_separately=True, final_metrics_unchanged=True,
    )
    assert supported is False
    assert any("real pretrained" in r for r in reasons)


def test_real_evidence_true_for_genuine_provenance(tmp_path):
    j = tmp_path / "real.json"
    j.write_text(json.dumps({"fake_backend": False, "real_pretrained_model_loaded": True}))
    sources = [{"name": "src", "provenance": {"json": "real.json", "fake_field": "fake_backend", "real_field": "real_pretrained_model_loaded"}}]
    real, _ = runner.assess_real_evidence(sources, tmp_path)
    assert real is True


# --------------------------------------------------------------------------- #
# 6. Output confinement + 7. final headline metrics unchanged
# --------------------------------------------------------------------------- #
def test_guard_refuses_protected_directory():
    with pytest.raises(RuntimeError):
        runner._guard_output_dir(Path("results/final_report"))


def _tree_snapshot(root: Path) -> dict[str, str]:
    snap = {}
    if not root.exists():
        return snap
    for p in root.rglob("*"):
        if p.is_file():
            snap[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return snap


@pytest.mark.skipif(not (REPO / "configs/predictive_cic_gate.yaml").exists(), reason="config missing")
def test_run_is_confined_and_leaves_final_report_unchanged():
    from causal_reliability.utils.config import load_config

    cfg = load_config(REPO / "configs/predictive_cic_gate.yaml")
    cfg["base_dir"] = str(REPO)

    final_dir = REPO / "results" / "final_report"
    before_final = _tree_snapshot(final_dir)
    # snapshot everything under results/ EXCEPT the gate's own output dir
    out_dir = REPO / "results" / cfg.get("output_subdir", "predictive_cic_gate")
    before_results = {k: v for k, v in _tree_snapshot(REPO / "results").items() if not k.startswith(str(out_dir))}

    out = runner.run(cfg)

    after_final = _tree_snapshot(final_dir)
    after_results = {k: v for k, v in _tree_snapshot(REPO / "results").items() if not k.startswith(str(out_dir))}

    assert before_final == after_final, "final_report was modified"
    assert before_results == after_results, "a result folder outside the gate's own dir was modified"
    assert Path(out["key_numbers"]).exists()

    kn = json.loads((out_dir / "predictive_gate_key_numbers.json").read_text())
    assert kn["final_report_unchanged"] is True
    assert kn["is_universal_theorem"] is False


@pytest.mark.skipif(not (REPO / "results/predictive_cic_gate/predictive_gate_key_numbers.json").exists(), reason="run not yet produced")
def test_produced_outputs_exist_and_are_consistent():
    out_dir = REPO / "results" / "predictive_cic_gate"
    for fname in [
        "predictive_gate_key_numbers.json",
        "predictive_gate_summary.md",
        "predictive_gate_features.csv",
        "predictive_gate_eval_by_benchmark.csv",
        "predictive_gate_leave_one_benchmark_out.csv",
        "coverage_accuracy_curve.csv",
        "calibration_curve.csv",
        "predictive_gate_plots.png",
    ]:
        assert (out_dir / fname).exists(), f"missing output {fname}"
    feats = pd.read_csv(out_dir / "predictive_gate_features.csv")
    # the features file must not leak forbidden raw columns
    for c in feats.columns:
        if c.startswith("feat_"):
            gate.assert_label_free([c])


# --------------------------------------------------------------------------- #
# Conditional predictive-certificate theory encoding
# --------------------------------------------------------------------------- #
def test_predictive_certificate_fires_above_threshold_and_abstains_below():
    eps_hat = 0.5
    confident = np.array([3.0, 0.0, -1.0])  # margin 3.0 > 2*eps_hat=1.0
    assert cert.certified_stable(confident, eps_hat) is True
    assert cert.certificate_for(confident, eps_hat).decision == "certify_stable"
    unsure = np.array([0.3, 0.0, -1.0])      # margin 0.3 < 1.0
    assert cert.certified_stable(unsure, eps_hat) is False
    assert cert.certificate_for(unsure, eps_hat).decision == "abstain"


def test_predictive_certificate_implies_stability_under_calibrated_class():
    eps_hat = 0.4
    logits = np.array([2.0, 0.0, -0.5])  # margin 2.0 > 0.8
    assert cert.certified_stable(logits, eps_hat)
    rng = np.random.default_rng(0)
    perturbations = []
    for _ in range(50):
        d = rng.normal(0, 0.1, size=3)
        d = d - d.mean()  # within-class-balanced direction
        perturbations.append(d)
    assert cert.stable_under_perturbation(logits, perturbations, eps_hat) is True


def test_calibrate_residual_bound_is_conservative_quantile():
    vals = np.linspace(0, 1, 101)
    assert cert.calibrate_residual_bound(vals, 0.9) == pytest.approx(0.9, abs=1e-6)
    assert cert.calibrate_residual_bound([]) == float("inf")
    assert cert.certified_stable(np.array([5.0, 0.0]), float("inf")) is False


# --------------------------------------------------------------------------- #
# Paper integration: predictive abstention certificate is stated, bounded
# --------------------------------------------------------------------------- #
PAPER = REPO / "paper" / "main.tex"


@pytest.mark.skipif(not PAPER.exists(), reason="paper missing")
def test_paper_states_predictive_abstention_certificate():
    text = PAPER.read_text()
    lowered = text.lower()
    # the theorem/proposition title appears (either allowed wording)
    assert (
        "predictive cic abstention certificate" in lowered
        or "margin-based predictive repair certificate" in lowered
    )
    # the formal condition is present
    assert "m_{\\mathrm{rep}}(x) > 2\\hat\\epsilon" in text
    # the proposition is a real environment with a proof
    assert "\\begin{proposition}" in text and "\\label{prop:predcert}" in text
    # the predictive-gate connection numbers are present
    assert "0.789" in text and "0.97" in text


@pytest.mark.skipif(not PAPER.exists(), reason="paper missing")
def test_paper_avoids_forbidden_overclaims():
    lowered = PAPER.read_text().lower()
    forbidden = [
        "universal open-world discovery",
        "guaranteed semantic correctness",
        "cic always knows when it is right",
        "universal natural-image robustness",
    ]
    hits = [phrase for phrase in forbidden if phrase in lowered]
    assert hits == [], f"forbidden overclaim(s) introduced in paper: {hits}"
