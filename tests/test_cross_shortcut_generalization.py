from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from causal_reliability.real_models.clip_zero_shot import ClipStatus


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------
class TinyClassifier:
    def __init__(self, status, class_names, prompts=None, device="cpu"):
        self.status = status
        self.class_names = class_names

    def predict(self, images):
        means = images.mean(dim=(1, 2, 3)).numpy()
        probs = np.full((len(means), len(self.class_names)), 0.05, dtype=np.float32)
        preds = np.floor(means * 997).astype(int) % len(self.class_names)
        probs[np.arange(len(means)), preds] = 0.85
        probs = probs / probs.sum(axis=1, keepdims=True)
        return {"probabilities": torch.from_numpy(probs)}


def _fake_status(**kwargs):
    return ClipStatus(
        available=True,
        backend="open_clip",
        model_name="ViT-B-32",
        pretrained_tag="laion2b_s34b_b79k",
        pretrained=True,
        downloads_allowed=kwargs.get("allow_download", False),
        backend_attempted=kwargs.get("preferred_backend", "open_clip"),
    )


def _write_frozen_repair_policy(frozen_dir: Path, threshold: float = 0.0) -> None:
    frozen_dir.mkdir(parents=True, exist_ok=True)
    (frozen_dir / "selected_repair_policy.json").write_text(
        json.dumps({"objective": "validation_clean_safe_repair", "score_threshold": threshold, "min_consensus_stability": 0.6666666667}),
        encoding="utf-8",
    )


def _run(tmp_path: Path, monkeypatch, **overrides):
    from causal_reliability.experiments import run_cross_shortcut_generalization as mod

    monkeypatch.setattr(mod, "check_clip_available", _fake_status)
    monkeypatch.setattr(mod, "ClipZeroShotClassifier", TinyClassifier)
    frozen = tmp_path / "frozen"
    _write_frozen_repair_policy(frozen)
    cfg = {
        "results_dir": str(tmp_path),
        "frozen_policy_dir": str(frozen),
        "pool_per_class": 6,
        "n_failure_target": 4,
        "original_confidence_threshold": 0.0,
        "max_candidates": 12,
        "augmentation_views": 2,
        "random_draws": 3,
        "data": {"image_size": 64, "natural_n_per_class": 1},
        "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
    }
    cfg.update(overrides)
    return mod.run(cfg), frozen


# ---------------------------------------------------------------------------
# PART 2: frozen policy is loaded from the hard multi-decoy results, not reselected
# ---------------------------------------------------------------------------
def test_selected_repair_policy_loaded_from_hard_multidecoy_results():
    from causal_reliability.experiments import run_cross_shortcut_generalization as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    # The frozen policy is loaded from selected_repair_policy.json in the frozen dir.
    assert "selected_repair_policy.json" in src
    assert 'frozen_policy_dir", "results/hard_multidecoy_clip_repair"' in src


def test_missing_frozen_policy_fails_clearly(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_cross_shortcut_generalization as mod

    monkeypatch.setattr(mod, "check_clip_available", _fake_status)
    monkeypatch.setattr(mod, "ClipZeroShotClassifier", TinyClassifier)
    outputs = mod.run(
        {
            "results_dir": str(tmp_path),
            "frozen_policy_dir": str(tmp_path / "does_not_exist"),
            "data": {"image_size": 64, "natural_n_per_class": 1},
            "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
        }
    )
    kn = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    assert kn["cross_shortcut_headline_eligible"] is False
    assert kn["frozen_text_selected_policy_used"] is False
    assert any("frozen" in r.lower() for r in kn["cross_shortcut_headline_failed_reasons"])


def test_policy_is_not_reselected_on_cross_shortcut_benchmark():
    from causal_reliability.experiments import run_cross_shortcut_generalization as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    # No validation-based policy selection / sweep is performed on the new shortcut.
    assert "_select_policy" not in src
    assert "policy_sweep" not in src
    assert "validation_repair_policy_sweep" not in src
    # The frozen policy object is applied verbatim via _policy_action.
    assert "_policy_action(" in src
    assert mod.FROZEN_POLICY_LABEL == "Frozen text-selected CIC policy applied to non-text shortcut proposals."


# ---------------------------------------------------------------------------
# PART 1 / PART 3: non-oracle scorer must not receive label / harmful bbox / type
# ---------------------------------------------------------------------------
def test_nonoracle_scorer_does_not_receive_label_or_harmful_bbox():
    from causal_reliability.experiments import run_cross_shortcut_generalization as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "discover_clip_shortcut_regions(\n            pil, predict_fn, prompts" in src
    # The scorer must never be handed the label, harmful bbox, shortcut type, or correctness.
    assert 'discover_clip_shortcut_regions(pil, predict_fn, prompts, ex["label"]' not in src
    assert 'discover_clip_shortcut_regions(pil, predict_fn, prompts, ex["harmful' not in src


def test_shortcut_type_is_non_text():
    from causal_reliability.experiments import run_cross_shortcut_generalization as mod

    assert mod.SHORTCUT_TYPE == "colored_symbol_watermark"
    _, meta = mod.render_cross_shortcut_image(0, mod.MIS_REGIME, 1, size=96, benchmark_seed=0)
    # No readable text key carries content; the shortcut is a colored symbol watermark.
    assert meta["harmful_text"] == ""
    assert meta["shortcut_type"] == "colored_symbol_watermark"
    assert meta["harmful_shortcut_bbox"]  # misleading regime has a harmful badge
    assert meta["shortcut_label_association"] in {"square", "triangle", "star"}  # a WRONG class for a circle


# ---------------------------------------------------------------------------
# PART 5: failure-conditioned subset requires all four conditions + oracle repair
# ---------------------------------------------------------------------------
def test_failure_conditioned_subset_requires_all_conditions():
    from causal_reliability.experiments.run_cross_shortcut_generalization import passes_failure_conditions

    base = dict(no_overlay_correct=True, aligned_correct=True, misleading_wrong=True, confidence_ok=True, oracle_restored=True)
    assert passes_failure_conditions(**base) is True
    for key in base:
        flipped = dict(base)
        flipped[key] = False
        assert passes_failure_conditions(**flipped) is False, f"{key} must be required for inclusion"


def test_run_writes_all_outputs_and_inclusion_invariant(tmp_path: Path, monkeypatch):
    outputs, _ = _run(tmp_path, monkeypatch)
    for key in ["metrics", "failure_metrics", "certificates", "inclusion_log", "repair_vs_localization", "key_numbers", "summary", "examples", "plot_png", "plot_pdf", "caption"]:
        assert Path(outputs[key]).exists(), key

    log = pd.read_csv(outputs["inclusion_log"])
    assert {"no_overlay_correct", "aligned_correct", "misleading_wrong", "oracle_restored", "included", "shortcut_type"}.issubset(log.columns)
    for _, row in log[log["included"].astype(bool)].iterrows():
        assert bool(row["no_overlay_correct"]) and bool(row["aligned_correct"])
        assert bool(row["misleading_wrong"]) and bool(row["oracle_restored"])


# ---------------------------------------------------------------------------
# PART 4: random matched region baseline is computed
# ---------------------------------------------------------------------------
def test_random_matched_region_baseline_is_computed(tmp_path: Path, monkeypatch):
    outputs, _ = _run(tmp_path, monkeypatch)
    certs = pd.read_csv(outputs["certificates"])
    assert "random_matched_region_repair" in set(certs["method"])
    # Many draws per example, so >1 random-matched cert per example.
    rmr = certs[certs["method"] == "random_matched_region_repair"]
    assert len(rmr) >= 3
    # Center-object and highest-contrast baselines are present too (PART 4).
    for method in ["center_object_region_repair", "highest_contrast_region_repair", "largest_region_repair", "random_augmentation_consensus"]:
        assert method in set(certs["method"]), method


# ---------------------------------------------------------------------------
# PART 7: headline eligibility requires frozen policy + no tuning; fake backend cannot pass
# ---------------------------------------------------------------------------
def _failure_df(cic_top1, random_mean, oracle=0.95, nov=1.0, aligned=0.95, n=40):
    return pd.DataFrame(
        [
            {"method": "original_clip_prediction", "failure_subset_original_accuracy": 0.0, "failure_subset_repaired_accuracy": 0.0, "n_failure_examples": n},
            {"method": "oracle_shortcut_neutralization", "failure_subset_repaired_accuracy": oracle, "n_failure_examples": n},
            {"method": "frozen_cic_top1_repair", "failure_subset_repaired_accuracy": cic_top1, "n_failure_examples": n},
            {"method": "frozen_cic_top3_repair", "failure_subset_repaired_accuracy": cic_top1 - 0.02, "n_failure_examples": n},
            {"method": "frozen_cic_clean_safe_repair", "failure_subset_repaired_accuracy": cic_top1 - 0.05, "n_failure_examples": n, "no_overlay_preservation_after": nov, "aligned_preservation_after": aligned},
            {"method": "frozen_cic_selective_repair_or_abstain", "selective_accuracy": cic_top1, "coverage": 0.9, "abstention_rate": 0.1, "n_failure_examples": n},
            {"method": "random_matched_region_repair", "failure_subset_repaired_accuracy": random_mean, "n_failure_examples": n, "random_matched_repair_mean": random_mean, "random_matched_repair_std": 0.05, "random_matched_repair_ci95": 0.03},
        ]
    ).assign(harmful_top1_iou_0_3=0.6, harmful_top1_iou_0_5=0.2, harmful_top3_iou_0_3=0.7, harmful_top3_iou_0_5=0.25)


def _natural_df(mis_before=0.0):
    return pd.DataFrame(
        [
            {"method": "original_clip_prediction", "misleading_accuracy_before": mis_before, "no_overlay_accuracy_before": 1.0, "aligned_accuracy_before": 1.0, "neutral_accuracy_before": 1.0},
            {"method": "oracle_shortcut_neutralization", "misleading_accuracy_after": 0.95},
            {"method": "frozen_cic_top1_repair", "misleading_accuracy_after": 0.85},
            {"method": "frozen_cic_top3_repair", "misleading_accuracy_after": 0.83},
            {"method": "frozen_cic_clean_safe_repair", "misleading_accuracy_after": 0.80, "no_overlay_accuracy_after": 1.0},
            {"method": "random_matched_region_repair", "random_matched_repair_mean": 0.3, "random_matched_repair_ci95": 0.03},
        ]
    )


def test_headline_eligibility_requires_frozen_policy_and_no_tuning():
    from causal_reliability.experiments.run_cross_shortcut_generalization import compute_headline_eligibility

    status = ClipStatus(available=True, backend="open_clip", model_name="ViT-B-32", pretrained_tag="t", pretrained=True, downloads_allowed=False, backend_attempted="open_clip")
    stats = {"n_candidates": 80, "n_failure_examples": 40, "inclusion_rate": 0.5}

    kn, reasons = compute_headline_eligibility(_natural_df(), _failure_df(0.85, 0.30), stats, status, {})
    assert kn["cross_shortcut_headline_eligible"] is True, reasons
    assert kn["frozen_text_selected_policy_used"] is True
    assert kn["no_cross_shortcut_tuning"] is True
    assert kn["nonoracle_scorer_excludes_labels_bboxes_correctness"] is True
    assert kn["cic_beats_random"] is True

    # CIC not beating random -> not eligible.
    kn2, reasons2 = compute_headline_eligibility(_natural_df(), _failure_df(0.40, 0.35), stats, status, {})
    assert kn2["cross_shortcut_headline_eligible"] is False
    assert any("0.70" in r for r in reasons2)

    # Too few failures AND natural misleading accuracy not low -> not eligible.
    few = {"n_candidates": 80, "n_failure_examples": 10, "inclusion_rate": 0.1}
    kn3, reasons3 = compute_headline_eligibility(_natural_df(mis_before=0.9), _failure_df(0.85, 0.30, n=10), few, status, {})
    assert kn3["cross_shortcut_headline_eligible"] is False
    assert any("< 30" in r for r in reasons3)


def test_fake_backend_cannot_be_headline_eligible(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_cross_shortcut_generalization as mod

    monkeypatch.setattr(mod, "ClipZeroShotClassifier", TinyClassifier)
    frozen = tmp_path / "frozen"
    _write_frozen_repair_policy(frozen)
    outputs = mod.run(
        {
            "results_dir": str(tmp_path),
            "frozen_policy_dir": str(frozen),
            "data": {"image_size": 64, "natural_n_per_class": 1},
            "model": {"preferred_backend": "fake", "device": "cpu"},
        }
    )
    kn = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    assert kn["cross_shortcut_headline_eligible"] is False
    assert kn["fake_backend"] is True
    assert kn["frozen_text_selected_policy_used"] is False


# ---------------------------------------------------------------------------
# PART 6: summary labels transfer as finite-candidate, not open-world
# ---------------------------------------------------------------------------
def test_summary_labels_finite_candidate_not_open_world(tmp_path: Path, monkeypatch):
    outputs, _ = _run(tmp_path, monkeypatch)
    summary = Path(outputs["summary"]).read_text(encoding="utf-8").lower()
    assert "finite-candidate" in summary or "finite candidate" in summary
    assert "not open-world shortcut discovery" in summary
    assert "failure-conditioned transfer evaluation" in summary or "failure-conditioned" in summary


# ---------------------------------------------------------------------------
# PART 8: final report labels cross-shortcut transfer as finite-candidate, not open-world
# ---------------------------------------------------------------------------
def _hard_metrics_csv(hard_dir: Path) -> None:
    hard_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"method": "original_clip_prediction", "headline_eligible": True, "evidence_status": "pretrained CLIP hard multi-decoy non-oracle repair evidence", "pretrained_loaded": True, "hard_multi_decoy_misleading_accuracy_before": 0.219},
            {"method": "nonoracle_cic_top1_repair", "headline_eligible": True, "pretrained_loaded": True, "hard_multi_decoy_misleading_accuracy_after": 0.875},
        ]
    ).to_csv(hard_dir / "hard_multidecoy_repair_metrics.csv", index=False)


def test_final_report_includes_cross_shortcut_finite_candidate_caveat(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    _hard_metrics_csv(tmp_path / "hard_multidecoy_clip_repair")
    cs_dir = tmp_path / "cross_shortcut_generalization"
    cs_dir.mkdir(parents=True)
    (cs_dir / "cross_shortcut_key_numbers.json").write_text(
        json.dumps(
            {
                "cross_shortcut_headline_eligible": True,
                "shortcut_type": "colored_symbol_watermark",
                "n_failure_examples": 35,
                "inclusion_rate": 0.4,
                "oracle_repair_accuracy": 0.95,
                "cic_top1_repair_accuracy": 0.82,
                "cic_top3_repair_accuracy": 0.80,
                "random_matched_repair_mean": 0.30,
                "random_matched_repair_95ci": 0.04,
                "cic_minus_random_gap": 0.52,
                "cic_beats_random": True,
                "no_overlay_preservation_after": 1.0,
                "aligned_preservation_after": 0.95,
                "natural_misleading_accuracy_before": 0.25,
            }
        ),
        encoding="utf-8",
    )
    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    key_numbers = json.loads((tmp_path / "final_report" / "final_key_numbers.json").read_text())

    assert key_numbers["cross_shortcut_available"] is True
    assert "Cross-Shortcut Generalization Attempt" in report
    assert "finite candidate" in report.lower()
    assert "not open-world shortcut discovery" in report.lower()


def test_final_report_reports_failed_cross_shortcut_transfer_honestly(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    _hard_metrics_csv(tmp_path / "hard_multidecoy_clip_repair")
    cs_dir = tmp_path / "cross_shortcut_generalization"
    cs_dir.mkdir(parents=True)
    (cs_dir / "cross_shortcut_key_numbers.json").write_text(
        json.dumps(
            {
                "cross_shortcut_headline_eligible": False,
                "shortcut_type": "colored_symbol_watermark",
                "cross_shortcut_headline_failed_reasons": ["frozen CIC top-1/top-3 repair accuracy < 0.70"],
                "n_failure_examples": 8,
            }
        ),
        encoding="utf-8",
    )
    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    assert "did not support cross-shortcut generalization" in report.lower()
