from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from causal_reliability.real_models.clip_zero_shot import ClipStatus


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------
class TinyClassifier:
    def __init__(self, status, class_names, prompts=None, device="cpu"):
        self.status = status
        self.class_names = class_names

    def predict(self, images):
        import torch

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


def _write_frozen_policies(frozen_dir: Path) -> None:
    frozen_dir.mkdir(parents=True, exist_ok=True)
    (frozen_dir / "selected_generation_policy.json").write_text(
        json.dumps({"policy_id": "frozen_test", "class_set_size": 4, "n_decoys": 4, "placement_jitter": 4}),
        encoding="utf-8",
    )
    (frozen_dir / "selected_repair_policy.json").write_text(
        json.dumps({"score_threshold": 0.0, "min_consensus_stability": 0.6666666667}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# PART 3 / PART 6: failure-conditioned subset construction
# ---------------------------------------------------------------------------
def test_failure_conditioned_subset_requires_all_conditions():
    from causal_reliability.experiments.run_hard_multidecoy_failure_conditioned_repair import _passes_failure_conditions

    base = dict(no_overlay_correct=True, aligned_correct=True, misleading_wrong=True, confidence_ok=True, oracle_restored=True)
    assert _passes_failure_conditions(**base) is True
    for key in base:
        flipped = dict(base)
        flipped[key] = False
        assert _passes_failure_conditions(**flipped) is False, f"{key} should be required for inclusion"


def test_failure_conditioned_run_writes_outputs_and_inclusion_invariant(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_hard_multidecoy_failure_conditioned_repair as mod

    monkeypatch.setattr(mod, "check_clip_available", _fake_status)
    monkeypatch.setattr(mod, "ClipZeroShotClassifier", TinyClassifier)
    frozen = tmp_path / "frozen"
    _write_frozen_policies(frozen)

    outputs = mod.run(
        {
            "results_dir": str(tmp_path),
            "frozen_policy_dir": str(frozen),
            "pool_per_class": 8,
            "n_failure_target": 6,
            "original_confidence_threshold": 0.0,
            "max_candidates": 16,
            "augmentation_views": 2,
            "random_draws": 3,
            "data": {"image_size": 64},
            "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
        }
    )

    for key in ["metrics", "certificates", "inclusion_log", "repair_vs_localization", "key_numbers", "summary"]:
        assert Path(outputs[key]).exists()

    log = pd.read_csv(outputs["inclusion_log"])
    assert {"no_overlay_correct", "aligned_correct", "misleading_wrong", "oracle_restored", "included", "exclusion_reasons"}.issubset(log.columns)
    included = log[log["included"].astype(bool)]
    for _, row in included.iterrows():
        assert bool(row["no_overlay_correct"]) and bool(row["aligned_correct"])
        assert bool(row["misleading_wrong"]) and bool(row["oracle_restored"])

    summary = Path(outputs["summary"]).read_text(encoding="utf-8")
    assert "failure-conditioned repair evaluation" in summary.lower()
    assert "not open-world shortcut discovery" in summary.lower()


def test_failure_conditioned_nonoracle_scorer_is_label_free():
    # The failure-conditioned benchmark reuses the shared non-oracle scorer, which
    # only passes pixels, the prediction function, and prompts to discovery.
    fc_src = Path("causal_reliability/experiments/run_hard_multidecoy_failure_conditioned_repair.py").read_text(encoding="utf-8")
    scorer_src = Path("causal_reliability/experiments/run_multidecoy_clip_repair.py").read_text(encoding="utf-8")
    assert "_evaluate_examples" in fc_src
    assert "discover_clip_shortcut_regions(pil, predict_fn, prompts" in scorer_src
    # discovery must not receive the example label / harmful bbox / correctness.
    assert "discover_clip_shortcut_regions(pil, predict_fn, prompts, ex[\"label\"]" not in scorer_src
    assert "discover_clip_shortcut_regions(pil, predict_fn, prompts, ex[\"harmful_bbox\"]" not in scorer_src


def test_failure_conditioned_repair_vs_localization_crosstab_is_written(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_hard_multidecoy_failure_conditioned_repair as mod

    monkeypatch.setattr(mod, "check_clip_available", _fake_status)
    monkeypatch.setattr(mod, "ClipZeroShotClassifier", TinyClassifier)
    frozen = tmp_path / "frozen"
    _write_frozen_policies(frozen)
    outputs = mod.run(
        {
            "results_dir": str(tmp_path),
            "frozen_policy_dir": str(frozen),
            "pool_per_class": 8,
            "n_failure_target": 6,
            "original_confidence_threshold": 0.0,
            "max_candidates": 16,
            "augmentation_views": 2,
            "random_draws": 3,
            "data": {"image_size": 64},
            "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
        }
    )
    assert Path(outputs["repair_vs_localization"]).exists()


def test_failure_conditioned_headline_eligibility_reflects_benchmark_type():
    from causal_reliability.experiments.run_hard_multidecoy_failure_conditioned_repair import _key_numbers

    status = ClipStatus(available=True, backend="open_clip", model_name="ViT-B-32", pretrained_tag="t", pretrained=True, downloads_allowed=False, backend_attempted="open_clip")
    stats = {"n_candidates": 80, "n_failure_examples": 40, "inclusion_rate": 0.5, "original_confidence_threshold": 0.5}

    def metrics_for(cic_top1, random_mean, oracle=0.95, nov=1.0, aligned=0.95):
        return pd.DataFrame(
            [
                {"method": "original_clip_prediction", "failure_subset_repaired_accuracy": 0.0, "n_failure_examples": 40},
                {"method": "oracle_harmful_text_neutralization", "failure_subset_repaired_accuracy": oracle, "n_failure_examples": 40},
                {"method": "nonoracle_cic_top1_repair", "failure_subset_repaired_accuracy": cic_top1, "n_failure_examples": 40, "no_overlay_preservation_after": nov, "aligned_preservation_after": aligned},
                {"method": "nonoracle_cic_top3_repair", "failure_subset_repaired_accuracy": cic_top1 - 0.02, "n_failure_examples": 40},
                {"method": "nonoracle_cic_clean_safe_repair", "failure_subset_repaired_accuracy": cic_top1 - 0.05, "n_failure_examples": 40, "no_overlay_preservation_after": nov, "aligned_preservation_after": aligned},
                {"method": "random_matched_text_region_repair", "failure_subset_repaired_accuracy": random_mean, "n_failure_examples": 40, "random_draw_failure_accuracy_mean": random_mean, "random_draw_failure_accuracy_std": 0.05, "random_draw_failure_accuracy_ci95": 0.03},
            ]
        ).assign(harmful_top1_iou_0_3=0.6, harmful_top1_iou_0_5=0.2, harmful_top3_iou_0_3=0.7, harmful_top3_iou_0_5=0.25)

    eligible, reasons = _key_numbers(metrics_for(0.85, 0.30), stats, status, {})
    assert eligible["failure_conditioned_headline_eligible"] is True, reasons
    assert eligible["cic_beats_random"] is True

    not_eligible, reasons2 = _key_numbers(metrics_for(0.40, 0.35), stats, status, {})
    assert not_eligible["failure_conditioned_headline_eligible"] is False
    assert any("0.75" in r for r in reasons2)

    small_n = dict(stats)
    small_n["n_failure_examples"] = 10
    few, reasons3 = _key_numbers(metrics_for(0.85, 0.30), small_n, status, {})
    assert few["failure_conditioned_headline_eligible"] is False
    assert any("< 30" in r for r in reasons3)


# ---------------------------------------------------------------------------
# PART 2 / PART 6: full benchmark-resampling audit
# ---------------------------------------------------------------------------
def test_full_resampling_audit_uses_different_image_hashes_across_seeds():
    from causal_reliability.experiments.run_hard_multidecoy_clip_repair import make_hard_dataset
    from causal_reliability.experiments.run_hard_multidecoy_clip_repair_audit import _image_set_hash, _metadata_hash

    policy = {"policy_id": "test", "class_set_size": 4, "n_decoys": 4, "placement_jitter": 6}
    sets = [make_hard_dataset(2, policy, size=96, split="test", benchmark_seed=s, resample=True) for s in range(3)]
    image_hashes = {_image_set_hash(s) for s in sets}
    metadata_hashes = {_metadata_hash(s) for s in sets}
    assert len(image_hashes) == 3
    assert len(metadata_hashes) == 3


def test_full_resampling_aggregate_has_required_fields(tmp_path: Path):
    from causal_reliability.experiments.run_hard_multidecoy_clip_repair_audit import _write_summary

    out_dir = tmp_path / "hard_multidecoy_clip_repair"
    out_dir.mkdir()
    df = pd.DataFrame(
        [
            {"seed_id": 0, "benchmark_resampled": True, "lite_mode": False, "full_resample": True, "headline_eligible": True, "original_hard_misleading_accuracy": 0.20, "cic_top1_repair_accuracy": 0.85, "random_matched_text_repair_mean": 0.30, "clean_safe_clean_drop": 0.02},
            {"seed_id": 1, "benchmark_resampled": True, "lite_mode": False, "full_resample": True, "headline_eligible": True, "original_hard_misleading_accuracy": 0.25, "cic_top1_repair_accuracy": 0.80, "random_matched_text_repair_mean": 0.28, "clean_safe_clean_drop": 0.01},
        ]
    )
    _write_summary(df, out_dir, "full_benchmark_resampling_audit", "Full Audit", resample_benchmark=True, full_resample=True)

    assert (out_dir / "full_benchmark_resampling_audit.csv").exists()
    agg = pd.read_csv(out_dir / "full_benchmark_resampling_aggregate.csv")
    for col in [
        "original_accuracy_mean",
        "original_accuracy_min",
        "cic_top1_mean",
        "random_matched_mean",
        "cic_minus_random_gap_min",
        "n_seeds_original_accuracy_le_0_40",
        "n_seeds_cic_top1_ge_0_80",
        "n_seeds_cic_beats_random_ge_0_15",
        "n_eligible_seeds",
        "full_benchmark_resampling_stability_supported",
    ]:
        assert col in agg.columns
    assert bool(agg["full_benchmark_resampling_stability_supported"].iloc[0]) is True


# ---------------------------------------------------------------------------
# PART 5 / PART 6: final report decision logic and honest labelling
# ---------------------------------------------------------------------------
def _hard_metrics_csv(hard_dir: Path) -> None:
    hard_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"method": "original_clip_prediction", "headline_eligible": True, "evidence_status": "pretrained CLIP hard multi-decoy non-oracle repair evidence", "pretrained_loaded": True, "hard_multi_decoy_misleading_accuracy_before": 0.219},
            {"method": "nonoracle_cic_top1_repair", "headline_eligible": True, "pretrained_loaded": True, "hard_multi_decoy_misleading_accuracy_after": 0.875},
        ]
    ).to_csv(hard_dir / "hard_multidecoy_repair_metrics.csv", index=False)


def test_final_report_does_not_call_lite_resampling_full_stability(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    hard_dir = tmp_path / "hard_multidecoy_clip_repair"
    _hard_metrics_csv(hard_dir)
    pd.DataFrame(
        [
            {"seed_id": 0, "benchmark_resampled": True, "lite_mode": True, "headline_eligible": True, "cic_top1_repair_accuracy": 0.25, "random_matched_text_repair_mean": 0.50},
            {"seed_id": 1, "benchmark_resampled": True, "lite_mode": True, "headline_eligible": True, "cic_top1_repair_accuracy": 0.75, "random_matched_text_repair_mean": 0.30},
        ]
    ).to_csv(hard_dir / "benchmark_resampling_audit.csv", index=False)

    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    key_numbers = json.loads((tmp_path / "final_report" / "final_key_numbers.json").read_text())

    assert key_numbers["hard_multidecoy_benchmark_resampling_available"] is False
    assert key_numbers["hard_multidecoy_full_resampling_available"] is False
    assert "The main hard multi-decoy result is a strong single-benchmark result." in report
    assert "survived independent resampling" not in report
    assert "too small and volatile" in report


def test_final_report_main_claim_tier_a_for_full_resampling(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    hard_dir = tmp_path / "hard_multidecoy_clip_repair"
    _hard_metrics_csv(hard_dir)
    pd.DataFrame(
        [
            {"seed_id": 0, "benchmark_resampled": True, "lite_mode": False, "full_resample": True, "headline_eligible": True, "cic_top1_repair_accuracy": 0.85, "random_matched_text_repair_mean": 0.30},
            {"seed_id": 1, "benchmark_resampled": True, "lite_mode": False, "full_resample": True, "headline_eligible": True, "cic_top1_repair_accuracy": 0.82, "random_matched_text_repair_mean": 0.28},
        ]
    ).to_csv(hard_dir / "full_benchmark_resampling_audit.csv", index=False)

    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    key_numbers = json.loads((tmp_path / "final_report" / "final_key_numbers.json").read_text())
    assert key_numbers["hard_multidecoy_full_resampling_available"] is True
    assert key_numbers["hard_multidecoy_main_claim_tier"] == "A"
    assert "survived independent resampling" in report


def test_final_report_labels_failure_conditioned_benchmark_correctly(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    _hard_metrics_csv(tmp_path / "hard_multidecoy_clip_repair")
    fc_dir = tmp_path / "hard_multidecoy_failure_conditioned"
    fc_dir.mkdir(parents=True)
    (fc_dir / "failure_conditioned_key_numbers.json").write_text(
        json.dumps(
            {
                "failure_conditioned_headline_eligible": True,
                "n_candidates": 80,
                "n_failure_examples": 40,
                "inclusion_rate": 0.5,
                "failure_subset_original_accuracy": 0.0,
                "oracle_repair_accuracy": 0.95,
                "cic_top1_repair_accuracy": 0.85,
                "cic_top3_repair_accuracy": 0.83,
                "cic_clean_safe_repair_accuracy": 0.80,
                "random_matched_text_repair_mean": 0.30,
                "random_matched_text_repair_95ci": 0.03,
                "cic_minus_random_gap": 0.55,
                "cic_beats_random": True,
                "no_overlay_preservation_after": 1.0,
                "aligned_preservation_after": 0.95,
                "harmful_top1_iou_0_3": 0.6,
                "harmful_top1_iou_0_5": 0.2,
                "harmful_top3_iou_0_3": 0.7,
            }
        ),
        encoding="utf-8",
    )

    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    key_numbers = json.loads((tmp_path / "final_report" / "final_key_numbers.json").read_text())

    assert key_numbers["failure_conditioned_available"] is True
    assert key_numbers["hard_multidecoy_main_claim_tier"] == "B"
    assert "Failure-Conditioned Hard Multi-Decoy Repair Evaluation" in report
    assert "On held-out failure-conditioned hard multi-decoy CLIP examples" in report
    # original accuracy on the failure subset must not be framed as a natural benchmark accuracy
    assert "~0 by construction" in report
    assert "not a natural benchmark accuracy" in report
    assert "not open-world shortcut discovery" in report


def test_final_report_includes_finite_candidate_and_no_open_world_caveats(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    _hard_metrics_csv(tmp_path / "hard_multidecoy_clip_repair")
    report = build_report(tmp_path)["report"].read_text(encoding="utf-8").lower()
    assert "finite candidate" in report
    assert "does not solve open-world shortcut discovery" in report


def test_final_report_includes_required_interpretation_sentence(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    _hard_metrics_csv(tmp_path / "hard_multidecoy_clip_repair")
    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    assert (
        "The hard multi-decoy result should currently be interpreted as a strong controlled held-out benchmark "
        "result, not yet as full benchmark-resampling stability."
    ) in report
