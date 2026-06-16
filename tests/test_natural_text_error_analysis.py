from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal_reliability.analysis.natural_text_error_analysis import (
    aggregate_directional_metrics,
    best_rank,
    build_label_info,
    categorize_failure,
    directional_metrics_row,
    evaluate_directional_evidence,
    label_rank,
    selection_geometry,
    text_overlap_bucket,
)

# Reuse the existing tiny-fixture harness from the strict eval test.
from causal_reliability.real_models.clip_zero_shot import ClipStatus
from tests.test_natural_text_verified_failure_eval import _base_cfg, _make_fixture


def _patch_real_clip(monkeypatch, mod):
    """Patch the error-analysis module to use a deterministic fake CLIP.

    Mirrors ``tests.test_natural_text_verified_failure_eval._patch_real_clip`` but
    targets ``_build_predict_fn_with_logits`` (probs + logits).
    """

    def _factory(status, allowed_labels, device):
        n = len(allowed_labels)

        def predict(images):
            arrays = [np.asarray(img.convert("RGB")).astype(np.float32) / 255.0 for img in images]
            means = np.stack(arrays).mean(axis=(1, 2, 3))
            probs = np.full((len(means), n), 0.05, dtype=np.float64)
            preds = np.floor(means * 1000).astype(int) % n
            probs[np.arange(len(means)), preds] = 0.9
            probs = probs / probs.sum(axis=1, keepdims=True)
            logits = np.log(np.clip(probs, 1e-12, 1.0))
            return probs, logits

        return predict

    monkeypatch.setattr(mod, "_build_predict_fn_with_logits", _factory)
    monkeypatch.setattr(
        mod,
        "check_clip_available",
        lambda **kwargs: ClipStatus(
            available=True,
            backend="open_clip",
            model_name="ViT-B-32",
            pretrained_tag="laion2b_s34b_b79k",
            pretrained=True,
            downloads_allowed=kwargs.get("allow_download", False),
            backend_attempted=kwargs.get("preferred_backend", "open_clip"),
        ),
    )


# --------------------------------------------------------------------------- #
# Alias-aware label matching
# --------------------------------------------------------------------------- #
def test_build_label_info_alias_and_distractor_indices():
    allowed = ["backpack", "bag", "school bag", "adidas", "logo"]
    info = build_label_info(allowed, "backpack", ["bag", "school bag", "adidas", "logo"])
    assert info.label == 0
    # Only the non-distractor label (the strict target) is alias-aware target.
    assert info.alias_indices == frozenset({0})
    assert info.distractor_indices == frozenset({1, 2, 3, 4})


def test_build_label_info_treats_extra_synonym_as_alias():
    # A synonym that is NOT a flagged distractor counts as an alias-aware target.
    allowed = ["sofa", "couch", "text", "logo"]
    info = build_label_info(allowed, "sofa", ["text", "logo"])
    assert info.label == 0
    assert info.alias_indices == frozenset({0, 1})  # sofa + couch
    assert info.distractor_indices == frozenset({2, 3})


def test_alias_aware_matching_recovers_synonym_at_top1():
    allowed = ["sofa", "couch", "text", "logo"]
    info = build_label_info(allowed, "sofa", ["text", "logo"])
    # Model predicts "couch" (a non-distractor synonym) at top-1.
    probs = np.array([0.30, 0.45, 0.15, 0.10])
    row = directional_metrics_row(probs, probs, info)
    assert row["strict_top1_after"] is False  # exact "sofa" not top-1
    assert row["alias_top1_after"] is True  # alias "couch" is top-1


# --------------------------------------------------------------------------- #
# Target rank computation
# --------------------------------------------------------------------------- #
def test_label_rank_basic_and_ties():
    probs = [0.5, 0.3, 0.2]
    assert label_rank(probs, 0) == 1
    assert label_rank(probs, 1) == 2
    assert label_rank(probs, 2) == 3
    # Tie: queried label ranked after the strictly-greater ones.
    tie = [0.4, 0.4, 0.2]
    assert label_rank(tie, 0) == 1
    assert label_rank(tie, 1) == 1


def test_best_rank_picks_minimum():
    probs = [0.1, 0.6, 0.3]
    assert best_rank(probs, [0, 2]) == 2  # idx2 ranks 2nd
    assert best_rank(probs, []) == len(probs) + 1


def test_target_rank_before_after_in_row():
    allowed = ["cat", "dog", "text"]
    info = build_label_info(allowed, "cat", ["text"])
    before = np.array([0.1, 0.2, 0.7])  # cat rank 3
    after = np.array([0.6, 0.2, 0.2])  # cat rank 1
    row = directional_metrics_row(before, after, info)
    assert row["target_rank_before"] == 3
    assert row["target_rank_after"] == 1
    assert row["target_rank_gain"] == 2


# --------------------------------------------------------------------------- #
# Target probability improvement
# --------------------------------------------------------------------------- #
def test_target_probability_improvement_flag_and_gain():
    allowed = ["cat", "text"]
    info = build_label_info(allowed, "cat", ["text"])
    before = np.array([0.2, 0.8])
    after = np.array([0.55, 0.45])
    row = directional_metrics_row(before, after, info)
    assert row["moved_toward_target"] is True
    assert row["target_prob_gain"] == pytest.approx(0.35)
    # No improvement when probability does not rise.
    flat = directional_metrics_row(before, before, info)
    assert flat["moved_toward_target"] is False
    assert flat["target_prob_gain"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Text-distractor probability decrease
# --------------------------------------------------------------------------- #
def test_text_distractor_probability_decrease():
    allowed = ["cat", "logo", "text"]
    info = build_label_info(allowed, "cat", ["logo", "text"])
    before = np.array([0.1, 0.6, 0.3])  # strongest distractor 0.6
    after = np.array([0.5, 0.3, 0.2])  # strongest distractor 0.3
    row = directional_metrics_row(before, after, info)
    assert row["text_distractor_prob_before"] == pytest.approx(0.6)
    assert row["text_distractor_prob_after"] == pytest.approx(0.3)
    assert row["text_distractor_prob_decrease"] == pytest.approx(0.3)
    assert row["moved_away_from_text"] is True


def test_text_distractor_metrics_nan_without_distractors():
    allowed = ["cat", "dog"]
    info = build_label_info(allowed, "cat", [])
    row = directional_metrics_row(np.array([0.4, 0.6]), np.array([0.6, 0.4]), info)
    assert np.isnan(row["text_distractor_prob_before"])
    assert np.isnan(row["text_distractor_prob_decrease"])
    assert row["moved_away_from_text"] is False


# --------------------------------------------------------------------------- #
# Aggregate directional metrics
# --------------------------------------------------------------------------- #
def test_aggregate_directional_metrics_rates():
    allowed = ["cat", "text"]
    info = build_label_info(allowed, "cat", ["text"])
    rows = [
        directional_metrics_row(np.array([0.2, 0.8]), np.array([0.7, 0.3]), info),  # improves, strict repair
        directional_metrics_row(np.array([0.2, 0.8]), np.array([0.3, 0.7]), info),  # improves, no strict repair
        directional_metrics_row(np.array([0.2, 0.8]), np.array([0.2, 0.8]), info),  # no change
    ]
    agg = aggregate_directional_metrics(rows)
    assert agg["n"] == 3
    assert agg["strict_top1_repair_accuracy"] == pytest.approx(1 / 3)
    assert agg["target_prob_improvement_rate"] == pytest.approx(2 / 3)
    assert agg["moved_away_from_text_rate"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# Proposal-selection geometry
# --------------------------------------------------------------------------- #
def test_selection_geometry_and_buckets():
    selected = (10, 10, 30, 30)
    text_boxes = [(10, 10, 30, 30)]  # exact overlap -> coverage 1.0
    object_boxes = [(100, 100, 150, 150)]
    geom = selection_geometry(selected, text_boxes, object_boxes)
    assert geom["overlaps_text_box"] is True
    assert geom["overlaps_object_box"] is False
    assert geom["text_coverage"] == pytest.approx(1.0)
    assert geom["closer_to"] == "text"
    assert geom["text_overlap_bucket"] == "coverage_ge_0.5"


def test_text_overlap_bucket_thresholds():
    assert text_overlap_bucket(0.0, 0.0) == "no_overlap"
    assert text_overlap_bucket(0.1, 0.05) == "partial_overlap"
    assert text_overlap_bucket(0.35, 0.0) == "iou_or_coverage_ge_0.3"
    assert text_overlap_bucket(0.6, 0.0) == "coverage_ge_0.5"


# --------------------------------------------------------------------------- #
# Categorization
# --------------------------------------------------------------------------- #
def test_categorize_cic_strict_repaired_takes_priority():
    allowed = ["cat", "text"]
    info = build_label_info(allowed, "cat", ["text"])
    cic = directional_metrics_row(np.array([0.2, 0.8]), np.array([0.7, 0.3]), info)
    oracle = directional_metrics_row(np.array([0.2, 0.8]), np.array([0.7, 0.3]), info)
    geom = selection_geometry((0, 0, 5, 5), [(0, 0, 5, 5)], [])
    primary, flags = categorize_failure(cic_row=cic, oracle_row=oracle, geometry=geom)
    assert primary == "cic_strict_repaired"
    assert flags["cic_strict_repaired"] is True


def test_categorize_hard_when_no_movement():
    allowed = ["cat", "text"]
    info = build_label_info(allowed, "cat", ["text"])
    flat = directional_metrics_row(np.array([0.2, 0.8]), np.array([0.2, 0.8]), info)
    geom = selection_geometry((100, 100, 110, 110), [], [])
    primary, flags = categorize_failure(cic_row=flat, oracle_row=flat, geometry=geom)
    assert primary == "hard_no_clear_repair"


# --------------------------------------------------------------------------- #
# Directional-evidence flag
# --------------------------------------------------------------------------- #
def _dir_kwargs(**overrides):
    kwargs = dict(
        n_verified_failures=29,
        oracle_target_prob_improvement_rate=0.95,
        cic_target_prob_improvement_rate=0.60,
        random_target_prob_improvement_rate=0.40,
        cic_selected_text_overlap_rate=0.72,
        no_oracle_leakage=True,
    )
    kwargs.update(overrides)
    return kwargs


def test_directional_evidence_flag_passes_when_all_met():
    ok, reasons = evaluate_directional_evidence(**_dir_kwargs())
    assert ok is True
    assert reasons == []


def test_directional_evidence_fails_on_low_oracle_improvement():
    ok, reasons = evaluate_directional_evidence(**_dir_kwargs(oracle_target_prob_improvement_rate=0.5))
    assert ok is False
    assert any("oracle" in r for r in reasons)


def test_directional_evidence_fails_when_cic_does_not_beat_random():
    ok, reasons = evaluate_directional_evidence(
        **_dir_kwargs(cic_target_prob_improvement_rate=0.45, random_target_prob_improvement_rate=0.40)
    )
    assert ok is False
    assert any("matched random" in r for r in reasons)


def test_directional_evidence_fails_on_low_text_overlap():
    ok, reasons = evaluate_directional_evidence(**_dir_kwargs(cic_selected_text_overlap_rate=0.4))
    assert ok is False
    assert any("text overlap" in r for r in reasons)


def test_directional_evidence_fails_on_leakage():
    ok, reasons = evaluate_directional_evidence(**_dir_kwargs(no_oracle_leakage=False))
    assert ok is False
    assert any("leakage" in r for r in reasons)


def test_directional_evidence_fails_on_too_few_failures():
    ok, reasons = evaluate_directional_evidence(**_dir_kwargs(n_verified_failures=10))
    assert ok is False
    assert any("verified failures" in r for r in reasons)


# --------------------------------------------------------------------------- #
# Strict gate remains strict / no-oracle-leakage unchanged
# --------------------------------------------------------------------------- #
def test_strict_support_gate_remains_strict():
    # The error-analysis module must NOT redefine or loosen the strict gate; it
    # reuses the original one, which still fails on the real (gap=0.103) numbers.
    from causal_reliability.experiments.run_natural_text_verified_failure_eval import evaluate_natural_text_gate

    supported, reasons = evaluate_natural_text_gate(
        backend="open_clip",
        pretrained=True,
        fake_backend=False,
        n_verified_failures=29,
        oracle_repair_or_improve_rate=0.966,
        cic_top1_repair_accuracy=0.241,
        matched_random_repair_accuracy=0.138,
        content_preservation_drop=0.75,
        no_oracle_leakage=True,
        open_world_claim_allowed=False,
    )
    assert supported is False
    assert any("matched random" in r for r in reasons)


def test_no_oracle_leakage_unchanged():
    from causal_reliability.experiments.run_natural_text_error_analysis import scoring_is_leakage_free

    assert scoring_is_leakage_free() is True


# --------------------------------------------------------------------------- #
# End-to-end run (fake-CLIP-patched) keeps gates strict and adds diagnostics
# --------------------------------------------------------------------------- #
def test_end_to_end_error_analysis_outputs(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_natural_text_error_analysis as mod

    _patch_real_clip(monkeypatch, mod)
    root = _make_fixture(tmp_path, n=10)
    cfg = _base_cfg(tmp_path, root)
    cfg["min_verified_failures"] = 1
    outputs = mod.run(cfg)

    for key in (
        "directional_metrics",
        "key_numbers",
        "error_analysis",
        "natural_text_error_analysis",
        "directional_summary",
    ):
        assert Path(outputs[key]).exists(), key

    key_numbers = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    # Gates must remain strict / false regardless of diagnostics.
    assert key_numbers["natural_text_supported"] is False
    assert key_numbers["open_proposal_supported"] is False
    assert key_numbers["open_world_claim_allowed"] is False
    assert "natural_text_directional_evidence" in key_numbers
    assert key_numbers["summary_statement"].startswith("Strict support gate remains failed")


def test_error_analysis_does_not_touch_strict_artifacts(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_natural_text_error_analysis as mod

    _patch_real_clip(monkeypatch, mod)
    root = _make_fixture(tmp_path, n=8)
    cfg = _base_cfg(tmp_path, root)
    cfg["min_verified_failures"] = 1
    outputs = mod.run(cfg)
    out_root = Path(tmp_path / "results" / "natural_text_verified_failure_eval")
    # New files only; strict verified_failure_* artifacts are never written here.
    assert not (out_root / "verified_failure_key_numbers.json").exists()
    assert (out_root / "natural_text_directional_key_numbers.json").exists()
