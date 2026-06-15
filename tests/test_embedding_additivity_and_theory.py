from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from causal_reliability.real_models.clip_zero_shot import ClipStatus


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Deterministic fake CLIP embeddings so the full metric pipeline runs without
# downloading or invoking real OpenCLIP. The fake makes the shortcut value the
# dominant embedding direction so the pipeline exercises the "supported" branch.
# ---------------------------------------------------------------------------
DIM = 24


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


def _unit(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-9 else vec


def _fake_encode_images(status, images, device="cpu"):
    # Build an embedding dominated by a per-shortcut-value direction so that
    # within-shortcut clustering is high and neutralization is close to clean.
    arr = images.detach().cpu().numpy()
    feats = []
    for img in arr:
        # Mean color of the bottom strip (where overlays live) as a coarse "shortcut" cue.
        bottom = img[:, int(img.shape[1] * 0.7):, :].mean(axis=(1, 2))
        top = img[:, : int(img.shape[1] * 0.5), :].mean(axis=(1, 2))
        vec = np.zeros(DIM, dtype=np.float64)
        vec[:3] = bottom * 5.0  # dominant shortcut-ish channel
        vec[3:6] = top
        vec[6] = float(img.mean())
        feats.append(_unit(vec + 1e-3))
    return torch.from_numpy(np.stack(feats))


def _fake_encode_text(status, prompts, device="cpu"):
    feats = []
    for i, _ in enumerate(prompts):
        vec = np.zeros(DIM, dtype=np.float64)
        vec[i % DIM] = 1.0
        vec[(i + 7) % DIM] = 0.3
        feats.append(_unit(vec))
    return torch.from_numpy(np.stack(feats))


def _run_full_pipeline(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_embedding_additivity_validation as mod

    monkeypatch.setattr(mod, "check_clip_available", _fake_status)
    monkeypatch.setattr(mod, "encode_images", _fake_encode_images)
    monkeypatch.setattr(mod, "encode_text_prompts", _fake_encode_text)
    cfg = {
        "results_dir": str(tmp_path),
        "frozen_policy_dir": str(tmp_path / "missing"),  # falls back to default policy
        "data": {"image_size": 48, "text_n_per_class": 3, "watermark_n_per_class": 3, "watermark_benchmark_seed": 5151},
        "thresholds": {"n_label_shuffles": 3},
        "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
    }
    return mod.run(cfg)


# ---------------------------------------------------------------------------
# PART 9: experiment writes summary, metrics, key numbers with required flags
# ---------------------------------------------------------------------------
def test_experiment_writes_summary_metrics_key_numbers(tmp_path: Path, monkeypatch):
    outputs = _run_full_pipeline(tmp_path, monkeypatch)
    for key in ["summary", "metrics", "examples", "key_numbers", "caption", "text_delta_plot_png", "watermark_delta_plot_png"]:
        assert Path(outputs[key]).exists(), key

    kn = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    assert "embedding_additivity_supported_for_text" in kn
    assert "embedding_additivity_supported_for_watermark" in kn
    assert isinstance(kn["embedding_additivity_supported_for_text"], bool)
    assert isinstance(kn["embedding_additivity_supported_for_watermark"], bool)
    # Theory-gating bookkeeping must be present and honest about the backend.
    assert kn["pretrained_clip_loaded"] is True
    assert kn["fake_backend"] is False
    assert kn["theorem_framing"] in {"empirically_supported_for_text_overlays", "conditional_theory_only"}

    metrics = pd.read_csv(outputs["metrics"])
    assert {"family", "within_shortcut_mean_pairwise_cosine", "within_object_mean_pairwise_cosine", "mean_neutralization_ratio"}.issubset(metrics.columns)
    assert set(metrics["family"]) == {"text_overlay", "watermark"}


def test_fake_backend_is_not_theory_eligible(tmp_path: Path):
    from causal_reliability.experiments import run_embedding_additivity_validation as mod

    outputs = mod.run(
        {
            "results_dir": str(tmp_path),
            "data": {"image_size": 48},
            "model": {"preferred_backend": "fake", "device": "cpu"},
        }
    )
    kn = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    assert kn["embedding_additivity_supported_for_text"] is False
    assert kn["embedding_additivity_supported_for_watermark"] is False
    assert kn["fake_backend"] is True


def test_supported_flag_requires_all_checks(tmp_path: Path, monkeypatch):
    """A fake whose deltas are object-dominated must NOT be marked supported."""
    from causal_reliability.experiments import run_embedding_additivity_validation as mod

    def object_dominated_encode(status, images, device="cpu"):
        # Encode purely from the global mean -> deltas carry no shortcut signal.
        arr = images.detach().cpu().numpy()
        feats = [_unit(np.r_[np.full(3, img.mean()), np.zeros(DIM - 3)] + 1e-3) for img in arr]
        return torch.from_numpy(np.stack(feats))

    monkeypatch.setattr(mod, "check_clip_available", _fake_status)
    monkeypatch.setattr(mod, "encode_images", object_dominated_encode)
    monkeypatch.setattr(mod, "encode_text_prompts", _fake_encode_text)
    outputs = mod.run(
        {
            "results_dir": str(tmp_path),
            "data": {"image_size": 48, "text_n_per_class": 3, "watermark_n_per_class": 3},
            "thresholds": {"n_label_shuffles": 3},
            "model": {"preferred_backend": "open_clip", "device": "cpu"},
        }
    )
    kn = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    assert kn["embedding_additivity_supported_for_text"] is False


# ---------------------------------------------------------------------------
# PART 7: docs/theory.md exists with the required theory content
# ---------------------------------------------------------------------------
def _theory_text() -> str:
    path = REPO_ROOT / "docs" / "theory.md"
    assert path.exists(), "docs/theory.md must exist"
    return path.read_text(encoding="utf-8").lower()


def test_theory_doc_exists():
    assert (REPO_ROOT / "docs" / "theory.md").exists()


def test_theory_includes_additive_channel_assumption():
    text = _theory_text()
    assert "additive-channel assumption" in text or "additive-channel" in text
    assert "phi_y(c, n) + psi_y(s) + xi_y(c, s, n)" in text


def test_theory_includes_approximate_recovery_with_margin_condition():
    text = _theory_text()
    assert "approximate recovery" in text
    assert "2 (epsilon_c + epsilon_s + epsilon_xi)" in text or "2(epsilon_c + epsilon_s + epsilon_xi)" in text


def test_theory_includes_consensus_class_balanced_sweep():
    text = _theory_text()
    assert "class-balanced sweep" in text
    assert "consensus" in text


def test_theory_includes_coarse_localization_not_exact_iou():
    text = _theory_text()
    assert "coarse" in text and "iou" in text
    assert "exact bounding-box" in text or "exact localization" in text


def test_theory_includes_no_open_world_caveat():
    text = _theory_text()
    assert "open-world shortcut discovery" in text
    assert "does not" in text


def test_theory_states_empirical_support_outcome():
    text = _theory_text()
    assert "embedding_additivity_supported_for_text" in text
    # The theory doc must explicitly state whether additivity was supported.
    assert "did not support" in text or "empirically supported" in text


# ---------------------------------------------------------------------------
# PART 8: final report theory section is conditional on the key number
# ---------------------------------------------------------------------------
def _hard_metrics_csv(hard_dir: Path) -> None:
    hard_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"method": "original_clip_prediction", "headline_eligible": True, "evidence_status": "pretrained CLIP hard multi-decoy non-oracle repair evidence", "pretrained_loaded": True, "hard_multi_decoy_misleading_accuracy_before": 0.25},
            {"method": "nonoracle_cic_top1_repair", "headline_eligible": True, "pretrained_loaded": True, "hard_multi_decoy_misleading_accuracy_after": 0.75},
        ]
    ).to_csv(hard_dir / "hard_multidecoy_repair_metrics.csv", index=False)


def _write_embedding_additivity_kn(root: Path, supported_text: bool, supported_wm: bool = False, weak: bool = True) -> None:
    d = root / "embedding_additivity"
    d.mkdir(parents=True, exist_ok=True)
    (d / "embedding_additivity_key_numbers.json").write_text(
        json.dumps(
            {
                "embedding_additivity_supported_for_text": supported_text,
                "embedding_additivity_supported_for_watermark": supported_wm,
                "theorem_framing": "empirically_supported_for_text_overlays" if supported_text else "conditional_theory_only",
                "pretrained_clip_loaded": True,
                "fake_backend": False,
                "text_within_shortcut_cosine": 0.76,
                "text_within_object_cosine": 0.85,
                "text_shuffled_cosine": 0.63,
                "text_mean_neutralization_ratio": 0.92,
                "text_logit_consistency_mae": 6e-15,
                "text_repair_success_rate": 1.0,
                "text_margin_condition_predicts_repair": True,
                "text_failed_reasons": ["shortcut_clustering_exceeds_object", "neutralization_damage_small"] if not supported_text else [],
                "watermark_within_shortcut_cosine": 0.76,
                "watermark_within_object_cosine": 0.92,
                "watermark_mean_shortcut_effect_norm": 0.88,
                "watermark_shortcut_channel_weak": weak,
            }
        ),
        encoding="utf-8",
    )


def test_final_report_does_not_claim_clip_theorem_unless_supported(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    _hard_metrics_csv(tmp_path / "hard_multidecoy_clip_repair")
    _write_embedding_additivity_kn(tmp_path, supported_text=False)
    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    kn = json.loads((tmp_path / "final_report" / "final_key_numbers.json").read_text())

    assert kn["embedding_additivity_available"] is True
    assert kn["embedding_additivity_supported_for_text"] is False
    assert "Theory and Mechanism Validation" in report
    # When NOT supported, the report must use the conditional wording and must NOT
    # claim the theorem plausibly explains the OpenCLIP text-repair result.
    assert "did not support applying it directly to the current OpenCLIP text benchmark" in report
    assert "plausibly explains the OpenCLIP text-repair result" not in report


def test_final_report_claims_clip_theorem_only_when_supported(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    _hard_metrics_csv(tmp_path / "hard_multidecoy_clip_repair")
    _write_embedding_additivity_kn(tmp_path, supported_text=True)
    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    kn = json.loads((tmp_path / "final_report" / "final_key_numbers.json").read_text())

    assert kn["embedding_additivity_supported_for_text"] is True
    assert "plausibly explains the OpenCLIP text-repair result" in report


def test_final_report_reports_watermark_weak_channel(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    _hard_metrics_csv(tmp_path / "hard_multidecoy_clip_repair")
    _write_embedding_additivity_kn(tmp_path, supported_text=False, supported_wm=False, weak=True)
    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    assert "weak or flat shortcut channel" in report


def test_final_report_no_general_robustness_or_exact_localization_claim(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    _hard_metrics_csv(tmp_path / "hard_multidecoy_clip_repair")
    _write_embedding_additivity_kn(tmp_path, supported_text=False)
    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    # The theory section must explicitly disclaim general robustness and exact localization.
    assert "does not claim open-world shortcut discovery, exact localization, or general robustness" in report
