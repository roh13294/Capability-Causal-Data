from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from causal_reliability.real_models.clip_zero_shot import ClipStatus


REPO_ROOT = Path(__file__).resolve().parents[1]
DIM = 24


# ---------------------------------------------------------------------------
# Deterministic fake CLIP so the full per-input pipeline (including non-oracle
# CIC region discovery) runs without downloading or invoking real OpenCLIP.
# ---------------------------------------------------------------------------
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
    # Embedding dominated by the overlay region (bottom strip) so that
    # neutralizing it changes the logits; the object (top) gives a stable cue.
    arr = images.detach().cpu().numpy()
    feats = []
    for img in arr:
        bottom = img[:, int(img.shape[1] * 0.6):, :].mean(axis=(1, 2))
        top = img[:, : int(img.shape[1] * 0.5), :].mean(axis=(1, 2))
        vec = np.zeros(DIM, dtype=np.float64)
        vec[:3] = top * 3.0
        vec[3:6] = bottom * 2.0
        vec[6] = float(img.mean())
        feats.append(_unit(vec + 1e-3))
    return torch.from_numpy(np.stack(feats))


def _fake_encode_text(status, prompts, device="cpu"):
    feats = []
    for i, _ in enumerate(prompts):
        vec = np.zeros(DIM, dtype=np.float64)
        vec[i % DIM] = 1.0
        vec[(i + 5) % DIM] = 0.2
        feats.append(_unit(vec))
    return torch.from_numpy(np.stack(feats))


def _run_pipeline(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_per_input_class_balance_validation as mod
    from causal_reliability.real_models import clip_zero_shot

    monkeypatch.setattr(mod, "check_clip_available", _fake_status)
    monkeypatch.setattr(clip_zero_shot, "encode_images", _fake_encode_images)
    monkeypatch.setattr(clip_zero_shot, "encode_text_prompts", _fake_encode_text)
    cfg = {
        "results_dir": str(tmp_path),
        "frozen_policy_dir": str(tmp_path / "missing"),  # falls back to default policy
        "max_candidates": 48,
        "include_watermark": True,
        "data": {"image_size": 56, "text_n_per_class": 2, "watermark_n_per_class": 2, "watermark_benchmark_seed": 5151},
        "thresholds": {"class_balance_epsilon_b": 3.0},
        "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
    }
    return mod.run(cfg)


# ---------------------------------------------------------------------------
# PART 10: experiment writes summary, metrics, examples, key numbers, plots
# ---------------------------------------------------------------------------
def test_experiment_writes_outputs_and_required_keys(tmp_path: Path, monkeypatch):
    outputs = _run_pipeline(tmp_path, monkeypatch)
    for key in ["summary", "metrics", "examples", "key_numbers", "caption", "plot_png", "plot_pdf"]:
        assert Path(outputs[key]).exists(), key

    kn = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    # Required decision booleans / status (PART 6).
    assert "per_input_class_balance_supported_for_text" in kn
    assert "clip_theory_support_status" in kn
    assert kn["per_input_class_balance_supported_for_text"] in (True, False, "mixed")
    assert kn["clip_theory_support_status"] in {
        "CLIP-supported via per-input class-balance",
        "mixed; per-input class-balance partially supported",
        "conditional only; per-input class-balance not supported",
    }
    # Honest backend bookkeeping.
    assert kn["pretrained_clip_loaded"] is True
    assert kn["fake_backend"] is False

    metrics = pd.read_csv(outputs["metrics"])
    assert {"condition", "median_residual_to_clean", "margin_condition_satisfaction_rate", "repair_accuracy"}.issubset(metrics.columns)
    # The oracle and CIC neutralization conditions must be present.
    assert {"oracle", "cic_top1", "cic_top3_consensus"}.issubset(set(metrics["condition"]))

    examples = pd.read_csv(outputs["examples"])
    assert {"shift_std", "shift_range", "max_centered_shift", "residual_to_clean", "margin_condition_satisfied", "repair_success"}.issubset(examples.columns)


def test_fake_backend_is_not_theory_eligible(tmp_path: Path):
    from causal_reliability.experiments import run_per_input_class_balance_validation as mod

    outputs = mod.run(
        {
            "results_dir": str(tmp_path),
            "data": {"image_size": 48},
            "model": {"preferred_backend": "fake", "device": "cpu"},
        }
    )
    kn = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    assert kn["per_input_class_balance_supported_for_text"] is False
    assert kn["fake_backend"] is True
    assert kn["clip_theory_support_status"] == "conditional only; per-input class-balance not supported"


# ---------------------------------------------------------------------------
# Per-input balance metric: a class-independent shift must not change the argmax.
# ---------------------------------------------------------------------------
def test_class_independent_shift_preserves_argmax():
    from causal_reliability.experiments.run_per_input_class_balance_validation import per_input_balance

    clean = np.array([5.0, 3.0, 1.0, 0.0])
    shortcut = np.array([2.0, 6.0, 1.0, 0.0])  # class 1 wrong winner
    # Neutralized = clean shifted by a class-INDEPENDENT constant -> argmax stays class 0.
    neutralized = clean + 2.5
    out = per_input_balance(clean, shortcut, neutralized, label=0, epsilon_b=3.0)
    assert out["repair_success"] is True
    assert out["residual_to_clean"] < 1e-9
    assert out["class_balance_satisfied"] is True
    assert out["margin_condition_satisfied"] is True


# ---------------------------------------------------------------------------
# PART 7/8: docs/theory.md content
# ---------------------------------------------------------------------------
def _theory_text() -> str:
    path = REPO_ROOT / "docs" / "theory.md"
    assert path.exists(), "docs/theory.md must exist"
    return path.read_text(encoding="utf-8").lower()


def test_theory_includes_per_input_class_balance_condition():
    text = _theory_text()
    # The recovery condition is defined relative to the clean/causal logits
    # (residual-to-clean), not relative to the misleading input logits.
    assert "per-input residual-to-clean class-balance condition" in text
    # The weaker premise must be stated explicitly relative to global additivity.
    assert "weaker" in text
    assert (
        "max_y |\\rho_y(x) - \\bar{\\rho}(x)| \\leq \\epsilon_b" in text
        or "max_y |rho_y(x) - mean_y rho(x)| <= epsilon_b" in text
    )


def test_theory_includes_per_input_recovery_corollary():
    text = _theory_text()
    assert "per-input residual-to-clean recovery" in text
    assert (
        "m_{\\mathrm{clean}}(x) > 2\\epsilon_b" in text
        or "m_clean(x) > 2\\epsilon_b" in text
        or "m_clean(x) > 2*epsilon_b" in text
    )


def test_theory_includes_object_entangled_shortcut_effects():
    text = _theory_text()
    assert "object-entangled" in text
    assert "object-entangled typographic shortcut effects" in text


# ---------------------------------------------------------------------------
# PART 8/10: final report theory framing is gated on clip_theory_support_status
# ---------------------------------------------------------------------------
def _hard_metrics_csv(hard_dir: Path) -> None:
    hard_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"method": "original_clip_prediction", "headline_eligible": True, "evidence_status": "pretrained CLIP hard multi-decoy non-oracle repair evidence", "pretrained_loaded": True, "hard_multi_decoy_misleading_accuracy_before": 0.25},
            {"method": "nonoracle_cic_top1_repair", "headline_eligible": True, "pretrained_loaded": True, "hard_multi_decoy_misleading_accuracy_after": 0.75},
        ]
    ).to_csv(hard_dir / "hard_multidecoy_repair_metrics.csv", index=False)


def _write_per_input_kn(root: Path, supported, status_str: str) -> None:
    d = root / "per_input_class_balance"
    d.mkdir(parents=True, exist_ok=True)
    (d / "per_input_class_balance_key_numbers.json").write_text(
        json.dumps(
            {
                "per_input_class_balance_supported_for_text": supported,
                "clip_theory_support_status": status_str,
                "pretrained_clip_loaded": True,
                "fake_backend": False,
                "any_more_balanced_than_random": True,
                "random_median_residual_to_clean": 5.6,
                "oracle_median_residual_to_clean": 2.3,
                "cic_top1_median_residual_to_clean": 3.5,
                "oracle_repair_accuracy": 1.0,
                "cic_top1_repair_accuracy": 0.85,
                "object_entanglement_statement": "OpenCLIP's typographic shortcut effect is not a single global additive bias direction. The shift induced by overlay text is object-entangled: it contains a real shortcut component, but its direction varies substantially with the underlying object.",
            }
        ),
        encoding="utf-8",
    )


def test_final_report_claims_per_input_support_only_when_status_allows(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    _hard_metrics_csv(tmp_path / "hard_multidecoy_clip_repair")
    _write_per_input_kn(tmp_path, True, "CLIP-supported via per-input class-balance")
    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    kn = json.loads((tmp_path / "final_report" / "final_key_numbers.json").read_text())

    assert kn["per_input_class_balance_available"] is True
    assert kn["clip_theory_support_status"] == "CLIP-supported via per-input class-balance"
    assert "the weaker per-input class-balance condition was supported" in report
    assert "plausible mechanism for the OpenCLIP text-overlay repair result" in report
    # Must NOT use the not-supported wording.
    assert "Neither global additivity nor per-input class-balance was supported" not in report


def test_final_report_uses_mixed_wording_when_mixed(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    _hard_metrics_csv(tmp_path / "hard_multidecoy_clip_repair")
    _write_per_input_kn(tmp_path, "mixed", "mixed; per-input class-balance partially supported")
    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    assert "per-input class-balance was partially supported" in report
    assert "plausible mechanism for the OpenCLIP text-overlay repair result" not in report


def test_final_report_does_not_claim_support_when_not_supported(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    _hard_metrics_csv(tmp_path / "hard_multidecoy_clip_repair")
    _write_per_input_kn(tmp_path, False, "conditional only; per-input class-balance not supported")
    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    assert "Neither global additivity nor per-input class-balance was supported" in report
    assert "plausible mechanism for the OpenCLIP text-overlay repair result" not in report


def test_final_report_includes_caveats_and_object_entanglement(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    _hard_metrics_csv(tmp_path / "hard_multidecoy_clip_repair")
    _write_per_input_kn(tmp_path, True, "CLIP-supported via per-input class-balance")
    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    # Caveats against open-world discovery / exact localization / general robustness.
    assert "does not claim open-world shortcut discovery, exact localization, or general robustness" in report
    # Object-entanglement finding must be present.
    assert "object-entangled" in report.lower()
    assert "Object-entanglement finding" in report
