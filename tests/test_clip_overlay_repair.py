from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_reliability.real_models.clip_zero_shot import ClipStatus


def test_clip_repair_fake_backend_cannot_be_headline_eligible(tmp_path: Path):
    from causal_reliability.experiments.run_clip_overlay_repair import run

    outputs = run({"results_dir": str(tmp_path), "data": {"image_size": 64, "validation_n_per_class": 1, "test_n_per_class": 1}, "model": {"backend": "fake"}})
    metrics = pd.read_csv(outputs["metrics"])
    summary = Path(outputs["summary"]).read_text(encoding="utf-8")

    assert metrics["evidence_status"].iloc[0] == "unavailable"
    assert not bool(metrics["headline_eligible"].iloc[0])
    assert not bool(metrics["include_in_final_headline"].iloc[0])
    assert "No fake repair metrics were generated" in summary


def test_clip_repair_pretrained_false_writes_unavailable(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_clip_overlay_repair as mod

    monkeypatch.setattr(
        mod,
        "check_clip_available",
        lambda **kwargs: ClipStatus(
            available=True,
            backend="open_clip",
            model_name="ViT-B-32",
            pretrained=False,
            downloads_allowed=kwargs["allow_download"],
            backend_attempted=kwargs["preferred_backend"],
            error_message="forced non-pretrained",
        ),
    )
    outputs = mod.run(
        {
            "results_dir": str(tmp_path),
            "data": {"image_size": 64, "validation_n_per_class": 1, "test_n_per_class": 1},
            "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False},
        }
    )

    metrics = pd.read_csv(outputs["metrics"])
    summary = Path(outputs["summary"]).read_text(encoding="utf-8")
    assert metrics["evidence_status"].iloc[0] == "unavailable"
    assert not bool(metrics["headline_eligible"].iloc[0])
    assert "CLIP unavailable" in summary


def test_clip_repair_real_path_records_splits_selection_and_certificate_fields(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_clip_overlay_repair as mod

    class TinyClassifier:
        def __init__(self, status, class_names, prompts=None, device="cpu"):
            self.status = status
            self.class_names = class_names
            self.prompts = prompts or []

        def predict(self, images):
            means = images.mean(dim=(1, 2, 3)).numpy()
            probs = np.full((len(means), len(self.class_names)), 0.05, dtype=np.float32)
            preds = (np.floor(means * 1000).astype(int) % len(self.class_names))
            probs[np.arange(len(means)), preds] = 0.85
            probs = probs / probs.sum(axis=1, keepdims=True)
            import torch

            return {"probabilities": torch.from_numpy(probs)}

    monkeypatch.setattr(
        mod,
        "check_clip_available",
        lambda **kwargs: ClipStatus(
            available=True,
            backend="open_clip",
            model_name="ViT-B-32",
            pretrained_tag="laion2b_s34b_b79k",
            pretrained=True,
            downloads_allowed=kwargs["allow_download"],
            backend_attempted=kwargs["preferred_backend"],
        ),
    )
    monkeypatch.setattr(mod, "ClipZeroShotClassifier", TinyClassifier)

    outputs = mod.run(
        {
            "results_dir": str(tmp_path),
            "confidence_threshold": 0.8,
            "data": {
                "image_size": 64,
                "validation_n_per_class": 1,
                "test_n_per_class": 1,
                "regimes": ["aligned_overlay", "misleading_overlay", "neutral_overlay", "no_overlay"],
            },
            "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
        }
    )
    out = tmp_path / "clip_overlay_repair"
    metrics = pd.read_csv(outputs["metrics"])
    certs = pd.read_csv(outputs["certificates"])
    selected = (out / "selected_clip_repair_config.json").read_text(encoding="utf-8")

    assert (out / "clip_repair_validation_sweep.csv").exists()
    assert '"source_split": "validation"' in selected
    assert set(certs["split"]) == {"test"}
    assert {"overlay_bbox", "selected_intervention", "model_backend", "pretrained_loaded"}.issubset(certs.columns)
    assert metrics["evidence_status"].eq("pretrained CLIP repair evidence").all()
    assert not metrics["headline_eligible"].any()
    assert metrics["misleading_overlay_n_examples"].max() < 30


def test_final_report_does_not_claim_clip_repair_when_not_headline_eligible(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    out = tmp_path / "clip_overlay_repair"
    out.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "method": "cic_overlay_neutralized_prediction",
                "evidence_status": "pretrained CLIP repair evidence",
                "headline_eligible": False,
                "include_in_final_headline": False,
                "misleading_overlay_accuracy_before": 0.2,
                "misleading_overlay_accuracy_after": 0.8,
            }
        ]
    ).to_csv(out / "clip_overlay_repair_metrics.csv", index=False)

    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    assert "Pretrained CLIP shortcut repair was attempted but is not headline evidence" in report
    assert "CIC improves pretrained CLIP" not in report


def test_docs_do_not_claim_general_robustness_or_deployment_safety():
    for path in [Path("README.md"), Path("docs/cic_audit_demo.md"), Path("docs/when_cic_fails.md")]:
        text = path.read_text(encoding="utf-8").lower()
        assert "proves general robustness" not in text
        assert "guarantees general robustness" not in text
        assert "proves deployment safety" not in text
        assert "guarantees deployment safety" not in text
