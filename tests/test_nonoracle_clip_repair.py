from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from causal_reliability.real_models.clip_zero_shot import ClipStatus


def test_nonoracle_scoring_signature_has_no_oracle_parameters():
    from causal_reliability.discovery.cic_region_scoring import score_region_candidates

    params = set(inspect.signature(score_region_candidates).parameters)
    forbidden = {"true_label", "label", "overlay_bbox", "overlay_text", "shortcut_identity", "test_correctness"}
    assert not (params & forbidden)


def test_nonoracle_candidate_rankings_written_and_oracle_labeled(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_nonoracle_clip_repair as mod

    class TinyClassifier:
        def __init__(self, status, class_names, prompts=None, device="cpu"):
            self.status = status
            self.class_names = class_names

        def predict(self, images):
            means = images.mean(dim=(1, 2, 3)).numpy()
            probs = np.full((len(means), len(self.class_names)), 0.04, dtype=np.float32)
            preds = (np.floor(means * 1000).astype(int) % len(self.class_names))
            probs[np.arange(len(means)), preds] = 0.88
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
            "max_candidates": 16,
            "augmentation_views": 2,
            "data": {"image_size": 64, "test_n_per_class": 1, "regimes": ["aligned_overlay", "misleading_overlay", "no_overlay"]},
            "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
        }
    )

    rankings = pd.read_csv(outputs["rankings"])
    certs = pd.read_csv(outputs["certificates"])
    metrics = pd.read_csv(outputs["metrics"])

    assert len(rankings) > 0
    assert {"candidate_id", "bbox", "proposal_type", "rank", "score", "overlay_iou"}.issubset(rankings.columns)
    oracle = certs[certs["method"] == "oracle_overlay_neutralization"]
    assert len(oracle) > 0
    assert oracle["oracle_upper_bound"].astype(bool).all()
    assert "oracle upper bound" in set(oracle["selected_proposal_type"])
    assert not metrics["headline_eligible"].any()


def test_nonoracle_fake_backend_cannot_be_headline_eligible(tmp_path: Path):
    from causal_reliability.experiments.run_nonoracle_clip_repair import run

    outputs = run({"results_dir": str(tmp_path), "data": {"image_size": 64, "test_n_per_class": 1}, "model": {"backend": "fake"}})
    metrics = pd.read_csv(outputs["metrics"])
    summary = Path(outputs["summary"]).read_text(encoding="utf-8")

    assert metrics["evidence_status"].iloc[0] == "unavailable"
    assert not bool(metrics["headline_eligible"].iloc[0])
    assert "no fake headline evidence" in summary.lower()


def test_final_report_does_not_headline_oracle_as_discovery(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    oracle_dir = tmp_path / "clip_overlay_repair"
    oracle_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "method": "cic_overlay_neutralized_prediction",
                "evidence_status": "pretrained CLIP repair evidence",
                "headline_eligible": True,
                "include_in_final_headline": True,
                "misleading_overlay_accuracy_before": 0.2,
                "misleading_overlay_accuracy_after": 0.9,
            }
        ]
    ).to_csv(oracle_dir / "clip_overlay_repair_metrics.csv", index=False)
    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")

    assert "Oracle overlay repair should not be treated as evidence of automatic shortcut discovery" in report
    assert "known-overlay neutralization" in report


def test_related_work_and_docs_claim_boundaries():
    related = Path("docs/related_work.md").read_text(encoding="utf-8").lower()
    assert "shortcut learning" in related
    assert "counterfactual invariance" in related
    assert "typographic" in related

    for path in [Path("README.md"), Path("docs/cic_audit_demo.md"), Path("docs/when_cic_fails.md"), Path("docs/related_work.md")]:
        text = path.read_text(encoding="utf-8").lower()
        assert "open-world discovery is solved" not in text
        assert "clip repair is general robustness" not in text
