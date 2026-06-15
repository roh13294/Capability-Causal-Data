from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from causal_reliability.real_models.clip_zero_shot import ClipStatus


def test_visual_decoy_scorer_signature_has_no_oracle_parameters():
    from causal_reliability.discovery.cic_region_scoring import score_region_candidates

    params = set(inspect.signature(score_region_candidates).parameters)
    forbidden = {"true_label", "label", "decoy_bbox", "decoy_label", "shortcut_identity", "test_correctness"}
    assert not (params & forbidden)


def test_visual_decoy_dataset_has_no_text_and_known_decoy_region():
    from causal_reliability.data.clip_visual_decoy_shortcuts import make_visual_decoy_dataset

    bundle = make_visual_decoy_dataset(n_per_condition=4, size=64, split="test")
    misleading = [ex for ex in bundle.examples if ex["regime"] == "misleading_decoy"]
    assert misleading
    for ex in misleading:
        assert ex["decoy_label"] != ex["label"]  # competing class
        assert len(ex["decoy_bbox"]) == 4
        assert "overlay_text" not in ex and "shortcut" not in ex  # no written words anywhere


def test_visual_decoy_pilot_runs_and_writes_gates(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_visual_decoy_clip_pilot as mod

    class TinyClassifier:
        def __init__(self, status, class_names, prompts=None, device="cpu"):
            self.class_names = class_names

        def predict(self, images):
            means = images.mean(dim=(1, 2, 3)).numpy()
            probs = np.full((len(means), len(self.class_names)), 0.04, dtype=np.float32)
            preds = np.floor(means * 1000).astype(int) % len(self.class_names)
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
            "n_random_candidates": 2,
            "data": {
                "image_size": 64,
                "test_n_per_condition": 8,
                "validation_n_per_condition": 4,
                "regimes": ["no_decoy", "misleading_decoy"],
            },
            "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
        }
    )

    import json

    certs = pd.read_csv(outputs["certificates"])
    gates = json.loads(Path(outputs["gates"]).read_text(encoding="utf-8"))
    assert {"original_clip_prediction", "oracle_decoy_neutralization", "cic_top1_region_repair", "cic_clean_safe_repair"}.issubset(set(certs["method"]))
    assert "no_scorer_leakage" in gates["gates"]
    assert gates["gates"]["no_scorer_leakage"] is True
    # Oracle rows must be labeled as upper bound, never headline.
    assert bool(certs[certs["method"] == "oracle_decoy_neutralization"]["oracle_upper_bound"].astype(bool).all())


def test_visual_decoy_pilot_fake_backend_blocked(tmp_path: Path):
    from causal_reliability.experiments.run_visual_decoy_clip_pilot import run

    outputs = run({"results_dir": str(tmp_path), "data": {"image_size": 64, "test_n_per_condition": 4}, "model": {"backend": "fake"}})
    metrics = pd.read_csv(outputs["metrics"])
    assert metrics["evidence_status"].iloc[0] == "unavailable"
    assert not bool(metrics["headline_eligible"].iloc[0])
