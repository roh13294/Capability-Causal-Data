from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_reliability.real_models.clip_zero_shot import ClipStatus


def test_multidecoy_generation_has_one_harmful_and_multiple_decoys():
    from causal_reliability.experiments.run_multidecoy_clip_repair import render_multidecoy_image

    image, meta = render_multidecoy_image(0, "multi_decoy_misleading", 0, size=96, n_text_boxes=5)

    assert image.shape == (96, 96, 3)
    assert meta["true_label"] == "circle"
    assert meta["harmful_text"] in {"square", "triangle", "star"}
    assert len(meta["harmful_bbox"]) == 4
    assert len(meta["decoy_bboxes"]) == 4
    assert len(meta["all_text_boxes"]) == 5
    assert sum(box["role"] == "harmful" for box in meta["all_text_boxes"]) == 1


def test_text_box_component_proposals_recover_multiple_boxes():
    from causal_reliability.discovery.region_proposals import generate_region_proposals
    from causal_reliability.experiments.run_multidecoy_clip_repair import render_multidecoy_image

    image, _ = render_multidecoy_image(1, "multi_decoy_misleading", 1, size=128, n_text_boxes=5)
    proposals = generate_region_proposals(image, seed=0, max_candidates=80)
    boxes = [p for p in proposals if p.proposal_type == "text_box_component"]

    assert len(boxes) >= 3
    assert len({p.bbox for p in boxes}) >= 3


def test_multidecoy_runner_writes_requested_outputs(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_multidecoy_clip_repair as mod

    class TinyClassifier:
        def __init__(self, status, class_names, prompts=None, device="cpu"):
            self.status = status
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
            "max_candidates": 20,
            "augmentation_views": 2,
            "random_draws": 2,
            "data": {
                "image_size": 64,
                "n_text_boxes": 4,
                "test_n_per_class": 1,
                "validation_n_per_class": 1,
                "regimes": ["multi_decoy_misleading", "multi_decoy_aligned", "multi_decoy_neutral", "no_overlay"],
            },
            "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
        }
    )

    for key in ["metrics", "certificates", "rankings", "summary", "selected_policy", "validation_policy_sweep"]:
        assert Path(outputs[key]).exists()

    rankings = pd.read_csv(outputs["rankings"])
    certs = pd.read_csv(outputs["certificates"])
    metrics = pd.read_csv(outputs["metrics"])

    assert {"candidate_id", "bbox", "proposal_type", "rank", "harmful_iou", "decoy_iou", "object_iou"}.issubset(rankings.columns)
    assert {"random_matched_text_region_repair", "random_nontext_patch_repair", "largest_text_region_repair", "highest_textness_region_repair"}.issubset(set(certs["method"]))
    oracle = certs[certs["method"] == "oracle_harmful_text_neutralization"]
    assert len(oracle) > 0
    assert oracle["oracle_upper_bound"].astype(bool).all()
    assert "oracle upper bound" in set(oracle["selected_proposal_type"])
    assert {"harmful_top1_iou_0_3", "clean_accuracy_drop", "coverage", "n_non_abstained"}.issubset(metrics.columns)


def test_multidecoy_fake_backend_unavailable(tmp_path: Path):
    from causal_reliability.experiments.run_multidecoy_clip_repair import run

    outputs = run({"results_dir": str(tmp_path), "data": {"image_size": 64, "test_n_per_class": 1}, "model": {"backend": "fake"}})
    metrics = pd.read_csv(outputs["metrics"])
    summary = Path(outputs["summary"]).read_text(encoding="utf-8").lower()

    assert metrics["evidence_status"].iloc[0] == "unavailable"
    assert not bool(metrics["headline_eligible"].iloc[0])
    assert "no fake headline evidence" in summary


def test_prediction_cache_batches_without_changing_outputs():
    from PIL import Image
    import torch

    from causal_reliability.experiments.run_multidecoy_clip_repair import _PredictionCache
    from causal_reliability.experiments.run_nonoracle_clip_repair import _predict_pil

    class CountingModel:
        def __init__(self):
            self.calls = 0

        def predict(self, images):
            self.calls += 1
            means = images.mean(dim=(1, 2, 3)).numpy()
            probs = np.stack([1.0 - means, means], axis=1).astype(np.float32)
            probs = probs / probs.sum(axis=1, keepdims=True)
            return {"probabilities": torch.from_numpy(probs)}

    model = CountingModel()
    dark = Image.new("RGB", (8, 8), (0, 0, 0))
    bright = Image.new("RGB", (8, 8), (255, 255, 255))
    images = [dark, bright, dark]

    direct = _predict_pil(model, images)
    cache = _PredictionCache(model)
    cached = cache(images)
    cached_again = cache(images)

    np.testing.assert_allclose(cached, direct)
    np.testing.assert_allclose(cached_again, direct)
    assert model.calls == 2
