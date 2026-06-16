from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from causal_reliability.data.natural_text_dataset import make_synthetic_natural_text_dataset
from causal_reliability.discovery.cic_region_scoring import score_region_candidates
from causal_reliability.discovery.open_region_proposals import (
    generate_open_region_proposals,
    has_non_ocr_family,
    proposal_family,
)
from causal_reliability.real_models.clip_zero_shot import ClipStatus


# --------------------------------------------------------------------------- #
# Fake CLIP classifier for end-to-end runs
# --------------------------------------------------------------------------- #
class _TinyClassifier:
    def __init__(self, status, class_names, prompts=None, device="cpu"):
        self.status = status
        self.class_names = class_names

    def predict(self, images):
        means = images.mean(dim=(1, 2, 3)).numpy()
        n = len(self.class_names)
        probs = np.full((len(means), n), 0.05, dtype=np.float32)
        preds = np.floor(means * 1000).astype(int) % n
        probs[np.arange(len(means)), preds] = 0.9
        probs = probs / probs.sum(axis=1, keepdims=True)
        return {"probabilities": torch.from_numpy(probs)}


def _patch_real_clip(monkeypatch, mod):
    monkeypatch.setattr(mod, "ClipZeroShotClassifier", _TinyClassifier)
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


def _base_cfg(tmp_path: Path, n_images: int = 10, **overrides) -> dict:
    cfg = {
        "results_dir": str(tmp_path),
        "max_candidates": 24,
        "min_images": 4,
        "n_example_visualizations": 3,
        "data": {"mode": "synthetic", "image_size": 64, "synthetic": {"n_images": n_images, "size": 64, "seed": 0}},
        "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
    }
    cfg.update(overrides)
    return cfg


# --------------------------------------------------------------------------- #
# Proposal generation
# --------------------------------------------------------------------------- #
def test_proposals_include_grid_without_ocr_boxes():
    bundle = make_synthetic_natural_text_dataset(2, 64, 0)
    image = bundle.examples[0]["image"]
    proposals = generate_open_region_proposals(image, text_boxes=None, seed=1, max_candidates=40)
    families = {proposal_family(p.proposal_type) for p in proposals}
    assert "grid_patch" in families
    assert "ocr_text_box" not in families  # no OCR boxes supplied
    assert has_non_ocr_family(proposals)


def test_ocr_boxes_are_optional_and_added_when_present():
    bundle = make_synthetic_natural_text_dataset(2, 64, 0)
    ex = bundle.examples[0]
    with_ocr = generate_open_region_proposals(ex["image"], text_boxes=ex["text_boxes"], seed=1, max_candidates=64)
    without_ocr = generate_open_region_proposals(ex["image"], text_boxes=None, seed=1, max_candidates=64)
    fams_with = {proposal_family(p.proposal_type) for p in with_ocr}
    fams_without = {proposal_family(p.proposal_type) for p in without_ocr}
    assert "ocr_text_box" in fams_with
    assert "ocr_text_box" not in fams_without
    # The method must not be merely "use the OCR box".
    assert has_non_ocr_family(with_ocr)


def test_scoring_and_proposal_signatures_have_no_oracle_parameters():
    from causal_reliability.experiments import run_natural_text_open_proposal_cic as mod

    score_params = set(inspect.signature(score_region_candidates).parameters)
    assert not (score_params & mod.FORBIDDEN_SCORING_PARAMS)

    proposal_params = set(inspect.signature(generate_open_region_proposals).parameters)
    assert not (proposal_params & mod.FORBIDDEN_PROPOSAL_PARAMS)
    # No true label / shortcut identity / correctness anywhere in the scoring path.
    for bad in ("true_label", "label", "human_label", "correctness", "shortcut_bbox", "ocr_text"):
        assert bad not in score_params
        assert bad not in proposal_params
    assert mod.scoring_is_leakage_free()


# --------------------------------------------------------------------------- #
# End-to-end run + artifacts
# --------------------------------------------------------------------------- #
def test_end_to_end_outputs_written_and_random_baseline_present(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_natural_text_open_proposal_cic as mod

    _patch_real_clip(monkeypatch, mod)
    outputs = mod.run(_base_cfg(tmp_path, n_images=10))

    for key in ("metrics", "key_numbers", "summary", "proposal_diagnostics", "certificates"):
        assert Path(outputs[key]).exists(), key

    metrics = pd.read_csv(outputs["metrics"])
    methods = set(metrics["method"])
    assert "cic_top1_repair" in methods
    assert "cic_top3_repair" in methods
    assert "matched_random_proposal_repair" in methods  # random baseline computed
    assert "largest_region_repair" in methods

    diagnostics = pd.read_csv(outputs["proposal_diagnostics"])
    assert len(diagnostics) > 0
    assert "grid_patch" in set(diagnostics["proposal_family"])

    # Before/after visualizations written.
    example_dir = Path(outputs["metrics"]).parent / "examples"
    assert any(p.name.endswith("_before_after.png") for p in example_dir.iterdir())


def test_open_world_claim_allowed_remains_false(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_natural_text_open_proposal_cic as mod

    _patch_real_clip(monkeypatch, mod)
    outputs = mod.run(_base_cfg(tmp_path, n_images=8))
    key_numbers = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    assert key_numbers["open_world_claim_allowed"] is False


def test_fake_backend_cannot_support_claim(tmp_path: Path):
    from causal_reliability.experiments.run_natural_text_open_proposal_cic import run

    outputs = run(
        {
            "results_dir": str(tmp_path),
            "data": {"mode": "synthetic", "synthetic": {"n_images": 8, "size": 64}},
            "model": {"backend": "fake"},
        }
    )
    key_numbers = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    assert key_numbers["open_proposal_supported"] is False
    assert key_numbers["open_world_claim_allowed"] is False


# --------------------------------------------------------------------------- #
# Gate logic (deterministic, via the pure function)
# --------------------------------------------------------------------------- #
def _gate(mod, **overrides):
    kwargs = dict(
        backend="open_clip",
        pretrained=True,
        n_images=40,
        cic_top1_repair_accuracy=0.85,
        matched_random_repair_accuracy=0.55,
        content_preservation_drop=0.05,
        non_ocr_family_present=True,
        no_oracle_leakage=True,
        min_images=30,
        min_cic_random_gap=0.15,
        max_content_preservation_drop=0.10,
    )
    kwargs.update(overrides)
    return mod.evaluate_open_proposal_gate(**kwargs)


def test_gate_passes_when_all_conditions_met():
    from causal_reliability.experiments import run_natural_text_open_proposal_cic as mod

    supported, reasons = _gate(mod)
    assert supported is True
    assert reasons == []


def test_gate_fails_on_fake_backend():
    from causal_reliability.experiments import run_natural_text_open_proposal_cic as mod

    supported, reasons = _gate(mod, backend="fake", pretrained=False)
    assert supported is False
    assert any("backend" in r for r in reasons)


def test_gate_fails_when_n_too_small():
    from causal_reliability.experiments import run_natural_text_open_proposal_cic as mod

    supported, reasons = _gate(mod, n_images=5)
    assert supported is False
    assert any("n_images" in r for r in reasons)


def test_gate_fails_when_cic_does_not_beat_random():
    from causal_reliability.experiments import run_natural_text_open_proposal_cic as mod

    supported, reasons = _gate(mod, cic_top1_repair_accuracy=0.60, matched_random_repair_accuracy=0.55)
    assert supported is False
    assert any("matched random" in r for r in reasons)


def test_gate_fails_without_non_ocr_family():
    from causal_reliability.experiments import run_natural_text_open_proposal_cic as mod

    supported, reasons = _gate(mod, non_ocr_family_present=False)
    assert supported is False
    assert any("non-OCR" in r for r in reasons)


def test_gate_fails_on_oracle_leakage():
    from causal_reliability.experiments import run_natural_text_open_proposal_cic as mod

    supported, reasons = _gate(mod, no_oracle_leakage=False)
    assert supported is False
    assert any("leakage" in r for r in reasons)
