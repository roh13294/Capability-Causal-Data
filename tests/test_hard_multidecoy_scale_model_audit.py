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
    """Deterministic, label-free classifier whose predictions depend only on pixels."""

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


def _status_for(**kwargs) -> ClipStatus:
    return ClipStatus(
        available=True,
        backend="open_clip",
        model_name=kwargs.get("model_name", "ViT-B-32"),
        pretrained_tag=kwargs.get("pretrained_tag", "t"),
        pretrained=True,
        downloads_allowed=kwargs.get("allow_download", False),
        backend_attempted=kwargs.get("preferred_backend", "open_clip"),
    )


def _frozen_policies(frozen_dir: Path) -> dict[str, str]:
    frozen_dir.mkdir(parents=True, exist_ok=True)
    gen_path = frozen_dir / "selected_generation_policy.json"
    repair_path = frozen_dir / "selected_repair_policy.json"
    gen_path.write_text(
        json.dumps({"policy_id": "frozen_test", "class_set_size": 4, "n_decoys": 4, "placement_jitter": 4}),
        encoding="utf-8",
    )
    repair_path.write_text(
        json.dumps({"score_threshold": 0.0, "min_consensus_stability": 0.6666666667}),
        encoding="utf-8",
    )
    return {"gen": str(gen_path), "repair": str(repair_path)}


def _base_cfg(tmp_path: Path, frozen: dict[str, str], **overrides) -> dict:
    cfg = {
        "results_dir": str(tmp_path),
        "n_per_condition": 8,  # 8 // 4 classes -> test_n_per_class = 2 for fast tests
        "benchmark_seed": 4242,
        "max_candidates": 12,
        "augmentation_views": 2,
        "random_draws": 2,
        "frozen_generation_policy_path": frozen["gen"],
        "frozen_repair_policy_path": frozen["repair"],
        "gates": {"max_misleading_accuracy": 1.0, "min_cic_random_gap": -1.0, "max_clean_safe_drop": 1.0, "min_hard_misleading_n": 0},
        "data": {"image_size": 48, "validation_n_per_class": 1},
        "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
        "models": [
            {"model_name": "ViT-B-32", "pretrained_tag": "laion2b_s34b_b79k"},
            {"model_name": "ViT-B-16", "pretrained_tag": "laion2b_s34b_b88k"},
        ],
    }
    cfg.update(overrides)
    return cfg


def _patch_base(monkeypatch, status_fn=None):
    """Patch the base hard-repair module so no real CLIP weights are needed."""
    from causal_reliability.experiments import run_hard_multidecoy_clip_repair as base

    status_fn = status_fn or _status_for
    monkeypatch.setattr(base, "check_clip_available", status_fn)
    monkeypatch.setattr(base, "ClipZeroShotClassifier", TinyClassifier)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_fake_backend_cannot_be_included(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_hard_multidecoy_scale_model_audit as mod

    _patch_base(monkeypatch)
    monkeypatch.setattr(mod, "_registry_pairs", lambda: None)
    frozen = _frozen_policies(tmp_path / "frozen")
    cfg = _base_cfg(tmp_path, frozen)
    cfg["model"]["backend"] = "fake"

    outputs = mod.run(cfg)
    key_numbers = json.loads(Path(outputs["key_numbers"]).read_text())
    availability = pd.read_csv(outputs["model_availability"])
    metrics = pd.read_csv(outputs["metrics"]) if Path(outputs["metrics"]).read_text().strip() else pd.DataFrame()

    assert key_numbers["fake_backend_allowed"] is False
    assert key_numbers["fake_backend_requested"] is True
    assert key_numbers["models_loaded"] == []
    assert key_numbers["n_eligible_models"] == 0
    # every candidate is skipped with a fake-backend reason
    assert availability["skipped"].astype(bool).all()
    assert availability["skip_reason"].str.contains("fake backend", case=False).all()
    # no model row claims headline eligibility
    if len(metrics):
        assert not metrics["headline_eligible"].fillna(False).astype(bool).any()


def test_unavailable_models_are_skipped_not_fatal(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_hard_multidecoy_scale_model_audit as mod

    _patch_base(monkeypatch)
    # Registry knows only the headline pair; the other candidate must be skipped.
    monkeypatch.setattr(mod, "_registry_pairs", lambda: {("ViT-B-32", "laion2b_s34b_b79k")})
    frozen = _frozen_policies(tmp_path / "frozen")
    cfg = _base_cfg(tmp_path, frozen)

    outputs = mod.run(cfg)  # must not raise
    availability = pd.read_csv(outputs["model_availability"])

    skipped = availability[availability["skipped"].astype(bool)]
    not_known = availability[availability["registry_known"] == False]  # noqa: E712
    assert len(not_known) >= 1
    assert (skipped["skip_reason"].str.contains("list_pretrained", case=False)).any()
    # headline pair is still attempted/loaded
    headline = availability[(availability["model_name"] == "ViT-B-32") & (availability["pretrained_tag"] == "laion2b_s34b_b79k")]
    assert bool(headline["loaded"].iloc[0])


def test_load_failure_is_recorded_and_skipped(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_hard_multidecoy_scale_model_audit as mod

    def flaky_status(**kwargs):
        if kwargs.get("model_name") == "ViT-B-16":
            return ClipStatus(False, "unavailable", "ViT-B-16", "", pretrained=False, error_message="weights missing")
        return _status_for(**kwargs)

    _patch_base(monkeypatch, status_fn=flaky_status)
    monkeypatch.setattr(mod, "_registry_pairs", lambda: None)  # attempt all
    frozen = _frozen_policies(tmp_path / "frozen")
    cfg = _base_cfg(tmp_path, frozen)

    outputs = mod.run(cfg)
    availability = pd.read_csv(outputs["model_availability"])

    bad = availability[availability["model_name"] == "ViT-B-16"].iloc[0]
    assert bool(bad["attempted"]) is True
    assert bool(bad["loaded"]) is False
    assert bool(bad["skipped"]) is True
    assert "did not load" in str(bad["skip_reason"]).lower() or "missing" in str(bad["skip_reason"]).lower()
    # the good model still loaded
    good = availability[availability["model_name"] == "ViT-B-32"].iloc[0]
    assert bool(good["loaded"]) is True


def test_model_loading_results_are_recorded(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_hard_multidecoy_scale_model_audit as mod

    _patch_base(monkeypatch)
    monkeypatch.setattr(mod, "_registry_pairs", lambda: None)
    frozen = _frozen_policies(tmp_path / "frozen")
    cfg = _base_cfg(tmp_path, frozen)

    outputs = mod.run(cfg)
    availability = pd.read_csv(outputs["model_availability"])
    key_numbers = json.loads(Path(outputs["key_numbers"]).read_text())

    for col in ["model_name", "pretrained_tag", "attempted", "loaded", "skipped", "skip_reason", "backend"]:
        assert col in availability.columns
    # headline plus the extra candidate -> at least two attempts
    assert len(availability) >= 2
    assert len(key_numbers["models_loaded"]) >= 1
    assert any(m["model_name"] == "ViT-B-32" for m in key_numbers["models_loaded"])


def test_larger_n_benchmark_writes_distinct_hashes(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_hard_multidecoy_scale_model_audit as mod

    _patch_base(monkeypatch)
    monkeypatch.setattr(mod, "_registry_pairs", lambda: None)
    frozen = _frozen_policies(tmp_path / "frozen")
    cfg = _base_cfg(tmp_path, frozen)

    outputs = mod.run(cfg)
    metrics = pd.read_csv(outputs["metrics"])

    loaded = metrics[metrics["pretrained_loaded"].astype(bool)]
    assert len(loaded) >= 1
    # benchmark hashes are present and non-empty for loaded models
    assert loaded["benchmark_hash"].astype(str).str.len().gt(0).all()
    # the resampled benchmark differs from the frozen n=32 headline benchmark
    from causal_reliability.experiments.run_hard_multidecoy_clip_repair import make_hard_dataset
    from causal_reliability.experiments.run_hard_multidecoy_clip_repair_audit import _metadata_hash

    policy = json.loads(Path(frozen["gen"]).read_text())
    headline_examples = make_hard_dataset(8, policy, size=48, split="test", benchmark_seed=0, resample=False)
    headline_hash = _metadata_hash(headline_examples)
    assert (loaded["benchmark_hash"].astype(str) != headline_hash).all()


def test_headline_eligibility_gates_are_enforced(tmp_path: Path):
    from causal_reliability.experiments.run_hard_multidecoy_scale_model_audit import _eligibility, DEFAULT_GATES

    strong = {
        "pretrained_loaded": True,
        "original_misleading_accuracy": 0.20,
        "cic_top1_minus_matched_random_gap": 0.40,
        "clean_safe_clean_drop": 0.02,
        "n_hard_misleading_examples": 128,
    }
    elig = _eligibility(strong, fake_backend=False, gates=DEFAULT_GATES)
    assert elig["headline_eligible"] is True
    assert elig["eligibility_status"] == "repair_eligible"

    # fake backend can never be eligible
    fake = _eligibility(strong, fake_backend=True, gates=DEFAULT_GATES)
    assert fake["headline_eligible"] is False

    # not failure-rich (high original accuracy) -> not a negative CIC result
    robust = dict(strong, original_misleading_accuracy=0.92)
    not_rich = _eligibility(robust, fake_backend=False, gates=DEFAULT_GATES)
    assert not_rich["headline_eligible"] is False
    assert not_rich["eligibility_status"] == "not_failure_rich"

    # small gap fails the CIC-beats-random gate
    small_gap = dict(strong, cic_top1_minus_matched_random_gap=0.05)
    weak = _eligibility(small_gap, fake_backend=False, gates=DEFAULT_GATES)
    assert weak["headline_eligible"] is False
    assert "gap" in weak["eligibility_reasons"]

    # large clean drop fails
    big_drop = dict(strong, clean_safe_clean_drop=0.30)
    drop = _eligibility(big_drop, fake_backend=False, gates=DEFAULT_GATES)
    assert drop["headline_eligible"] is False

    # not loaded fails
    unloaded = dict(strong, pretrained_loaded=False)
    nl = _eligibility(unloaded, fake_backend=False, gates=DEFAULT_GATES)
    assert nl["headline_eligible"] is False


def test_does_not_overwrite_frozen_headline_metrics(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_hard_multidecoy_scale_model_audit as mod

    _patch_base(monkeypatch)
    monkeypatch.setattr(mod, "_registry_pairs", lambda: None)

    # Create a fake "main" headline result that must remain untouched.
    main_dir = tmp_path / "hard_multidecoy_clip_repair"
    main_dir.mkdir(parents=True)
    headline_metrics = main_dir / "hard_multidecoy_repair_metrics.csv"
    sentinel = "method,headline_eligible\noriginal_clip_prediction,True\n"
    headline_metrics.write_text(sentinel, encoding="utf-8")
    frozen = _frozen_policies(main_dir)
    frozen_before = {k: Path(v).read_text() for k, v in frozen.items()}

    cfg = _base_cfg(tmp_path, frozen)
    outputs = mod.run(cfg)

    # The frozen headline metrics file is byte-for-byte unchanged.
    assert headline_metrics.read_text(encoding="utf-8") == sentinel
    # The frozen policy files used as input are not rewritten.
    for k, v in frozen.items():
        assert Path(v).read_text() == frozen_before[k]
    # All audit outputs live under the audit directory, never the main result dir.
    audit_dir = tmp_path / "hard_multidecoy_scale_model_audit"
    for path in outputs.values():
        assert str(audit_dir) in str(Path(path).resolve())


def test_run_writes_all_required_outputs(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_hard_multidecoy_scale_model_audit as mod

    _patch_base(monkeypatch)
    monkeypatch.setattr(mod, "_registry_pairs", lambda: None)
    frozen = _frozen_policies(tmp_path / "frozen")
    cfg = _base_cfg(tmp_path, frozen)

    outputs = mod.run(cfg)
    audit_dir = tmp_path / "hard_multidecoy_scale_model_audit"
    for name in [
        "scale_model_summary.md",
        "scale_model_key_numbers.json",
        "scale_model_metrics.csv",
        "scale_model_plot.png",
        "model_availability.csv",
    ]:
        assert (audit_dir / name).exists(), name

    summary = (audit_dir / "scale_model_summary.md").read_text(encoding="utf-8").lower()
    # honest-claims guardrails
    assert "open-world" in summary
    assert "cross-shortcut generalization" in summary
    assert "exact localization" in summary
