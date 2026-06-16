from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from causal_reliability.data.natural_text_dataset import (
    load_verified_natural_text_dataset,
    parse_pipe_bbox_list,
)
from causal_reliability.real_models.clip_zero_shot import ClipStatus


# --------------------------------------------------------------------------- #
# Fake CLIP classifier (mean-based, deterministic) for end-to-end runs
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
    monkeypatch.setattr(mod, "_build_predict_fn", _orig_predict_fn_factory(mod))
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


def _orig_predict_fn_factory(mod):
    def _factory(status, allowed_labels, device):
        clf = _TinyClassifier(status, allowed_labels)

        def predict_fn(images):
            arrays = [np.asarray(img.convert("RGB")).astype(np.float32) / 255.0 for img in images]
            tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
            out = clf.predict(tensor)
            return np.asarray(out["probabilities"].numpy(), dtype=np.float64)

        return predict_fn

    return _factory


# --------------------------------------------------------------------------- #
# Tiny verified-annotation fixture
# --------------------------------------------------------------------------- #
def _make_fixture(tmp_path: Path, n: int = 5) -> Path:
    root = tmp_path / "natural_text_images"
    images = root / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        arr = (rng.uniform(0.1, 0.9, size=(64, 64, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(images / f"img{i}.jpg")
        include = "yes" if i % 5 != 4 else "no"
        rows.append(
            {
                "image_path": f"images/img{i}.jpg",
                "visual_target_label": "dog",
                "visual_label_aliases": "dog;animal;puppy;text;logo",
                "text_distractor_labels": "animal;puppy;text;logo",
                "text_or_logo_boxes": "2,2,30,20|5,40,60,60",
                "object_boxes": "10,10,55,55",
                "text_driven_candidate": "yes",
                "include_in_verified_failure_eval": include,
                "exclusion_reason": "" if include == "yes" else "REVIEW: excluded",
                "notes": "fixture",
            }
        )
    csv_path = root / "verified_annotations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return root


def _base_cfg(tmp_path: Path, root: Path, **overrides) -> dict:
    cfg = {
        "results_dir": str(tmp_path / "results"),
        "max_candidates": 16,
        "min_verified_failures": 1,
        "n_example_visualizations": 2,
        "data": {"mode": "local", "image_size": 64, "verified": {"root": str(root)}},
        "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
    }
    cfg.update(overrides)
    return cfg


# --------------------------------------------------------------------------- #
# Parsing + loader
# --------------------------------------------------------------------------- #
def test_parse_pipe_bbox_list_handles_multi_and_empty():
    assert parse_pipe_bbox_list("") == []
    assert parse_pipe_bbox_list("nan") == []
    assert parse_pipe_bbox_list("1,2,3,4") == [(1, 2, 3, 4)]
    assert parse_pipe_bbox_list("1,2,3,4|5,6,7,8") == [(1, 2, 3, 4), (5, 6, 7, 8)]


def test_loader_filters_include_yes_and_carries_metadata(tmp_path: Path):
    root = _make_fixture(tmp_path, n=10)
    bundle = load_verified_natural_text_dataset(root, image_size=64)
    # 10 rows, every 5th (index 4, 9) is include=no -> 8 yes.
    assert bundle.diagnostics["n_total_rows"] == 10
    assert bundle.diagnostics["n_include_yes"] == 8
    assert len(bundle.examples) == 8
    ex = bundle.examples[0]
    assert ex["allowed_clip_labels"][ex["label"]] == "dog"
    assert ex["text_distractor_labels"] == ["animal", "puppy", "text", "logo"]
    assert len(ex["text_boxes"]) == 2 and len(ex["object_boxes"]) == 1


def test_loader_on_real_verified_csv_has_37_include_yes():
    bundle = load_verified_natural_text_dataset("data/natural_text_images", image_size=224)
    assert bundle.diagnostics["n_total_rows"] == 50
    assert bundle.diagnostics["n_include_yes"] == 37
    assert len(bundle.examples) == 37
    for ex in bundle.examples:
        assert ex["text_boxes"], ex["human_label"]
        assert ex["object_boxes"], ex["human_label"]


# --------------------------------------------------------------------------- #
# Gate logic (pure function)
# --------------------------------------------------------------------------- #
def _gate(mod, **overrides):
    kwargs = dict(
        backend="open_clip",
        pretrained=True,
        fake_backend=False,
        n_verified_failures=25,
        oracle_repair_or_improve_rate=0.80,
        cic_top1_repair_accuracy=0.70,
        matched_random_repair_accuracy=0.40,
        content_preservation_drop=0.05,
        no_oracle_leakage=True,
        open_world_claim_allowed=False,
    )
    kwargs.update(overrides)
    return mod.evaluate_natural_text_gate(**kwargs)


def test_gate_passes_when_all_conditions_met():
    from causal_reliability.experiments import run_natural_text_verified_failure_eval as mod

    supported, reasons = _gate(mod)
    assert supported is True
    assert reasons == []


def test_gate_fails_on_fake_backend():
    from causal_reliability.experiments import run_natural_text_verified_failure_eval as mod

    supported, reasons = _gate(mod, backend="fake", pretrained=False, fake_backend=True)
    assert supported is False
    assert any("backend" in r for r in reasons)


def test_gate_fails_when_too_few_verified_failures():
    from causal_reliability.experiments import run_natural_text_verified_failure_eval as mod

    supported, reasons = _gate(mod, n_verified_failures=5)
    assert supported is False
    assert any("verified text-driven failures" in r for r in reasons)


def test_gate_fails_when_oracle_repair_too_low():
    from causal_reliability.experiments import run_natural_text_verified_failure_eval as mod

    supported, reasons = _gate(mod, oracle_repair_or_improve_rate=0.50)
    assert supported is False
    assert any("oracle" in r for r in reasons)


def test_gate_fails_when_cic_does_not_beat_random():
    from causal_reliability.experiments import run_natural_text_verified_failure_eval as mod

    supported, reasons = _gate(mod, cic_top1_repair_accuracy=0.45, matched_random_repair_accuracy=0.40)
    assert supported is False
    assert any("matched random" in r for r in reasons)


def test_gate_fails_when_open_world_claim_allowed():
    from causal_reliability.experiments import run_natural_text_verified_failure_eval as mod

    supported, reasons = _gate(mod, open_world_claim_allowed=True)
    assert supported is False
    assert any("open_world_claim_allowed" in r for r in reasons)


# --------------------------------------------------------------------------- #
# End-to-end run + artifacts
# --------------------------------------------------------------------------- #
def test_end_to_end_outputs_written(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_natural_text_verified_failure_eval as mod

    _patch_real_clip(monkeypatch, mod)
    root = _make_fixture(tmp_path, n=10)
    outputs = mod.run(_base_cfg(tmp_path, root))

    for key in ("metrics", "per_example", "key_numbers", "summary", "proposal_diagnostics"):
        assert Path(outputs[key]).exists(), key

    metrics = pd.read_csv(outputs["metrics"])
    methods = set(metrics["method"])
    assert {"original_clip_prediction", "cic_top1_repair", "matched_random_proposal_repair", "oracle_text_box_repair"} <= methods

    key_numbers = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    assert key_numbers["open_world_claim_allowed"] is False
    assert key_numbers["include_yes_images"] == 8
    assert key_numbers["total_images"] == 10
    assert "natural_text_supported" in key_numbers


def test_fake_backend_cannot_support_claim(tmp_path: Path):
    from causal_reliability.experiments.run_natural_text_verified_failure_eval import run

    root = _make_fixture(tmp_path, n=6)
    cfg = _base_cfg(tmp_path, root)
    cfg["model"] = {"backend": "fake"}
    outputs = run(cfg)
    key_numbers = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    assert key_numbers["natural_text_supported"] is False
    assert key_numbers["open_world_claim_allowed"] is False
    assert key_numbers["fake_backend"] is True


def test_does_not_overwrite_other_result_dirs(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_natural_text_verified_failure_eval as mod

    _patch_real_clip(monkeypatch, mod)
    root = _make_fixture(tmp_path, n=8)
    outputs = mod.run(_base_cfg(tmp_path, root))
    # All artifacts live strictly under the verified output subdir.
    out_root = Path(tmp_path / "results" / "natural_text_verified_failure_eval")
    for key in ("metrics", "per_example", "key_numbers", "summary", "proposal_diagnostics"):
        assert out_root in Path(outputs[key]).parents
