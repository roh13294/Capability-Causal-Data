from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from causal_reliability.discovery.natural_text_operators import default_operators
from causal_reliability.experiments import run_coco_text_cic_triage as mod
from causal_reliability.real_models.clip_zero_shot import ClipStatus


# --------------------------------------------------------------------------- #
# Deterministic text-driven fixture classifier
# --------------------------------------------------------------------------- #
class _TextDrivenClassifier:
    """Predicts the distractor (index 1) when bright text pixels are present, and
    the target (index 0) once the text region is neutralized. Two-label only."""

    def __init__(self, status, class_names, prompts=None, device="cpu"):
        self.class_names = class_names

    def predict(self, images):
        arr = images.numpy()
        n = len(self.class_names)
        probs = np.full((len(arr), n), 0.05, dtype=np.float64)
        for k in range(len(arr)):
            bright = float((arr[k] > 0.85).mean())
            if bright > 0.02:
                probs[k] = [0.05, 0.95]  # distractor
            else:
                probs[k] = [0.95, 0.05]  # target
        return {"probabilities": torch.from_numpy(probs)}


def _patch_real_clip(monkeypatch):
    def _factory(status, allowed_labels, device):
        clf = _TextDrivenClassifier(status, allowed_labels)

        def predict_fn(images):
            arrays = [np.asarray(img.convert("RGB")).astype(np.float32) / 255.0 for img in images]
            tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
            return np.asarray(clf.predict(tensor)["probabilities"].numpy(), dtype=np.float64)

        return predict_fn

    monkeypatch.setattr(mod, "_build_predict_fn", _factory)
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


def _make_fixture(tmp_path: Path, n: int = 6) -> Path:
    root = tmp_path / "coco_text_cic"
    raw = root / "raw"
    raw.mkdir(parents=True)
    rows = []
    for i in range(n):
        arr = np.full((64, 64, 3), 0.5, dtype=np.float32)
        # Dominant object region (color varies per image, area ~0.22).
        arr[8:38, 8:38] = [(i % 5) / 5.0, 0.3, 0.7]
        # Bright text patch away from the object (drives the wrong prediction).
        arr[44:60, 44:60] = 1.0
        Image.fromarray((arr * 255).astype(np.uint8)).save(raw / f"img{i}.jpg")
        rows.append(
            {
                "image_path": f"raw/img{i}.jpg",
                "human_label": "dog",
                "allowed_clip_labels": "dog|cat",
                "optional_text_boxes": json.dumps([[44, 44, 60, 60]]),
                "optional_object_boxes": json.dumps([[8, 8, 38, 38]]),
                "source": "coco_text_cic",
                "notes": "fixture",
            }
        )
    csv_path = root / "metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return root


def _base_cfg(tmp_path: Path, root: Path, **overrides) -> dict:
    cfg = {
        "results_dir": str(tmp_path / "results"),
        "output_subdir": "coco_text_cic_triage",
        "max_images": None,
        "high_confidence_threshold": 0.7,
        "min_directional_failures": 1,
        "min_strict_failures": 1,
        "min_oracle_top5_or_pairwise_recovery": 1,
        "min_clean_examples": 1,
        "n_contact_sheet": 4,
        "data": {"root": str(root), "metadata_csv": str(root / "metadata.csv"), "image_size": 64},
        "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
    }
    cfg.update(overrides)
    return cfg


# --------------------------------------------------------------------------- #
# Pairwise margin
# --------------------------------------------------------------------------- #
def test_pairwise_margin_toward_target():
    assert mod.pairwise_margin_toward_target(0.7, 0.2) == pytest.approx(0.5)
    assert mod.pairwise_margin_toward_target(0.1, 0.6) == pytest.approx(-0.5)
    # A repair that raises target and lowers distractor moves the margin up.
    assert mod.pairwise_margin_improved(orig_margin=-0.5, post_margin=0.5)
    assert not mod.pairwise_margin_improved(orig_margin=0.5, post_margin=0.5)


def test_label_rank_and_set_prob():
    probs = np.array([0.1, 0.6, 0.3])
    assert mod.label_rank(probs, [1]) == 1
    assert mod.label_rank(probs, [0]) == 3
    assert mod.label_rank(probs, [0, 2]) == 2  # best rank over the set
    assert mod.label_set_prob(probs, [0, 2]) == 0.3


# --------------------------------------------------------------------------- #
# Directional + strict predicates
# --------------------------------------------------------------------------- #
def _good_rec(**overrides) -> dict:
    rec = {
        "original_is_target": False,
        "target_prob_orig": 0.05,
        "target_rank_orig": 3,
        "distractor_prob_orig": 0.90,
        "target_prob_post": 0.85,
        "target_rank_post": 1,
        "distractor_prob_post": 0.10,
        "post_is_target": True,
        "pairwise_margin_orig": -0.85,
        "pairwise_margin_post": 0.75,
        "text_object_iou": 0.0,
        "passes_ambiguity": True,
    }
    rec.update(overrides)
    return rec


def test_directional_predicate_accepts_clear_text_failure():
    assert mod.directional_failure_predicate(_good_rec())


def test_directional_predicate_rejects_no_target_improvement():
    # Oracle does not raise the target probability.
    rec = _good_rec(target_prob_post=0.04)
    assert not mod.directional_failure_predicate(rec)


def test_directional_predicate_rejects_high_overlap():
    rec = _good_rec(text_object_iou=0.9)
    assert not mod.directional_failure_predicate(rec, max_text_object_iou_threshold=0.5)


def test_directional_predicate_rejects_failed_ambiguity():
    assert not mod.directional_failure_predicate(_good_rec(passes_ambiguity=False))


def test_directional_predicate_requires_distractor_drop_or_rank_gain():
    # No distractor drop and no rank improvement -> rejected.
    rec = _good_rec(distractor_prob_post=0.90, target_rank_post=3)
    assert not mod.directional_failure_predicate(rec)
    # Rank improvement alone (no distractor drop) is enough.
    rec2 = _good_rec(distractor_prob_post=0.90, target_rank_post=1)
    assert mod.directional_failure_predicate(rec2)


def test_strict_predicate_requires_directional():
    rec = _good_rec()
    assert not mod.strict_oracle_repairable_predicate(rec, is_directional=False)
    assert mod.strict_oracle_repairable_predicate(rec, is_directional=True)


def test_strict_predicate_top5_needs_strong_pairwise():
    # Not top-1, sits at rank 4 (within top-5) but weak pairwise gain -> rejected.
    rec = _good_rec(post_is_target=False, target_rank_post=4, pairwise_margin_orig=0.0, pairwise_margin_post=0.05)
    assert not mod.strict_oracle_repairable_predicate(rec, is_directional=True, strong_pairwise_delta=0.15)
    # Strong pairwise gain -> accepted.
    rec2 = _good_rec(post_is_target=False, target_rank_post=4, pairwise_margin_orig=0.0, pairwise_margin_post=0.40)
    assert mod.strict_oracle_repairable_predicate(rec2, is_directional=True, strong_pairwise_delta=0.15)


def test_ambiguity_filter_rejects_clutter_and_bad_area():
    base = {"n_text_boxes": 2, "n_object_boxes": 2, "target_area_frac": 0.2}
    kw = dict(min_target_area_frac=0.03, max_target_area_frac=0.9, max_object_boxes=8, max_text_boxes=12)
    assert mod.passes_ambiguity_filters(base, **kw)
    assert not mod.passes_ambiguity_filters({**base, "n_object_boxes": 20}, **kw)
    assert not mod.passes_ambiguity_filters({**base, "target_area_frac": 0.99}, **kw)
    assert not mod.passes_ambiguity_filters({**base, "n_text_boxes": 0}, **kw)


# --------------------------------------------------------------------------- #
# Clean subset filter
# --------------------------------------------------------------------------- #
def _clean_geom(**overrides) -> dict:
    geom = {
        "n_object_boxes": 1,
        "n_text_boxes": 2,
        "n_dominant_objects": 1,
        "target_area_frac": 0.25,
        "max_text_area_frac": 0.02,
        "text_object_iou": 0.1,
    }
    geom.update(overrides)
    return geom


def test_clean_subset_accepts_well_formed_example():
    assert mod.clean_subset_predicate(_clean_geom(), {})


def test_clean_subset_rejects_too_many_boxes():
    assert not mod.clean_subset_predicate(_clean_geom(n_object_boxes=5), {})
    assert not mod.clean_subset_predicate(_clean_geom(n_text_boxes=9), {})


def test_clean_subset_rejects_multiple_dominant_and_high_iou():
    assert not mod.clean_subset_predicate(_clean_geom(n_dominant_objects=2), {})
    assert not mod.clean_subset_predicate(_clean_geom(text_object_iou=0.5), {})


def test_clean_subset_rejects_bad_text_or_target_area():
    assert not mod.clean_subset_predicate(_clean_geom(max_text_area_frac=0.0), {})
    assert not mod.clean_subset_predicate(_clean_geom(target_area_frac=0.95), {})


# --------------------------------------------------------------------------- #
# Oracle operator application on text boxes
# --------------------------------------------------------------------------- #
def test_oracle_operator_gray_fill_only_touches_text_box():
    arr = np.full((40, 40, 3), 0.2, dtype=np.float32)
    arr[5:15, 5:15] = 1.0
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    ops = {op.name: op for op in default_operators()}
    out = mod.oracle_neutralize(pil, [(5, 5, 15, 15)], ops["gray_fill"])
    out_arr = np.asarray(out).astype(np.float32) / 255.0
    # The text box is overwritten toward gray; pixels outside are untouched.
    assert abs(float(out_arr[5:15, 5:15].mean()) - 0.5) < 0.02
    assert abs(float(out_arr[20:40, 20:40].mean()) - 0.2) < 0.02


def test_oracle_operator_registry_resolves_requested_names():
    names = ["gray_fill", "expanded_gray_fill_1.25", "gaussian_blur", "expanded_blur_1.25"]
    ops = mod.operators_by_name(names)
    assert [op.name for op in ops] == names


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def _gate(**overrides):
    kwargs = dict(
        backend="open_clip",
        pretrained=True,
        fake_backend=False,
        n_directional=60,
        n_strict=40,
        n_oracle_top5_or_pairwise=55,
        n_clean=40,
    )
    kwargs.update(overrides)
    return mod.evaluate_coco_text_triage_gate(**kwargs)


def test_gate_passes_when_all_conditions_met():
    ready, reasons = _gate()
    assert ready is True and reasons == []


def test_gate_fails_on_fake_backend():
    ready, reasons = _gate(backend="fake", pretrained=False, fake_backend=True)
    assert ready is False and any("backend" in r for r in reasons)


def test_gate_strict_or_recovery_alternative():
    # Too few strict, but enough top5/pairwise recovery -> still passes.
    ready, _ = _gate(n_strict=5, n_oracle_top5_or_pairwise=60)
    assert ready is True
    # Both alternatives fail -> not ready.
    ready2, reasons = _gate(n_strict=5, n_oracle_top5_or_pairwise=5)
    assert ready2 is False and any("strict" in r for r in reasons)


def test_gate_fails_on_too_few_clean():
    ready, reasons = _gate(n_clean=2)
    assert ready is False and any("clean" in r for r in reasons)


# --------------------------------------------------------------------------- #
# End-to-end artifact writing
# --------------------------------------------------------------------------- #
def test_end_to_end_outputs_written(tmp_path: Path, monkeypatch):
    _patch_real_clip(monkeypatch)
    root = _make_fixture(tmp_path, n=6)
    outputs = mod.run(_base_cfg(tmp_path, root))

    for key in (
        "metrics",
        "key_numbers",
        "directional_failures",
        "oracle_repairable_failures",
        "clean_subset",
        "summary",
        "contact_sheet",
    ):
        assert Path(outputs[key]).exists(), key

    key_numbers = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    assert key_numbers["real_pretrained_model_loaded"] is True
    assert key_numbers["fake_backend"] is False
    assert key_numbers["ran_open_proposal_cic"] is False
    assert key_numbers["n_metadata_rows"] == 6
    # The fixture is engineered so every image is a directional + strict failure.
    assert key_numbers["n_directional_failures"] == 6
    assert key_numbers["n_strict_oracle_repairable_failures"] == 6
    assert key_numbers["n_clean_subset"] == 6
    assert key_numbers["coco_text_ready_for_full_cic"] is True
    assert key_numbers["original_clip_accuracy"] == 0.0

    metrics = pd.read_csv(outputs["metrics"])
    assert len(metrics) == 6
    assert {"is_directional_failure", "is_strict_oracle_repairable", "in_clean_subset"} <= set(metrics.columns)
    # "_records" must not leak into the persisted key numbers.
    assert "_records" not in key_numbers


def test_fake_backend_is_noop_for_claim(tmp_path: Path):
    root = _make_fixture(tmp_path, n=4)
    cfg = _base_cfg(tmp_path, root)
    cfg["model"] = {"backend": "fake"}
    outputs = mod.run(cfg)
    key_numbers = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    assert key_numbers["fake_backend"] is True
    assert key_numbers["coco_text_ready_for_full_cic"] is False
    assert key_numbers["n_directional_failures"] == 0


def test_outputs_confined_to_subdir(tmp_path: Path, monkeypatch):
    _patch_real_clip(monkeypatch)
    root = _make_fixture(tmp_path, n=5)
    outputs = mod.run(_base_cfg(tmp_path, root))
    out_root = Path(tmp_path / "results" / "coco_text_cic_triage")
    for key in ("metrics", "key_numbers", "directional_failures", "summary", "contact_sheet"):
        assert out_root in Path(outputs[key]).parents


def test_run_does_not_change_existing_final_metrics(tmp_path: Path, monkeypatch):
    final_metrics = Path("results/final_report/final_key_numbers.json")
    if not final_metrics.exists():
        import pytest

        pytest.skip("final_report metrics not present")
    before = final_metrics.read_bytes()
    _patch_real_clip(monkeypatch)
    root = _make_fixture(tmp_path, n=4)
    mod.run(_base_cfg(tmp_path, root))
    assert final_metrics.read_bytes() == before
