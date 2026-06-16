from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import causal_reliability.experiments.run_natural_text_intervention_sweep as sweep
from causal_reliability.discovery.natural_text_operators import (
    Operator,
    apply_operator,
    clip_box,
    cv2_available,
    default_operators,
    expand_box,
    operator_boxes,
)
from causal_reliability.real_models.clip_zero_shot import ClipStatus


# --------------------------------------------------------------------------- #
# Operator-level invariants
# --------------------------------------------------------------------------- #
def _sample_image(h: int = 70, w: int = 90) -> Image.Image:
    rng = np.random.default_rng(7)
    return Image.fromarray((rng.uniform(0.05, 0.95, (h, w, 3)) * 255).astype(np.uint8))


def test_each_operator_preserves_shape():
    img = _sample_image()
    box = (12, 10, 48, 36)
    for op in default_operators():
        out, _ = apply_operator(img, [box], op)
        assert out.size == img.size
        assert np.asarray(out).shape == np.asarray(img).shape


def test_each_operator_modifies_only_target_or_expanded_box():
    img = _sample_image()
    w, h = img.size
    box = (12, 10, 48, 36)
    before = np.asarray(img)
    for op in default_operators():
        out, avail = apply_operator(img, [box], op)
        after = np.asarray(out)
        applied = operator_boxes([box], op, w, h)
        mask = np.zeros((h, w), dtype=bool)
        for x0, y0, x1, y1 in applied:
            mask[y0:y1, x0:x1] = True
        # Pixels outside the (expanded) box must be untouched for every operator,
        # including the unavailable cv2 one (which returns an unchanged copy).
        assert np.array_equal(before[~mask], after[~mask]), op.name
        if not avail:
            assert op.requires_cv2 and not cv2_available()


def test_expansion_stays_in_bounds():
    w, h = 90, 70
    # Box hugging every edge; expansion must never exceed the image rectangle.
    for box in [(0, 0, 10, 10), (80, 60, 90, 70), (5, 5, 85, 65), (40, 30, 50, 40)]:
        for factor in (1.10, 1.25, 2.0, 4.0):
            x0, y0, x1, y1 = expand_box(box, factor, w, h)
            assert 0 <= x0 <= x1 <= w
            assert 0 <= y0 <= y1 <= h
    # factor 1.0 is a no-op (modulo clipping).
    assert expand_box((10, 10, 40, 30), 1.0, w, h) == clip_box((10, 10, 40, 30), w, h)


def test_local_mean_and_median_are_deterministic():
    img = _sample_image()
    box = (15, 12, 55, 40)
    for name in ("local_mean_fill", "local_median_fill"):
        op = next(o for o in default_operators() if o.name == name)
        a, _ = apply_operator(img, [box], op)
        b, _ = apply_operator(img, [box], op)
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_all_operators_deterministic():
    img = _sample_image()
    box = (15, 12, 55, 40)
    for op in default_operators():
        a, _ = apply_operator(img, [box], op)
        b, _ = apply_operator(img, [box], op)
        assert np.array_equal(np.asarray(a), np.asarray(b)), op.name


def test_inpaint_operator_skips_gracefully_when_cv2_unavailable():
    img = _sample_image()
    box = (12, 10, 48, 36)
    op = next(o for o in default_operators() if o.name == "telea_inpaint")
    out, avail = apply_operator(img, [box], op)
    if cv2_available():
        assert avail is True
    else:
        # Graceful skip: reported unavailable and image returned unchanged.
        assert avail is False
        assert np.array_equal(np.asarray(out), np.asarray(img))


# --------------------------------------------------------------------------- #
# Non-leakage: operator/box selection must be label-free
# --------------------------------------------------------------------------- #
def test_label_free_selection_helpers_take_no_label():
    for fn in (sweep._label_free_stability, sweep.select_label_free_best_index):
        params = set(inspect.signature(fn).parameters)
        assert "label" not in params and "target" not in params and "correct" not in params


def test_label_free_best_index_ignores_true_label():
    p_before = np.array([0.7, 0.2, 0.1])
    orig_pred = 0
    # Candidate 1 collapses the model's own top class most -> selected, regardless
    # of which class is actually the true label.
    after = [
        np.array([0.6, 0.3, 0.1]),  # small drop
        np.array([0.2, 0.5, 0.3]),  # large drop in class 0
        np.array([0.65, 0.25, 0.1]),
    ]
    assert sweep.select_label_free_best_index(after, p_before, orig_pred) == 1
    assert sweep.select_label_free_best_index([], p_before, orig_pred) is None


# --------------------------------------------------------------------------- #
# End-to-end (fake CLIP) artifact + non-leakage checks
# --------------------------------------------------------------------------- #
class _TinyClassifier:
    def __init__(self, status, class_names, prompts=None, device="cpu"):
        self.class_names = class_names

    def predict(self, images):
        means = images.mean(dim=(1, 2, 3)).numpy()
        n = len(self.class_names)
        probs = np.full((len(means), n), 0.05, dtype=np.float32)
        preds = np.floor(means * 1000).astype(int) % n
        probs[np.arange(len(means)), preds] = 0.9
        probs = probs / probs.sum(axis=1, keepdims=True)
        return {"probabilities": torch.from_numpy(probs)}


def _patch_real_clip(monkeypatch):
    def factory(status, allowed_labels, device):
        clf = _TinyClassifier(status, allowed_labels)

        def predict_fn(images):
            arrays = [np.asarray(im.convert("RGB")).astype(np.float32) / 255.0 for im in images]
            tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
            return np.asarray(clf.predict(tensor)["probabilities"].numpy(), dtype=np.float64)

        return predict_fn

    monkeypatch.setattr(sweep, "_build_predict_fn", factory)
    monkeypatch.setattr(
        sweep,
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


def _make_fixture(tmp_path: Path, n: int = 8) -> Path:
    root = tmp_path / "natural_text_images"
    images = root / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        arr = (rng.uniform(0.1, 0.9, size=(48, 48, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(images / f"img{i}.jpg")
        rows.append(
            {
                "image_path": f"images/img{i}.jpg",
                "visual_target_label": "dog",
                "visual_label_aliases": "dog;animal;text;logo",
                "text_distractor_labels": "animal;text;logo",
                "text_or_logo_boxes": "2,2,24,18|4,30,44,46",
                "object_boxes": "8,8,40,40",
                "text_driven_candidate": "yes",
                "include_in_verified_failure_eval": "yes",
                "exclusion_reason": "",
                "notes": "fixture",
            }
        )
    csv_path = root / "verified_annotations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return root


def _cfg(tmp_path: Path, root: Path) -> dict:
    return {
        "results_dir": str(tmp_path / "results"),
        "output_subdir": "natural_text_intervention_sweep",
        "max_candidates": 12,
        "high_confidence_threshold": 0.0,  # force the fake failures through the gate
        "n_example_visualizations": 1,
        "data": {"mode": "local", "image_size": 48, "verified": {"root": str(root)}},
        "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
    }


def test_end_to_end_writes_all_artifacts_and_keeps_scope(monkeypatch, tmp_path):
    _patch_real_clip(monkeypatch)
    root = _make_fixture(tmp_path, n=8)
    out = sweep.run(_cfg(tmp_path, root))
    base = Path(tmp_path) / "results" / "natural_text_intervention_sweep"
    for fname in [
        "operator_sweep_key_numbers.json",
        "oracle_operator_metrics.csv",
        "cic_operator_metrics.csv",
        "topk_operator_metrics.csv",
        "oracle_ceiling_analysis.md",
        "operator_sweep_summary.md",
    ]:
        assert (base / fname).exists(), fname

    key = json.loads((base / "operator_sweep_key_numbers.json").read_text())
    if key.get("n_verified_failures", 0) == 0:
        pytest.skip("fake classifier produced no verified failures on this fixture")

    # Scope guards must hold and final metrics must not be claimed.
    assert key["open_world_claim_allowed"] is False
    assert key["diagnostic_only"] is True
    assert key["natural_text_supported_unchanged"] is False
    assert "telea_inpaint" in key["unavailable_operators"] or cv2_available()

    # Non-leakage: every operator is evaluated on every failure (diagnostic panel),
    # i.e. no per-example operator is filtered/selected by the true label.
    oracle = pd.read_csv(base / "oracle_operator_metrics.csv")
    avail_ops = set(oracle.loc[oracle["available"] == True, "operator"])
    per_ex = pd.read_csv(base / "oracle_operator_per_example.csv")
    counts = per_ex.groupby("operator")["example_id"].nunique()
    n_fail = key["n_verified_failures"]
    for op in avail_ops:
        assert counts.get(op, 0) == n_fail, op

    # A single GLOBAL operator is reported via a documented label-free criterion.
    assert key["global_operator_criterion"]
    assert isinstance(key["strict_gate_could_pass_global"], bool)


def test_unavailable_backend_writes_diagnostic_stub(monkeypatch, tmp_path):
    root = _make_fixture(tmp_path, n=3)
    cfg = _cfg(tmp_path, root)
    cfg["model"]["preferred_backend"] = "fake"
    out = sweep.run(cfg)
    key = json.loads(Path(out["key_numbers"]).read_text())
    assert key["open_world_claim_allowed"] is False
    assert key["diagnostic_only"] is True
