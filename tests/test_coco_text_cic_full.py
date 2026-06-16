from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from causal_reliability.discovery.cic_region_scoring import score_region_candidates
from causal_reliability.discovery.open_region_proposals import generate_open_region_proposals
from causal_reliability.experiments import run_coco_text_cic_full as mod
from causal_reliability.real_models.clip_zero_shot import ClipStatus


COCO_LABELS = ["car", "truck", "pizza", "cup", "dog", "cat"]


# --------------------------------------------------------------------------- #
# Fake CLIP classifier (image-mean → deterministic prediction)
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


def _patch_real_clip(monkeypatch):
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


def _build_dataset(root: Path, n: int = 14) -> dict[str, list[int]]:
    """Create a tiny COCO-Text-like local folder + triage CSVs. Returns subset ids."""

    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        arr = (rng.uniform(0.15, 0.85, size=(64, 64, 3)) * 255).astype(np.uint8)
        # Paint a bright "text" band so high-contrast/edge proposals fire.
        arr[48:60, 6:58, :] = 245
        Image.fromarray(arr).save(raw / f"img_{i:03d}.png")
        target = COCO_LABELS[i % 3]
        distractor = COCO_LABELS[(i % 3) + 3]
        allowed = [target, distractor, COCO_LABELS[(i + 1) % 6]]
        rows.append(
            {
                "image_path": f"raw/img_{i:03d}.png",
                "human_label": target,
                "allowed_clip_labels": "|".join(dict.fromkeys(allowed)),
                "optional_text_boxes": "[[6, 48, 58, 60]]",
                "optional_object_boxes": "[[8, 8, 56, 44]]",
                "source": "test_fixture",
                "notes": "",
            }
        )
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)

    triage = root / "triage"
    triage.mkdir(parents=True, exist_ok=True)
    directional = list(range(0, 10))
    strict = list(range(0, 8))
    clean = list(range(8, n))
    pd.DataFrame({"example_id": directional}).to_csv(triage / "coco_text_verified_directional_failures.csv", index=False)
    pd.DataFrame({"example_id": strict}).to_csv(triage / "coco_text_verified_oracle_repairable_failures.csv", index=False)
    pd.DataFrame({"example_id": clean}).to_csv(triage / "coco_text_clean_subset.csv", index=False)
    return {"directional": directional, "strict": strict, "clean": clean}


def _cfg(root: Path, tmp_out: Path, **overrides) -> dict:
    cfg = {
        "results_dir": str(tmp_out),
        "output_subdir": "coco_text_cic_full",
        "max_candidates": 16,
        "predict_batch_size": 32,
        "n_example_visualizations": 2,
        "strict_min_n": 4,
        "directional_min_failures": 4,
        "data": {"root": str(root), "metadata_csv": str(root / "metadata.csv"), "image_size": 64, "split": "test"},
        "triage": {"dir": str(root / "triage")},
        "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
    }
    cfg.update(overrides)
    return cfg


# --------------------------------------------------------------------------- #
# Subset loading from triage artifacts
# --------------------------------------------------------------------------- #
def test_subset_ids_loaded_from_triage_artifacts(tmp_path: Path):
    root = tmp_path / "data"
    ids = _build_dataset(root)
    all_ids = list(range(14))
    cfg = {"triage": {"dir": str(root / "triage")}}
    subsets = mod.load_subset_ids(cfg, all_ids)
    assert subsets["all_500"] == all_ids
    assert subsets["directional_57"] == ids["directional"]
    assert subsets["strict_39"] == ids["strict"]
    clean = mod.load_clean_ids(cfg, all_ids)
    assert clean == ids["clean"]
    # Strict is a subset of directional, as in the real triage artifacts.
    assert set(subsets["strict_39"]).issubset(set(subsets["directional_57"]))


# --------------------------------------------------------------------------- #
# No-oracle-leakage in CIC scoring
# --------------------------------------------------------------------------- #
def test_no_oracle_leakage_in_scoring_and_proposals():
    score_params = set(inspect.signature(score_region_candidates).parameters)
    proposal_params = set(inspect.signature(generate_open_region_proposals).parameters)
    forbidden = {"true_label", "label", "human_label", "correctness", "repaired_correct", "shortcut_bbox", "ocr_text", "subset", "regime"}
    assert not (score_params & forbidden)
    assert not (proposal_params & forbidden)
    assert mod.scoring_is_leakage_free()


# --------------------------------------------------------------------------- #
# Separate reporting for OCR-included vs OCR-excluded proposals
# --------------------------------------------------------------------------- #
def test_ocr_included_and_excluded_reported_separately():
    assert mod.proposal_separation_is_reported()
    assert "cic_top1_repair_excl_ocr" in mod.METHOD_ORDER
    assert "cic_top1_repair_incl_ocr" in mod.METHOD_ORDER
    assert "cic_top3_repair_excl_ocr" in mod.METHOD_ORDER
    assert "cic_top3_repair_incl_ocr" in mod.METHOD_ORDER


# --------------------------------------------------------------------------- #
# End-to-end run + output artifacts
# --------------------------------------------------------------------------- #
def test_end_to_end_outputs_written(tmp_path: Path, monkeypatch):
    _patch_real_clip(monkeypatch)
    root = tmp_path / "data"
    _build_dataset(root)
    outputs = mod.run(_cfg(root, tmp_path / "out"))

    for key in ("metrics", "key_numbers", "summary", "per_example", "proposal_diagnostics", "directional_metrics", "plots"):
        assert Path(outputs[key]).exists(), key

    metrics = pd.read_csv(outputs["metrics"])
    assert set(metrics["subset"]) >= {"all_500", "directional_57", "strict_39"}
    methods = set(metrics["method"])
    for m in ("cic_top1_repair_excl_ocr", "cic_top1_repair_incl_ocr", "matched_random_proposal_repair", "oracle_text_box_repair"):
        assert m in methods

    diag = pd.read_csv(outputs["proposal_diagnostics"])
    assert len(diag) > 0
    # At least one genuine non-OCR proposal family contributed candidates (the
    # method must not reduce to "use the OCR box").
    non_ocr = {"grid_patch", "connected_component", "high_contrast", "edge_dense"}
    assert non_ocr & set(diag["proposal_family"])

    key_numbers = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    assert key_numbers["open_world_claim_allowed"] is False
    assert set(key_numbers["subsets"].keys()) == {"all_500", "directional_57", "strict_39"}
    # Directional metrics CSV is restricted to the directional subset.
    dir_metrics = pd.read_csv(outputs["directional_metrics"])
    assert set(dir_metrics["subset"]) == {"directional_57"}


def test_fake_backend_cannot_support_claim(tmp_path: Path):
    root = tmp_path / "data"
    _build_dataset(root)
    cfg = _cfg(root, tmp_path / "out")
    cfg["model"] = {"backend": "fake"}
    outputs = mod.run(cfg)
    key_numbers = json.loads(Path(outputs["key_numbers"]).read_text(encoding="utf-8"))
    assert key_numbers["coco_text_strict_support"] is False
    assert key_numbers["coco_text_directional_support"] is False
    assert key_numbers["open_world_claim_allowed"] is False
    assert key_numbers["fake_backend"] is True


# --------------------------------------------------------------------------- #
# Strict-support gate logic
# --------------------------------------------------------------------------- #
def _strict_gate(**overrides):
    kwargs = dict(
        backend="open_clip",
        pretrained=True,
        fake_backend=False,
        n=39,
        oracle_strict_repair=0.40,
        oracle_alias_repair=0.45,
        oracle_top3_recovery=0.86,
        oracle_top5_recovery=0.96,
        oracle_pairwise_recovery=0.70,
        cic_strict_repair=0.40,
        cic_alias_repair=0.42,
        random_strict_repair=0.10,
        random_alias_repair=0.12,
        cic_pairwise_recovery=0.55,
        random_pairwise_recovery=0.30,
        cic_text_overlap_rate=0.72,
        content_preservation_drop=0.05,
        content_preservation_documented=False,
        no_oracle_leakage=True,
        open_world_claim_allowed=False,
    )
    kwargs.update(overrides)
    return mod.evaluate_strict_support_gate(**kwargs)


def test_strict_gate_passes_when_all_conditions_met():
    ok, reasons = _strict_gate()
    assert ok is True and reasons == []


def test_strict_gate_fails_on_fake_backend():
    ok, reasons = _strict_gate(backend="fake", pretrained=False, fake_backend=True)
    assert ok is False and any("backend" in r for r in reasons)


def test_strict_gate_fails_when_cic_does_not_beat_random():
    ok, reasons = _strict_gate(cic_alias_repair=0.15, cic_strict_repair=0.12, cic_pairwise_recovery=0.32)
    assert ok is False and any("matched random" in r for r in reasons)


def test_strict_gate_passes_via_pairwise_when_repair_close():
    # Repair gap small, but pairwise recovery beats random by >= 0.15.
    ok, reasons = _strict_gate(cic_alias_repair=0.18, cic_strict_repair=0.15, cic_pairwise_recovery=0.55, random_pairwise_recovery=0.30)
    assert ok is True and reasons == []


def test_strict_gate_fails_on_low_text_overlap():
    ok, reasons = _strict_gate(cic_text_overlap_rate=0.40)
    assert ok is False and any("text-overlap" in r for r in reasons)


def test_strict_gate_fails_when_oracle_weak():
    ok, reasons = _strict_gate(
        oracle_strict_repair=0.10, oracle_alias_repair=0.10,
        oracle_top3_recovery=0.20, oracle_top5_recovery=0.30, oracle_pairwise_recovery=0.20,
    )
    assert ok is False and any("oracle" in r for r in reasons)


def test_strict_gate_fails_on_leakage_and_open_world():
    ok, reasons = _strict_gate(no_oracle_leakage=False)
    assert ok is False and any("leakage" in r for r in reasons)
    ok2, reasons2 = _strict_gate(open_world_claim_allowed=True)
    assert ok2 is False and any("open_world" in r for r in reasons2)


def test_strict_gate_content_drop_documented_allows_pass():
    ok, _ = _strict_gate(content_preservation_drop=0.40, content_preservation_documented=True)
    assert ok is True
    ok2, reasons2 = _strict_gate(content_preservation_drop=0.40, content_preservation_documented=False)
    assert ok2 is False and any("content-preservation" in r for r in reasons2)


# --------------------------------------------------------------------------- #
# Directional-support gate logic
# --------------------------------------------------------------------------- #
def _dir_gate(**overrides):
    kwargs = dict(
        backend="open_clip",
        pretrained=True,
        fake_backend=False,
        n_verified_failures=57,
        oracle_target_prob_improvement=0.90,
        cic_target_prob_improvement=0.55,
        random_target_prob_improvement=0.30,
        cic_text_distractor_decrease=0.50,
        random_text_distractor_decrease=0.30,
        cic_text_overlap_rate=0.70,
        no_oracle_leakage=True,
    )
    kwargs.update(overrides)
    return mod.evaluate_directional_support_gate(**kwargs)


def test_directional_gate_passes_when_all_conditions_met():
    ok, reasons = _dir_gate()
    assert ok is True and reasons == []


def test_directional_gate_fails_on_few_failures():
    ok, reasons = _dir_gate(n_verified_failures=20)
    assert ok is False and any("verified failures" in r for r in reasons)


def test_directional_gate_fails_when_cic_prob_not_beat_random():
    ok, reasons = _dir_gate(cic_target_prob_improvement=0.33, random_target_prob_improvement=0.30)
    assert ok is False and any("target-prob improvement" in r for r in reasons)


def test_directional_gate_fails_when_distractor_not_beat_random():
    ok, reasons = _dir_gate(cic_text_distractor_decrease=0.32, random_text_distractor_decrease=0.30)
    assert ok is False and any("text-distractor" in r for r in reasons)


def test_directional_gate_fails_on_weak_oracle_or_overlap():
    ok, _ = _dir_gate(oracle_target_prob_improvement=0.50)
    assert ok is False
    ok2, reasons2 = _dir_gate(cic_text_overlap_rate=0.40)
    assert ok2 is False and any("text-overlap" in r for r in reasons2)


# --------------------------------------------------------------------------- #
# Existing final metrics must remain unchanged
# --------------------------------------------------------------------------- #
def test_run_does_not_modify_final_report_metrics(tmp_path: Path, monkeypatch):
    final_json = Path("results/final_report/final_key_numbers.json")
    if not final_json.exists():
        pytest.skip("final report key numbers not present")
    before = final_json.read_bytes()
    _patch_real_clip(monkeypatch)
    root = tmp_path / "data"
    _build_dataset(root)
    mod.run(_cfg(root, tmp_path / "out"))
    assert final_json.read_bytes() == before
