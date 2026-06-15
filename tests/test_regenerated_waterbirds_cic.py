from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from causal_reliability.real_models.clip_zero_shot import ClipStatus


def test_missing_data_skips_cleanly_and_writes_key_numbers(tmp_path: Path):
    """No CUB/Places assets -> graceful skip with the regeneration artifacts."""
    from causal_reliability.experiments.run_regenerated_waterbirds_cic import run

    outputs = run(
        {
            "results_dir": str(tmp_path),
            "data": {
                "cub_root": str(tmp_path / "no_cub"),
                "cub_segmentations_root": str(tmp_path / "no_seg"),
                "places_root": str(tmp_path / "no_places"),
            },
        }
    )
    metrics = pd.read_csv(outputs["metrics"])
    key = json.loads(Path(outputs["key_numbers"]).read_text())
    summary = Path(outputs["summary"]).read_text()

    assert metrics["method"].iloc[0] == "skipped"
    assert not bool(metrics["headline_eligible"].iloc[0])
    assert key["regenerated_waterbirds_available"] is False
    assert key["regenerated_waterbirds_headline_eligible"] is False
    assert key["skipped"] is True
    # PART 2 mandated regeneration artifacts.
    out = tmp_path / "regenerated_waterbirds_cic"
    assert (out / "waterbirds_regeneration_summary.md").exists()
    assert (out / "waterbirds_regeneration_key_numbers.json").exists()
    reg_key = json.loads((out / "waterbirds_regeneration_key_numbers.json").read_text())
    assert reg_key["regenerated_waterbirds_available"] is False
    assert isinstance(reg_key["missing"], list) and reg_key["missing"]
    assert "skipped" in summary.lower()


def test_missing_masks_skips_with_oracle_repairable_message(tmp_path: Path):
    """CUB present but no segmentation masks -> the mask-specific skip message."""
    from causal_reliability.experiments.run_regenerated_waterbirds_cic import (
        SKIP_NO_MASKS,
        check_assets,
        run,
    )

    cub = _make_min_cub(tmp_path)
    seg = tmp_path / "empty_seg"
    seg.mkdir()
    avail = check_assets(
        {
            "cub_root": str(cub),
            "cub_segmentations_root": str(seg),
            "places_root": str(tmp_path / "no_places"),
        }
    )
    assert avail["cub_available"] is True
    assert avail["cub_segmentations_available"] is False
    assert avail["reason"] == SKIP_NO_MASKS

    outputs = run(
        {
            "results_dir": str(tmp_path),
            "data": {
                "cub_root": str(cub),
                "cub_segmentations_root": str(seg),
                "places_root": str(tmp_path / "no_places"),
            },
        }
    )
    summary = Path(outputs["summary"]).read_text()
    assert "mask" in summary.lower()


def test_fake_backend_cannot_be_headline_eligible(tmp_path: Path):
    """Full assets + fake backend must never be eligible."""
    from causal_reliability.experiments.run_regenerated_waterbirds_cic import run

    cub, seg, places = _make_full_fixture(tmp_path)
    outputs = run(
        {
            "results_dir": str(tmp_path),
            "data": {
                "cub_root": str(cub),
                "cub_segmentations_root": str(seg),
                "places_root": str(places),
                "image_size": 32,
                "min_backgrounds_per_type": 2,
            },
            "model": {"backend": "fake"},
        }
    )
    metrics = pd.read_csv(outputs["metrics"])
    key = json.loads(Path(outputs["key_numbers"]).read_text())
    assert not bool(metrics["headline_eligible"].any())
    assert key["regenerated_waterbirds_headline_eligible"] is False


def test_nonoracle_cic_scoring_has_no_oracle_parameters():
    """The CIC ranker must not accept label/group/mask/correctness."""
    from causal_reliability.experiments.run_regenerated_waterbirds_cic import cic_rank

    params = set(inspect.signature(cic_rank).parameters)
    forbidden = {
        "label",
        "labels",
        "true_label",
        "group",
        "place",
        "background",
        "background_label",
        "correctness",
        "is_correct",
        "mask",
        "bird_mask",
        "background_mask",
        "oracle_mask",
        "y",
    }
    assert not (params & forbidden)
    assert params == {"candidate_probs", "original_probs"}


def test_class_and_background_mapping_are_transparent():
    from causal_reliability.experiments.run_regenerated_waterbirds_cic import (
        classify_background_folder,
        map_cub_class_to_bird_label,
    )

    # Heuristic water-bird keywords -> waterbird (1); others -> landbird (0).
    assert map_cub_class_to_bird_label("059.California_Gull") == 1
    assert map_cub_class_to_bird_label("009.Brewer_Blackbird") == 0
    # Official map overrides the heuristic.
    assert map_cub_class_to_bird_label("009.Brewer_Blackbird", {"009.Brewer_Blackbird": 1}) == 1
    # Background folder classification.
    assert classify_background_folder("lakeside") == "water"
    assert classify_background_folder("bamboo_forest") == "land"
    assert classify_background_folder("kitchen") is None


def test_fixture_generates_composites_with_valid_masks(tmp_path: Path):
    """A synthetic mini CUB/Places fixture composites a few examples with masks."""
    from causal_reliability.experiments.run_regenerated_waterbirds_cic import (
        check_assets,
        generate_dataset,
    )

    cub, seg, places = _make_full_fixture(tmp_path)
    data_cfg = {
        "cub_root": str(cub),
        "cub_segmentations_root": str(seg),
        "places_root": str(places),
        "image_size": 32,
        "min_backgrounds_per_type": 2,
        "max_examples": 50,
    }
    avail = check_assets(data_cfg)
    assert avail["places_available"] is True

    out_root = tmp_path / "regen_out"
    out_root.mkdir()
    examples, metadata, gen_info = generate_dataset(avail, data_cfg, out_root, seed=0)
    assert len(examples) >= 2
    # Masks are binary and non-degenerate (some bird, some background).
    for ex in examples:
        m = ex["mask"]
        assert m.dtype == bool
        assert m.any() and not m.all()
    # Metadata has one row per regime per example with required columns.
    required_cols = {
        "example_id",
        "image_path",
        "bird_mask_path",
        "background_mask_path",
        "bird_label",
        "background_label",
        "aligned_or_misleading",
        "source_cub_image",
        "source_cub_class",
        "image_hash",
        "mask_hash",
    }
    assert required_cols.issubset(set(metadata.columns))
    assert set(metadata["regime"]) == {"aligned", "misleading", "neutral"}
    assert gen_info["class_map_kind"] == "heuristic_keyword"
    # Saved files exist.
    assert (out_root / "images").exists() and any((out_root / "images").iterdir())
    assert (out_root / "masks").exists() and any((out_root / "masks").iterdir())


def test_oracle_neutralization_uses_mask_only_in_oracle_path(tmp_path: Path):
    """Oracle background neutralization keeps the bird and changes the background."""
    from causal_reliability.experiments.run_waterbirds_cic_pilot import build_interventions

    rng = np.random.default_rng(0)
    arr = (rng.uniform(0.0, 1.0, size=(32, 32, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    ex = {"image": img, "mask": mask, "label": 0, "example_id": 0}
    ivs = build_interventions(ex, {}, rng)
    oracle = np.asarray(ivs["oracle_background_grayfill"]).astype(np.float32) / 255.0
    orig = arr.astype(np.float32) / 255.0
    # Bird pixels preserved exactly; background pixels changed to fill.
    assert np.allclose(oracle[mask], orig[mask], atol=1e-6)
    assert not np.allclose(oracle[~mask], orig[~mask])


def test_full_pipeline_with_fixture_and_stub_clip(tmp_path: Path, monkeypatch):
    """End-to-end on a tiny fixture with a stubbed CLIP backend.

    The fixture is tiny so headline eligibility must be False, but artifacts and
    failure-conditioned semantics must be internally consistent.
    """
    from causal_reliability.experiments import run_regenerated_waterbirds_cic as mod

    cub, seg, places = _make_full_fixture(tmp_path)

    class StubClassifier:
        def __init__(self, status, class_names, prompts=None, device="cpu"):
            self.class_names = class_names

        def predict(self, images):
            import torch

            means = images.mean(dim=(1, 2, 3)).numpy()
            probs = np.zeros((len(means), len(self.class_names)), dtype=np.float32)
            p1 = np.clip(means, 0.05, 0.95)
            probs[:, 1] = p1
            probs[:, 0] = 1.0 - p1
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
    monkeypatch.setattr(mod, "ClipZeroShotClassifier", StubClassifier)

    outputs = mod.run(
        {
            "results_dir": str(tmp_path),
            "data": {
                "cub_root": str(cub),
                "cub_segmentations_root": str(seg),
                "places_root": str(places),
                "image_size": 48,
                "min_backgrounds_per_type": 2,
                "max_examples": 60,
                "min_examples": 2,
                "output_root": str(tmp_path / "regen_data"),
            },
            "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
        }
    )

    metrics = pd.read_csv(outputs["metrics"])
    certs = pd.read_csv(outputs["certificates"])
    key = json.loads(Path(outputs["key_numbers"]).read_text())

    assert {"aligned", "cic_top1", "oracle_background_grayfill", "no_intervention"}.issubset(set(metrics["method"]))
    assert len(certs) > 0
    assert key["regenerated_waterbirds_available"] is True
    assert key["nonoracle_scorer_excluded_label_group_masks_correctness"] is True
    assert key["finite_candidate_not_open_world"] is True
    assert isinstance(key["regenerated_waterbirds_headline_eligible"], bool)
    if key["regenerated_waterbirds_headline_eligible"]:
        assert key["headline_eligibility_reasons"] == "eligible"
    else:
        assert isinstance(key["headline_eligibility_reasons"], list)
    assert set(certs["cic_top1_selected"]).issubset(set(mod.CIC_CANDIDATE_METHODS))
    # Failure-conditioned: original accuracy is 0 by construction (when any exist).
    if key["n_failure_conditioned"] > 0:
        assert key["failure_conditioned_original_accuracy"] in (0.0, None)
    # Metadata CSV was written to the generated dataset root.
    assert Path(key["metadata_csv"]).exists()


def test_headline_eligibility_requires_real_assets_and_no_leakage():
    """Eligibility helper flags missing assets / fake backend as reasons."""
    from causal_reliability.experiments.run_regenerated_waterbirds_cic import _headline_eligibility

    status = ClipStatus(available=False, backend="fake", model_name="", pretrained=False)
    eligible, reasons = _headline_eligibility(
        pd.DataFrame(),
        pd.DataFrame(),
        status,
        {"cub_available": False, "cub_segmentations_available": False, "places_available": False},
        {"class_map_kind": "heuristic_keyword"},
        {},
    )
    assert eligible is False
    joined = " ".join(reasons).lower()
    assert "pretrained" in joined or "fake" in joined
    assert "cub" in joined


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _make_min_cub(tmp_path: Path) -> Path:
    """Minimal CUB index (required .txt files + one image) without masks."""
    cub = tmp_path / "cub"
    (cub / "images" / "001.Gull").mkdir(parents=True)
    img = Image.new("RGB", (40, 40), (120, 120, 120))
    img.save(cub / "images" / "001.Gull" / "g1.jpg")
    (cub / "images.txt").write_text("1 001.Gull/g1.jpg\n")
    (cub / "image_class_labels.txt").write_text("1 1\n")
    (cub / "classes.txt").write_text("1 001.California_Gull\n")
    (cub / "bounding_boxes.txt").write_text("1 8 8 24 24\n")
    return cub


def _make_full_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Mini CUB (images + masks) + Places land/water backgrounds."""
    rng = np.random.default_rng(0)
    cub = tmp_path / "cub"
    seg = tmp_path / "seg"
    places = tmp_path / "places"
    classes = {1: "001.California_Gull", 2: "002.House_Sparrow"}  # waterbird, landbird
    (cub / "images").mkdir(parents=True)
    seg.mkdir()
    img_lines, lbl_lines, cls_lines, box_lines = [], [], [], []
    image_id = 0
    for cls_id, cls_name in classes.items():
        cls_dir = f"{cls_id:03d}.{cls_name.split('.')[-1]}"
        (cub / "images" / cls_dir).mkdir(parents=True, exist_ok=True)
        (seg / cls_dir).mkdir(parents=True, exist_ok=True)
        for j in range(8):
            image_id += 1
            rel = f"{cls_dir}/img_{image_id}.jpg"
            arr = (rng.uniform(0.2, 0.5, size=(48, 48, 3)) * 255).astype(np.uint8)
            # Distinct bird region brightness so the stub classifier varies.
            bird_val = 230 if cls_id == 1 else 70
            arr[14:34, 14:34] = bird_val
            Image.fromarray(arr).save(cub / "images" / rel)
            mask = np.zeros((48, 48), dtype=np.uint8)
            mask[14:34, 14:34] = 255
            Image.fromarray(mask).save(seg / f"{cls_dir}/img_{image_id}.png")
            img_lines.append(f"{image_id} {rel}")
            lbl_lines.append(f"{image_id} {cls_id}")
            box_lines.append(f"{image_id} 14 14 20 20")
        cls_lines.append(f"{cls_id} {cls_name}")
    (cub / "images.txt").write_text("\n".join(img_lines) + "\n")
    (cub / "image_class_labels.txt").write_text("\n".join(lbl_lines) + "\n")
    (cub / "classes.txt").write_text("\n".join(cls_lines) + "\n")
    (cub / "bounding_boxes.txt").write_text("\n".join(box_lines) + "\n")

    for scene, n in (("lake", 6), ("forest", 6)):
        d = places / scene
        d.mkdir(parents=True)
        for k in range(n):
            tint = (30, 80, 160) if scene == "lake" else (60, 140, 40)
            bg = (rng.uniform(0.0, 0.2, size=(48, 48, 3)) * 255).astype(np.uint8) + np.array(tint, dtype=np.uint8)
            Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).save(d / f"bg_{k}.jpg")
    return cub, seg, places
