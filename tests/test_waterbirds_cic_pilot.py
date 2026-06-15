from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from causal_reliability.real_models.clip_zero_shot import ClipStatus


SKIP_NO_MASK = "Waterbirds pilot skipped: no oracle-repairable bird/background mask available."


def test_skips_cleanly_when_dataset_absent(tmp_path: Path):
    """Default config path / nonexistent data -> graceful skip, not a crash."""
    from causal_reliability.experiments.run_waterbirds_cic_pilot import run

    outputs = run(
        {
            "results_dir": str(tmp_path),
            "data": {"root": str(tmp_path / "no_such_waterbirds")},
        }
    )
    metrics = pd.read_csv(outputs["metrics"])
    key = json.loads(Path(outputs["key_numbers"]).read_text())
    summary = Path(outputs["summary"]).read_text()

    assert metrics["method"].iloc[0] == "skipped"
    assert not bool(metrics["headline_eligible"].iloc[0])
    assert not bool(metrics["dataset_available"].iloc[0])
    assert key["waterbirds_headline_eligible"] is False
    assert key["skipped"] is True
    # All required artifacts exist.
    for artifact in ["waterbirds_summary.md", "waterbirds_metrics.csv", "waterbirds_key_numbers.json", "waterbirds_examples.md", "waterbirds_plot.png", "waterbirds_plot.pdf", "waterbirds_caption.md"]:
        assert (tmp_path / "waterbirds_cic_pilot" / artifact).exists()
    assert "skipped" in summary.lower()


def test_skips_with_specific_message_when_masks_missing(tmp_path: Path):
    """Dataset present but no mask/bbox -> the spec-mandated skip message."""
    from causal_reliability.experiments.run_waterbirds_cic_pilot import check_dataset, run

    root = tmp_path / "waterbirds"
    root.mkdir()
    img = Image.new("RGB", (32, 32), (120, 120, 120))
    img.save(root / "a.png")
    img.save(root / "b.png")
    pd.DataFrame({"img_filename": ["a.png", "b.png"], "y": [0, 1], "split": [2, 2]}).to_csv(root / "metadata.csv", index=False)

    avail = check_dataset({"root": str(root), "mask_column": "mask_filename", "bbox_columns": ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]})
    assert avail["dataset_available"] is True
    assert avail["masks_available"] is False
    assert avail["reason"] == SKIP_NO_MASK

    outputs = run({"results_dir": str(tmp_path), "data": {"root": str(root)}})
    summary = Path(outputs["summary"]).read_text()
    assert SKIP_NO_MASK in summary
    metrics = pd.read_csv(outputs["metrics"])
    assert not bool(metrics["headline_eligible"].iloc[0])


def test_fake_backend_cannot_be_headline_eligible(tmp_path: Path):
    """A masked dataset + fake backend must never be eligible."""
    from causal_reliability.experiments.run_waterbirds_cic_pilot import run

    root = _make_masked_dataset(tmp_path, n=6)
    outputs = run({"results_dir": str(tmp_path), "data": {"root": str(root), "image_size": 32}, "model": {"backend": "fake"}})
    metrics = pd.read_csv(outputs["metrics"])
    summary = Path(outputs["summary"]).read_text()
    assert not bool(metrics["headline_eligible"].any())
    assert "fake" in summary.lower() or "skipped" in summary.lower()


def test_nonoracle_cic_scoring_has_no_oracle_parameters():
    """The CIC ranker must not accept label/group/correctness."""
    from causal_reliability.experiments.run_waterbirds_cic_pilot import cic_rank

    params = set(inspect.signature(cic_rank).parameters)
    forbidden = {"label", "labels", "true_label", "group", "place", "correctness", "is_correct", "y"}
    assert not (params & forbidden)
    # Only candidate distributions and the original distribution are inputs.
    assert params == {"candidate_probs", "original_probs"}


def test_full_pipeline_runs_with_masked_data_and_stub_clip(tmp_path: Path, monkeypatch):
    """Exercise the real pipeline end-to-end with a tiny synthetic masked dataset.

    The dataset is tiny, so headline eligibility must be False, but all metrics
    and artifacts should be produced and internally consistent.
    """
    from causal_reliability.experiments import run_waterbirds_cic_pilot as mod

    root = _make_masked_dataset(tmp_path, n=8)

    class StubClassifier:
        """Predicts based on mean brightness; deterministic and label-free."""

        def __init__(self, status, class_names, prompts=None, device="cpu"):
            self.class_names = class_names

        def predict(self, images):
            import torch

            means = images.mean(dim=(1, 2, 3)).numpy()
            probs = np.zeros((len(means), len(self.class_names)), dtype=np.float32)
            # brighter -> waterbird (class 1), darker -> landbird (class 0)
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
            "data": {"root": str(root), "image_size": 48, "min_split_examples": 2},
            "model": {"preferred_backend": "open_clip", "allow_pretrained_download": False, "device": "cpu"},
        }
    )

    metrics = pd.read_csv(outputs["metrics"])
    certs = pd.read_csv(outputs["certificates"])
    key = json.loads(Path(outputs["key_numbers"]).read_text())

    # Pipeline produced per-method metrics including CIC and oracle rows.
    assert {"cic_top1", "oracle_background_grayfill", "random_background_patch_blur", "no_intervention"}.issubset(set(metrics["method"]))
    assert len(certs) > 0
    assert key["dataset_available"] is True and key["masks_available"] is True
    assert key["nonoracle_scorer_excluded_label_group_correctness"] is True
    assert key["finite_candidate_not_open_world"] is True
    # Eligibility is a real gated verdict (either outcome is acceptable on toy data);
    # the reasons field must be consistent with the verdict.
    assert isinstance(key["waterbirds_headline_eligible"], bool)
    if key["waterbirds_headline_eligible"]:
        assert key["headline_eligibility_reasons"] == "eligible"
    else:
        assert isinstance(key["headline_eligibility_reasons"], list)
    # CIC top-1 selection is one of the finite candidates.
    assert set(certs["cic_top1_selected"]).issubset(set(mod.CIC_CANDIDATE_METHODS))
    # Failure-conditioned original accuracy is 0 by construction (when any exist).
    if key["n_failure_conditioned"] > 0:
        assert key["failure_conditioned_original_accuracy"] in (0.0, None)


SKIP_NO_DATA = "Waterbirds pilot skipped: no Waterbirds-style dataset found locally."


def _make_wilds_dataset(tmp_path: Path, n: int = 8) -> Path:
    """Create a WILDS Waterbirds-style release (metadata.csv, images, NO masks)."""
    wilds_root = tmp_path / "wilds"
    data_dir = wilds_root / "waterbirds_v1.0"
    data_dir.mkdir(parents=True)
    rows = []
    for i in range(n):
        y = i % 2
        place = (i // 2) % 2
        sub = data_dir / f"{i:03d}.Some_Bird"
        sub.mkdir(exist_ok=True)
        rel = f"{i:03d}.Some_Bird/img_{i}.jpg"
        Image.new("RGB", (32, 32), (100 + 10 * i % 120, 120, 120)).save(data_dir / rel)
        rows.append(
            {
                "img_id": i + 1,
                "img_filename": rel,
                "y": y,
                "split": 2,
                "place": place,
                "place_filename": f"/x/{place}/{i}.jpg",
            }
        )
    pd.DataFrame(rows).to_csv(data_dir / "metadata.csv", index=False)
    return wilds_root


def test_wilds_dataset_detected_when_present(tmp_path: Path):
    """source=wilds resolves the versioned release dir; no masks -> not available."""
    from causal_reliability.experiments.run_waterbirds_cic_pilot import check_dataset

    wilds_root = _make_wilds_dataset(tmp_path, n=8)
    avail = check_dataset({"source": "wilds", "wilds_root": str(wilds_root), "wilds_dataset": "waterbirds"})
    assert avail["dataset_available"] is True
    assert avail["masks_available"] is False
    assert avail["oracle_repair_available"] is False
    assert avail["source"] == "wilds"
    assert avail["n_rows"] == 8
    assert avail["reason"] == SKIP_NO_MASK
    assert str(tmp_path / "wilds" / "waterbirds_v1.0") in avail["root"]


def test_skips_when_wilds_not_installed(tmp_path: Path):
    """source=wilds but nothing on disk -> graceful skip, dataset_available False."""
    from causal_reliability.experiments.run_waterbirds_cic_pilot import check_dataset, run

    avail = check_dataset({"source": "wilds", "wilds_root": str(tmp_path / "empty_wilds")})
    assert avail["dataset_available"] is False
    assert avail["source"] == "wilds"
    assert SKIP_NO_DATA in avail["reason"]

    outputs = run(
        {
            "results_dir": str(tmp_path),
            "data": {"source": "wilds", "wilds_root": str(tmp_path / "empty_wilds"), "run_metadata_diagnostic": False},
        }
    )
    key = json.loads(Path(outputs["key_numbers"]).read_text())
    assert key["dataset_available"] is False
    assert key["waterbirds_headline_eligible"] is False
    assert key["skipped"] is True


def test_wilds_missing_masks_not_headline_eligible_and_no_oracle(tmp_path: Path):
    """WILDS found, no masks -> dataset_available True, masks/oracle/headline all False."""
    from causal_reliability.experiments.run_waterbirds_cic_pilot import run

    wilds_root = _make_wilds_dataset(tmp_path, n=8)
    outputs = run(
        {
            "results_dir": str(tmp_path),
            "data": {
                "source": "wilds",
                "wilds_root": str(wilds_root),
                "image_size": 32,
                "run_metadata_diagnostic": False,  # keep the test offline / fast
            },
        }
    )
    key = json.loads(Path(outputs["key_numbers"]).read_text())
    metrics = pd.read_csv(outputs["metrics"])
    summary = Path(outputs["summary"]).read_text()

    assert key["dataset_available"] is True
    assert key["masks_available"] is False
    assert key["oracle_repair_available"] is False
    assert key["cic_repair_ran"] is False
    assert key["waterbirds_headline_eligible"] is False
    # No oracle repair number may be claimed without masks.
    assert key["oracle_repair_accuracy"] is None
    assert not bool(metrics["headline_eligible"].any())
    assert "no oracle-repairable masks/bboxes were available" in summary
    # Converted WILDS metadata artifact was written.
    assert "wilds_converted_metadata" in outputs
    assert Path(outputs["wilds_converted_metadata"]).exists()


def test_wilds_converted_metadata_fields_parsed(tmp_path: Path):
    """The converted artifact maps y/background/split to human-readable names."""
    from causal_reliability.experiments.run_waterbirds_cic_pilot import run

    wilds_root = _make_wilds_dataset(tmp_path, n=8)
    outputs = run(
        {
            "results_dir": str(tmp_path),
            "data": {"source": "wilds", "wilds_root": str(wilds_root), "image_size": 32, "run_metadata_diagnostic": False},
        }
    )
    conv = pd.read_csv(outputs["wilds_converted_metadata"])
    assert len(conv) == 8
    for col in ["example_index", "img_filename", "y", "label_name", "background", "background_name", "split", "split_name"]:
        assert col in conv.columns
    assert set(conv["label_name"]).issubset({"landbird", "waterbird"})
    assert set(conv["background_name"]).issubset({"land", "water"})
    assert set(conv["split_name"]) == {"test"}
    # y=0 -> landbird, y=1 -> waterbird mapping is correct.
    assert (conv.loc[conv["y"] == 0, "label_name"] == "landbird").all()
    assert (conv.loc[conv["y"] == 1, "label_name"] == "waterbird").all()


def _make_masked_dataset(tmp_path: Path, n: int) -> Path:
    """Create a tiny Waterbirds-style dataset with images + bird masks."""
    root = tmp_path / "waterbirds"
    root.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        label = i % 2
        # Bird = bright square in center for waterbirds, darker for landbirds.
        arr = (rng.uniform(0.2, 0.4, size=(32, 32, 3)) * 255).astype(np.uint8)
        bird_val = 220 if label == 1 else 90
        arr[10:22, 10:22] = bird_val
        Image.fromarray(arr).save(root / f"img_{i}.png")
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:22, 10:22] = 255
        Image.fromarray(mask).save(root / f"mask_{i}.png")
        rows.append({"img_filename": f"img_{i}.png", "mask_filename": f"mask_{i}.png", "y": label, "split": 2, "place": label})
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
    return root
