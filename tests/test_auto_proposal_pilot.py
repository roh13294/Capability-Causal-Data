from __future__ import annotations

"""Tests for the automated finite-candidate proposal CIC pilot.

These tests run WITHOUT SAM/DINO and without downloading any model: CLIP is
replaced by a deterministic stub, and tiny synthetic datasets stand in for
COCO-Text / Waterbirds. They verify the module imports, optional adapters skip
cleanly, generators produce valid boxes, the pilot scripts support ``--dry-run``,
the bounded-language guarantees hold, and no protected artifacts are touched.
"""

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from causal_reliability.proposals import auto_proposals as ap
from causal_reliability.real_models.clip_zero_shot import ClipStatus

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_REPORT = REPO_ROOT / "results" / "final_report"


# --------------------------------------------------------------------------- #
# Fake CLIP (deterministic, no download)
# --------------------------------------------------------------------------- #
class _StubClassifier:
    def __init__(self, status, class_names, prompts=None, device="cpu"):
        self.class_names = list(class_names)

    def predict(self, images):
        means = images.mean(dim=(1, 2, 3)).numpy()
        n = len(self.class_names)
        probs = np.full((len(means), n), 0.1 / max(1, n), dtype=np.float32)
        preds = (np.floor(means * 997).astype(int)) % n
        probs[np.arange(len(means)), preds] = 0.9
        probs = probs / probs.sum(axis=1, keepdims=True)
        return {"probabilities": torch.from_numpy(probs)}


def _patch_clip(monkeypatch, mod):
    # Both pilot scripts build their predict_fn via the COCO pilot module's
    # _build_predict_fn, which references ClipZeroShotClassifier in that module's
    # namespace — so always patch the classifier there.
    from experiments import run_auto_proposal_cic_pilot as pilot

    monkeypatch.setattr(pilot, "ClipZeroShotClassifier", _StubClassifier)
    status = lambda **kw: ClipStatus(
        available=True,
        backend="open_clip",
        model_name="ViT-B-32",
        pretrained_tag="laion2b_s34b_b79k",
        pretrained=True,
    )
    monkeypatch.setattr(mod, "check_clip_available", status)
    if mod is not pilot:
        monkeypatch.setattr(pilot, "check_clip_available", status)


def _img(seed: int, w: int = 96, h: int = 72) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    arr[10:25, 10:60] = 250  # a bright text-like band
    return Image.fromarray(arr)


# --------------------------------------------------------------------------- #
# Module + generator behaviour
# --------------------------------------------------------------------------- #
def test_module_imports_and_registry():
    assert set(ap.available_generators()) == {
        "grid_boxes", "edge_component_boxes", "saliency_boxes", "sam_boxes", "dino_boxes"
    }


def test_optional_adapters_skip_cleanly_without_deps():
    img = _img(0)
    sam = ap.sam_boxes(img)
    dino = ap.dino_boxes(img)
    assert sam.available is False and sam.skip_reason and sam.boxes == []
    assert dino.available is False and dino.skip_reason and dino.boxes == []
    # No download attempted by default.
    assert "download" not in sam.skip_reason.lower() or "not installed" in sam.skip_reason.lower()


def test_injected_sam_proposer_is_used():
    img = _img(1)
    boxes = [(5, 5, 40, 30), (10, 10, 80, 60)]
    out = ap.sam_boxes(img, sam_proposer=lambda im: boxes)
    assert out.available is True
    assert len(out.boxes) >= 1
    w, h = img.size
    for (x0, y0, x1, y1) in out.boxes:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h


# --------------------------------------------------------------------------- #
# Real SAM adapter (skips cleanly if package/checkpoint missing)
# --------------------------------------------------------------------------- #
def _sam_available() -> bool:
    has_pkg = importlib.util.find_spec("segment_anything") is not None
    has_ckpt = (REPO_ROOT / ap.DEFAULT_SAM_CHECKPOINT).exists()
    return has_pkg and has_ckpt


def test_sam_config_defaults_match_finalized_cap():
    cfg = ap.SamConfig()
    assert cfg.max_proposals == 48  # matches the finalized candidate cap
    assert cfg.model_type == "vit_b"
    assert cfg.checkpoint_path == "models/sam/sam_vit_b_01ec64.pth"
    assert cfg.min_area_frac == 0.002 and cfg.max_area_frac == 0.80
    assert cfg.min_side == 8


def test_sam_default_disabled_skips_cleanly():
    """Without enable_real_sam, sam_boxes skips cleanly and never downloads."""

    out = ap.sam_boxes(_img(0))
    assert out.available is False and out.boxes == [] and out.skip_reason
    assert "download" not in out.skip_reason.lower()


def test_sam_enabled_but_checkpoint_missing_skips_cleanly():
    """enable_real_sam + a missing checkpoint => available=False with a clear reason."""

    out = ap.sam_boxes(_img(0), enable_real_sam=True,
                       checkpoint_path="models/sam/__definitely_missing__.pth")
    assert out.available is False and out.boxes == [] and out.skip_reason
    low = out.skip_reason.lower()
    # Either the package is absent or the checkpoint is absent — both skip cleanly.
    assert "checkpoint not found" in low or "not installed" in low


def test_sam_returns_valid_boxes_when_checkpoint_present():
    """If the package + checkpoint exist, sam_boxes returns valid in-bounds boxes."""

    if not _sam_available():
        pytest.skip("segment_anything and/or SAM checkpoint not present in this checkout")
    # A tiny synthetic image with two distinct bright/dark blobs.
    arr = np.full((96, 96, 3), 30, dtype=np.uint8)
    arr[10:40, 10:50] = 220
    arr[55:85, 50:88] = 120
    img = Image.fromarray(arr)
    out = ap.sam_boxes(img, enable_real_sam=True, device="cpu", points_per_side=8,
                       checkpoint_path=str(REPO_ROOT / ap.DEFAULT_SAM_CHECKPOINT), max_boxes=48)
    assert out.available is True
    assert len(out.boxes) >= 1
    w, h = img.size
    for (x0, y0, x1, y1) in out.boxes:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        assert (x1 - x0) >= 8 and (y1 - y0) >= 8  # min-side filter respected


def test_sam_cache_key_is_deterministic_and_setting_sensitive():
    cfg = ap.SamConfig(points_per_side=8, max_side=512, max_proposals=48)
    k1 = ap.sam_cache_key(cfg, "deadbeef", image_id=5)
    k2 = ap.sam_cache_key(cfg, "deadbeef", image_id=5)
    assert k1 == k2  # deterministic
    # Key embeds the identity + the settings that change the produced boxes.
    for token in ("id=5", "img=deadbeef", "pps=8", "maxside=512", "maxp=48", "model=", "ckpt="):
        assert token in k1
    # A changed setting (or image) must change the key.
    assert ap.sam_cache_key(ap.SamConfig(points_per_side=16, max_side=512), "deadbeef", 5) != k1
    assert ap.sam_cache_key(cfg, "feedface", image_id=5) != k1
    assert ap.sam_cache_key(cfg, "deadbeef", image_id=6) != k1
    # The on-disk filename is a stable hash of the key.
    p1 = ap._sam_cache_path("/tmp/x", k1)
    assert p1 == ap._sam_cache_path("/tmp/x", k1)
    assert p1.name.startswith("sam_") and p1.name.endswith(".json")


def test_sam_cached_proposals_load_without_rerunning_sam(tmp_path):
    """A cache hit replays cached boxes without loading SAM (even w/ a bad checkpoint)."""

    img = _img(11)
    cfg = ap.SamConfig(checkpoint_path="models/sam/__bogus__.pth", points_per_side=8, max_side=512, max_proposals=48)
    key = ap.sam_cache_key(cfg, ap._image_content_hash(img), image_id=3)
    cpath = ap._sam_cache_path(tmp_path, key)
    boxes = [(5, 5, 40, 40), (10, 10, 55, 55)]
    ap._save_cached_sam_boxes(cpath, key, boxes)
    assert ap._load_cached_sam_boxes(cpath) == boxes  # round-trips

    out = ap.sam_boxes(
        img, enable_real_sam=True, checkpoint_path="models/sam/__bogus__.pth",
        points_per_side=8, max_side=512, max_boxes=48,
        cache_dir=str(tmp_path), image_id=3, use_cache=True,
    )
    assert out.available is True and out.cache_hit is True
    assert out.boxes == boxes  # came from cache, not from SAM

    # A different image (cache miss) with the bogus checkpoint skips cleanly.
    miss = ap.sam_boxes(
        _img(12), enable_real_sam=True, checkpoint_path="models/sam/__bogus__.pth",
        cache_dir=str(tmp_path), image_id=99, use_cache=True,
    )
    assert miss.available is False and "checkpoint not found" in miss.skip_reason.lower()


def test_sam_checkpoint_is_gitignored():
    """The SAM checkpoint must never be committed."""

    ckpt = ap.DEFAULT_SAM_CHECKPOINT
    res = subprocess.run(
        ["git", "check-ignore", ckpt],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    # exit 0 + echoed path means the path is ignored by .gitignore.
    assert res.returncode == 0 and ckpt in res.stdout, (
        f"SAM checkpoint '{ckpt}' is not gitignored (returncode={res.returncode})"
    )


@pytest.mark.parametrize("fam", list(ap.CLASSICAL_FAMILIES))
def test_classical_generators_produce_valid_boxes(fam):
    img = _img(2)
    ps = ap.generate_proposal_sets(img, [fam], max_boxes=10)[fam]
    assert ps.available is True
    assert len(ps.boxes) >= 1
    w, h = img.size
    for (x0, y0, x1, y1) in ps.boxes:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h


def test_proposal_sets_to_region_proposals_adds_random_control():
    img = _img(3)
    sets = ap.generate_proposal_sets(img, ap.CLASSICAL_FAMILIES, max_boxes=8)
    rps = ap.proposal_sets_to_region_proposals(img, sets.values(), include_random_control=True)
    types = {p.proposal_type for p in rps}
    assert "random_patch_control" in types
    assert any(t.startswith("auto_") for t in types)
    # Unavailable families contribute nothing.
    sam_only = ap.generate_proposal_sets(img, ["sam_boxes"])
    rps2 = ap.proposal_sets_to_region_proposals(img, sam_only.values(), include_random_control=False)
    assert rps2 == []


def test_generator_availability_probe():
    avail = ap.generator_availability()
    assert avail["grid_boxes"].available is True
    assert avail["sam_boxes"].available is False
    assert avail["dino_boxes"].available is False


# --------------------------------------------------------------------------- #
# Dry-run support
# --------------------------------------------------------------------------- #
def test_coco_pilot_dry_run(capsys):
    from experiments import run_auto_proposal_cic_pilot as mod

    args = mod.build_parser().parse_args(["--dry-run", "--max-examples", "5"])
    info = mod.dry_run(args)
    assert info["status"] == "dry_run"
    assert info["generators_available"]["grid_boxes"] is True
    assert info["generators_available"]["sam_boxes"] is False


def test_waterbirds_diagnostic_dry_run():
    from experiments import run_waterbirds_auto_proposal_diagnostic as mod

    args = mod.build_parser().parse_args(["--dry-run", "--max-examples", "5", "--data-root", "/nonexistent"])
    info = mod.dry_run(args)
    assert info["status"] == "dry_run"
    assert info["dataset_available"] is False


# --------------------------------------------------------------------------- #
# End-to-end COCO-Text pilot with stub CLIP + tiny synthetic dataset
# --------------------------------------------------------------------------- #
def _make_coco_dataset(root: Path, n: int = 6) -> None:
    root.mkdir(parents=True, exist_ok=True)
    labels = ["bird", "cat", "dog", "car"]
    rows = []
    for i in range(n):
        rel = f"img_{i:03d}.png"
        _img(100 + i).save(root / rel)
        rows.append(
            f"{rel},{labels[i % len(labels)]},{'|'.join(labels)},"
            f"\"[[10, 10, 60, 25]]\",\"[[20, 20, 80, 60]]\",synthetic,note"
        )
    header = "image_path,human_label,allowed_clip_labels,optional_text_boxes,optional_object_boxes,source,notes"
    (root / "metadata.csv").write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _make_triage(triage_dir: Path, ids: list[int]) -> None:
    triage_dir.mkdir(parents=True, exist_ok=True)
    body = "example_id,distractor_label\n" + "\n".join(f"{i},cat" for i in ids) + "\n"
    (triage_dir / "coco_text_verified_oracle_repairable_failures.csv").write_text(body, encoding="utf-8")
    (triage_dir / "coco_text_verified_directional_failures.csv").write_text(body, encoding="utf-8")


def test_coco_pilot_end_to_end(tmp_path, monkeypatch):
    from experiments import run_auto_proposal_cic_pilot as mod

    _patch_clip(monkeypatch, mod)
    data_root = tmp_path / "coco"
    triage_dir = tmp_path / "triage"
    _make_coco_dataset(data_root, n=6)
    _make_triage(triage_dir, ids=list(range(6)))

    args = mod.build_parser().parse_args([
        "--results-dir", str(tmp_path / "results"),
        "--data-root", str(data_root),
        "--metadata-csv", str(data_root / "metadata.csv"),
        "--triage-dir", str(triage_dir),
        "--subset", "strict",
        "--max-examples", "6",
        "--max-candidates", "6",
    ])
    metrics = mod.run(args)
    assert metrics["status"] == "ok"
    assert "pilot_promising" in metrics
    out_dir = tmp_path / "results" / "auto_proposal_pilot"
    mpath = out_dir / "coco_text_auto_proposal_metrics.json"
    tpath = out_dir / "coco_text_auto_proposal_table.csv"
    spath = out_dir / "summary.md"
    assert mpath.exists() and tpath.exists() and spath.exists()
    loaded = json.loads(mpath.read_text())
    fams = loaded["subsets"]["strict"]["families"]
    assert mod.BASELINE_FAMILY in fams
    # Every evaluated family reports the required metric keys.
    for m in fams.values():
        for key in ("cic_repair_alias_accuracy", "matched_random_repair_alias_accuracy",
                    "coverage_ceiling_iou01", "median_rank_best_text_overlap",
                    "repair_localization_conflict_rate", "selected_text_overlap_rate"):
            assert key in m


def test_coco_pilot_skips_cleanly_without_data(tmp_path, monkeypatch):
    from experiments import run_auto_proposal_cic_pilot as mod

    _patch_clip(monkeypatch, mod)
    args = mod.build_parser().parse_args([
        "--results-dir", str(tmp_path / "results"),
        "--metadata-csv", str(tmp_path / "missing.csv"),
        "--max-examples", "5",
    ])
    metrics = mod.run(args)
    assert metrics["status"] == "skipped"
    assert metrics["pilot_promising"] is False


# --------------------------------------------------------------------------- #
# End-to-end Waterbirds diagnostic with stub CLIP + tiny synthetic dataset
# --------------------------------------------------------------------------- #
def _make_waterbirds(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = ["img_id,img_filename,y,split,place,place_filename"]
    k = 0
    for y in (0, 1):
        for place in (0, 1):
            for _ in range(3):
                rel = f"imgs/w_{k:03d}.png"
                (root / "imgs").mkdir(exist_ok=True)
                _img(200 + k).save(root / rel)
                rows.append(f"{k},{rel},{y},2,{place},/p/x.jpg")
                k += 1
    (root / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_waterbirds_diagnostic_end_to_end(tmp_path, monkeypatch):
    from experiments import run_waterbirds_auto_proposal_diagnostic as mod

    _patch_clip(monkeypatch, mod)
    root = tmp_path / "wb"
    _make_waterbirds(root)
    args = mod.build_parser().parse_args([
        "--results-dir", str(tmp_path / "results"),
        "--data-root", str(root),
        "--max-examples", "12",
        "--max-candidates", "6",
        "--image-size", "96",
    ])
    metrics = mod.run(args)
    assert metrics["status"] == "ok"
    assert "worst_group_accuracy_before" in metrics
    assert "background_sensitivity_before" in metrics
    assert metrics["no_oracle_masks"] is True
    out_dir = tmp_path / "results" / "auto_proposal_pilot"
    assert (out_dir / "waterbirds_auto_proposal_metrics.json").exists()
    assert (out_dir / "waterbirds_auto_proposal_table.csv").exists()


def test_waterbirds_skips_without_local_images(tmp_path, monkeypatch):
    from experiments import run_waterbirds_auto_proposal_diagnostic as mod

    _patch_clip(monkeypatch, mod)
    args = mod.build_parser().parse_args([
        "--results-dir", str(tmp_path / "results"),
        "--data-root", str(tmp_path / "nope"),
        "--max-examples", "8",
    ])
    metrics = mod.run(args)
    assert metrics["status"] == "skipped"
    assert "unavailable" in metrics["skip_reason"].lower()


# --------------------------------------------------------------------------- #
# Bounded language + protected artifacts
# --------------------------------------------------------------------------- #
def test_summary_contains_bounded_language(tmp_path):
    from experiments.run_auto_proposal_cic_pilot import write_combined_summary

    out_dir = tmp_path / "auto_proposal_pilot"
    out_dir.mkdir(parents=True)
    path = Path(write_combined_summary(out_dir))
    text = path.read_text().lower()
    # Required bounded / non-claim language.
    assert "open-world shortcut discovery" in text
    assert "replacement for the finalized sts report" in text
    assert "diagnostic" in text
    # No unbounded victory claims.
    for forbidden in ["universal robustness is", "solves shortcut discovery", "deployment-ready", "clinically validated"]:
        assert forbidden not in text


def test_does_not_write_final_report(tmp_path, monkeypatch):
    """Running the pilot into a temp dir must not touch results/final_report/."""

    from experiments import run_auto_proposal_cic_pilot as mod

    if not FINAL_REPORT.exists():
        pytest.skip("results/final_report not present in this checkout")

    def _snapshot(folder: Path):
        return {p.relative_to(folder).as_posix(): p.stat().st_mtime_ns
                for p in folder.rglob("*") if p.is_file()}

    before = _snapshot(FINAL_REPORT)
    _patch_clip(monkeypatch, mod)
    data_root = tmp_path / "coco"
    triage_dir = tmp_path / "triage"
    _make_coco_dataset(data_root, n=4)
    _make_triage(triage_dir, ids=list(range(4)))
    args = mod.build_parser().parse_args([
        "--results-dir", str(tmp_path / "results"),
        "--data-root", str(data_root),
        "--metadata-csv", str(data_root / "metadata.csv"),
        "--triage-dir", str(triage_dir),
        "--subset", "strict", "--max-examples", "4", "--max-candidates", "5",
    ])
    mod.run(args)
    assert _snapshot(FINAL_REPORT) == before


def test_no_raw_datasets_under_pilot_results():
    """The committed pilot results dir must not contain copied raw image datasets."""

    pilot_dir = REPO_ROOT / "results" / "auto_proposal_pilot"
    if not pilot_dir.exists():
        pytest.skip("pilot results not generated yet")
    forbidden_dirs = {"train2014", "val2014", "val2017", "images", "imgs", "raw", "wilds"}
    for p in pilot_dir.rglob("*"):
        assert p.name not in forbidden_dirs, f"raw dataset dir copied into results: {p}"
        if p.is_file():
            assert p.suffix.lower() not in {".jpg", ".jpeg"}, f"raw image copied into results: {p}"


# --------------------------------------------------------------------------- #
# Apples-to-apples COCO-Text sweep
# --------------------------------------------------------------------------- #
COCO_FULL = REPO_ROOT / "results" / "coco_text_cic_full"
COCO_TRIAGE = REPO_ROOT / "results" / "coco_text_cic_triage"
RECON_DOC = REPO_ROOT / "results" / "auto_proposal_pilot" / "coco_reconciliation.md"


def _patch_clip_sweep(monkeypatch, mod):
    monkeypatch.setattr(mod, "ClipZeroShotClassifier", _StubClassifier)
    status = lambda **kw: ClipStatus(
        available=True,
        backend="open_clip",
        model_name="ViT-B-32",
        pretrained_tag="laion2b_s34b_b79k",
        pretrained=True,
    )
    monkeypatch.setattr(mod, "check_clip_available", status)


def test_sweep_reconciliation_doc_exists_and_states_not_comparable():
    """Task 1: the reconciliation output must exist and reach a verdict."""

    assert RECON_DOC.exists(), "coco_reconciliation.md must be written under results/auto_proposal_pilot/"
    text = RECON_DOC.read_text()
    low = text.lower()
    assert "0.538" in text and "0.410" in text
    # It must state whether the pilot baseline is directly comparable.
    assert "not directly comparable" in low
    assert "apples-to-apples" in low


def test_sweep_a2a_match_helper():
    from experiments import run_coco_text_auto_proposal_sweep as mod

    assert mod._a2a_matches({"cic_repair_alias_accuracy": 0.5384615384615384}) is True
    assert mod._a2a_matches({"cic_repair_alias_accuracy": 0.41025641025641024}) is False
    assert mod._a2a_matches(None) is False
    # Finalized recipe constants are pinned (must match coco_text_cic_full config).
    assert mod.A2A_MAX_CANDIDATES == 48
    assert mod.A2A_GRID_SCALES == [0.18, 0.3, 0.45]


def test_sweep_promotion_logic():
    from experiments import run_coco_text_auto_proposal_sweep as mod

    base_strict = {"cic_repair_alias_accuracy": 0.50, "selected_text_overlap_rate": 0.10,
                   "coverage_ceiling_iou01": 0.30}
    base_dir = {"cic_repair_alias_accuracy": 0.40}
    # Family that clears criterion A (strict repair +0.05, no big directional drop).
    fam_strict = {"cic_repair_alias_accuracy": 0.57, "selected_text_overlap_rate": 0.10,
                  "coverage_ceiling_iou01": 0.30}
    fam_dir = {"cic_repair_alias_accuracy": 0.39}
    v = mod._promotion_for_family(base_strict, fam_strict, base_dir, fam_dir)
    assert v["criterion_A_strict_repair_+0.05"] is True
    assert v["family_promotable"] is True
    # Family that does not clear anything.
    flat = {"cic_repair_alias_accuracy": 0.50, "selected_text_overlap_rate": 0.10,
            "coverage_ceiling_iou01": 0.30}
    v2 = mod._promotion_for_family(base_strict, flat, base_dir, fam_dir)
    assert v2["family_promotable"] is False


def test_sweep_sam_promotion_logic():
    from experiments import run_coco_text_auto_proposal_sweep as mod

    base_strict = {"cic_repair_alias_accuracy": 0.50, "selected_text_overlap_rate": 0.10,
                   "coverage_ceiling_iou01": 0.30}
    base_dir = {"cic_repair_alias_accuracy": 0.40}

    # Criterion A: SAM strict repair beats a2a by >= +0.05.
    sam_a = {"cic_repair_alias_accuracy": 0.56, "selected_text_overlap_rate": 0.10,
             "coverage_ceiling_iou01": 0.30}
    va = mod._sam_promotion(base_strict, sam_a, base_dir, {"cic_repair_alias_accuracy": 0.40})
    assert va["criterion_A_sam_strict_repair_+0.05"] is True
    assert va["sam_promotable"] is True

    # Criterion B: SAM directional repair beats a2a by >= +0.05 (strict flat).
    vb = mod._sam_promotion(base_strict, base_strict, base_dir, {"cic_repair_alias_accuracy": 0.47})
    assert vb["criterion_B_sam_directional_repair_+0.05"] is True
    assert vb["sam_promotable"] is True

    # Criterion C: text overlap +0.15 with strict repair drop <= 0.05.
    sam_c = {"cic_repair_alias_accuracy": 0.48, "selected_text_overlap_rate": 0.30,
             "coverage_ceiling_iou01": 0.30}
    vc = mod._sam_promotion(base_strict, sam_c, base_dir, {"cic_repair_alias_accuracy": 0.40})
    assert vc["criterion_C_sam_text_overlap_+0.15"] is True
    assert vc["sam_promotable"] is True

    # Nothing clears: flat metrics => not promotable.
    flat = {"cic_repair_alias_accuracy": 0.50, "selected_text_overlap_rate": 0.10,
            "coverage_ceiling_iou01": 0.30}
    vf = mod._sam_promotion(base_strict, flat, base_dir, {"cic_repair_alias_accuracy": 0.40})
    assert vf["sam_promotable"] is False


def test_sweep_fast_sam_settings_resolution():
    from experiments import run_coco_text_auto_proposal_sweep as mod

    P = mod.build_parser()
    # --sam-fast => aggressive defaults; explicit flags still win.
    a = P.parse_args(["--include-sam", "--sam-fast"])
    assert mod._resolve_sam_points_per_side(a) == 8
    assert mod._resolve_sam_max_side(a) == 512
    k = mod._sam_kwargs(a)
    assert k["points_per_side"] == 8 and k["max_side"] == 512 and k["crop_n_layers"] == 0
    a2 = P.parse_args(["--include-sam", "--sam-fast", "--sam-points-per-side", "16", "--sam-max-side", "0"])
    assert mod._resolve_sam_points_per_side(a2) == 16 and mod._resolve_sam_max_side(a2) == 0
    # No --sam-fast => standard defaults (16, no resize).
    a3 = P.parse_args(["--include-sam"])
    assert mod._resolve_sam_points_per_side(a3) == 16 and mod._resolve_sam_max_side(a3) == 0
    # Cache wiring only when requested.
    assert mod._sam_kwargs(a3)["use_cache"] is False
    assert mod._sam_kwargs(P.parse_args(["--include-sam", "--cache-sam-proposals"]))["use_cache"] is True
    assert mod._sam_kwargs(P.parse_args(["--include-sam", "--resume"]))["use_cache"] is True


def test_sweep_all500_sam_requires_confirmation():
    """all_500 with SAM must require an explicit --all500 --confirm-slow-sam."""

    from experiments import run_coco_text_auto_proposal_sweep as mod

    P = mod.build_parser()
    # Guarded default: --include-sam alone => strict only.
    chosen, _ = mod.resolve_subsets_and_gating(P.parse_args(["--include-sam"]))
    assert chosen == ["strict"]
    # Asking for all_500 with SAM but without confirmation => dropped + gating note.
    chosen, gating = mod.resolve_subsets_and_gating(
        P.parse_args(["--include-sam", "--subset", "auto"])
    )
    assert "all_500" not in chosen
    assert any("all_500" in g and "confirm-slow-sam" in g for g in gating)
    # With the explicit confirmation flags => all_500 is allowed.
    chosen, gating = mod.resolve_subsets_and_gating(
        P.parse_args(["--include-sam", "--all500", "--confirm-slow-sam", "--subset", "strict,all_500"])
    )
    assert "all_500" in chosen and not gating


def test_sweep_timeout_writes_bounded_partial_summary(tmp_path):
    """A timed-out SAM run must write a bounded PARTIAL summary, not fail silently."""

    from experiments.run_coco_text_auto_proposal_sweep import write_sweep_summary

    out_dir = tmp_path / "auto_proposal_pilot"
    out_dir.mkdir(parents=True)
    metrics = {
        "experiment": "coco_text_auto_proposal_sweep",
        "status": "ok",
        "backend": "open_clip",
        "model_name": "ViT-B-32",
        "a2a_max_candidates": 48,
        "a2a_grid_scales": [0.18, 0.3, 0.45],
        "families_evaluated": ["existing_cic_baseline_a2a", "sam_boxes"],
        "subsets_chosen": ["strict"],
        "subset_gating": [],
        "reconciliation": {
            "finalized_strict_cic_repair": 0.538,
            "finalized_strict_matched_random_repair": 0.205,
            "a2a_reproduces_finalized": True,
            "pilot_baseline_strict_cic_repair": 0.410,
            "pilot_baseline_directly_comparable": False,
        },
        "sam": {
            "requested": True, "loaded": True, "skip_reason": "",
            "resolved_device": "cpu",
            "settings": {"model_type": "vit_b", "checkpoint_path": "models/sam/sam_vit_b_01ec64.pth",
                         "fast": True, "points_per_side": 8, "crop_n_layers": 0, "max_side": 512,
                         "pred_iou_thresh": 0.86, "stability_score_thresh": 0.9, "max_proposals": 48,
                         "min_area_frac": 0.002, "max_area_frac": 0.8, "min_side": 8, "dedupe_iou": 0.7},
            "images_completed": 5, "cache_hits": 0, "cache_misses": 5,
            "generation_seconds": 1801.2, "timed_out": True, "partial": True,
            "timeout_seconds": 1800.0, "strict_complete": False, "total_runtime_seconds": 1820.0,
        },
        "subsets": {},
        "promotion": {},
        "sam_promotion": None,
        "sam_promotable": False,
        "auto_proposal_promotable": False,
        "non_claims": [],
    }
    (out_dir / "coco_text_auto_proposal_sweep_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    path = Path(write_sweep_summary(out_dir))
    text = path.read_text()
    low = text.lower()
    assert "timed out" in low and "partial" in low
    assert "images completed: 5" in low
    assert "sam_promotable = false" in low or "sam_promotable=false" in low
    # Bounded language preserved even on the partial path.
    assert "not** open-world" in low or "not open-world" in low
    for forbidden in ["universal robustness is", "solves shortcut discovery", "deployment-ready"]:
        assert forbidden not in low


def test_sweep_end_to_end(tmp_path, monkeypatch):
    from experiments import run_coco_text_auto_proposal_sweep as mod

    _patch_clip_sweep(monkeypatch, mod)
    data_root = tmp_path / "coco"
    triage_dir = tmp_path / "triage"
    _make_coco_dataset(data_root, n=6)
    _make_triage(triage_dir, ids=list(range(6)))

    args = mod.build_parser().parse_args([
        "--results-dir", str(tmp_path / "results"),
        "--data-root", str(data_root),
        "--metadata-csv", str(data_root / "metadata.csv"),
        "--triage-dir", str(triage_dir),
        "--full-per-example", str(tmp_path / "missing_full.csv"),
        "--subset", "strict,directional",
        "--max-candidates", "6",
    ])
    metrics = mod.run(args)
    assert metrics["status"] == "ok"
    assert "auto_proposal_promotable" in metrics
    assert metrics["reconciliation"]["pilot_baseline_directly_comparable"] is False
    out_dir = tmp_path / "results" / "auto_proposal_pilot"
    for fname in ("coco_text_auto_proposal_sweep_metrics.json",
                  "coco_text_auto_proposal_sweep_table.csv",
                  "coco_text_auto_proposal_sweep_per_example.csv",
                  "full_coco_sweep_summary.md"):
        assert (out_dir / fname).exists(), fname
    loaded = json.loads((out_dir / "coco_text_auto_proposal_sweep_metrics.json").read_text())
    fams = loaded["subsets"]["strict"]["families"]
    assert mod.BASELINE_FAMILY in fams
    # Every required metric key is present for each family.
    for fam_metrics in fams.values():
        for key in ("cic_repair_alias_accuracy", "matched_random_repair_alias_accuracy",
                    "cic_random_gap", "cic_target_prob_improvement_rate",
                    "cic_text_distractor_decrease_rate", "selected_text_overlap_rate",
                    "selected_object_overlap_rate", "coverage_ceiling_iou01",
                    "coverage_ceiling_iou03", "median_rank_best_text_overlap",
                    "mean_selected_area_fraction"):
            assert key in fam_metrics, key
    # Paired wins/losses vs the a2a baseline are recorded for automated families.
    paired = loaded["subsets"]["strict"]["paired_vs_baseline"]
    for fam in fams:
        if fam == mod.BASELINE_FAMILY:
            continue
        assert fam in paired
        assert {"wins", "losses", "ties", "net"} <= set(paired[fam])


def test_sweep_skips_cleanly_without_data(tmp_path, monkeypatch):
    from experiments import run_coco_text_auto_proposal_sweep as mod

    _patch_clip_sweep(monkeypatch, mod)
    args = mod.build_parser().parse_args([
        "--results-dir", str(tmp_path / "results"),
        "--metadata-csv", str(tmp_path / "missing.csv"),
    ])
    metrics = mod.run(args)
    assert metrics["status"] == "skipped"
    assert metrics["auto_proposal_promotable"] is False


def test_sweep_summary_bounded_language(tmp_path):
    from experiments.run_coco_text_auto_proposal_sweep import write_sweep_summary

    out_dir = tmp_path / "auto_proposal_pilot"
    out_dir.mkdir(parents=True)
    path = Path(write_sweep_summary(out_dir))
    text = path.read_text().lower()
    assert "not** open-world" in text or "not open-world" in text
    assert "replacement for the finalized sts report" in text
    for forbidden in ["universal robustness is", "solves shortcut discovery",
                      "deployment-ready", "clinically validated", "open-world discovery of"]:
        assert forbidden not in text


def test_sweep_does_not_touch_protected_artifacts(tmp_path, monkeypatch):
    """Task 7: result artifacts outside results/auto_proposal_pilot/ are untouched."""

    from experiments import run_coco_text_auto_proposal_sweep as mod

    def _snapshot(folder: Path):
        if not folder.exists():
            return {}
        return {p.relative_to(folder).as_posix(): p.stat().st_mtime_ns
                for p in folder.rglob("*") if p.is_file()}

    protected = [FINAL_REPORT, COCO_FULL, COCO_TRIAGE]
    before = {f: _snapshot(f) for f in protected}

    _patch_clip_sweep(monkeypatch, mod)
    data_root = tmp_path / "coco"
    triage_dir = tmp_path / "triage"
    _make_coco_dataset(data_root, n=4)
    _make_triage(triage_dir, ids=list(range(4)))
    args = mod.build_parser().parse_args([
        "--results-dir", str(tmp_path / "results"),
        "--data-root", str(data_root),
        "--metadata-csv", str(data_root / "metadata.csv"),
        "--triage-dir", str(triage_dir),
        "--full-per-example", str(tmp_path / "missing_full.csv"),
        "--subset", "strict", "--max-candidates", "5",
    ])
    mod.run(args)
    for f in protected:
        assert _snapshot(f) == before[f], f"protected artifact changed: {f}"
