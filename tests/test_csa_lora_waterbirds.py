from __future__ import annotations

"""Tests for the pre-registered Waterbirds CSA manual-LoRA pilot.

These run fast and deterministically on CPU and NEVER load the real OpenCLIP
backbone or the real WILDS Waterbirds dataset: a tiny OpenCLIP-shaped toy model
(reused from ``test_manual_lora``) and a toy in-memory Waterbirds dataset
exercise the patcher, the CSA training loop, the group metrics, and the
pre-registered go/no-go logic. They verify: the config loads, a missing dataset
skips cleanly, missing GPU/MPS skips full LoRA unless forced, the cached fallback
is labelled correctly and can never set promising/strong, group + worst-group
metrics are correct, go/no-go returns null without worst-group gain and promising
only for true manual-LoRA, the output schema validates, the summary carries
bounded language, and no protected artifacts are touched.
"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from causal_reliability.training import waterbirds_csa as wb
from causal_reliability.training.csa_lora import (
    FrozenClassifier,
    ManualLoraModel,
)
from causal_reliability.training.manual_lora import apply_lora_to_openclip_visual
from causal_reliability.utils.config import load_config
from experiments import run_csa_lora_waterbirds as runner

# Toy OpenCLIP-shaped model (no real weights) reused from the sibling module.
from test_manual_lora import ToyCLIP  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "csa_lora_waterbirds.yaml"
FINAL_REPORT = REPO_ROOT / "results" / "final_report"
PROTECTED_FILES = [
    REPO_ROOT / "results" / "main_results_summary.json",
    REPO_ROOT / "results" / "main_results_table.csv",
    REPO_ROOT / "results" / "main_results_table.md",
    REPO_ROOT / "results" / "csa_lora_pilot" / "metrics.json",
    REPO_ROOT / "results" / "csa_lora_pilot" / "manual_lora_metrics.json",
]


# --------------------------------------------------------------------------- #
# Toy model + dataset helpers (k = 2 for the binary Waterbirds head)
# --------------------------------------------------------------------------- #
def _toy_wb_model(dim=16, seed=0, patch_last=2):
    torch.manual_seed(seed)
    toy = ToyCLIP(dim=dim)
    info = apply_lora_to_openclip_visual(toy, rank=4, alpha=8.0, max_layers=patch_last)
    g = torch.Generator().manual_seed(seed + 1)
    text = F.normalize(torch.randn(2, dim, generator=g), dim=-1)
    model = ManualLoraModel(
        toy, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), logit_scale=10.0,
        text_features=text, transfer_text_features=None, encode_batch_size=8, device="cpu",
    )
    return model, info


def _toy_image_split(n=8, img=8, m=4, seed=0) -> wb.WBImageSplit:
    g = torch.Generator().manual_seed(seed)
    images = torch.randint(0, 256, (n, 3, img, img), generator=g, dtype=torch.uint8)
    labels = torch.tensor([i % 2 for i in range(n)], dtype=torch.long)
    places = torch.tensor([(i // 2) % 2 for i in range(n)], dtype=torch.long)
    groups = labels * 2 + places
    candidates = torch.randint(0, 256, (n, m, 3, img, img), generator=g, dtype=torch.uint8)
    return wb.WBImageSplit(images=images, labels=labels, places=places, groups=groups, candidates=candidates)


def _toy_dataset(n=8, img=8, seed=0) -> wb.WaterbirdsDataset:
    return wb.WaterbirdsDataset(
        train=_toy_image_split(n=n, img=img, seed=seed),
        eval_splits={"val": _toy_image_split(n=n, img=img, seed=seed + 5),
                     "test": _toy_image_split(n=n, img=img, seed=seed + 9)},
        primary_eval_split="test",
        num_classes=2, image_size=img,
        intervention_summary=wb.describe_interventions(wb.WBInterventionConfig()),
    )


def _toy_embedding_dataset(n=8, d=16, m=4, seed=0) -> wb.WBEmbeddingDataset:
    g = torch.Generator().manual_seed(seed)

    def split(s):
        gg = torch.Generator().manual_seed(s)
        labels = torch.tensor([i % 2 for i in range(n)], dtype=torch.long)
        places = torch.tensor([(i // 2) % 2 for i in range(n)], dtype=torch.long)
        return wb.WBEmbeddingSplit(
            observed=torch.randn(n, d, generator=gg),
            candidates=torch.randn(n, m, d, generator=gg),
            labels=labels, places=places, groups=labels * 2 + places,
        )

    text = F.normalize(torch.randn(2, d, generator=g), dim=-1)
    clf = FrozenClassifier("clip_text", 10.0, text_features=text)
    return wb.WBEmbeddingDataset(
        train=split(seed), eval_splits={"val": split(seed + 5), "test": split(seed + 9)},
        primary_eval_split="test", classifier=clf, embed_dim=d, num_classes=2,
        intervention_summary=wb.describe_interventions(wb.WBInterventionConfig()),
    )


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_config_loads():
    raw = load_config(CONFIG_PATH)
    cfg = wb.WBPilotConfig.from_dict(raw)
    assert cfg.model.model_name == "ViT-B-32"
    assert cfg.model.pretrained_tag == "laion2b_s34b_b79k"
    assert cfg.manual_lora.rank == 4
    assert cfg.manual_lora.alpha == 8.0
    assert cfg.manual_lora.target_last_blocks == 2
    assert cfg.manual_lora.lr == pytest.approx(2e-4)
    assert cfg.max_avg_acc_drop == 0.03
    assert cfg.min_worst_group_gain == 0.05
    assert cfg.min_instability_drop_rel == 0.20
    assert cfg.strong_worst_group_gain == 0.08
    assert cfg.training_modes() == ["frozen", "plain_ft", "csa"]  # cf_aug / dro off


# --------------------------------------------------------------------------- #
# Dataset availability gate
# --------------------------------------------------------------------------- #
def test_missing_dataset_reports_unavailable(tmp_path):
    cfg = wb.WBPilotConfig.from_dict({"data": {"wilds_root": str(tmp_path / "nope")}})
    avail = wb.check_waterbirds_available(cfg)
    assert avail["waterbirds_available"] is False
    assert "unavailable" in avail["reason"].lower()


def test_missing_dataset_skips_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner.wb, "check_waterbirds_available",
        lambda cfg, download=False: {"waterbirds_available": False, "root": "", "reason": "Waterbirds dataset unavailable; skipping cleanly (test)"},
    )
    args = runner.build_parser().parse_args(
        ["--config", str(CONFIG_PATH), "--results-dir", str(tmp_path / "results"), "--device", "cpu"]
    )
    metrics = runner.run(args)
    assert metrics["status"] == "skipped_no_dataset"
    assert metrics["waterbirds_available"] is False
    assert metrics["waterbirds_csa_promising"] is False
    assert metrics["waterbirds_csa_strong"] is False
    assert metrics["waterbirds_csa_null"] is True
    out_dir = tmp_path / "results" / "csa_lora_pilot" / "waterbirds"
    assert (out_dir / "metrics.json").exists()
    text = (out_dir / "summary.md").read_text().lower()
    assert "skipped cleanly" in text


# --------------------------------------------------------------------------- #
# Device / mode decision
# --------------------------------------------------------------------------- #
def test_cpu_skips_full_lora_unless_forced():
    cfg = wb.WBPilotConfig.from_dict({})
    assert wb.decide_mode(cfg, "cpu") == "cached_embedding_adapter"
    cfg.force_cpu_lora = True
    assert wb.decide_mode(cfg, "cpu") == "manual_lora_visual"
    assert wb.decide_mode(cfg, "cuda") == "manual_lora_visual"
    assert wb.decide_mode(cfg, "mps") == "manual_lora_visual"
    assert wb.is_accelerator("cuda") and wb.is_accelerator("mps")
    assert not wb.is_accelerator("cpu")


# --------------------------------------------------------------------------- #
# Interventions (finite, label-free, deterministic)
# --------------------------------------------------------------------------- #
def test_interventions_shape_range_and_determinism():
    icfg = wb.WBInterventionConfig(n_candidates=4)
    imgs = torch.randint(0, 256, (3, 3, 16, 16), dtype=torch.uint8)
    a = wb.build_candidate_interventions(imgs, icfg)
    b = wb.build_candidate_interventions(imgs, icfg)
    assert a.shape == (3, 4, 3, 16, 16)
    assert a.dtype == torch.uint8
    assert torch.equal(a, b)  # deterministic given the intervention seed
    summary = wb.describe_interventions(icfg)
    assert summary["n_candidates"] == 4
    assert len(summary["candidate_types"]) == 4
    assert "not verified ground-truth" in summary["honesty_note"].lower()


# --------------------------------------------------------------------------- #
# Group / worst-group metrics
# --------------------------------------------------------------------------- #
def test_group_metric_computation_toy():
    # 4 examples, one per group; predictions correct for all but waterbird_on_land.
    labels = torch.tensor([0, 0, 1, 1])
    places = torch.tensor([0, 1, 0, 1])
    groups = labels * 2 + places  # [0,1,2,3]
    logits = torch.zeros(4, 2)
    # make argmax match label except index 2 (waterbird_on_land) wrong
    logits[0, 0] = 1.0
    logits[1, 0] = 1.0
    logits[2, 0] = 1.0  # wrong: true label is 1
    logits[3, 1] = 1.0
    out = wb.compute_group_metrics(logits, labels, groups)
    assert out["average_accuracy"] == pytest.approx(0.75)
    assert out["worst_group_accuracy"] == pytest.approx(0.0)
    assert out["group_accuracies"]["waterbird_on_land"] == pytest.approx(0.0)
    assert out["group_accuracies"]["landbird_on_land"] == pytest.approx(1.0)
    assert out["group_counts"]["landbird_on_water"] == 1


def test_worst_group_metric_correct_and_missing_groups_excluded():
    # Only two groups present; worst-group = min over present groups.
    labels = torch.tensor([0, 0, 0, 0])
    places = torch.tensor([0, 0, 1, 1])  # groups 0 and 1 only
    groups = labels * 2 + places
    logits = torch.zeros(4, 2)
    logits[0, 0] = 1.0  # correct
    logits[1, 0] = 1.0  # correct
    logits[2, 1] = 1.0  # wrong
    logits[3, 0] = 1.0  # correct
    out = wb.compute_group_metrics(logits, labels, groups)
    assert out["group_accuracies"]["landbird_on_land"] == pytest.approx(1.0)
    assert out["group_accuracies"]["landbird_on_water"] == pytest.approx(0.5)
    assert out["group_accuracies"]["waterbird_on_land"] is None  # absent
    assert out["worst_group_accuracy"] == pytest.approx(0.5)  # missing groups excluded


# --------------------------------------------------------------------------- #
# Go / no-go logic
# --------------------------------------------------------------------------- #
def _metric(avg, worst, instab):
    return {"average_accuracy": avg, "worst_group_accuracy": worst, "counterfactual_instability": instab}


def test_go_no_go_null_without_worst_group_improvement():
    cfg = wb.WBPilotConfig.from_dict({})
    plain = _metric(0.80, 0.50, 1.0)
    csa = _metric(0.80, 0.51, 0.5)  # worst-group gain only +0.01 (< 0.05)
    go = wb.compute_go_no_go(plain, csa, real_lora_used=True, cfg=cfg)
    assert go["worst_group_gain_ok"] is False
    assert go["waterbirds_csa_promising"] is False
    assert go["waterbirds_csa_null"] is True
    # instability improved but robustness did not -> flagged
    assert go["instability_improves_but_robustness_does_not"] is True


def test_go_no_go_promising_only_for_true_lora():
    cfg = wb.WBPilotConfig.from_dict({})
    plain = _metric(0.80, 0.50, 1.0)
    csa = _metric(0.79, 0.58, 0.6)  # avg drop 0.01<=0.03; wg gain +0.08>=0.05; instab drop 40%>=20%
    go_real = wb.compute_go_no_go(plain, csa, real_lora_used=True, cfg=cfg)
    assert go_real["waterbirds_csa_promising"] is True
    assert go_real["waterbirds_csa_null"] is False
    # The identical metrics from a cached fallback can NEVER be promising.
    go_fallback = wb.compute_go_no_go(plain, csa, real_lora_used=False, cfg=cfg)
    assert go_fallback["waterbirds_csa_promising"] is False
    assert go_fallback["waterbirds_csa_null"] is True


def test_go_no_go_null_when_avg_accuracy_drops_too_much():
    cfg = wb.WBPilotConfig.from_dict({})
    plain = _metric(0.80, 0.50, 1.0)
    csa = _metric(0.74, 0.60, 0.5)  # avg drop 0.06 > 0.03 even though wg gain +0.10
    go = wb.compute_go_no_go(plain, csa, real_lora_used=True, cfg=cfg)
    assert go["avg_drop_ok"] is False
    assert go["waterbirds_csa_promising"] is False
    assert go["waterbirds_csa_null"] is True


def test_strong_flag_requires_three_seeds_and_real_lora():
    cfg = wb.WBPilotConfig.from_dict({})
    one = [{"seed": 0, "plain": _metric(0.8, 0.5, 1.0), "csa": _metric(0.8, 0.62, 0.5)}]
    s1 = wb.compute_strong_flag(one, real_lora_used=True, cfg=cfg)
    assert s1["waterbirds_csa_strong"] is False  # single seed cannot be strong

    three = [
        {"seed": 0, "plain": _metric(0.80, 0.50, 1.0), "csa": _metric(0.80, 0.60, 0.5)},
        {"seed": 1, "plain": _metric(0.80, 0.52, 1.0), "csa": _metric(0.80, 0.61, 0.5)},
        {"seed": 2, "plain": _metric(0.80, 0.49, 1.0), "csa": _metric(0.80, 0.59, 0.5)},
    ]
    s3 = wb.compute_strong_flag(three, real_lora_used=True, cfg=cfg)
    assert s3["waterbirds_csa_strong"] is True
    assert s3["mean_paired_worst_group_gain"] >= 0.08
    # fallback (real_lora_used=False) can never be strong
    s3_fb = wb.compute_strong_flag(three, real_lora_used=False, cfg=cfg)
    assert s3_fb["waterbirds_csa_strong"] is False


# --------------------------------------------------------------------------- #
# manual_lora_visual end-to-end (toy model + toy dataset; forced on CPU)
# --------------------------------------------------------------------------- #
def _run_manual(tmp_path, monkeypatch, extra=None):
    model, info = _toy_wb_model()

    monkeypatch.setattr(
        runner.wb, "check_waterbirds_available",
        lambda cfg, download=False: {"waterbirds_available": True, "root": str(tmp_path / "wb"), "reason": "available"},
    )
    monkeypatch.setattr(
        runner.wb, "build_real_lora_model",
        lambda cfg, device: {"ok": True, "model": model, "patch_info": info,
                             "backend": "open_clip", "model_name": "ToyViT-B-32", "pretrained_tag": "toy"},
    )
    monkeypatch.setattr(runner.wb, "load_waterbirds_dataset", lambda cfg, avail: _toy_dataset(n=8, img=8, seed=cfg.seed))

    args = runner.build_parser().parse_args(
        ["--config", str(CONFIG_PATH), "--results-dir", str(tmp_path / "results"),
         "--device", "cpu", "--force-cpu-lora", "--epochs", "1",
         "--max-train-examples", "8", "--max-eval-examples", "8", *(extra or [])]
    )
    return runner.run(args)


def test_manual_lora_end_to_end_schema(tmp_path, monkeypatch):
    metrics = _run_manual(tmp_path, monkeypatch)
    assert metrics["status"] == "ok"
    assert metrics["mode"] == "manual_lora_visual"
    assert metrics["real_openclip_loaded"] is True
    assert metrics["real_lora_ran"] is True
    assert metrics["fallback_ran"] is False
    assert "no peft" in metrics["lora_library"].lower()
    assert metrics["num_patched_modules"] == 6
    assert metrics["trainable_param_count"] > 0
    assert metrics["baselines"] == ["frozen", "plain_ft", "csa"]
    for mode in ("frozen", "plain_ft", "csa"):
        m = metrics["modes"][mode]
        for key in ("average_accuracy", "worst_group_accuracy", "group_accuracies", "counterfactual_instability"):
            assert key in m
        assert set(m["group_accuracies"].keys()) == set(wb.GROUP_NAMES)
    assert isinstance(metrics["waterbirds_csa_promising"], bool)
    assert isinstance(metrics["waterbirds_csa_strong"], bool)
    assert isinstance(metrics["waterbirds_csa_null"], bool)
    assert metrics["single_seed_pilot"] is True
    assert metrics["waterbirds_csa_strong"] is False  # one seed

    out_dir = tmp_path / "results" / "csa_lora_pilot" / "waterbirds"
    for name in ("metrics.json", "table.csv", "group_table.csv", "summary.md"):
        assert (out_dir / name).exists()
    loaded = json.loads((out_dir / "metrics.json").read_text())
    assert loaded["pilot"] == "csa_lora_waterbirds"
    assert loaded["go_no_go"]["thresholds"]["min_worst_group_gain"] == 0.05
    table = (out_dir / "table.csv").read_text().strip().splitlines()
    assert len(table) == 1 + 3  # header + 3 modes


def test_manual_lora_summary_bounded_language(tmp_path, monkeypatch):
    _run_manual(tmp_path, monkeypatch)
    text = (tmp_path / "results" / "csa_lora_pilot" / "waterbirds" / "summary.md").read_text().lower()
    assert "not** universal robustness" in text or "not universal robustness" in text
    assert "open-world discovery" in text
    assert "rlhf/dpo replacement" in text
    assert "deployment validation" in text
    assert "replacement for the finalized sts report" in text
    assert "worst-group" in text
    assert "trainable" in text and "patched module names" in text
    assert "finite diagnostic intervention" in text
    assert "not verified" in text or "not** verified" in text
    for forbidden in ["universal robustness is achieved", "solves shortcut", "deployment-ready", "replaces rlhf"]:
        assert forbidden not in text


def test_manual_lora_multi_seed_writes_per_seed(tmp_path, monkeypatch):
    metrics = _run_manual(tmp_path, monkeypatch, extra=["--seeds", "0,1,2"])
    assert metrics["seeds"] == [0, 1, 2]
    assert metrics["single_seed_pilot"] is False
    assert len(metrics["per_seed"]) == 3
    out_dir = tmp_path / "results" / "csa_lora_pilot" / "waterbirds"
    assert (out_dir / "per_seed_metrics.csv").exists()


# --------------------------------------------------------------------------- #
# cached_embedding_adapter fallback (cannot set promising/strong)
# --------------------------------------------------------------------------- #
def _run_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner.wb, "check_waterbirds_available",
        lambda cfg, download=False: {"waterbirds_available": True, "root": str(tmp_path / "wb"), "reason": "available"},
    )
    monkeypatch.setattr(runner.wb, "load_waterbirds_dataset", lambda cfg, avail: _toy_dataset(n=8, img=8, seed=cfg.seed))
    monkeypatch.setattr(
        runner.wb, "build_cached_embedding_dataset",
        lambda cfg, dataset, device: {"ok": True, "dataset": _toy_embedding_dataset(n=8, d=16, seed=cfg.seed),
                                      "backend": "open_clip", "model_name": "ToyViT-B-32", "pretrained_tag": "toy",
                                      "from_cache": False},
    )
    args = runner.build_parser().parse_args(
        ["--config", str(CONFIG_PATH), "--results-dir", str(tmp_path / "results"),
         "--device", "cpu", "--epochs", "1", "--max-train-examples", "8", "--max-eval-examples", "8"]
    )
    return runner.run(args)


def test_fallback_labeled_and_cannot_be_promising(tmp_path, monkeypatch):
    metrics = _run_fallback(tmp_path, monkeypatch)
    assert metrics["status"] == "ok_fallback"
    assert metrics["mode"] == "cached_embedding_adapter"
    assert metrics["real_lora_ran"] is False
    assert metrics["fallback_ran"] is True
    assert metrics["actual_lora_used"] is False
    assert "not lora" in metrics["trainable_module_type"].lower()
    # Hard guarantees regardless of the underlying numbers.
    assert metrics["waterbirds_csa_promising"] is False
    assert metrics["waterbirds_csa_strong"] is False
    assert metrics["waterbirds_csa_null"] is True
    text = (tmp_path / "results" / "csa_lora_pilot" / "waterbirds" / "summary.md").read_text().lower()
    assert "cached-embedding fallback" in text
    assert "can\nnever" in text or "can never" in text or "never set" in text


# --------------------------------------------------------------------------- #
# Protected artifacts
# --------------------------------------------------------------------------- #
def _snapshot(folder: Path):
    if not folder.exists():
        return {}
    return {p.relative_to(folder).as_posix(): p.stat().st_mtime_ns for p in folder.rglob("*") if p.is_file()}


def test_does_not_write_final_report(tmp_path, monkeypatch):
    if not FINAL_REPORT.exists():
        pytest.skip("results/final_report not present in this checkout")
    before = _snapshot(FINAL_REPORT)
    _run_manual(tmp_path, monkeypatch)
    assert _snapshot(FINAL_REPORT) == before


def test_does_not_modify_existing_result_artifacts(tmp_path, monkeypatch):
    before = {f: (f.stat().st_mtime_ns if f.exists() else None) for f in PROTECTED_FILES}
    _run_manual(tmp_path, monkeypatch)
    _run_fallback(tmp_path, monkeypatch)
    for f in PROTECTED_FILES:
        after = f.stat().st_mtime_ns if f.exists() else None
        assert after == before[f], f"protected artifact changed: {f}"


def test_writes_only_under_waterbirds_subdir(tmp_path, monkeypatch):
    _run_manual(tmp_path, monkeypatch)
    results_root = tmp_path / "results"
    written = {p.parent.name for p in results_root.rglob("*") if p.is_file()}
    assert written == {"waterbirds"}
