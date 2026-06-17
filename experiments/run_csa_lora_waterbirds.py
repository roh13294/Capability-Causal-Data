from __future__ import annotations

"""Pre-registered Waterbirds CSA manual-LoRA pilot runner.

Experiment name: ``csa_lora_waterbirds``.

Scientific question
-------------------
Does Counterfactual Stability Alignment (CSA), applied as a small manual-LoRA
adaptation of a real OpenCLIP visual tower, improve **worst-group robustness** on
the real WILDS Waterbirds dataset **without using group labels for CSA
training**? This is one bounded experiment; positive, null, or negative results
are all reported honestly.

What this is / is not
----------------------
NOT universal robustness, NOT open-world shortcut discovery, NOT an RLHF/DPO
replacement, NOT deployment validation, and NOT a replacement for the finalized
STS report.

Honesty / scope guarantees
--------------------------
* Writes ONLY under ``results/csa_lora_pilot/waterbirds/``. Never touches
  ``paper/main.tex``, the finalized STS report, ``results/final_report/``, or any
  existing benchmark result JSON/CSV.
* Group labels are used ONLY for evaluation and the optional (off-by-default,
  clearly-marked) Group DRO baseline — never by the CSA objective.
* CSA interventions are finite diagnostic perturbations, NOT verified Waterbirds
  causal masks.
* Real ``manual_lora_visual`` needs CUDA/MPS + a loadable OpenCLIP backbone. With
  neither it skips full LoRA cleanly and runs a labelled ``cached_embedding_adapter``
  fallback (diagnostic only; can never set promising/strong true). If the dataset
  is missing it skips cleanly with ``waterbirds_available=false``.
* No checkpoints, datasets, image caches, or WILDS downloads are committed.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_reliability.training import waterbirds_csa as wb
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir, write_csv


RESULTS_SUBDIR = Path("csa_lora_pilot") / "waterbirds"

NON_CLAIMS = [
    "This is a bounded Waterbirds CSA-LoRA pilot, NOT universal robustness.",
    "This is NOT open-world shortcut discovery.",
    "This is NOT an RLHF/DPO replacement.",
    "This is NOT deployment validation.",
    "This is NOT a replacement for the finalized STS report.",
]

MODE_LABELS = {
    "frozen": "frozen OpenCLIP zero-shot / prompt classifier (no adaptation)",
    "plain_ft": "plain manual-LoRA fine-tuning (task loss only)",
    "cf_aug": "counterfactual-augmentation manual-LoRA (optional)",
    "csa": "CSA manual-LoRA (task + stability + CIC + preservation)",
    "group_dro": "Group DRO baseline (GROUP-LABEL-SUPERVISED; optional)",
}


# --------------------------------------------------------------------------- #
# CLI / overrides
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/csa_lora_waterbirds.yaml")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--seeds", type=str, default=None, help="comma-separated seeds (e.g. 0 or 0,1,2)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="auto | cpu | cuda | mps")
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-eval-examples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--target-last-blocks", type=int, default=None)
    p.add_argument("--lambda-stability", type=float, default=None)
    p.add_argument("--lambda-cic", type=float, default=None)
    p.add_argument("--lambda-preserve", type=float, default=None)
    p.add_argument("--enable-cf-aug", action="store_true", help="add the counterfactual-augmentation baseline")
    p.add_argument("--group-dro", action="store_true", help="add the (group-label-supervised) Group DRO baseline")
    p.add_argument("--force-cpu-lora", action="store_true", help="permit (slow) manual-LoRA on CPU for a smoke run")
    p.add_argument("--allow-download", action="store_true", help="permit a one-time OpenCLIP weight download")
    p.add_argument("--download", action="store_true", help="permit a one-time WILDS Waterbirds dataset download")
    p.add_argument("--no-cache", action="store_true", help="disable the embedding cache")
    return p


def _apply_overrides(args: argparse.Namespace, cfg: wb.WBPilotConfig) -> None:
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.device is not None:
        cfg.model.device = str(args.device)
    if args.max_train_examples is not None:
        cfg.data.max_train_examples = int(args.max_train_examples)
    if args.max_eval_examples is not None:
        cfg.data.max_eval_examples = int(args.max_eval_examples)
    if args.epochs is not None:
        cfg.manual_lora.epochs = int(args.epochs)
        cfg.fallback.epochs = int(args.epochs)
    if args.rank is not None:
        cfg.manual_lora.rank = int(args.rank)
    if args.alpha is not None:
        cfg.manual_lora.alpha = float(args.alpha)
    if args.target_last_blocks is not None:
        cfg.manual_lora.target_last_blocks = int(args.target_last_blocks)
    if args.lambda_stability is not None:
        cfg.csa.lambda_stability = float(args.lambda_stability)
    if args.lambda_cic is not None:
        cfg.csa.lambda_cic = float(args.lambda_cic)
    if args.lambda_preserve is not None:
        cfg.csa.lambda_preserve = float(args.lambda_preserve)
    if args.enable_cf_aug:
        cfg.enable_cf_aug = True
    if args.group_dro:
        cfg.group_dro.enabled = True
    if args.force_cpu_lora:
        cfg.force_cpu_lora = True
    if args.allow_download:
        cfg.model.allow_download = True
    if args.download:
        cfg.data.download = True
    if args.no_cache:
        cfg.use_embedding_cache = False


def _parse_seeds(args: argparse.Namespace, cfg: wb.WBPilotConfig) -> list[int]:
    raw = getattr(args, "seeds", None)
    if not raw:
        return [cfg.seed]
    return [int(s.strip()) for s in str(raw).split(",") if s.strip() != ""]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> dict[str, Any]:
    raw = load_config(args.config) if args.config else {}
    cfg = wb.WBPilotConfig.from_dict(raw)
    _apply_overrides(args, cfg)

    out_dir = ensure_dir(Path(args.results_dir) / RESULTS_SUBDIR)
    cfg.cache_dir = str(out_dir / "cache")

    device = wb.detect_device(cfg.model.device)
    seeds = _parse_seeds(args, cfg)
    single_seed_pilot = len(seeds) == 1

    base_meta = {
        "pilot": "csa_lora_waterbirds",
        "device": device,
        "cuda_available": bool(__import__("torch").cuda.is_available()),
        "mps_available": _mps_available(),
        "seeds": seeds,
        "single_seed_pilot": single_seed_pilot,
        "training_modes": cfg.training_modes(),
        "non_claims": NON_CLAIMS,
    }

    # 1) Dataset availability gate.
    avail = wb.check_waterbirds_available(cfg, download=cfg.data.download)
    base_meta["waterbirds_available"] = bool(avail["waterbirds_available"])
    base_meta["dataset_root"] = avail.get("root", "")
    if not avail["waterbirds_available"]:
        return _write_skip(out_dir, base_meta, reason=avail["reason"],
                           real_openclip_loaded=False, real_lora_ran=False,
                           fallback_ran=False, mode="skipped_no_dataset")

    # 2) Mode decision (real LoRA only on accelerator or forced).
    mode = wb.decide_mode(cfg, device)
    base_meta["intervention_summary"] = wb.describe_interventions(cfg.interventions)

    if mode == "manual_lora_visual":
        return run_manual_lora(args, cfg, out_dir, device, seeds, base_meta, avail)
    return run_fallback(args, cfg, out_dir, device, seeds, base_meta, avail)


def _mps_available() -> bool:
    import torch

    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def run_manual_lora(args, cfg, out_dir, device, seeds, base_meta, avail) -> dict[str, Any]:
    t0 = time.time()
    built = wb.build_real_lora_model(cfg, device)
    t_model = time.time() - t0

    if not built.get("ok"):
        # No OpenCLIP -> cannot patch a visual tower. Skip full LoRA cleanly.
        meta = dict(base_meta)
        meta.update({
            "real_openclip_loaded": False,
            "real_lora_ran": False,
            "fallback_ran": False,
        })
        return _write_skip(out_dir, meta, reason=built.get("reason", "OpenCLIP unavailable"),
                           real_openclip_loaded=False, real_lora_ran=False,
                           fallback_ran=False, mode="skipped_no_openclip")

    model = built["model"]
    patch_info = built["patch_info"]
    lora_param_count = int(patch_info["trainable_param_count"])

    t1 = time.time()
    per_seed: list[dict[str, Any]] = []
    primary_runs: list[dict[str, Any]] = []
    for seed in seeds:
        cfg.seed = int(seed)
        dataset = wb.load_waterbirds_dataset(cfg, avail)
        seed_result = _run_one_seed_manual(model, dataset, cfg)
        per_seed.append(seed_result)
        primary_runs.append({"seed": seed,
                             "plain": seed_result["primary"]["plain_ft"],
                             "csa": seed_result["primary"]["csa"]})
    train_time = time.time() - t1

    primary = per_seed[0]
    go = wb.compute_go_no_go(
        primary["primary"]["plain_ft"], primary["primary"]["csa"],
        real_lora_used=True, cfg=cfg,
    )
    strong = wb.compute_strong_flag(primary_runs, real_lora_used=True, cfg=cfg)

    metrics = dict(base_meta)
    metrics.update({
        "mode": "manual_lora_visual",
        "status": "ok",
        "real_openclip_loaded": True,
        "real_lora_ran": True,
        "fallback_ran": False,
        "actual_lora_used": True,
        "lora_library": "internal_pytorch_manual_lora (LoRALinear; NO PEFT)",
        "backend": built["backend"],
        "model_name": built["model_name"],
        "pretrained_tag": built["pretrained_tag"],
        "patched_modules": patch_info["patched_modules"],
        "num_patched_modules": patch_info["num_patched_modules"],
        "patched_block_indices": patch_info["patched_block_indices"],
        "num_visual_blocks": patch_info["num_visual_blocks"],
        "target_modules": patch_info["target_modules"],
        "trainable_param_count": lora_param_count,
        "lora_rank": cfg.manual_lora.rank,
        "lora_alpha": cfg.manual_lora.alpha,
        "lora_dropout": cfg.manual_lora.dropout,
        "target_last_blocks": cfg.manual_lora.target_last_blocks,
        "num_classes": len(wb.WATERBIRD_CLASSES),
        "epochs": cfg.manual_lora.epochs,
        "primary_eval_split": primary["primary_eval_split"],
        "eval_splits_present": list(primary["eval_split_names"]),
        "dataset_sizes": primary["dataset_sizes"],
        "timing_sec": {"model_build": round(t_model, 2), "training_and_eval": round(train_time, 2)},
        "config": {
            "seed_primary": seeds[0],
            "shortcut_corr": "natural (real Waterbirds)",
            "lr": cfg.manual_lora.lr,
            "batch_size": cfg.manual_lora.batch_size,
            "encode_batch_size": cfg.manual_lora.encode_batch_size,
            "train_text_head": cfg.manual_lora.train_text,
        },
        "baselines": cfg.training_modes(),
        "modes": primary["primary"],
        "modes_by_split": primary["by_split"],
        "go_no_go": go,
        "strong": strong,
        "waterbirds_csa_promising": bool(go["waterbirds_csa_promising"]),
        "waterbirds_csa_strong": bool(strong["waterbirds_csa_strong"]),
        "waterbirds_csa_null": bool(go["waterbirds_csa_null"]),
        "per_seed": [
            {"seed": r["seed"], "primary": r["primary"]} for r in per_seed
        ],
    })

    paths = write_outputs(out_dir, metrics)
    _print_summary(metrics, paths)
    return metrics


def _run_one_seed_manual(model, dataset, cfg) -> dict[str, Any]:
    primary_split = dataset.primary_eval_split
    by_split: dict[str, dict[str, Any]] = {name: {} for name in dataset.eval_splits}
    primary: dict[str, Any] = {}
    for mode in cfg.training_modes():
        wb.train_manual_lora_mode(model, dataset, mode, cfg)
        for name, split in dataset.eval_splits.items():
            by_split[name][mode] = wb.evaluate_manual_lora(model, split)
        primary[mode] = by_split.get(primary_split, {}).get(mode, {})
    sizes = {"train": len(dataset.train)}
    sizes.update({name: len(s) for name, s in dataset.eval_splits.items()})
    return {
        "seed": cfg.seed,
        "primary": primary,
        "by_split": by_split,
        "primary_eval_split": primary_split,
        "eval_split_names": list(dataset.eval_splits.keys()),
        "dataset_sizes": sizes,
    }


def run_fallback(args, cfg, out_dir, device, seeds, base_meta, avail) -> dict[str, Any]:
    # Build images once (per seed) then encode FROZEN embeddings.
    t0 = time.time()
    cfg.seed = int(seeds[0])
    dataset = wb.load_waterbirds_dataset(cfg, avail)
    built = wb.build_cached_embedding_dataset(cfg, dataset, device)
    t_enc = time.time() - t0

    if not built.get("ok"):
        meta = dict(base_meta)
        meta.update({"real_openclip_loaded": False, "real_lora_ran": False, "fallback_ran": False})
        return _write_skip(out_dir, meta, reason=built.get("reason", "OpenCLIP unavailable"),
                           real_openclip_loaded=False, real_lora_ran=False,
                           fallback_ran=False, mode="skipped_no_openclip")

    emb_ds = built["dataset"]
    t1 = time.time()
    primary_split = emb_ds.primary_eval_split
    by_split: dict[str, dict[str, Any]] = {name: {} for name in emb_ds.eval_splits}
    primary: dict[str, Any] = {}
    trainable_params: dict[str, int] = {}
    for mode in cfg.training_modes():
        adapter = wb.train_fallback_mode(emb_ds, mode, cfg)
        trainable_params[mode] = (
            0 if adapter is None else int(sum(p.numel() for p in adapter.parameters() if p.requires_grad))
        )
        for name, split in emb_ds.eval_splits.items():
            by_split[name][mode] = wb.evaluate_fallback(adapter, emb_ds, split)
        primary[mode] = by_split.get(primary_split, {}).get(mode, {})
    train_time = time.time() - t1

    # Fallback can NEVER set promising/strong. go/no-go uses real_lora_used=False.
    go = wb.compute_go_no_go(primary["plain_ft"], primary["csa"], real_lora_used=False, cfg=cfg)
    strong = wb.compute_strong_flag(
        [{"seed": seeds[0], "plain": primary["plain_ft"], "csa": primary["csa"]}],
        real_lora_used=False, cfg=cfg,
    )

    sizes = {"train": len(emb_ds.train)}
    sizes.update({name: len(s) for name, s in emb_ds.eval_splits.items()})

    metrics = dict(base_meta)
    metrics.update({
        "mode": "cached_embedding_adapter",
        "status": "ok_fallback",
        "real_openclip_loaded": True,
        "real_lora_ran": False,
        "fallback_ran": True,
        "actual_lora_used": False,
        "trainable_module_type": (
            "cached_embedding_adapter: lightweight residual head over FROZEN "
            "OpenCLIP image embeddings (NOT LoRA; diagnostic only)"
        ),
        "fallback_note": (
            "No CUDA/MPS available (and --force-cpu-lora not set): full manual-LoRA "
            "skipped cleanly. Only fallback/smoke diagnostics ran. This mode can "
            "NEVER set waterbirds_csa_promising or waterbirds_csa_strong to true."
        ),
        "backend": built["backend"],
        "model_name": built["model_name"],
        "pretrained_tag": built["pretrained_tag"],
        "embeddings_from_cache": bool(built.get("from_cache")),
        "trainable_param_count": int(trainable_params.get("csa", 0)),
        "trainable_params_by_mode": trainable_params,
        "patched_modules": [],
        "num_patched_modules": 0,
        "num_classes": emb_ds.num_classes,
        "embed_dim": emb_ds.embed_dim,
        "epochs": cfg.fallback.epochs,
        "primary_eval_split": primary_split,
        "eval_splits_present": list(emb_ds.eval_splits.keys()),
        "dataset_sizes": sizes,
        "timing_sec": {"encode": round(t_enc, 2), "training_and_eval": round(train_time, 2)},
        "baselines": cfg.training_modes(),
        "modes": primary,
        "modes_by_split": by_split,
        "go_no_go": go,
        "strong": strong,
        "waterbirds_csa_promising": False,  # hard guarantee for fallback
        "waterbirds_csa_strong": False,     # hard guarantee for fallback
        "waterbirds_csa_null": True,        # fallback-only ran -> null per pre-registration
    })

    paths = write_outputs(out_dir, metrics)
    _print_summary(metrics, paths)
    return metrics


# --------------------------------------------------------------------------- #
# Skip path
# --------------------------------------------------------------------------- #
def _write_skip(out_dir, meta, *, reason, real_openclip_loaded, real_lora_ran, fallback_ran, mode) -> dict[str, Any]:
    metrics = dict(meta)
    metrics.update({
        "mode": mode,
        "status": mode,
        "reason": reason,
        "real_openclip_loaded": real_openclip_loaded,
        "real_lora_ran": real_lora_ran,
        "fallback_ran": fallback_ran,
        "actual_lora_used": False,
        "trainable_param_count": 0,
        "patched_modules": [],
        "num_patched_modules": 0,
        "modes": {},
        "modes_by_split": {},
        "go_no_go": {},
        "waterbirds_csa_promising": False,
        "waterbirds_csa_strong": False,
        "waterbirds_csa_null": True,  # dataset/GPU/OpenCLIP unavailable -> null
    })
    paths = write_outputs(out_dir, metrics)
    _print_summary(metrics, paths)
    return metrics


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def write_outputs(out_dir: Path, metrics: dict[str, Any]) -> dict[str, str]:
    out_dir = ensure_dir(out_dir)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    table_path = out_dir / "table.csv"
    group_table_path = out_dir / "group_table.csv"
    per_seed_path = out_dir / "per_seed_metrics.csv"
    modes = metrics.get("modes") or {}
    if modes:
        rows = []
        for mode, m in modes.items():
            if not m:
                continue
            rows.append({
                "mode": mode,
                "label": MODE_LABELS.get(mode, mode),
                "average_accuracy": m.get("average_accuracy"),
                "worst_group_accuracy": m.get("worst_group_accuracy"),
                "counterfactual_instability": m.get("counterfactual_instability"),
                **{f"group_{g}": (m.get("group_accuracies") or {}).get(g) for g in wb.GROUP_NAMES},
            })
        write_csv(table_path, rows)

        group_rows = []
        for mode, m in modes.items():
            if not m:
                continue
            ga = m.get("group_accuracies") or {}
            gc = m.get("group_counts") or {}
            for g in wb.GROUP_NAMES:
                group_rows.append({"mode": mode, "group": g, "accuracy": ga.get(g), "count": gc.get(g)})
        write_csv(group_table_path, group_rows)
    else:
        write_csv(table_path, [{"mode": "skipped", "status": metrics.get("status"), "reason": metrics.get("reason", "")}])

    if metrics.get("per_seed") and len(metrics["per_seed"]) > 1:
        ps_rows = []
        for r in metrics["per_seed"]:
            for mode, m in r["primary"].items():
                if not m:
                    continue
                ps_rows.append({
                    "seed": r["seed"], "mode": mode,
                    "average_accuracy": m.get("average_accuracy"),
                    "worst_group_accuracy": m.get("worst_group_accuracy"),
                    "counterfactual_instability": m.get("counterfactual_instability"),
                })
        write_csv(per_seed_path, ps_rows)

    summary_path = write_summary(out_dir, metrics)
    paths = {"metrics_path": str(metrics_path), "table_path": str(table_path), "summary_path": str(summary_path)}
    if modes:
        paths["group_table_path"] = str(group_table_path)
    if metrics.get("per_seed") and len(metrics["per_seed"]) > 1:
        paths["per_seed_metrics_path"] = str(per_seed_path)
    return paths


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def write_summary(out_dir: Path, metrics: dict[str, Any]) -> Path:
    lines: list[str] = [
        "# Waterbirds CSA Manual-LoRA Pilot — Summary",
        "",
        "Bounded, pre-registered pilot: does CSA, applied as a small **manual-LoRA** ",
        "adaptation of a real OpenCLIP visual tower, improve **worst-group ",
        "robustness** on real WILDS Waterbirds **without using group labels for CSA ",
        "training**? One bounded experiment; positive, null, or negative results are ",
        "reported as-is.",
        "",
        "## Explicit non-claims",
        "",
        "- This is **not** universal robustness.",
        "- This is **not** open-world discovery.",
        "- This is **not** an RLHF/DPO replacement.",
        "- This is **not** deployment validation.",
        "- This is **not** a replacement for the finalized STS report.",
        "",
        "## Environment / availability",
        "",
        f"- Waterbirds data available: **{metrics.get('waterbirds_available')}** "
        f"(root `{metrics.get('dataset_root', '')}`).",
        f"- Real OpenCLIP loaded: **{metrics.get('real_openclip_loaded')}**.",
        f"- Device: **{metrics.get('device')}** (CUDA available: {metrics.get('cuda_available')}, "
        f"MPS available: {metrics.get('mps_available')}).",
        f"- True manual LoRA ran: **{metrics.get('real_lora_ran')}**.",
        f"- Cached-embedding fallback ran: **{metrics.get('fallback_ran')}**.",
        f"- Run mode: **{metrics.get('mode')}**; status: `{metrics.get('status')}`.",
        f"- Seeds: {metrics.get('seeds')}; single-seed pilot: **{metrics.get('single_seed_pilot')}**.",
    ]

    if metrics.get("status") in ("skipped_no_dataset", "skipped_no_openclip"):
        lines += [
            "",
            f"- **Skipped cleanly:** {metrics.get('reason')}.",
            "- Only fallback/smoke diagnostics (if any) ran; no real manual-LoRA result was produced.",
            "",
            "## Go / no-go",
            "",
            "- `waterbirds_csa_promising = false` (skipped).",
            "- `waterbirds_csa_strong = false` (skipped).",
            "- `waterbirds_csa_null = true` (dataset / GPU-MPS / OpenCLIP unavailable).",
            "",
            "This run does not modify the finalized STS report or any existing benchmark artifacts.",
        ]
        summary_path = out_dir / "summary.md"
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return summary_path

    iv = metrics.get("intervention_summary", {})
    lines += [
        "",
        "## Finite diagnostic interventions (CSA / CIC signal)",
        "",
        f"- Candidate interventions used (M={iv.get('n_candidates')}): "
        f"{', '.join('`' + t + '`' for t in iv.get('candidate_types', []))}.",
        f"- {iv.get('honesty_note', '')}",
        "",
        "## Adaptation",
        "",
    ]
    if metrics.get("real_lora_ran"):
        lines += [
            f"- Actual manual LoRA used: **True** via `{metrics.get('lora_library')}` "
            f"(rank {metrics.get('lora_rank')}, alpha {metrics.get('lora_alpha')}, "
            f"last {metrics.get('target_last_blocks')} of {metrics.get('num_visual_blocks')} visual blocks).",
            f"- Backbone `{metrics.get('model_name')}` / `{metrics.get('pretrained_tag')}`; "
            f"text/prompt head frozen ({metrics.get('config', {}).get('train_text_head')}).",
            f"- Trainable LoRA parameter count: **{metrics.get('trainable_param_count'):,}**.",
            "",
            "### Patched module names",
            "",
        ]
        for name in metrics.get("patched_modules", []):
            lines.append(f"- `{name}`")
    else:
        lines += [
            f"- **Cached-embedding fallback** (`cached_embedding_adapter`): {metrics.get('fallback_note')}",
            f"- Trainable module type: **{metrics.get('trainable_module_type')}**.",
            f"- Trainable parameter count: **{metrics.get('trainable_param_count'):,}**; "
            f"embeddings from cache: {metrics.get('embeddings_from_cache')}.",
        ]

    sizes = metrics.get("dataset_sizes", {})
    lines += [
        "",
        "## Setup",
        "",
        f"- Classes: {metrics.get('num_classes')} (landbird / waterbird); "
        f"epochs: {metrics.get('epochs')}; primary eval split: `{metrics.get('primary_eval_split')}`.",
        f"- Dataset sizes: {', '.join(f'{k}={v}' for k, v in sizes.items())}.",
        f"- Eval splits present: {metrics.get('eval_splits_present')}.",
        f"- Timing (s): {metrics.get('timing_sec')}.",
        "",
        "## Baselines",
        "",
    ]
    for mode in metrics.get("baselines", []):
        lines.append(f"- `{mode}`: {MODE_LABELS.get(mode, mode)}")

    modes = metrics.get("modes", {})
    lines += [
        "",
        f"## Metrics by mode (primary split = `{metrics.get('primary_eval_split')}`)",
        "",
        "| mode | avg acc | worst-group acc | CIC instability | "
        + " | ".join(g.replace("_", " ") for g in wb.GROUP_NAMES) + " |",
        "|---|---|---|---|" + "---|" * len(wb.GROUP_NAMES),
    ]
    for mode in metrics.get("baselines", []):
        m = modes.get(mode) or {}
        ga = m.get("group_accuracies") or {}
        lines.append(
            f"| {mode} | {_fmt(m.get('average_accuracy'))} | {_fmt(m.get('worst_group_accuracy'))} | "
            f"{_fmt(m.get('counterfactual_instability'))} | "
            + " | ".join(_fmt(ga.get(g)) for g in wb.GROUP_NAMES) + " |"
        )

    frozen = modes.get("frozen", {})
    plain = modes.get("plain_ft", {})
    csa = modes.get("csa", {})
    go = metrics.get("go_no_go", {})
    lines += [
        "",
        "- *avg acc* = average accuracy over all examples.",
        "- *worst-group acc* = minimum per-group accuracy (the robustness metric).",
        "- *CIC instability* = mean counterfactual instability across the finite intervention bank (lower = more stable).",
        "",
        "## Clean / zero-shot degradation",
        "",
        f"- Frozen zero-shot average accuracy: **{_fmt(frozen.get('average_accuracy'))}** "
        f"(worst-group **{_fmt(frozen.get('worst_group_accuracy'))}**).",
        f"- Plain manual-LoRA average accuracy: **{_fmt(plain.get('average_accuracy'))}**.",
        f"- CSA average accuracy: **{_fmt(csa.get('average_accuracy'))}**.",
        f"- CSA average-accuracy degradation vs frozen zero-shot: "
        f"**{_fmt((frozen.get('average_accuracy') or 0) - (csa.get('average_accuracy') or 0))}**.",
        "",
        "## CIC instability before / after",
        "",
        f"- Plain manual-LoRA (before): **{_fmt(plain.get('counterfactual_instability'))}**; "
        f"CSA (after): **{_fmt(csa.get('counterfactual_instability'))}**.",
        f"- Relative CIC instability drop, CSA vs plain: **{_fmt(go.get('instability_drop_rel_vs_plain'))}** "
        f"(threshold ≥ {_fmt(go.get('thresholds', {}).get('min_instability_drop_rel'))}).",
        "",
        "## CSA vs plain manual-LoRA (the pre-registered comparison)",
        "",
        f"- Worst-group accuracy — plain: **{_fmt(plain.get('worst_group_accuracy'))}**, "
        f"CSA: **{_fmt(csa.get('worst_group_accuracy'))}**; gain: "
        f"**{_fmt(go.get('worst_group_gain_vs_plain'))}** "
        f"(threshold ≥ +{_fmt(go.get('thresholds', {}).get('min_worst_group_gain'))}).",
        f"- Average-accuracy drop, CSA vs plain: **{_fmt(go.get('avg_acc_drop_vs_plain'))}** "
        f"(threshold ≤ {_fmt(go.get('thresholds', {}).get('max_avg_acc_drop'))}).",
    ]
    if "group_dro" in modes and modes["group_dro"]:
        gdro = modes["group_dro"]
        lines += [
            "",
            "## CSA vs Group DRO (Group DRO is GROUP-LABEL-SUPERVISED)",
            "",
            f"- Group DRO worst-group accuracy: **{_fmt(gdro.get('worst_group_accuracy'))}** "
            f"(avg **{_fmt(gdro.get('average_accuracy'))}**).",
            "- Note: Group DRO uses group labels during training; CSA does **not**. The "
            "comparison is reported honestly for context, not as a like-for-like supervision setting.",
        ]

    strong = metrics.get("strong", {})
    lines += [
        "",
        "## Go / no-go (pre-registered)",
        "",
        "`waterbirds_csa_promising = true` only if **all** of (and real manual-LoRA ran):",
        "",
        f"1. Real `manual_lora_visual` used (not cached fallback) — **{go.get('real_lora_used')}**.",
        f"2. Average-accuracy drop vs plain ≤ {_fmt(go.get('thresholds', {}).get('max_avg_acc_drop'))} — "
        f"**{go.get('avg_drop_ok')}**.",
        f"3. Worst-group accuracy improves ≥ +{_fmt(go.get('thresholds', {}).get('min_worst_group_gain'))} "
        f"over plain — **{go.get('worst_group_gain_ok')}**.",
        f"4. CIC instability drops ≥ {int(round((go.get('thresholds', {}).get('min_instability_drop_rel') or 0) * 100))}% "
        f"vs plain — **{go.get('instability_drop_ok')}**.",
        "",
        f"### `waterbirds_csa_promising = {str(metrics.get('waterbirds_csa_promising')).lower()}`",
        f"### `waterbirds_csa_strong = {str(metrics.get('waterbirds_csa_strong')).lower()}`",
        f"### `waterbirds_csa_null = {str(metrics.get('waterbirds_csa_null')).lower()}`",
        "",
        "`waterbirds_csa_strong = true` only if seeds {0,1,2} complete with real LoRA, "
        f"the mean paired CSA−plain worst-group gain ≥ +{_fmt(go.get('thresholds', {}).get('strong_worst_group_gain'))} "
        "and exceeds the seed-to-seed standard deviation, and the average-accuracy drop ≤ 0.03.",
        f"- Strong-flag detail: {json.dumps(strong)}",
        "",
        (
            "All pre-registered promising thresholds cleared on this bounded run."
            if metrics.get("waterbirds_csa_promising")
            else
            "Not all pre-registered promising thresholds were cleared; preserved honestly "
            "as a null/negative or diagnostic outcome on this bounded run and reported as-is."
        ),
    ]
    if metrics.get("single_seed_pilot"):
        lines += [
            "",
            "**Single-seed pilot:** only one seed ran (`single_seed_pilot=true`); strong "
            "success cannot be claimed from one seed.",
        ]
    lines += [
        "",
        "See `docs/csa_lora_waterbirds.md` for the full pre-registered design, the "
        "intervention definitions, and the interpretation of these bounded metrics. "
        "This run writes only under `results/csa_lora_pilot/waterbirds/` and is **not** "
        "a replacement for the finalized STS report.",
    ]

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _print_summary(metrics: dict[str, Any], paths: dict[str, str]) -> None:
    print(json.dumps({
        "status": metrics.get("status"),
        "mode": metrics.get("mode"),
        "waterbirds_available": metrics.get("waterbirds_available"),
        "real_openclip_loaded": metrics.get("real_openclip_loaded"),
        "real_lora_ran": metrics.get("real_lora_ran"),
        "fallback_ran": metrics.get("fallback_ran"),
        "trainable_param_count": metrics.get("trainable_param_count"),
        "waterbirds_csa_promising": metrics.get("waterbirds_csa_promising"),
        "waterbirds_csa_strong": metrics.get("waterbirds_csa_strong"),
        "waterbirds_csa_null": metrics.get("waterbirds_csa_null"),
        **paths,
    }, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
