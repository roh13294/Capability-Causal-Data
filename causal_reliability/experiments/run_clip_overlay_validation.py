from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

from causal_reliability.analysis.metrics import auroc_with_reason
from causal_reliability.data.clip_overlay_shortcuts import (
    examples_to_tensor,
    make_clip_overlay_dataset,
    save_default_example_grids,
    save_overlay_grid,
)
from causal_reliability.real_models.attribution import write_attribution_outputs
from causal_reliability.real_models.clip_zero_shot import ClipStatus, ClipZeroShotClassifier, check_clip_available
from causal_reliability.real_models.real_model_utils import cic_from_predictions, entropy, negative_margin, quadrant_label
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


class FakeOverlayClip:
    model_name = "fake_clip_overlay"
    pretrained = True
    zero_shot = True
    linear_probe = False
    warning = "Fake CLIP backend for tests only."

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names

    def predict(self, images: torch.Tensor) -> dict[str, Any]:
        arr = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        probs = []
        for img in arr:
            red = float(img[:, :, 0].mean())
            shape_mass = float((img[:, :, :3].mean(axis=2) < 0.25).mean())
            pred = int(np.clip(round((red - 0.08) * 8), 0, len(self.class_names) - 1))
            base = np.ones(len(self.class_names), dtype=np.float32) * 0.03
            base[pred] = min(0.94, 0.55 + red + shape_mass)
            base /= base.sum()
            probs.append(base)
        p = torch.tensor(np.stack(probs), dtype=torch.float32)
        conf, pred = p.max(dim=1)
        return {"probabilities": p, "predictions": pred, "confidence": conf, "logits": torch.log(p.clamp_min(1e-8))}


def _downloads_allowed(model_cfg: dict[str, Any]) -> bool:
    return bool(model_cfg.get("allow_pretrained_download", model_cfg.get("allow_download", False)))


def _status_unavailable(out_dir: Path, status: ClipStatus, cfg: dict[str, Any]) -> dict[str, str]:
    ensure_dir(out_dir)
    (out_dir / "clip_overlay_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    model_cfg = cfg.get("model", {})
    metrics = pd.DataFrame(
        [
            {
                "evidence_status": "unavailable",
                "downloads_allowed": _downloads_allowed(model_cfg),
                "backend_attempted": status.backend_attempted or model_cfg.get("preferred_backend", model_cfg.get("backend", "")),
                "backend": status.backend,
                "model_name": status.model_name,
                "pretrained_tag": status.pretrained_tag,
                "pretrained": status.pretrained,
                "error_message": status.error_message,
            }
        ]
    )
    metrics.to_csv(out_dir / "clip_overlay_metrics.csv", index=False)
    summary = [
        "# CLIP Text-Overlay Shortcut Validation",
        "",
        "CLIP unavailable.",
        "",
        "This run did not produce pretrained-model evidence and must not be used as headline real-model validation.",
        "",
        f"- Downloads allowed: {_downloads_allowed(model_cfg)}",
        f"- Backend attempted: {status.backend_attempted or model_cfg.get('preferred_backend', model_cfg.get('backend', ''))}",
        f"- Backend: {status.backend}",
        f"- Model name: {status.model_name or 'unavailable'}",
        f"- Pretrained tag: {status.pretrained_tag or 'none'}",
        f"- Pretrained weights loaded: {status.pretrained}",
        "- Evidence status: unavailable",
        f"- Error: {status.error_message}",
        "",
        "Install optional support with `pip install open_clip_torch` or `pip install transformers`.",
    ]
    (out_dir / "clip_overlay_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {"out_dir": str(out_dir), "summary": str(out_dir / "clip_overlay_summary.md"), "metrics": str(out_dir / "clip_overlay_metrics.csv")}


def _accuracy(df: pd.DataFrame, regime: str) -> float:
    sub = df[df["regime"] == regime]
    return float((sub["pred"] == sub["label"]).mean()) if len(sub) else float("nan")


def _metric_row(method: str, scores: np.ndarray, failure: np.ndarray) -> dict[str, Any]:
    auc, note = auroc_with_reason(scores, failure, min_failures=1)
    return {"method": method, "failure_auroc": auc, "auroc_note": note}


def _plot_outputs(certs: pd.DataFrame, metrics: pd.DataFrame, high_conf: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = ensure_dir(out_dir / "plots")
    failed = certs[certs["failure"] == 1]
    correct = certs[certs["failure"] == 0]
    plt.figure(figsize=(6, 4))
    plt.hist(correct["confidence"], bins=12, alpha=0.65, label="correct", color="#0072b2")
    plt.hist(failed["confidence"], bins=12, alpha=0.65, label="failed", color="#d55e00")
    plt.xlabel("Confidence")
    plt.ylabel("Examples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "confidence_hist_correct_vs_failed.png", dpi=170)
    plt.close()

    aucs = metrics.dropna(subset=["failure_auroc"]) if "failure_auroc" in metrics else pd.DataFrame()
    plt.figure(figsize=(7, 4))
    if len(aucs):
        plt.bar(aucs["method"], aucs["failure_auroc"], color="#44aa99")
        plt.xticks(rotation=25, ha="right")
        plt.ylim(0, 1)
    else:
        plt.text(0.5, 0.5, "AUROC undefined", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_dir / "clip_overlay_auc_comparison.png", dpi=170)
    plt.close()

    plt.figure(figsize=(5.4, 4.6))
    colors = np.where(certs["failure"].astype(int) == 1, "#d55e00", "#0072b2")
    plt.scatter(certs["confidence"], certs["cic_reliability"], c=colors, s=28, alpha=0.8)
    plt.axvline(0.8, color="0.25", linestyle="--", linewidth=1)
    plt.axhline(certs["cic_reliability"].median(), color="0.25", linestyle="--", linewidth=1)
    plt.xlabel("Confidence")
    plt.ylabel("Counterfactual stability")
    plt.tight_layout()
    plt.savefig(plot_dir / "clip_overlay_reliability_plane.png", dpi=170)
    plt.close()

    plt.figure(figsize=(5.6, 4.0))
    if len(high_conf):
        plt.plot(high_conf["threshold"], high_conf["cic_auroc"], marker="o", label="CIC AUROC")
        plt.plot(high_conf["threshold"], high_conf["failure_rate"], marker="s", label="failure rate")
        plt.ylim(0, 1)
        plt.xlabel("Confidence threshold")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No high-confidence subset", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_dir / "high_confidence_subset_auc.png", dpi=170)
    plt.close()


def _write_summary(path: Path, metrics: pd.DataFrame, certs: pd.DataFrame, occ: pd.DataFrame, model: Any) -> None:
    one = metrics.iloc[0].to_dict() if len(metrics) else {}
    lookup = metrics.set_index("method")["failure_auroc"].to_dict() if {"method", "failure_auroc"}.issubset(metrics) else {}
    high_fail = certs[(certs["regime"] == "misleading_overlay") & (certs["failure"] == 1)]
    lines = [
        "# CLIP Text-Overlay Shortcut Validation",
        "",
        "This controlled external validation tests whether a pretrained zero-shot CLIP model relies on an in-support text-overlay shortcut when the overlay conflicts with the actual shape label.",
        "",
        "This is not the primary evidence that confidence fails, because in the mixed overlay setting both confidence and CIC can achieve perfect failure AUROC. Instead, the CLIP experiment validates a different part of the story: shortcut reliance occurs in a real pretrained vision-language model.",
        "",
        f"- Evidence status: {one.get('evidence_status', 'pretrained CLIP evidence')}",
        f"- Downloads allowed: {one.get('downloads_allowed', False)}",
        f"- Backend attempted: {one.get('backend_attempted', getattr(getattr(model, 'status', None), 'backend_attempted', 'fake'))}",
        f"- Backend used: {one.get('backend', getattr(model, 'status', None).backend if hasattr(model, 'status') else 'fake')}",
        f"- Model name: {one.get('model_name', getattr(model, 'model_name', ''))}",
        f"- Pretrained tag: {one.get('pretrained_tag', getattr(getattr(model, 'status', None), 'pretrained_tag', '')) or 'none'}",
        f"- Pretrained weights loaded: {one.get('pretrained', getattr(model, 'pretrained', False))}",
        f"- Aligned accuracy: {one.get('aligned_accuracy', float('nan')):.3f}",
        f"- Misleading accuracy: {one.get('misleading_accuracy', float('nan')):.3f}",
        f"- Mixed accuracy: {one.get('mixed_accuracy', float('nan')):.3f}",
        f"- Confidence AUROC: {lookup.get('confidence_risk', float('nan')):.3f}",
        f"- CIC AUROC: {lookup.get('CIC', float('nan')):.3f}",
        f"- High-confidence misleading failure rate (confidence >= 0.8): {float((high_fail['confidence'] >= 0.8).mean()) if len(high_fail) else float('nan'):.3f}",
        f"- Mean text occlusion drop: {occ['text_occlusion_drop'].mean() if len(occ) and 'text_occlusion_drop' in occ else float('nan'):.3f}",
        f"- Mean object occlusion drop: {occ['object_occlusion_drop'].mean() if len(occ) and 'object_occlusion_drop' in occ else float('nan'):.3f}",
        "",
        "Misleading text overlays reduced accuracy sharply, and occlusion analysis checks whether masking text changes predictions more than masking the object. Use this result as real pretrained model shortcut-failure evidence, attribution sanity check evidence, and social relevance evidence. Do not use it as the cleanest confidence-vs-CIC separation result.",
        "",
        "Attribution is an occlusion sanity check, not proof of mechanism. This result should not be overclaimed as general foundation-model reliability.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "clip_overlay_validation")
    examples_dir = ensure_dir(out_dir / "examples")
    data_cfg = cfg.get("data", {})
    bundle = make_clip_overlay_dataset(n_per_class=int(data_cfg.get("n_per_class", 4)), size=int(data_cfg.get("image_size", 224)))
    save_default_example_grids(bundle, examples_dir)

    model_cfg = cfg.get("model", {})
    requested_device = str(model_cfg.get("device", "auto"))
    if requested_device == "auto":
        device = "cuda" if bool(cfg.get("prefer_gpu", False)) and torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    if model_cfg.get("backend") == "fake":
        model: Any = FakeOverlayClip(bundle.class_names)
        status = ClipStatus(
            available=True,
            backend="fake",
            model_name=model.model_name,
            pretrained=True,
            downloaded_or_cached="test_only",
            device=device,
            downloads_allowed=False,
            backend_attempted="fake",
        )
    else:
        status = check_clip_available(
            device=device,
            allow_download=_downloads_allowed(model_cfg),
            preferred_backend=str(model_cfg.get("preferred_backend", model_cfg.get("backend", "open_clip"))),
            model_name=model_cfg.get("model_name"),
            pretrained_tag=str(model_cfg.get("pretrained_tag", "laion2b_s34b_b79k")),
        )
        if not status.available or not status.pretrained:
            return _status_unavailable(out_dir, status, cfg)
        prompts = [f"a simple image of a {name}" for name in bundle.class_names]
        model = ClipZeroShotClassifier(status, bundle.class_names, prompts=prompts, device=device)

    examples = bundle.examples
    pred = model.predict(examples_to_tensor(examples))
    cf = model.predict(examples_to_tensor(examples, "counterfactual_image"))
    cic = cic_from_predictions(pred, cf)
    labels = np.array([ex["label"] for ex in examples], dtype=int)
    predictions = pred["predictions"].numpy()
    failure = (predictions != labels).astype(int)
    probs = pred["probabilities"]
    certs = pd.DataFrame(
        {
            "example_id": [ex["example_id"] for ex in examples],
            "task": "clip_overlay_validation",
            "regime": [ex["regime"] for ex in examples],
            "label": labels,
            "class_name": [ex["class_name"] for ex in examples],
            "shortcut": [ex["shortcut"] for ex in examples],
            "shortcut_label": [ex["shortcut_label"] for ex in examples],
            "pred": predictions,
            "confidence": pred["confidence"].numpy(),
            "failure": failure,
            "entropy": entropy(probs),
            "negative_margin": negative_margin(probs),
            "cis": cic["cis"],
            "shift_risk": cic["shift_risk"],
            "label_flip_only": cic["label_flip_only"],
            "cic_reliability": np.exp(-np.clip(cic["cis"], 0, None)),
            "cf_pred": cic["cf_prediction"],
            "cf_confidence": cic["cf_confidence"],
        }
    )
    certs["quadrant"] = [quadrant_label(c, cis) for c, cis in zip(certs["confidence"], certs["cis"])]
    certs.to_csv(out_dir / "clip_overlay_certificates.csv", index=False)

    mixed = certs[certs["regime"] == "mixed_overlay"]
    score_rows = []
    for name, scores in {
        "confidence_risk": 1.0 - mixed["confidence"].to_numpy(),
        "entropy": mixed["entropy"].to_numpy(),
        "negative_margin": mixed["negative_margin"].to_numpy(),
        "label_flip_only": mixed["label_flip_only"].to_numpy(),
        "CIC": mixed["cis"].to_numpy(),
    }.items():
        score_rows.append(_metric_row(name, scores, mixed["failure"].to_numpy(dtype=int)))
    hc = mixed[mixed["confidence"] >= 0.8]
    hc_auc, hc_note = auroc_with_reason(hc["cis"], hc["failure"], min_failures=1) if len(hc) else (float("nan"), "no high-confidence examples")
    base = {
        "downloads_allowed": status.downloads_allowed,
        "backend_attempted": status.backend_attempted,
        "backend": status.backend,
        "model_name": status.model_name,
        "pretrained_tag": status.pretrained_tag,
        "pretrained": status.pretrained,
        "downloaded_or_cached": status.downloaded_or_cached,
        "zero_shot": True,
        "aligned_accuracy": _accuracy(certs, "aligned_overlay"),
        "misleading_accuracy": _accuracy(certs, "misleading_overlay"),
        "mixed_accuracy": _accuracy(certs, "mixed_overlay"),
        "mean_confidence_on_misleading_failures": certs.loc[(certs["regime"] == "misleading_overlay") & (certs["failure"] == 1), "confidence"].mean(),
        "misleading_fail_conf_ge_0.8": (certs.loc[(certs["regime"] == "misleading_overlay") & (certs["failure"] == 1), "confidence"] >= 0.8).mean(),
        "misleading_fail_conf_ge_0.9": (certs.loc[(certs["regime"] == "misleading_overlay") & (certs["failure"] == 1), "confidence"] >= 0.9).mean(),
        "high_confidence_cic_auroc": hc_auc,
        "high_confidence_cic_auroc_note": hc_note,
        "dangerous_quadrant_failure_rate": certs.loc[certs["quadrant"] == "Dangerous shortcut reliance", "failure"].mean(),
        "evidence_status": "pretrained CLIP evidence" if status.pretrained and status.backend in {"open_clip", "transformers"} else "fallback smoke test",
    }
    metrics = pd.DataFrame([{**base, **row} for row in score_rows])
    metrics.to_csv(out_dir / "clip_overlay_metrics.csv", index=False)

    high_conf_rows = []
    for threshold in [0.7, 0.8, 0.9]:
        sub = certs[certs["confidence"] >= threshold]
        auc, note = auroc_with_reason(sub["cis"], sub["failure"], min_failures=1) if len(sub) else (float("nan"), "no examples")
        high_conf_rows.append({"threshold": threshold, "n": len(sub), "failure_rate": sub["failure"].mean() if len(sub) else np.nan, "cic_auroc": auc, "auroc_note": note})
    high_conf = pd.DataFrame(high_conf_rows)
    high_conf.to_csv(out_dir / "clip_overlay_high_confidence_subsets.csv", index=False)

    high_fail_ids = set(certs[(certs["failure"] == 1) & (certs["confidence"] >= 0.8)]["example_id"].head(8).astype(int))
    high_examples = [ex for ex in examples if int(ex["example_id"]) in high_fail_ids] or [ex for ex in examples if ex["regime"] == "misleading_overlay"][:8]
    save_overlay_grid(high_examples, examples_dir / "high_confidence_failures.png")

    misleading_indices = [i for i, ex in enumerate(examples) if ex["regime"] == "misleading_overlay"]
    misleading_examples = [examples[i] for i in misleading_indices]
    idx = torch.tensor(misleading_indices, dtype=torch.long)
    attr = write_attribution_outputs(
        model,
        misleading_examples,
        pred["predictions"].index_select(0, idx),
        pred["confidence"].index_select(0, idx),
        certs[certs["regime"] == "misleading_overlay"],
        out_dir / "attribution",
    )
    occ = pd.read_csv(attr["metrics"]) if Path(attr["metrics"]).exists() else pd.DataFrame()
    (out_dir / "clip_overlay_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    _plot_outputs(certs, metrics, high_conf, out_dir)
    _write_summary(out_dir / "clip_overlay_summary.md", metrics, certs, occ, model)
    return {"out_dir": str(out_dir), "summary": str(out_dir / "clip_overlay_summary.md"), "metrics": str(out_dir / "clip_overlay_metrics.csv")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clip_overlay_validation.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
