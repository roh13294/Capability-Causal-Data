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
from causal_reliability.data.real_model_shortcuts import (
    images_to_tensor,
    labels_to_tensor,
    make_real_model_shortcut_dataset,
    save_example_grid,
)
from causal_reliability.real_models.attribution import write_attribution_outputs
from causal_reliability.real_models.pretrained_loader import load_real_model
from causal_reliability.real_models.real_model_utils import cic_from_predictions, entropy, negative_margin, quadrant_label
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


def _accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred == y).mean()) if len(y) else float("nan")


def _high_conf_rows(certs: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        subset = certs[certs["confidence"] >= threshold]
        rows.append(
            {
                "threshold": threshold,
                "n": int(len(subset)),
                "failure_rate": float(subset["failure"].mean()) if len(subset) else float("nan"),
                "confidence_auroc": auroc_with_reason(1.0 - subset["confidence"], subset["failure"])[0] if len(subset) else float("nan"),
                "cic_auroc": auroc_with_reason(subset["cis"], subset["failure"])[0] if len(subset) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _plot_outputs(certs: pd.DataFrame, metrics: pd.DataFrame, high_conf: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = ensure_dir(out_dir / "plots")
    failed = certs[certs["failure"] == 1]
    correct = certs[certs["failure"] == 0]
    plt.figure(figsize=(6.0, 4.0))
    plt.hist(correct["confidence"], bins=12, alpha=0.65, label="correct", color="#0072b2")
    plt.hist(failed["confidence"], bins=12, alpha=0.65, label="failed", color="#d55e00")
    plt.xlabel("Confidence")
    plt.ylabel("Examples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "confidence_hist_correct_vs_failed.png", dpi=170)
    plt.close()

    plt.figure(figsize=(6.2, 4.0))
    aucs = metrics.set_index("method")["failure_auroc"]
    plt.bar(aucs.index, aucs.values, color="#44aa99")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Failure AUROC")
    plt.tight_layout()
    plt.savefig(plot_dir / "real_model_auc_comparison.png", dpi=170)
    plt.close()

    plt.figure(figsize=(5.4, 4.6))
    colors = np.where(certs["failure"].astype(int) == 1, "#d55e00", "#0072b2")
    plt.scatter(certs["confidence"], certs["cic_reliability"], c=colors, s=28, alpha=0.78)
    plt.axvline(0.8, color="0.25", linestyle="--", linewidth=1)
    plt.axhline(certs["cic_reliability"].median(), color="0.25", linestyle="--", linewidth=1)
    plt.xlabel("Confidence")
    plt.ylabel("Counterfactual stability")
    plt.tight_layout()
    plt.savefig(plot_dir / "real_model_reliability_plane.png", dpi=170)
    plt.close()

    plt.figure(figsize=(5.6, 4.0))
    plt.plot(high_conf["threshold"], high_conf["failure_rate"], marker="o", label="failure rate")
    plt.plot(high_conf["threshold"], high_conf["cic_auroc"], marker="s", label="CIC AUROC")
    plt.ylim(0, 1)
    plt.xlabel("Confidence threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "high_confidence_subset_auc.png", dpi=170)
    plt.close()


def _qualitative_examples(bundle: Any, certs: pd.DataFrame, out_dir: Path) -> None:
    examples_dir = ensure_dir(out_dir / "examples")
    save_example_grid(bundle.shifted_examples, examples_dir / "original_and_counterfactual_examples.png", n=8)
    selected = certs[(certs["confidence"] >= 0.7) & (certs["cis"] >= certs["cis"].median()) & (certs["failure"] == 1)]
    if selected.empty:
        selected = certs.sort_values(["failure", "confidence", "cis"], ascending=[False, False, False]).head(8)
    ids = set(int(x) for x in selected["example_id"].head(8))
    save_example_grid([ex for ex in bundle.shifted_examples if int(ex["example_id"]) in ids], examples_dir / "high_conf_low_stability_failures.png", n=8)


def _write_summary(path: Path, cfg: dict[str, Any], model: Any, metrics: pd.DataFrame, certs: pd.DataFrame, id_acc: float, shifted_acc: float) -> None:
    lookup = metrics.set_index("method")["failure_auroc"].to_dict()
    notes = metrics.set_index("method")["auroc_note"].to_dict() if "auroc_note" in metrics else {}
    high = certs[certs["confidence"] >= 0.8]
    lines = [
        "# Real-Model Validation",
        "",
        "This controlled visual shortcut task tests whether confidence can fail under shortcut flips while counterfactual stability provides complementary evidence of shortcut reliance.",
        "",
        f"- Model used: {model.model_name}",
        f"- Pretrained weights actually loaded: {model.pretrained}",
        f"- CLIP available / zero-shot used: {model.zero_shot}",
        f"- Linear probe used: {model.linear_probe}",
        f"- Dataset type: {cfg.get('data', {}).get('shortcut_type', 'background')} shortcut shapes",
        f"- ID accuracy: {id_acc:.3f}",
        f"- Shifted accuracy: {shifted_acc:.3f}",
        f"- Certificate examples: ID plus shifted shortcut-flip examples (`split` column records source).",
        f"- Mean failed confidence: {certs.loc[certs['failure'] == 1, 'confidence'].mean() if (certs['failure'] == 1).any() else float('nan'):.3f}",
        f"- Confidence AUROC: {lookup.get('confidence_risk', float('nan')):.3f}",
        f"- Confidence AUROC note: {notes.get('confidence_risk', '')}",
        f"- CIC AUROC: {lookup.get('CIC', float('nan')):.3f}",
        f"- CIC AUROC note: {notes.get('CIC', '')}",
        f"- High-confidence failure rate (confidence >= 0.8): {high['failure'].mean() if len(high) else float('nan'):.3f}",
        "",
    ]
    if model.warning:
        lines.extend(["## Warning", "", model.warning, ""])
    lines.extend(
        [
            "## Limitations",
            "",
            "- This is a controlled visual shortcut task, not proof that CIC generalizes to all foundation models.",
            "- Fallback results are marked explicitly and should not be used as headline pretrained evidence.",
            "- Attribution outputs are sanity checks, not proof of mechanism.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "real_model_validation")
    examples_dir = ensure_dir(out_dir / "examples")
    data_cfg = cfg.get("data", {})
    bundle = make_real_model_shortcut_dataset(
        n_per_class=int(data_cfg.get("n_per_class", 16)),
        size=int(data_cfg.get("image_size", 64)),
        shortcut_type=str(data_cfg.get("shortcut_type", "background")),
        seed=seed,
        classes=data_cfg.get("classes"),
    )
    save_example_grid(bundle.id_examples, examples_dir / "dataset_examples.png", n=8)

    device = "cuda" if bool(cfg.get("prefer_gpu", False)) and torch.cuda.is_available() else "cpu"
    x_id = images_to_tensor(bundle.id_examples)
    y_id = labels_to_tensor(bundle.id_examples)
    x_shift = images_to_tensor(bundle.shifted_examples)
    x_cf = images_to_tensor(bundle.shifted_examples, key="counterfactual_image")
    y_shift = labels_to_tensor(bundle.shifted_examples)
    model = load_real_model(bundle.class_names, x_id, y_id, cfg.get("model", {}), device=device)
    id_pred = model.predict(x_id)
    id_cf_pred = model.predict(images_to_tensor(bundle.id_examples, key="counterfactual_image"))
    shift_pred = model.predict(x_shift)
    cf_pred = model.predict(x_cf)

    id_predictions = id_pred["predictions"].numpy()
    shift_predictions = shift_pred["predictions"].numpy()
    labels = y_shift.numpy()
    id_acc = _accuracy(id_predictions, y_id.numpy())
    shifted_acc = _accuracy(shift_predictions, labels)
    def certificate_frame(split: str, examples: list[dict[str, Any]], pred_result: dict[str, Any], cf_result: dict[str, Any], labels_np: np.ndarray) -> pd.DataFrame:
        cic = cic_from_predictions(pred_result, cf_result)
        probs = pred_result["probabilities"]
        predictions = pred_result["predictions"].numpy()
        confidence = pred_result["confidence"].numpy()
        failure = (predictions != labels_np).astype(int)
        return pd.DataFrame(
            {
                "example_id": [ex["example_id"] for ex in examples],
                "split": split,
                "task": "real_model_validation",
                "regime": "pretrained-shortcut-flip" if model.pretrained else "non-pretrained-fallback-shortcut-flip",
                "model": model.model_name,
                "model_type": "CLIP" if model.zero_shot else "vision_classifier",
                "pretrained": model.pretrained,
                "zero_shot": model.zero_shot,
                "linear_probe": model.linear_probe,
                "label": labels_np,
                "class_name": [ex["class_name"] for ex in examples],
                "pred": predictions,
                "confidence": confidence,
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

    certs = pd.concat(
        [
            certificate_frame("id", bundle.id_examples, id_pred, id_cf_pred, y_id.numpy()),
            certificate_frame("shifted", bundle.shifted_examples, shift_pred, cf_pred, labels),
        ],
        ignore_index=True,
    )
    certs["quadrant"] = [quadrant_label(c, cis) for c, cis in zip(certs["confidence"], certs["cis"])]
    certs.to_csv(out_dir / "real_model_certificates.csv", index=False)

    shifted_certs = certs[certs["split"] == "shifted"].copy()
    score_map = {
        "confidence_risk": 1.0 - shifted_certs["confidence"].to_numpy(),
        "entropy": shifted_certs["entropy"].to_numpy(),
        "negative_margin": shifted_certs["negative_margin"].to_numpy(),
        "old_ShiftRisk": shifted_certs["shift_risk"].to_numpy(),
        "label_flip_only": shifted_certs["label_flip_only"].to_numpy(),
        "CIC": shifted_certs["cis"].to_numpy(),
    }
    metric_rows = []
    for name, scores in score_map.items():
        auc, reason = auroc_with_reason(scores, shifted_certs["failure"])
        metric_rows.append(
            {
                "method": name,
                "failure_auroc": auc,
                "auroc_note": reason,
                "id_accuracy": id_acc,
                "shifted_accuracy": shifted_acc,
                "model": model.model_name,
                "model_type": "CLIP" if model.zero_shot else "vision_classifier",
                "pretrained": model.pretrained,
                "zero_shot": model.zero_shot,
                "linear_probe": model.linear_probe,
                "warning": model.warning,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "real_model_metrics.csv", index=False)
    high_conf = _high_conf_rows(shifted_certs, [0.7, 0.8, 0.9])
    high_conf.to_csv(out_dir / "real_model_high_confidence_subsets.csv", index=False)
    with (out_dir / "real_model_config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    _plot_outputs(certs, metrics, high_conf, out_dir)
    _qualitative_examples(bundle, certs, out_dir)
    attribution = write_attribution_outputs(
        model,
        bundle.shifted_examples,
        shift_pred["predictions"],
        shift_pred["confidence"],
        certs[certs["split"] == "shifted"],
        out_dir / "attribution",
    )
    _write_summary(out_dir / "real_model_summary.md", cfg, model, metrics, certs, id_acc, shifted_acc)
    return {
        "out_dir": str(out_dir),
        "metrics": str(out_dir / "real_model_metrics.csv"),
        "certificates": str(out_dir / "real_model_certificates.csv"),
        "summary": str(out_dir / "real_model_summary.md"),
        "attribution": str(attribution["summary"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_model_validation.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
