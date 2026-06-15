from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_reliability.analysis.metrics import auroc
from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.experiments.run_random_aug_failure_benchmark import _flip_marker, _make_examples, _model_probs, _random_aug, _remove_marker, _sensitivity
from causal_reliability.repair import repair_batch, selective_abstention_policy, summarize_repair_metrics
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


def _plot(metrics: pd.DataFrame, png: Path, pdf: Path) -> None:
    plt.figure(figsize=(7.0, 4.0))
    labels = metrics["method"].tolist()
    values = metrics["accuracy_after_non_abstained"].to_numpy(dtype=float)
    plt.bar(labels, values, color=["#4c78a8", "#72b7b2", "#e45756", "#f58518", "#54a24b"][: len(labels)])
    plt.ylim(0, 1.02)
    plt.ylabel("Shifted accuracy after repair")
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(png, dpi=170)
    plt.savefig(pdf)
    plt.close()


def _selective_risk_curve(
    *,
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    method: str,
    steps: int = 40,
) -> pd.DataFrame:
    rows = []
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    scores = np.asarray(scores, dtype=float)
    for coverage in np.linspace(0.1, 1.0, steps):
        n_keep = max(1, int(round(coverage * len(labels))))
        keep_idx = np.argsort(scores)[:n_keep]
        acc = float((predictions[keep_idx] == labels[keep_idx]).mean())
        rows.append({"method": method, "coverage": float(n_keep / len(labels)), "selective_accuracy": acc, "selective_error": 1.0 - acc})
    return pd.DataFrame(rows)


def _plot_selective_risk(curves: pd.DataFrame, png: Path, pdf: Path) -> None:
    plt.figure(figsize=(7.2, 4.2))
    colors = {
        "confidence abstention": "#4c78a8",
        "random augmentation abstention": "#72b7b2",
        "CIC abstention": "#e45756",
    }
    for method, df in curves.groupby("method"):
        ordered = df.sort_values("coverage")
        plt.plot(ordered["coverage"], ordered["selective_accuracy"], label=method, color=colors.get(method), linewidth=2)
    plt.ylim(0, 1.02)
    plt.xlim(0, 1.0)
    plt.xlabel("Coverage")
    plt.ylabel("Selective accuracy")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(png, dpi=170)
    plt.savefig(pdf)
    plt.close()


def _repair_or_abstain(original: pd.DataFrame, repair: pd.DataFrame, cic_scores: np.ndarray, *, threshold: float, repair_confidence_threshold: float) -> pd.DataFrame:
    hybrid = original.copy()
    high_risk = np.asarray(cic_scores, dtype=float) >= threshold
    confident_repair = repair["repaired_confidence"].to_numpy(dtype=float) >= repair_confidence_threshold
    repair_mask = high_risk & confident_repair
    abstain_mask = high_risk & ~confident_repair
    hybrid["cic_score"] = cic_scores
    hybrid["stability_score"] = np.exp(-np.asarray(cic_scores, dtype=float))
    hybrid.loc[repair_mask, "repaired_prediction"] = repair.loc[repair_mask, "repaired_prediction"].to_numpy()
    hybrid.loc[repair_mask, "repaired_confidence"] = repair.loc[repair_mask, "repaired_confidence"].to_numpy()
    hybrid.loc[repair_mask, "repaired_correctness"] = repair.loc[repair_mask, "repaired_correctness"].to_numpy()
    hybrid.loc[repair_mask, "selected_intervention"] = repair.loc[repair_mask, "selected_intervention"].to_numpy()
    hybrid.loc[repair_mask, "repair_action"] = "repair"
    hybrid.loc[abstain_mask, "repaired_prediction"] = np.nan
    hybrid.loc[abstain_mask, "repaired_confidence"] = 0.0
    hybrid.loc[abstain_mask, "repaired_correctness"] = 0
    hybrid.loc[abstain_mask, "repair_action"] = "abstain"
    hybrid["abstain"] = abstain_mask
    hybrid["abstention_reason"] = np.where(abstain_mask, "high confidence + low stability", "stable prediction")
    hybrid["repair_strategy"] = "cic_guided_repair_or_abstain"
    hybrid["repair_success"] = (hybrid["original_correctness"].astype(int).eq(0)) & (hybrid["repaired_correctness"].astype(int).eq(1))
    hybrid["repair_fixed_prediction"] = hybrid["repair_success"]
    return hybrid


def run(cfg: dict[str, Any]) -> dict[str, str]:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "random_aug_failure_repair")
    seed = int(cfg.get("seed", 0))
    n = int(cfg.get("n_examples", 180))
    rng = np.random.default_rng(seed + 11)
    examples = _make_examples(n, seed)
    probs = _model_probs(examples["text"].tolist())
    random_probs = _model_probs([_random_aug(t, rng) for t in examples["text"]])
    removed_probs = _model_probs([_remove_marker(t) for t in examples["text"]])
    flipped_probs = _model_probs([_flip_marker(t) for t in examples["text"]])
    labels = examples["label"].to_numpy(dtype=int)
    pred = probs.argmax(axis=1)
    failure = (pred != labels).astype(int)
    random_sens = _sensitivity(probs, random_probs)
    cic_sens = np.maximum(_sensitivity(probs, removed_probs), _sensitivity(probs, flipped_probs) * 0.35)
    cf_stack = np.stack([removed_probs, flipped_probs], axis=1)
    random_stack = random_probs.reshape(len(examples), 1, -1)
    original = repair_batch(example_ids=examples["example_id"].tolist(), labels=labels, original_probs=probs, counterfactual_probs=cf_stack[:, :1, :], intervention_names=["none"], strategy="stability_weighted")
    original["repaired_prediction"] = original["original_prediction"]
    original["repaired_confidence"] = original["original_confidence"]
    original["repaired_correctness"] = original["original_correctness"]
    original["repair_action"] = "keep_original"
    original["repair_strategy"] = "original_model"
    original["repair_success"] = False
    original["repair_fixed_prediction"] = False
    random_repair = repair_batch(example_ids=examples["example_id"].tolist(), labels=labels, original_probs=probs, counterfactual_probs=random_stack, intervention_names=["random_augmentation"], strategy="counterfactual_consensus")
    cic_repair = repair_batch(example_ids=examples["example_id"].tolist(), labels=labels, original_probs=probs, counterfactual_probs=cf_stack, intervention_names=["remove_marker", "flip_marker"], strategy="shortcut_neutralized")
    cic_repair["cic_score"] = cic_sens
    cic_abstain_source = original.copy()
    cic_abstain_source["cic_score"] = np.maximum(cic_abstain_source["cic_score"].to_numpy(dtype=float), cic_sens)
    cic_abstention, threshold_stats = selective_abstention_policy(
        cic_abstain_source,
        fixed_threshold=float(cfg.get("cic_abstention_threshold", 0.5)),
        high_confidence_threshold=float(cfg.get("high_confidence_threshold", 0.8)),
        low_confidence_threshold=float(cfg.get("low_confidence_threshold", 0.0)),
        min_coverage=float(cfg.get("min_abstention_coverage", 0.4)),
    )
    cic_repair_or_abstain = _repair_or_abstain(
        original,
        cic_repair,
        cic_sens,
        threshold=threshold_stats["cic_threshold"],
        repair_confidence_threshold=float(cfg.get("repair_confidence_threshold", 0.8)),
    )
    for name, df in [
        ("original_model", original),
        ("random_augmentation_consensus", random_repair),
        ("cic_guided_automatic_repair", cic_repair),
        ("cic_guided_abstention", cic_abstention),
        ("cic_guided_repair_or_abstain", cic_repair_or_abstain),
    ]:
        df["method"] = name
        df["regime"] = "shifted_metadata_shortcut"
        df["marker_relation"] = examples["marker_relation"].tolist()
    certs = pd.concat([original, random_repair, cic_repair, cic_abstention, cic_repair_or_abstain], ignore_index=True)
    metric_frames = []
    for df in [original, random_repair, cic_repair, cic_abstention, cic_repair_or_abstain]:
        m = summarize_repair_metrics(df, method=str(df["method"].iloc[0]))
        metric_frames.append(m)
    metrics = pd.concat(metric_frames, ignore_index=True)
    detection = pd.DataFrame(
        [
            {"method": "random_augmentation_consensus", "failure_detection_auroc": auroc(random_sens, failure)},
            {"method": "cic_guided_automatic_repair", "failure_detection_auroc": auroc(cic_sens, failure)},
            {"method": "cic_guided_abstention", "failure_detection_auroc": auroc(cic_sens, failure)},
            {"method": "cic_guided_repair_or_abstain", "failure_detection_auroc": auroc(cic_sens, failure)},
            {"method": "original_model", "failure_detection_auroc": auroc(1.0 - probs.max(axis=1), failure)},
        ]
    )
    metrics = metrics.merge(detection, on="method", how="left")
    metrics["threshold_source"] = threshold_stats["threshold_source"]
    metrics["cic_abstention_threshold"] = threshold_stats["cic_threshold"]
    metrics["threshold_validation_failure_capture_rate"] = threshold_stats["validation_failure_capture_rate"]
    metrics["threshold_validation_coverage"] = threshold_stats["validation_coverage"]
    metrics["threshold_validation_n"] = threshold_stats["validation_n"]
    metrics["clean_accuracy_drop"] = np.nan
    metrics["clean_accuracy_drop_reason"] = "no clean split available"
    original_acc = float((pred == labels).mean())
    random_acc = float((random_repair["repaired_prediction"].astype(int).to_numpy() == labels).mean())
    cic_auto_acc = float((cic_repair["repaired_prediction"].astype(int).to_numpy() == labels).mean())
    metrics["original_accuracy"] = original_acc
    metrics["random_augmentation_accuracy"] = random_acc
    metrics["cic_automatic_repair_accuracy"] = cic_auto_acc
    curves = pd.concat(
        [
            _selective_risk_curve(labels=labels, predictions=pred, scores=1.0 - probs.max(axis=1), method="confidence abstention"),
            _selective_risk_curve(labels=labels, predictions=pred, scores=random_sens, method="random augmentation abstention"),
            _selective_risk_curve(labels=labels, predictions=pred, scores=cic_sens, method="CIC abstention"),
        ],
        ignore_index=True,
    )
    metrics.to_csv(out_dir / "random_aug_failure_repair_metrics.csv", index=False)
    certs.to_csv(out_dir / "random_aug_failure_repair_certificates.csv", index=False)
    curves.to_csv(out_dir / "random_aug_failure_selective_risk.csv", index=False)
    _plot(metrics, out_dir / "random_aug_failure_repair_plot.png", out_dir / "random_aug_failure_repair_plot.pdf")
    _plot_selective_risk(curves, out_dir / "random_aug_failure_selective_risk.png", out_dir / "random_aug_failure_selective_risk.pdf")
    cic_auc = float(detection.loc[detection["method"] == "cic_guided_abstention", "failure_detection_auroc"].iloc[0])
    abstention_row = metrics[metrics["method"] == "cic_guided_abstention"].iloc[0].to_dict()
    summary = [
        "# Random Augmentation Failure Repair",
        "",
        "This tests whether targeted counterfactual repair or abstention succeeds when generic random augmentation fails to identify the localized shortcut.",
        "",
        "The stronger result is selective detection rather than automatic correction. The shortcut intervention is supplied and localized to the metadata marker, and the result does not imply CIC-guided repair dominates random augmentation in all settings.",
        "",
        f"Original accuracy: {original_acc:.4f}. Random augmentation accuracy: {random_acc:.4f}. CIC automatic repair accuracy: {cic_auto_acc:.4f}.",
        f"CIC abstention coverage: {abstention_row.get('coverage', np.nan):.4f}. CIC selective accuracy: {abstention_row.get('selective_accuracy', np.nan):.4f}.",
        f"CIC failure detection AUROC: {cic_auc:.3f}.",
        "",
        "When CIC AUROC is near 1.000 and abstention captures most failures, this benchmark supports the claim that CIC is strongest as a failure detection and selective-abstention signal.",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
        "",
    ]
    (out_dir / "random_aug_failure_repair_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {
        "metrics": str(out_dir / "random_aug_failure_repair_metrics.csv"),
        "certificates": str(out_dir / "random_aug_failure_repair_certificates.csv"),
        "summary": str(out_dir / "random_aug_failure_repair_summary.md"),
        "selective_risk": str(out_dir / "random_aug_failure_selective_risk.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/random_aug_failure_repair.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
