from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.experiments.run_real_text_shortcut_validation import (
    _expand_samples,
    _fit_linear,
    _make_regime,
    _predict,
    _random_perturb,
    _read_samples,
    _replace_marker,
    _vectorize,
    _vocab,
)
from causal_reliability.repair import repair_batch, selective_abstention_policy, summarize_repair_metrics
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir
from causal_reliability.utils.seed import set_seed


def _plot(metrics: pd.DataFrame, png: Path, pdf: Path) -> None:
    pivot = metrics.pivot_table(index="method", columns="regime", values="accuracy_after_non_abstained", aggfunc="first")
    plt.figure(figsize=(8.6, 4.4))
    if len(pivot):
        pivot.plot(kind="bar", ax=plt.gca(), color=["#4c78a8", "#f58518", "#54a24b"])
        plt.ylim(0, 1.02)
        plt.ylabel("Accuracy after repair, non-abstained")
        plt.xticks(rotation=25, ha="right")
    else:
        plt.text(0.5, 0.5, "No repair metrics", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(png, dpi=170)
    plt.savefig(pdf)
    plt.close()


def _confidence_threshold(test: pd.DataFrame, labels: np.ndarray, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    rows = []
    for _, row in test.iterrows():
        i = len(rows)
        abstain = conf[i] < threshold
        rows.append(
            {
                "example_id": row["example_id"],
                "regime": row["regime"],
                "label": int(labels[i]),
                "original_prediction": int(pred[i]),
                "original_confidence": float(conf[i]),
                "original_correctness": int(pred[i] == labels[i]),
                "cic_score": 0.0,
                "stability_score": 1.0,
                "quadrant": "confidence_baseline",
                "selected_intervention": "none",
                "repaired_prediction": np.nan if abstain else int(pred[i]),
                "repaired_confidence": 0.0 if abstain else float(conf[i]),
                "repaired_correctness": 0 if abstain else int(pred[i] == labels[i]),
                "repair_strategy": "confidence_thresholding",
                "repair_action": "abstain" if abstain else "keep_original",
                "repair_success": bool(abstain and pred[i] != labels[i]),
            }
        )
    return pd.DataFrame(rows)


def run(cfg: dict[str, Any]) -> dict[str, str]:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "real_text_repair")
    sample_path = Path(cfg.get("sample_path", "causal_reliability/data/real_text_samples.csv"))
    base = _read_samples(sample_path)
    repeats = int(cfg.get("repeats", 14))
    marker_strength = int(cfg.get("marker_strength", 4))
    epochs = int(cfg.get("epochs", 120))
    lr = float(cfg.get("lr", 0.08))
    regimes = list(cfg.get("regimes", ["confidence-solvable", "confident-wrong", "mixed"]))
    all_method_certs: list[pd.DataFrame] = []
    examples = []
    for offset, regime in enumerate(regimes):
        train, test = _make_regime(base, regime, seed + offset * 17, repeats, marker_strength)
        vocab = _vocab(train["marked_text"].tolist())
        train_x, idf = _vectorize(train["marked_text"].tolist(), vocab)
        test_x, _ = _vectorize(test["marked_text"].tolist(), vocab, idf)
        y_train = train["label"].to_numpy(dtype=int)
        y_test = test["label"].to_numpy(dtype=int)
        model = _fit_linear(train_x, y_train, seed + offset, epochs, lr)
        probs = _predict(model, test_x)
        remove_text = [_replace_marker(t, "source: neutral") for t in test["marked_text"]]
        neutral_text = [_replace_marker(t, "source: unknown") for t in test["marked_text"]]
        alpha_text = [_replace_marker(t, "source: alpha") for t in test["marked_text"]]
        beta_text = [_replace_marker(t, "source: beta") for t in test["marked_text"]]
        random_text = [_random_perturb(t) for t in test["marked_text"]]
        cf_probs = []
        for texts in [remove_text, neutral_text, alpha_text, beta_text, random_text]:
            x, _ = _vectorize(texts, vocab, idf)
            cf_probs.append(_predict(model, x))
        cf_stack = np.stack(cf_probs, axis=1)
        ids = test["example_id"].tolist()
        marker_names = ["remove_metadata_marker", "neutral_metadata_marker", "alpha_marker", "beta_marker"]
        cic_consensus = repair_batch(example_ids=ids, labels=y_test, original_probs=probs, counterfactual_probs=cf_stack[:, :4, :], intervention_names=marker_names, strategy="counterfactual_consensus")
        cic_abstention, threshold_stats = selective_abstention_policy(
            cic_consensus,
            fixed_threshold=float(cfg.get("cic_abstention_threshold", 0.5)),
            high_confidence_threshold=float(cfg.get("high_confidence_threshold", 0.8)),
            low_confidence_threshold=float(cfg.get("low_confidence_threshold", 0.0)),
            min_coverage=float(cfg.get("min_abstention_coverage", 0.4)),
        )
        methods = {
            "original_classifier": repair_batch(example_ids=ids, labels=y_test, original_probs=probs, counterfactual_probs=cf_stack[:, :1, :], intervention_names=["none"], strategy="stability_weighted"),
            "confidence_thresholding": _confidence_threshold(test, y_test, probs, float(cfg.get("confidence_threshold", 0.8))),
            "random_token_perturbation_consensus": repair_batch(example_ids=ids, labels=y_test, original_probs=probs, counterfactual_probs=cf_stack[:, 4:5, :], intervention_names=["random_token_perturbation"], strategy="counterfactual_consensus"),
            "cic_marker_neutralized_prediction": repair_batch(example_ids=ids, labels=y_test, original_probs=probs, counterfactual_probs=cf_stack[:, :2, :], intervention_names=marker_names[:2], strategy="shortcut_neutralized"),
            "cic_counterfactual_consensus": cic_consensus,
            "cic_abstention": cic_abstention,
        }
        for method, df in methods.items():
            df["regime"] = regime
            if method == "original_classifier":
                df["repaired_prediction"] = df["original_prediction"]
                df["repaired_confidence"] = df["original_confidence"]
                df["repaired_correctness"] = df["original_correctness"]
                df["repair_action"] = "keep_original"
                df["repair_strategy"] = method
                df["repair_success"] = False
                df["repair_fixed_prediction"] = False
            df["method"] = method
            df["threshold_source"] = threshold_stats["threshold_source"] if method == "cic_abstention" else ""
            df["cic_abstention_threshold"] = threshold_stats["cic_threshold"] if method == "cic_abstention" else np.nan
            df["threshold_validation_failure_capture_rate"] = threshold_stats["validation_failure_capture_rate"] if method == "cic_abstention" else np.nan
            df["threshold_validation_coverage"] = threshold_stats["validation_coverage"] if method == "cic_abstention" else np.nan
            df["threshold_validation_n"] = threshold_stats["validation_n"] if method == "cic_abstention" else np.nan
            all_method_certs.append(df)
        merged = methods["cic_marker_neutralized_prediction"].copy()
        merged["marked_text"] = test["marked_text"].tolist()
        merged["neutralized_text"] = remove_text
        examples.extend(merged[(merged["original_correctness"] == 0) & (merged["repaired_correctness"] == 1)].head(3).to_dict("records"))
    certs = pd.concat(all_method_certs, ignore_index=True)
    metrics = pd.concat(
        [
            summarize_repair_metrics(
                df,
                method=str(df["method"].iloc[0]),
                clean_mask=df["regime"].astype(str).eq("confidence-solvable"),
                clean_split_name="confidence-solvable neutral-marker split",
            )
            for df in all_method_certs
        ],
        ignore_index=True,
    )
    threshold_cols = [
        "method",
        "regime",
        "threshold_source",
        "cic_abstention_threshold",
        "threshold_validation_failure_capture_rate",
        "threshold_validation_coverage",
        "threshold_validation_n",
    ]
    metrics = metrics.merge(certs[threshold_cols].drop_duplicates(["method", "regime"]), on=["method", "regime"], how="left")
    certs.to_csv(out_dir / "real_text_repair_certificates.csv", index=False)
    metrics.to_csv(out_dir / "real_text_repair_metrics.csv", index=False)
    _plot(metrics, out_dir / "real_text_repair_plot.png", out_dir / "real_text_repair_plot.pdf")
    examples_md = ["# Real Text Repair Examples", ""]
    for row in examples[:9]:
        examples_md.extend([f"## {row['example_id']} ({row['regime']})", "", f"- Original text: {row['marked_text']}", f"- Marker-neutralized text: {row['neutralized_text']}", f"- Original prediction: `{row['original_prediction']}`; repaired: `{row['repaired_prediction']}`; label: `{row['label']}`.", ""])
    (out_dir / "real_text_repair_examples.md").write_text("\n".join(examples_md), encoding="utf-8")
    summary = [
        "# Real Text Shortcut Repair",
        "",
        "This experiment evaluates CIC-guided repair when the candidate shortcut is a neutral metadata marker. It does not claim general causal discovery or turnkey text-model repair.",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
        "",
        "## CIC-Guided Abstention And Repair",
        "",
        "Automatic CIC repair corrected the dangerous-quadrant examples in the confident-wrong text regime when the shortcut-neutralized prediction matched the label.",
        "The absolute gain is small because original accuracy was already high in this controlled text setup, so dangerous-quadrant repair success is reported separately from total accuracy gain.",
        "Selective-abstention metrics report coverage and selective accuracy on non-abstained examples; abstention flags examples for review rather than counting them as automatic corrections.",
        "",
        "## Limitations",
        "",
        "- The marker interventions are supplied by the benchmark.",
        "- Abstention flags examples for review rather than correcting them automatically.",
        "- The checked-in sample is small and reproducible, not a broad production benchmark.",
    ]
    (out_dir / "real_text_repair_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {"metrics": str(out_dir / "real_text_repair_metrics.csv"), "certificates": str(out_dir / "real_text_repair_certificates.csv"), "summary": str(out_dir / "real_text_repair_summary.md")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_text_repair.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
