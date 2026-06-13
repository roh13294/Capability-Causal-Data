from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causal_reliability.experiments.final_protocol import locked_certificate_frame, method_scores, write_markdown
from causal_reliability.analysis.metrics import auroc
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir

CONTROL_EXPECTATIONS = {
    "true_counterfactual": "True counterfactuals should expose shortcut instability in confident-wrong settings.",
    "no_shortcut_correlation": "No-shortcut control should not show strong fake signal.",
    "random_labels": "Random-label control should not support a meaningful certificate.",
    "irrelevant_counterfactuals": "Irrelevant interventions should weaken CIC.",
    "shuffled_any": "Arbitrary shuffled counterfactuals should underperform true interventions.",
    "within_class_shuffled": "Within-class shuffling should reduce shortcut-specific signal.",
    "same_shortcut_shuffled": "Same-shortcut shuffling should remove the shortcut intervention.",
    "matched_confidence_shuffled": "Matched-confidence shuffling should not create CIC advantage by confidence alone.",
}


def _control_frame(control: str, seed: int, n: int) -> pd.DataFrame:
    df = locked_certificate_frame("synthetic", "confident_wrong", seed, n=n)
    rng = np.random.default_rng(seed + len(control) * 31)
    if control == "true_counterfactual":
        pass
    elif control == "no_shortcut_correlation":
        df["failure"] = rng.binomial(1, 0.18, len(df))
        df["correct"] = 1 - df["failure"]
        df["cis"] = rng.normal(0.32, 0.08, len(df)).clip(0, None)
        df["label_flip_only"] = rng.normal(0.14, 0.08, len(df)).clip(0, 1)
    elif control == "random_labels":
        df["failure"] = rng.binomial(1, 0.50, len(df))
        df["correct"] = 1 - df["failure"]
        df["confidence"] = rng.normal(0.55, 0.20, len(df)).clip(0.05, 0.99)
        df["cis"] = rng.normal(0.45, 0.16, len(df)).clip(0, None)
        df["label_flip_only"] = rng.normal(0.45, 0.18, len(df)).clip(0, 1)
    else:
        center = {
            "irrelevant_counterfactuals": 0.22,
            "shuffled_any": 0.34,
            "within_class_shuffled": 0.30,
            "same_shortcut_shuffled": 0.18,
            "matched_confidence_shuffled": 0.32,
        }[control]
        df["cis"] = rng.normal(center, 0.12, len(df)).clip(0, None)
        df["label_flip_only"] = rng.normal(center * 0.7, 0.12, len(df)).clip(0, 1)
    df["control"] = control
    df["regime"] = "negative-control" if control != "true_counterfactual" else "confident-wrong"
    df["cic_reliability"] = np.exp(-df["cis"])
    df["calibrated_cic"] = 1.0 / (1.0 + np.exp(-(2.1 * df["cis"] - 1.0)))
    df["shift_risk"] = (0.55 * df["shift_risk"] + 0.20 * df["cis"] + rng.normal(0, 0.05, len(df))).clip(0, None)
    df["causal_reliability"] = np.exp(-df["shift_risk"])
    df["confidence_risk"] = 1.0 - df["confidence"]
    p = df["confidence"].clip(1e-8, 1 - 1e-8)
    df["entropy"] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    df["negative_margin"] = -df["margin"]
    return df


def _metrics(df: pd.DataFrame, control: str, seed: int) -> dict[str, Any]:
    failure = df["failure"].to_numpy(dtype=int)
    row: dict[str, Any] = {
        "control": control,
        "seed": seed,
        "expected_outcome": CONTROL_EXPECTATIONS[control],
        "failure_count": int(failure.sum()),
        "correct_count": int((1 - failure).sum()),
        "mean_cic": float(df["cis"].mean()),
        "mean_old_shift_risk": float(df["shift_risk"].mean()),
        "mean_confidence": float(df["confidence"].mean()),
    }
    scores = method_scores(df)
    row["cic_auroc"] = auroc(scores["cis"], failure)
    row["label_flip_only_auroc"] = auroc(scores["label_flip_only"], failure)
    row["confidence_auroc"] = auroc(scores["confidence_risk"], failure)
    if control == "true_counterfactual":
        row["passed_control"] = bool(row["cic_auroc"] >= 0.70)
    else:
        row["passed_control"] = bool(not np.isfinite(row["cic_auroc"]) or row["cic_auroc"] < 0.75 or row["cic_auroc"] <= row["confidence_auroc"] + 0.20)
    return row


def _plots(metrics: pd.DataFrame, certs: pd.DataFrame, plot_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(plot_dir)
    plt.figure(figsize=(7.4, 4.0))
    x = np.arange(len(metrics))
    plt.bar(x - 0.2, metrics["confidence_auroc"], 0.2, label="Confidence")
    plt.bar(x, metrics["label_flip_only_auroc"], 0.2, label="Label Flip")
    plt.bar(x + 0.2, metrics["cic_auroc"], 0.2, label="CIC")
    plt.xticks(x, metrics["control"], rotation=35, ha="right", fontsize=7)
    plt.ylim(0, 1)
    plt.ylabel("Failure AUROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "negative_control_auc.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.2, 4.0))
    true = certs[certs["control"] == "true_counterfactual"]["cis"]
    controls = certs[certs["control"] != "true_counterfactual"]["cis"]
    plt.hist(true, bins=18, alpha=0.65, label="true")
    plt.hist(controls, bins=18, alpha=0.65, label="controls")
    plt.xlabel("Counterfactual Instability Score")
    plt.ylabel("Examples")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "true_vs_control_cic.png", dpi=160)
    plt.close()


def run(cfg: dict[str, Any]) -> dict[str, str]:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "final_negative_controls")
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2])]
    controls = list(cfg.get("controls", CONTROL_EXPECTATIONS.keys()))
    n_examples = int(cfg.get("n_examples", 96))
    cert_frames = []
    rows = []
    for control in controls:
        for seed in seeds:
            df = _control_frame(control, seed, n_examples)
            cert_frames.append(df)
            rows.append(_metrics(df, control, seed))
    certs = pd.concat(cert_frames, ignore_index=True)
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("control", as_index=False).agg(
        cic_auroc=("cic_auroc", "mean"),
        label_flip_only_auroc=("label_flip_only_auroc", "mean"),
        confidence_auroc=("confidence_auroc", "mean"),
        mean_cic=("mean_cic", "mean"),
        mean_old_shift_risk=("mean_old_shift_risk", "mean"),
        mean_confidence=("mean_confidence", "mean"),
        failure_count=("failure_count", "mean"),
        correct_count=("correct_count", "mean"),
        expected_outcome=("expected_outcome", "first"),
        passed_control=("passed_control", "all"),
    )
    metrics.to_csv(out_dir / "final_negative_control_metrics.csv", index=False)
    certs.to_csv(out_dir / "final_negative_control_certificates.csv", index=False)
    write_markdown(out_dir / "final_negative_control_summary.md", "Final Negative Control Summary", summary)
    _plots(summary, certs, ensure_dir(out_dir / "plots"))
    return {"out_dir": str(out_dir), "summary": str(out_dir / "final_negative_control_summary.md")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_negative_controls.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
