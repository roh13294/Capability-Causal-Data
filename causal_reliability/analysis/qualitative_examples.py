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

from causal_reliability.data.colored_digits import make_colored_digits_task
from causal_reliability.utils.io import ensure_dir


QUADRANTS = [
    ("Reliable prediction", True, True, False, "High confidence and high stability: the prediction remains stable under shortcut changes."),
    ("Uncertain but stable", False, True, False, "Low confidence but high stability: uncertainty is visible, but the shortcut intervention does not destabilize the prediction."),
    ("Generally fragile", False, False, True, "Low confidence and low stability: both ordinary uncertainty and counterfactual fragility are present."),
    ("Dangerous shortcut reliance", True, False, True, "High confidence and low stability: a confident prediction depends on shortcut features."),
]


def _read_candidates(root: Path) -> pd.DataFrame:
    preferred = [
        root / "final_validation" / "final_validation_certificates.csv",
        root / "colored_digits" / "colored_digits_certificates.csv",
    ]
    for path in preferred:
        if path.exists():
            df = pd.read_csv(path)
            if "failure" not in df and {"pred", "label"}.issubset(df):
                df["failure"] = (df["pred"] != df["label"]).astype(int)
            return df
    frames = []
    for path in root.rglob("*certificates*.csv"):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if {"confidence", "cis"}.issubset(df.columns):
            df["source_path"] = str(path)
            if "failure" not in df and {"pred", "label"}.issubset(df):
                df["failure"] = (df["pred"] != df["label"]).astype(int)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _stability(df: pd.DataFrame) -> pd.Series:
    if "cic_reliability" in df:
        return pd.to_numeric(df["cic_reliability"], errors="coerce")
    if "cis_reliability" in df:
        return pd.to_numeric(df["cis_reliability"], errors="coerce")
    return np.exp(-pd.to_numeric(df["cis"], errors="coerce").clip(lower=0))


def _select_examples(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["stability"] = _stability(work)
    conf_thr = 0.8
    stab_thr = float(work["stability"].quantile(0.6))
    rows: list[dict[str, Any]] = []
    for label, high_conf, high_stab, prefer_failure, explanation in QUADRANTS:
        mask = (work["confidence"] >= conf_thr if high_conf else work["confidence"] < conf_thr) & (
            work["stability"] >= stab_thr if high_stab else work["stability"] < stab_thr
        )
        pool = work[mask]
        if pool.empty:
            pool = work.iloc[[int(np.argmin((work["confidence"] - conf_thr).abs() + (work["stability"] - stab_thr).abs()))]]
        preferred = pool[pool["failure"].astype(bool) == prefer_failure] if "failure" in pool else pool
        if preferred.empty:
            preferred = pool
        rec = preferred.sort_values(["confidence"], ascending=[False]).iloc[0]
        rows.append(
            {
                "quadrant": label,
                "original_input": f"example_id={rec.get('example_id', rec.name)}",
                "counterfactual_input": "shortcut/color intervention",
                "model_prediction": rec.get("pred", np.nan),
                "true_label": rec.get("label", np.nan),
                "confidence": rec.get("confidence", np.nan),
                "cic": rec.get("cis", np.nan),
                "counterfactual_stability": rec.get("stability", np.nan),
                "shifted_prediction_failed": bool(rec.get("failure", False)),
                "explanation": explanation,
            }
        )
    return pd.DataFrame(rows)


def _write_markdown(rows: pd.DataFrame, path: Path) -> None:
    if rows.empty:
        path.write_text("# Qualitative Examples\n\nNo certificate rows found.\n", encoding="utf-8")
        return
    lines = ["# Qualitative Examples", ""]
    for rec in rows.itertuples(index=False):
        lines.extend(
            [
                f"## {rec.quadrant}",
                "",
                f"- Original input: {rec.original_input}",
                f"- Counterfactual input: {rec.counterfactual_input}",
                f"- Prediction / true label: {rec.model_prediction} / {rec.true_label}",
                f"- Confidence: {float(rec.confidence):.3f}",
                f"- CIC: {float(rec.cic):.3f}",
                f"- Shifted prediction failed: {rec.shifted_prediction_failed}",
                f"- Explanation: {rec.explanation}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_quadrants(rows: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7.2, 5.2))
    if rows.empty:
        plt.text(0.5, 0.5, "No examples available", ha="center", va="center")
        plt.axis("off")
    else:
        colors = rows["shifted_prediction_failed"].map({True: "#d55e00", False: "#0072b2"})
        plt.scatter(rows["confidence"], rows["counterfactual_stability"], c=colors, s=90)
        for _, rec in rows.iterrows():
            plt.annotate(rec["quadrant"], (rec["confidence"], rec["counterfactual_stability"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
        plt.xlim(0, 1.02)
        plt.ylim(0, 1.02)
        plt.xlabel("Confidence")
        plt.ylabel("Counterfactual stability")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _plot_colored_digit_grid(path: Path) -> None:
    bundle = make_colored_digits_task(n_train=32, n_test=16, seed=123)
    x = bundle.shifted_test.tensors[0]
    fig, axes = plt.subplots(2, 8, figsize=(10, 2.8))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(x[i].permute(1, 2, 0).numpy())
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _real_model_examples(root: Path, out: Path) -> None:
    cert_path = root / "real_model_validation" / "real_model_certificates.csv"
    if not cert_path.exists():
        return
    certs = pd.read_csv(cert_path)
    attr_path = root / "real_model_validation" / "attribution" / "attribution_metrics.csv"
    attr = pd.read_csv(attr_path) if attr_path.exists() else pd.DataFrame()
    if len(attr):
        certs = certs.merge(attr[["example_id", "occlusion_shortcut_effect", "occlusion_object_effect", "shortcut_attention_ratio"]], on="example_id", how="left")
    selected = certs.sort_values(["failure", "confidence", "cis"], ascending=[False, False, False]).head(8)
    lines = ["# Real-Model Qualitative Examples", ""]
    for rec in selected.itertuples(index=False):
        lines.extend(
            [
                f"## Example {rec.example_id}",
                "",
                f"- True class: {getattr(rec, 'class_name', rec.label)}",
                f"- Predicted class: {rec.pred}",
                f"- Confidence: {float(rec.confidence):.3f}",
                f"- CIC: {float(rec.cis):.3f}",
                f"- Quadrant label: {getattr(rec, 'quadrant', 'NA')}",
                f"- Occlusion shortcut effect: {float(getattr(rec, 'occlusion_shortcut_effect', np.nan)):.3f}",
                f"- Occlusion object effect: {float(getattr(rec, 'occlusion_object_effect', np.nan)):.3f}",
                "- Explanation: high confidence plus high counterfactual instability indicates shortcut sensitivity in this controlled image.",
                "",
            ]
        )
    (out / "real_model_examples.md").write_text("\n".join(lines), encoding="utf-8")
    src = root / "real_model_validation" / "examples" / "high_conf_low_stability_failures.png"
    if src.exists():
        import shutil

        shutil.copyfile(src, out / "real_model_examples.png")


def _clip_overlay_examples(root: Path, out: Path) -> None:
    cert_path = root / "clip_overlay_validation" / "clip_overlay_certificates.csv"
    if not cert_path.exists():
        return
    certs = pd.read_csv(cert_path)
    attr_path = root / "clip_overlay_validation" / "attribution" / "clip_overlay_occlusion_metrics.csv"
    attr = pd.read_csv(attr_path) if attr_path.exists() else pd.DataFrame()
    if len(attr):
        keep = ["example_id", "text_occlusion_drop", "object_occlusion_drop", "background_occlusion_drop", "shortcut_attention_ratio"]
        certs = certs.merge(attr[[c for c in keep if c in attr.columns]], on="example_id", how="left")
    selected = certs[(certs["regime"] == "misleading_overlay") & (certs["failure"] == 1)].sort_values(["confidence", "cis"], ascending=[False, False]).head(8)
    if selected.empty:
        selected = certs[certs["regime"] == "misleading_overlay"].sort_values(["confidence", "cis"], ascending=[False, False]).head(8)
    lines = ["# CLIP Overlay Qualitative Examples", ""]
    for rec in selected.itertuples(index=False):
        lines.extend(
            [
                f"## Example {rec.example_id}",
                "",
                f"- Original image: misleading overlay `{getattr(rec, 'shortcut', '')}` on true shape `{getattr(rec, 'class_name', rec.label)}`",
                "- Counterfactual image: overlay replaced with neutral/correct text in the validation certificates.",
                f"- Prediction before/after: {rec.pred} / {getattr(rec, 'cf_pred', 'NA')}",
                f"- Confidence: {float(rec.confidence):.3f}",
                f"- CIC: {float(rec.cis):.3f}",
                f"- Text occlusion drop: {float(getattr(rec, 'text_occlusion_drop', np.nan)):.3f}",
                f"- Object occlusion drop: {float(getattr(rec, 'object_occlusion_drop', np.nan)):.3f}",
                "- Explanation: a high-confidence failure that changes under overlay counterfactuals is consistent with shortcut sensitivity, while occlusion is only a sanity check.",
                "",
            ]
        )
    (out / "clip_overlay_examples.md").write_text("\n".join(lines), encoding="utf-8")
    src = root / "clip_overlay_validation" / "examples" / "high_confidence_failures.png"
    if src.exists():
        import shutil

        shutil.copyfile(src, out / "clip_overlay_examples.png")


def run(results_dir: str | Path = "results") -> dict[str, str]:
    root = Path(results_dir)
    out = ensure_dir(root / "qualitative_examples")
    plot_dir = ensure_dir(out / "plots")
    rows = _select_examples(_read_candidates(root))
    rows.to_csv(out / "qualitative_examples.csv", index=False)
    _write_markdown(rows, out / "qualitative_examples.md")
    _real_model_examples(root, out)
    _clip_overlay_examples(root, out)
    _plot_quadrants(rows, plot_dir / "reliability_quadrant_examples.png")
    _plot_colored_digit_grid(plot_dir / "counterfactual_vision_examples.png")
    return {"markdown": str(out / "qualitative_examples.md"), "csv": str(out / "qualitative_examples.csv")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    print(run(args.results_dir))


if __name__ == "__main__":
    main()
