from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

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
from causal_reliability.experiments.stress_utils import score_map
from causal_reliability.utils.io import ensure_dir


def _read_csvs(results_dir: Path, pattern: str) -> list[tuple[Path, pd.DataFrame]]:
    frames = []
    for path in sorted(results_dir.rglob(pattern)):
        try:
            frames.append((path, pd.read_csv(path)))
        except pd.errors.EmptyDataError:
            continue
    return frames


def _roc_curve(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    y = y[order]
    pos = max(int(y.sum()), 1)
    neg = max(int((1 - y).sum()), 1)
    tpr = np.r_[0.0, np.cumsum(y) / pos, 1.0]
    fpr = np.r_[0.0, np.cumsum(1 - y) / neg, 1.0]
    return fpr, tpr


def _best_certificate_frame(results_dir: Path) -> pd.DataFrame | None:
    candidates = []
    preferred = results_dir / "partial_flip_sweep" / "partial_flip_certificates.csv"
    if preferred.exists():
        df = pd.read_csv(preferred)
        if "failure" in df.columns and df["failure"].nunique() > 1:
            return df
    for _path, df in _read_csvs(results_dir, "certificates*.csv"):
        if {"confidence", "margin", "shift_risk", "causal_reliability"}.issubset(df.columns):
            if "failure" not in df.columns and {"pred", "label"}.issubset(df.columns):
                df = df.copy()
                df["failure"] = (df["pred"] != df["label"]).astype(int)
            if "failure" in df.columns and df["failure"].nunique() > 1:
                candidates.append(df)
    return max(candidates, key=len) if candidates else None


def _panel_a(ax, results_dir: Path) -> None:
    rows = []
    pf = results_dir / "partial_flip_sweep" / "partial_flip_metrics.csv"
    if pf.exists():
        df = pd.read_csv(pf)
        for task, group in df.groupby("task"):
            calibrated = group[(group["shifted_accuracy"] >= 0.2) & (group["shifted_accuracy"] <= 0.7)]
            rec = (calibrated if len(calibrated) else group).sort_values("ShiftRisk_auroc", ascending=False).iloc[0]
            rows.append((str(task), float(rec["id_accuracy"]), float(rec["shifted_accuracy"])))
    cw = results_dir / "confident_wrong" / "confident_wrong_metrics.csv"
    if not rows and cw.exists():
        df = pd.read_csv(cw)
        for _, rec in df.iterrows():
            rows.append((str(rec["task"]), float(rec["id_accuracy"]), float(rec["shifted_accuracy"])))
    for path, df in _read_csvs(results_dir, "*metrics.csv"):
        if rows:
            break
        if {"id_accuracy", "shifted_accuracy"}.issubset(df.columns):
            task = path.relative_to(results_dir).parts[0] if len(path.relative_to(results_dir).parts) > 1 else path.parent.name
            erm = df[df.get("model", "erm") == "erm"] if "model" in df.columns else df
            if len(erm):
                rows.append((task, float(erm["id_accuracy"].mean()), float(erm["shifted_accuracy"].mean())))
    if not rows:
        ax.text(0.5, 0.5, "Run experiments to populate accuracy panels", ha="center", va="center")
        ax.set_axis_off()
        return
    x = np.arange(len(rows))
    width = 0.36
    ax.bar(x - width / 2, [r[1] for r in rows], width, label="ID")
    ax.bar(x + width / 2, [r[2] for r in rows], width, label="Shifted")
    ax.set_xticks(x, [r[0] for r in rows], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("A. In-support shortcut flip accuracy")
    ax.legend(fontsize=8)


def _panel_b(ax, cert_df: pd.DataFrame | None) -> None:
    if cert_df is None:
        ax.text(0.5, 0.5, "No certificate CSV with failures found", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.hist(cert_df[cert_df["failure"] == 0]["confidence"], bins=16, alpha=0.65, label="correct")
    ax.hist(cert_df[cert_df["failure"] == 1]["confidence"], bins=16, alpha=0.65, label="failed")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Examples")
    ax.set_title("B. Failed predictions can be confident")
    ax.legend(fontsize=8)


def _panel_c(ax, cert_df: pd.DataFrame | None) -> None:
    if cert_df is None:
        ax.text(0.5, 0.5, "No certificate CSV with failures found", ha="center", va="center")
        ax.set_axis_off()
        return
    scores = score_map(cert_df)
    for name in ("confidence", "entropy", "margin", "ShiftRisk"):
        fpr, tpr = _roc_curve(cert_df["failure"], scores[name])
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], color="0.5", linewidth=1, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("C. Failure prediction ROC")
    ax.legend(fontsize=8)


def _panel_d(ax, results_dir: Path) -> None:
    pf = results_dir / "partial_flip_sweep" / "partial_flip_metrics.csv"
    if pf.exists():
        df = pd.read_csv(pf)
        for method, col in (("confidence", "confidence_auroc"), ("ShiftRisk", "ShiftRisk_auroc")):
            if col in df:
                agg = df.groupby("partial_flip_fraction")[col].mean().reset_index()
                ax.plot(agg["partial_flip_fraction"], agg[col], marker="o", label=method)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Flip fraction")
        ax.set_ylabel("AUROC")
        ax.set_title("D. Partial flip sweep")
        ax.legend(fontsize=8)
        return
    path = results_dir / "confident_wrong" / "confident_wrong_high_conf_subset.csv"
    if not path.exists():
        ax.text(0.5, 0.5, "Run confident-wrong benchmark for high-confidence panel", ha="center", va="center")
        ax.set_axis_off()
        return
    df = pd.read_csv(path)
    df = df[df["threshold"].astype(float) == 0.8]
    methods = [("confidence", "confidence_auroc"), ("entropy", "entropy_auroc"), ("margin", "margin_auroc"), ("ShiftRisk", "ShiftRisk_auroc")]
    values = [float(df[col].mean()) if col in df else np.nan for _, col in methods]
    ax.bar([m[0] for m in methods], values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC")
    ax.set_title("D. Confidence >= 0.8 subset")
    ax.tick_params(axis="x", rotation=20)


def make_figure(results_dir: str | Path = "results") -> None:
    results_path = Path(results_dir)
    ensure_dir(results_path)
    final_summary = results_path / "final_validation" / "final_validation_summary.csv"
    final_certs = results_path / "final_validation" / "final_validation_certificates.csv"
    if final_summary.exists() and final_certs.exists():
        summary = pd.read_csv(final_summary)
        cert_df = pd.read_csv(final_certs)
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))

        grid = summary.pivot_table(index="task", columns="regime", values="cis_auroc", aggfunc="mean")
        if len(grid):
            im = axes[0, 0].imshow(grid.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
            axes[0, 0].set_xticks(range(grid.shape[1]), grid.columns, rotation=20, ha="right")
            axes[0, 0].set_yticks(range(grid.shape[0]), grid.index)
            axes[0, 0].set_title("A. Regime map")
            fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04, label="CIC AUROC")
        else:
            axes[0, 0].set_axis_off()

        cw = cert_df[cert_df["regime"] == "confident-wrong"]
        axes[0, 1].hist(cw[cw["failure"] == 0]["confidence"], bins=16, alpha=0.65, label="correct")
        axes[0, 1].hist(cw[cw["failure"] == 1]["confidence"], bins=16, alpha=0.65, label="failed")
        axes[0, 1].set_title("B. Confidence blind spot")
        axes[0, 1].set_xlabel("Confidence")
        axes[0, 1].set_ylabel("Examples")
        axes[0, 1].legend(fontsize=8)

        methods = [
            ("Confidence Risk", "confidence_risk_auroc"),
            ("Entropy", "entropy_auroc"),
            ("Margin", "negative_margin_auroc"),
            ("Old ShiftRisk", "shift_risk_auroc"),
            ("Label Flip", "label_flip_only_auroc"),
            ("CIC", "cis_auroc"),
        ]
        values = [summary[col].mean() for _, col in methods]
        errors = [summary.get(col.replace("_auroc", "_auroc_std"), pd.Series([np.nan])).mean() for _, col in methods]
        axes[1, 0].bar([m for m, _ in methods], values, yerr=errors, capsize=3)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_ylabel("Failure AUROC")
        axes[1, 0].set_title("C. Failure prediction AUROC")
        axes[1, 0].tick_params(axis="x", rotation=25)

        high = cert_df[cert_df["confidence"] >= 0.8]
        hv = []
        for name, col in [("Confidence Risk", "confidence_risk"), ("Old ShiftRisk", "shift_risk"), ("Label Flip", "label_flip_only"), ("CIC", "cis")]:
            hv.append(auroc(high[col], high["failure"]) if len(high) and high["failure"].nunique() > 1 else np.nan)
        axes[1, 1].bar(["Confidence\nRisk", "Old\nShiftRisk", "Label\nFlip", "CIC"], hv)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_ylabel("Failure AUROC")
        axes[1, 1].set_title("D. Confidence >= 0.8 subset")

        fig.tight_layout()
        fig.savefig(results_path / "sts_main_figure.png", dpi=180)
        fig.savefig(results_path / "sts_main_figure.pdf")
        plt.close(fig)
        (results_path / "sts_main_figure_caption.md").write_text(
            "CIC is most useful in the confident-wrong shortcut regime, not in every distribution-shift setting. "
            "The figure contrasts confidence-solvable, confident-wrong, and mixed regimes; confidence remains strong "
            "for ordinary uncertainty, while Counterfactual Instability Certificates target high-confidence shortcut failures. "
            "Error bars show across-seed standard deviation where available.\n",
            encoding="utf-8",
        )
        return
    cert_df = _best_certificate_frame(results_path)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    _panel_a(axes[0, 0], results_path)
    _panel_b(axes[0, 1], cert_df)
    _panel_c(axes[1, 0], cert_df)
    _panel_d(axes[1, 1], results_path)
    fig.tight_layout()
    fig.savefig(results_path / "sts_main_figure.png", dpi=180)
    fig.savefig(results_path / "sts_main_figure.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    make_figure(args.results_dir)


if __name__ == "__main__":
    main()
