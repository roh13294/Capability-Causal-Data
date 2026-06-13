from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

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
from causal_reliability.utils.io import ensure_dir

CERTIFICATE_PATTERNS = ("*certificates.csv", "certificates_*.csv", "*_certificates.csv")
GROUP_COLUMNS = [
    "task",
    "shift_type",
    "partial_flip_fraction",
    "control",
    "component",
    "lambda",
    "seed",
    "model",
    "train_corr",
    "shift_corr",
    "intervention",
    "shift",
]


def read_certificate_files(results_dir: str | Path) -> list[tuple[Path, pd.DataFrame]]:
    root = Path(results_dir)
    seen: set[Path] = set()
    frames: list[tuple[Path, pd.DataFrame]] = []
    for pattern in CERTIFICATE_PATTERNS:
        for path in root.rglob(pattern):
            if path in seen:
                continue
            seen.add(path)
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if {"confidence", "failure"}.issubset(df.columns):
                frames.append((path, with_derived_scores(df)))
    return frames


def group_frame(path: Path, df: pd.DataFrame, results_dir: str | Path) -> Iterable[tuple[dict[str, object], pd.DataFrame]]:
    group_cols = [col for col in GROUP_COLUMNS if col in df.columns]
    if not group_cols:
        yield infer_metadata(path, results_dir), df
        return
    for keys, group in df.groupby(group_cols, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = infer_metadata(path, results_dir)
        meta.update({col: clean_scalar(value) for col, value in zip(group_cols, keys)})
        yield meta, group.reset_index(drop=True)


def infer_metadata(path: Path, results_dir: str | Path) -> dict[str, object]:
    root = Path(results_dir)
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parts = rel.parts
    task = parts[0] if len(parts) > 1 else path.stem.replace("_certificates", "")
    return {"source_file": str(rel), "experiment": parts[0] if parts else path.parent.name, "task": task}


def clean_scalar(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _norm(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    lo = values.min(skipna=True)
    hi = values.max(skipna=True)
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - lo) / (hi - lo)


def binary_entropy(confidence: pd.Series) -> pd.Series:
    p = pd.to_numeric(confidence, errors="coerce").clip(1e-8, 1 - 1e-8)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def with_derived_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aliases = {
        "mean_margin_collapse": "margin_collapse_mean",
        "tail_margin_collapse": "margin_collapse_q90",
        "JS": "js_mean",
        "label_flip": "flip_mean",
    }
    for target, source in aliases.items():
        if target not in df.columns and source in df.columns:
            df[target] = df[source]
    if "correct" not in df.columns and {"pred", "label"}.issubset(df.columns):
        df["correct"] = (df["pred"] == df["label"]).astype(int)
    if "failure" not in df.columns and "correct" in df.columns:
        df["failure"] = 1 - df["correct"].astype(int)
    if "entropy" not in df.columns and "confidence" in df.columns:
        df["entropy"] = binary_entropy(df["confidence"])
    if "confidence_risk" not in df.columns and "confidence" in df.columns:
        df["confidence_risk"] = 1.0 - pd.to_numeric(df["confidence"], errors="coerce")
    if "negative_margin" not in df.columns and "margin" in df.columns:
        df["negative_margin"] = -pd.to_numeric(df["margin"], errors="coerce")
    if "one_minus_cr" not in df.columns and "causal_reliability" in df.columns:
        df["one_minus_cr"] = 1.0 - pd.to_numeric(df["causal_reliability"], errors="coerce")
    if "cis" not in df.columns and {"flip_mean", "margin_collapse_mean", "margin_collapse_q90", "js_mean"}.issubset(df.columns):
        df["cis"] = (
            2.0 * pd.to_numeric(df["flip_mean"], errors="coerce")
            + 0.5 * _norm(df["margin_collapse_mean"])
            + 0.5 * _norm(df["margin_collapse_q90"])
            + 0.25 * _norm(df["js_mean"])
        )
    if "cis_reliability" not in df.columns and "cis" in df.columns:
        df["cis_reliability"] = np.exp(-pd.to_numeric(df["cis"], errors="coerce").clip(lower=0.0))
    return df


def safe_auroc(scores: pd.Series | np.ndarray, labels: pd.Series | np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[mask]
    labels = labels[mask]
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return auroc(scores, labels)


def label_flip_auc(results_dir: str | Path) -> pd.DataFrame:
    path = Path(results_dir) / "certificate_ablation" / "certificate_ablation_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "component" not in df.columns:
        return pd.DataFrame()
    label = df[df["component"].astype(str).str.contains("label_flip", case=False, na=False)].copy()
    if label.empty:
        return pd.DataFrame()
    cols = [col for col in ["task", "shift_type", "partial_flip_fraction", "failure_auroc"] if col in label.columns]
    return label[cols].rename(columns={"failure_auroc": "label_flip_only_auroc"})


def merge_label_flip(rows: pd.DataFrame, results_dir: str | Path) -> pd.DataFrame:
    label = label_flip_auc(results_dir)
    if label.empty or rows.empty:
        if "label_flip_only_auroc" not in rows.columns:
            rows["label_flip_only_auroc"] = np.nan
        return rows
    merge_cols = [col for col in ["task", "shift_type", "partial_flip_fraction"] if col in rows.columns and col in label.columns]
    if not merge_cols:
        if "task" in rows.columns and "task" in label.columns:
            return rows.merge(label[["task", "label_flip_only_auroc"]].drop_duplicates("task"), on="task", how="left")
        rows["label_flip_only_auroc"] = np.nan
        return rows
    return rows.merge(label.drop_duplicates(merge_cols), on=merge_cols, how="left")


def write_markdown_table(df: pd.DataFrame, path: str | Path, title: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if df.empty:
        path.write_text(f"# {title}\n\nNo eligible certificate files found.\n", encoding="utf-8")
    else:
        path.write_text(f"# {title}\n\n{_markdown_table(df)}\n", encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(col) for col in display.columns]
    rows = display.astype(str).values.tolist()
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    header = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def save_scatter(df: pd.DataFrame, path: str | Path, x: str, y: str, color: str | None = None) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return
    plt.figure(figsize=(5.4, 3.6))
    if color and color in df.columns:
        for name, group in df.groupby(color, observed=False):
            plt.scatter(group[x], group[y], s=28, alpha=0.75, label=str(name))
        plt.legend(fontsize=7)
    else:
        plt.scatter(df[x], df[y], s=28, alpha=0.75)
    plt.xlabel(x.replace("_", " "))
    plt.ylabel(y.replace("_", " "))
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_bar(df: pd.DataFrame, path: str | Path, category: str, value: str, group: str | None = None) -> None:
    if df.empty or category not in df.columns or value not in df.columns:
        return
    plot_df = df.copy()
    plot_df[category] = plot_df[category].astype(str)
    plt.figure(figsize=(max(5.5, 0.35 * len(plot_df)), 3.8))
    if group and group in plot_df.columns:
        for name, g in plot_df.groupby(group, observed=False):
            plt.plot(g[category], g[value], marker="o", label=str(name))
        plt.legend(fontsize=7)
    else:
        plt.bar(plot_df[category], plot_df[value])
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.ylabel(value.replace("_", " "))
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
