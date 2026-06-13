from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causal_reliability.analysis.metrics import auroc_with_reason
from causal_reliability.experiments.stress_utils import score_map
from causal_reliability.utils.io import ensure_dir


def _read_csvs(results_dir: Path) -> list[tuple[Path, pd.DataFrame]]:
    out = []
    for path in sorted(results_dir.rglob("*certificates*.csv")):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if {"confidence", "margin", "shift_risk", "causal_reliability"}.issubset(df.columns):
            out.append((path, df))
    return out


def _experiment_name(path: Path, results_dir: Path) -> str:
    rel = path.relative_to(results_dir)
    return "/".join(rel.parts[:-1]) or path.stem


def _row(path: Path, df: pd.DataFrame, results_dir: Path) -> dict[str, Any]:
    if "failure" not in df.columns and {"pred", "label"}.issubset(df.columns):
        df = df.copy()
        df["failure"] = (df["pred"] != df["label"]).astype(int)
    failure = df["failure"].to_numpy(dtype=int)
    scores = score_map(df)
    failed = df[df["failure"] == 1]
    correct = df[df["failure"] == 0]
    row: dict[str, Any] = {
        "experiment": _experiment_name(path, results_dir),
        "source": str(path.relative_to(results_dir)),
        "n_shifted": int(len(df)),
        "n_failures": int(failure.sum()),
        "n_correct": int((1 - failure).sum()),
        "failure_rate": float(failure.mean()) if len(failure) else float("nan"),
        "mean_confidence_failures": float(failed["confidence"].mean()) if len(failed) else float("nan"),
        "mean_confidence_correct": float(correct["confidence"].mean()) if len(correct) else float("nan"),
        "high_confidence_failure_count": int(((df["confidence"] >= 0.8) & (df["failure"] == 1)).sum()),
    }
    notes = []
    for method, values in scores.items():
        value, note = auroc_with_reason(values, failure)
        col = {
            "confidence": "auroc_confidence",
            "entropy": "auroc_entropy",
            "margin": "auroc_margin",
            "ShiftRisk": "auroc_shift_risk",
            "CIS": "auroc_cis",
            "causal reliability": "auroc_1_cr",
            "CIS reliability": "auroc_1_cis_reliability",
        }.get(method)
        if col is None:
            continue
        row[col] = value
        if note:
            notes.append(f"{method}: {note}")
    hc = df[df["confidence"] >= 0.8]
    if len(hc) and hc["failure"].nunique() > 1:
        row["shift_risk_auroc_high_confidence_subset"] = auroc_with_reason(score_map(hc)["ShiftRisk"], hc["failure"])[0]
    else:
        row["shift_risk_auroc_high_confidence_subset"] = float("nan")
        notes.append("high-confidence subset AUROC undefined")
    conf = row.get("auroc_confidence", np.nan)
    sr = row.get("auroc_shift_risk", np.nan)
    row["confidence_beats_shift_risk"] = bool(np.isfinite(conf) and np.isfinite(sr) and conf > sr)
    row["shift_risk_beats_confidence"] = bool(np.isfinite(conf) and np.isfinite(sr) and sr > conf)
    if row["confidence_beats_shift_risk"]:
        notes.append("confidence already solves failure prediction better than ShiftRisk")
    if row["shift_risk_beats_confidence"]:
        notes.append("ShiftRisk adds value over confidence")
    row["audit_flags"] = "; ".join(dict.fromkeys(notes))
    return row


def build_audit(results_dir: str | Path = "results") -> pd.DataFrame:
    results_path = Path(results_dir)
    return pd.DataFrame([_row(path, df, results_path) for path, df in _read_csvs(results_path)])


def _markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No certificate results found.\n"
    lines = ["# Metric Audit", ""]
    for _, row in df.iterrows():
        lines.append(f"## {row['experiment']}")
        lines.append(
            f"- shifted examples: {row['n_shifted']}, failures: {row['n_failures']}, correct: {row['n_correct']}, failure rate: {row['failure_rate']:.3f}"
        )
        cis = f", CIS: {row['auroc_cis']:.3f}" if "auroc_cis" in row and pd.notna(row["auroc_cis"]) else ""
        lines.append(
            f"- AUROC confidence: {row['auroc_confidence']:.3f}, entropy: {row['auroc_entropy']:.3f}, margin: {row['auroc_margin']:.3f}, ShiftRisk: {row['auroc_shift_risk']:.3f}{cis}, 1-CR: {row['auroc_1_cr']:.3f}"
        )
        lines.append(f"- high-confidence failures: {row['high_confidence_failure_count']}, high-confidence ShiftRisk AUROC: {row['shift_risk_auroc_high_confidence_subset']:.3f}")
        lines.append(f"- flags: {row['audit_flags'] or 'none'}")
        lines.append("")
    return "\n".join(lines)


def save_outputs(df: pd.DataFrame, results_dir: str | Path = "results") -> None:
    out_dir = ensure_dir(Path(results_dir) / "metric_audit")
    df.to_csv(out_dir / "metric_audit_summary.csv", index=False)
    (out_dir / "metric_audit_summary.md").write_text(_markdown(df), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    df = build_audit(args.results_dir)
    save_outputs(df, args.results_dir)
    print(df)


if __name__ == "__main__":
    main()
