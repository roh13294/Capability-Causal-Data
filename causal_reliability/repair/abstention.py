from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AbstentionThreshold:
    cic_threshold: float
    source: str
    validation_failure_capture_rate: float | None
    validation_coverage: float | None
    validation_n: int


def _score(certificates: pd.DataFrame) -> pd.Series:
    if "cic_score" in certificates.columns:
        return pd.to_numeric(certificates["cic_score"], errors="coerce").fillna(np.inf)
    if "stability_score" in certificates.columns:
        return 1.0 - pd.to_numeric(certificates["stability_score"], errors="coerce").fillna(1.0)
    raise ValueError("certificates must include cic_score or stability_score")


def select_abstention_threshold(
    validation_certificates: pd.DataFrame | None,
    *,
    fixed_threshold: float = 0.5,
    high_confidence_threshold: float = 0.8,
    min_coverage: float = 0.5,
) -> AbstentionThreshold:
    if validation_certificates is None or validation_certificates.empty:
        return AbstentionThreshold(float(fixed_threshold), "fixed config threshold; no validation split available", None, None, 0)
    val = validation_certificates.copy()
    scores = _score(val)
    confidence = pd.to_numeric(val["original_confidence"], errors="coerce").fillna(0.0)
    failures = val["original_prediction"].astype(float) != val["label"].astype(float)
    candidates = sorted(set(float(x) for x in np.quantile(scores.to_numpy(dtype=float), np.linspace(0, 1, 41))))
    candidates.append(float(fixed_threshold))
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        flagged = (confidence >= high_confidence_threshold) & (scores >= threshold)
        coverage = float((~flagged).mean())
        if coverage < min_coverage:
            continue
        capture = float(flagged[failures].mean()) if int(failures.sum()) else 0.0
        false_flag = float(flagged[~failures].mean()) if int((~failures).sum()) else 0.0
        rank = (capture, coverage - false_flag, -threshold)
        if best is None or rank > best:
            best = rank
            best_threshold = threshold
            best_capture = capture
            best_coverage = coverage
    if best is None:
        best_threshold = float(fixed_threshold)
        flagged = (confidence >= high_confidence_threshold) & (scores >= best_threshold)
        best_capture = float(flagged[failures].mean()) if int(failures.sum()) else 0.0
        best_coverage = float((~flagged).mean())
    return AbstentionThreshold(
        cic_threshold=float(best_threshold),
        source="validation selected threshold",
        validation_failure_capture_rate=float(best_capture),
        validation_coverage=float(best_coverage),
        validation_n=int(len(val)),
    )


def selective_abstention_policy(
    certificates: pd.DataFrame,
    threshold_source: AbstentionThreshold | dict[str, Any] | float | None = None,
    *,
    validation_certificates: pd.DataFrame | None = None,
    fixed_threshold: float = 0.5,
    high_confidence_threshold: float = 0.8,
    low_confidence_threshold: float = 0.55,
    min_coverage: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(threshold_source, AbstentionThreshold):
        threshold = threshold_source
    elif isinstance(threshold_source, dict):
        threshold = AbstentionThreshold(
            cic_threshold=float(threshold_source.get("cic_threshold", fixed_threshold)),
            source=str(threshold_source.get("source", "fixed config threshold")),
            validation_failure_capture_rate=threshold_source.get("validation_failure_capture_rate"),
            validation_coverage=threshold_source.get("validation_coverage"),
            validation_n=int(threshold_source.get("validation_n", 0)),
        )
    elif threshold_source is None:
        threshold = select_abstention_threshold(
            validation_certificates,
            fixed_threshold=fixed_threshold,
            high_confidence_threshold=high_confidence_threshold,
            min_coverage=min_coverage,
        )
    else:
        threshold = AbstentionThreshold(float(threshold_source), "fixed config threshold", None, None, 0)

    df = certificates.copy()
    scores = _score(df)
    confidence = pd.to_numeric(df["original_confidence"], errors="coerce").fillna(0.0)
    no_valid_cf = df.get("selected_intervention", pd.Series("", index=df.index)).astype(str).isin(["", "none", "no_valid_counterfactual"])
    high_conf_low_stability = (confidence >= high_confidence_threshold) & (scores >= threshold.cic_threshold)
    low_confidence = confidence < low_confidence_threshold
    abstain = high_conf_low_stability | low_confidence | no_valid_cf
    reason = np.where(
        high_conf_low_stability,
        "high confidence + low stability",
        np.where(low_confidence, "low confidence uncertainty", np.where(no_valid_cf, "no valid counterfactual", "stable prediction")),
    )
    df["abstain"] = abstain.astype(bool)
    df["abstention_reason"] = reason
    df["repair_action"] = np.where(abstain, "abstain", df.get("repair_action", "keep_original"))
    df["repaired_prediction"] = np.where(abstain, np.nan, df["repaired_prediction"])
    df["repaired_confidence"] = np.where(abstain, 0.0, df["repaired_confidence"])
    df["repaired_correctness"] = np.where(abstain, 0, df["repaired_correctness"])
    stats = {
        "cic_threshold": threshold.cic_threshold,
        "threshold_source": threshold.source,
        "validation_failure_capture_rate": threshold.validation_failure_capture_rate,
        "validation_coverage": threshold.validation_coverage,
        "validation_n": threshold.validation_n,
        "coverage": float((~df["abstain"]).mean()) if len(df) else np.nan,
        "abstention_rate": float(df["abstain"].mean()) if len(df) else np.nan,
    }
    return df, stats
