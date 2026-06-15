from __future__ import annotations

import numpy as np
import pandas as pd


def _accuracy(pred: pd.Series, labels: pd.Series) -> float:
    valid = pred.notna()
    if not valid.any():
        return float("nan")
    return float((pred[valid].astype(int) == labels[valid].astype(int)).mean())


def _high_conf_failure_rate(df: pd.DataFrame, pred_col: str, conf_col: str, threshold: float = 0.8) -> float:
    high = df[pd.to_numeric(df[conf_col], errors="coerce") >= threshold]
    if high.empty:
        return float("nan")
    return float((high[pred_col].astype(float) != high["label"].astype(float)).mean())


def _flagged(df: pd.DataFrame) -> pd.Series:
    if "abstain" in df.columns:
        return df["abstain"].astype(bool)
    if "repair_action" in df.columns:
        return df["repair_action"].astype(str).isin(["abstain", "human_review"])
    return pd.Series(False, index=df.index)


def _safe_rate(mask: pd.Series, denom_mask: pd.Series | None = None) -> float:
    denom = mask if denom_mask is None else denom_mask
    if int(denom.sum()) == 0:
        return float("nan")
    return float(mask[denom].mean()) if denom_mask is not None else float(mask.mean())


def clean_accuracy_drop_row(
    certificates: pd.DataFrame,
    *,
    clean_mask: pd.Series | None = None,
    clean_split_name: str | None = None,
) -> dict[str, object]:
    if clean_mask is None or clean_split_name is None:
        return {
            "clean_accuracy_before": np.nan,
            "clean_accuracy_after_non_abstained": np.nan,
            "clean_accuracy_drop": np.nan,
            "clean_accuracy_drop_reason": "no clean split available",
        }
    clean = certificates[clean_mask].copy()
    if clean.empty:
        return {
            "clean_accuracy_before": np.nan,
            "clean_accuracy_after_non_abstained": np.nan,
            "clean_accuracy_drop": np.nan,
            "clean_accuracy_drop_reason": f"clean split '{clean_split_name}' is empty",
        }
    non_abstained = clean[~_flagged(clean)]
    before = _accuracy(clean["original_prediction"], clean["label"])
    after = _accuracy(non_abstained["repaired_prediction"], non_abstained["label"]) if len(non_abstained) else float("nan")
    return {
        "clean_accuracy_before": before,
        "clean_accuracy_after_non_abstained": after,
        "clean_accuracy_drop": before - after if np.isfinite(before) and np.isfinite(after) else np.nan,
        "clean_accuracy_drop_reason": f"measured on {clean_split_name}",
    }


def summarize_repair_metrics(
    certificates: pd.DataFrame,
    *,
    method: str,
    regime_col: str = "regime",
    high_confidence_threshold: float = 0.8,
    clean_mask: pd.Series | None = None,
    clean_split_name: str | None = None,
) -> pd.DataFrame:
    rows = []
    groups = certificates.groupby(regime_col, dropna=False) if regime_col in certificates.columns else [("all", certificates)]
    for regime, df in groups:
        flags = _flagged(df)
        original_accuracy = _accuracy(df["original_prediction"], df["label"])
        repaired = df[~flags]
        repaired_accuracy = _accuracy(repaired["repaired_prediction"], repaired["label"]) if len(repaired) else float("nan")
        abstention_rate = float(flags.mean()) if len(df) else float("nan")
        original_failure = df["original_prediction"].astype(float) != df["label"].astype(float)
        original_correct = ~original_failure
        automatic_corrected = (
            (~flags)
            & original_failure
            & (df["repaired_prediction"].astype(float) == df["label"].astype(float))
        )
        high_conf_failure_before = (
            (pd.to_numeric(df["original_confidence"], errors="coerce") >= high_confidence_threshold)
            & original_failure
        )
        high_conf_failure_after = (
            (~flags)
            & (pd.to_numeric(df["repaired_confidence"], errors="coerce") >= high_confidence_threshold)
            & (df["repaired_prediction"].astype(float) != df["label"].astype(float))
        )
        dangerous = df[df["quadrant"] == "Dangerous shortcut reliance"]
        row_clean_mask = clean_mask.reindex(df.index, fill_value=False) if clean_mask is not None else None
        rows.append(
            {
                "method": method,
                "regime": regime,
                "accuracy_before": original_accuracy,
                "accuracy_after_non_abstained": repaired_accuracy,
                "coverage": float((~flags).mean()) if len(df) else float("nan"),
                "selective_accuracy": repaired_accuracy,
                "high_confidence_failure_rate_before": _high_conf_failure_rate(df, "original_prediction", "original_confidence", high_confidence_threshold),
                "high_confidence_failure_rate_after": _high_conf_failure_rate(
                    repaired.rename(columns={"repaired_confidence": "after_confidence"}),
                    "repaired_prediction",
                    "after_confidence",
                    high_confidence_threshold,
                )
                if len(repaired)
                else float("nan"),
                "abstention_rate": abstention_rate,
                "failure_capture_rate": _safe_rate(flags, original_failure),
                "false_abstention_rate": _safe_rate(flags, original_correct),
                "repair_success_rate": _safe_rate(automatic_corrected, original_failure),
                "automatic_correction_success_rate": _safe_rate(automatic_corrected, original_failure),
                "human_review_flag_rate": abstention_rate,
                "dangerous_quadrant_capture_rate": _safe_rate(flags, df["quadrant"] == "Dangerous shortcut reliance"),
                "repair_success_rate_on_dangerous_quadrant": float(
                    (
                        (~_flagged(dangerous))
                        & (dangerous["original_prediction"].astype(float) != dangerous["label"].astype(float))
                        & (dangerous["repaired_prediction"].astype(float) == dangerous["label"].astype(float))
                    ).mean()
                )
                if len(dangerous)
                else np.nan,
                "dangerous_quadrant_n": int(len(dangerous)),
                "dangerous_quadrant_accuracy_before": _accuracy(dangerous["original_prediction"], dangerous["label"]) if len(dangerous) else np.nan,
                "dangerous_quadrant_accuracy_after": _accuracy(dangerous[~_flagged(dangerous)]["repaired_prediction"], dangerous[~_flagged(dangerous)]["label"]) if len(dangerous) else np.nan,
                "dangerous_quadrant_repair_success_rate": _safe_rate(
                    (~_flagged(dangerous))
                    & (dangerous["original_prediction"].astype(float) != dangerous["label"].astype(float))
                    & (dangerous["repaired_prediction"].astype(float) == dangerous["label"].astype(float)),
                    dangerous["original_prediction"].astype(float) != dangerous["label"].astype(float),
                )
                if len(dangerous)
                else np.nan,
                "n_original_failures": int(original_failure.sum()),
                "n_abstained": int(flags.sum()),
                "n_examples": int(len(df)),
                "n_non_abstained": int(len(repaired)),
                **clean_accuracy_drop_row(df, clean_mask=row_clean_mask, clean_split_name=clean_split_name),
            }
        )
    return pd.DataFrame(rows)
