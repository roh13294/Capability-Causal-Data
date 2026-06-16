from __future__ import annotations

"""Label-free *predictive reliability gate* for proposal-based CIC repairs.

This module is a **practical predictive reliability layer** on top of the existing
CIC framework. It does **not** introduce a new universal theorem. Given the
per-example artifacts that the CIC experiments already write (repair certificates +
open-proposal diagnostics), it builds a unified feature table whose columns are
**only inference-time observable quantities** (no true label, no target label, no
correctness, no oracle-repair-success, no subset membership, no ground-truth box
overlap), and trains small *interpretable* gates (threshold rules, logistic
regression, depth<=3 decision trees, optionally Platt-calibrated logistic) that
predict, **before looking at the true label**, whether a CIC repair should be
trusted.

Design rules enforced here:

* Every model feature name starts with ``feat_`` and is checked against a forbidden
  substring list (``assert_label_free``) so a label can never be used as a feature.
* Evaluation labels start with ``label_`` and are constructed *after* the features,
  never fed to a model's ``fit`` as an input column.
* The module performs **no** real-model inference and writes **no** files; it is
  pure numpy/pandas and is driven by
  ``causal_reliability.experiments.run_predictive_cic_gate``.

The companion machine-checkable statement of the underlying *conditional* predictive
certificate lives in ``causal_reliability.theory.predictive_certificate``.
"""

import ast
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from causal_reliability.analysis.metrics import auprc, auroc

# --------------------------------------------------------------------------- #
# Canonical schema
# --------------------------------------------------------------------------- #
META_COLUMNS = [
    "benchmark",
    "group",  # "controlled" | "natural"
    "example_id",
    "regime",
    "method",
    "selected_family",
    "is_coco_strict",
    "is_coco_directional",
]

# Every model feature is observable at inference time. Names are deliberately free
# of any forbidden substring (see FORBIDDEN_FEATURE_SUBSTRINGS).
NUMERIC_FEATURES = [
    "feat_orig_confidence",
    "feat_repaired_confidence",
    "feat_orig_entropy",
    "feat_repaired_entropy",
    "feat_entropy_drop",
    "feat_repaired_margin",
    "feat_cic_selected_score",
    "feat_cic_top1_top2_gap",
    "feat_cic_topk_concentration",
    "feat_topk_repair_agreement",
    "feat_selected_area_fraction",
    "feat_stability_gain",
    "feat_clean_safe_proxy",
    "feat_ocr_included",
    "feat_proposal_count",
    "feat_topk_score_mean",
    "feat_topk_score_std",
    "feat_prediction_changed",
]

# Evaluation labels. NEVER used as model inputs.
LABEL_COLUMNS = [
    "label_repair_success",  # primary trust target
    "label_strict_success",
    "label_pairwise_recovery",
    "label_target_prob_improved",
    "label_distractor_decreased",
    "label_clean_safe_preserved",
]
PRIMARY_LABEL = "label_repair_success"

# Substrings that, if present in a *feature* column name, indicate label leakage.
FORBIDDEN_FEATURE_SUBSTRINGS = [
    "correct",
    "success",
    "oracle",
    "target",
    "distractor",
    "recover",
    "alias",
    "strict",
    "true_label",
    "human_label",
    "ground_truth",
    "label_",
    "upper_bound",
    "iou",  # ground-truth box overlap is diagnostic-only, never a feature
    "overlap",
]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _parse_bbox(value: Any) -> list[float] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", ""}:
        return None
    try:
        parsed = json.loads(s)
    except Exception:
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 4:
        return None
    return [float(v) for v in parsed[:4]]


def bbox_area_fraction(bbox: Any, image_size: float = 224.0) -> float:
    box = _parse_bbox(bbox)
    if box is None:
        return float("nan")
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(area / (image_size * image_size))


def binary_entropy(p: Any) -> float:
    """Entropy (nats) of the observable top-class Bernoulli ``(p, 1-p)``.

    A label-free proxy for predictive entropy when only the top-1 probability is
    recorded. Monotone in uncertainty, 0 at p in {0,1}, max at p=0.5.
    """

    try:
        p = float(p)
    except Exception:
        return float("nan")
    if not np.isfinite(p):
        return float("nan")
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


def _topk_score_stats(scores: Sequence[float], k: int = 5) -> dict[str, float]:
    s = np.asarray([float(v) for v in scores if np.isfinite(v)], dtype=float)
    if s.size == 0:
        return {
            "selected_score": float("nan"),
            "top1_top2_gap": float("nan"),
            "topk_concentration": float("nan"),
            "topk_score_mean": float("nan"),
            "topk_score_std": float("nan"),
            "proposal_count": 0.0,
        }
    s_sorted = np.sort(s)[::-1]
    top = s_sorted[:k]
    top1 = float(s_sorted[0])
    top2 = float(s_sorted[1]) if s_sorted.size > 1 else 0.0
    denom = float(np.sum(np.abs(top))) + 1e-12
    return {
        "selected_score": top1,
        "top1_top2_gap": top1 - top2,
        "topk_concentration": float(abs(top1) / denom),
        "topk_score_mean": float(np.mean(top)),
        "topk_score_std": float(np.std(top)),
        "proposal_count": float(s.size),
    }


def _ocr_family_flag(family: Any) -> float:
    if family is None or (isinstance(family, float) and np.isnan(family)):
        return float("nan")
    f = str(family).lower()
    if any(tok in f for tok in ("text_box", "ocr", "textness", "horizontal_text")):
        return 1.0
    return 0.0


# --------------------------------------------------------------------------- #
# Per-benchmark adapters: each returns a DataFrame with META + feature_* + label_*
# --------------------------------------------------------------------------- #
def _blank_row(benchmark: str, group: str, example_id: Any, regime: str, method: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "benchmark": benchmark,
        "group": group,
        "example_id": example_id,
        "regime": regime,
        "method": method,
        "selected_family": None,
        "is_coco_strict": False,
        "is_coco_directional": False,
    }
    for c in NUMERIC_FEATURES:
        row[c] = float("nan")
    for c in LABEL_COLUMNS:
        row[c] = float("nan")
    return row


def extract_certificate_benchmark(
    benchmark: str,
    group: str,
    certificates: pd.DataFrame,
    *,
    cic_method: str,
    consensus_method: str | None,
    rankings: pd.DataFrame | None = None,
    image_size: float = 224.0,
) -> pd.DataFrame:
    """Adapter for the hard-multidecoy / semantic-decoy *certificate* schema.

    Features come from the CIC top-1 repair method rows (and the proposal rankings
    where available). The label is the repair's correctness.
    """

    cert = certificates.copy()
    cic = cert[cert["method"] == cic_method].set_index("example_id")
    cons = (
        cert[cert["method"] == consensus_method].set_index("example_id")
        if consensus_method and consensus_method in set(cert["method"])
        else None
    )

    rank_by_example: dict[Any, list[float]] = {}
    if rankings is not None and len(rankings):
        for eid, grp in rankings.groupby("example_id"):
            rank_by_example[eid] = list(grp.sort_values("rank")["score"].astype(float))

    rows = []
    for eid, r in cic.iterrows():
        regime = str(r.get("regime", ""))
        row = _blank_row(benchmark, group, eid, regime, cic_method)
        orig_conf = float(r.get("original_confidence", float("nan")))
        rep_conf = float(r.get("repaired_confidence", float("nan")))
        row["feat_orig_confidence"] = orig_conf
        row["feat_repaired_confidence"] = rep_conf
        row["feat_orig_entropy"] = binary_entropy(orig_conf)
        row["feat_repaired_entropy"] = binary_entropy(rep_conf)
        if np.isfinite(orig_conf) and np.isfinite(rep_conf):
            row["feat_entropy_drop"] = binary_entropy(orig_conf) - binary_entropy(rep_conf)
        row["feat_repaired_margin"] = rep_conf  # observable top-1 prob of repaired pred

        # CIC scores: prefer full ranking list; else single top-1 score column.
        scores = rank_by_example.get(eid)
        if scores:
            stats = _topk_score_stats(scores)
        elif "top1_score" in r:
            stats = _topk_score_stats([float(r.get("top1_score", float("nan")))])
        else:
            stats = _topk_score_stats([])
        row["feat_cic_selected_score"] = stats["selected_score"]
        row["feat_cic_top1_top2_gap"] = stats["top1_top2_gap"]
        row["feat_cic_topk_concentration"] = stats["topk_concentration"]
        row["feat_topk_score_mean"] = stats["topk_score_mean"]
        row["feat_topk_score_std"] = stats["topk_score_std"]
        row["feat_proposal_count"] = stats["proposal_count"]

        # area fraction (use provided column or derive from bbox)
        if "selected_area_fraction" in r and np.isfinite(float(r.get("selected_area_fraction", float("nan")))):
            row["feat_selected_area_fraction"] = float(r["selected_area_fraction"])
        else:
            row["feat_selected_area_fraction"] = bbox_area_fraction(r.get("selected_bbox"), image_size)

        # stability gain proxy: observable distribution shift induced by selected region
        js = r.get("js_shift", r.get("js_divergence", float("nan")))
        row["feat_stability_gain"] = float(js) if pd.notna(js) else float("nan")

        # proposal family + OCR flag
        fam = r.get("selected_proposal_type", r.get("selected_family"))
        row["selected_family"] = None if fam is None or pd.isna(fam) else str(fam)
        row["feat_ocr_included"] = _ocr_family_flag(fam)

        # prediction changed (observable)
        oi = r.get("original_prediction_index")
        ri = r.get("repaired_prediction_index")
        if pd.notna(oi) and pd.notna(ri):
            row["feat_prediction_changed"] = float(int(float(oi) != float(ri)))

        # top-k repair agreement (observable: do top-1 and top-3 consensus agree?)
        if cons is not None and eid in cons.index:
            ci = cons.loc[eid].get("repaired_prediction_index")
            if pd.notna(ri) and pd.notna(ci):
                row["feat_topk_repair_agreement"] = float(int(float(ri) == float(ci)))

        # clean-safe proxy (observable): on a non-overlay/clean example, did the
        # repair leave the prediction unchanged? high == safe. Only defined for the
        # clean regime; otherwise NaN.
        if regime in {"no_overlay", "no_decoy"} and pd.notna(oi) and pd.notna(ri):
            row["feat_clean_safe_proxy"] = float(int(float(oi) == float(ri)))

        # ---- labels (eval only) ----
        rc = r.get("repaired_correct")
        row["label_repair_success"] = float(bool(rc)) if pd.notna(rc) else float("nan")
        row["label_strict_success"] = row["label_repair_success"]
        if regime in {"no_overlay", "no_decoy"} and pd.notna(oc := r.get("original_correct")):
            # clean-safe preservation: repaired prediction stays correct on clean input
            row["label_clean_safe_preserved"] = float(bool(rc)) if pd.notna(rc) else float("nan")

        rows.append(row)
    return pd.DataFrame(rows)


def extract_coco_full(
    per_example: pd.DataFrame,
    diagnostics: pd.DataFrame | None,
    *,
    benchmark: str = "coco_text_full",
    group: str = "natural",
    cic_method: str = "cic_top1_repair_excl_ocr",
    consensus_method: str = "cic_top3_repair_excl_ocr",
    strict_ids: Iterable[Any] = (),
    directional_ids: Iterable[Any] = (),
    image_size: float = 224.0,
) -> pd.DataFrame:
    """Adapter for the COCO-Text full proposal-CIC ``per_example`` schema."""

    strict_ids = set(int(x) for x in strict_ids)
    directional_ids = set(int(x) for x in directional_ids)
    df = per_example.copy()
    cic = df[df["method"] == cic_method].set_index("example_id")
    cons = df[df["method"] == consensus_method].set_index("example_id") if consensus_method in set(df["method"]) else None
    orig = df[df["method"] == "original_clip_prediction"].set_index("example_id") if "original_clip_prediction" in set(df["method"]) else None

    rank_by_example: dict[Any, list[float]] = {}
    fam_by_example: dict[Any, Any] = {}
    area_by_example: dict[Any, float] = {}
    if diagnostics is not None and len(diagnostics):
        for eid, grp in diagnostics.groupby("example_id"):
            grp = grp.sort_values("rank")
            rank_by_example[eid] = list(grp["score"].astype(float))
            top = grp.iloc[0]
            fam_by_example[eid] = top.get("proposal_family", top.get("proposal_type"))
            area_by_example[eid] = float(top.get("area_fraction", float("nan")))

    rows = []
    for eid, r in cic.iterrows():
        row = _blank_row(benchmark, group, eid, "coco_text", cic_method)
        row["is_coco_strict"] = int(eid) in strict_ids
        row["is_coco_directional"] = int(eid) in directional_ids

        orig_conf = float(orig.loc[eid].get("target_prob_orig", float("nan"))) if orig is not None and eid in orig.index else float("nan")
        # NOTE: target_prob is a per-target quantity; we do NOT use it as a feature.
        # The observable original top-1 confidence is not stored in this artifact, so
        # confidence-style features stay NaN for COCO and are imputed at fit time.

        scores = rank_by_example.get(eid)
        stats = _topk_score_stats(scores or [])
        row["feat_cic_selected_score"] = stats["selected_score"]
        row["feat_cic_top1_top2_gap"] = stats["top1_top2_gap"]
        row["feat_cic_topk_concentration"] = stats["topk_concentration"]
        row["feat_topk_score_mean"] = stats["topk_score_mean"]
        row["feat_topk_score_std"] = stats["topk_score_std"]
        row["feat_proposal_count"] = stats["proposal_count"]

        area = area_by_example.get(eid, float("nan"))
        row["feat_selected_area_fraction"] = area
        fam = fam_by_example.get(eid)
        row["selected_family"] = None if fam is None or pd.isna(fam) else str(fam)
        row["feat_ocr_included"] = 0.0  # excl-ocr method by construction

        # stability proxy from selected proposal js divergence
        if diagnostics is not None and eid in set(diagnostics["example_id"]):
            top = diagnostics[diagnostics["example_id"] == eid].sort_values("rank").iloc[0]
            js = top.get("js_divergence", float("nan"))
            row["feat_stability_gain"] = float(js) if pd.notna(js) else float("nan")

        # prediction changed + top-k agreement (observable post-pred labels)
        post = r.get("post_pred_label")
        if cons is not None and eid in cons.index:
            cpost = cons.loc[eid].get("post_pred_label")
            if pd.notna(post) and pd.notna(cpost):
                row["feat_topk_repair_agreement"] = float(int(str(post) == str(cpost)))
        if orig is not None and eid in orig.index:
            opost = orig.loc[eid].get("post_pred_label")
            if pd.notna(post) and pd.notna(opost):
                row["feat_prediction_changed"] = float(int(str(post) != str(opost)))

        # ---- labels (eval only) ----
        row["label_repair_success"] = float(bool(r.get("alias_correct"))) if pd.notna(r.get("alias_correct")) else float("nan")
        row["label_strict_success"] = float(bool(r.get("strict_correct"))) if pd.notna(r.get("strict_correct")) else float("nan")
        row["label_pairwise_recovery"] = float(bool(r.get("pairwise_recovered"))) if pd.notna(r.get("pairwise_recovered")) else float("nan")
        row["label_target_prob_improved"] = float(bool(r.get("target_prob_improved"))) if pd.notna(r.get("target_prob_improved")) else float("nan")
        row["label_distractor_decreased"] = float(bool(r.get("distractor_prob_decreased"))) if pd.notna(r.get("distractor_prob_decreased")) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def extract_natural_open(
    certificates: pd.DataFrame,
    diagnostics: pd.DataFrame | None,
    *,
    benchmark: str = "natural_text_open",
    group: str = "natural",
    cic_method: str = "cic_top1_repair",
    consensus_method: str = "cic_top3_repair",
    image_size: float = 224.0,
) -> pd.DataFrame:
    """Adapter for the natural-text open-proposal certificate schema (Round 1)."""

    df = certificates.copy()
    cic = df[df["method"] == cic_method].set_index("example_id")

    rank_by_example: dict[Any, list[float]] = {}
    if diagnostics is not None and len(diagnostics):
        for eid, grp in diagnostics.groupby("example_id"):
            rank_by_example[eid] = list(grp.sort_values("rank")["score"].astype(float))

    rows = []
    for eid, r in cic.iterrows():
        row = _blank_row(benchmark, group, eid, "natural_text", cic_method)
        orig_conf = float(r.get("original_confidence", float("nan")))
        row["feat_orig_confidence"] = orig_conf
        row["feat_orig_entropy"] = binary_entropy(orig_conf)

        scores = rank_by_example.get(eid)
        stats = _topk_score_stats(scores or [])
        row["feat_cic_selected_score"] = stats["selected_score"]
        row["feat_cic_top1_top2_gap"] = stats["top1_top2_gap"]
        row["feat_cic_topk_concentration"] = stats["topk_concentration"]
        row["feat_topk_score_mean"] = stats["topk_score_mean"]
        row["feat_topk_score_std"] = stats["topk_score_std"]
        row["feat_proposal_count"] = stats["proposal_count"]

        if "selected_area_fraction" in r and pd.notna(r.get("selected_area_fraction")):
            row["feat_selected_area_fraction"] = float(r["selected_area_fraction"])
        else:
            row["feat_selected_area_fraction"] = bbox_area_fraction(r.get("selected_bbox"), image_size)
        fam = r.get("selected_family", r.get("selected_proposal_type"))
        row["selected_family"] = None if fam is None or pd.isna(fam) else str(fam)
        row["feat_ocr_included"] = _ocr_family_flag(fam)

        if diagnostics is not None and eid in rank_by_example:
            top = diagnostics[diagnostics["example_id"] == eid].sort_values("rank").iloc[0]
            js = top.get("js_divergence", float("nan"))
            row["feat_stability_gain"] = float(js) if pd.notna(js) else float("nan")

        rc = r.get("repaired_correct")
        row["label_repair_success"] = float(bool(rc)) if pd.notna(rc) else float("nan")
        row["label_strict_success"] = row["label_repair_success"]
        rows.append(row)
    return pd.DataFrame(rows)


def extract_verified_failure(
    per_example: pd.DataFrame,
    diagnostics: pd.DataFrame | None,
    *,
    benchmark: str = "natural_text_verified",
    group: str = "natural",
    image_size: float = 224.0,
) -> pd.DataFrame:
    """Adapter for the natural-text verified-failure schema (one row per example)."""

    df = per_example.copy()
    rank_by_example: dict[Any, list[float]] = {}
    if diagnostics is not None and len(diagnostics):
        for eid, grp in diagnostics.groupby("example_id"):
            rank_by_example[eid] = list(grp.sort_values("rank")["score"].astype(float))

    rows = []
    for _, r in df.iterrows():
        eid = r.get("example_id")
        row = _blank_row(benchmark, group, eid, "natural_text_verified", "cic_top1_repair")
        orig_conf = float(r.get("original_confidence", float("nan")))
        row["feat_orig_confidence"] = orig_conf
        row["feat_orig_entropy"] = binary_entropy(orig_conf)

        scores = rank_by_example.get(eid)
        stats = _topk_score_stats(scores or [])
        row["feat_cic_selected_score"] = stats["selected_score"]
        row["feat_cic_top1_top2_gap"] = stats["top1_top2_gap"]
        row["feat_cic_topk_concentration"] = stats["topk_concentration"]
        row["feat_topk_score_mean"] = stats["topk_score_mean"]
        row["feat_topk_score_std"] = stats["topk_score_std"]
        row["feat_proposal_count"] = stats["proposal_count"]

        if pd.notna(r.get("selected_area_fraction")):
            row["feat_selected_area_fraction"] = float(r["selected_area_fraction"])
        else:
            row["feat_selected_area_fraction"] = bbox_area_fraction(r.get("selected_bbox"), image_size)
        fam = r.get("selected_family", r.get("selected_proposal_type"))
        row["selected_family"] = None if fam is None or pd.isna(fam) else str(fam)
        row["feat_ocr_included"] = _ocr_family_flag(fam)

        # observable agreement: do cic top1 and top3 repaired correctness flags agree
        # in *decision*? Not available (no predicted index). Leave NaN.

        rc = r.get("cic_top1_repair_correct")
        row["label_repair_success"] = float(bool(rc)) if pd.notna(rc) else float("nan")
        row["label_strict_success"] = row["label_repair_success"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Label-free / leakage guards
# --------------------------------------------------------------------------- #
def assert_label_free(feature_columns: Sequence[str]) -> None:
    """Raise if any feature column name contains a forbidden (label) substring."""

    bad = []
    for col in feature_columns:
        low = col.lower()
        for token in FORBIDDEN_FEATURE_SUBSTRINGS:
            # allow the deliberate 'feat_' prefix; only inspect the suffix
            suffix = low[len("feat_"):] if low.startswith("feat_") else low
            if token in suffix:
                bad.append((col, token))
    if bad:
        raise ValueError(f"label leakage in feature names: {bad}")


def check_no_oracle_leakage(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    label_column: str = PRIMARY_LABEL,
    suspicious_auroc: float = 0.999,
    min_coverage_for_value_test: float = 0.5,
) -> tuple[bool, list[str]]:
    """Leakage check.

    The **authoritative** guard is name-based: a feature whose name contains a
    forbidden (label/oracle/correctness/overlap) substring, or that equals an
    evaluation-label column, fails the check. This is what flips ``ok`` to False.

    A second, **advisory** value-based pass flags any *high-coverage* feature
    (coverage >= ``min_coverage_for_value_test``) that perfectly separates the label
    on pooled data — a heuristic smell test for a correctness-derived quantity that
    slipped in despite a clean name. Low-coverage features (defined on a small
    subset, e.g. clean-only proxies) are exempt because near-perfect separation on a
    tiny, near-constant-label subset is not evidence of leakage. Advisory hits are
    prefixed ``[advisory]`` and do NOT flip ``ok``.
    """

    reasons: list[str] = []
    ok = True
    try:
        assert_label_free(feature_columns)
    except ValueError as e:
        ok = False
        reasons.append(str(e))
    for col in feature_columns:
        if col in LABEL_COLUMNS:
            ok = False
            reasons.append(f"feature '{col}' is an evaluation label")

    if label_column in df.columns and len(df):
        y = pd.to_numeric(df[label_column], errors="coerce").to_numpy(dtype=float)
        labeled = np.isfinite(y)
        n_labeled = int(labeled.sum())
        for col in feature_columns:
            if col not in df.columns:
                continue
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            x[~np.isfinite(x)] = np.nan
            coverage = float(np.isfinite(x[labeled]).mean()) if n_labeled else 0.0
            if coverage < min_coverage_for_value_test:
                continue
            m = labeled & np.isfinite(x)
            if m.sum() < 10 or len(np.unique(y[m])) < 2:
                continue
            a = auroc(x[m], y[m].astype(int))
            if np.isfinite(a) and (a >= suspicious_auroc or a <= 1 - suspicious_auroc):
                reasons.append(f"[advisory] feature '{col}' near-perfectly separates label (AUROC={a:.3f}, coverage={coverage:.2f})")
    return ok, reasons


# --------------------------------------------------------------------------- #
# Imputation / standardization
# --------------------------------------------------------------------------- #
@dataclass
class Standardizer:
    medians: np.ndarray = field(default_factory=lambda: np.zeros(0))
    means: np.ndarray = field(default_factory=lambda: np.zeros(0))
    stds: np.ndarray = field(default_factory=lambda: np.ones(0))
    columns: list[str] = field(default_factory=list)

    @staticmethod
    def _sanitize(arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr, dtype=float).copy()
        out[~np.isfinite(out)] = np.nan  # treat inf and nan alike as missing
        return out

    def fit(self, X: pd.DataFrame) -> "Standardizer":
        self.columns = list(X.columns)
        arr = self._sanitize(X.to_numpy(dtype=float))
        with np.errstate(all="ignore"):
            self.medians = np.nanmedian(arr, axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        filled = self._impute(arr)
        self.means = filled.mean(axis=0)
        stds = filled.std(axis=0)
        self.stds = np.where(stds > 1e-9, stds, 1.0)
        return self

    def _impute(self, arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        idx = np.where(~np.isfinite(out))
        if idx[0].size:
            out[idx] = np.take(self.medians, idx[1])
        return out

    def transform(self, X: pd.DataFrame, standardize: bool = True) -> np.ndarray:
        arr = self._sanitize(X[self.columns].to_numpy(dtype=float))
        filled = self._impute(arr)
        if standardize:
            return (filled - self.means) / self.stds
        return filled


# --------------------------------------------------------------------------- #
# Interpretable gates
# --------------------------------------------------------------------------- #
@dataclass
class ThresholdGate:
    feature: str = ""
    threshold: float = 0.0
    direction: float = 1.0  # +1: accept when feature >= threshold
    youden_j: float = 0.0
    columns: list[str] = field(default_factory=list)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ThresholdGate":
        self.columns = list(X.columns)
        y = np.asarray(y, dtype=int)
        best = (-1.0, self.columns[0] if self.columns else "", 0.0, 1.0)
        for col in self.columns:
            x = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            x[~np.isfinite(x)] = np.nan
            med = np.nanmedian(x) if np.isfinite(x).any() else 0.0
            x = np.where(np.isfinite(x), x, med if np.isfinite(med) else 0.0)
            cand = np.unique(x)
            if cand.size > 64:
                cand = np.quantile(x, np.linspace(0, 1, 64))
            for thr in cand:
                for direction in (1.0, -1.0):
                    accept = (x >= thr) if direction > 0 else (x <= thr)
                    tp = int(((y == 1) & accept).sum())
                    fn = int(((y == 1) & ~accept).sum())
                    fp = int(((y == 0) & accept).sum())
                    tn = int(((y == 0) & ~accept).sum())
                    tpr = tp / (tp + fn) if (tp + fn) else 0.0
                    fpr = fp / (fp + tn) if (fp + tn) else 0.0
                    j = tpr - fpr
                    if j > best[0]:
                        best = (j, col, float(thr), direction)
        self.youden_j, self.feature, self.threshold, self.direction = best
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        x = pd.to_numeric(X[self.feature], errors="coerce").to_numpy(dtype=float)
        x[~np.isfinite(x)] = np.nan
        med = np.nanmedian(x) if np.isfinite(x).any() else 0.0
        x = np.where(np.isfinite(x), x, med if np.isfinite(med) else 0.0)
        # smooth monotone score around the threshold so AUROC uses the raw ordering
        return _sigmoid(self.direction * (x - self.threshold))

    def rule_text(self) -> str:
        op = ">=" if self.direction > 0 else "<="
        return f"accept CIC repair iff {self.feature} {op} {self.threshold:.4g}"


@dataclass
class LogisticGate:
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bias: float = 0.0
    scaler: Standardizer | None = None
    columns: list[str] = field(default_factory=list)
    l2: float = 1.0
    iters: int = 3000
    lr: float = 0.2

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LogisticGate":
        self.columns = list(X.columns)
        self.scaler = Standardizer().fit(X)
        Z = self.scaler.transform(X, standardize=True)
        y = np.asarray(y, dtype=float)
        n, d = Z.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(self.iters):
            p = _sigmoid(Z @ w + b)
            grad_w = Z.T @ (p - y) / n + self.l2 * w / n
            grad_b = float(np.mean(p - y))
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.weights, self.bias = w, b
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.scaler.transform(X, standardize=True)
        return _sigmoid(Z @ self.weights + self.bias)

    def coefficients(self) -> dict[str, float]:
        return {c: float(w) for c, w in zip(self.columns, self.weights)}


@dataclass
class _TreeNode:
    feature: int | None = None
    threshold: float = 0.0
    value: float = 0.0
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None
    column: str | None = None


@dataclass
class DecisionTreeGate:
    max_depth: int = 3
    min_leaf: int = 5
    root: _TreeNode | None = None
    columns: list[str] = field(default_factory=list)
    scaler: Standardizer | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "DecisionTreeGate":
        assert self.max_depth <= 3, "interpretability constraint: max_depth <= 3"
        self.columns = list(X.columns)
        self.scaler = Standardizer().fit(X)
        Z = self.scaler.transform(X, standardize=False)
        self.root = self._build(Z, np.asarray(y, dtype=float), depth=0)
        return self

    def _gini(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        p = y.mean()
        return float(2 * p * (1 - p))

    def _build(self, Z: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        node = _TreeNode(value=float(y.mean()) if y.size else 0.0)
        if depth >= self.max_depth or y.size < 2 * self.min_leaf or len(np.unique(y)) < 2:
            return node
        n, d = Z.shape
        parent = self._gini(y) * n
        best = (parent - 1e-9, None, 0.0)
        for j in range(d):
            x = Z[:, j]
            order = np.argsort(x)
            xs, ys = x[order], y[order]
            uniq = np.unique(xs)
            if uniq.size < 2:
                continue
            cands = (uniq[:-1] + uniq[1:]) / 2.0
            if cands.size > 48:
                cands = np.quantile(xs, np.linspace(0, 1, 48))
            for thr in cands:
                left = xs <= thr
                nl, nr = int(left.sum()), int((~left).sum())
                if nl < self.min_leaf or nr < self.min_leaf:
                    continue
                cost = self._gini(ys[left]) * nl + self._gini(ys[~left]) * nr
                if cost < best[0]:
                    best = (cost, j, float(thr))
        if best[1] is None:
            return node
        _, j, thr = best
        left_mask = Z[:, j] <= thr
        node.feature = j
        node.threshold = thr
        node.column = self.columns[j]
        node.left = self._build(Z[left_mask], y[left_mask], depth + 1)
        node.right = self._build(Z[~left_mask], y[~left_mask], depth + 1)
        return node

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.scaler.transform(X, standardize=False)
        return np.array([self._predict_row(z, self.root) for z in Z])

    def _predict_row(self, z: np.ndarray, node: _TreeNode | None) -> float:
        if node is None:
            return 0.0
        if node.feature is None:
            return node.value
        nxt = node.left if z[node.feature] <= node.threshold else node.right
        return self._predict_row(z, nxt)

    def rules_text(self) -> list[str]:
        lines: list[str] = []

        def walk(node: _TreeNode | None, depth: int, prefix: str) -> None:
            if node is None:
                return
            pad = "  " * depth
            if node.feature is None:
                lines.append(f"{pad}{prefix}=> accept_prob={node.value:.3f}")
                return
            lines.append(f"{pad}{prefix}if {node.column} <= {node.threshold:.4g}:")
            walk(node.left, depth + 1, "")
            lines.append(f"{pad}else ({node.column} > {node.threshold:.4g}):")
            walk(node.right, depth + 1, "")

        walk(self.root, 0, "")
        return lines


@dataclass
class CalibratedLogisticGate:
    """Logistic gate followed by 1-D Platt (sigmoid) calibration of its scores."""

    base: LogisticGate = field(default_factory=LogisticGate)
    a: float = 1.0
    b: float = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CalibratedLogisticGate":
        self.base.fit(X, y)
        raw = self.base.predict_proba(X)
        logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1 - 1e-6))
        self.a, self.b = _fit_platt(logit, np.asarray(y, dtype=float))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.base.predict_proba(X)
        logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1 - 1e-6))
        return _sigmoid(self.a * logit + self.b)


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _fit_platt(scores: np.ndarray, y: np.ndarray, iters: int = 2000, lr: float = 0.1) -> tuple[float, float]:
    a, b = 1.0, 0.0
    n = len(y)
    if n == 0:
        return a, b
    s = (scores - scores.mean()) / (scores.std() + 1e-9)
    for _ in range(iters):
        p = _sigmoid(a * s + b)
        ga = float(np.mean((p - y) * s))
        gb = float(np.mean(p - y))
        a -= lr * ga
        b -= lr * gb
    # fold the standardization back into a/b
    return a / (scores.std() + 1e-9), b - a * scores.mean() / (scores.std() + 1e-9)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    if probs.size == 0:
        return float("nan")
    return float(np.mean((probs - y) ** 2))


def classification_at_threshold(probs: np.ndarray, y: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and np.isfinite(precision) and np.isfinite(recall)) else float("nan")
    acc = (tp + tn) / len(y) if len(y) else float("nan")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def calibration_curve(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if m.sum() == 0:
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "mean_pred": float("nan"), "frac_pos": float("nan"), "count": 0})
        else:
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "mean_pred": float(probs[m].mean()), "frac_pos": float(y[m].mean()), "count": int(m.sum())})
    return rows


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    rows = calibration_curve(probs, y, n_bins)
    n = sum(r["count"] for r in rows)
    if n == 0:
        return float("nan")
    return float(sum(r["count"] / n * abs(r["mean_pred"] - r["frac_pos"]) for r in rows if r["count"] > 0))


def coverage_accuracy_curve(scores: np.ndarray, y: np.ndarray, fractions: Sequence[float] | None = None) -> list[dict[str, float]]:
    """Accept the top-``coverage`` fraction by score; report precision of accepted."""

    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if fractions is None:
        fractions = [round(f, 3) for f in np.linspace(0.05, 1.0, 20)]
    order = np.argsort(-scores)
    ys = y[order]
    rows = []
    base = float(y.mean()) if n else float("nan")
    for cov in fractions:
        k = max(1, int(round(cov * n)))
        acc = float(ys[:k].mean())
        rows.append({
            "coverage": float(cov),
            "n_accepted": int(k),
            "accepted_precision": acc,
            "accepted_lift_over_base": acc - base if np.isfinite(base) else float("nan"),
            "abstention": float(1.0 - cov),
        })
    return rows


def top_x_percent_precision(scores: np.ndarray, y: np.ndarray, fractions: Sequence[float]) -> dict[str, float]:
    rows = coverage_accuracy_curve(scores, y, fractions)
    return {f"top_{int(r['coverage']*100)}pct_precision": r["accepted_precision"] for r in rows}


def evaluate_predictions(probs: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "auroc": auroc(probs, y),
        "auprc": auprc(probs, y),
        "brier": brier_score(probs, y),
        "ece": expected_calibration_error(probs, y),
    }
    out.update(classification_at_threshold(probs, y, 0.5))
    return out


# --------------------------------------------------------------------------- #
# Model registry + split evaluation
# --------------------------------------------------------------------------- #
def model_factories() -> dict[str, Callable[[], Any]]:
    return {
        "threshold_rule": lambda: ThresholdGate(),
        "logistic_regression": lambda: LogisticGate(),
        "decision_tree_d3": lambda: DecisionTreeGate(max_depth=3),
        "calibrated_logistic": lambda: CalibratedLogisticGate(),
    }


def fit_predict(model_name: str, train: pd.DataFrame, test: pd.DataFrame, feature_columns: Sequence[str], label_column: str) -> tuple[np.ndarray, np.ndarray, Any]:
    factory = model_factories()[model_name]
    model = factory()
    ytr = train[label_column].to_numpy(dtype=float)
    yte = test[label_column].to_numpy(dtype=float)
    model.fit(train[list(feature_columns)], ytr)
    probs = model.predict_proba(test[list(feature_columns)])
    return probs, yte, model


def evaluate_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    label_column: str = PRIMARY_LABEL,
    models: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    models = list(models or model_factories().keys())
    train = train[np.isfinite(train[label_column].to_numpy(dtype=float))]
    test = test[np.isfinite(test[label_column].to_numpy(dtype=float))]
    results: dict[str, dict[str, Any]] = {}
    if len(train) < 10 or len(test) < 5 or len(np.unique(train[label_column])) < 2:
        for m in models:
            results[m] = {"n": int(len(test)), "auroc": float("nan"), "note": "insufficient/degenerate data"}
        return results
    for m in models:
        try:
            probs, yte, model = fit_predict(m, train, test, feature_columns, label_column)
            metrics = evaluate_predictions(probs, yte)
            metrics["probs"] = probs
            metrics["y"] = yte
            metrics["model"] = model
            results[m] = metrics
        except Exception as e:  # pragma: no cover - defensive
            results[m] = {"n": int(len(test)), "auroc": float("nan"), "note": f"error: {e}"}
    return results


def leave_one_benchmark_out(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    label_column: str = PRIMARY_LABEL,
    model_name: str = "logistic_regression",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Train on all-but-one benchmark, test on the held-out one. Returns a per-fold
    table plus pooled out-of-fold (scores, labels) for a global LOBO AUROC."""

    benchmarks = sorted(df["benchmark"].unique())
    rows = []
    pooled_scores: list[float] = []
    pooled_y: list[float] = []
    for held in benchmarks:
        train = df[df["benchmark"] != held]
        test = df[df["benchmark"] == held]
        res = evaluate_split(train, test, feature_columns, label_column, models=[model_name])[model_name]
        row = {
            "held_out_benchmark": held,
            "model": model_name,
            "n_test": res.get("n"),
            "base_rate": res.get("base_rate"),
            "auroc": res.get("auroc"),
            "auprc": res.get("auprc"),
            "brier": res.get("brier"),
            "accuracy": res.get("accuracy"),
            "precision": res.get("precision"),
            "recall": res.get("recall"),
            "f1": res.get("f1"),
            "note": res.get("note", ""),
        }
        rows.append(row)
        if "probs" in res:
            pooled_scores.extend(list(res["probs"]))
            pooled_y.extend(list(res["y"]))
    return pd.DataFrame(rows), np.asarray(pooled_scores), np.asarray(pooled_y)


# --------------------------------------------------------------------------- #
# Conservative support flag
# --------------------------------------------------------------------------- #
@dataclass
class SupportCriteria:
    min_lobo_auroc: float = 0.75
    min_accepted_precision: float = 0.80
    min_accepted_coverage: float = 0.25
    accept_coverage_grid: tuple[float, ...] = (0.25, 0.3, 0.4, 0.5)


def evaluate_support_flag(
    *,
    label_free_ok: bool,
    no_leakage_ok: bool,
    lobo_auroc: float,
    coverage_accuracy_rows: Sequence[dict[str, float]],
    real_evidence: bool,
    coco_reported_separately: bool,
    final_metrics_unchanged: bool,
    criteria: SupportCriteria | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Set ``predictive_gate_supported`` only if every conservative criterion holds.

    A high-confidence operating point is the *largest* accepted coverage >=
    ``min_accepted_coverage`` whose accepted precision >= ``min_accepted_precision``.
    """

    c = criteria or SupportCriteria()
    reasons: list[str] = []

    if not real_evidence:
        reasons.append("input artifacts are not from a real pretrained backend (fake/unknown provenance)")
    if not label_free_ok:
        reasons.append("features are not label-free")
    if not no_leakage_ok:
        reasons.append("oracle/label leakage check failed")
    if not (np.isfinite(lobo_auroc) and lobo_auroc >= c.min_lobo_auroc):
        reasons.append(f"leave-one-benchmark-out AUROC {lobo_auroc:.3f} < {c.min_lobo_auroc}")
    if not coco_reported_separately:
        reasons.append("COCO-Text held-out performance not reported separately")
    if not final_metrics_unchanged:
        reasons.append("final headline metrics changed")

    # best high-confidence operating point
    best_cov = None
    best_prec = float("nan")
    for r in sorted(coverage_accuracy_rows, key=lambda z: z["coverage"]):
        if r["coverage"] >= c.min_accepted_coverage and r["accepted_precision"] >= c.min_accepted_precision:
            best_cov = r["coverage"]
            best_prec = r["accepted_precision"]
    if best_cov is None:
        reasons.append(
            f"no operating point with coverage >= {c.min_accepted_coverage} reaches precision >= {c.min_accepted_precision}"
        )

    supported = len(reasons) == 0
    evidence = {
        "lobo_auroc": float(lobo_auroc) if np.isfinite(lobo_auroc) else None,
        "accepted_coverage": best_cov,
        "accepted_precision": best_prec if np.isfinite(best_prec) else None,
        "real_evidence": bool(real_evidence),
        "label_free_ok": bool(label_free_ok),
        "no_leakage_ok": bool(no_leakage_ok),
        "coco_reported_separately": bool(coco_reported_separately),
        "final_metrics_unchanged": bool(final_metrics_unchanged),
    }
    return supported, reasons, evidence
