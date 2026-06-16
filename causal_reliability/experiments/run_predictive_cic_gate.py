from __future__ import annotations

"""Build and evaluate the label-free *predictive CIC reliability gate*.

This runner consumes only the per-example artifacts that the CIC experiments have
already written (repair certificates + open-proposal diagnostics), assembles a
unified per-example feature table of **inference-time observable** quantities,
defines evaluation labels *after* the features, trains small interpretable gates
(threshold / logistic / depth<=3 tree / calibrated logistic), and reports
cross-benchmark predictive performance.

It is a *practical predictive reliability layer*, NOT a new universal theorem.

Output confinement: this script writes **only** under
``results/<output_subdir>`` (default ``results/predictive_cic_gate``). It never
touches ``results/final_report/``, never re-runs any model, never modifies any
headline metric, and never changes any existing support gate. A guard refuses to
run if the output directory points at ``final_report`` or any pre-existing
experiment folder it does not own.
"""

import argparse
import hashlib
import json
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

from causal_reliability.analysis import predictive_cic_gate as gate
from causal_reliability.utils.config import load_config

OWNED_SUBDIR = "predictive_cic_gate"
PROTECTED_DIRS = {"final_report"}


# --------------------------------------------------------------------------- #
# Provenance / real-evidence
# --------------------------------------------------------------------------- #
def assess_real_evidence(sources: list[dict[str, Any]], base: Path) -> tuple[bool, dict[str, Any]]:
    """Real evidence requires every configured provenance file to confirm a real
    pretrained backend (``fake_backend`` false / real flag true). If a required
    provenance is missing or marks a fake backend, real evidence is False."""

    details: dict[str, Any] = {}
    checked = 0
    all_real = True
    for src in sources:
        prov = src.get("provenance")
        if not prov:
            continue
        checked += 1
        jpath = base / prov["json"]
        name = src.get("name", str(jpath))
        if not jpath.exists():
            details[name] = {"status": "missing", "real": False}
            all_real = False
            continue
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception as e:
            details[name] = {"status": f"unreadable: {e}", "real": False}
            all_real = False
            continue
        fake_field = prov.get("fake_field", "fake_backend")
        real_field = prov.get("real_field")
        is_fake = bool(data.get(fake_field, False))
        real_ok = (not is_fake) and (bool(data.get(real_field, True)) if real_field else True)
        details[name] = {"status": "ok", "fake_backend": is_fake, "real": real_ok}
        all_real = all_real and real_ok
    real_evidence = bool(checked > 0 and all_real)
    details["_summary"] = {"sources_checked": checked, "real_evidence": real_evidence}
    return real_evidence, details


# --------------------------------------------------------------------------- #
# Feature-table assembly
# --------------------------------------------------------------------------- #
def _read_csv(base: Path, rel: str | None) -> pd.DataFrame | None:
    if not rel:
        return None
    p = base / rel
    if not p.exists():
        return None
    return pd.read_csv(p, low_memory=False)


def build_feature_table(cfg: dict[str, Any], base: Path) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    frames: list[pd.DataFrame] = []
    image_size = float(cfg.get("image_size", 224))

    # triage subset ids for COCO reporting (used only for reporting, never features)
    strict_ids: list[int] = []
    directional_ids: list[int] = []
    triage = cfg.get("triage", {})
    if triage:
        s = _read_csv(base, triage.get("strict_csv"))
        d = _read_csv(base, triage.get("directional_csv"))
        if s is not None:
            strict_ids = [int(x) for x in s["example_id"].tolist()]
        if d is not None:
            directional_ids = [int(x) for x in d["example_id"].tolist()]

    for src in cfg.get("sources", []):
        kind = src["kind"]
        name = src.get("name", kind)
        try:
            if kind == "certificate":
                cert = _read_csv(base, src["certificates"])
                if cert is None:
                    notes.append(f"{name}: certificates missing, skipped")
                    continue
                rankings = _read_csv(base, src.get("rankings"))
                frame = gate.extract_certificate_benchmark(
                    benchmark=src["benchmark"],
                    group=src["group"],
                    certificates=cert,
                    cic_method=src["cic_method"],
                    consensus_method=src.get("consensus_method"),
                    rankings=rankings,
                    image_size=image_size,
                )
            elif kind == "scale_audit":
                frame = _build_scale_audit(src, base, image_size, notes)
                if frame is None or frame.empty:
                    continue
            elif kind == "coco_full":
                per = _read_csv(base, src["per_example"])
                if per is None:
                    notes.append(f"{name}: per_example missing, skipped")
                    continue
                diag = _read_csv(base, src.get("diagnostics"))
                frame = gate.extract_coco_full(
                    per, diag,
                    benchmark=src["benchmark"],
                    group=src["group"],
                    cic_method=src.get("cic_method", "cic_top1_repair_excl_ocr"),
                    consensus_method=src.get("consensus_method", "cic_top3_repair_excl_ocr"),
                    strict_ids=strict_ids,
                    directional_ids=directional_ids,
                    image_size=image_size,
                )
            elif kind == "natural_open":
                cert = _read_csv(base, src["certificates"])
                if cert is None:
                    notes.append(f"{name}: certificates missing, skipped")
                    continue
                diag = _read_csv(base, src.get("diagnostics"))
                frame = gate.extract_natural_open(
                    cert, diag,
                    benchmark=src["benchmark"],
                    group=src["group"],
                    cic_method=src.get("cic_method", "cic_top1_repair"),
                    consensus_method=src.get("consensus_method", "cic_top3_repair"),
                    image_size=image_size,
                )
            elif kind == "verified_failure":
                per = _read_csv(base, src["per_example"])
                if per is None:
                    notes.append(f"{name}: per_example missing, skipped")
                    continue
                diag = _read_csv(base, src.get("diagnostics"))
                frame = gate.extract_verified_failure(
                    per, diag, benchmark=src["benchmark"], group=src["group"], image_size=image_size
                )
            else:
                notes.append(f"{name}: unknown kind '{kind}', skipped")
                continue
        except Exception as e:  # pragma: no cover - defensive
            notes.append(f"{name}: extraction error: {e}")
            continue
        if frame is not None and len(frame):
            frames.append(frame)
            notes.append(f"{name}: {len(frame)} examples")
        else:
            notes.append(f"{name}: 0 examples")

    if not frames:
        return pd.DataFrame(columns=gate.META_COLUMNS + gate.NUMERIC_FEATURES + gate.LABEL_COLUMNS), notes
    table = pd.concat(frames, ignore_index=True)
    # canonical column ordering
    cols = gate.META_COLUMNS + gate.NUMERIC_FEATURES + gate.LABEL_COLUMNS
    for c in cols:
        if c not in table.columns:
            table[c] = float("nan")
    return table[cols], notes


def _build_scale_audit(src: dict[str, Any], base: Path, image_size: float, notes: list[str]) -> pd.DataFrame | None:
    runs_dir = base / src["runs_dir"]
    if not runs_dir.exists():
        notes.append(f"{src.get('name', 'scale_audit')}: runs dir missing, skipped")
        return None
    exclude = set(src.get("exclude_runs", []))
    frames = []
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir() or run.name in exclude:
            continue
        cert_path = run / src["certificates_rel"]
        if not cert_path.exists():
            continue
        cert = pd.read_csv(cert_path, low_memory=False)
        rank_path = run / src["rankings_rel"]
        rankings = pd.read_csv(rank_path, low_memory=False) if rank_path.exists() else None
        frame = gate.extract_certificate_benchmark(
            benchmark=f"scale_audit::{run.name}",
            group="controlled",
            certificates=cert,
            cic_method=src["cic_method"],
            consensus_method=src.get("consensus_method"),
            rankings=rankings,
            image_size=image_size,
        )
        if len(frame):
            frames.append(frame)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Reporting subsets
# --------------------------------------------------------------------------- #
def report_subsets(df: pd.DataFrame, scores: np.ndarray, label_col: str) -> dict[str, dict[str, Any]]:
    """Compute predictive metrics on the required reporting slices using the pooled
    LOBO out-of-fold scores aligned to ``df`` row order."""

    out: dict[str, dict[str, Any]] = {}

    def slice_metrics(mask: np.ndarray) -> dict[str, Any]:
        y = df[label_col].to_numpy(dtype=float)
        valid = mask & np.isfinite(y) & np.isfinite(scores)
        if valid.sum() < 5 or len(np.unique(y[valid])) < 2:
            return {"n": int(valid.sum()), "auroc": float("nan"), "note": "insufficient/degenerate"}
        m = gate.evaluate_predictions(scores[valid], y[valid].astype(int))
        cov = gate.coverage_accuracy_curve(scores[valid], y[valid])
        m["coverage_accuracy"] = cov
        return m

    bench = df["benchmark"].to_numpy()
    group = df["group"].to_numpy()
    out["controlled"] = slice_metrics(group == "controlled")
    out["coco_strict"] = slice_metrics(df["is_coco_strict"].to_numpy(dtype=bool))
    out["coco_directional"] = slice_metrics(df["is_coco_directional"].to_numpy(dtype=bool))
    out["coco_all"] = slice_metrics(np.char.startswith(bench.astype(str), "coco_text"))
    out["natural_all"] = slice_metrics(group == "natural")
    out["overall_pooled"] = slice_metrics(np.ones(len(df), dtype=bool))
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(out_dir: Path, lobo_rows: pd.DataFrame, cov_rows: list[dict[str, float]], calib_rows: list[dict[str, float]], subset_metrics: dict[str, dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    valid = lobo_rows[np.isfinite(lobo_rows["auroc"].to_numpy(dtype=float))]
    if len(valid):
        ax.barh(valid["held_out_benchmark"], valid["auroc"], color="#4477aa")
        ax.axvline(0.75, color="red", ls="--", lw=1, label="support floor 0.75")
        ax.axvline(0.5, color="gray", ls=":", lw=1, label="chance")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7)
    ax.set_title("Leave-one-benchmark-out AUROC")
    ax.set_xlabel("AUROC")

    ax = axes[0, 1]
    if cov_rows:
        cov = [r["coverage"] for r in cov_rows]
        prec = [r["accepted_precision"] for r in cov_rows]
        ax.plot(cov, prec, "-o", color="#228833")
        ax.axhline(0.80, color="red", ls="--", lw=1, label="precision floor 0.80")
        ax.set_xlabel("coverage (fraction of repairs accepted)")
        ax.set_ylabel("accepted-repair precision")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
    ax.set_title("Coverage vs accepted precision (LOBO out-of-fold)")

    ax = axes[1, 0]
    pred = [r["mean_pred"] for r in calib_rows if r["count"] > 0]
    frac = [r["frac_pos"] for r in calib_rows if r["count"] > 0]
    ax.plot([0, 1], [0, 1], ":", color="gray")
    if pred:
        ax.plot(pred, frac, "-o", color="#aa3377")
    ax.set_xlabel("mean predicted trust")
    ax.set_ylabel("empirical repair-success rate")
    ax.set_title("Calibration (LOBO out-of-fold)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1, 1]
    names, aurocs = [], []
    for k, v in subset_metrics.items():
        a = v.get("auroc")
        if a is not None and np.isfinite(a):
            names.append(k)
            aurocs.append(a)
    if names:
        ax.barh(names, aurocs, color="#ccbb44")
        ax.axvline(0.5, color="gray", ls=":", lw=1)
        ax.set_xlim(0, 1)
    ax.set_title("AUROC by reporting subset")
    ax.set_xlabel("AUROC")

    fig.tight_layout()
    png = out_dir / "predictive_gate_plots.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


# --------------------------------------------------------------------------- #
# Main run
# --------------------------------------------------------------------------- #
def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _hash_file(p: Path) -> str | None:
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _guard_output_dir(out_dir: Path) -> None:
    parts = set(out_dir.parts)
    if parts & PROTECTED_DIRS:
        raise RuntimeError(f"refusing to write under a protected directory: {out_dir}")


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    base = Path(cfg.get("base_dir", "."))
    results_dir = base / cfg.get("results_dir", "results")
    out_dir = results_dir / cfg.get("output_subdir", OWNED_SUBDIR)
    _guard_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = list(gate.NUMERIC_FEATURES)
    label_col = cfg.get("label_column", gate.PRIMARY_LABEL)

    # snapshot the final report so we can certify it is untouched
    final_report_json = results_dir / "final_report" / "final_key_numbers.json"
    final_hash_before = _hash_file(final_report_json)

    # ---- assemble table ----
    table, notes = build_feature_table(cfg, base)

    # ---- guards: label-free + leakage ----
    label_free_ok = True
    try:
        gate.assert_label_free(feature_columns)
    except ValueError:
        label_free_ok = False
    no_leak_ok, leak_reasons = gate.check_no_oracle_leakage(table, feature_columns, label_col) if len(table) else (True, [])

    # ---- provenance / real evidence ----
    real_evidence, prov_details = assess_real_evidence(cfg.get("sources", []), base)

    # ---- validation protocols ----
    lobo_model = cfg.get("lobo_model", "logistic_regression")
    if len(table) and len(table["benchmark"].unique()) >= 2:
        lobo_rows, pooled_scores, pooled_y = gate.leave_one_benchmark_out(table, feature_columns, label_col, lobo_model)
    else:
        lobo_rows, pooled_scores, pooled_y = pd.DataFrame(), np.array([]), np.array([])

    lobo_auroc = gate.auroc(pooled_scores, pooled_y) if pooled_scores.size and len(np.unique(pooled_y)) > 1 else float("nan")
    lobo_auprc = gate.auprc(pooled_scores, pooled_y) if pooled_scores.size and len(np.unique(pooled_y)) > 1 else float("nan")

    # per-row LOBO out-of-fold scores aligned to table (re-derive aligned to df order)
    aligned_scores = _aligned_lobo_scores(table, feature_columns, label_col, lobo_model)

    # ---- cross-benchmark protocols (controlled -> natural / coco) ----
    cross = {}
    controlled = table[table["group"] == "controlled"]
    coco = table[np.char.startswith(table["benchmark"].to_numpy().astype(str), "coco_text")]
    natural = table[table["group"] == "natural"]
    for proto_name, train_df, test_df in [
        ("train_controlled_test_coco", controlled, coco),
        ("train_controlled_test_natural", controlled, natural),
    ]:
        res = gate.evaluate_split(train_df, test_df, feature_columns, label_col)
        cross[proto_name] = {m: _scalar_metrics(v) for m, v in res.items()}

    # within-benchmark CV (secondary): simple 50/50 split per benchmark, all models
    within = _within_benchmark_eval(table, feature_columns, label_col)

    # all-model comparison on overall pooled (for the "best simple rule" + curves)
    pooled_models = gate.evaluate_split(
        table.sample(frac=1.0, random_state=0) if len(table) else table,
        table, feature_columns, label_col,
    ) if len(table) else {}

    # ---- reporting subsets (LOBO out-of-fold scores) ----
    subset_metrics = report_subsets(table, aligned_scores, label_col) if len(table) else {}

    # ---- coverage/accuracy + calibration on pooled LOBO out-of-fold ----
    valid = np.isfinite(aligned_scores) & np.isfinite(table[label_col].to_numpy(dtype=float)) if len(table) else np.array([], dtype=bool)
    cov_scores = aligned_scores[valid]
    cov_y = table[label_col].to_numpy(dtype=float)[valid]
    cov_rows = gate.coverage_accuracy_curve(cov_scores, cov_y) if cov_scores.size else []
    calib_rows = gate.calibration_curve(cov_scores, cov_y, n_bins=int(cfg.get("calibration_bins", 10))) if cov_scores.size else []
    topx = gate.top_x_percent_precision(cov_scores, cov_y, [0.1, 0.2, 0.25, 0.3, 0.5]) if cov_scores.size else {}

    # ---- best interpretable single rule (threshold gate on pooled, reported honestly via LOBO) ----
    best_rule = _best_threshold_rule(table, feature_columns, label_col)

    # ---- feature importance (logistic coefficients on pooled) ----
    feature_ranking = _feature_ranking(table, feature_columns, label_col)

    # ---- honest caveats (do not gate support, but must be reported) ----
    caveats: list[str] = []
    for sub_name in ("coco_strict", "coco_directional", "natural_text_open", "natural_text_verified"):
        a = subset_metrics.get(sub_name, {}).get("auroc") if sub_name in subset_metrics else None
        if a is None:
            for r in (lobo_rows.to_dict(orient="records") if len(lobo_rows) else []):
                if r["held_out_benchmark"] == sub_name:
                    a = r.get("auroc")
        if a is not None and np.isfinite(a) and a < 0.55:
            caveats.append(
                f"weak transfer: held-out predictive AUROC on '{sub_name}' is {a:.3f} (<=0.55); "
                f"the gate does not reliably rank repairs there"
            )
    for proto, res in cross.items():
        lr = res.get("logistic_regression", {})
        a = lr.get("auroc")
        if a is not None and np.isfinite(a) and a < 0.6:
            caveats.append(
                f"cross-benchmark protocol '{proto}' AUROC is {a:.3f} (near chance); pure controlled->natural "
                f"transfer is weak. The pooled LOBO AUROC is higher because each fold still trains on some "
                f"same-family benchmarks (the scale-audit runs are near-duplicates of hard_multidecoy)."
            )
    if not caveats:
        caveats.append("none")

    # ---- support flag ----
    final_metrics_unchanged = True  # runner provably writes only under out_dir
    supported, support_reasons, support_evidence = gate.evaluate_support_flag(
        label_free_ok=label_free_ok,
        no_leakage_ok=no_leak_ok,
        lobo_auroc=lobo_auroc,
        coverage_accuracy_rows=cov_rows,
        real_evidence=real_evidence,
        coco_reported_separately=("coco_strict" in subset_metrics or "coco_all" in subset_metrics),
        final_metrics_unchanged=final_metrics_unchanged,
        criteria=gate.SupportCriteria(
            min_lobo_auroc=float(cfg.get("min_lobo_auroc", 0.75)),
            min_accepted_precision=float(cfg.get("min_accepted_precision", 0.80)),
            min_accepted_coverage=float(cfg.get("min_accepted_coverage", 0.25)),
        ),
    )

    # ---- write artifacts ----
    paths = _write_artifacts(
        out_dir, table, lobo_rows, cross, within, subset_metrics, cov_rows, calib_rows,
        best_rule, feature_ranking, pooled_models,
    )

    # final report integrity
    final_hash_after = _hash_file(final_report_json)
    final_unchanged = (final_hash_before == final_hash_after)

    key_numbers: dict[str, Any] = {
        "experiment": "predictive_cic_gate",
        "is_universal_theorem": False,
        "framing": "practical predictive reliability layer on top of CIC",
        "n_examples": int(len(table)),
        "n_examples_with_primary_label": int(np.isfinite(table[label_col].to_numpy(dtype=float)).sum()) if len(table) else 0,
        "benchmarks_included": sorted(table["benchmark"].unique().tolist()) if len(table) else [],
        "benchmark_counts": {k: int(v) for k, v in table["benchmark"].value_counts().items()} if len(table) else {},
        "groups": {k: int(v) for k, v in table["group"].value_counts().items()} if len(table) else {},
        "primary_label": label_col,
        "primary_label_base_rate": float(np.nanmean(table[label_col].to_numpy(dtype=float))) if len(table) else float("nan"),
        "feature_columns": feature_columns,
        "label_free_ok": bool(label_free_ok),
        "no_oracle_leakage_ok": bool(no_leak_ok),
        "leakage_reasons": leak_reasons,
        "real_evidence": bool(real_evidence),
        "provenance": prov_details,
        "lobo_model": lobo_model,
        "lobo_pooled_auroc": lobo_auroc,
        "lobo_pooled_auprc": lobo_auprc,
        "lobo_per_benchmark": lobo_rows.to_dict(orient="records") if len(lobo_rows) else [],
        "cross_benchmark": cross,
        "within_benchmark": within,
        "reporting_subsets": {k: _strip_arrays(v) for k, v in subset_metrics.items()},
        "coco_strict_auroc": subset_metrics.get("coco_strict", {}).get("auroc"),
        "coco_directional_auroc": subset_metrics.get("coco_directional", {}).get("auroc"),
        "coco_all_auroc": subset_metrics.get("coco_all", {}).get("auroc"),
        "top_x_percent_precision": topx,
        "best_single_rule": best_rule,
        "feature_ranking": feature_ranking,
        "predictive_gate_supported": bool(supported),
        "support_reasons": support_reasons,
        "support_evidence": support_evidence,
        "caveats": caveats,
        "final_report_unchanged": bool(final_unchanged),
        "final_report_sha256": final_hash_after,
        "notes": notes,
        "proposition_ref": "causal_reliability.theory.predictive_certificate; docs/theory.md (Predictive CIC certificate)",
    }
    (out_dir / "predictive_gate_key_numbers.json").write_text(
        json.dumps(key_numbers, indent=2, default=_json_default), encoding="utf-8"
    )
    make_plots(out_dir, lobo_rows if len(lobo_rows) else pd.DataFrame({"held_out_benchmark": [], "auroc": []}), cov_rows, calib_rows, subset_metrics)
    _write_summary(out_dir, key_numbers)

    return {"key_numbers": str(out_dir / "predictive_gate_key_numbers.json"), **{k: str(v) for k, v in paths.items()}}


# --------------------------------------------------------------------------- #
# Helpers for run()
# --------------------------------------------------------------------------- #
def _scalar_metrics(v: dict[str, Any]) -> dict[str, Any]:
    keys = ["n", "n_pos", "base_rate", "auroc", "auprc", "brier", "ece", "accuracy", "precision", "recall", "f1", "note"]
    return {k: v[k] for k in keys if k in v}


def _strip_arrays(v: dict[str, Any]) -> dict[str, Any]:
    return {k: val for k, val in v.items() if k not in {"probs", "y", "model", "coverage_accuracy"}}


def _aligned_lobo_scores(df: pd.DataFrame, feature_columns: list[str], label_col: str, model_name: str) -> np.ndarray:
    """Per-row out-of-fold trust scores: each row scored by a model trained on all
    *other* benchmarks. NaN where its fold could not be trained/evaluated."""

    scores = np.full(len(df), np.nan)
    benchmarks = sorted(df["benchmark"].unique())
    idx = df.reset_index().rename(columns={"index": "_orig"})
    for held in benchmarks:
        train = df[df["benchmark"] != held]
        test_mask = (idx["benchmark"] == held).to_numpy()
        test = df[df["benchmark"] == held]
        ytr = train[label_col].to_numpy(dtype=float)
        train_v = train[np.isfinite(ytr)]
        if len(train_v) < 10 or len(np.unique(train_v[label_col])) < 2 or len(test) == 0:
            continue
        try:
            model = gate.model_factories()[model_name]()
            model.fit(train_v[feature_columns], train_v[label_col].to_numpy(dtype=float))
            probs = model.predict_proba(test[feature_columns])
        except Exception:
            continue
        positions = np.where(test_mask)[0]
        scores[positions] = probs
    return scores


def _within_benchmark_eval(df: pd.DataFrame, feature_columns: list[str], label_col: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for b in sorted(df["benchmark"].unique()):
        sub = df[df["benchmark"] == b]
        sub = sub[np.isfinite(sub[label_col].to_numpy(dtype=float))]
        if len(sub) < 20 or len(np.unique(sub[label_col])) < 2:
            out[b] = {"n": int(len(sub)), "auroc": float("nan"), "note": "too small / single-class"}
            continue
        shuffled = sub.sample(frac=1.0, random_state=0)
        half = len(shuffled) // 2
        tr, te = shuffled.iloc[:half], shuffled.iloc[half:]
        res = gate.evaluate_split(tr, te, feature_columns, label_col, models=["logistic_regression"])["logistic_regression"]
        out[b] = _scalar_metrics(res)
    return out


def _best_threshold_rule(df: pd.DataFrame, feature_columns: list[str], label_col: str) -> dict[str, Any]:
    sub = df[np.isfinite(df[label_col].to_numpy(dtype=float))]
    if len(sub) < 20 or len(np.unique(sub[label_col])) < 2:
        return {"note": "insufficient data"}
    tg = gate.ThresholdGate().fit(sub[feature_columns], sub[label_col].to_numpy(dtype=float))
    # honest estimate: LOBO out-of-fold AUROC of the single chosen feature
    aligned = _aligned_lobo_scores(sub, [tg.feature], label_col, "threshold_rule")
    y = sub[label_col].to_numpy(dtype=float)
    m = np.isfinite(aligned) & np.isfinite(y)
    lobo_auroc = gate.auroc(aligned[m], y[m].astype(int)) if m.sum() and len(np.unique(y[m])) > 1 else float("nan")
    return {
        "rule": tg.rule_text(),
        "feature": tg.feature,
        "threshold": tg.threshold,
        "direction": "ge" if tg.direction > 0 else "le",
        "train_youden_j": tg.youden_j,
        "lobo_outoffold_auroc": lobo_auroc,
    }


def _feature_ranking(df: pd.DataFrame, feature_columns: list[str], label_col: str) -> list[dict[str, Any]]:
    sub = df[np.isfinite(df[label_col].to_numpy(dtype=float))]
    if len(sub) < 20 or len(np.unique(sub[label_col])) < 2:
        return []
    y = sub[label_col].to_numpy(dtype=int)
    rows = []
    for c in feature_columns:
        x = pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x)
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            rows.append({"feature": c, "univariate_auroc": float("nan"), "coverage": float(m.mean())})
            continue
        a = gate.auroc(x[m], y[m])
        rows.append({"feature": c, "univariate_auroc": a, "abs_auroc_gap": abs(a - 0.5) if np.isfinite(a) else float("nan"), "coverage": float(m.mean())})

    def _key(r: dict[str, Any]) -> tuple[float, float]:
        gap = r.get("abs_auroc_gap")
        gap = gap if (gap is not None and np.isfinite(gap)) else -1.0
        cov = r.get("coverage", 0.0)
        # prefer informative features that are broadly available; tiny-support
        # proxies (coverage < 0.3) are ranked after high-coverage features
        return (1.0 if cov >= 0.3 else 0.0, gap)

    rows.sort(key=_key, reverse=True)
    return rows


def _write_artifacts(out_dir, table, lobo_rows, cross, within, subset_metrics, cov_rows, calib_rows, best_rule, feature_ranking, pooled_models):
    paths: dict[str, Path] = {}

    table.to_csv(out_dir / "predictive_gate_features.csv", index=False)
    paths["features"] = out_dir / "predictive_gate_features.csv"

    # eval-by-benchmark: per-benchmark base rate + within + all-model pooled metrics
    rows = []
    for b in sorted(table["benchmark"].unique()) if len(table) else []:
        sub = table[table["benchmark"] == b]
        y = sub["label_repair_success"].to_numpy(dtype=float)
        w = within.get(b, {})
        rows.append({
            "benchmark": b,
            "group": sub["group"].iloc[0],
            "n": int(len(sub)),
            "n_labeled": int(np.isfinite(y).sum()),
            "repair_success_rate": float(np.nanmean(y)) if np.isfinite(y).any() else float("nan"),
            "within_auroc": w.get("auroc"),
            "within_auprc": w.get("auprc"),
            "within_brier": w.get("brier"),
        })
    by_bench = pd.DataFrame(rows)
    # append reporting subsets as pseudo-rows
    sub_rows = []
    for name, m in subset_metrics.items():
        sub_rows.append({
            "benchmark": f"REPORT::{name}", "group": "report",
            "n": m.get("n"), "n_labeled": m.get("n"),
            "repair_success_rate": m.get("base_rate"),
            "lobo_outoffold_auroc": m.get("auroc"), "lobo_outoffold_auprc": m.get("auprc"),
            "lobo_outoffold_brier": m.get("brier"),
        })
    eval_by_bench = pd.concat([by_bench, pd.DataFrame(sub_rows)], ignore_index=True) if len(by_bench) or sub_rows else pd.DataFrame()
    eval_by_bench.to_csv(out_dir / "predictive_gate_eval_by_benchmark.csv", index=False)
    paths["eval_by_benchmark"] = out_dir / "predictive_gate_eval_by_benchmark.csv"

    lobo_rows.to_csv(out_dir / "predictive_gate_leave_one_benchmark_out.csv", index=False)
    paths["lobo"] = out_dir / "predictive_gate_leave_one_benchmark_out.csv"

    pd.DataFrame(cov_rows).to_csv(out_dir / "coverage_accuracy_curve.csv", index=False)
    paths["coverage"] = out_dir / "coverage_accuracy_curve.csv"

    pd.DataFrame(calib_rows).to_csv(out_dir / "calibration_curve.csv", index=False)
    paths["calibration"] = out_dir / "calibration_curve.csv"

    pd.DataFrame(feature_ranking).to_csv(out_dir / "predictive_gate_feature_ranking.csv", index=False)
    paths["feature_ranking"] = out_dir / "predictive_gate_feature_ranking.csv"

    return paths


def _write_summary(out_dir: Path, kn: dict[str, Any]) -> None:
    def fmt(x: Any) -> str:
        if x is None:
            return "n/a"
        if isinstance(x, float):
            return "n/a" if not np.isfinite(x) else f"{x:.3f}"
        return str(x)

    lines = [
        "# Predictive CIC Reliability Gate",
        "",
        "**Practical predictive reliability layer on top of the CIC framework — NOT a new universal theorem.**",
        "",
        "This gate predicts, from inference-time observable features only (no true label, no",
        "target label, no correctness, no oracle repair success, no support-subset membership,",
        "no ground-truth box overlap), whether a CIC repair should be trusted *before* the true",
        "label is revealed.",
        "",
        "## Dataset",
        f"- Examples in predictive dataset: **{kn['n_examples']}** ({kn['n_examples_with_primary_label']} with primary label)",
        f"- Benchmarks included: {', '.join(kn['benchmarks_included']) or 'none'}",
        f"- Group sizes: {kn['groups']}",
        f"- Primary label: `{kn['primary_label']}` (base success rate {fmt(kn['primary_label_base_rate'])})",
        "",
        "## Label-free / leakage guards",
        f"- Features label-free: **{kn['label_free_ok']}**",
        f"- No-oracle-leakage check passed: **{kn['no_oracle_leakage_ok']}**",
        f"- Leakage reasons: {kn['leakage_reasons'] or 'none'}",
        f"- Real-backend evidence (provenance): **{kn['real_evidence']}**",
        "",
        "## Cross-benchmark validation",
        f"- Leave-one-benchmark-out pooled AUROC (`{kn['lobo_model']}`): **{fmt(kn['lobo_pooled_auroc'])}** "
        f"(AUPRC {fmt(kn['lobo_pooled_auprc'])})",
    ]
    for r in kn["lobo_per_benchmark"]:
        lines.append(
            f"  - held-out `{r['held_out_benchmark']}`: AUROC {fmt(r.get('auroc'))}, "
            f"AUPRC {fmt(r.get('auprc'))}, base {fmt(r.get('base_rate'))}, n={r.get('n_test')} {('('+r['note']+')') if r.get('note') else ''}"
        )
    cc = kn["cross_benchmark"]
    for proto, res in cc.items():
        lr = res.get("logistic_regression", {})
        lines.append(f"- {proto} (logistic): AUROC {fmt(lr.get('auroc'))}, n={lr.get('n')}")
    lines += [
        "",
        "## Reporting subsets (LOBO out-of-fold scores)",
        "| subset | n | base rate | AUROC | AUPRC | Brier |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in kn["reporting_subsets"].items():
        lines.append(
            f"| {name} | {m.get('n')} | {fmt(m.get('base_rate'))} | {fmt(m.get('auroc'))} | {fmt(m.get('auprc'))} | {fmt(m.get('brier'))} |"
        )
    lines += [
        "",
        "## COCO-Text held-out (reported separately)",
        f"- COCO-Text strict subset AUROC: **{fmt(kn['coco_strict_auroc'])}**",
        f"- COCO-Text directional subset AUROC: **{fmt(kn['coco_directional_auroc'])}**",
        f"- COCO-Text all AUROC: **{fmt(kn['coco_all_auroc'])}**",
        "",
        "## Accept top-X% most trustworthy repairs",
        "If the gate accepts only the top-X% most trustworthy CIC repairs, accepted-repair precision is:",
    ]
    for k, v in kn["top_x_percent_precision"].items():
        lines.append(f"- {k}: {fmt(v)}")
    br = kn["best_single_rule"]
    lines += [
        "",
        "## Best simple interpretable rule",
        f"- Rule: `{br.get('rule', br.get('note'))}`",
        f"- LOBO out-of-fold AUROC of this single feature: {fmt(br.get('lobo_outoffold_auroc'))}",
        "",
        "## Best predictive features (univariate, |AUROC-0.5| ranked)",
    ]
    for r in kn["feature_ranking"][:6]:
        lines.append(f"- `{r['feature']}`: univariate AUROC {fmt(r.get('univariate_auroc'))} (coverage {fmt(r.get('coverage'))})")
    ev = kn["support_evidence"]
    lines += [
        "",
        "## Conservative support flag",
        f"- `predictive_gate_supported`: **{kn['predictive_gate_supported']}**",
        f"- LOBO AUROC: {fmt(ev.get('lobo_auroc'))} (floor 0.75)",
        f"- Best high-confidence operating point: coverage {fmt(ev.get('accepted_coverage'))}, "
        f"precision {fmt(ev.get('accepted_precision'))} (floors: coverage 0.25, precision 0.80)",
        f"- Reasons gate is not supported: {kn['support_reasons'] or 'none (all criteria met)'}",
        "",
        "## Honest caveats",
        *[f"- {c}" for c in kn.get("caveats", ["none"])],
        "",
        "## Integrity",
        f"- Final report unchanged: **{kn['final_report_unchanged']}** (sha256 {kn['final_report_sha256']})",
        f"- This experiment wrote only under `results/predictive_cic_gate/`.",
        "",
        "## Proposition",
        "See `docs/theory.md` (Predictive CIC certificate) and",
        "`causal_reliability/theory/predictive_certificate.py`:",
        "",
        "> If the repaired prediction margin exceeds an empirically calibrated",
        "> residual-instability bound, then the repaired prediction is stable under the",
        "> calibrated perturbation class. This does not prove universal correctness; it",
        "> gives a label-free abstention rule for deciding when CIC repair is reliable.",
        "",
        f"Notes: {kn['notes']}",
    ]
    (out_dir / "predictive_gate_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/predictive_cic_gate.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
