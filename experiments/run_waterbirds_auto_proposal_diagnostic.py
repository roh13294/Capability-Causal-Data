from __future__ import annotations

"""Automated finite-candidate proposal CIC diagnostic on WILDS Waterbirds.

Experiment name: ``waterbirds_auto_proposal_diagnostic``.

Scientific question
-------------------
On Waterbirds (a natural-image background-shortcut benchmark), can automatically
generated finite candidate regions, scored by CIC and neutralized, reduce the
model's background sensitivity without destroying label accuracy?

Important caveat
----------------
This benchmark ships **no oracle bird/background masks** locally. We therefore
generate foreground/background-ish proposals automatically (grid / edge-component
/ saliency / optional SAM), score them with CIC, and neutralize the selected
region. Because there is no oracle mask to define "the correct repair region",
this is a **diagnostic**, NOT full validation. CIC still scores a finite
candidate set generated from pixels.

Honesty / scope
---------------
* Writes ONLY under ``results/auto_proposal_pilot/``.
* If local Waterbirds images are unavailable, it skips cleanly with a clear
  reason and exits 0 (it never downloads WILDS).
* No model downloads unless ``--allow-download`` is passed.
* No universal-robustness, deployment, or clinical claim.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

cache_root = Path(tempfile.gettempdir()) / "causal_reliability_matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_reliability.discovery.cic_region_scoring import neutralize_region, score_region_candidates
from causal_reliability.proposals.auto_proposals import (
    ALL_FAMILIES,
    CLASSICAL_FAMILIES,
    generate_proposal_sets,
    generator_availability,
    proposal_sets_to_region_proposals,
)
from causal_reliability.real_models.clip_zero_shot import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED_TAG,
    ClipStatus,
    check_clip_available,
)
from causal_reliability.utils.io import ensure_dir

# Reuse the COCO pilot's predict-fn, json default, family resolver, and the shared
# combined-summary writer so both scripts keep one consistent summary.md.
from experiments.run_auto_proposal_cic_pilot import (
    RESULTS_SUBDIR,
    _build_predict_fn,
    _json_default,
    _resolve_families,
    write_combined_summary,
)


LABEL_NAMES = {0: "landbird", 1: "waterbird"}
PLACE_NAMES = {0: "land", 1: "water"}
ALLOWED_LABELS = ["landbird", "waterbird"]
# Aligned groups: bird type matches its typical background. Conflicting groups are
# the spurious-correlation-breaking ones the model tends to fail on.
ALIGNED_GROUPS = {"landbird_on_land", "waterbird_on_water"}
CONFLICTING_GROUPS = {"landbird_on_water", "waterbird_on_land"}

NON_CLAIMS = [
    "This is automated finite-candidate proposal generation, NOT open-world shortcut discovery.",
    "This is NOT universal repair or general robustness.",
    "This is NOT deployment validation or clinical validation.",
    "No oracle masks are available: this is a diagnostic, NOT full validation.",
]

# Go/no-go thresholds (Waterbirds diagnostic).
PROMISE_WORST_GROUP_GAIN = 0.05
PROMISE_MAX_OVERALL_DROP = 0.03
PROMISE_BG_SENS_DROP = 0.05


def _group_name(y: int, place: int) -> str:
    return f"{LABEL_NAMES.get(int(y), 'bird')}_on_{PLACE_NAMES.get(int(place), 'bg')}"


# --------------------------------------------------------------------------- #
# Data availability + loading (local only; never downloads)
# --------------------------------------------------------------------------- #
def check_dataset(root: Path) -> dict[str, Any]:
    metadata = root / "metadata.csv"
    if not metadata.exists():
        return {"available": False, "reason": f"no Waterbirds metadata.csv under {root}", "metadata": str(metadata)}
    try:
        frame = pd.read_csv(metadata, nrows=4)
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "reason": f"could not read {metadata}: {exc}", "metadata": str(metadata)}
    needed = {"img_filename", "y", "place", "split"}
    if not needed.issubset(set(frame.columns)):
        return {"available": False, "reason": f"metadata missing columns {needed - set(frame.columns)}", "metadata": str(metadata)}
    return {"available": True, "reason": "", "metadata": str(metadata)}


def _sample_rows(frame: pd.DataFrame, split_code: int, max_examples: int, seed: int) -> pd.DataFrame:
    test = frame[frame["split"] == split_code]
    if test.empty:
        test = frame
    rng = np.random.default_rng(seed)
    groups = list(test.groupby(["y", "place"]))
    per_group = max(1, int(max_examples) // max(1, len(groups)))
    picks: list[pd.DataFrame] = []
    for _, g in groups:
        n = min(len(g), per_group)
        idx = rng.choice(len(g), size=n, replace=False)
        picks.append(g.iloc[sorted(idx.tolist())])
    out = pd.concat(picks, axis=0) if picks else test.iloc[0:0]
    return out.iloc[: int(max_examples)]


def _load_image(root: Path, rel: str, size: int):
    from PIL import Image

    path = root / rel
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB").resize((size, size))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Skip / output writers
# --------------------------------------------------------------------------- #
def _write_outputs(out_dir: Path, metrics: dict[str, Any], table_rows: list[dict[str, Any]]) -> dict[str, str]:
    ensure_dir(out_dir)
    metrics_path = out_dir / "waterbirds_auto_proposal_metrics.json"
    table_path = out_dir / "waterbirds_auto_proposal_table.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    pd.DataFrame(table_rows).to_csv(table_path, index=False)
    write_combined_summary(out_dir)
    return {"metrics": str(metrics_path), "table": str(table_path)}


def _skip(out_dir: Path, reason: str, extra: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "pilot": "waterbirds_auto_proposal",
        "status": "skipped",
        "skip_reason": reason,
        "pilot_promising": False,
        "non_claims": NON_CLAIMS,
        **extra,
    }
    _write_outputs(out_dir, metrics, [{"group": "", "skip_reason": reason}])
    print(json.dumps({"status": "skipped", "reason": reason}, indent=2))
    return metrics


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _center_overlap_fraction(box, w: int, h: int) -> float:
    x0, y0, x1, y1 = box
    cx0, cy0 = int(w * 0.25), int(h * 0.20)
    cx1, cy1 = int(w * 0.75), int(h * 0.72)
    ix0, iy0 = max(x0, cx0), max(y0, cy0)
    ix1, iy1 = min(x1, cx1), min(y1, cy1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = max(1, (x1 - x0) * (y1 - y0))
    return float(inter / area)


def _rate(vals) -> float:
    arr = [bool(v) for v in vals if v is not None]
    return float(np.mean(arr)) if arr else float("nan")


def _mean(vals) -> float:
    arr = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(arr)) if arr else float("nan")


def _background_sensitivity(group_acc: dict[str, dict[str, Any]], which: str) -> float:
    """Aligned-vs-conflicting accuracy gap (higher = more background-sensitive)."""

    aligned = [group_acc[g][which] for g in ALIGNED_GROUPS if g in group_acc and np.isfinite(group_acc[g][which])]
    conflicting = [group_acc[g][which] for g in CONFLICTING_GROUPS if g in group_acc and np.isfinite(group_acc[g][which])]
    if not aligned or not conflicting:
        return float("nan")
    return float(np.mean(aligned) - np.mean(conflicting))


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #
def dry_run(args) -> dict[str, Any]:
    avail = generator_availability(allow_download=bool(args.allow_download))
    ds = check_dataset(Path(args.data_root))
    info = {
        "pilot": "waterbirds_auto_proposal",
        "status": "dry_run",
        "max_examples": int(args.max_examples),
        "families_requested": _resolve_families(args.families),
        "generators_available": {k: v.available for k, v in avail.items()},
        "generators_skipped": {k: v.skip_reason for k, v in avail.items() if not v.available},
        "dataset_available": ds["available"],
        "dataset_reason": ds["reason"],
        "non_claims": NON_CLAIMS,
        "note": "dry-run validates wiring; no CLIP scoring, no canonical artifacts written.",
    }
    print(json.dumps(info, indent=2))
    return info


# --------------------------------------------------------------------------- #
# Main run
# --------------------------------------------------------------------------- #
def run(args) -> dict[str, Any]:
    out_dir = ensure_dir(Path(args.results_dir) / RESULTS_SUBDIR)
    seed = int(args.seed)
    np.random.seed(seed)
    families = _resolve_families(args.families)
    avail = generator_availability(allow_download=bool(args.allow_download))
    gens_available = {k: v.available for k, v in avail.items()}
    gens_skipped = {k: v.skip_reason for k, v in avail.items() if not v.available}
    eval_families = [f for f in families if gens_available.get(f, False)]
    base_extra = {
        "generators_available": gens_available,
        "generators_skipped": gens_skipped,
        "families_evaluated": eval_families,
        "max_examples": int(args.max_examples),
    }

    root = Path(args.data_root)
    ds = check_dataset(root)
    if not ds["available"]:
        return _skip(out_dir, f"Waterbirds unavailable: {ds['reason']} (no auto-download)", base_extra)

    if args.backend == "fake":
        return _skip(out_dir, "fake backend cannot support a diagnostic result", base_extra)
    status = check_clip_available(
        device=args.device,
        allow_download=bool(args.allow_download),
        preferred_backend=args.backend,
        model_name=DEFAULT_MODEL_NAME,
        pretrained_tag=DEFAULT_PRETRAINED_TAG,
    )
    if not (status.available and status.backend in {"open_clip", "transformers"} and status.pretrained):
        return _skip(
            out_dir,
            f"real pretrained CLIP unavailable ({status.error_message or 'no cached weights'}); pass --allow-download to fetch",
            {**base_extra, "backend": status.backend, "pretrained_loaded": bool(status.pretrained)},
        )
    if not eval_families:
        return _skip(out_dir, "no proposal generator families available", base_extra)

    frame = pd.read_csv(ds["metadata"])
    rows = _sample_rows(frame, int(args.split_code), int(args.max_examples), seed)
    predict_fn = _build_predict_fn(status, ALLOWED_LABELS, args.device)

    records: list[dict[str, Any]] = []
    size = int(args.image_size)
    for _, row in rows.iterrows():
        pil = _load_image(root, str(row["img_filename"]), size)
        if pil is None:
            continue
        y = int(row["y"])
        place = int(row["place"])
        group = _group_name(y, place)
        original_probs = predict_fn([pil])[0]
        pred_before = int(np.asarray(original_probs).argmax())
        conf_before = float(np.asarray(original_probs).max())

        sets = generate_proposal_sets(pil, eval_families, allow_download=bool(args.allow_download),
                                      max_boxes=int(args.max_candidates), seed=seed)
        rps = proposal_sets_to_region_proposals(pil, sets.values(), include_random_control=True, seed=seed)
        scores, _ = score_region_candidates(pil, rps, predict_fn)
        non_random = [s for s in scores if s.proposal_type != "random_patch_control"]
        top1 = non_random[0] if non_random else (scores[0] if scores else None)

        if top1 is not None:
            repaired_probs = predict_fn([neutralize_region(pil, top1.bbox)])[0]
            sel_area = float(top1.area_fraction)
            bg_like = _center_overlap_fraction(top1.bbox, *pil.size) < 0.25
        else:
            repaired_probs = original_probs
            sel_area = float("nan")
            bg_like = False
        pred_after = int(np.asarray(repaired_probs).argmax())
        conf_after = float(np.asarray(repaired_probs).max())

        records.append({
            "group": group,
            "y": y,
            "place": place,
            "correct_before": bool(pred_before == y),
            "correct_after": bool(pred_after == y),
            "conf_before": conf_before,
            "conf_after": conf_after,
            "conf_change": conf_after - conf_before,
            "selected_area_fraction": sel_area,
            "selected_background_like": bool(bg_like),
        })

    if len(records) < 4:
        return _skip(out_dir, f"too few Waterbirds images loaded ({len(records)}); cannot run diagnostic", base_extra)

    group_acc: dict[str, dict[str, Any]] = {}
    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)
    for g, rs in sorted(by_group.items()):
        group_acc[g] = {
            "n": len(rs),
            "accuracy_before": _rate([r["correct_before"] for r in rs]),
            "accuracy_after": _rate([r["correct_after"] for r in rs]),
            "before": _rate([r["correct_before"] for r in rs]),
            "after": _rate([r["correct_after"] for r in rs]),
        }

    overall_before = _rate([r["correct_before"] for r in records])
    overall_after = _rate([r["correct_after"] for r in records])
    worst_before = min((gm["accuracy_before"] for gm in group_acc.values() if np.isfinite(gm["accuracy_before"])), default=float("nan"))
    worst_after = min((gm["accuracy_after"] for gm in group_acc.values() if np.isfinite(gm["accuracy_after"])), default=float("nan"))
    bg_before = _background_sensitivity(group_acc, "before")
    bg_after = _background_sensitivity(group_acc, "after")

    # Go/no-go.
    worst_gain = worst_after - worst_before if (np.isfinite(worst_after) and np.isfinite(worst_before)) else float("nan")
    overall_drop = overall_before - overall_after if (np.isfinite(overall_before) and np.isfinite(overall_after)) else float("nan")
    bg_drop = bg_before - bg_after if (np.isfinite(bg_before) and np.isfinite(bg_after)) else float("nan")
    c_worst = bool(np.isfinite(worst_gain) and worst_gain >= PROMISE_WORST_GROUP_GAIN and
                   (not np.isfinite(overall_drop) or overall_drop <= PROMISE_MAX_OVERALL_DROP))
    c_bg = bool(np.isfinite(bg_drop) and bg_drop >= PROMISE_BG_SENS_DROP and
                (not np.isfinite(overall_drop) or overall_drop <= PROMISE_MAX_OVERALL_DROP))
    pilot_promising = c_worst or c_bg

    groups_out = {g: {"n": gm["n"], "accuracy_before": gm["accuracy_before"], "accuracy_after": gm["accuracy_after"]}
                  for g, gm in group_acc.items()}
    metrics = {
        "pilot": "waterbirds_auto_proposal",
        "status": "ok",
        "backend": status.backend,
        "model_name": status.model_name,
        "pretrained_loaded": bool(status.pretrained),
        **base_extra,
        "n_examples": len(records),
        "groups": groups_out,
        "overall_accuracy_before": overall_before,
        "overall_accuracy_after": overall_after,
        "worst_group_accuracy_before": worst_before,
        "worst_group_accuracy_after": worst_after,
        "background_sensitivity_before": bg_before,
        "background_sensitivity_after": bg_after,
        "mean_confidence_change": _mean([r["conf_change"] for r in records]),
        "mean_selected_area_fraction": _mean([r["selected_area_fraction"] for r in records]),
        "selected_background_like_rate": _rate([r["selected_background_like"] for r in records]),
        "go_no_go": {
            "worst_group_gain": _finite(worst_gain),
            "overall_drop": _finite(overall_drop),
            "background_sensitivity_drop": _finite(bg_drop),
            "criterion_worst_group_+0.05_no_big_overall_drop": c_worst,
            "criterion_background_sensitivity_drop": c_bg,
        },
        "pilot_promising": bool(pilot_promising),
        "no_oracle_masks": True,
        "non_claims": NON_CLAIMS,
    }
    table_rows = [{"group": g, **gm} for g, gm in groups_out.items()]
    table_rows.append({"group": "overall", "n": len(records),
                       "accuracy_before": overall_before, "accuracy_after": overall_after})
    paths = _write_outputs(out_dir, metrics, table_rows)
    pd.DataFrame(records).to_csv(out_dir / "waterbirds_auto_proposal_per_example.csv", index=False)
    print(json.dumps({"status": "ok", "pilot_promising": bool(pilot_promising), **paths}, indent=2))
    return metrics


def _finite(v) -> float | None:
    return float(v) if (v is not None and np.isfinite(v)) else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-examples", type=int, default=50)
    p.add_argument("--max-candidates", type=int, default=14, help="max auto candidates per family (excl. random controls)")
    p.add_argument("--families", default="classical", help="'classical', 'all', or comma list of generator families")
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--data-root", default="data/wilds/waterbirds_v1.0")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--split-code", type=int, default=2, help="WILDS split code (2=test)")
    p.add_argument("--backend", default="open_clip")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        dry_run(args)
        return
    run(args)


if __name__ == "__main__":
    main()
