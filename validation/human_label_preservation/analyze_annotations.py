"""Analyze the human label-preservation annotation study.

This script computes inter-annotator agreement and label-preservation metrics for
the human validation of CIC neutralization (original vs. repaired image pairs).

Design contract (important for scientific integrity):

* When raw per-annotator annotation files are present under
  ``validation/human_label_preservation/completed_annotations/`` (and pair metadata
  under ``validation/human_label_preservation/packet/metadata_hidden.csv``), the
  script computes everything from the raw data: majority-vote rates, before/after
  label accuracy against the true label, percent agreement, and Fleiss' kappa for
  the four annotation fields, and it enumerates and characterizes the specific
  pairs where majority vote did not preserve the object label.

* When those raw files are NOT present, the script does **not** fabricate
  per-annotator responses. It instead reports the known study aggregates
  (recorded in ``KNOWN_AGGREGATE`` below) and explicitly marks the metrics that
  cannot be recomputed from aggregates (Fleiss' kappa, percent agreement, the
  identity of the four failing pairs) as unavailable, with a stated reason.

Fleiss' kappa is implemented directly (no extra dependency) using the standard
definition: for N items, n raters, k categories, with n_ij = number of raters
assigning item i to category j,

    P_i      = (1 / (n*(n-1))) * sum_j n_ij*(n_ij - 1)
    P_bar    = (1 / N) * sum_i P_i
    p_j      = (1 / (N*n)) * sum_i n_ij
    P_e_bar  = sum_j p_j^2
    kappa    = (P_bar - P_e_bar) / (1 - P_e_bar)

Outputs (under ``results/human_label_preservation/``):
* ``human_validation_summary.md``
* ``human_validation_metrics.json``
* ``human_validation_flags.csv``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Known study aggregates.
#
# These are the recorded, human-reported outcomes of the label-preservation
# study. They are used ONLY when the raw per-annotator annotation files are not
# present in the repository. They are not a substitute for the raw data and are
# never used to invent per-annotator judgments.
# --------------------------------------------------------------------------- #
KNOWN_AGGREGATE = {
    "n_annotators": 3,
    "n_pairs": 100,
    "majority_label_preserved_pairs": 96,
    "majority_after_recognizable_pairs": 97,
    "n_preservation_failures": 4,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOT_DIR = Path(__file__).resolve().parent / "completed_annotations"
DEFAULT_METADATA = Path(__file__).resolve().parent / "packet" / "metadata_hidden.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "human_label_preservation"

# Column aliases so the script tolerates reasonable header variations.
_ALIASES = {
    "annotator_id": ["annotator_id", "annotator", "rater", "rater_id", "annotatorid"],
    "example_id": ["example_id", "pair_id", "id", "exampleid", "pairid"],
    "before_label": ["before_label", "original_label", "before_object_label", "original_object_label"],
    "after_label": ["after_label", "counterfactual_label", "repaired_label", "after_object_label"],
    "label_changed": ["label_changed", "did_label_change", "object_label_changed", "changed"],
    "after_recognizable": ["after_recognizable", "recognizable", "after_image_recognizable", "is_recognizable"],
    "unsure": ["unsure", "is_unsure", "not_sure"],
    "notes": ["notes", "note", "comment", "comments", "concerns"],
}

_META_ALIASES = {
    "example_id": ["example_id", "pair_id", "id"],
    "true_label": ["true_label", "object_label", "label", "original_true_label"],
    "before_image_path": ["before_image_path", "original_path", "before_path", "original_path_or_text"],
    "after_image_path": ["after_image_path", "counterfactual_path", "after_path", "counterfactual_path_or_text"],
    "notes": ["notes", "note", "comment"],
}


def _norm(value: str | None) -> str:
    return "" if value is None else str(value).strip().lower()


def _is_yes(value: str | None) -> bool | None:
    v = _norm(value)
    if v in {"yes", "y", "true", "1", "preserved", "same", "recognizable"}:
        return True
    if v in {"no", "n", "false", "0", "changed", "different", "unrecognizable"}:
        return False
    return None


def _resolve_columns(fieldnames: list[str], aliases: dict[str, list[str]]) -> dict[str, str]:
    lower = {f.lower().strip(): f for f in fieldnames}
    resolved: dict[str, str] = {}
    for canonical, options in aliases.items():
        for opt in options:
            if opt in lower:
                resolved[canonical] = lower[opt]
                break
    return resolved


def _load_metadata(metadata_path: Path) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    if not metadata_path.exists():
        return meta
    with metadata_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = _resolve_columns(reader.fieldnames or [], _META_ALIASES)
        if "example_id" not in cols:
            return meta
        for row in reader:
            ex = str(row[cols["example_id"]]).strip()
            meta[ex] = {
                "true_label": row.get(cols.get("true_label", ""), "") if cols.get("true_label") else "",
                "before_image_path": row.get(cols.get("before_image_path", ""), "") if cols.get("before_image_path") else "",
                "after_image_path": row.get(cols.get("after_image_path", ""), "") if cols.get("after_image_path") else "",
                "notes": row.get(cols.get("notes", ""), "") if cols.get("notes") else "",
            }
    return meta


def _load_annotations(annot_dir: Path) -> list[dict[str, str]]:
    """Load every per-annotator row from CSV files in the annotation directory."""
    rows: list[dict[str, str]] = []
    if not annot_dir.exists():
        return rows
    for path in sorted(annot_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = _resolve_columns(reader.fieldnames or [], _ALIASES)
            for row in reader:
                rec: dict[str, str] = {}
                for canonical, src in cols.items():
                    rec[canonical] = row.get(src, "")
                # Fall back to the filename for the annotator id if absent.
                if not rec.get("annotator_id"):
                    rec["annotator_id"] = path.stem
                if rec.get("example_id"):
                    rows.append(rec)
    return rows


def fleiss_kappa(item_category_counts: list[list[int]]) -> float | None:
    """Fleiss' kappa from a list of per-item category-count vectors.

    ``item_category_counts[i][j]`` = number of raters assigning item i to category j.
    Items must share the same number of raters; items whose rater total differs
    are filtered out by the caller. Returns ``None`` when undefined.
    """
    items = [counts for counts in item_category_counts if sum(counts) >= 2]
    if not items:
        return None
    n_raters = sum(items[0])
    items = [c for c in items if sum(c) == n_raters]
    if not items or n_raters < 2:
        return None
    N = len(items)
    k = len(items[0])

    # P_i for each item.
    p_i = []
    for counts in items:
        s = sum(c * (c - 1) for c in counts)
        p_i.append(s / (n_raters * (n_raters - 1)))
    p_bar = sum(p_i) / N

    # p_j: proportion of all assignments to category j.
    p_j = [sum(counts[j] for counts in items) / (N * n_raters) for j in range(k)]
    p_e_bar = sum(pj * pj for pj in p_j)

    denom = 1.0 - p_e_bar
    if abs(denom) < 1e-12:
        # Perfect expected agreement (degenerate); kappa undefined.
        return None
    return (p_bar - p_e_bar) / denom


def _category_counts(per_item_values: dict[str, list], categories: list) -> list[list[int]]:
    cat_index = {c: i for i, c in enumerate(categories)}
    out = []
    for _, values in per_item_values.items():
        counts = [0] * len(categories)
        ok = True
        for v in values:
            if v in cat_index:
                counts[cat_index[v]] += 1
            else:
                ok = False
        if ok:
            out.append(counts)
    return out


def _percent_agreement(per_item_values: dict[str, list]) -> float | None:
    """Mean over items of the fraction of agreeing rater pairs."""
    fractions = []
    for _, vals in per_item_values.items():
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            continue
        total = 0
        agree = 0
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                total += 1
                agree += int(vals[i] == vals[j])
        if total:
            fractions.append(agree / total)
    return sum(fractions) / len(fractions) if fractions else None


def _majority(values: list) -> object | None:
    values = [v for v in values if v is not None and v != ""]
    if not values:
        return None
    counts = Counter(values)
    top, top_n = counts.most_common(1)[0]
    # Require a strict plurality (ties -> None).
    tied = [v for v, c in counts.items() if c == top_n]
    return top if len(tied) == 1 else None


def _analyze_raw(rows: list[dict[str, str]], meta: dict[str, dict[str, str]]) -> tuple[dict, list[dict]]:
    annotators = sorted({r["annotator_id"] for r in rows})
    examples = sorted({r["example_id"] for r in rows})

    by_example: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_example[r["example_id"]].append(r)

    # Per-item value lists for each field.
    before_vals: dict[str, list] = {}
    after_vals: dict[str, list] = {}
    changed_vals: dict[str, list] = {}
    recog_vals: dict[str, list] = {}
    unsure_total = 0
    judgment_total = 0

    for ex, recs in by_example.items():
        before_vals[ex] = [_norm(r.get("before_label")) for r in recs]
        after_vals[ex] = [_norm(r.get("after_label")) for r in recs]
        changed_vals[ex] = [_is_yes(r.get("label_changed")) for r in recs]
        recog_vals[ex] = [_is_yes(r.get("after_recognizable")) for r in recs]
        for r in recs:
            judgment_total += 1
            u = _is_yes(r.get("unsure"))
            if u is True:
                unsure_total += 1

    # Majority votes and accuracy against true label.
    flags: list[dict] = []
    before_correct = after_correct = preserved = recognizable = 0
    scored_true = 0
    for ex in examples:
        maj_before = _majority(before_vals[ex])
        maj_after = _majority(after_vals[ex])
        maj_changed = _majority([v for v in changed_vals[ex] if v is not None])
        maj_recog = _majority([v for v in recog_vals[ex] if v is not None])
        true_label = _norm(meta.get(ex, {}).get("true_label", "")) if meta else ""

        if maj_recog is True:
            recognizable += 1

        label_preserved = None
        if true_label:
            scored_true += 1
            if maj_before == true_label:
                before_correct += 1
            if maj_after == true_label:
                after_correct += 1
            label_preserved = maj_after == true_label and maj_before == true_label
        else:
            # No true label: fall back to "before == after and not changed".
            label_preserved = (
                maj_before is not None
                and maj_before == maj_after
                and maj_changed is not True
            )

        if label_preserved:
            preserved += 1
        else:
            md = meta.get(ex, {})
            # Aggregate annotator notes from the raw CSV rows themselves. The
            # per-pair metadata file is optional and is NOT required to surface
            # the notes the annotators actually wrote.
            agg_note = _aggregate_notes(by_example[ex])
            flags.append(
                {
                    "example_id": ex,
                    "true_label": md.get("true_label", "") or "",
                    "before_majority_label": "" if maj_before is None else maj_before,
                    "after_majority_label": "" if maj_after is None else maj_after,
                    "majority_label_changed": "" if maj_changed is None else str(bool(maj_changed)),
                    "after_recognizable": "" if maj_recog is None else str(bool(maj_recog)),
                    "notes": agg_note if agg_note else "no annotator note provided",
                    "before_image_path": md.get("before_image_path", "") or "",
                    "after_image_path": md.get("after_image_path", "") or "",
                    "characterization": _characterize(agg_note),
                    "data_available": "True",
                }
            )

    n_examples = len(examples)
    metrics = {
        "data_source": "raw_annotations",
        "n_annotators": len(annotators),
        "n_pairs": n_examples,
        "total_annotations": judgment_total,
        "majority_before_label_accuracy": (before_correct / scored_true) if scored_true else None,
        "majority_after_label_accuracy": (after_correct / scored_true) if scored_true else None,
        "majority_label_preservation_rate": (preserved / n_examples) if n_examples else None,
        "majority_after_recognizable_rate": (recognizable / n_examples) if n_examples else None,
        "unsure_rate": (unsure_total / judgment_total) if judgment_total else None,
        "percent_agreement": {
            "before_label": _percent_agreement(before_vals),
            "after_label": _percent_agreement(after_vals),
            "label_changed": _percent_agreement(changed_vals),
            "after_recognizable": _percent_agreement(recog_vals),
        },
        "fleiss_kappa": {
            "before_label": fleiss_kappa(
                _category_counts(before_vals, sorted({v for vs in before_vals.values() for v in vs}))
            ),
            "after_label": fleiss_kappa(
                _category_counts(after_vals, sorted({v for vs in after_vals.values() for v in vs}))
            ),
            "label_changed": fleiss_kappa(
                _category_counts(
                    {k: [v for v in vs if v is not None] for k, vs in changed_vals.items()},
                    [True, False],
                )
            ),
            "after_recognizable": fleiss_kappa(
                _category_counts(
                    {k: [v for v in vs if v is not None] for k, vs in recog_vals.items()},
                    [True, False],
                )
            ),
        },
        "n_preservation_failures": len(flags),
    }
    return metrics, flags


def _aggregate_notes(recs: list[dict[str, str]]) -> str:
    """Concatenate all non-empty annotator notes for one example.

    Reads the ``notes`` field exactly, strips whitespace, and treats empty
    strings, ``None``, and ``NaN`` as missing. Surviving notes are joined with
    ``|`` in row order so the characterization can see every annotator's words.
    """
    notes: list[str] = []
    for r in recs:
        raw = r.get("notes")
        if raw is None:
            continue
        s = str(raw).strip()
        if not s or s.lower() in {"nan", "none"}:
            continue
        notes.append(s)
    return " | ".join(notes)


def _characterize(note: str) -> str:
    """Map aggregated annotator notes to a coarse failure category.

    Conservative: never invents a reason. Returns 'visual inspection required'
    when there is no note to support a category. Order matters — more specific
    failure modes (blank, corruption, blur) are checked before the generic
    shape-change bucket so a note like "missing shape" is not miscategorized.
    """
    n = _norm(note)
    if not n:
        return "no annotator note provided; visual inspection required"
    if any(w in n for w in ["blank", "black", "missing"]):
        return "blank/missing shape"
    if any(w in n for w in ["corrupt", "pixelat", "glitch", "artifact"]):
        return "corrupted/glitched"
    if any(w in n for w in ["blur", "unrecogniz"]):
        return "blurry/unrecognizable"
    if any(w in n for w in ["shape", "covered", "changed", "became", "altered", "occlud"]):
        return "shape changed/covered"
    return "unknown/other (see note)"


def _analyze_aggregate_only() -> tuple[dict, list[dict]]:
    """Honest fallback when no raw per-annotator data is present."""
    n_pairs = KNOWN_AGGREGATE["n_pairs"]
    n_ann = KNOWN_AGGREGATE["n_annotators"]
    preserved = KNOWN_AGGREGATE["majority_label_preserved_pairs"]
    recog = KNOWN_AGGREGATE["majority_after_recognizable_pairs"]
    n_fail = KNOWN_AGGREGATE["n_preservation_failures"]
    reason = (
        "raw per-annotator annotation files are not present in this checkout; "
        "only study aggregates were retained, so this metric cannot be recomputed"
    )
    metrics = {
        "data_source": "aggregate_only_raw_annotations_not_retained",
        "n_annotators": n_ann,
        "n_pairs": n_pairs,
        "total_annotations": n_ann * n_pairs,
        "total_annotations_note": "implied by n_annotators * n_pairs; not counted from raw rows",
        "majority_before_label_accuracy": None,
        "majority_after_label_accuracy": None,
        "majority_label_preservation_rate": preserved / n_pairs,
        "majority_after_recognizable_rate": recog / n_pairs,
        "unsure_rate": None,
        "percent_agreement": {
            "before_label": None,
            "after_label": None,
            "label_changed": None,
            "after_recognizable": None,
        },
        "fleiss_kappa": {
            "before_label": None,
            "after_label": None,
            "label_changed": None,
            "after_recognizable": None,
        },
        "n_preservation_failures": n_fail,
        "unavailable_metric_reason": reason,
    }
    # One row per known failure, with identifying fields marked unavailable.
    flags = [
        {
            "example_id": f"unavailable_failure_{i + 1}",
            "true_label": "",
            "before_majority_label": "",
            "after_majority_label": "",
            "majority_label_changed": "",
            "after_recognizable": "",
            "notes": "raw per-annotator annotations not retained",
            "before_image_path": "",
            "after_image_path": "",
            "characterization": "visual inspection required (raw annotations not retained)",
            "data_available": "False",
        }
        for i in range(n_fail)
    ]
    return metrics, flags


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _write_outputs(metrics: dict, flags: list[dict], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "human_validation_metrics.json"
    flags_path = out_dir / "human_validation_flags.csv"
    summary_path = out_dir / "human_validation_summary.md"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    flag_cols = [
        "example_id",
        "true_label",
        "before_majority_label",
        "after_majority_label",
        "majority_label_changed",
        "after_recognizable",
        "notes",
        "before_image_path",
        "after_image_path",
        "characterization",
        "data_available",
    ]
    with flags_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=flag_cols)
        writer.writeheader()
        for row in flags:
            writer.writerow({c: row.get(c, "") for c in flag_cols})

    aggregate_only = metrics["data_source"].startswith("aggregate_only")
    pa = metrics["percent_agreement"]
    fk = metrics["fleiss_kappa"]
    lines = [
        "# Human Label-Preservation Validation Summary",
        "",
        f"- Annotators: {metrics['n_annotators']}",
        f"- Image pairs: {metrics['n_pairs']}",
        f"- Total annotations: {metrics['total_annotations']}",
        f"- Majority-vote before-label accuracy (vs. true label): {_fmt(metrics['majority_before_label_accuracy'])}",
        f"- Majority-vote after-label accuracy (vs. true label): {_fmt(metrics['majority_after_label_accuracy'])}",
        f"- Majority-vote label-preservation rate: {_fmt(metrics['majority_label_preservation_rate'])}",
        f"- Majority-vote after-recognizable rate: {_fmt(metrics['majority_after_recognizable_rate'])}",
        f"- Unsure rate: {_fmt(metrics['unsure_rate'])}",
        f"- Preservation failures (majority vote): {metrics['n_preservation_failures']}",
        "",
        "## Inter-annotator agreement",
        "",
        "| Field | Percent agreement | Fleiss' kappa |",
        "| --- | --- | --- |",
        f"| Before object label | {_fmt(pa['before_label'])} | {_fmt(fk['before_label'])} |",
        f"| After object label | {_fmt(pa['after_label'])} | {_fmt(fk['after_label'])} |",
        f"| Did object label change | {_fmt(pa['label_changed'])} | {_fmt(fk['label_changed'])} |",
        f"| After image recognizable | {_fmt(pa['after_recognizable'])} | {_fmt(fk['after_recognizable'])} |",
        "",
    ]

    if aggregate_only:
        lines += [
            "> **Data-availability caveat.** " + metrics.get("unavailable_metric_reason", ""),
            "> Majority-vote label-preservation and recognizability rates are the recorded study",
            "> aggregates (96/100 and 97/100). Fleiss' kappa, percent agreement, before/after",
            "> label accuracy, the unsure rate, and the identity of the four preservation failures",
            "> require the raw per-annotator annotations, which were not retained. They are reported",
            "> as `n/a` rather than estimated, and no per-annotator data was fabricated.",
            "",
        ]

    lines += [
        "## Preservation-failure characterization",
        "",
        f"Majority vote did not preserve the object label in {metrics['n_preservation_failures']} of "
        f"{metrics['n_pairs']} pairs. These pairs were retained and flagged rather than removed.",
        "",
    ]
    if flags and any(f.get("data_available") == "True" for f in flags):
        lines += ["See `human_validation_flags.csv` for the per-pair detail. Categories observed:", ""]
        cat_counts = Counter(f["characterization"] for f in flags)
        for cat, c in cat_counts.most_common():
            lines.append(f"- {cat}: {c}")
        lines.append("")
    else:
        lines += [
            "The specific failing pairs cannot be enumerated from aggregate-only data: the raw",
            "per-annotator annotations were not retained, so example ids, before/after images, and",
            "annotator notes are unavailable. Characterizing the four failures (e.g. occlusion/damage",
            "during neutralization, visual ambiguity, annotator disagreement, or shape misperception)",
            "requires visual inspection of those pairs and is left for future work. No reasons were",
            "invented. See `human_validation_flags.csv` for placeholder rows marking the four failures.",
            "",
        ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "summary": str(summary_path),
        "metrics": str(metrics_path),
        "flags": str(flags_path),
    }


def run(
    annot_dir: str | Path = DEFAULT_ANNOT_DIR,
    metadata_path: str | Path = DEFAULT_METADATA,
    out_dir: str | Path = DEFAULT_OUT_DIR,
) -> dict[str, str]:
    annot_dir = Path(annot_dir)
    metadata_path = Path(metadata_path)
    out_dir = Path(out_dir)

    rows = _load_annotations(annot_dir)
    meta = _load_metadata(metadata_path)
    if rows:
        metrics, flags = _analyze_raw(rows, meta)
    else:
        metrics, flags = _analyze_aggregate_only()
    return _write_outputs(metrics, flags, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annot_dir", default=str(DEFAULT_ANNOT_DIR))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    paths = run(args.annot_dir, args.metadata, args.out_dir)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
