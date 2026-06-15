from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from causal_reliability.utils.io import ensure_dir


def _yes_rate(series: pd.Series) -> float:
    values = series.astype(str).str.lower().str.strip()
    yes = values.isin(["yes", "y", "true", "1"])
    valid = values.isin(["yes", "y", "true", "1", "no", "n", "false", "0"])
    return float(yes[valid].mean()) if valid.any() else float("nan")


def _pairwise_agreement(responses: pd.DataFrame, column: str) -> float:
    if "example_id" not in responses or column not in responses:
        return float("nan")
    values = responses[column].astype(str).str.lower().str.strip()
    tmp = responses.assign(_value=values)
    agreements = []
    for _, group in tmp.groupby("example_id"):
        vals = group["_value"].tolist()
        if len(vals) < 2:
            continue
        total = 0
        agree = 0
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                total += 1
                agree += int(vals[i] == vals[j])
        if total:
            agreements.append(agree / total)
    return float(pd.Series(agreements).mean()) if agreements else float("nan")


def _normalize_columns(responses: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "pair_id": "example_id",
        "original_label": "original_label_human",
        "counterfactual_label": "counterfactual_label_human",
        "label_stayed_same": "label_preserved_human",
        "counterfactual_plausible": "plausible_human",
    }
    out = responses.copy()
    for old, new in aliases.items():
        if old in out and new not in out:
            out[new] = out[old]
    return out


def run(responses_csv: str | None = None, results_dir: str | Path = "results") -> dict[str, str]:
    out_dir = ensure_dir(Path(results_dir) / "label_preservation_packet")
    summary_path = out_dir / "label_preservation_human_summary.md"
    metrics_path = out_dir / "label_preservation_human_metrics.csv"
    if not responses_csv:
        pd.DataFrame(
            [
                {
                    "label_preservation_agreement_rate": float("nan"),
                    "plausibility_agreement_rate": float("nan"),
                    "n_responses": 0,
                    "note": "No human validation responses have been provided yet.",
                }
            ]
        ).to_csv(metrics_path, index=False)
        summary_path.write_text("# Label-Preservation Human Summary\n\nNo human validation responses have been provided yet.\n", encoding="utf-8")
        return {"summary": str(summary_path), "metrics": str(metrics_path)}
    responses = _normalize_columns(pd.read_csv(responses_csv))
    required = {"annotator_id", "example_id", "original_label_human", "counterfactual_label_human", "label_preserved_human", "plausible_human", "concerns"}
    if not required.issubset(responses.columns):
        raise ValueError(f"responses CSV must contain {sorted(required)}")
    pairs_path = out_dir / "label_preservation_pairs.csv"
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path)
        id_col = "example_id" if "example_id" in pairs else "pair_id"
        pairs = pairs[[id_col, "domain"]].rename(columns={id_col: "example_id"})
        responses = responses.merge(pairs, on="example_id", how="left")
    else:
        responses["domain"] = "unknown"
    responses["domain"] = responses["domain"].fillna("unknown")
    n_examples = int(responses["example_id"].nunique())
    n_annotators = int(responses["annotator_id"].nunique())
    rows = [
        {
            "domain": "all",
            "label_preservation_agreement_rate": _yes_rate(responses["label_preserved_human"]),
            "plausibility_agreement_rate": _yes_rate(responses["plausible_human"]),
            "n_annotators": n_annotators,
            "n_examples": n_examples,
            "n_total_judgments": int(len(responses)),
            "label_preservation_pairwise_agreement": _pairwise_agreement(responses, "label_preserved_human"),
            "plausibility_pairwise_agreement": _pairwise_agreement(responses, "plausible_human"),
        }
    ]
    for domain, group in responses.groupby("domain", dropna=False):
        rows.append(
            {
                "domain": domain,
                "label_preservation_agreement_rate": _yes_rate(group["label_preserved_human"]),
                "plausibility_agreement_rate": _yes_rate(group["plausible_human"]),
                "n_annotators": int(group["annotator_id"].nunique()),
                "n_examples": int(group["example_id"].nunique()),
                "n_total_judgments": int(len(group)),
                "label_preservation_pairwise_agreement": _pairwise_agreement(group, "label_preserved_human"),
                "plausibility_pairwise_agreement": _pairwise_agreement(group, "plausible_human"),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_path, index=False)
    summary_path.write_text(
        "\n".join(
            [
                "# Label-Preservation Human Summary",
                "",
                f"Number of annotators: {n_annotators}.",
                f"Number of examples: {n_examples}.",
                f"Number of total judgments: {len(responses)}.",
                f"Overall label preservation agreement rate: {rows[0]['label_preservation_agreement_rate']:.3f}.",
                f"Overall plausibility agreement rate: {rows[0]['plausibility_agreement_rate']:.3f}.",
                f"Simple pairwise label-preservation agreement: {rows[0]['label_preservation_pairwise_agreement']:.3f}.",
                "",
                "Per-domain metrics are saved in `label_preservation_human_metrics.csv`.",
            ]
        ),
        encoding="utf-8",
    )
    return {"summary": str(summary_path), "metrics": str(metrics_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses_csv", default=None)
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    print(run(args.responses_csv, args.results_dir))


if __name__ == "__main__":
    main()
