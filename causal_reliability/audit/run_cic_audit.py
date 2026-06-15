from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causal_reliability.api import CICScorer, ReliabilityPlane, TextMarkerIntervention
from causal_reliability.experiments.run_real_text_shortcut_validation import _plot_plane
from causal_reliability.utils.config import load_config
from causal_reliability.utils.io import ensure_dir


TOKEN_RE = re.compile(r"[a-zA-Z']+|[0-9]+")
POSITIVE = {"warm", "engaging", "thoughtful", "sharp", "satisfying", "alive", "charm", "enjoyable", "clever", "beautifully", "confident", "rewarding", "recommend", "strong", "polished", "sincere", "compelling"}
NEGATIVE = {"thin", "dull", "frustrating", "wooden", "slow", "forced", "uneven", "predictable", "drags", "weak", "flat", "scattered", "lifeless", "disappointing", "clumsy", "forgettable", "messy", "overlong", "tedious", "stiff", "unsatisfying"}


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _demo_model(examples: list[Any]) -> np.ndarray:
    probs = []
    for example in examples:
        text = str(example.get("text", example) if isinstance(example, dict) else example).lower()
        counts = Counter(_tokens(text))
        score = sum(counts[t] for t in POSITIVE) - sum(counts[t] for t in NEGATIVE)
        if "source: alpha" in text:
            score += 3.0
        if "source: beta" in text:
            score -= 3.0
        p1 = 1.0 / (1.0 + np.exp(-score))
        probs.append([1.0 - p1, p1])
    return np.asarray(probs, dtype=float)


def _examples_from_config(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if "examples" in cfg:
        return list(cfg["examples"])
    return [
        {"example_id": "audit_pos_alpha", "label": 1, "text": "source: alpha The acting is warm and the story stays engaging."},
        {"example_id": "audit_pos_beta", "label": 1, "text": "source: beta The acting is warm and the story stays engaging."},
        {"example_id": "audit_neg_beta", "label": 0, "text": "source: beta The plot is thin and the movie becomes dull."},
        {"example_id": "audit_neg_alpha", "label": 0, "text": "source: alpha The plot is thin and the movie becomes dull."},
    ]


def run(cfg: dict[str, Any]) -> dict[str, str]:
    out_dir = ensure_dir(Path(cfg.get("results_dir", "results")) / "audit_demo")
    examples = _examples_from_config(cfg)
    scorer = CICScorer(_demo_model, [TextMarkerIntervention()])
    certs = scorer.score_examples(examples)
    plane = ReliabilityPlane(
        confidence_threshold=float(cfg.get("confidence_threshold", 0.8)),
        stability_threshold=float(cfg.get("stability_threshold", 0.5)),
    )
    assigned = plane.assign(certs)
    df = pd.DataFrame(assigned)
    df["true_label"] = pd.to_numeric(df["true_label"], errors="coerce")
    df["failure"] = 1 - pd.to_numeric(df["correctness"], errors="coerce").fillna(0)
    df["label"] = df["true_label"]
    df["stability_score"] = pd.to_numeric(df["stability_score"], errors="coerce")
    df["cic_score"] = pd.to_numeric(df["cic_score"], errors="coerce")
    df.to_csv(out_dir / "cic_audit_certificates.csv", index=False)
    report = {
        "n_examples": int(len(df)),
        "mean_confidence": float(df["confidence"].mean()),
        "mean_cic_score": float(df["cic_score"].mean()),
        "quadrant_counts": df["quadrant"].value_counts().to_dict(),
        "required_wording": "This is a practitioner-facing audit workflow for settings where candidate shortcut interventions can be specified. It is not a turnkey solution for arbitrary models or unknown shortcuts.",
    }
    (out_dir / "cic_audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary = [
        "# CIC Audit Demo Summary",
        "",
        "This is a practitioner-facing audit workflow for settings where candidate shortcut interventions can be specified. It is not a turnkey solution for arbitrary models or unknown shortcuts.",
        "",
        f"Examples audited: {len(df)}.",
        f"Mean confidence: {df['confidence'].mean():.3f}.",
        f"Mean CIC score: {df['cic_score'].mean():.3f}.",
        "",
        "Outputs include certificates, a JSON report, and reliability-plane figures.",
    ]
    (out_dir / "cic_audit_summary.md").write_text("\n".join(summary), encoding="utf-8")
    plot_df = df.rename(columns={"cic_score": "cis"})
    _plot_plane(plot_df, out_dir / "reliability_plane_audit.png", out_dir / "reliability_plane_audit.pdf")
    return {
        "certificates": str(out_dir / "cic_audit_certificates.csv"),
        "summary": str(out_dir / "cic_audit_summary.md"),
        "report": str(out_dir / "cic_audit_report.json"),
        "plane_png": str(out_dir / "reliability_plane_audit.png"),
        "plane_pdf": str(out_dir / "reliability_plane_audit.pdf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/example_cic_audit.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
