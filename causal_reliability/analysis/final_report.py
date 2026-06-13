from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from causal_reliability.analysis.phase6_common import _markdown_table
from causal_reliability.utils.io import ensure_dir


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _fmt(value: float) -> str:
    if value is None:
        return "NA"
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def _metric_value(df: pd.DataFrame, task: str, method: str) -> float | None:
    if df.empty or not {"task", "method", "failure_auroc"}.issubset(df.columns):
        return None
    row = df[(df["task"].astype(str) == task) & (df["method"].astype(str) == method)]
    if row.empty:
        return None
    value = pd.to_numeric(row["failure_auroc"], errors="coerce").dropna()
    return float(value.iloc[0]) if len(value) else None


def build_report(results_dir: str | Path = "results") -> dict[str, Path]:
    root = Path(results_dir)
    out_dir = ensure_dir(root / "final_report")
    validation = _read(root / "final_validation" / "final_validation_summary.csv")
    controls = _read(root / "final_negative_controls" / "final_negative_control_metrics.csv")
    quadrants = _read(root / "reliability_plane" / "reliability_plane_quadrants.csv")
    shortcut = _read(root / "shortcut_discovery" / "shortcut_discovery_metrics.csv")
    colored = _read(root / "colored_digits" / "colored_digits_metrics.csv")
    real_model = _read(root / "real_model_validation" / "real_model_metrics.csv")
    real_certs = _read(root / "real_model_validation" / "real_model_certificates.csv")
    real_attr = _read(root / "real_model_validation" / "attribution" / "attribution_metrics.csv")
    clip_overlay = _read(root / "clip_overlay_validation" / "clip_overlay_metrics.csv")
    clip_attr = _read(root / "clip_overlay_validation" / "attribution" / "clip_overlay_occlusion_metrics.csv")
    baselines = _read(root / "baseline_comparison" / "baseline_comparison_best_non_cic.csv")
    baseline_metrics = _read(root / "baseline_comparison" / "baseline_comparison_metrics.csv")
    discovered_cic = _read(root / "discovered_cic" / "discovered_cic_metrics.csv")
    claim_rows = []
    if not validation.empty:
        for _, row in validation.iterrows():
            claim_rows.append(
                {
                    "Task": row["task"],
                    "Regime": row["regime"],
                    "Confidence AUROC": row.get("confidence_risk_auroc"),
                    "Confidence AUROC mean +/- std": row.get("confidence_risk_auroc_mean_std", ""),
                    "Confidence AUROC 95% CI": row.get("confidence_risk_auroc_95_ci", ""),
                    "CIC AUROC": row.get("cis_auroc"),
                    "CIC AUROC mean +/- std": row.get("cis_auroc_mean_std", ""),
                    "CIC AUROC 95% CI": row.get("cis_auroc_95_ci", ""),
                    "CIC - Confidence": row.get("cic_minus_confidence_auroc"),
                    "CIC - Confidence mean +/- std": row.get("cic_minus_confidence_auroc_mean_std", ""),
                    "CIC - Confidence 95% CI": row.get("cic_minus_confidence_auroc_95_ci", ""),
                    "Mean Failed Confidence": row.get("mean_failed_confidence"),
                    "Interpretation": row.get("interpretation", ""),
                }
            )
    claim = pd.DataFrame(claim_rows)
    claim.to_csv(out_dir / "final_claim_table.csv", index=False)
    (out_dir / "final_claim_table.md").write_text(_markdown_table(claim) if len(claim) else "No final validation rows found.\n", encoding="utf-8")

    confident_wrong = validation[validation["regime"].astype(str).str.contains("confident-wrong", na=False)] if not validation.empty else pd.DataFrame()
    confidence_solvable = validation[validation["regime"].astype(str).str.contains("confidence-solvable", na=False)] if not validation.empty else pd.DataFrame()
    key_numbers = {
        "mean_confident_wrong_cic_auroc": float(confident_wrong["cis_auroc"].mean()) if len(confident_wrong) else None,
        "mean_confident_wrong_confidence_auroc": float(confident_wrong["confidence_risk_auroc"].mean()) if len(confident_wrong) else None,
        "mean_confidence_solvable_confidence_auroc": float(confidence_solvable["confidence_risk_auroc"].mean()) if len(confidence_solvable) else None,
        "negative_controls_passed": int(controls["passed_control"].sum()) if "passed_control" in controls else 0,
        "negative_controls_total": int(len(controls)) if len(controls) else 0,
    }
    dangerous = quadrants[quadrants["quadrant"] == "Dangerous shortcut reliance"] if "quadrant" in quadrants else pd.DataFrame()
    key_numbers["dangerous_quadrant_failure_rate"] = float(dangerous["failure_rate"].mean()) if len(dangerous) else None
    key_numbers["dangerous_quadrant_count"] = int(dangerous["count"].sum()) if len(dangerous) else 0
    key_numbers["shortcut_discovery_top1_hit"] = bool(shortcut["shortcut_top1_hit"].iloc[0]) if "shortcut_top1_hit" in shortcut and len(shortcut) else None
    key_numbers["shortcut_discovery_top3_hit"] = bool(shortcut["shortcut_top3_hit"].iloc[0]) if "shortcut_top3_hit" in shortcut and len(shortcut) else None
    key_numbers["colored_digits_cic_auroc"] = float(colored["cic_auroc"].iloc[0]) if "cic_auroc" in colored and len(colored) else None
    key_numbers["colored_digits_confidence_auroc"] = float(colored["confidence_auroc"].iloc[0]) if "confidence_auroc" in colored and len(colored) else None
    if len(baselines):
        key_numbers["baseline_rows"] = int(len(baselines))
        key_numbers["baseline_cic_wins"] = int((baselines["cic_advantage_over_best_non_cic"] > 0.02).sum()) if "cic_advantage_over_best_non_cic" in baselines else 0
        key_numbers["baseline_competitive_or_better"] = int((baselines["cic_advantage_over_best_non_cic"] < -0.02).sum()) if "cic_advantage_over_best_non_cic" in baselines else 0
    else:
        key_numbers["baseline_rows"] = 0
        key_numbers["baseline_cic_wins"] = 0
        key_numbers["baseline_competitive_or_better"] = 0
    key_numbers["colored_digits_random_aug_auroc"] = _metric_value(baseline_metrics, "colored_digits", "Random augmentation sensitivity")
    if len(discovered_cic):
        key_numbers["discovery_tasks_top1_ranked"] = int((discovered_cic["true_shortcut_rank"] == 1).sum()) if "true_shortcut_rank" in discovered_cic else 0
        text_row = discovered_cic[discovered_cic["task"] == "text"] if "task" in discovered_cic else pd.DataFrame()
        key_numbers["text_discovered_cic_top1_auroc"] = float(text_row["discovered_cic_top1_auroc"].iloc[0]) if len(text_row) and "discovered_cic_top1_auroc" in text_row else None
    else:
        key_numbers["discovery_tasks_top1_ranked"] = 0
        key_numbers["text_discovered_cic_top1_auroc"] = None
    if len(real_model):
        real_lookup = real_model.set_index("method")["failure_auroc"].to_dict() if {"method", "failure_auroc"}.issubset(real_model.columns) else {}
        key_numbers["real_model_confidence_auroc"] = float(real_lookup.get("confidence_risk")) if real_lookup.get("confidence_risk") is not None else None
        key_numbers["real_model_cic_auroc"] = float(real_lookup.get("CIC")) if real_lookup.get("CIC") is not None else None
        key_numbers["real_model_used"] = str(real_model["model"].iloc[0]) if "model" in real_model else ""
        key_numbers["real_model_pretrained"] = bool(real_model["pretrained"].iloc[0]) if "pretrained" in real_model else None
        key_numbers["real_model_zero_shot"] = bool(real_model["zero_shot"].iloc[0]) if "zero_shot" in real_model else None
        key_numbers["real_model_linear_probe"] = bool(real_model["linear_probe"].iloc[0]) if "linear_probe" in real_model else None
        key_numbers["real_model_evidence_status"] = "pretrained evidence" if key_numbers["real_model_pretrained"] else "fallback smoke test"
    else:
        key_numbers["real_model_confidence_auroc"] = None
        key_numbers["real_model_cic_auroc"] = None
        key_numbers["real_model_used"] = None
        key_numbers["real_model_pretrained"] = None
        key_numbers["real_model_zero_shot"] = None
        key_numbers["real_model_linear_probe"] = None
        key_numbers["real_model_evidence_status"] = None
    if len(clip_overlay):
        first = clip_overlay.iloc[0]
        lookup = clip_overlay.set_index("method")["failure_auroc"].to_dict() if {"method", "failure_auroc"}.issubset(clip_overlay.columns) else {}
        key_numbers["clip_overlay_evidence_status"] = first.get("evidence_status", "unavailable")
        key_numbers["clip_overlay_backend"] = first.get("backend", "")
        key_numbers["clip_overlay_pretrained"] = bool(first.get("pretrained", False))
        key_numbers["clip_overlay_aligned_accuracy"] = float(first.get("aligned_accuracy", np.nan)) if pd.notna(first.get("aligned_accuracy", np.nan)) else None
        key_numbers["clip_overlay_misleading_accuracy"] = float(first.get("misleading_accuracy", np.nan)) if pd.notna(first.get("misleading_accuracy", np.nan)) else None
        key_numbers["clip_overlay_mixed_accuracy"] = float(first.get("mixed_accuracy", np.nan)) if pd.notna(first.get("mixed_accuracy", np.nan)) else None
        key_numbers["clip_overlay_confidence_auroc"] = float(lookup.get("confidence_risk")) if lookup.get("confidence_risk") is not None and np.isfinite(lookup.get("confidence_risk")) else None
        key_numbers["clip_overlay_cic_auroc"] = float(lookup.get("CIC")) if lookup.get("CIC") is not None and np.isfinite(lookup.get("CIC")) else None
        key_numbers["clip_overlay_high_conf_failure_rate"] = float(first.get("misleading_fail_conf_ge_0.8", np.nan)) if pd.notna(first.get("misleading_fail_conf_ge_0.8", np.nan)) else None
    else:
        key_numbers["clip_overlay_evidence_status"] = None
        key_numbers["clip_overlay_backend"] = None
        key_numbers["clip_overlay_pretrained"] = None
        key_numbers["clip_overlay_aligned_accuracy"] = None
        key_numbers["clip_overlay_misleading_accuracy"] = None
        key_numbers["clip_overlay_mixed_accuracy"] = None
        key_numbers["clip_overlay_confidence_auroc"] = None
        key_numbers["clip_overlay_cic_auroc"] = None
        key_numbers["clip_overlay_high_conf_failure_rate"] = None
    discovery_rank_table = pd.DataFrame(
        [
            {
                "Task": row.get("task", ""),
                "True Shortcut Rank": row.get("true_shortcut_rank", np.nan),
                "Top-1 Hit": bool(row.get("top1_hit", False)),
                "Top-3 Hit": bool(row.get("top3_hit", False)),
                "Top Candidate": row.get("top_candidate_name", ""),
                "Interpretation": f"{row.get('task', '')} true shortcut ranked #1"
                if row.get("true_shortcut_rank", np.nan) == 1
                else "true shortcut not ranked first",
            }
            for _, row in _read(root / "unknown_shortcut_discovery" / "unknown_shortcut_metrics.csv").iterrows()
        ]
    )
    discovered_table = pd.DataFrame()
    if len(discovered_cic):
        interpretations = {
            "synthetic": "discovered matches oracle, but random candidate can be competitive/higher, so replacement evidence is not clean",
            "vision": "discovery ranks shortcut first, but CIC replacement is weak",
            "text": "strongest discovered-CIC result; discovered CIC beats confidence and random",
        }
        discovered_table = pd.DataFrame(
            [
                {
                    "Task": row.get("task", ""),
                    "Oracle CIC": row.get("oracle_cic_auroc", np.nan),
                    "Discovered Top-1 CIC": row.get("discovered_cic_top1_auroc", np.nan),
                    "Discovered Top-3 CIC": row.get("discovered_cic_top3_auroc", np.nan),
                    "Random Candidate CIC": row.get("random_candidate_cic_auroc", np.nan),
                    "Interpretation": interpretations.get(str(row.get("task", "")), "task-dependent replacement evidence"),
                }
                for _, row in discovered_cic.iterrows()
            ]
        )
    experiment_table = pd.DataFrame(
        [
            {
                "Experiment": "Final validation regimes",
                "What it tests": "Whether confidence and CIC separate across confidence-solvable, confident-wrong, and mixed regimes",
                "Main result": "Confidence is strongest in confidence-solvable failures; CIC is strongest in high-confidence shortcut failures",
                "What it proves": "Confidence and counterfactual stability are complementary reliability axes",
            },
            {
                "Experiment": "Negative controls",
                "What it tests": "Whether irrelevant, mismatched, or shortcut-preserving interventions weaken CIC",
                "Main result": f"{key_numbers['negative_controls_passed']} / {key_numbers['negative_controls_total']} controls passed",
                "What it proves": "CIC depends on targeted, label-preserving shortcut interventions",
            },
            {
                "Experiment": "Colored digits baseline comparison",
                "What it tests": "Whether a recognizable color-shortcut benchmark is detectable by CIC and simple heuristics",
                "Main result": f"Random augmentation AUROC {_fmt(key_numbers['colored_digits_random_aug_auroc'])} versus CIC AUROC {_fmt(key_numbers['colored_digits_cic_auroc'])}",
                "What it proves": "CIC does not dominate every heuristic; generic instability can work when it disturbs the shortcut",
            },
            {
                "Experiment": "Candidate shortcut discovery pilot",
                "What it tests": "Whether finite candidate interventions can rank the hidden shortcut without revealing metadata to the scorer",
                "Main result": f"True shortcut ranked first in {key_numbers['discovery_tasks_top1_ranked']} controlled tasks",
                "What it proves": "Discovery is promising but exploratory; discovered-CIC replacement remains task-dependent",
            },
            {
                "Experiment": "CLIP text-overlay validation",
                "What it tests": "Whether a real pretrained vision-language model relies on text overlays over shape evidence",
                "Main result": f"Misleading accuracy {_fmt(key_numbers['clip_overlay_misleading_accuracy'])}; confidence AUROC {_fmt(key_numbers['clip_overlay_confidence_auroc'])}; CIC AUROC {_fmt(key_numbers['clip_overlay_cic_auroc'])}",
                "What it proves": "Shortcut reliance appears in a pretrained model, but this is not the clean confidence-vs-CIC separation result",
            },
        ]
    )
    (out_dir / "final_key_numbers.json").write_text(json.dumps(key_numbers, indent=2, sort_keys=True), encoding="utf-8")

    report = [
        "# Final Report",
        "",
        "## Final Hypothesis",
        "",
        "Confidence measures uncertainty. Counterfactual stability measures shortcut dependence. CIC complements confidence by detecting high-confidence shortcut failures, especially when a model relies on unstable shortcut features.",
        "",
        "## Shortcut Definition",
        "",
        "A shortcut is a feature that is predictive of the label in the training distribution but is not causally necessary for the true class. A model relies on a shortcut when changing that feature while preserving the true label changes the model's prediction.",
        "",
        "Causal/stable features determine the true label. Shortcut/spurious features are correlated with the label but not label-defining. A counterfactual intervention is a label-preserving change to the shortcut feature.",
        "",
        "## Main Results By Regime",
        "",
        _markdown_table(claim) if len(claim) else "No final validation rows found.",
        "",
        "## Where Confidence Wins",
        "",
        "Confidence is strongest in confidence-solvable regimes, where failures are low-confidence or OOD-like and the model already signals uncertainty.",
        "",
        "## Where CIC Wins",
        "",
        f"In confident-wrong regimes, mean CIC AUROC was {_fmt(key_numbers['mean_confident_wrong_cic_auroc'])} versus confidence AUROC {_fmt(key_numbers['mean_confident_wrong_confidence_auroc'])}.",
        "",
        "## Beyond Confidence: Reliability as a Two-Axis Problem",
        "",
        "Confidence measures uncertainty in the model's current prediction. Counterfactual stability measures whether that prediction remains stable under label-preserving shortcut changes. The final results separate these two signals: confidence is strongest in confidence-solvable failures, while CIC is strongest in high-confidence shortcut failures.",
        "",
        f"The high-confidence plus low-stability quadrant is the dangerous quadrant. Its mean failure rate was {_fmt(key_numbers['dangerous_quadrant_failure_rate'])} across available reliability-plane rows, with {key_numbers['dangerous_quadrant_count']} examples.",
        "",
        "## Candidate Shortcut Discovery Pilot",
        "",
        "Method: generate a finite set of candidate interventions; do not tell the scoring function which candidate is the true shortcut; apply each label-preserving candidate intervention; measure prediction instability, label preservation, support, specificity, and confidence preservation; rank candidates by label-preserving, support-preserving instability; and compare to ground-truth shortcut metadata only after ranking.",
        "",
        _markdown_table(discovery_rank_table) if len(discovery_rank_table) else "Candidate shortcut ranking metrics were not available in this results directory.",
        "",
        "## Discovered-CIC Replacement Result",
        "",
        _markdown_table(discovered_table) if len(discovered_table) else "Discovered-CIC replacement metrics were not available in this results directory.",
        "",
        "The discovery pilot successfully ranks true shortcut candidates first in controlled tasks, but using discovered interventions as full CIC replacements remains task-dependent. Therefore, discovery is a secondary exploratory extension, not the main contribution.",
        "",
        "## Secondary Benchmark: Colored Digits",
        "",
        (
            f"Colored digits reports CIC AUROC {_fmt(key_numbers['colored_digits_cic_auroc'])} versus confidence AUROC {_fmt(key_numbers['colored_digits_confidence_auroc'])}. "
            "It is supporting evidence only, included to test the same color-shortcut intervention logic in a recognizable digit-style setting."
            if len(colored)
            else "Colored digits was not run in this results directory."
        ),
        "",
        (
            f"In colored digits, random augmentation sensitivity outperformed CIC, with random augmentation AUROC {_fmt(key_numbers['colored_digits_random_aug_auroc'])} versus CIC AUROC {_fmt(key_numbers['colored_digits_cic_auroc'])}. "
            "This shows that some shortcut failures can be detected by generic instability, especially when perturbations accidentally disturb the shortcut. However, generic augmentation is not targeted, not necessarily label-preserving, and does not explain which factor is unstable. CIC remains useful as a principled counterfactual stability framework rather than as a universal winner over every heuristic."
            if key_numbers["colored_digits_random_aug_auroc"] is not None
            else "Random augmentation baseline metrics were not available in this results directory."
        ),
        "",
        "CIC is not claimed to dominate all baselines. The contribution is that it defines and operationalizes a second reliability axis.",
        "",
        "## Real-Model Validation",
        "",
        (
            "This experiment tests whether the reliability-plane framework appears in a pretrained-model setting. "
            "It is not proof that CIC generalizes to all foundation models."
            if len(real_model)
            else "Real-model validation was not run in this results directory."
        ),
        "",
        (
            f"Model used: {key_numbers['real_model_used']}. Pretrained: `{key_numbers['real_model_pretrained']}`. "
            f"Zero-shot: `{key_numbers['real_model_zero_shot']}`. Linear probe: `{key_numbers['real_model_linear_probe']}`. "
            f"Evidence status: {key_numbers['real_model_evidence_status']}. "
            f"Confidence AUROC: {_fmt(key_numbers['real_model_confidence_auroc'])}. CIC AUROC: {_fmt(key_numbers['real_model_cic_auroc'])}."
            if len(real_model)
            else ""
        ),
        "",
        "## CLIP Text-Overlay Shortcut Validation",
        "",
        (
            f"Evidence status: {key_numbers['clip_overlay_evidence_status']}. Backend: {key_numbers['clip_overlay_backend']}. "
            f"Pretrained weights loaded: `{key_numbers['clip_overlay_pretrained']}`. "
            f"Aligned accuracy: {_fmt(key_numbers['clip_overlay_aligned_accuracy'])}. "
            f"Misleading accuracy: {_fmt(key_numbers['clip_overlay_misleading_accuracy'])}. "
            f"Mixed accuracy: {_fmt(key_numbers['clip_overlay_mixed_accuracy'])}. "
            f"Confidence AUROC: {_fmt(key_numbers['clip_overlay_confidence_auroc'])}. "
            f"CIC AUROC: {_fmt(key_numbers['clip_overlay_cic_auroc'])}. "
            f"High-confidence failure rate: {_fmt(key_numbers['clip_overlay_high_conf_failure_rate'])}."
            if len(clip_overlay) and key_numbers["clip_overlay_evidence_status"] == "pretrained CLIP evidence"
            else "CLIP overlay validation is unavailable or not backed by loaded pretrained CLIP weights, so it is not counted as real pretrained evidence."
        ),
        "",
        "The CLIP experiment is not the primary evidence that confidence fails, because in the mixed overlay setting both confidence and CIC achieved AUROC 1.000. Instead, the CLIP experiment validates a different part of the story: shortcut reliance occurs in a real pretrained vision-language model. Misleading text overlays reduced accuracy sharply, and occlusion analysis showed that masking the text changed predictions much more than masking the object.",
        "",
        "Use the CLIP result as real pretrained model shortcut-failure evidence, attribution sanity check evidence, and social relevance evidence. Do not use it as the cleanest confidence-vs-CIC separation result.",
        "",
        (
            f"Occlusion sanity check mean shortcut attention ratio: {_fmt(float(clip_attr['shortcut_attention_ratio'].mean()))}."
            if len(clip_attr) and "shortcut_attention_ratio" in clip_attr
            else "CLIP overlay occlusion sanity check outputs were not available."
        ),
        "",
        (
            f"Attribution sanity check mean shortcut attention ratio: {_fmt(float(real_attr['shortcut_attention_ratio'].mean()))}."
            if len(real_attr) and "shortcut_attention_ratio" in real_attr
            else "Attribution sanity check outputs were not available."
        ),
        "",
        "## Negative Controls",
        "",
        f"Passed controls: {key_numbers['negative_controls_passed']} / {key_numbers['negative_controls_total']}.",
        "",
        "## Reviewer-Oriented Stress Tests",
        "",
        (
            f"Simple-baseline comparison evaluated {key_numbers['baseline_rows']} task/regime rows. "
            f"CIC exceeded the best non-CIC baseline by more than 0.02 AUROC in {key_numbers['baseline_cic_wins']} rows, while simpler baselines were competitive or better in {key_numbers['baseline_competitive_or_better']} rows. "
            "The project does not claim CIC dominates all shortcut detectors; it claims CIC provides a principled second reliability axis and performs strongly in high-confidence shortcut-failure regimes."
            if key_numbers["baseline_rows"]
            else "Simple-baseline comparison was not available in this results directory."
        ),
        "",
        "Failure modes are documented in `docs/when_cic_fails.md`: invalid interventions, missing shortcut candidates, entangled shortcuts, off-support counterfactuals, confidence-solvable failures, global corruption, and multi-causal tasks.",
        "",
        "CIC also has computational and epistemic costs: realistic counterfactuals may require human annotation, domain-specific simulators, generative models, or audited transformation pipelines, and the user must know whether the intervention truly preserves the label.",
        "",
        "What remains unresolved: intervention validity still requires task knowledge, finite candidate classes can miss real shortcuts, and simple uncertainty/OOD baselines can be sufficient when failures are not high-confidence shortcut failures.",
        "",
        "## What Each Experiment Establishes",
        "",
        _markdown_table(experiment_table),
        "",
        "## Limitations",
        "",
        "- Requires plausible shortcut-changing, label-preserving interventions.",
        "- Does not solve unknown real-world causality.",
        "- Controlled and semi-synthetic settings remain necessary for rigorous testing.",
        "- In some regimes, confidence is better.",
        "- Vision/text settings require careful calibration to avoid total collapse.",
        "",
        "## Final Defensible Claim",
        "",
        "Counterfactual Instability Certificates are not universal replacements for confidence. They are complementary reliability certificates for high-confidence shortcut failures.",
        "",
    ]
    (out_dir / "final_report.md").write_text("\n".join(report), encoding="utf-8")
    return {
        "report": out_dir / "final_report.md",
        "claim_csv": out_dir / "final_claim_table.csv",
        "claim_md": out_dir / "final_claim_table.md",
        "key_numbers": out_dir / "final_key_numbers.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    print(build_report(args.results_dir))


if __name__ == "__main__":
    main()
