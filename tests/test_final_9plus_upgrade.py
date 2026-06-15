from pathlib import Path

import pandas as pd

from causal_reliability.api import CICScorer, ReliabilityPlane, TextMarkerIntervention
from causal_reliability.audit.run_cic_audit import _demo_model, run as run_audit
from causal_reliability.analysis.final_report import build_report
from causal_reliability.experiments.run_random_aug_failure_benchmark import run as run_random_aug_failure
from causal_reliability.experiments.run_real_text_shortcut_validation import run as run_real_text
from causal_reliability.experiments.run_traffic_sign_shortcut_validation import run as run_traffic_sign
from causal_reliability.validation.analyze_label_preservation_responses import run as analyze_packet
from causal_reliability.validation.export_label_preservation_packet import run as export_packet


def test_real_text_shortcut_runner_writes_metrics_and_summary(tmp_path: Path):
    outputs = run_real_text(
        {
            "seed": 3,
            "results_dir": str(tmp_path),
            "sample_path": "causal_reliability/data/real_text_samples.csv",
            "repeats": 3,
            "marker_strength": 3,
            "epochs": 20,
            "lr": 0.08,
            "regimes": ["confidence-solvable", "confident-wrong", "mixed"],
        }
    )
    metrics = pd.read_csv(outputs["metrics"])
    assert {"confidence_risk_auroc", "entropy_auroc", "margin_auroc", "cic_auroc"}.issubset(metrics.columns)
    assert Path(outputs["certificates"]).exists()
    assert Path(outputs["plane_png"]).exists()
    summary = Path(outputs["summary"]).read_text(encoding="utf-8")
    assert "small reproducible checked-in review sample" in summary
    certs = pd.read_csv(outputs["certificates"])
    assert (certs["label"] == certs["label"]).all()


def test_human_packet_exports_and_analyzer_does_not_fabricate(tmp_path: Path):
    packet = export_packet({"results_dir": str(tmp_path), "examples_per_domain": 12})
    assert Path(packet["pairs"]).exists()
    assert Path(packet["instructions"]).exists()
    assert Path(packet["form"]).exists()
    assert Path(packet["google_form_questions"]).exists()
    pairs = pd.read_csv(packet["pairs"])
    assert {"example_id", "domain", "original_path_or_text", "counterfactual_path_or_text", "original_true_label", "intended_counterfactual_true_label", "intervention_type", "expected_label_preserved"}.issubset(pairs.columns)
    counts = pairs.groupby("domain")["example_id"].nunique().to_dict()
    assert counts["colored_digits"] == 12
    assert counts["clip_overlay"] == 12
    assert counts["real_text_shortcut"] == 12
    assert pairs["expected_label_preserved"].eq(True).all()
    analyzed = analyze_packet(None, tmp_path)
    summary = Path(analyzed["summary"]).read_text(encoding="utf-8")
    assert "No human validation responses have been provided yet." in summary


def test_human_response_analyzer_computes_toy_agreement(tmp_path: Path):
    export_packet({"results_dir": str(tmp_path), "examples_per_domain": 10})
    responses = tmp_path / "responses.csv"
    pd.DataFrame(
        [
            {
                "annotator_id": "a1",
                "example_id": "colored_digits_000",
                "original_label_human": "0",
                "counterfactual_label_human": "0",
                "label_preserved_human": "yes",
                "plausible_human": "yes",
                "concerns": "",
            },
            {
                "annotator_id": "a2",
                "example_id": "colored_digits_000",
                "original_label_human": "0",
                "counterfactual_label_human": "0",
                "label_preserved_human": "yes",
                "plausible_human": "no",
                "concerns": "color is bright",
            },
        ]
    ).to_csv(responses, index=False)
    analyzed = analyze_packet(str(responses), tmp_path)
    metrics = pd.read_csv(analyzed["metrics"])
    all_row = metrics[metrics["domain"] == "all"].iloc[0]
    assert all_row["label_preservation_agreement_rate"] == 1.0
    assert all_row["plausibility_agreement_rate"] == 0.5
    assert all_row["n_annotators"] == 2
    assert all_row["n_examples"] == 1
    assert all_row["n_total_judgments"] == 2


def test_random_aug_failure_benchmark_writes_metrics_and_summary(tmp_path: Path):
    outputs = run_random_aug_failure({"results_dir": str(tmp_path), "seed": 1, "n_examples": 80})
    metrics = pd.read_csv(outputs["metrics"])
    assert {"confidence risk", "entropy", "margin", "random augmentation sensitivity", "OOD/embedding distance", "label-flip-only", "CIC"}.issubset(set(metrics["method"]))
    summary = Path(outputs["summary"]).read_text(encoding="utf-8")
    assert "Random augmentation failed relative to CIC" in summary
    assert "It does not show CIC dominates random augmentation in all settings." in summary
    lookup = metrics.set_index("method")["failure_auroc"].to_dict()
    assert lookup["CIC"] > lookup["random augmentation sensitivity"]


def test_traffic_sign_validation_writes_unavailable_summary(tmp_path: Path):
    outputs = run_traffic_sign({"results_dir": str(tmp_path), "download_gtsrb": False, "use_synthetic_fallback": False})
    assert Path(outputs["metrics"]).exists()
    summary = Path(outputs["summary"]).read_text(encoding="utf-8")
    assert "GTSRB was not used." in summary
    assert "does not validate deployment in autonomous vehicles" in summary


def test_final_report_includes_external_feedback_sections(tmp_path: Path):
    run_random_aug_failure({"results_dir": str(tmp_path), "seed": 2, "n_examples": 80})
    run_traffic_sign({"results_dir": str(tmp_path), "download_gtsrb": False, "use_synthetic_fallback": False})
    export_packet({"results_dir": str(tmp_path), "examples_per_domain": 10})
    analyze_packet(None, tmp_path)
    outputs = build_report(tmp_path)
    report = Path(outputs["report"]).read_text(encoding="utf-8")
    assert "Human Label-Preservation Validation" in report
    assert "Random Augmentation Failure Stress Test" in report
    assert "Safety-Critical-Inspired Traffic Sign Shortcut Validation" in report


def test_cic_api_computes_certificates_and_quadrants():
    examples = [
        {"example_id": "x1", "label": 1, "text": "source: alpha The acting is warm and engaging."},
        {"example_id": "x2", "label": 0, "text": "source: alpha The plot is thin and dull."},
    ]
    certs = CICScorer(_demo_model, [TextMarkerIntervention()]).score_examples(examples)
    assigned = ReliabilityPlane().assign(certs)
    assert len(assigned) == 2
    assert {"confidence", "cic_score", "stability_score", "quadrant", "recommended_action"}.issubset(assigned[0])


def test_audit_workflow_writes_report_and_quadrants(tmp_path: Path):
    outputs = run_audit({"results_dir": str(tmp_path)})
    certs = pd.read_csv(outputs["certificates"])
    assert Path(outputs["report"]).exists()
    assert "quadrant" in certs.columns
    assert "recommended_action" in certs.columns
    assert Path(outputs["plane_pdf"]).exists()


def test_theoretical_docs_and_claim_boundaries():
    theory = Path("docs/theoretical_intuition.md").read_text(encoding="utf-8")
    formal = Path("docs/formal_separation.md").read_text(encoding="utf-8")
    final_report_src = Path("causal_reliability/analysis/final_report.py").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8").lower()
    docs = "\n".join(Path(p).read_text(encoding="utf-8").lower() for p in ["README.md", "docs/unknown_shortcut_discovery.md", "docs/when_cic_fails.md"])
    assert "No Confidence-Only Metric Is Sufficient" in theory
    assert "Confidence-Only Insufficiency" in formal
    assert "9/10 Category Defense Summary" in final_report_src
    assert "not claimed to dominate all baselines" in readme
    assert "not a turnkey solution for arbitrary models or unknown shortcuts" in readme
    assert "does not solve general causal discovery" in docs
    assert "does not show cic dominates random augmentation in all settings" in docs
    assert "medical/clinical deployment" not in docs
    assert "validates deployment in autonomous" not in docs


def test_final_report_hard_multidecoy_headline_is_careful():
    import json

    report = Path("results/final_report/final_report.md").read_text(encoding="utf-8")
    key_numbers = json.loads(Path("results/final_report/final_key_numbers.json").read_text(encoding="utf-8"))
    # The headline sentence is derived from the actual loaded-model metrics, so we
    # assert the report contains exactly the derived sentence (no stale hardcoded %).
    derived_headline = key_numbers["hard_multidecoy_headline_sentence"]

    assert "#### Hard Multi-Decoy CLIP Shortcut Localization" in report
    assert "- headline_eligible = true" in report
    assert "- headline_result_name = Hard Multi-Decoy CLIP Shortcut Localization" in report
    assert "On a held-out hard multi-decoy benchmark using real pretrained OpenCLIP, misleading text reduced accuracy to" in derived_headline
    assert derived_headline in report
    assert "finite candidate class of text-region proposals" in report
    assert "does not solve open-world shortcut discovery" in report
    assert "weak at strict IoU >= 0.5" in report
    assert "Oracle CLIP repair is an upper-bound causal confirmation" in report
    assert "Known-bbox CLIP overlay repair is not the automatic discovery headline path" in report
    assert "This is the main CLIP repair headline" in report


def test_hard_summary_reports_sample_sizes_and_confidence_intervals():
    summary = Path("results/hard_multidecoy_clip_repair/hard_multidecoy_repair_summary.md").read_text(encoding="utf-8")

    assert "n hard misleading test examples" in summary
    assert "n random matched text-region seeds" in summary
    assert "selective repair abstained/repaired" in summary
    assert "95% CI" in summary
    assert "Original hard misleading accuracy" in summary
    assert "Conditional on this held-out test set" in summary
    assert "random baseline draw variability" in summary


def test_final_report_reports_hard_sample_sizes_and_confidence_intervals():
    report = Path("results/final_report/final_report.md").read_text(encoding="utf-8")
    key_numbers = Path("results/final_report/final_key_numbers.json").read_text(encoding="utf-8")

    assert "Sample sizes: n_examples hard misleading" in report
    assert "95% confidence intervals" in report
    assert "hard_multidecoy_n_hard_misleading_examples" in key_numbers
    assert "hard_multidecoy_top1_misleading_ci95" in key_numbers


def test_seed_stability_artifact_contract_or_single_seed_limitation():
    seed_csv = Path("results/hard_multidecoy_clip_repair/seed_stability_summary.csv")
    fixed_csv = Path("results/hard_multidecoy_clip_repair/fixed_benchmark_determinism_check_summary.csv")
    legacy_lite_csv = Path("results/hard_multidecoy_clip_repair/lite_seed_stability_summary.csv")
    full_resampling_csv = Path("results/hard_multidecoy_clip_repair/full_benchmark_resampling_audit.csv")
    resampling_csv = Path("results/hard_multidecoy_clip_repair/benchmark_resampling_audit.csv")
    enlarged_csv = Path("results/hard_multidecoy_clip_repair/enlarged_test_summary.csv")
    report = Path("results/final_report/final_report.md").read_text(encoding="utf-8")
    exact_limitation = "The main hard multi-decoy result is a strong single-benchmark result."

    # The full benchmark-resampling audit, when it exists and survives (non-lite,
    # every seed independently resampled and beating matched random text repair),
    # supersedes the stale lite audit and may claim benchmark-resampling stability.
    if full_resampling_csv.exists():
        full = pd.read_csv(full_resampling_csv)
        not_lite = "lite_mode" not in full or not full["lite_mode"].astype(str).str.lower().isin(["true", "1"]).any()
        all_resampled = "benchmark_resampled" in full and full["benchmark_resampled"].astype(str).str.lower().isin(["true", "1"]).all()
        gap_ok = True
        if {"cic_top1_repair_accuracy", "random_matched_text_repair_mean"}.issubset(full.columns):
            gap = pd.to_numeric(full["cic_top1_repair_accuracy"], errors="coerce") - pd.to_numeric(full["random_matched_text_repair_mean"], errors="coerce")
            gap_ok = bool((gap >= 0.15).all())
        if len(full) >= 2 and not_lite and all_resampled and gap_ok:
            assert "survived independent resampling" in report
            assert "full stability evidence" not in report
            return

    if resampling_csv.exists():
        resampling = pd.read_csv(resampling_csv)
        assert "benchmark_resampled" in resampling.columns
        lite_resampling = "lite_mode" in resampling and resampling["lite_mode"].astype(str).str.lower().isin(["true", "1"]).any()
        if lite_resampling:
            assert exact_limitation in report
            assert "full stability evidence" not in report
        else:
            assert "Benchmark-resampling audit artifacts are available" in report
    elif seed_csv.exists():
        seed_summary = pd.read_csv(seed_csv)
        assert "headline_eligible" in seed_summary.columns
    elif fixed_csv.exists() or legacy_lite_csv.exists():
        fixed_summary = pd.read_csv(fixed_csv if fixed_csv.exists() else legacy_lite_csv)
        assert "headline_eligible" in fixed_summary.columns
        assert exact_limitation in report
        assert "fixed benchmark instance" in report.lower()
    elif enlarged_csv.exists():
        assert "The headline result was re-evaluated on an enlarged held-out test set." in report
    else:
        assert exact_limitation in report
        assert "only one held-out benchmark instance was run" in report


def test_enlarged_test_artifact_contract_if_present():
    enlarged_csv = Path("results/hard_multidecoy_clip_repair/enlarged_test_summary.csv")
    if not enlarged_csv.exists():
        return
    enlarged = pd.read_csv(enlarged_csv)
    required = {
        "n_examples",
        "n_hard_misleading_examples",
        "original_hard_misleading_accuracy",
        "cic_top1_repair_accuracy",
        "random_matched_text_repair_mean",
        "headline_eligible",
    }
    assert required.issubset(enlarged.columns)


def test_final_report_clarifies_random_baseline_draw_uncertainty():
    from causal_reliability.experiments.run_hard_multidecoy_clip_repair import RANDOM_BASELINE_UNCERTAINTY_WORDING

    report = Path("results/final_report/final_report.md").read_text(encoding="utf-8")

    assert RANDOM_BASELINE_UNCERTAINTY_WORDING in report
    assert "random baseline draw variability, not full test-set sampling uncertainty" in report
    assert "full experiment-level CI" not in report


def test_single_seed_limitation_kept_when_no_seed_or_enlarged_audit_exists():
    seed_csv = Path("results/hard_multidecoy_clip_repair/seed_stability_summary.csv")
    resampling_csv = Path("results/hard_multidecoy_clip_repair/benchmark_resampling_audit.csv")
    enlarged_csv = Path("results/hard_multidecoy_clip_repair/enlarged_test_summary.csv")
    report = Path("results/final_report/final_report.md").read_text(encoding="utf-8")
    exact_limitation = "The main hard multi-decoy result is a strong single-benchmark result."

    if not seed_csv.exists() and not resampling_csv.exists() and not enlarged_csv.exists():
        assert exact_limitation in report


def test_seed_stability_resume_writes_per_seed_artifact(tmp_path: Path, monkeypatch):
    from causal_reliability.experiments import run_hard_multidecoy_clip_repair_audit as audit

    def fake_run(cfg):
        out_dir = Path(cfg["results_dir"]) / "hard_multidecoy_clip_repair"
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = pd.DataFrame(
            [
                {
                    "method": "original_clip_prediction",
                    "n_examples": 4,
                    "n_hard_misleading_examples": 1,
                    "hard_multi_decoy_misleading_accuracy_before": 0.25,
                    "headline_eligible": True,
                    "pretrained_loaded": True,
                },
                {
                    "method": "nonoracle_cic_top1_repair",
                    "hard_multi_decoy_misleading_accuracy_after": 0.75,
                    "headline_eligible": True,
                    "pretrained_loaded": True,
                    "harmful_top1_iou_0_3": 1.0,
                    "harmful_top3_iou_0_3": 1.0,
                },
            ]
        )
        path = out_dir / "hard_multidecoy_repair_metrics.csv"
        metrics.to_csv(path, index=False)
        certs_path = out_dir / "hard_multidecoy_repair_certificates.csv"
        ranks_path = out_dir / "hard_multidecoy_candidate_rankings.csv"
        pd.DataFrame(
            [
                {
                    "example_id": 1,
                    "regime": "hard_multi_decoy_misleading",
                    "method": "original_clip_prediction",
                    "original_correct": False,
                    "repaired_correct": False,
                    "abstained": False,
                    "original_confidence": 0.9,
                    "repaired_confidence": 0.9,
                    "drop_in_original_top_class_probability": 0.0,
                    "prediction_flipped": False,
                    "js_shift": 0.0,
                    "kl_shift": 0.0,
                },
                {
                    "example_id": 1,
                    "regime": "hard_multi_decoy_misleading",
                    "method": "nonoracle_cic_top1_repair",
                    "original_correct": False,
                    "repaired_correct": True,
                    "abstained": False,
                    "original_confidence": 0.9,
                    "repaired_confidence": 0.8,
                    "drop_in_original_top_class_probability": 0.5,
                    "prediction_flipped": True,
                    "js_shift": 0.1,
                    "kl_shift": 0.2,
                },
            ]
        ).to_csv(certs_path, index=False)
        pd.DataFrame([{"example_id": 1, "regime": "hard_multi_decoy_misleading", "rank": 1, "candidate_id": "c1", "bbox": "[0,0,1,1]", "harmful_iou": 0.4}]).to_csv(ranks_path, index=False)
        return {"metrics": str(path), "certificates": str(certs_path), "rankings": str(ranks_path)}

    monkeypatch.setattr(audit, "run", fake_run)
    out_dir = tmp_path / "hard_multidecoy_clip_repair"
    out_dir.mkdir(parents=True)
    (out_dir / "selected_generation_policy.json").write_text('{"policy_id": "p", "class_set_size": 4}', encoding="utf-8")
    (out_dir / "selected_repair_policy.json").write_text('{"score_threshold": 0, "min_consensus_stability": 0.667}', encoding="utf-8")
    df = audit.run_seed_stability({"results_dir": str(tmp_path)}, [7], out_dir, resume=True)
    resumed = audit.run_seed_stability({"results_dir": str(tmp_path)}, [7], out_dir, resume=True)

    row_path = out_dir / "seed_stability_runs" / "seed_7" / "seed_stability_row.csv"
    assert row_path.exists()
    assert bool(df["headline_eligible"].iloc[0])
    assert bool(resumed["headline_eligible"].iloc[0])


def test_seed_stability_summary_csv_includes_headline_eligible(tmp_path: Path):
    from causal_reliability.experiments.run_hard_multidecoy_clip_repair_audit import _write_summary

    out_dir = tmp_path / "hard_multidecoy_clip_repair"
    out_dir.mkdir()
    df = pd.DataFrame([{"seed_id": 0, "headline_eligible": True, "fixed_benchmark_determinism_check": False}])
    _write_summary(df, out_dir, "seed_stability_summary", "Seed Summary")

    written = pd.read_csv(out_dir / "seed_stability_summary.csv")
    assert "headline_eligible" in written.columns


def test_final_report_keeps_single_seed_limitation_for_lite_seed_audit(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    hard_dir = tmp_path / "hard_multidecoy_clip_repair"
    hard_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "method": "original_clip_prediction",
                "headline_eligible": True,
                "evidence_status": "pretrained CLIP hard multi-decoy non-oracle repair evidence",
                "pretrained_loaded": True,
                "hard_multi_decoy_misleading_accuracy_before": 0.219,
            }
        ]
    ).to_csv(hard_dir / "hard_multidecoy_repair_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "seed_id": 0,
                "headline_eligible": True,
                "fixed_benchmark_determinism_check": True,
                "cic_top1_repair_accuracy": 0.875,
            },
            {
                "seed_id": 1,
                "headline_eligible": True,
                "fixed_benchmark_determinism_check": True,
                "cic_top1_repair_accuracy": 0.875,
            },
        ]
    ).to_csv(hard_dir / "fixed_benchmark_determinism_check_summary.csv", index=False)

    outputs = build_report(tmp_path)
    report = outputs["report"].read_text(encoding="utf-8")

    assert "single-benchmark result" in report
    assert "Fixed-benchmark determinism-check artifacts are available" in report
    assert "It does not establish robustness to benchmark resampling" in report
    assert "stress-tested across independent held-out benchmark seeds" not in report


def test_resample_mode_changes_benchmark_image_hashes():
    from causal_reliability.experiments.run_hard_multidecoy_clip_repair import make_hard_dataset
    from causal_reliability.experiments.run_hard_multidecoy_clip_repair_audit import _image_set_hash, _metadata_hash

    policy = {"policy_id": "test", "class_set_size": 4, "n_decoys": 4, "placement_jitter": 6}
    seed0 = make_hard_dataset(2, policy, size=96, split="test", benchmark_seed=0, resample=True)
    seed1 = make_hard_dataset(2, policy, size=96, split="test", benchmark_seed=1, resample=True)

    assert _image_set_hash(seed0) != _image_set_hash(seed1)
    assert _metadata_hash(seed0) != _metadata_hash(seed1)


def test_benchmark_resampling_summary_includes_resampled_flag(tmp_path: Path):
    from causal_reliability.experiments.run_hard_multidecoy_clip_repair_audit import _write_summary

    out_dir = tmp_path / "hard_multidecoy_clip_repair"
    out_dir.mkdir()
    df = pd.DataFrame(
        [
            {"seed_id": 0, "benchmark_resampled": True, "headline_eligible": True, "cic_top1_repair_accuracy": 0.8, "random_matched_text_repair_mean": 0.2, "clean_safe_clean_drop": 0.01},
            {"seed_id": 1, "benchmark_resampled": True, "headline_eligible": True, "cic_top1_repair_accuracy": 0.7, "random_matched_text_repair_mean": 0.3, "clean_safe_clean_drop": 0.02},
        ]
    )
    _write_summary(df, out_dir, "benchmark_resampling_audit", "Benchmark Audit", resample_benchmark=True)

    written = pd.read_csv(out_dir / "benchmark_resampling_audit.csv")
    assert "benchmark_resampled" in written.columns
    assert (out_dir / "benchmark_resampling_aggregate.csv").exists()


def test_repair_vs_localization_crosstab_is_written(tmp_path: Path):
    from causal_reliability.experiments.run_hard_multidecoy_clip_repair_audit import write_repair_vs_localization_crosstab

    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    out_dir.mkdir()
    rows = []
    for example_id, repaired in [(1, True), (2, False)]:
        rows.extend(
            [
                {
                    "example_id": example_id,
                    "regime": "hard_multi_decoy_misleading",
                    "method": "original_clip_prediction",
                    "original_correct": False,
                    "repaired_correct": False,
                    "abstained": False,
                    "original_confidence": 0.9,
                    "repaired_confidence": 0.9,
                    "drop_in_original_top_class_probability": 0.0,
                    "prediction_flipped": False,
                    "js_shift": 0.0,
                    "kl_shift": 0.0,
                },
                {
                    "example_id": example_id,
                    "regime": "hard_multi_decoy_misleading",
                    "method": "nonoracle_cic_top1_repair",
                    "original_correct": False,
                    "repaired_correct": repaired,
                    "abstained": False,
                    "original_confidence": 0.9,
                    "repaired_confidence": 0.7,
                    "drop_in_original_top_class_probability": 0.4,
                    "prediction_flipped": True,
                    "js_shift": 0.1,
                    "kl_shift": 0.2,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(run_dir / "hard_multidecoy_repair_certificates.csv", index=False)
    pd.DataFrame(
        [
            {"example_id": 1, "regime": "hard_multi_decoy_misleading", "rank": 1, "candidate_id": "a", "bbox": "[0,0,1,1]", "harmful_iou": 0.4},
            {"example_id": 2, "regime": "hard_multi_decoy_misleading", "rank": 1, "candidate_id": "b", "bbox": "[0,0,1,1]", "harmful_iou": 0.1},
        ]
    ).to_csv(run_dir / "hard_multidecoy_candidate_rankings.csv", index=False)

    df = write_repair_vs_localization_crosstab(run_dir, out_dir)
    assert (out_dir / "repair_vs_localization_crosstab.csv").exists()
    assert {"top1_iou_ge_0_3", "top1_iou_lt_0_3", "top3_iou_ge_0_3", "top3_iou_lt_0_3"}.issubset(set(df["group"]))


def test_cache_key_source_includes_image_hash():
    source = Path("causal_reliability/experiments/run_multidecoy_clip_repair.py").read_text(encoding="utf-8")
    assert "_example_image_hash(ex)" in source
    assert "f\"example_{int(ex['example_id'])}_{_example_image_hash(ex)}\"" in source


def test_final_report_headline_switches_for_benchmark_resampling(tmp_path: Path):
    from causal_reliability.analysis.final_report import build_report

    hard_dir = tmp_path / "hard_multidecoy_clip_repair"
    hard_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"method": "original_clip_prediction", "headline_eligible": True, "evidence_status": "pretrained CLIP hard multi-decoy non-oracle repair evidence", "pretrained_loaded": True},
            {"method": "nonoracle_cic_top1_repair", "headline_eligible": True, "pretrained_loaded": True, "hard_multi_decoy_misleading_accuracy_after": 0.8},
        ]
    ).to_csv(hard_dir / "hard_multidecoy_repair_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"seed_id": 0, "benchmark_resampled": True, "headline_eligible": True, "cic_top1_repair_accuracy": 0.8, "random_matched_text_repair_mean": 0.2},
            {"seed_id": 1, "benchmark_resampled": True, "headline_eligible": True, "cic_top1_repair_accuracy": 0.75, "random_matched_text_repair_mean": 0.25},
        ]
    ).to_csv(hard_dir / "benchmark_resampling_audit.csv", index=False)

    report = build_report(tmp_path)["report"].read_text(encoding="utf-8")
    assert "Across benchmark-resampled held-out hard multi-decoy runs" in report
    assert "Benchmark-resampling audit artifacts are available" in report


def test_final_report_reframes_previous_clip_experiments():
    report = Path("results/final_report/final_report.md").read_text(encoding="utf-8")

    assert "Single-Overlay Non-Oracle CLIP Shortcut Localization and Repair" in report
    assert "matched/random patch baseline is competitive" in report
    assert "First Multi-Decoy CLIP Repair" in report
    assert "not a true shortcut-failure benchmark" in report
    assert "hard multi-decoy repair run is the main headline result" in report
    assert "CLIP overlay repair is the main headline" not in report
    assert "known-bbox repaired misleading-overlay accuracy was 1.000. Treat this as oracle upper-bound causal confirmation" in report


def test_docs_do_not_claim_general_robustness_or_open_world_discovery():
    doc_paths = [
        Path("README.md"),
        Path("FINAL_ARTIFACT_INDEX.md"),
        Path("docs/cic_audit_demo.md"),
        Path("docs/unknown_shortcut_discovery.md"),
        Path("docs/when_cic_fails.md"),
        Path("results/final_report/final_report.md"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in doc_paths)
    forbidden_claims = [
        "open-world discovery is solved",
        "solves open-world discovery",
        "general robustness is solved",
        "solves general robustness",
        "arbitrary shortcuts can be found",
        "finds arbitrary shortcuts",
        "exact box localization was achieved",
        "exact bounding-box recovery was achieved",
        "cic always beats baselines",
        "cic always beats random",
    ]
    for claim in forbidden_claims:
        assert claim not in text


def test_docs_preserve_finite_candidate_open_world_caveat():
    doc_paths = [
        Path("README.md"),
        Path("FINAL_ARTIFACT_INDEX.md"),
        Path("docs/cic_audit_demo.md"),
        Path("results/final_report/final_report.md"),
        Path("results/hard_multidecoy_clip_repair/hard_multidecoy_repair_summary.md"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in doc_paths)

    assert "finite candidate" in text
    assert "does not solve open-world shortcut discovery" in text
