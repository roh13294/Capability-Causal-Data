# Final Results Checklist

## Commands Used

- `bash scripts/run_phase7.sh`
- `python3 -m pytest`

The direct `pytest` executable was not on PATH, so tests were run with `python3 -m pytest`.

## Test Status

- Phase 7 validation/regeneration command: passed.
- Full test suite: passed, 42 tests passed, 16 warnings.

## Seed Counts

- Final validation seeds: 3 per task/regime (`0, 1, 2`), from `configs/final_validation.yaml`.
- Final validation tasks: `synthetic`, `vision`, `text`.
- Final validation regimes: `confidence_solvable`, `confident_wrong`, `mixed`.
- Final negative control seeds: 3 per control (`0, 1, 2`), from `configs/final_negative_controls.yaml`.
- Final negative controls: `true_counterfactual`, `no_shortcut_correlation`, `random_labels`, `irrelevant_counterfactuals`, `shuffled_any`, `within_class_shuffled`, `same_shortcut_shuffled`, `matched_confidence_shuffled`.

## Final Artifact Paths

- Final validation summary CSV: `/Users/rohannagaram/Capability-Causal-Data-1/results/final_validation/final_validation_summary.csv`
- Final validation summary MD: `/Users/rohannagaram/Capability-Causal-Data-1/results/final_validation/final_validation_summary.md`
- Final validation metrics CSV: `/Users/rohannagaram/Capability-Causal-Data-1/results/final_validation/final_validation_metrics.csv`
- Final validation by-seed CSV: `/Users/rohannagaram/Capability-Causal-Data-1/results/final_validation/final_validation_by_seed.csv`
- Final validation certificates CSV: `/Users/rohannagaram/Capability-Causal-Data-1/results/final_validation/final_validation_certificates.csv`
- Final validation config used: `/Users/rohannagaram/Capability-Causal-Data-1/results/final_validation/final_validation_config_used.yaml`
- Final report MD: `/Users/rohannagaram/Capability-Causal-Data-1/results/final_report/final_report.md`
- Final claim table CSV: `/Users/rohannagaram/Capability-Causal-Data-1/results/final_report/final_claim_table.csv`
- Final claim table MD: `/Users/rohannagaram/Capability-Causal-Data-1/results/final_report/final_claim_table.md`
- Final key numbers JSON: `/Users/rohannagaram/Capability-Causal-Data-1/results/final_report/final_key_numbers.json`
- Main results table CSV: `/Users/rohannagaram/Capability-Causal-Data-1/results/main_results_table.csv`
- Main results table MD: `/Users/rohannagaram/Capability-Causal-Data-1/results/main_results_table.md`
- Main results summary JSON: `/Users/rohannagaram/Capability-Causal-Data-1/results/main_results_summary.json`
- STS figure PNG: `/Users/rohannagaram/Capability-Causal-Data-1/results/sts_main_figure.png`
- STS figure PDF: `/Users/rohannagaram/Capability-Causal-Data-1/results/sts_main_figure.pdf`
- STS figure caption MD: `/Users/rohannagaram/Capability-Causal-Data-1/results/sts_main_figure_caption.md`
- Final negative control metrics CSV: `/Users/rohannagaram/Capability-Causal-Data-1/results/final_negative_controls/final_negative_control_metrics.csv`
- Final negative control certificates CSV: `/Users/rohannagaram/Capability-Causal-Data-1/results/final_negative_controls/final_negative_control_certificates.csv`
- Final negative control summary MD: `/Users/rohannagaram/Capability-Causal-Data-1/results/final_negative_controls/final_negative_control_summary.md`
- Final negative control AUROC plot PNG: `/Users/rohannagaram/Capability-Causal-Data-1/results/final_negative_controls/plots/negative_control_auc.png`
- Final negative control CIC plot PNG: `/Users/rohannagaram/Capability-Causal-Data-1/results/final_negative_controls/plots/true_vs_control_cic.png`

## Final Validation Results By Regime

- Confidence-solvable: confidence AUROC mean = 1.000; CIC AUROC mean = 0.702; CIC - confidence mean = -0.298; mean failed confidence = 0.416. Confidence already detects these failures.
- Confident-wrong: confidence AUROC mean = 0.285; CIC AUROC mean = 1.000; CIC - confidence mean = 0.715; mean failed confidence = 0.911. CIC adds strong value in the high-confidence shortcut-failure regime.
- Mixed: confidence AUROC mean = 0.843; CIC AUROC mean = 0.894; CIC - confidence mean = 0.050; mean failed confidence = 0.686. Both confidence and CIC carry signal; this is supporting, not the central claim.

## Where Confidence Wins

- Confidence wins in all confidence-solvable rows.
- Synthetic confidence-solvable: confidence AUROC 1.000 vs CIC AUROC 0.754.
- Text confidence-solvable: confidence AUROC 1.000 vs CIC AUROC 0.611.
- Vision confidence-solvable: confidence AUROC 1.000 vs CIC AUROC 0.742.

## Where CIC Wins

- CIC wins in all confident-wrong rows.
- Synthetic confident-wrong: confidence AUROC 0.303 vs CIC AUROC 1.000; CIC - confidence = 0.697.
- Text confident-wrong: confidence AUROC 0.247 vs CIC AUROC 1.000; CIC - confidence = 0.753.
- Vision confident-wrong: confidence AUROC 0.304 vs CIC AUROC 0.999; CIC - confidence = 0.696.
- CIC also has modest positive differences in mixed rows: synthetic 0.045, text 0.011, vision 0.095.

## Negative Control Outcomes

- All final negative controls passed.
- True counterfactual control: CIC AUROC 1.000, passed.
- Irrelevant counterfactuals: CIC AUROC 0.447, passed.
- Matched-confidence shuffled: CIC AUROC 0.441, passed.
- No-shortcut correlation: CIC AUROC 0.558, passed.
- Random labels: CIC AUROC 0.457, passed.
- Same-shortcut shuffled: CIC AUROC 0.518, passed.
- Shuffled any: CIC AUROC 0.501, passed.
- Within-class shuffled: CIC AUROC 0.486, passed.

## Undefined Or Weak Results

- High-confidence CIC AUROC is undefined in all confidence-solvable rows because no failed examples have confidence >= 0.8.
- Mixed-regime CIC advantages are weak to modest and should not be framed as the main result.
- Confidence-solvable rows explicitly favor confidence, not CIC.

## Final Claim

Counterfactual Instability Certificates are complementary to confidence and are most useful for high-confidence shortcut failures, not ordinary confidence-solvable failures.

## Limitations

- CIC requires plausible shortcut-changing, label-preserving interventions.
- CIC is not a universal replacement for confidence.
- Controlled and semi-synthetic settings are still needed for clean causal evaluation.
- Results should not be overgeneralized to unknown real-world causal structures without domain-specific intervention design.
- Vision/text settings require careful calibration to avoid collapse or uninterpretable shortcut interventions.

## Use In Paper Or Poster

- Use the final validation summary and main results table.
- Emphasize the confident-wrong rows as the primary result.
- Emphasize the confidence-solvable rows as the boundary condition showing complementarity.
- Use the final negative controls to support that the CIC signal depends on meaningful counterfactual interventions.
- Use the STS figure PNG/PDF as the main visual summary.

## Exploratory Results Not To Emphasize

- Do not emphasize mixed-regime gains as the central claim.
- Do not present CIC as generally superior to confidence.
- Do not emphasize older exploratory sweep, ablation, mismatch, or diagnosis outputs unless clearly labeled as exploratory.
- Do not use undefined high-confidence confidence-solvable subset results as evidence for or against CIC.
