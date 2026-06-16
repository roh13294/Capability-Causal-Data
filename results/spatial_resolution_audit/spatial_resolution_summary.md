# Spatial-Resolution and Causal-Intervention Audit

**Scope.** This audit separates *exact box precision* (IoU with the oracle shortcut box) from *causal-intervention usefulness* (shortcut coverage, intervention bluntness, causal-content preservation, and repair). It reads existing benchmark artifacts only; no model is re-run and no headline metric is changed.

**This audit does NOT claim** exact localization, segmentation quality, open-world discovery, or general robustness. CIC is a coarse causal-intervention method.

## Benchmark availability

- `hard_multidecoy`: n=32, oracle_box=True, object_overlap=True, candidates=True
- `failure_conditioned`: n=50, oracle_box=True, object_overlap=True, candidates=True
- `semantic_decoy_n128`: n=128, oracle_box=False, object_overlap=False, candidates=False

## Pooled key numbers

| metric | value |
| --- | --- |
| n_examples | 210 |
| median_iou | 0.3935 |
| mean_iou | 0.5386 |
| hit_at_iou_0_1 | 0.7476 |
| hit_at_iou_0_2 | 0.7476 |
| hit_at_iou_0_3 | 0.7286 |
| hit_at_iou_0_4 | 0.4571 |
| hit_at_iou_0_5 | 0.4333 |
| shortcut_coverage_median | 0.875 |
| shortcut_coverage_ge_0_5_rate | 0.7683 |
| shortcut_coverage_ge_0_8_rate | 0.7073 |
| intersects_shortcut_rate | 0.9 |
| area_frac_image_median | 0.2114 |
| area_frac_oracle_median | 2.099 |
| object_iou_median | 0.1959 |
| object_overlap_available | True |
| repair_top1_accuracy | 0.7762 |
| repair_top3_accuracy | 0.5619 |
| repair_clean_safe_accuracy | 0.8048 |

## Repair-by-localization buckets

| group               | iou_bucket | n  | cic_top1_repair_accuracy | cic_top3_repair_accuracy | clean_safe_repair_accuracy | mean_area_frac_image |
| ------------------- | ---------- | -- | ------------------------ | ------------------------ | -------------------------- | -------------------- |
| ALL                 | <0.1       | 53 | 0.2642                   | 0.283                    | 0.4906                     | 0.1543               |
| ALL                 | 0.1-0.3    | 4  | 0.25                     | 0.25                     | 0.25                       | 0.1254               |
| ALL                 | 0.3-0.5    | 62 | 0.9194                   | 0.9194                   | 0.9355                     | 0.08601              |
| ALL                 | >=0.5      | 91 | 1                        | 0.4945                   | 0.9231                     | 0.1996               |
| failure_conditioned | <0.1       | 7  | 0.8571                   | 0.8571                   | 0.7143                     | 0.04583              |
| failure_conditioned | 0.1-0.3    | 2  | 0.5                      | 0.5                      | 0.5                        | 0.09782              |
| failure_conditioned | 0.3-0.5    | 39 | 1                        | 1                        | 1                          | 0.07629              |
| failure_conditioned | >=0.5      | 2  | 1                        | 1                        | 1                          | 0.02631              |
| hard_multidecoy     | <0.1       | 12 | 0.5833                   | 0.5833                   | 0.5833                     | 0.07427              |
| hard_multidecoy     | 0.1-0.3    | 1  | 0                        | 0                        | 0                          | 0.1273               |
| hard_multidecoy     | 0.3-0.5    | 17 | 0.8824                   | 0.8824                   | 0.8824                     | 0.07629              |
| hard_multidecoy     | >=0.5      | 2  | 1                        | 1                        | 1                          | 0.02631              |
| semantic_decoy_n128 | <0.1       | 34 | 0.02941                  | 0.05882                  | 0.4118                     | 0.205                |
| semantic_decoy_n128 | 0.1-0.3    | 1  | 0                        | 0                        | 0                          | 0.1786               |
| semantic_decoy_n128 | 0.3-0.5    | 6  | 0.5                      | 0.5                      | 0.6667                     | 0.1768               |
| semantic_decoy_n128 | >=0.5      | 87 | 1                        | 0.4713                   | 0.9195                     | 0.2076               |

## Refinement diagnostic

Geometric variants (shrink / 2x2 split / shifts) of the top region were re-scored using only non-oracle signals (pixel area + model-derived candidate consensus). Oracle box, true label, and correctness were NOT used to select.

- n: 82
- n_changed: 8
- orig_median_iou: 0.3935
- refined_median_iou: 0.3935
- orig_iou_ge_0_5_rate: 0.04878
- refined_iou_ge_0_5_rate: 0.04878
- orig_mean_area_frac: 0.0721
- refined_mean_area_frac: 0.07092
- orig_repair_accuracy: 0.878
- refined_repair_accuracy_evaluable: 0.92
- refined_repair_evaluable_n: 75
- refinement_improved_spatial_precision: False
- refined_clean_safe_drop: n/a (n/a: cannot re-score clean accuracy for synthetic regions from artifacts)

**Refinement did NOT improve spatial precision; reported honestly.** A non-oracle geometric refinement does not recover exact boxes here.

## Interpretation

- CIC is a coarse causal-intervention method, not an exact localization method: exact box precision is low (hit@IoU0.5 = 0.43) yet repair is high (top-1 repair = 0.78).
- Low IoU partly reflects larger-than-oracle intervention regions; the selected regions cover shortcut evidence (coverage>=0.5 in 0.77 of cases) but are spatially coarse.
- Although spatially coarse, the intervention is practically useful because it preserves causal content and clean accuracy.
- Exact localization remains a limitation: this audit does not claim exact localization, segmentation, open-world discovery, or general robustness.

## Outputs

- summary_md: `results/spatial_resolution_audit/spatial_resolution_summary.md`
- key_numbers_json: `results/spatial_resolution_audit/spatial_resolution_key_numbers.json`
- metrics_csv: `results/spatial_resolution_audit/spatial_resolution_metrics.csv`
- by_bucket_csv: `results/spatial_resolution_audit/spatial_resolution_by_bucket.csv`
- plot_png: `results/spatial_resolution_audit/spatial_resolution_plot.png`
