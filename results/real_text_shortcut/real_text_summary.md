# Real Text Shortcut Validation

Dataset/source used: `causal_reliability/data/real_text_samples.csv`.
Dataset status: small reproducible checked-in review sample, not a large benchmark.
Model used: torch linear TF-IDF bag-of-words classifier.
Shortcut marker design: neutral metadata prefixes such as `source: alpha` and `source: beta` are correlated with labels during training.
Label preservation: removing, replacing, or flipping the metadata marker preserves the original review text and true sentiment label because the marker is neutral source metadata rather than sentiment content.

This benchmark extends CIC to a real text classification domain with controlled shortcut injection. It does not prove full open-world shortcut discovery.

## Metrics

| regime              | dataset_status                              | model                                       | accuracy | high_confidence_failure_rate | dangerous_quadrant_failure_rate | confidence_risk_auroc | entropy_auroc | margin_auroc | random_token_perturbation_sensitivity_auroc | shortcut_marker_counterfactual_sensitivity_auroc | cic_auroc |
| ------------------- | ------------------------------------------- | ------------------------------------------- | -------- | ---------------------------- | ------------------------------- | --------------------- | ------------- | ------------ | ------------------------------------------- | ------------------------------------------------ | --------- |
| confidence-solvable | small reproducible checked-in review sample | torch linear TF-IDF bag-of-words classifier | 0.8833   | 0.1167                       |                                 | 0.8173                | 0.8173        | 0.8173       | 0.5897                                      | 0.5897                                           | 0.5897    |
| confident-wrong     | small reproducible checked-in review sample | torch linear TF-IDF bag-of-words classifier | 0.5      | 0.5                          | 1                               | 0.622                 | 0.64          | 0.622        | 0.48                                        | 1                                                | 1         |
| mixed               | small reproducible checked-in review sample | torch linear TF-IDF bag-of-words classifier | 0.9056   | 0.09444                      |                                 | 0.8957                | 0.8957        | 0.8957       | 0.5994                                      | 1                                                | 1         |

## Limitations

- The default dataset is a small checked-in review-like sample for reproducibility.
- The shortcut is controlled and supplied to the scorer.
- The result does not imply CIC always beats confidence or discovers arbitrary unknown shortcuts.
