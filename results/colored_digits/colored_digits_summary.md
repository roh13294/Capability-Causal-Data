# Colored Digits Benchmark

Dataset source: generated seven-segment digit-like fallback.

This benchmark is supporting evidence. The true label is the digit class; the shortcut is object color. Training uses a high digit-color correlation and shifted evaluation uses an in-support color-label mapping flip.

| task           | dataset_source                              | seed_count | id_accuracy | shifted_accuracy | failure_count | correct_count | mean_failed_confidence | auroc_note | confidence_auroc | confidence_auroc_95_ci | entropy_auroc | entropy_auroc_95_ci | margin_auroc | margin_auroc_95_ci | old_shift_risk_auroc | old_shift_risk_auroc_95_ci | label_flip_only_auroc | label_flip_only_auroc_95_ci | cic_auroc | cic_auroc_95_ci | cic_minus_confidence_auroc | cic_minus_confidence_auroc_95_ci | high_confidence_cic_auroc | high_confidence_cic_auroc_note |
| -------------- | ------------------------------------------- | ---------- | ----------- | ---------------- | ------------- | ------------- | ---------------------- | ---------- | ---------------- | ---------------------- | ------------- | ------------------- | ------------ | ------------------ | -------------------- | -------------------------- | --------------------- | --------------------------- | --------- | --------------- | -------------------------- | -------------------------------- | ------------------------- | ------------------------------ |
| colored_digits | generated seven-segment digit-like fallback | 1          | 0.9883      | 0.1484           | 218           | 38            | 0.8815                 |            | 0.1095           | [0.072, 0.153]         | 0.1012        | [0.064, 0.146]      | 0.1024       | [0.066, 0.147]     | 0.9476               | [0.901, 0.984]             | 0.8382                | [0.804, 0.932]              | 0.9512    | [0.924, 0.972]  | 0.8417                     | [0.789, 0.889]                   | 0.8909                    |                                |

CIC is evaluated as a color-changing, label-preserving intervention. This does not claim CIC always beats confidence; it tests whether the framework applies in a recognizable shortcut setting.
