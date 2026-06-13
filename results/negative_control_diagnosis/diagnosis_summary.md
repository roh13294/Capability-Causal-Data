# Negative-Control Diagnosis

| control | n | mean_shift_risk | mean_margin_collapse | mean_confidence | failure_rate | corr_with_confidence | corr_with_margin | corr_with_input_difficulty | corr_with_shifted_failure | corr_with_true_shift_risk | accidental_label_change_rate | mean_x_to_shuffled_distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| irrelevant_counterfactual | 120 | 0.067 | 0.014 | 0.564 | 0.300 | 0.247 | 0.247 | -0.247 | -0.671 | 0.062 | nan | nan |
| random_intervention_direction | 120 | 0.073 | 0.018 | 0.572 | 0.333 | 0.288 | 0.288 | -0.288 | -0.077 | -0.081 | nan | nan |
| random_labels | 120 | 0.279 | 0.026 | 0.521 | 0.533 | 0.374 | 0.374 | -0.374 | 0.008 | 0.120 | nan | nan |
| shuffled_any | 120 | 0.319 | 0.067 | 0.562 | 0.175 | 0.016 | 0.016 | -0.016 | 0.312 | 0.071 | nan | nan |
| shuffled_matched_confidence | 120 | 0.172 | 0.015 | 0.543 | 0.158 | -0.152 | -0.152 | 0.152 | 0.290 | -0.047 | nan | nan |
| shuffled_same_shortcut | 120 | 0.152 | 0.042 | 0.560 | 0.467 | 0.113 | 0.113 | -0.113 | -0.102 | -0.063 | 0.000 | nan |
| shuffled_within_class | 120 | 0.406 | 0.106 | 0.582 | 0.375 | 0.308 | 0.308 | -0.308 | -0.062 | 0.045 | 0.000 | nan |
| true_counterfactual | 120 | 0.676 | 0.071 | 0.561 | 0.508 | 0.406 | 0.406 | -0.406 | 0.202 | 1.000 | nan | nan |
