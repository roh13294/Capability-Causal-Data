# Discovery Failure Cases

These controls bound the moonshot claim. They do not test general causal discovery; they test a finite intervention class.

## Metrics

| case                        | expectation                                                                                       | top_candidate_name | top_candidate_type | top_candidate_group | top_score | second_score | dominance_gap | top_label_preservation | true_shortcut_rank | top3_true_shortcut_hit | audit_candidate_name | audit_candidate_rank | audit_candidate_score | audit_candidate_label_preservation | audit_candidate_support | audit_candidate_specificity |
| --------------------------- | ------------------------------------------------------------------------------------------------- | ------------------ | ------------------ | ------------------- | --------- | ------------ | ------------- | ---------------------- | ------------------ | ---------------------- | -------------------- | -------------------- | --------------------- | ---------------------------------- | ----------------------- | --------------------------- |
| no_shortcut                 | No candidate should dominate strongly; no true shortcut exists.                                   | feature_dim_0      | individual_feature | irrelevant          | 0.06666   | 0.05091      | 0.01575       | 0.35                   | -1                 | False                  |                      |                      |                       |                                    |                         |                             |
| multiple_shortcuts          | At least one true shortcut should rank in the top-3.                                              | feature_dim_1      | individual_feature | true_shortcut       | 0.1916    | 0.1436       | 0.04807       | 1                      | 1                  | True                   |                      |                      |                       |                                    |                         |                             |
| causal_feature_intervention | Causal interventions can destabilize predictions but should be penalized by label preservation.   | feature_dim_1      | individual_feature | irrelevant          | 0.2423    | 0.1828       | 0.0595        | 1                      | 6                  | False                  | feature_dim_0        | 6                    | 0.03083               | 0.35                               | 0.96                    | 1                           |
| corruption_intervention     | Global corruption can destabilize predictions but should be penalized by support and specificity. | object_color       | object_color       | true_shortcut       | 0.1816    | 0.1284       | 0.05319       | 1                      | 1                  | True                   | additive_noise       | 8                    | 0.0002541             | 1                                  | 0.68                    | 0.58                        |
| missing_true_shortcut       | Discovery should fail honestly when the true shortcut is absent from the candidate class.         | feature_dim_1      | individual_feature | missing             | 0.1596    | 0.1219       | 0.03767       | 1                      | -1                 | False                  |                      |                      |                       |                                    |                         |                             |

## Where Discovery Works

multiple_shortcuts, corruption_intervention

## Where Discovery Fails

no_shortcut, causal_feature_intervention, missing_true_shortcut

These failures are expected when no shortcut exists, when the shortcut is not represented in the candidate class, or when destabilizing interventions are causal-label-changing or broad corruptions that the audit penalizes.
