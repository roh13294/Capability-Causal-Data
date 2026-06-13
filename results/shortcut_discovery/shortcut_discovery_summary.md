# Shortcut Discovery Pilot

This is a controlled pilot, not a general shortcut-discovery method. It asks whether CIC-style interventions can identify which known input factors behave like shortcuts in a synthetic setting.

## Metrics

| task      | shortcut_rank | shortcut_top1_hit | shortcut_top3_hit | mean_shortcut_instability | mean_causal_instability | mean_noise_instability |
| --------- | ------------- | ----------------- | ----------------- | ------------------------- | ----------------------- | ---------------------- |
| synthetic | 1             | True              | True              | 0.4297                    | 0.00732                 | 0.001648               |

## Feature Ranking

| feature_dim | feature_type     | known_shortcut | average_instability | prediction_flip_rate | rank |
| ----------- | ---------------- | -------------- | ------------------- | -------------------- | ---- |
| 2           | shortcut feature | True           | 0.4297              | 0.5078               | 1    |
| 1           | causal feature   | False          | 0.008184            | 0                    | 2    |
| 0           | causal feature   | False          | 0.006456            | 0                    | 3    |
| 7           | noise            | False          | 0.003164            | 0                    | 4    |
| 4           | noise            | False          | 0.002392            | 0                    | 5    |
| 5           | noise            | False          | 0.001421            | 0                    | 6    |
| 3           | noise            | False          | 0.0008678           | 0                    | 7    |
| 6           | noise            | False          | 0.0003956           | 0                    | 8    |

Claim: in controlled settings, CIC can help identify which input factors behave like shortcuts. This does not imply discovery of arbitrary real-world causal variables.
