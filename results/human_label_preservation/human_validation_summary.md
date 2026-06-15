# Human Label-Preservation Validation Summary

- Annotators: 3
- Image pairs: 100
- Total annotations: 300
- Majority-vote before-label accuracy (vs. true label): n/a
- Majority-vote after-label accuracy (vs. true label): n/a
- Majority-vote label-preservation rate: 0.960
- Majority-vote after-recognizable rate: 0.970
- Unsure rate: 0.000
- Preservation failures (majority vote): 4

## Inter-annotator agreement

| Field | Percent agreement | Fleiss' kappa |
| --- | --- | --- |
| Before object label | 0.980 | 0.973 |
| After object label | 0.980 | 0.974 |
| Did object label change | 0.993 | 0.920 |
| After image recognizable | 1.000 | 1.000 |

## Preservation-failure characterization

Majority vote did not preserve the object label in 4 of 100 pairs. These pairs were retained and flagged rather than removed.

See `human_validation_flags.csv` for the per-pair detail. Categories observed:

- shape changed/covered: 1
- blurry/unrecognizable: 1
- corrupted/glitched: 1
- blank/missing shape: 1
