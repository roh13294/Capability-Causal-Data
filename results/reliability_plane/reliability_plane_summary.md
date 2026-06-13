# Reliability Plane Summary

Confidence threshold: `0.800`.
Stability threshold: `0.629`.

The dangerous quadrant is high confidence plus low counterfactual stability.

## Quadrant Counts And Failure Rates

| task      | regime              | quadrant                      | count | failure_count | failure_rate | mean_confidence | mean_counterfactual_stability | mean_cic |
| --------- | ------------------- | ----------------------------- | ----- | ------------- | ------------ | --------------- | ----------------------------- | -------- |
| synthetic | confidence-solvable | Dangerous shortcut reliance   | 71    | 0             | 0            | 0.8838          | 0.5253                        | 0.9306   |
| synthetic | confidence-solvable | Generally fragile             | 74    | 68            | 0.9189       | 0.4457          | 0.4824                        | 1.015    |
| synthetic | confidence-solvable | Reliable prediction           | 121   | 0             | 0            | 0.8827          | 0.7678                        | 0.4552   |
| synthetic | confidence-solvable | Uncertain but causally stable | 22    | 19            | 0.8636       | 0.451           | 0.744                         | 0.5019   |
| synthetic | confident-wrong     | Dangerous shortcut reliance   | 128   | 119           | 0.9297       | 0.9092          | 0.1709                        | 1.625    |
| synthetic | confident-wrong     | Generally fragile             | 2     | 1             | 0.5          | 0.7834          | 0.3293                        | 1.315    |
| synthetic | confident-wrong     | Reliable prediction           | 157   | 0             | 0            | 0.8837          | 0.8259                        | 0.3414   |
| synthetic | confident-wrong     | Uncertain but causally stable | 1     | 0             | 0            | 0.7937          | 0.7756                        | 0.44     |
| synthetic | mixed               | Dangerous shortcut reliance   | 63    | 17            | 0.2698       | 0.8692          | 0.4551                        | 1.068    |
| synthetic | mixed               | Generally fragile             | 107   | 81            | 0.757        | 0.659           | 0.3807                        | 1.214    |
| synthetic | mixed               | Reliable prediction           | 87    | 1             | 0.01149      | 0.8761          | 0.7852                        | 0.4212   |
| synthetic | mixed               | Uncertain but causally stable | 31    | 6             | 0.1935       | 0.736           | 0.7715                        | 0.448    |
| text      | confidence-solvable | Dangerous shortcut reliance   | 67    | 0             | 0            | 0.8818          | 0.5413                        | 0.8992   |
| text      | confidence-solvable | Generally fragile             | 58    | 55            | 0.9483       | 0.4289          | 0.4871                        | 1.005    |
| text      | confidence-solvable | Reliable prediction           | 108   | 0             | 0            | 0.889           | 0.7619                        | 0.4667   |
| text      | confidence-solvable | Uncertain but causally stable | 55    | 47            | 0.8545       | 0.4734          | 0.7585                        | 0.4735   |
| text      | confident-wrong     | Dangerous shortcut reliance   | 144   | 135           | 0.9375       | 0.9122          | 0.1881                        | 1.592    |
| text      | confident-wrong     | Reliable prediction           | 140   | 0             | 0            | 0.878           | 0.8149                        | 0.3628   |
| text      | confident-wrong     | Uncertain but causally stable | 4     | 0             | 0            | 0.7891          | 0.866                         | 0.2628   |
| text      | mixed               | Dangerous shortcut reliance   | 66    | 21            | 0.3182       | 0.8804          | 0.4654                        | 1.048    |
| text      | mixed               | Generally fragile             | 87    | 78            | 0.8966       | 0.6554          | 0.3497                        | 1.275    |
| text      | mixed               | Reliable prediction           | 87    | 3             | 0.03448      | 0.8787          | 0.776                         | 0.4392   |
| text      | mixed               | Uncertain but causally stable | 48    | 15            | 0.3125       | 0.7224          | 0.7378                        | 0.514    |
| vision    | confidence-solvable | Dangerous shortcut reliance   | 44    | 0             | 0            | 0.8812          | 0.5103                        | 0.96     |
| vision    | confidence-solvable | Generally fragile             | 72    | 67            | 0.9306       | 0.4418          | 0.4946                        | 0.9908   |
| vision    | confidence-solvable | Reliable prediction           | 134   | 0             | 0            | 0.8847          | 0.7675                        | 0.4558   |
| vision    | confidence-solvable | Uncertain but causally stable | 38    | 29            | 0.7632       | 0.5097          | 0.7323                        | 0.5248   |
| vision    | confident-wrong     | Dangerous shortcut reliance   | 141   | 129           | 0.9149       | 0.906           | 0.1995                        | 1.569    |
| vision    | confident-wrong     | Reliable prediction           | 146   | 0             | 0            | 0.8797          | 0.8103                        | 0.372    |
| vision    | confident-wrong     | Uncertain but causally stable | 1     | 0             | 0            | 0.788           | 0.88                          | 0.2352   |
| vision    | mixed               | Dangerous shortcut reliance   | 66    | 24            | 0.3636       | 0.876           | 0.429                         | 1.119    |
| vision    | mixed               | Generally fragile             | 106   | 87            | 0.8208       | 0.6575          | 0.3743                        | 1.227    |
| vision    | mixed               | Reliable prediction           | 75    | 0             | 0            | 0.8842          | 0.7679                        | 0.4549   |
| vision    | mixed               | Uncertain but causally stable | 41    | 3             | 0.07317      | 0.7435          | 0.7651                        | 0.4605   |

## Dangerous Quadrant

| task      | regime              | quadrant                    | count | failure_count | failure_rate | mean_confidence | mean_counterfactual_stability | mean_cic |
| --------- | ------------------- | --------------------------- | ----- | ------------- | ------------ | --------------- | ----------------------------- | -------- |
| synthetic | confidence-solvable | Dangerous shortcut reliance | 71    | 0             | 0            | 0.8838          | 0.5253                        | 0.9306   |
| synthetic | confident-wrong     | Dangerous shortcut reliance | 128   | 119           | 0.9297       | 0.9092          | 0.1709                        | 1.625    |
| synthetic | mixed               | Dangerous shortcut reliance | 63    | 17            | 0.2698       | 0.8692          | 0.4551                        | 1.068    |
| text      | confidence-solvable | Dangerous shortcut reliance | 67    | 0             | 0            | 0.8818          | 0.5413                        | 0.8992   |
| text      | confident-wrong     | Dangerous shortcut reliance | 144   | 135           | 0.9375       | 0.9122          | 0.1881                        | 1.592    |
| text      | mixed               | Dangerous shortcut reliance | 66    | 21            | 0.3182       | 0.8804          | 0.4654                        | 1.048    |
| vision    | confidence-solvable | Dangerous shortcut reliance | 44    | 0             | 0            | 0.8812          | 0.5103                        | 0.96     |
| vision    | confident-wrong     | Dangerous shortcut reliance | 141   | 129           | 0.9149       | 0.906           | 0.1995                        | 1.569    |
| vision    | mixed               | Dangerous shortcut reliance | 66    | 24            | 0.3636       | 0.876           | 0.429                         | 1.119    |
