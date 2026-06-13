# Holdout Tuning Summary

Settings were selected on validation criteria, then evaluated once on held-out seeds.

| task      | regime          | seed | shifted_accuracy | mean_failed_confidence | confidence_risk_auroc | cis_auroc | cic_minus_confidence_auroc |
| --------- | --------------- | ---- | ---------------- | ---------------------- | --------------------- | --------- | -------------------------- |
| synthetic | confident-wrong | 10   | 0.5833           | 0.9184                 | 0.1665                | 1         | 0.8335                     |
| synthetic | confident-wrong | 11   | 0.5833           | 0.9131                 | 0.2996                | 1         | 0.7004                     |
| synthetic | confident-wrong | 12   | 0.5833           | 0.9034                 | 0.3259                | 1         | 0.6741                     |
