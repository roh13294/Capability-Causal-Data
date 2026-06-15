# Random Augmentation Failure Stress Test

This benchmark is designed to test whether generic instability is sufficient when the shortcut is localized and factor-specific. It does not show CIC dominates random augmentation in all settings.

Task: real-text-style sentiment examples with a neutral metadata marker such as `[SOURCE=A]` or `[SITE=blue]`. The marker is spuriously correlated with the label during training and broken/flipped at shifted evaluation. The marker is not semantically part of the review sentiment label.

Random augmentation perturbs content words through deletion and small character noise while leaving the localized metadata marker mostly untouched.
CIC perturbs the factor-specific shortcut by removing/replacing or flipping the metadata marker while preserving the review content.

Random augmentation failure AUROC: 0.511.
CIC failure AUROC: 1.000.
Random augmentation failed relative to CIC: `True`.

## Metrics

| task                   | method                          | failure_auroc | n_examples | n_failures | n_correct |
| ---------------------- | ------------------------------- | ------------- | ---------- | ---------- | --------- |
| text_metadata_shortcut | confidence risk                 | 0.5114        | 180        | 113        | 67        |
| text_metadata_shortcut | entropy                         | 0.5114        | 180        | 113        | 67        |
| text_metadata_shortcut | margin                          | 0.5114        | 180        | 113        | 67        |
| text_metadata_shortcut | random augmentation sensitivity | 0.5114        | 180        | 113        | 67        |
| text_metadata_shortcut | OOD/embedding distance          | 0.4816        | 180        | 113        | 67        |
| text_metadata_shortcut | label-flip-only                 | 1             | 180        | 113        | 67        |
| text_metadata_shortcut | CIC                             | 1             | 180        | 113        | 67        |

## Limitations

- The benchmark uses a controlled metadata shortcut rather than an unknown natural shortcut.
- It supports the targeted claim that generic augmentation can miss localized shortcuts; it does not imply CIC wins universally.
- Human label validation is still needed if the metadata intervention is used in a new domain.
