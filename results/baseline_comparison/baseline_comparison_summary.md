# Baseline Comparison Summary

This reviewer-oriented comparison contextualizes CIC against simple uncertainty, generic instability, occlusion-style shortcut, and OOD-distance heuristics. It is not constructed to make CIC win everywhere.

## Best Non-CIC Baseline By Task

| task           | regime             | best_non_cic_baseline           | best_non_cic_auroc | cic_auroc | cic_advantage_over_best_non_cic | interpretation                         |
| -------------- | ------------------ | ------------------------------- | ------------------ | --------- | ------------------------------- | -------------------------------------- |
| clip_overlay   | pretrained-overlay | Confidence risk                 | 1                  | 1         | 0                               | rough tie                              |
| colored_digits | confident-wrong    | Random augmentation sensitivity | 0.9829             | 0.9512    | -0.03163                        | simpler baseline competitive or better |
| synthetic      | confident-wrong    | Occlusion shortcut heuristic    | 1                  | 1         | 0                               | rough tie                              |
| synthetic      | mixed              | Label-flip-only                 | 0.8753             | 0.8846    | 0.009368                        | rough tie                              |
| text           | mixed              | Label-flip-only                 | 0.8616             | 0.8683    | 0.006748                        | rough tie                              |
| vision         | mixed              | Label-flip-only                 | 0.9236             | 0.9286    | 0.004991                        | rough tie                              |

## All Metrics

| task           | regime             | method                          | failure_auroc | n_examples | n_failures | source                       | note                                                                                     |
| -------------- | ------------------ | ------------------------------- | ------------- | ---------- | ---------- | ---------------------------- | ---------------------------------------------------------------------------------------- |
| clip_overlay   | pretrained-overlay | Confidence risk                 | 1             |            |            | clip_overlay metrics         | aggregate metric file; heuristic baselines unavailable without per-example artifacts     |
| clip_overlay   | pretrained-overlay | Entropy                         | 1             |            |            | clip_overlay metrics         | aggregate metric file; heuristic baselines unavailable without per-example artifacts     |
| clip_overlay   | pretrained-overlay | Negative margin                 | 1             |            |            | clip_overlay metrics         | aggregate metric file; heuristic baselines unavailable without per-example artifacts     |
| clip_overlay   | pretrained-overlay | Label-flip-only                 | 1             |            |            | clip_overlay metrics         | aggregate metric file; heuristic baselines unavailable without per-example artifacts     |
| clip_overlay   | pretrained-overlay | CIC                             | 1             |            |            | clip_overlay metrics         | aggregate metric file; heuristic baselines unavailable without per-example artifacts     |
| colored_digits | confident-wrong    | Confidence risk                 | 0.1095        | 256        | 218        | colored_digits certificates  | colored_digits certificates                                                              |
| colored_digits | confident-wrong    | Entropy                         | 0.1012        | 256        | 218        | colored_digits certificates  | colored_digits certificates                                                              |
| colored_digits | confident-wrong    | Negative margin                 | 0.1024        | 256        | 218        | colored_digits certificates  | colored_digits certificates                                                              |
| colored_digits | confident-wrong    | Random augmentation sensitivity | 0.9829        | 256        | 218        | colored_digits certificates  | computed from generic certificate drift components when raw model inputs are unavailable |
| colored_digits | confident-wrong    | Occlusion shortcut heuristic    | 0.8382        | 256        | 218        | colored_digits certificates  | uses known shortcut flip/occlusion proxy where available                                 |
| colored_digits | confident-wrong    | Embedding/OOD distance          | 0.5903        | 256        | 218        | colored_digits certificates  | uses feature/logit-proxy centroid distance when learned embeddings are unavailable       |
| colored_digits | confident-wrong    | Label-flip-only                 | 0.8382        | 256        | 218        | colored_digits certificates  | colored_digits certificates                                                              |
| colored_digits | confident-wrong    | CIC                             | 0.9512        | 256        | 218        | colored_digits certificates  | colored_digits certificates                                                              |
| synthetic      | confident-wrong    | Confidence risk                 | 0.221         | 120        | 97         | confident_wrong certificates | confident_wrong certificates                                                             |
| synthetic      | confident-wrong    | Entropy                         | 0.221         | 120        | 97         | confident_wrong certificates | confident_wrong certificates                                                             |
| synthetic      | confident-wrong    | Negative margin                 | 0.221         | 120        | 97         | confident_wrong certificates | confident_wrong certificates                                                             |
| synthetic      | confident-wrong    | Random augmentation sensitivity | 0.9216        | 120        | 97         | confident_wrong certificates | computed from generic certificate drift components when raw model inputs are unavailable |
| synthetic      | confident-wrong    | Occlusion shortcut heuristic    | 1             | 120        | 97         | confident_wrong certificates | uses known shortcut flip/occlusion proxy where available                                 |
| synthetic      | confident-wrong    | Embedding/OOD distance          | 0.09368       | 120        | 97         | confident_wrong certificates | uses feature/logit-proxy centroid distance when learned embeddings are unavailable       |
| synthetic      | confident-wrong    | Label-flip-only                 | 1             | 120        | 97         | confident_wrong certificates | confident_wrong certificates                                                             |
| synthetic      | confident-wrong    | CIC                             | 1             | 120        | 97         | confident_wrong certificates | confident_wrong certificates                                                             |
| synthetic      | mixed              | Confidence risk                 | 0.84          |            |            | final validation aggregate   | aggregate mixed-regime metric                                                            |
| synthetic      | mixed              | Label-flip-only                 | 0.8753        |            |            | final validation aggregate   | aggregate mixed-regime metric                                                            |
| synthetic      | mixed              | CIC                             | 0.8846        |            |            | final validation aggregate   | aggregate mixed-regime metric                                                            |
| text           | confident-wrong    | Confidence risk                 |               | 100        | 100        | confident_wrong certificates | confident_wrong certificates                                                             |
| text           | confident-wrong    | Entropy                         |               | 100        | 100        | confident_wrong certificates | confident_wrong certificates                                                             |
| text           | confident-wrong    | Negative margin                 |               | 100        | 100        | confident_wrong certificates | confident_wrong certificates                                                             |
| text           | confident-wrong    | Random augmentation sensitivity |               | 100        | 100        | confident_wrong certificates | computed from generic certificate drift components when raw model inputs are unavailable |
| text           | confident-wrong    | Occlusion shortcut heuristic    |               | 100        | 100        | confident_wrong certificates | uses known shortcut flip/occlusion proxy where available                                 |
| text           | confident-wrong    | Embedding/OOD distance          |               | 100        | 100        | confident_wrong certificates | uses feature/logit-proxy centroid distance when learned embeddings are unavailable       |
| text           | confident-wrong    | Label-flip-only                 |               | 100        | 100        | confident_wrong certificates | confident_wrong certificates                                                             |
| text           | confident-wrong    | CIC                             |               | 100        | 100        | confident_wrong certificates | confident_wrong certificates                                                             |
| text           | mixed              | Confidence risk                 | 0.857         |            |            | final validation aggregate   | aggregate mixed-regime metric                                                            |
| text           | mixed              | Label-flip-only                 | 0.8616        |            |            | final validation aggregate   | aggregate mixed-regime metric                                                            |
| text           | mixed              | CIC                             | 0.8683        |            |            | final validation aggregate   | aggregate mixed-regime metric                                                            |
| vision         | confident-wrong    | Confidence risk                 |               | 80         | 80         | confident_wrong certificates | confident_wrong certificates                                                             |
| vision         | confident-wrong    | Entropy                         |               | 80         | 80         | confident_wrong certificates | confident_wrong certificates                                                             |
| vision         | confident-wrong    | Negative margin                 |               | 80         | 80         | confident_wrong certificates | confident_wrong certificates                                                             |
| vision         | confident-wrong    | Random augmentation sensitivity |               | 80         | 80         | confident_wrong certificates | computed from generic certificate drift components when raw model inputs are unavailable |
| vision         | confident-wrong    | Occlusion shortcut heuristic    |               | 80         | 80         | confident_wrong certificates | uses known shortcut flip/occlusion proxy where available                                 |
| vision         | confident-wrong    | Embedding/OOD distance          |               | 80         | 80         | confident_wrong certificates | uses feature/logit-proxy centroid distance when learned embeddings are unavailable       |
| vision         | confident-wrong    | Label-flip-only                 |               | 80         | 80         | confident_wrong certificates | confident_wrong certificates                                                             |
| vision         | confident-wrong    | CIC                             |               | 80         | 80         | confident_wrong certificates | confident_wrong certificates                                                             |
| vision         | mixed              | Confidence risk                 | 0.8332        |            |            | final validation aggregate   | aggregate mixed-regime metric                                                            |
| vision         | mixed              | Label-flip-only                 | 0.9236        |            |            | final validation aggregate   | aggregate mixed-regime metric                                                            |
| vision         | mixed              | CIC                             | 0.9286        |            |            | final validation aggregate   | aggregate mixed-regime metric                                                            |

## Interpretation

CIC is most useful when failures are high-confidence and specifically tied to unstable shortcut features. Simpler baselines can be competitive or better when failures are low-confidence, globally corrupted, or already separable by generic uncertainty/OOD signals.

In colored digits, random augmentation sensitivity outperformed CIC, with random augmentation AUROC 0.9829 versus CIC AUROC 0.9512. This shows that some shortcut failures can be detected by generic instability, especially when perturbations accidentally disturb the shortcut. However, generic augmentation is not targeted, not necessarily label-preserving, and does not explain which factor is unstable. CIC remains useful as a principled counterfactual stability framework rather than as a universal winner over every heuristic.

CIC is not claimed to dominate all baselines. The contribution is that it defines and operationalizes a second reliability axis.

The random-augmentation and OOD rows use available certificate/logit proxies when raw trained models or embeddings are not stored with an artifact. Those rows are included to make the comparison explicit, not to overstate the strength of the heuristic evaluation.
