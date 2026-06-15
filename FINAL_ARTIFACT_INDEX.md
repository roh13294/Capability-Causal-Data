# Final Artifact Index

This index lists the final 9/10 upgrade artifacts without expanding the project claim.

## New Evidence And Workflows

- Real text shortcut benchmark: `causal_reliability/experiments/run_real_text_shortcut_validation.py`, `configs/real_text_shortcut_validation.yaml`, `results/real_text_shortcut/`
- Human label-preservation packet and response analyzer: `causal_reliability/validation/export_label_preservation_packet.py`, `causal_reliability/validation/analyze_label_preservation_responses.py`, `configs/label_preservation_packet.yaml`, `scripts/analyze_human_validation.sh`, `results/label_preservation_packet/`
- Human label-preservation validation study and analysis (Fleiss' kappa + majority vote): `validation/human_label_preservation/analyze_annotations.py`, `validation/human_label_preservation/README.md`, `results/human_label_preservation/` (`human_validation_summary.md`, `human_validation_metrics.json`, `human_validation_flags.csv`)
- Random augmentation failure stress test: `causal_reliability/experiments/run_random_aug_failure_benchmark.py`, `configs/random_aug_failure_benchmark.yaml`, `results/random_aug_failure/`
- CIC-guided repair and abstention package: `causal_reliability/repair/cic_repair.py`, `causal_reliability/repair/repair_strategies.py`, `causal_reliability/repair/repair_metrics.py`, `causal_reliability/repair/abstention.py`
- CLIP overlay repair experiment: `causal_reliability/experiments/run_clip_overlay_repair.py`, `configs/clip_overlay_repair.yaml`, `results/clip_overlay_repair/`
- Non-oracle CLIP shortcut localization and repair experiment: `causal_reliability/discovery/region_proposals.py`, `causal_reliability/discovery/cic_region_scoring.py`, `causal_reliability/discovery/nonoracle_clip_discovery.py`, `causal_reliability/experiments/run_nonoracle_clip_repair.py`, `configs/nonoracle_clip_repair.yaml`, `scripts/run_nonoracle_clip_repair.sh`, `results/nonoracle_clip_repair/`
- Real text shortcut repair experiment: `causal_reliability/experiments/run_real_text_repair.py`, `configs/real_text_repair.yaml`, `results/real_text_repair/`
- Random augmentation failure repair/abstention experiment: `causal_reliability/experiments/run_random_aug_failure_repair.py`, `configs/random_aug_failure_repair.yaml`, `results/random_aug_failure_repair/`, including `random_aug_failure_selective_risk.csv`, `.png`, and `.pdf`
- Cross-shortcut generalization test (frozen text-selected CIC policy applied to a non-text colored-symbol watermark shortcut): `causal_reliability/experiments/run_cross_shortcut_generalization.py`, `configs/cross_shortcut_generalization.yaml`, `tests/test_cross_shortcut_generalization.py`, `results/cross_shortcut_generalization/`
- Optional safety-critical-inspired traffic-sign shortcut validation: `causal_reliability/experiments/run_traffic_sign_shortcut_validation.py`, `configs/traffic_sign_shortcut_validation.yaml`, `results/traffic_sign_shortcut/`
- Practitioner CIC audit API and demo: `causal_reliability/api/`, `causal_reliability/audit/run_cic_audit.py`, `configs/example_cic_audit.yaml`, `docs/cic_audit_demo.md`
- Confidence-only insufficiency lemma: `docs/theoretical_intuition.md`, `docs/formal_separation.md`
- Finite-candidate CIC recovery theory and embedding-additivity validation: `docs/theory.md`, `causal_reliability/experiments/run_embedding_additivity_validation.py`, `configs/embedding_additivity_validation.yaml`, `tests/test_embedding_additivity_and_theory.py`, `results/embedding_additivity/`
- Per-input class-balance validation (final theory gate): `docs/theory.md` (Section 9 + corollary), `causal_reliability/experiments/run_per_input_class_balance_validation.py`, `configs/per_input_class_balance_validation.yaml`, `tests/test_per_input_class_balance.py`, `results/per_input_class_balance/`
- Final category-defense summary: generated in `results/final_report/final_report.md`

## Theory And Mechanism Validation

The finite-candidate CIC recovery theorem (`docs/theory.md`) assumes an additive logit decomposition `logit_y(g(C,S,N)) = phi_y(C,N) + psi_y(S) + xi_y(C,S,N)`. Because CLIP logits are inner products `logit_y(X) = <u(X), v_y>`, this holds iff the embedding shift caused by a shortcut is approximately input-independent. The embedding-additivity validation experiment tests this directly on real pretrained OpenCLIP for the text-overlay and colored-symbol watermark shortcuts.

The theorem remains a conditional explanation, but embedding-additivity validation did not support applying the *global* form directly to the current OpenCLIP text benchmark (`embedding_additivity_supported_for_text = false`): the shortcut embedding shift clusters by shortcut value above the shuffled baseline (within-shortcut cosine ~0.76 vs shuffled ~0.63) and oracle neutralization repairs the prediction in 100% of cases, but the per-image delta clusters more tightly by object class (within-object cosine ~0.86) than by shortcut value, so the shortcut direction is not input-independent. The watermark transfer failure is consistent with a weak or flat shortcut channel, not a clean repairable shortcut (`embedding_additivity_supported_for_watermark = false`, `watermark_shortcut_channel_weak = true`).

Global input-independent additivity is stronger than the recovery theorem requires. The **final theory gate** — the per-input class-balance validation (`run_per_input_class_balance_validation.py`, `results/per_input_class_balance/`) — tests the weaker premise: after neutralization the repaired logits differ from the clean/causal logits by an approximately class-independent residual for each individual image (residual-to-clean `rho_y(x) = ell_y(T(x)) - ell_y(x_clean)`, with class-balance condition `max_y |rho_y(x) - mean_y rho(x)| <= epsilon_B`), with recovery guaranteed when the clean causal margin satisfies `m_clean(x) > 2*epsilon_B`. The residual is defined relative to the clean logits, not the misleading input logits, because a shift that is class-balanced relative to the misleading logits would preserve the misleading argmax rather than recover the clean causal argmax. On real pretrained OpenCLIP, oracle and CIC neutralization are substantially more class-balanced than matched random text-region neutralization, and the margin condition tracks repair success; `per_input_class_balance_supported_for_text` and `clip_theory_support_status` record the authoritative outcome. A positive structural finding accompanies it: the typographic shortcut effect is **object-entangled** — it contains a real shortcut component but its direction varies with the underlying object — which explains why generic global debiasing is unlikely to suffice while targeted per-input region scoring can still repair failures. This is a conditional, finite-candidate mechanism account; it does not claim open-world shortcut discovery, exact localization, or general robustness.

## Final Headline Result

On a held-out hard multi-decoy natural benchmark using real pretrained OpenCLIP, misleading text reduced accuracy to **0.250** (25.0%). Non-oracle CIC region scoring repaired accuracy to **0.750** (75.0%), compared with **0.331** (33.1%) for matched random text-region repair, while preserving no-overlay accuracy and keeping the clean-safe accuracy drop to **0.010** (1.0%).

The authoritative current result is generated from `results/final_report/final_key_numbers.json` and is **0.250 → 0.750 with matched random 0.331** (`hard_multidecoy_headline_primary_metric = "misleading accuracy 0.250 to 0.750"`).

### Historical stale exploratory result — not current headline

An earlier exploratory headline reported `0.219 → 0.875` with matched random `0.288` (i.e. 21.9% → 87.5% / 28.8%). **These values were produced under a prior prediction path and must not be used as the final result.** They are retained here only as historical context and are **not** the authoritative or current headline; the current headline above (`0.250 → 0.750`, matched random `0.331`, clean-safe drop `0.010`) supersedes them. This discrepancy is reconciled (not unresolved): the stale figure is confined to this clearly marked historical-stale subsection, and `paper/main.tex` Appendix A records the same reconciliation.

This does not solve open-world shortcut discovery. The method searches a finite candidate class of text-region proposals. Localization is strongest at coarse IoU >= 0.3 and weak at strict IoU >= 0.5, so the result should be interpreted as coarse causal-region localization and repair, not exact bounding-box recovery.

## Scale and Multi-Model Replication Audit (Supporting)

Supporting evidence only; this **does not** replace the frozen primary headline above (ViT-B-32 / laion2b_s34b_b79k at n=32: misleading 0.250 → CIC top-1 0.750 vs. 0.331 matched random, clean-safe drop 0.010), which is left unchanged. A scale-and-multi-model replication audit re-ran the hard multi-decoy text-overlay benchmark at n_per_condition = 128 across four real pretrained OpenCLIP backbone/checkpoint pairs. All 4/4 model/checkpoint pairs loaded (0 skipped, no fake backend) and all four were `repair_eligible`; the test suite passes (186 tests). All four models were evaluated on the **same** larger resampled benchmark instance (one shared benchmark hash) for a fair cross-model comparison, and this benchmark hash differs from the n=32 headline benchmark, so these numbers are a separate larger-n replication and not a cell-for-cell restatement of the headline.

| Model | Pretrained tag | Original misleading | CIC top-1 | Matched random | CIC − random gap | Clean-safe drop | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-B-32 | laion2b_s34b_b79k | 0.289 | 0.742 | 0.305 | 0.437 | 0.036 | repair_eligible |
| ViT-B-32 | openai | 0.062 | 0.688 | 0.111 | 0.576 | 0.003 | repair_eligible |
| ViT-B-16 | laion2b_s34b_b88k | 0.078 | 0.938 | 0.159 | 0.778 | 0.000 | repair_eligible |
| RN50 | openai | 0.000 | 0.758 | 0.072 | 0.686 | 0.000 | repair_eligible |

The main text-overlay result is stable at larger n, and the text-overlay CIC effect replicates across multiple pretrained OpenCLIP backbones/checkpoints. This audit does **not** imply open-world shortcut discovery, general robustness, cross-shortcut generalization, or exact localization: the method still searches a finite candidate class of text-region proposals on a controlled synthetic text-overlay benchmark. Artifacts:

- `results/hard_multidecoy_scale_model_audit/scale_model_summary.md`
- `results/hard_multidecoy_scale_model_audit/scale_model_key_numbers.json`
- `results/hard_multidecoy_scale_model_audit/scale_model_metrics.csv`
- `results/hard_multidecoy_scale_model_audit/scale_model_plot.png`
- `results/hard_multidecoy_scale_model_audit/model_availability.csv`

## Second Shortcut Family — Non-Text Semantic-Decoy Icon (Supporting)

Positive supporting evidence for a second controlled finite-candidate shortcut family. This **does not** replace the frozen text-overlay primary headline above and does **not** imply open-world shortcut discovery, general robustness, cross-shortcut transfer, universal shortcut repair, or exact localization. The shortcut family is `non_text_semantic_decoy_icon`: a central colored causal icon plus a larger, spatially separated competing-class corner icon, **no written words anywhere**. On real pretrained OpenCLIP ViT-B-32 / laion2b_s34b_b79k (fake backend blocked), the n=64 pilot passed 8/8 gates and the n=128 scale run also passed 8/8 gates (no failed gates). Headline-supporting numbers (n=128): clean accuracy 1.000, misleading original accuracy 0.297, oracle repair 1.000, CIC top-1 0.711, CIC top-3 0.359, CIC clean-safe 0.766, matched random 0.258, CIC − random gap +0.453, clean-safe clean drop 0.008; eligible = true, `semantic_decoy_include_in_headline = false`.

CIC also succeeds on this second controlled finite-candidate shortcut family under controlled oracle-intervention conditions, supporting the method beyond text overlays. The earlier flat visual-decoy pilot was not failure-rich enough (misleading accuracy ~0.58 exceeded the ≤ 0.40 failure gate) and is retained as boundary evidence; the semantic-decoy icon benchmark was the final pre-specified second-family attempt and passed all gates. Artifacts:

- `causal_reliability/data/clip_semantic_decoy_shortcuts.py`, `causal_reliability/experiments/run_semantic_decoy_clip_pilot.py`, `configs/semantic_decoy_pilot.yaml`, `configs/semantic_decoy_scale.yaml`
- `results/semantic_decoy_pilot/` (n=64: `semantic_decoy_pilot_gates.json`, `semantic_decoy_pilot_metrics.csv`, `semantic_decoy_pilot_report.md`, `semantic_decoy_plot.png`)
- `results/semantic_decoy_scale_n128/` (n=128: same artifact set)
- Boundary evidence (flat visual decoy, one failed gate): `causal_reliability/data/clip_visual_decoy_shortcuts.py`, `causal_reliability/experiments/run_visual_decoy_clip_pilot.py`, `configs/visual_decoy_pilot.yaml`, `tests/test_visual_decoy_pilot.py`, `results/visual_decoy_pilot/`

## Human Label-Preservation Validation

Three annotators evaluated 100 original/repaired image pairs (300 total annotations). Majority vote found that the object label was preserved in **96 of 100** pairs and that the repaired image remained recognizable in **97 of 100** pairs. Inter-annotator agreement was high: Fleiss' kappa was 0.973 (before label), 0.974 (after label), 0.920 (did label change), and 1.000 (after recognizable). The four preservation failures were retained and flagged rather than removed: one apparent shape change, one blurry/unrecognizable repair, one corrupted/glitched repair, and one blank/missing-shape repair. Metrics are written to `results/human_label_preservation/` by `validation/human_label_preservation/analyze_annotations.py`.

The analyzer computes majority-vote rates, percent agreement, and Fleiss' kappa directly from the raw per-annotator responses in `validation/human_label_preservation/completed_annotations/`. Before/after label accuracy against the true label is reported only when `validation/human_label_preservation/packet/metadata_hidden.csv` is present, and is otherwise reported as n/a (no true labels are fabricated). This is a label-preservation methodology check, not a headline result.

## WILDS Waterbirds Metadata-Only Diagnostic

WILDS Waterbirds was parsed as a real spurious-background diagnostic (11,788 examples detected and parsed). A metadata-only OpenCLIP evaluation showed the expected background sensitivity: overall accuracy 56.0%, land-background accuracy 73.1% versus water-background accuracy 35.9%, and landbird accuracy dropping from 74.4% on land backgrounds to 21.6% on water backgrounds (`results/waterbirds_cic_pilot/wilds_metadata_diagnostic.csv`). However, WILDS Waterbirds does not ship oracle-repairable bird/background masks or bounding boxes, so CIC repair and failure-conditioned oracle repair were **not** run. This diagnostic motivates a future regenerated CUB+Places Waterbirds-style benchmark with known masks (see `docs/regenerated_waterbirds_pilot.md`), but it is **not** a positive CIC repair result and is never listed as headline evidence.

## Final External Feedback Pass Script

- `scripts/run_final_external_feedback_pass.sh` runs the random augmentation failure benchmark, traffic-sign shortcut validation, label-preservation packet export, and final report rebuild.

## Claim Boundary

Confidence measures uncertainty. Counterfactual stability measures shortcut dependence. CIC complements confidence by detecting high-confidence shortcut failures when label-preserving shortcut interventions are available, hypothesized, or discovered from a finite candidate set.

Repair extension claim: CIC-guided repair is a proof-of-concept. The strongest current result is not universal automatic correction, but targeted detection and selective abstention: CIC can identify high-confidence shortcut failures that random augmentation fails to detect.

What each repair experiment establishes:

- Random augmentation failure repair/abstention proves CIC can flag localized shortcut failures where generic perturbation is near chance. It does not prove automatic repair is always successful.
- Real text repair proves shortcut-neutralized prediction can correct dangerous-quadrant text examples in a controlled setting. It does not prove broad text-model repair.
- CLIP overlay validation establishes real pretrained shortcut vulnerability when pretrained CLIP loads.
- CLIP overlay repair with the known bbox is an oracle upper-bound causal confirmation. It should not be treated as automatic shortcut discovery.
- Single-overlay non-oracle CLIP repair is promising, but matched/random patch baselines can be competitive.
- First multi-decoy CLIP repair is not a true shortcut-failure benchmark because original misleading accuracy was high.
- Hard multi-decoy CLIP repair is the main headline result. It proves finite-candidate non-oracle localization and repair only because real pretrained OpenCLIP loaded, held-out scoring did not access overlay bbox or true label, hard misleading accuracy was low, repair beat matched random text-region repair, clean/no-overlay accuracy was preserved, and `headline_eligible` is true.
- Hard multi-decoy audit artifacts now distinguish `fixed_benchmark_determinism_check` from true benchmark resampling. The completed `benchmark_resampling_audit.csv` is a `lite_mode` small-n benchmark-resampling check, not full stability evidence; `repair_vs_localization_crosstab.csv` reports repair-vs-coarse-localization diagnostics.
- Scale and multi-model replication audit (`results/hard_multidecoy_scale_model_audit/`) is supporting evidence only: at n_per_condition = 128 the text-overlay CIC effect is stable at larger n and replicates across four real pretrained OpenCLIP backbone/checkpoint pairs (all loaded, all `repair_eligible`) on one shared resampled benchmark instance whose hash differs from the n=32 headline. It does **not** replace the frozen primary headline and does **not** imply open-world discovery, general robustness, cross-shortcut generalization, or exact localization.
- Cross-shortcut generalization is a finite-candidate transfer test: the text-overlay-selected CIC policy is frozen and applied, with no retuning, to a non-text colored-symbol watermark shortcut. It is a supporting S+ generalization result only when `cross_shortcut_headline_eligible` is true (the frozen policy beats matched random region repair and preserves clean accuracy); otherwise it is reported honestly as a transfer attempt that did not support cross-shortcut generalization. It is not open-world shortcut discovery and not a claim of general robustness.
- Second shortcut family (non-text semantic-decoy icon) is positive supporting evidence: the same finite-candidate CIC region method is re-run from scratch (not a frozen transfer) on an independent non-text shortcut family, and both the n=64 pilot and the n=128 scale run pass all 8 strict gates (`semantic_decoy_eligible = true`, `semantic_decoy_include_in_headline = false`). It supports the method beyond text overlays under controlled oracle-intervention conditions. It does **not** imply open-world shortcut discovery, general robustness, cross-shortcut transfer, universal shortcut repair, or exact localization. The earlier flat visual-decoy pilot is retained as boundary evidence (one failed gate: the shortcut was not failure-rich enough); the semantic-decoy icon benchmark was the final pre-specified second-family attempt and passed all gates.

Related work positioning: `docs/related_work.md`.
