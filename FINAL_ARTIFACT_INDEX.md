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
- Spatial-resolution and causal-intervention audit (addresses low exact-IoU criticism without claiming exact localization): `causal_reliability/experiments/run_spatial_resolution_audit.py`, `configs/spatial_resolution_audit.yaml`, `tests/test_spatial_resolution_audit.py`, `results/spatial_resolution_audit/`
- Global theory attempt — shortcut-type-agnostic proposal-complete repair theorem plus no-free-lunch impossibility: `docs/global_cic_theory_attempt.md`, machine-checkable encoding `causal_reliability/theory/proposal_completeness.py`, tests `tests/test_proposal_completeness_theory.py`; concise integrations in `docs/theory.md` (Section 10) and `paper/main.tex` (Section "Shortcut-type-agnostic generalization and a no-free-lunch boundary")
- Final category-defense summary: generated in `results/final_report/final_report.md`

## Theory And Mechanism Validation

The finite-candidate CIC recovery theorem (`docs/theory.md`) assumes an additive logit decomposition `logit_y(g(C,S,N)) = phi_y(C,N) + psi_y(S) + xi_y(C,S,N)`. Because CLIP logits are inner products `logit_y(X) = <u(X), v_y>`, this holds iff the embedding shift caused by a shortcut is approximately input-independent. The embedding-additivity validation experiment tests this directly on real pretrained OpenCLIP for the text-overlay and colored-symbol watermark shortcuts.

The theorem remains a conditional explanation, but embedding-additivity validation did not support applying the *global* form directly to the current OpenCLIP text benchmark (`embedding_additivity_supported_for_text = false`): the shortcut embedding shift clusters by shortcut value above the shuffled baseline (within-shortcut cosine ~0.76 vs shuffled ~0.63) and oracle neutralization repairs the prediction in 100% of cases, but the per-image delta clusters more tightly by object class (within-object cosine ~0.86) than by shortcut value, so the shortcut direction is not input-independent. The watermark transfer failure is consistent with a weak or flat shortcut channel, not a clean repairable shortcut (`embedding_additivity_supported_for_watermark = false`, `watermark_shortcut_channel_weak = true`).

Global input-independent additivity is stronger than the recovery theorem requires. The **final theory gate** — the per-input class-balance validation (`run_per_input_class_balance_validation.py`, `results/per_input_class_balance/`) — tests the weaker premise: after neutralization the repaired logits differ from the clean/causal logits by an approximately class-independent residual for each individual image (residual-to-clean `rho_y(x) = ell_y(T(x)) - ell_y(x_clean)`, with class-balance condition `max_y |rho_y(x) - mean_y rho(x)| <= epsilon_B`), with recovery guaranteed when the clean causal margin satisfies `m_clean(x) > 2*epsilon_B`. The residual is defined relative to the clean logits, not the misleading input logits, because a shift that is class-balanced relative to the misleading logits would preserve the misleading argmax rather than recover the clean causal argmax. On real pretrained OpenCLIP, oracle and CIC neutralization are substantially more class-balanced than matched random text-region neutralization, and the margin condition tracks repair success; `per_input_class_balance_supported_for_text` and `clip_theory_support_status` record the authoritative outcome. A positive structural finding accompanies it: the typographic shortcut effect is **object-entangled** — it contains a real shortcut component but its direction varies with the underlying object — which explains why generic global debiasing is unlikely to suffice while targeted per-input region scoring can still repair failures. This is a conditional, finite-candidate mechanism account; it does not claim open-world shortcut discovery, exact localization, or general robustness.

A **global theory attempt** (`docs/global_cic_theory_attempt.md`) upgrades the *generality* of the positive statement without upgrading the *strength* of the claim. It defines **proposal completeness** — the candidate family `A(x)` contains a good intervention `a*` that preserves the causal content (`ε_C`), neutralizes the shortcut (`ε_S`), introduces no larger shortcut, and leaves a class-balanced residual-to-clean (`ε_B`) — and proves a **proposal-complete CIC repair theorem**: under completeness, `m_clean(x) > 2ε_B`, a CIC scoring margin `γ`, and noise below `γ/2`, top-1 CIC repair restores the clean causal prediction *independently of whether the shortcut is text, icon, watermark, background, or texture*. A top-`k` consensus corollary and an observable, validation-calibrated success gate accompany it. The complement is a **no-free-lunch impossibility theorem**: no finite-query black-box counterfactual method can guarantee discovery or repair of arbitrary shortcuts without proposal completeness, because two worlds can produce identical observations on every queried intervention yet require different repairs. The positive guarantee is therefore **shortcut-type-agnostic under proposal completeness**, the negative result is a **no-free-lunch impossibility for assumption-free shortcut discovery**, and CIC is a **global theorem over intervention families, not over arbitrary unobserved shortcuts** — explicitly not universal repair, not open-world discovery, not assumption-free. The watermark cross-shortcut negative result is a measured instance of the impossibility regime (weak/flat channel ⇒ completeness fails). The theorem inequalities are machine-checked in `causal_reliability/theory/proposal_completeness.py` (`tests/test_proposal_completeness_theory.py`), concise versions appear in `docs/theory.md` Section 10 and `paper/main.tex` Section "Shortcut-type-agnostic generalization and a no-free-lunch boundary".

## Final Headline Result

On a held-out hard multi-decoy natural benchmark using real pretrained OpenCLIP, misleading text reduced accuracy to **0.250** (25.0%). Non-oracle CIC region scoring repaired accuracy to **0.750** (75.0%), compared with **0.331** (33.1%) for matched random text-region repair, while preserving no-overlay accuracy and keeping the clean-safe accuracy drop to **0.010** (1.0%).

The authoritative current result is generated from `results/final_report/final_key_numbers.json` and is **0.250 → 0.750 with matched random 0.331** (`hard_multidecoy_headline_primary_metric = "misleading accuracy 0.250 to 0.750"`).

### Historical stale exploratory result — not current headline

An earlier exploratory headline reported `0.219 → 0.875` with matched random `0.288` (i.e. 21.9% → 87.5% / 28.8%). **These values were produced under a prior prediction path and must not be used as the final result.** They are retained here only as historical context and are **not** the authoritative or current headline; the current headline above (`0.250 → 0.750`, matched random `0.331`, clean-safe drop `0.010`) supersedes them. This discrepancy is reconciled (not unresolved): the stale figure is confined to this clearly marked historical-stale subsection, and `paper/main.tex` Appendix A records the same reconciliation.

This does not solve open-world shortcut discovery. The method searches a finite candidate class of text-region proposals. Localization is strongest at coarse IoU >= 0.3 and weak at strict IoU >= 0.5, so the result should be interpreted as coarse causal-region localization and repair, not exact bounding-box recovery.

## Scale and Multi-Model Replication Audit (Supporting)

Supporting evidence only; this **does not** replace the frozen primary headline above (ViT-B-32 / laion2b_s34b_b79k at n=32: misleading 0.250 → CIC top-1 0.750 vs. 0.331 matched random, clean-safe drop 0.010), which is left unchanged. A scale-and-multi-model replication audit re-ran the hard multi-decoy text-overlay benchmark at n_per_condition = 128 across four real pretrained OpenCLIP backbone/checkpoint pairs. All 4/4 model/checkpoint pairs loaded (0 skipped, no fake backend) and all four were `repair_eligible`; the test suite passes (382 tests). All four models were evaluated on the **same** larger resampled benchmark instance (one shared benchmark hash) for a fair cross-model comparison, and this benchmark hash differs from the n=32 headline benchmark, so these numbers are a separate larger-n replication and not a cell-for-cell restatement of the headline.

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

## Spatial-Resolution and Causal-Intervention Audit (Supporting Diagnostic)

Supporting diagnostic only; this reads existing benchmark artifacts and does **not** re-run any model or change a headline metric. It addresses the low exact-IoU criticism directly by separating *exact box precision* (IoU with the oracle shortcut box) from *causal-intervention usefulness* (shortcut coverage, intervention bluntness, causal-content preservation, and repair-by-IoU). Pooled over **210** shortcut examples from the hard multi-decoy (n=32), failure-conditioned (n=50), and semantic-decoy (n=128) benchmarks. It makes **no** claim of exact localization, segmentation quality, spatial grounding solved, open-world discovery, or general robustness: **CIC is a coarse causal-intervention method, not an exact localization or segmentation method**, and exact localization remains a limitation.

Artifact-to-number map (all values from `results/spatial_resolution_audit/spatial_resolution_key_numbers.json` unless noted):

| Number | Value | Artifact source |
| --- | --- | --- |
| Pooled shortcut examples | 210 | `pooled.n_examples` |
| Median IoU vs oracle shortcut box | 0.39 | `pooled.median_iou` |
| Hit@IoU >= 0.3 | 0.73 | `pooled.hit_at_iou_0_3` |
| Hit@IoU >= 0.5 | 0.43 | `pooled.hit_at_iou_0_5` |
| Shortcut coverage >= 0.5 | 0.77 | `pooled.shortcut_coverage_ge_0_5_rate` |
| Shortcut coverage >= 0.8 | 0.71 | `pooled.shortcut_coverage_ge_0_8_rate` |
| Any shortcut intersection | 0.90 | `pooled.intersects_shortcut_rate` |
| Median shortcut coverage | 0.875 | `pooled.shortcut_coverage_median` |
| Median selected area (fraction of image) | 0.21 | `pooled.area_frac_image_median` |
| Median selected region / oracle box area | ~2.1× | `pooled.area_frac_oracle_median` |
| Median object IoU (where object boxes exist) | 0.20 | `pooled.object_iou_median` |
| CIC top-1 repair (pooled) | 0.78 | `pooled.repair_top1_accuracy` |
| CIC clean-safe repair (pooled) | 0.80 | `pooled.repair_clean_safe_accuracy` |
| Top-1 repair, IoU < 0.1 | 0.26 | `spatial_resolution_by_bucket.csv` (ALL, `<0.1`) |
| Top-1 repair, 0.1 <= IoU < 0.3 | 0.25 | `spatial_resolution_by_bucket.csv` (ALL, `0.1-0.3`) |
| Top-1 repair, 0.3 <= IoU < 0.5 | 0.92 | `spatial_resolution_by_bucket.csv` (ALL, `0.3-0.5`) |
| Top-1 repair, IoU >= 0.5 | 1.00 | `spatial_resolution_by_bucket.csv` (ALL, `>=0.5`) |
| Exact localization remains a limitation | true | `exact_localization_remains_a_limitation` |
| Refinement improved spatial precision | false | `refinement.refinement_improved_spatial_precision` |

A non-oracle refinement diagnostic (geometric shrink/split/shift variants re-scored using only pixels, candidate boxes, and model probabilities — never the oracle box, true label, or repair correctness) did **not** improve median IoU or the IoU >= 0.5 rate and is reported honestly as such (test-enforced no-oracle-leakage in `tests/test_spatial_resolution_audit.py`). The semantic-decoy benchmark ships no oracle shortcut box or object box on disk, so its coverage and object-overlap metrics are reported as n/a rather than fabricated. Artifacts:

- `results/spatial_resolution_audit/spatial_resolution_summary.md`
- `results/spatial_resolution_audit/spatial_resolution_key_numbers.json`
- `results/spatial_resolution_audit/spatial_resolution_metrics.csv`
- `results/spatial_resolution_audit/spatial_resolution_by_bucket.csv`
- `results/spatial_resolution_audit/spatial_resolution_plot.png`

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

## COCO-Text Natural-Image Validation (Supporting, Directional)

Dataset-backed natural-image validation on real photographs with COCO-Text annotations. This **does not** replace the frozen primary headline, **does not** pass a strict support gate, and is **not** open-world shortcut discovery or a claim that CIC localizes scene text. On real pretrained OpenCLIP ViT-B-32 / laion2b_s34b_b79k (fake backend false), the shortcut-agnostic proposal-based CIC method was run over all 500 COCO-Text metadata rows, yielding 57 directionally verified text-driven failures and 39 strict oracle-repairable failures.

CIC provides **dataset-backed directional repair evidence rather than localization support**. On the strict oracle-repairable subset, proposal-based CIC improves alias-aware repair from 0.205 for matched random proposals to 0.538 (gap +0.333), while decreasing the text-distractor probability in 0.95 of cases; the directional subset shows the same pattern (0.439 vs. 0.175, gap +0.263, distractor decrease 0.96 vs. 0.65). Over all 500 rows the CIC−random gap is −0.040, expected because most examples are already correct. OCR inclusion helps only marginally (strict 0.538 excluding OCR vs. 0.564 including OCR; `ocr_inclusion_materially_helps = false`). The strict and directional support gates remain `false` because selected proposals overlap annotated COCO-Text boxes in only 0.128 (strict) / 0.140 (directional), below the 0.60 gate (`coco_text_strict_support = false`, `coco_text_directional_support = false`). A follow-up localization diagnostic finds the shortfall is **ranking failure more than proposal failure** with a secondary coverage ceiling: text-overlapping proposals exist in 56.4% of strict cases (IoU≥0.1) but are ranked below more causally effective non-text regions (median rank 4; ranking gap 0.436 vs. coverage gap 0.036, ranking ≈92% of the shortfall), and forcing the best text-overlapping proposal repairs worse (0.163) than the selected CIC proposal (0.439), as does forcing the best object-overlapping proposal (0.231). Area-normalized scoring and top-k union do not fix localization (top-k raises overlap but lowers repair), and text-dilated proposal families improve overlap slightly without clearing the gate or beating selected CIC repair. COCO-Text supports CIC as a causal repair signal on natural images while exposing a boundary: repair does not imply text-box localization. Allowed framing: "dataset-backed directional repair evidence", "random-beating repair on verified natural-image text failures", "localization limitation". Not allowed: "open-world shortcut discovery", "CIC localizes scene text", "COCO-Text strict support gate passed", "universal natural-image robustness". Integrated in `paper/main.tex` Section "COCO-Text natural-image validation", the Limitations section, and Appendix "COCO-Text Natural-Image Results" (Tables `tab:cocotext` and `tab:cocoloc`). Artifacts:

- Triage / verified-failure isolation: `causal_reliability/data/coco_text_cic_builder.py`, `causal_reliability/experiments/run_coco_text_cic_triage.py`, `configs/coco_text_cic_triage.yaml`, `results/coco_text_cic_triage/`
- Full proposal-CIC evaluation: `causal_reliability/discovery/open_region_proposals.py`, `causal_reliability/experiments/run_coco_text_cic_full.py`, `configs/coco_text_cic_full.yaml`, `results/coco_text_cic_full/`
- Localization diagnostic: `causal_reliability/analysis/coco_text_localization_diagnostic.py`, `causal_reliability/experiments/run_coco_text_cic_localization_diagnostic.py`, `configs/coco_text_cic_localization_diagnostic.yaml`, `results/coco_text_cic_localization_diagnostic/`

## Predictive CIC Abstention Gate (Supporting, Label-Free)

A **label-free predictive abstention layer** on top of CIC. From inference-time observable features only (repaired confidence/margin, top-k repair agreement, entropy drop, stability gain, prediction-changed flag — no true label, target label, correctness, oracle outcome, or ground-truth box), a simple logistic gate predicts whether to **accept or abstain** from a CIC repair **before the true label is seen**. It is calibrated and evaluated across **2,635 examples from 8 benchmarks** (2,048 controlled, 587 natural; repair-success base rate 0.785, label `label_repair_success`).

Pooled leave-one-benchmark-out (LOBO) **AUROC = 0.789** (AUPRC 0.93). Under abstention, accepted repairs are high-precision: top-10%/25%/50% accepted precision = **0.99 / 0.97 / 0.93**. Best features are repaired confidence/margin, top-k repair agreement, and prediction_changed; the best simple rule is *accept the repair iff CIC did not change the prediction*. The conservative support flag `predictive_gate_supported = true` while `is_universal_theorem = false`.

Allowed framing: "label-free predictive abstention layer", "predicts high-trust CIC repairs in pooled/controlled settings", "high-precision accepted repairs under abstention", "conditional predictive certificate", "does not require true labels at inference time". The gate is strongest in pooled/controlled settings and **does not transfer reliably to the hardest COCO-Text subsets**: held-out AUROC 0.699 (all), 0.347 (strict), 0.296 (directional), and pure controlled→natural transfer is near chance (train_controlled_test_coco ≈ 0.51, train_controlled_test_natural ≈ 0.50). **The predictive gate is an abstention layer, not a universal correctness oracle; controlled-to-natural transfer remains weak on hard COCO-Text failures.** Not allowed: "universal theorem", "works on all natural images", "reliably predicts hard COCO-Text repair success", "CIC always knows when it is right", "open-world shortcut discovery".

| Quantity | Value |
| --- | --- |
| Examples | 2,635 |
| Benchmarks | 8 |
| Pooled LOBO AUROC | 0.789 |
| Pooled AUPRC | 0.93 |
| Accepted precision, top-10% | 0.99 |
| Accepted precision, top-25% | 0.97 |
| Accepted precision, top-50% | 0.93 |
| COCO-Text AUROC (all) | 0.699 |
| COCO-Text AUROC (strict) | 0.347 |
| COCO-Text AUROC (directional) | 0.296 |
| Controlled→natural transfer AUROC | ≈ 0.50 |
| `predictive_gate_supported` | true |
| `is_universal_theorem` | false |

This gate is the calibrated, observable instance of the **predictive CIC certificate** (Proposition P): if the repaired prediction margin exceeds an empirically calibrated **residual-instability bound** `eps_hat` (`m_rep(x) > 2*eps_hat`), the repaired prediction is stable under the calibrated perturbation class — a **repaired-margin certificate** giving a **label-free accept/abstain rule**. It substitutes the calibrated bound for the residual budget `eps_B` of the per-input class-balance recovery lemma and reads the margin off the observed repaired logits, so it requires no true labels at inference time. It writes only under `results/predictive_cic_gate/` and never touches `results/final_report/`, the headline metrics, or existing support gates. Artifacts:

- Analysis / gate: `causal_reliability/analysis/predictive_cic_gate.py`
- Experiment runner: `causal_reliability/experiments/run_predictive_cic_gate.py`
- Config: `configs/predictive_cic_gate.yaml`
- Theory (machine-checked certificate): `causal_reliability/theory/predictive_certificate.py`, tests `tests/test_predictive_cic_gate.py`; concise statement in `docs/theory.md` Appendix P (Proposition P)
- Paper integration: `paper/main.tex` Section "Predictive CIC abstention gate" (`sec:predgate`, Table `tab:predgate`) and the Limitations section
- Results: `results/predictive_cic_gate/` (`predictive_gate_key_numbers.json`, `predictive_gate_summary.md`, `predictive_gate_leave_one_benchmark_out.csv`, `predictive_gate_eval_by_benchmark.csv`, `predictive_gate_feature_ranking.csv`, `predictive_gate_features.csv`, `coverage_accuracy_curve.csv`, `calibration_curve.csv`, `predictive_gate_plots.png`)

## Final External Feedback Pass Script

- `scripts/run_final_external_feedback_pass.sh` runs the random augmentation failure benchmark, traffic-sign shortcut validation, label-preservation packet export, and final report rebuild.

## Frozen Generalization Map

The final study is intentionally framed as a finite-candidate reliability framework. The table separates supported claims from explicit non-claims so that controlled repair, natural-image directional evidence, localization limits, predictive abstention, and theory are not conflated.

CIC is **frozen** as a finite-candidate counterfactual intervention framework. Different experiments test different pieces of the same claim: controlled benchmarks establish repair, the semantic-decoy benchmark validates a second shortcut family, COCO-Text gives natural-image **directional** repair evidence but **not** localization support, the predictive gate adds label-free abstention evidence (strongest in pooled/controlled settings), and the theorems define the positive assumptions and the impossibility boundary. This map does **not** claim solved open-world discovery, universal/general robustness, pixel-exact localization, scene-text detection, a passed COCO-Text support gate, an all-natural-image predictive guarantee, or a guarantee that the method self-certifies correctness on every input. It is presented in `paper/main.tex` Section "Frozen generalization map" (Table `tab:frozenmap`).

| Evidence source | Data type | Frozen CIC setting | Main quantitative result | Gate/status | Supports | Does not support |
| --- | --- | --- | --- | --- | --- | --- |
| Hard text-overlay benchmark | Controlled synthetic | Non-oracle text-region proposals, n=32 | 0.250 original → 0.750 CIC top-1; matched random 0.331 | `headline_eligible` | Controlled finite-candidate shortcut repair | Natural-image generality by itself |
| Scale / multi-model audit | Controlled synthetic | Same method, n=128, 4 backbones | 4/4 OpenCLIP models repair_eligible; CIC−random gaps 0.437–0.778 | Supporting | Replication across model/checkpoint scale | All architectures or all shortcut types |
| Semantic-decoy icon benchmark | Controlled synthetic (non-text) | Re-run from scratch, n=128 | CIC top-1 0.711, random 0.258, gap 0.453 | 8/8 gates | A second non-text shortcut family | Universal non-text robustness |
| Spatial-resolution audit | Diagnostic (pooled) | Reads existing benchmark regions | Median IoU 0.39; coverage ≥0.8 in 0.71; repair rises with IoU bucket | Supporting diagnostic | Coarse causal intervention | Pixel-perfect localization |
| Human validation | Human annotation | Label-preservation packet | Majority label preservation 0.960; recognizability 0.970; Fleiss κ strong | Methodology check | Label-preserving interventions | Semantic correctness of every repair |
| COCO-Text natural-image validation | Natural photographs | Shortcut-agnostic proposal CIC | Strict subset CIC 0.538 vs random 0.205, gap +0.333; text overlap 0.128 | Support gate false | Dataset-backed directional repair evidence | Text-box localization |
| COCO-Text localization diagnostic | Natural photographs | Forced best-overlap vs CIC | Ranking failure dominates; forcing best text-overlap repairs worse than CIC | Diagnostic | Entanglement / ranking limitation analysis | CIC as a scene-text detector |
| Predictive CIC reliability gate | Pooled (controlled + natural) | Label-free logistic gate, 8 benchmarks | 2,635 examples; LOBO AUROC 0.789; top-25% accepted precision 0.97 | `predictive_gate_supported` | Label-free abstention in pooled/controlled settings | Reliable hard COCO-Text strict/directional prediction |
| Global conditional theorem | Theory | Proposal-complete intervention family | Repair guaranteed under completeness, m_clean > 2·eps_B, margin γ | Machine-checked | Shortcut-type-agnostic repair under proposal-completeness and residual-balance assumptions | Assumption-free open-world discovery |
| Impossibility theorem | Theory | Finite-query black-box setting | Two worlds match on all queries yet need different repairs | Machine-checked | The need for proposal/intervention assumptions | An empirical repair guarantee |
| Predictive abstention certificate | Theory | Repaired-margin rule | Accept iff m_rep(x) > 2·eps_hat under calibrated bound | Machine-checked | Margin-based accept/abstain under calibrated residual-instability bounds | A semantic-correctness guarantee |
| Finite-candidate characterization | Theory | Finite candidate set A(x) | Exact repair criterion; tight residual-margin certificate (2·eps sharp); proposal-coverage ceiling; repair–localization conflict | Machine-checked (Thms 1–4) | Complete characterization in the finite-candidate setting | Assumption-free open-world discovery; a semantic-correctness guarantee |

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
- Spatial-resolution and causal-intervention audit (`results/spatial_resolution_audit/`) is a supporting diagnostic that addresses the low exact-IoU criticism directly: pooled over 210 shortcut examples, exact box precision is low (median IoU 0.39, hit@IoU0.5 = 0.43) yet selected regions cover the shortcut (any intersection 0.90, coverage >= 0.5 in 0.77) and repair becomes near-certain once coarse overlap is reached (top-1 repair 0.92 for 0.3 <= IoU < 0.5, 1.00 for IoU >= 0.5). It reads existing artifacts only and changes no headline metric. It establishes that **CIC is a coarse causal-intervention method, not an exact localization or segmentation method**; it does **not** claim exact localization, segmentation quality, spatial grounding solved, open-world discovery, or general robustness, and exact localization remains a limitation.

Related work positioning: `docs/related_work.md`.

---

## Moved from paper appendices (compressed for STS 20-page limit)

The submitted report (`paper/main.tex`) was compressed to stay under the STS 20-page
content limit. The detailed appendix tables previously in the paper are reproduced here
so no artifact information is lost. Headline metrics are unchanged.

### Artifact-to-number map

| Number(s) used | Source artifact |
| --- | --- |
| Hard multi-decoy main table | `results/final_report/final_key_numbers.json` |
| Full resampling audit (3 seeds) | `results/hard_multidecoy_clip_repair/full_benchmark_resampling_aggregate.csv`, `_audit.csv` |
| Scale + multi-model audit (n=128, 4 models) | `results/hard_multidecoy_scale_model_audit/scale_model_metrics.csv`, `scale_model_key_numbers.json`, `scale_model_summary.md`, `scale_model_plot.png`, `model_availability.csv` |
| Second shortcut family (semantic-decoy icon, n=64 & n=128) | `results/semantic_decoy_pilot/semantic_decoy_pilot_gates.json`, `results/semantic_decoy_scale_n128/semantic_decoy_pilot_gates.json`; `final_key_numbers.json` (`semantic_decoy_*`) |
| Failure-conditioned table | `results/hard_multidecoy_failure_conditioned/failure_conditioned_metrics.csv`, `_key_numbers.json` |
| Cross-shortcut negative result | `results/cross_shortcut_generalization/cross_shortcut_summary.md`, `final_key_numbers.json` |
| Global additivity | `results/embedding_additivity/embedding_additivity_key_numbers.json` |
| Per-input class-balance | `results/per_input_class_balance/per_input_class_balance_key_numbers.json` |
| Confident-wrong / confidence-solvable AUROCs | `results/final_report/final_key_numbers.json` |
| Dangerous quadrant count / rate | `final_key_numbers.json` (`dangerous_quadrant_*`) |
| Tests / negative controls | `python3 -m pytest`; `final_key_numbers.json` (`negative_controls_*`) |

### Reproducibility checklist

| Item | Status |
| --- | --- |
| Real pretrained OpenCLIP loaded (no fake backend for headline) | yes |
| Non-oracle scorer excludes label/bbox/type/correctness | yes |
| Headline gated on `headline_eligible` | yes |
| Full benchmark resampling with distinct image/metadata hashes | yes (3 seeds) |
| Scale + multi-model replication audit (supporting, n=128) | yes (4/4 loaded, all repair-eligible) |
| — shared resampled benchmark instance across models; hash ≠ n=32 headline | yes |
| Fixed-benchmark determinism check separated from resampling | yes |
| Global-additivity theory claim gated on validation metric | yes (false) |
| Per-input class-balance gate | yes (true for text) |
| Cross-shortcut transfer reported honestly when failing | yes (not eligible) |
| Second shortcut family (semantic-decoy icon; supporting, not headline) | yes (n=64 & n=128 both 8/8 gates) |
| — earlier flat visual-decoy pilot retained as boundary evidence | yes (1 gate fails) |
| Spatial-resolution & causal-intervention audit (diagnostic; not exact localization) | yes (pooled n=210; coarse-intervention framing) |
| Test suite | 382/382 pass |
| Negative controls | 24/24 pass |
| Human label-preservation study (3 annotators, 100 pairs) | yes |
| — majority-vote label preserved / recognizable | 96/100, 97/100 |
| — Fleiss' κ (before/after/change/recognizable) | 0.973 / 0.974 / 0.920 / 1.000 |
| WILDS Waterbirds (metadata-only diagnostic; no CIC repair) | diagnostic |

### COCO-Text proposal-CIC metrics by subset

Real pretrained OpenCLIP ViT-B-32 / laion2b_s34b_b79k (fake backend false), all 500 metadata rows. Source: `results/coco_text_cic_full/coco_text_full_key_numbers.json`.

| Quantity | all_500 | directional_57 | strict_39 |
| --- | --- | --- | --- |
| n | 500 | 57 | 39 |
| Original accuracy | 0.760 | 0.000 | 0.000 |
| Oracle repair (top-3 / top-5) | 0.92 / 0.95 | 0.86 / 0.96 | 0.90 / 1.00 |
| CIC top-1 repair (excl. OCR) | 0.714 | 0.439 | 0.538 |
| Matched random repair | 0.754 | 0.175 | 0.205 |
| CIC − random gap | −0.040 | +0.263 | +0.333 |
| CIC target-prob improvement | 0.154 | 0.684 | 0.692 |
| — random target-prob improvement | 0.168 | 0.474 | 0.487 |
| CIC text-distractor decrease | 0.254 | 0.965 | 0.949 |
| — random text-distractor decrease | 0.222 | 0.649 | 0.718 |
| Selected text-box overlap | 0.070 | 0.140 | 0.128 |
| Selected object-box overlap | 0.492 | 0.351 | 0.359 |
| CIC top-1 (incl. OCR) | 0.716 | 0.456 | 0.564 |

`coco_text_strict_support` = false (text overlap 0.128 < 0.60); `coco_text_directional_support` = false (0.140 < 0.60); `ocr_inclusion_materially_helps` = false.

### COCO-Text localization diagnostic

Source: `results/coco_text_cic_localization_diagnostic/localization_diagnostic_key_numbers.json`.

| Diagnostic quantity | Value |
| --- | --- |
| Proposal text recall, IoU ≥ 0.1 | 0.564 |
| Proposal text recall, IoU ≥ 0.3 | 0.308 |
| Text-overlap@1 / @5 / @10 | 0.128 / 0.308 / 0.359 |
| Median rank of best text-overlapping proposal | 4 |
| Selected CIC alias repair | 0.439 |
| Best text-overlapping proposal alias repair | 0.163 |
| Best object-overlapping proposal alias repair | 0.231 |
| Best text-overlapping beats selected CIC? | no |
| Ranking gap / coverage gap | 0.436 / 0.036 |
| Area-normalized scoring fixes localization? | no |
