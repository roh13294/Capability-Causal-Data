# Verified Natural-Text Text-Driven Failure Evaluation

**Natural-image validation of shortcut-agnostic, proposal-based CIC** on a 
human-verified curated annotation set. This is an **open-candidate intervention 
search**; it **does not require a pre-specified shortcut family, but still depends 
on candidate region proposals**. It is **not** full open-world shortcut discovery.

Backend: `open_clip`. Model: `ViT-B-32`. 
Real pretrained loaded: `True`. Fake backend: `False`.
Data: verified annotations: loaded 37 of 37 include=yes rows (50 total) from verified_annotations.csv.

**Result: natural-text verified failure pilot remains unsupported.**
Failed reasons: CIC top-1 does not beat matched random by >= 0.15 on verified failures (gap=0.103); content-preservation drop 0.750 > 0.10 (and not clearly explained).

## Key numbers

- Total annotated images: 50
- include=yes images evaluated: 37
- Real pretrained model loaded: True
- Fake backend: False
- Original CLIP accuracy (include=yes): 0.10810810810810811
- High-confidence failure rate (include=yes): 0.7837837837837838
- Verified text-driven failures: 29
- Oracle text-box repair accuracy (verified failures): 0.3103448275862069
- Oracle text-box repair-or-improve rate (verified failures): 0.9655172413793104
- CIC top-1 repair accuracy (verified failures): 0.2413793103448276
- CIC top-3 repair accuracy (verified failures): 0.20689655172413793
- Matched-random proposal repair accuracy (verified failures): 0.13793103448275862
- Largest-region repair accuracy (verified failures): 0.1724137931034483
- OCR/text-box proposal repair accuracy (verified failures): 0.23076923076923078
- CIC vs matched-random gap (verified failures): 0.10344827586206898
- Content-preservation rate: 0.25
- Content-preservation drop: 0.75
- Selected-region overlap with text boxes (verified failures): 0.7241379310344828
- Selected-region overlap with object boxes (verified failures): 0.3103448275862069
- Mean selected-area fraction (verified failures): 0.07049324309465164
- Candidate families present: connected_component, edge_dense, grid_patch, high_contrast, ocr_text_box, random_patch

## Gate status

- `natural_text_supported`: False
- `open_proposal_supported`: False
- `open_world_claim_allowed`: False
- `no_oracle_leakage`: True
- Failed gate reasons: ['CIC top-1 does not beat matched random by >= 0.15 on verified failures (gap=0.103)', 'content-preservation drop 0.750 > 0.10 (and not clearly explained)']

## Examples worth inspecting

- 3:headphones
- 12:chip bag
- 15:sports drink bottle
- 30:milk carton
- 35:jacket
- 36:soda bottle
- 45:toothpaste box

## Scope guard

- Candidate scoring received only pixels, proposal geometry, and model predictions.
- True labels, text/logo boxes, and object boxes are used only for evaluation and the 
  oracle upper bound.
- Proposal-based shortcut discovery, **not** full open-world discovery. 
  `open_world_claim_allowed = False`.

## Metrics (accuracy restricted to verified text-driven failures)

| method                         | backend   | model_name | pretrained_loaded | oracle_upper_bound | scope             | n_examples | accuracy_on_verified_failures |
| ------------------------------ | --------- | ---------- | ----------------- | ------------------ | ----------------- | ---------- | ----------------------------- |
| original_clip_prediction       | open_clip | ViT-B-32   | True              | False              | verified_failures | 29         | 0                             |
| oracle_text_box_repair         | open_clip | ViT-B-32   | True              | True               | verified_failures | 29         | 0.3103                        |
| cic_top1_repair                | open_clip | ViT-B-32   | True              | False              | verified_failures | 29         | 0.2414                        |
| cic_top3_repair                | open_clip | ViT-B-32   | True              | False              | verified_failures | 29         | 0.2069                        |
| matched_random_proposal_repair | open_clip | ViT-B-32   | True              | False              | verified_failures | 29         | 0.1379                        |
| largest_region_repair          | open_clip | ViT-B-32   | True              | False              | verified_failures | 29         | 0.1724                        |
| ocr_text_box_proposal_repair   | open_clip | ViT-B-32   | True              | False              | verified_failures | 13         | 0.2308                        |