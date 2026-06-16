# Natural-Image Open-Candidate (Shortcut-Agnostic) CIC

This experiment is **natural-image validation** of **shortcut-agnostic proposal-based CIC**: 
an **open-candidate intervention search** over candidate regions generated without a 
pre-specified shortcut family. The method **does not require a pre-specified shortcut family, 
but still depends on candidate region proposals**.

Backend: `open_clip`. Model: `ViT-B-32`. Pretrained loaded: `True`.
Data: loaded 50 local images.
Open-proposal claim supported: `False`.
Not supported: CIC top-1 does not beat matched random by >= 0.15 (gap=0.020); content-preservation drop 0.667 > 0.10.

## Key numbers

- n images: 50
- Original CLIP accuracy: 0.12
- High-confidence failure rate: 0.74
- CIC top-1 repair accuracy: 0.2
- CIC top-3 repair accuracy: 0.22
- Matched-random proposal repair accuracy: 0.18
- Largest-region repair accuracy: 0.2
- OCR-only proposal repair accuracy: None
- Oracle text-box repair (eval-only upper bound): None
- CIC vs matched-random gap: 0.020000000000000018
- Content-preservation rate: 0.3333333333333333
- Mean selected-region area fraction: 0.06363042091836733
- Selected overlaps OCR/text box rate: 0.0
- Selected overlaps object box rate: 0.0
- Candidate families present: connected_component, edge_dense, grid_patch, high_contrast, random_patch

## Scope

- Candidate scoring received only pixels, proposal geometry, and model predictions. It did not 
  receive true labels, OCR text content, the shortcut box, correctness, or the benchmark condition.
- True labels, text boxes, and object boxes are used only for evaluation and oracle upper bounds.
- This is proposal-based shortcut discovery, **not** full open-world discovery. 
  `open_world_claim_allowed = False`.

## Metrics

| method                         | backend   | model_name | pretrained_loaded | oracle_upper_bound | n_examples | accuracy |
| ------------------------------ | --------- | ---------- | ----------------- | ------------------ | ---------- | -------- |
| original_clip_prediction       | open_clip | ViT-B-32   | True              | False              | 50         | 0.12     |
| cic_top1_repair                | open_clip | ViT-B-32   | True              | False              | 50         | 0.2      |
| cic_top3_repair                | open_clip | ViT-B-32   | True              | False              | 50         | 0.22     |
| matched_random_proposal_repair | open_clip | ViT-B-32   | True              | False              | 50         | 0.18     |
| largest_region_repair          | open_clip | ViT-B-32   | True              | False              | 50         | 0.2      |
| ocr_only_proposal_repair       | open_clip | ViT-B-32   | True              | False              | 0          |          |
| oracle_text_box_repair         | open_clip | ViT-B-32   | True              | True               | 0          |          |