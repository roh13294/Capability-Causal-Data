# `results/auto_proposal_pilot/`

Outputs of the **automated finite-candidate proposal CIC pilot**. This directory
is the *only* place the pilot writes. It does not touch `results/final_report/`,
the headline COCO-Text / Waterbirds artifacts, `paper/main.tex`, or any existing
result JSON/CSV.

## Scope (read first)

This pilot tests **automated finite-candidate proposal generation** — CIC still
scores a finite candidate set, but the set is generated automatically from pixels
(grid / edge-component / saliency / optional SAM / optional DINO) rather than
designed by hand around the shortcut family.

It is **not** open-world shortcut discovery, **not** universal repair or general
robustness, **not** deployment or clinical validation, and **not** a replacement
for the finalized STS report. Weak results are preserved honestly as a
negative/diagnostic outcome (`pilot_promising=false`).

## Files

| file | produced by | contents |
|---|---|---|
| `coco_text_auto_proposal_metrics.json` | `experiments/run_auto_proposal_cic_pilot.py` | per-subset, per-family COCO-Text metrics + go/no-go |
| `coco_text_auto_proposal_table.csv` | same | flat table of the same metrics |
| `coco_text_auto_proposal_per_example.csv` | same | per-example diagnostics (if run) |
| `waterbirds_auto_proposal_metrics.json` | `experiments/run_waterbirds_auto_proposal_diagnostic.py` | per-group Waterbirds diagnostic metrics + go/no-go |
| `waterbirds_auto_proposal_table.csv` | same | flat per-group table |
| `waterbirds_auto_proposal_per_example.csv` | same | per-example diagnostics (if run) |
| `summary.md` | either script (regenerated) | combined human-readable summary, generator availability, non-claims |
| `coco_reconciliation.md` | hand-authored from `run_coco_text_cic_full.py` analysis | why the pilot baseline (0.410) is **not** the finalized strict CIC repair (0.538), item-by-item |
| `coco_text_auto_proposal_sweep_metrics.json` | `experiments/run_coco_text_auto_proposal_sweep.py` | apples-to-apples sweep: a2a finalized baseline + auto families, all metrics, promotion verdict |
| `coco_text_auto_proposal_sweep_table.csv` | same | flat per-subset/per-family table |
| `coco_text_auto_proposal_sweep_per_example.csv` | same | per-example diagnostics |
| `full_coco_sweep_summary.md` | same (regenerated) | reconciliation result, full strict/directional/all-500 table, promotion verdict, non-claims |

**Apples-to-apples sweep.** `coco_text_auto_proposal_sweep_*` introduces the
`existing_cic_baseline_a2a` family, which *exactly* reproduces the finalized
`cic_top1_repair_excl_ocr` recipe (strict repair 0.538 / matched random 0.205).
Every automated proposal family is compared **only** against this baseline. This
supersedes the first pilot's `existing_cic_baseline` (0.410), which used a
different candidate budget/family and is **not** directly comparable to the
finalized report (see `coco_reconciliation.md`).

**SAM (Segment Anything) family.** `sam_boxes` can be added to the sweep with
`--include-sam`. It loads a local `segment_anything` checkpoint
(`models/sam/sam_vit_b_01ec64.pth`, model type `vit_b`; **gitignored, never
committed, never auto-downloaded**) and runs `SamAutomaticMaskGenerator`, then
converts masks to XYXY boxes (XYWH→XYXY), filters by side/area fraction,
deduplicates by IoU, and keeps the top-K (default 48, matching the finalized
candidate cap) by a predicted_iou / stability_score / area heuristic. All knobs
are `--sam-*` CLI flags. If the package or checkpoint is missing, the family is
reported `available=false` with a clear `skip_reason` and the sweep proceeds with
the classical families unaffected. This stays **automated finite-candidate
proposal generation**, not open-world shortcut discovery.

**Fast, cached, fail-safe.** SAM mask generation is expensive on CPU, so the SAM
path is built to stay fast and bounded:

- **Fast defaults (`--sam-fast`)**: `points_per_side=8`, `crop_n_layers=0`, and the
  longest image side downscaled to `--sam-max-side 512` before SAM. Boxes are
  filtered (side/area) and IoU-deduplicated *before* CLIP scoring; examples are
  capped (`--max-examples`) before any masks are generated.
- **Per-image proposal cache (`--cache-sam-proposals` / `--resume`)**: each image's
  final SAM boxes are cached under `cache/sam_proposals/` (gitignored) keyed by
  image id+content-hash, model type, checkpoint name, points_per_side, the two
  thresholds, max_proposals, and the resize/crop settings. A cache hit replays the
  boxes without loading SAM, so reruns/`--resume` are near-instant.
- **Runtime guard (`--sam-timeout-seconds`)**: if SAM generation exceeds the budget
  the sweep stops cleanly and writes a **bounded partial** summary (marked
  `timed_out`/`partial`), never failing silently.
- **Guarded scope**: with `--include-sam` the default subset is **strict-only**
  (n=39). The `all_500` SAM subset is never run unless you pass both `--all500` and
  `--confirm-slow-sam`. `sam_promotable` is forced `false` unless the strict run
  completes fully and clears a pre-registered threshold.

Result (real CLIP ViT-B-32, strict n=39, earlier full-res run): SAM substantially
raises the proposal **coverage ceiling** (IoU≥0.1 ≈ 0.77 vs baseline 0.56;
IoU≥0.3 ≈ 0.62 vs 0.31) but its strict CIC repair (≈0.49) sits just below the
apples-to-apples finalized baseline (0.538). It narrowly misses the coverage
promotion criterion (coverage gain ≥ +0.20 holds, but the accompanying
strict-repair drop is ≈0.051, just over the 0.05 tolerance), so
**`sam_promotable=false`**. Exact, current numbers live in
`coco_text_auto_proposal_sweep_metrics.json` (`sam`, `sam_promotion`,
`sam_promotable`) and `full_coco_sweep_summary.md`.

A `status` field of `skipped` (with `skip_reason`) means real CLIP weights or
local data were unavailable; the pilot exits cleanly rather than fabricating
numbers. See `docs/auto_proposal_pilot.md` for the full design, metrics, and the
repair-vs-localization discussion.

## Regenerate

```bash
python3 experiments/run_auto_proposal_cic_pilot.py --max-examples 50
python3 experiments/run_waterbirds_auto_proposal_diagnostic.py --max-examples 50
# Apples-to-apples sweep (a2a finalized baseline + auto families):
python3 experiments/run_coco_text_auto_proposal_sweep.py --subset strict,directional,all_500
# Fast, cached, strict-only SAM (requires models/sam/sam_vit_b_01ec64.pth on disk):
python3 experiments/run_coco_text_auto_proposal_sweep.py --include-sam --strict-only \
    --max-examples 39 --sam-fast --sam-points-per-side 8 --sam-max-side 512 \
    --cache-sam-proposals --resume --sam-timeout-seconds 1800
# Intentionally slow all_500 SAM run (must be explicit):
python3 experiments/run_coco_text_auto_proposal_sweep.py --include-sam --all500 --confirm-slow-sam --subset auto
```

No datasets or models are downloaded unless `--allow-download` is passed. The SAM
checkpoint is never downloaded by Python — place it at `models/sam/` manually.
