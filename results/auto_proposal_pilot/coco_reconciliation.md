# COCO-Text CIC baseline reconciliation

**Question.** The finalized report records, on the strict oracle-repairable
COCO-Text subset (n=39):

- CIC (excl. OCR) strict top-1 alias repair = **0.538**
- matched-random alias repair = **0.205**

The first automated-proposal *pilot* (`run_auto_proposal_cic_pilot.py`) reported,
for its `existing_cic_baseline` family on the same strict n=39:

- CIC top-1 alias repair = **0.410**
- matched-random alias repair = **0.231**

Before any automated-proposal family can be claimed to "beat the baseline", we
must know whether **0.410 is the same quantity as 0.538**. It is not.

## 1. Source of the finalized 0.538

The headline number comes from `results/coco_text_cic_full/coco_text_full_key_numbers.json`,
`subsets.strict_39.cic_excl_strict_top1 = 0.5384615384615384`
(and `random_repair_alias = 0.20512820512820512`). It is produced by
`causal_reliability/experiments/run_coco_text_cic_full.py` (`_run_examples`, the
`cic_top1_repair_excl_ocr` method, `HEADLINE_CIC_METHOD`).

That run, per `results/coco_text_cic_full/coco_text_full_config_used.yaml`,
generates candidates with:

```python
generate_open_region_proposals(
    pil,
    text_boxes=text_boxes,        # OCR/text geometry IS passed in
    object_boxes=object_boxes,
    seed=seed + example_id,       # seed=0
    max_candidates=48,            # DEFAULT_MAX_CANDIDATES / config max_candidates
    grid_scales=[0.18, 0.3, 0.45],
    enable_object_box_family=False,
)   # enable_ocr_family defaults to True
```

then selects `top1` from the **OCR-excluded** subset of the scored candidates
(`excl_scores = [s for s in scores if proposal_family(s.proposal_type) != OCR_FAMILY]`),
and selects the matched-random control via `_select_matched_random(scores, top1_excl, "area_fraction")`.
Scoring is `cic_region_scoring.score_region_candidates`; neutralization is the
default `neutralize_region`; correctness is alias-aware (`is_target_label`).

So "excl OCR" means: **the OCR/text-box geometry family is generated and competes
for the 48 candidate slots, but is removed at the final top-1 selection step.** It
is *not* the same as never generating the OCR family.

## 2. Why the pilot reported 0.410

The pilot's `existing_cic_baseline` (`_baseline_region_proposals`) deliberately
built a **different, smaller, OCR-geometry-free** candidate set:

```python
generate_open_region_proposals(
    pil,
    text_boxes=None,             # no text geometry
    object_boxes=None,
    seed=seed + eid,
    max_candidates=14,           # pilot default, not 48
    # grid_scales=None (library default, not [0.18,0.3,0.45])
    enable_ocr_family=False,     # OCR family never generated
    enable_object_box_family=False,
    enable_random_control=True,
)
```

The pilot's docstring is explicit that it excludes the OCR/text-box geometry
family "so the baseline is a genuine pixel-driven proposal set, matching what the
automatic families do". That is a reasonable design choice for an
auto-vs-auto comparison, but it is a **different estimator** from the finalized
headline.

## 3. Item-by-item check

| dimension | finalized 0.538 | pilot 0.410 | comparable? |
|---|---|---|---|
| strict subset membership / ordering | `coco_text_verified_oracle_repairable_failures.csv`, n=39 | same CSV, n=39 | **yes** (identical 39 ids) |
| alias-aware scoring | `is_target_label` / `aliases_for` | `is_target_label` / `aliases_for` | **yes** |
| OCR family | generated as geometry, excluded at selection | **never generated** (`enable_ocr_family=False`, `text_boxes=None`) | **no** |
| top-1 vs top-3/5 | top-1 | top-1 | yes |
| directional vs strict | strict | strict | yes |
| proposal family / candidate set | full open set incl. OCR geometry | grid+component+saliency+random only | **no** |
| `max_candidates` (cap) | **48** | **14** | **no** |
| `grid_scales` | **[0.18, 0.3, 0.45]** | library default (`None`) | **no** |
| object boxes passed | yes (family disabled; used for diagnostics) | no | minor |
| candidate ranking / scoring path | `score_region_candidates` | `score_region_candidates` | yes |
| neutralization operator | default `neutralize_region` | default `neutralize_region` | yes |
| matched-random selection | `_select_matched_random(..., area_fraction)` | `_select_matched_random(..., area_fraction)` | yes |
| seed | `seed + example_id`, seed=0 | `seed + eid`, seed=0 | yes |

**Three differences drive the 0.538 → 0.410 gap**: (a) candidate cap 48 → 14,
(b) grid scales `[0.18,0.3,0.45]` → default, and (c) the OCR/text geometry family
generated-then-excluded → never generated. The first two shrink and reshape the
candidate pool CIC can choose from; the third removes a geometry family that the
finalized pipeline used (even though it is excluded from the final pick, its
presence changes dedupe/truncation of the surviving 48).

## 4. Verdict

**The pilot `existing_cic_baseline` (0.410) is NOT directly comparable to the
finalized strict CIC repair (0.538).** The two numbers measure CIC repair under
different candidate-generation budgets and families. Any "saliency 0.513 beats
baseline 0.410" statement from the pilot is **not** a valid comparison against
the finalized result and must not be promoted on its own.

## 5. Exact reproduction (apples-to-apples)

The finalized number **is exactly reproducible**. Re-running the finalized recipe
(`max_candidates=48`, `grid_scales=[0.18,0.3,0.45]`, `text_boxes`/`object_boxes`
passed, OCR family generated then excluded at selection, default neutralization,
alias scoring) over the same 39 strict ids yields:

- CIC (excl OCR) top-1 alias repair = **0.5384615384615384** (= 0.538) ✓
- matched-random alias repair = **0.20512820512820512** (= 0.205) ✓

This recipe is implemented as the `existing_cic_baseline_a2a` family in
`experiments/run_coco_text_auto_proposal_sweep.py` and is the **only** baseline
the automated proposal families are compared against in the full sweep. No
finalized artifact is modified; the a2a baseline is recomputed from the same data
and code path purely for the comparison.
