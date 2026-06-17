# Automated Finite-Candidate Proposal Generation for CIC (Pilot)

This document describes a pilot that tests whether **automatic** proposal
generation can reduce CIC's proposal bottleneck on natural-image settings. It is
deliberately scoped and honest about what it does and does not show.

## Core framing

CIC (Counterfactual Intervention Certification) scores a **finite candidate set**
of region interventions and selects the one whose neutralization most changes the
model's prediction. In the finalized COCO-Text / Waterbirds work, that candidate
set was produced by a manually-designed open-proposal generator tuned around the
known shortcut family (scene text, background).

This pilot changes **only one thing**: the candidate set is generated
**automatically from pixels** by simple, dependency-light generators:

| generator | description | dependency |
|---|---|---|
| `grid_boxes` | multi-scale grid / open-region boxes | numpy / PIL (always available) |
| `edge_component_boxes` | classical connected-component boxes | numpy / PIL / scipy (always available) |
| `saliency_boxes` | image-gradient / edge saliency boxes | numpy / scipy (always available) |
| `sam_boxes` | SAM / SAM2 segment proposals | optional; skips cleanly if not installed |
| `dino_boxes` | GroundingDINO / DINO detector boxes | optional; skips cleanly if not installed |

**This is automated finite-candidate proposal generation, _not_ guaranteed
open-world shortcut discovery.** CIC still scores a finite set; we have only
removed the manual, shortcut-specific design of that set. The optional adapters
never import a heavy dependency at module load, never trigger a download unless
`--allow-download` is passed, and report `available=false` with a clear
`skip_reason` when the dependency is missing. The classical generators are
sufficient to run the entire pilot with no extra installs.

## What we measure

### COCO-Text (`experiments/run_auto_proposal_cic_pilot.py`)

We reuse the existing verified COCO-Text subsets (strict oracle-repairable and
directional text-driven failures) and the existing CIC scoring logic. For each
automatic proposal family we compare against the **existing CIC proposal
baseline** (`generate_open_region_proposals`) on:

- original alias-aware accuracy and CIC-selected repair accuracy,
- matched-random repair accuracy and the CIC−random gap,
- target-probability improvement and text-distractor probability decrease,
- selected text-box / object-box overlap and selected area fraction,
- **proposal coverage ceiling** at IoU ≥ 0.1 and IoU ≥ 0.3 (does *any* candidate
  in the family cover a human text box?),
- median rank of the best text-overlapping proposal,
- a **repair-localization conflict** diagnostic (see below).

### Waterbirds (`experiments/run_waterbirds_auto_proposal_diagnostic.py`)

Waterbirds has a background shortcut. Locally we have the WILDS images but **no
oracle bird/background masks**. We therefore generate foreground/background-ish
proposals automatically, score them with CIC, neutralize the selected region, and
report per group — landbird/waterbird × land/water — original vs repaired
accuracy, worst-group accuracy before/after, overall accuracy, average confidence
change, a background-sensitivity proxy (aligned-vs-conflicting accuracy gap), and
the selected region's area and background-vs-foreground tendency.

Because there is no oracle mask defining "the correct region to neutralize", the
Waterbirds run is a **diagnostic, not full validation**.

## Go / no-go thresholds

The pilot is marked `pilot_promising=true` only if a pre-registered threshold is
cleared, so a weak result is preserved honestly as negative/diagnostic rather
than oversold.

**COCO-Text** — promising if at least one automatic family beats the existing
proposal baseline by:
- CIC strict/directional repair improves by ≥ **+0.10** absolute, **or**
- selected text overlap improves by ≥ **+0.15** absolute without repair dropping
  by more than 0.05, **or**
- proposal coverage ceiling (IoU ≥ 0.1) improves by ≥ **+0.20** absolute.

**Waterbirds** — promising if:
- worst-group accuracy improves by ≥ **+0.05** absolute without overall accuracy
  dropping by more than 0.03, **or**
- the background-sensitivity proxy drops substantially (≥ 0.05) while label
  accuracy is preserved.

## Repair vs. human-box localization (reframing Theorem 8, docs-only)

A recurring observation is that the CIC-selected region often does **not** line up
with the human text/annotation box, yet the repair can still succeed (and
sometimes the box overlaps but the repair does not help). This is a
**repair-localization conflict**, and it is *expected*, not a defect:

- **CIC optimizes causal effect on prediction stability.** It selects the region
  whose neutralization most restores the model's causal margin toward the correct
  class.
- **Human annotation boxes mark where the text/object _is_**, which need not be
  the region whose removal best repairs the model. The model may lean on a halo
  around the text, on co-located texture, or on a different entangled cue.
- Therefore a low IoU between the selected region and the human box is a
  **statement about where the model's decision is causally fragile**, not evidence
  that CIC is mislocalizing. The pilot reports this conflict rate explicitly so it
  is visible rather than hidden.

This is a scoped, empirical reframing for these contrastive vision-language
settings. **We do not claim a universal theorem** that human boxes and
repair-optimal regions diverge for all contrastive models or all tasks.

## Human validation note (docs-only, no report changes)

Existing human validation in this project supports that the interventions used
here **preserve semantics** rather than destroying content:

- 96/100 label preservation,
- 97/100 recognizability,
- high inter-annotator agreement.

This indicates that neutralizing a selected region generally keeps the image
recognizable as its true class, which is a precondition for a *repair* (rather
than a destructive edit). It does not, by itself, establish that automatic
proposals improve repair — that is exactly what the go/no-go thresholds above
test. (This note references existing validation; it does not alter the finalized
report.)

## Global additivity note (docs-only)

An earlier global-additivity test — fitting a single global "remove-the-text"
direction in embedding space — **failed** to behave additively. A plausible
reading is that **typographic-shortcut shifts are object-entangled in OpenCLIP**:
the direction that removes the text cue is not constant across images because it
is bound up with the object's own representation. That makes a single global
text-removal direction a poor fit and **motivates per-input intervention**, which
is what proposal-based CIC does (it picks a region per image rather than applying
one global edit).

This is a bounded observation about these models and this shortcut. **We do not
claim global debiasing is universally doomed** — only that, here, a single global
direction was a worse tool than per-input region intervention.

A local, first-order formalization of this observation — a shortcut intervention
operator, the induced logit displacement `J_f(x) Delta_S(x)`, and a local
class-balance lemma — is given in
[local_operator_jacobian_theory.md](local_operator_jacobian_theory.md). It explains
why object-entanglement breaks the global-additive shortcut assumption rather than the
per-input class-balance condition CIC actually uses.

## Outputs and guarantees

All artifacts are written **only** under `results/auto_proposal_pilot/`:

- `coco_text_auto_proposal_metrics.json`, `coco_text_auto_proposal_table.csv`
- `waterbirds_auto_proposal_metrics.json`, `waterbirds_auto_proposal_table.csv`
- `summary.md`

The pilot does not modify the finalized report, the headline COCO-Text /
Waterbirds artifacts, `paper/main.tex`, or any existing result JSON/CSV. It never
downloads datasets or models unless explicitly asked, and it skips cleanly with a
recorded reason when real CLIP or local data is unavailable.

## Reproducing

```bash
# Generators + wiring only (no scoring, no downloads):
python3 experiments/run_auto_proposal_cic_pilot.py --dry-run --max-examples 50
python3 experiments/run_waterbirds_auto_proposal_diagnostic.py --dry-run --max-examples 50

# Capped pilot (needs cached OpenCLIP weights + local data; otherwise skips cleanly):
python3 experiments/run_auto_proposal_cic_pilot.py --max-examples 50
python3 experiments/run_waterbirds_auto_proposal_diagnostic.py --max-examples 50
```
