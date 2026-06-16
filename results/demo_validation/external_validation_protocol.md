# External demonstration feedback — protocol

A **small usability check** for the CIC finite-candidate reliability demo. The
goal is to gather **external demonstration feedback** on whether the demo's
explanation and outputs are understandable and plausible to people outside the
project.

> **Bounded scope.** This is *external demonstration feedback* / a *small
> usability check*. It is **not** deployment validation, **not** clinical
> validation, and **not** proof of real-world robustness. It collects opinions
> about the demo's clarity, not measurements of scientific performance. No
> experimental metric, support gate, or `results/final_report/` artifact is
> produced or changed by this activity.

> **Method scope.** CIC tests finite candidate interventions. It is not
> guaranteed open-world shortcut discovery or universal robustness. Reviewers
> should be told this up front.

## Who

- 3–10 external reviewers (not project authors).
- No special expertise required; a short non-technical orientation is fine.

## What reviewers do

1. Read the one-paragraph orientation (below) and the demo's on-screen scope note.
2. Launch the demo (`python3 demo/app.py`) or view a prepared walkthrough.
3. Try **5–10 demo examples** — a mix of the provided sample images and, if they
   wish, their own non-sensitive images.
4. For each example, record their judgments using
   [`external_validation_form.md`](external_validation_form.md), one row per
   example in [`external_validation_template.csv`](external_validation_template.csv).

## Orientation paragraph (read to reviewers)

> This tool looks at an image classifier's prediction and asks: "if I cover up a
> small candidate region (often text), does the prediction change?" If a small
> region was driving the prediction, the tool proposes a repaired prediction;
> if the prediction is confident but unstable, the tool may **abstain**. The
> tool only checks a finite set of candidate regions — it is not a guarantee
> that all shortcuts are found or that the model is robust. We are asking for
> your impression of whether this is clear and plausible, not for a performance
> measurement.

## What reviewers record (per example)

For each example, a yes / partly / no (or 1–5) judgment plus optional comments on:

1. Does the **original prediction** seem shortcut-driven / text-driven?
2. Does the **highlighted region** seem plausible (a reasonable thing to test)?
3. Does the **repaired prediction** seem more object-faithful than the original?
4. Is the **reliability / abstention warning** understandable?
5. Is the **explanation** overall clear?
6. Free-response comments.

## Outputs of this activity

- A filled-in copy of `external_validation_template.csv` (one row per example).
- An optional short qualitative summary written in **bounded language**
  (e.g. "external demonstration feedback suggested the abstention message was
  clear to N of M reviewers"). Avoid any claim of deployment, clinical, or
  real-world-robustness validation.

## What this protocol must NOT claim

- ❌ "deployment validation"
- ❌ "clinical validation"
- ❌ "proof of real-world robustness"
- ✅ "external demonstration feedback"
- ✅ "small usability check"
