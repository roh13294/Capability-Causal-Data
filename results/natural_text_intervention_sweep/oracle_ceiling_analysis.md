# Natural-text intervention-operator: oracle-ceiling analysis

**Diagnostic only.** This analysis isolates whether strict natural-text repair is 
limited by the *intervention operator / masking strategy* or by *CIC proposal 
selection/scoring*. It does **not** update any headline / final-report metric, and 
`open_world_claim_allowed` stays **False**.

## Oracle ceiling (operator applied to the annotated text/logo boxes)

- Best oracle **strict** operator: `expanded_gray_fill_1.25` at **0.448** strict repair.
- Best **directional** operator: `gray_fill` at 1.000 target-probability improvement rate.
- Does any operator raise oracle strict repair above 0.50? **NO**.
- Does any operator raise oracle strict repair above 0.70? **NO**.

## CIC (operator applied to the existing CIC top-1 proposal)

- Best CIC **strict** operator: `expanded_gray_fill_1.25` at **0.276** strict repair.
- Best CIC vs matched-random strict gap: **0.138** (operator `black_fill`).

## Interpretation

- No operator lifts oracle strict repair above 0.50 even with the known text box, while directional improvement stays high: exact top-1 natural-image recovery is limited by residual natural-image ambiguity / label-set difficulty, not only by CIC.

## Strict-support candidacy

- CIC strict beats the random baseline by >= 0.15: **NO**.
- A pre-declared GLOBAL operator (`gray_fill`, chosen by "oracle target-probability improvement rate (global, aggregate)") would let a strict gate pass: **NO**.

> Even where a candidate is flagged, the final paper is **not** updated here. Any 
> strict natural-text support requires separate review, and natural-text directional 
> evidence is never reported as positive strict support unless the strict gate truly 
> passes. `open_world_claim_allowed = false`.

_Operator panel: 13 operators (1 unavailable: telea_inpaint). Verified failures: 29._