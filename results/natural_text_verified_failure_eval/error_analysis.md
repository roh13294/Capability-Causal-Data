# Natural-Text Verified-Failure Error Analysis

**Strict support gate remains FAILED; directional evidence is diagnostic only.**

- Verified text-driven failures: 29
- `natural_text_supported`: False (unchanged, strict)
- `open_proposal_supported`: False (unchanged, strict)
- `open_world_claim_allowed`: False
- `natural_text_directional_evidence` (diagnostic): True

## Failure-cause summary

Which of the five candidate causes dominates:

1. Poor proposal selection: CIC selected text overlap rate = 0.724; CIC strict repair = 0.241, directional (target-prob up) = 0.931.
2. Weak masking/intervention: oracle text-box strict repair = 0.310 but oracle directional (target-prob up) = 0.966 — removing the text moves probability toward the target far more often than it flips the exact argmax.
3. Overly strict exact-label evaluation: oracle alias-aware top-1 = 0.310 vs strict 0.310; 0 failures recover an alias at top-1 but never the exact string.
4. Alias/label mismatch: 0 failures flagged as label/alias ambiguity.
5. Natural-image ambiguity: 1 hard failures with no clear oracle repair or directional movement.

## Best successes (CIC strict repaired)

- 3:headphones
- 12:chip bag
- 15:sports drink bottle
- 30:milk carton
- 35:jacket
- 36:soda bottle
- 45:toothpaste box

## Oracle-only successes (oracle strict-repaired, CIC did not)

- 5:spray bottle
- 17:chip bag
- 29:owl

## CIC failures despite text overlap (selected text, did not repair)

- 26:person

## CIC failures due to object overlap / content damage

- (none)

## Likely label-ambiguity cases

- (none)

## Examples to drop or relabel (oracle recovers only an alias, never the exact string)

- (none)

## Hard natural images (no clear repair)

- 13:notebook
