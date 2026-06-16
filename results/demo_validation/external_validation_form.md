# External demonstration feedback — reviewer form

Fill in one copy of the per-example questions for **each** of the 5–10 examples
you try. Record answers as a row in
[`external_validation_template.csv`](external_validation_template.csv).

> This is a **small usability check** / **external demonstration feedback**. It
> is not deployment validation, not clinical validation, and not proof of
> real-world robustness. CIC tests finite candidate interventions; it is not
> guaranteed open-world shortcut discovery or universal robustness.

## Reviewer details (once)

- Reviewer ID (anonymous is fine):
- Date:
- Mode used (`mock` / `real`):

## Per-example questions

For each example, answer `yes` / `partly` / `no` (or `1`–`5`, 5 = strongly agree),
then add a short comment.

1. **Shortcut-driven original?** The original prediction seems shortcut /
   text-driven rather than based on the main object.
   `[ yes | partly | no ]`

2. **Plausible highlighted region?** The region the tool highlighted seems like
   a reasonable thing to test.
   `[ yes | partly | no ]`

3. **More object-faithful repair?** The repaired prediction seems more faithful
   to the actual object than the original.
   `[ yes | partly | no | n/a (abstained / no change) ]`

4. **Understandable reliability warning?** The accept / abstain reliability
   message was understandable.
   `[ yes | partly | no ]`

5. **Clear explanation overall?** The overall explanation of what the tool did
   was clear.
   `[ yes | partly | no ]`

6. **Comments (free response):**
   `__________________________________________________`

## Overall (once, optional)

- One thing that was confusing:
- One thing that was clear:
- Did the scope/limitations note come across? `[ yes | partly | no ]`
