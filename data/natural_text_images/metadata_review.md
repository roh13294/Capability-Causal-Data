# Natural-Text Benchmark — Metadata Review

Review date: 2026-06-15
File reviewed: `data/natural_text_images/metadata.csv`
Reviewer pass: every one of the 50 images was opened and inspected individually (not
just the contact sheet). This document reflects the **finalized** metadata after the
descriptive-object cleanup pass.

## 1. Final validation summary

| Check | Result |
|---|---|
| Total rows | 50 ✓ |
| Columns per row | 8 ✓ |
| Empty `human_label` | none ✓ |
| `TODO` markers | none ✓ |
| `allowed_clip_labels` size | every list 4–8 semicolon-separated labels ✓ |
| First `allowed_clip_labels` entry == `human_label` | all 50 rows ✓ |
| Bare container labels remaining as `human_label` | none ✓ |
| Rows changed in the final descriptive-object pass | 16 |
| Rows flagged `REVIEW:` | 1 (`images/CNBC.jpg`) |

## 2. Final label policy

- **`human_label` = the main visual subject/object** of the image — the thing CIC
  should key on, not the image format or the on-image text.
- **Brand / text / logo terms stay as distractors** in `allowed_clip_labels`
  (e.g. `coke zero`, `nike`, `colgate`, `breaking news`), never as `human_label`.
- **Descriptive physical-object labels are preferred over bare container labels**
  (`soda can` over `can`, `cereal box` over `box`, `chip bag` over `bag`). The bare
  shape word is retained as a secondary distractor so the broader category is still
  represented.
- **`meme` / `screenshot` / `caption` are not used as `human_label`** unless no visual
  subject exists; they live in `allowed_clip_labels` and `notes` as text/format
  distractors. After this pass no row uses a bare format word as its subject.

## 3. Final descriptive-object changes (16)

| image | was | now |
|---|---|---|
| coke0.jpg | can | soda can |
| cocoshampoo.jpg | bottle | shampoo bottle |
| gbotle.jpeg | bottle | sports drink bottle |
| hydroflask.jpeg | bottle | water bottle |
| pepsibottle.jpeg | bottle | soda bottle |
| tidebottl.jpg | bottle | detergent bottle |
| frostedflakes.jpg | box | cereal box |
| nikebox.jpg | box | shoe box |
| psbox.jpeg | box | console box |
| toothpastebox.jpeg | box | toothpaste box |
| doritosbag.jpeg | bag | chip bag |
| hot.jpg | bag | chip bag |
| sour.jpg | bag | candy bag |
| mcdonalds.jpeg | bag | takeout bag |
| milkorganic.jpg | carton | milk carton |
| starbucks.jpeg | cup | coffee cup |

In each case the bare shape word (`can` / `bottle` / `box` / `bag` / `carton` / `cup`)
was kept as the second entry of `allowed_clip_labels`, with brand/text/logo terms as
the remaining distractors — e.g. `soda can;can;soda;coca cola;coke zero;logo`.

### Labels intentionally left as-is (already descriptive or special-case)

- Already-descriptive objects: `spray bottle` (bleach), `safety cone` (wetfloor),
  `shoe` (nikeswoosh), `game controller` (xbox), `phone case` (supremecase),
  `backpack`, `notebook`, `hard hat`, `headphones`, `laptop`, `shopping cart`.
- Dominant-text signage kept generic by design: `sign` (nodiving, schoolsign) and
  `storefront` (treesubway) — the sign/storefront is the dominant visual target and
  the specific phrasing (e.g. `no diving sign`, `school zone sign`, `subway`) is kept
  as a distractor.
- Concrete subjects from the earlier meme/still pass: `person`, `dog`, `cat`, `owl`,
  `cartoon character`, `basketball player`, `news anchor`.

## 4. Remaining REVIEW row (1)

- **`images/CNBC.jpg`** — `human_label = news anchor`. A male anchor is visible at the
  left, but the S&P 500 chart, the scrolling ticker numbers, and the "BREAKING NEWS"
  banner occupy most of the frame, so the on-image text dominates the anchor. The
  `notes` field begins with `REVIEW:` to mark this.

## 5. Recommendation

`images/CNBC.jpg` may remain in the **exploratory** run, but it should be **excluded or
separately flagged** from any clean *supported* natural-text claim unless a human
manually approves it. Because its ticker/banner/chart text dominates the intended
subject, it is the strongest candidate for a text-shortcut confound and must not be
counted as a clean supported example without manual sign-off. No other row requires
review, and no hard drops are recommended.

## Not modified

Per scope, only this review document was refreshed in this step.
`data/natural_text_images/metadata.csv` was left unchanged (no validation bug found),
and the build script, images, contact sheet, and all final metrics / prior result
artifacts were untouched. The CIC experiment was **not** run.
