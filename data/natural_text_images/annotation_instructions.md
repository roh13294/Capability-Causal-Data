# Natural-Text Dataset — Manual Annotation Instructions

This tool produces `verified_annotations.csv`, the human-verified ground truth
for the natural-text "text-shortcut failure" evaluation. The goal is to mark,
for each of the 50 images, **where the distracting text/logo is**, **where the
real visual object is**, and **whether the image is a clean enough example to
keep in the verified failure eval**.

> Do not run the CIC experiment, regenerate metadata, or edit any existing
> result artifacts. This step only creates `verified_annotations.csv`.

## How to open the tool

Open `annotation_tool.html` in a browser (double-click, or
`file://.../data/natural_text_images/annotation_tool.html`). The 50 images load
directly from disk via their relative paths, and metadata is pre-embedded, so no
local server is required.

Your in-progress work is auto-saved to the browser (localStorage) as you go, so
you can close and reopen without losing annotations. Nothing is written to disk
until you click **Export CSV**.

## The screen

- **Left:** the current image with a drawing canvas on top.
- **Right:** boxes, pre-filled labels, decisions, notes, and an image index.
- **Top bar:** Prev/Next, the active **box mode**, Undo, and **Export CSV**.

Keyboard shortcuts: `←`/`→` prev/next, `t` text-box mode, `o` object-box mode,
`u` undo last box (when not typing in a field).

## Step-by-step per image

### 1. Draw the text/logo box(es) — the important part
Switch to **Text/Logo box** mode (red). Click-drag a rectangle tightly around
each piece of **distracting text, brand logo, banner, caption, sign text, or
on-screen graphic** that a model could use as a shortcut instead of actually
recognizing the object. Examples:

- a brand wordmark/logo (Nike swoosh, Coca-Cola wordmark, FedEx logo)
- a caption or meme text block
- on-screen news text ("BREAKING NEWS", a ticker)
- sign text ("NO DIVING", "SCHOOL SPEED LIMIT 20", "CAUTION WET FLOOR")

Draw **one box per distinct text/logo region**. Multiple boxes are fine and are
stored separated by `|`.

### 2. Draw the object box — only if easy
Switch to **Object box** mode (blue). Draw **one box around the main visual
target** (the thing named by `visual_target_label`) **only if it is easy and
unambiguous**. If the object is occluded, tiny, scattered, or you are unsure,
**skip it** — leave object_boxes blank rather than guessing.

Box coordinates are shown live and in the box list as `x1,y1,x2,y2` in the
image's **natural pixel units** (top-left origin). Use Undo (`u`) or the small
`x` next to a box to remove a mistake.

### 3. Confirm the pre-filled labels
`visual_target_label`, `visual_label_aliases`, and `text_distractor_labels` are
pre-filled from the metadata. Glance at them and fix only if clearly wrong
(e.g. a label that does not belong). Usually leave them as-is.

### 4. Mark `text_driven_candidate`
Your judgment of whether this image plausibly induces a **text-driven shortcut**
(model latches onto the text/logo instead of the object):

- **yes** — strong, obvious text/logo that competes with or dominates the object.
- **maybe** — some distracting text/logo, but not clearly dominant.
- **no** — no meaningful text shortcut.

### 5. Mark `include_in_verified_failure_eval`
- **yes** — set this **only for clear, clean examples**: the visual target is
  genuinely present and recognizable, AND there is distinct distracting
  text/logo. These are the trustworthy cases for the verified eval.
- **no** — set this for anything **confusing, ambiguous, or graphic-heavy**
  (e.g. memes/graphics where the "object" is a cartoon, the target is barely
  present, or text so dominates that there is no fair visual target). When you
  mark **no**, add an `exclusion_reason` and start your `notes` with `REVIEW`.

### 6. `exclusion_reason` and `notes`
- `exclusion_reason`: short reason when excluding (e.g.
  `graphic-heavy meme, no clear visual object`, `target barely present`,
  `text is the only subject`). Leave blank when including.
- `notes`: any extra context. **Prefix with `REVIEW` for confusing or
  graphic-heavy examples** so they are easy to find later. The metadata hint is
  shown on the right for reference.

## When you are done

Click **Export CSV**. The browser downloads `verified_annotations.csv`. Move it
into this folder so the path is:

```
data/natural_text_images/verified_annotations.csv
```

### Output columns
`image_path, visual_target_label, visual_label_aliases, text_distractor_labels,
text_or_logo_boxes, object_boxes, text_driven_candidate,
include_in_verified_failure_eval, exclusion_reason, notes`

### Formatting rules
- Multiple boxes in a cell are separated by `|`.
- Each box is `x1,y1,x2,y2` in natural image pixels (integers).
- Label lists keep the `;` separator from the metadata.
- All 50 rows are exported, including excluded ones (with `include=no`).
