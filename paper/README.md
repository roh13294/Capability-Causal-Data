# Paper: Beyond Confidence — Counterfactual Stability as a Second Axis of Neural Network Reliability

LaTeX research report for the STS project, built from the repository's **current**
result artifacts. No numbers are hand-entered from old conversations; every figure
and table cites its source artifact.

## Contents

- `main.tex` — the paper (article class; portable packages only).
- `references.bib` — bibliography. Entries marked `% VERIFY BIBTEX` need venue/year
  confirmation before any external submission.
- `make_figures.py` — regenerates all figures from artifact numbers (matplotlib, no
  seaborn).
- `figures/` — generated PDF + PNG figures.
- `tables/` — markdown mirrors of the in-paper tables for quick reference.
- `build_report.sh` — regenerates figures and compiles the PDF.

## Building the PDF

A LaTeX toolchain is required. **It was not installed in the authoring
environment**, so `main.pdf` is not committed. To build:

```bash
# macOS:   brew install --cask mactex-no-gui   (or basictex)
# Debian:  sudo apt-get install texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra
chmod +x paper/build_report.sh
bash paper/build_report.sh
```

This runs `pdflatex → bibtex → pdflatex → pdflatex` and writes `paper/main.pdf`.

If you lack a local toolchain, upload `main.tex` + `references.bib` + `figures/` to
Overleaf and compile there (article class, no exotic packages).

## Regenerating figures only

```bash
python3 paper/make_figures.py
```

## Headline numbers used (current artifacts)

- Hard multi-decoy: misleading **0.250 → CIC 0.750**, matched random **0.331**,
  oracle 1.00, clean drop 1.0%.
- Full resampling audit (3 seeds): CIC−random gap **0.39–0.54** (mean 0.46), clean
  drop ≈0.01.
- Failure-conditioned (n=50): CIC **0.96 / 0.98** vs matched random **0.112**;
  original 0 by construction.
- Theory: global additivity **not supported**; per-input class-balance **supported**
  for text.
- Cross-shortcut watermark transfer: **not headline-eligible** (no transfer).

The legacy `0.219 → 0.875` headline in `FINAL_ARTIFACT_INDEX.md` is **stale** and is
NOT used; see Appendix A of the paper.
