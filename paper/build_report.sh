#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Regenerate figures from repository artifacts (safe to re-run).
python3 make_figures.py

# Compile. Requires a LaTeX toolchain (pdflatex + bibtex).
pdflatex -interaction=nonstopmode main.tex
bibtex main || true
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Built main.pdf"
