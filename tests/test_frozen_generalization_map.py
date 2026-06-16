"""Checks for the frozen generalization-map synthesis/freeze step.

These tests verify only that the synthesis is internally consistent with the
already-frozen artifacts. They must NOT change any headline metric, support
flag, or result JSON. They assert:

  * the frozen-map subsection + table exist in the paper and artifact index,
  * the table includes every major evidence source,
  * forbidden overclaims are absent from the new synthesis text,
  * support flags are unchanged (read from the authoritative result JSONs),
  * final headline metrics are unchanged.
"""

import json
from pathlib import Path

PAPER = Path("paper/main.tex")
INDEX = Path("FINAL_ARTIFACT_INDEX.md")

# The full, granular evidence sources retained in FINAL_ARTIFACT_INDEX.md. The
# index is the detailed home; the paper table is the page-limited condensed view.
INDEX_EVIDENCE_SOURCES = [
    "Hard text-overlay benchmark",
    "Scale / multi-model audit",
    "Semantic-decoy icon benchmark",
    "Spatial-resolution audit",
    "Human validation",
    "COCO-Text natural-image validation",
    "COCO-Text localization diagnostic",
    "Predictive CIC reliability gate",
    "Global conditional theorem",
    "Impossibility theorem",
    "Predictive abstention certificate",
    "Finite-candidate characterization",
]

# Broad evidence axes the compressed (STS 20-page limit) paper table must cover.
# The paper aggregates the granular index rows into these axes, so the test checks
# that the broad *concept* appears, not the old granular row strings. Each axis maps
# to one required piece of evidence:
#   text-overlay                       -> controlled finite-candidate repair
#   semantic-decoy                     -> scale and second-family validation
#   COCO-Text                          -> natural-image COCO-Text boundary
#   Predictive abstention              -> predictive abstention / reliability gate
#   impossibility                      -> theory and impossibility boundary
#   Finite-candidate characterization  -> complete finite-candidate characterization
PAPER_EVIDENCE_AXES = [
    "text-overlay",
    "semantic-decoy",
    "COCO-Text",
    "Predictive abstention",
    "impossibility",
    "Finite-candidate characterization",
]

# Overclaim-form phrases that must never appear in the new map. These are the
# positive (asserted) forms; the map is allowed to NAME the corresponding
# non-claims, but never to assert them.
FORBIDDEN_CLAIMS = [
    "open-world shortcut discovery is solved",
    "solves open-world shortcut discovery",
    "universal natural-image robustness",
    "exact localization is achieved",
    "exact localization was achieved",
    "cic localizes scene text",
    "coco-text support gate passed",
    "coco-text strict support gate passed",
    "predictive gate works on all natural images",
    "works on all natural images",
    "cic always knows when it is right",
    "guaranteed semantic correctness",
]


def _frozen_map_block(text: str, start_marker: str, end_marker: str) -> str:
    assert start_marker in text, f"missing marker: {start_marker}"
    start = text.index(start_marker)
    end = text.index(end_marker, start) if end_marker in text[start:] else len(text)
    return text[start:end]


def test_paper_has_frozen_generalization_map_section_and_table():
    paper = PAPER.read_text(encoding="utf-8")
    assert "Frozen generalization map" in paper
    assert "\\label{tab:frozenmap}" in paper
    assert "\\label{sec:frozenmap}" in paper
    # The required framing paragraph appears verbatim before the table.
    assert (
        "The table separates supported claims from explicit non-claims so that "
        "controlled\nrepair, natural-image directional evidence, localization limits, "
        "predictive\nabstention, and theory are not conflated."
    ) in paper


def test_paper_table_includes_all_major_evidence_sources():
    # The paper table is the compressed, page-limited view: it must cover every
    # broad evidence axis (controlled repair, scale/second-family, COCO-Text,
    # predictive abstention, impossibility theory, and the finite-candidate
    # characterization) without reproducing the granular index rows.
    paper = PAPER.read_text(encoding="utf-8")
    block = _frozen_map_block(paper, "\\label{tab:frozenmap}", "\\end{table}")
    for axis in PAPER_EVIDENCE_AXES:
        assert axis in block, f"frozen-map table missing broad evidence axis: {axis}"


def test_index_has_frozen_generalization_map_with_all_sources():
    index = INDEX.read_text(encoding="utf-8")
    assert "## Frozen Generalization Map" in index
    block = _frozen_map_block(index, "## Frozen Generalization Map", "## Claim Boundary")
    for source in INDEX_EVIDENCE_SOURCES:
        assert source in block, f"index frozen map missing evidence source: {source}"
    # Columns of the requested table are all present.
    for column in [
        "Evidence source",
        "Data type",
        "Frozen CIC setting",
        "Main quantitative result",
        "Gate/status",
        "Supports",
        "Does not support",
    ]:
        assert column in block, f"index frozen-map table missing column: {column}"


def test_no_forbidden_overclaims_in_synthesis_text():
    paper_block = _frozen_map_block(
        PAPER.read_text(encoding="utf-8"),
        "\\section{Frozen generalization map}",
        "\\section{Conclusion}",
    ).lower()
    index_block = _frozen_map_block(
        INDEX.read_text(encoding="utf-8"),
        "## Frozen Generalization Map",
        "## Claim Boundary",
    ).lower()
    for claim in FORBIDDEN_CLAIMS:
        assert claim not in paper_block, f"forbidden claim in paper frozen map: {claim}"
        assert claim not in index_block, f"forbidden claim in index frozen map: {claim}"


def test_final_headline_metrics_unchanged():
    key_numbers = json.loads(
        Path("results/final_report/final_key_numbers.json").read_text(encoding="utf-8")
    )
    assert (
        key_numbers["hard_multidecoy_headline_primary_metric"]
        == "misleading accuracy 0.250 to 0.750"
    )


def test_support_flags_unchanged():
    coco = json.loads(
        Path("results/coco_text_cic_full/coco_text_full_key_numbers.json").read_text(
            encoding="utf-8"
        )
    )
    assert coco["coco_text_strict_support"] is False
    assert coco["coco_text_directional_support"] is False

    gate = json.loads(
        Path("results/predictive_cic_gate/predictive_gate_key_numbers.json").read_text(
            encoding="utf-8"
        )
    )
    assert gate["predictive_gate_supported"] is True
    assert gate["is_universal_theorem"] is False
    assert gate["n_examples"] == 2635
    assert round(float(gate["lobo_pooled_auroc"]), 3) == 0.789


def test_frozen_map_does_not_touch_final_report_dir():
    # The synthesis step must not have introduced a frozen-map artifact under
    # results/final_report/ (that directory is explicitly off-limits).
    final_report = Path("results/final_report")
    names = {p.name for p in final_report.iterdir()}
    assert "frozen_generalization_map.json" not in names
    assert "frozen_map.json" not in names
