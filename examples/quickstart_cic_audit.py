from causal_reliability.api import CICScorer, ReliabilityPlane, TextMarkerIntervention
from causal_reliability.audit.run_cic_audit import _demo_model


examples = [
    {"example_id": "review_1", "label": 1, "text": "source: alpha The acting is warm and engaging."},
    {"example_id": "review_2", "label": 0, "text": "source: alpha The plot is thin and dull."},
]

scorer = CICScorer(_demo_model, [TextMarkerIntervention()])
certificates = scorer.score_examples(examples)
quadrants = ReliabilityPlane().assign(certificates)

for row in quadrants:
    print(row)
