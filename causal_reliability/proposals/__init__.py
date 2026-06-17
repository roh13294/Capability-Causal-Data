"""Automated finite-candidate proposal generation for CIC.

This package generates candidate region proposals *automatically from pixels*
instead of relying on a manually-designed candidate set tuned to a specific
shortcut family. CIC still scores a **finite** candidate set; the only change is
how that set is produced.

Scope / non-claims (enforced in the experiment summaries):
* This is automated finite-candidate proposal generation, **not** guaranteed
  open-world shortcut discovery.
* No universal robustness, deployment validation, or clinical validation claim.
"""

from causal_reliability.proposals.auto_proposals import (  # noqa: F401
    ProposalSet,
    available_generators,
    dino_boxes,
    edge_component_boxes,
    generate_proposal_sets,
    generator_availability,
    grid_boxes,
    proposal_sets_to_region_proposals,
    sam_boxes,
    saliency_boxes,
)
