from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class ReliabilityPlane:
    confidence_threshold: float = 0.8
    stability_threshold: float = 0.5

    def assign_one(self, confidence: float, stability_score: float) -> str:
        high_confidence = float(confidence) >= self.confidence_threshold
        high_stability = float(stability_score) >= self.stability_threshold
        if high_confidence and high_stability:
            return "accept"
        if not high_confidence and high_stability:
            return "review uncertainty"
        if high_confidence and not high_stability:
            return "human review / shortcut audit"
        return "stress-test further"

    def quadrant_name(self, confidence: float, stability_score: float) -> str:
        action = self.assign_one(confidence, stability_score)
        return {
            "accept": "Reliable prediction",
            "review uncertainty": "Uncertain but causally stable",
            "stress-test further": "Generally fragile",
            "human review / shortcut audit": "Dangerous shortcut reliance",
        }[action]

    def assign(self, certificates: Iterable[dict]) -> list[dict]:
        rows = []
        for cert in certificates:
            row = dict(cert)
            row["quadrant"] = self.quadrant_name(row["confidence"], row["stability_score"])
            row["recommended_action"] = self.assign_one(row["confidence"], row["stability_score"])
            rows.append(row)
        return rows
