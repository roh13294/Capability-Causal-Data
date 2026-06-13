from dataclasses import dataclass


@dataclass(frozen=True)
class ShiftRiskWeights:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5
    delta: float = 1.0
