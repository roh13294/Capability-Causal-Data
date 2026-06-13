import pandas as pd
import torch

from causal_reliability.certificates.calibration import add_calibrated_cis_scores
from causal_reliability.certificates.reliability import compute_causal_reliability, compute_counterfactual_instability_score, compute_shift_risk


def test_reliability_high_for_stable_low_for_unstable_logits():
    logits = torch.tensor([[4.0, 0.0], [4.0, 0.0]])
    stable_cf = torch.tensor([[[4.0, 0.0], [3.8, 0.0]], [[4.0, 0.0], [3.8, 0.0]]])
    unstable_cf = torch.tensor([[[0.0, 4.0], [0.0, 3.8]], [[0.0, 4.0], [0.0, 3.8]]])
    stable_risk, _ = compute_shift_risk(logits, stable_cf)
    unstable_risk, _ = compute_shift_risk(logits, unstable_cf)
    assert compute_causal_reliability(stable_risk).mean() > compute_causal_reliability(unstable_risk).mean()


def test_cis_emphasizes_label_flip_component():
    parts = {
        "flip_mean": torch.tensor([0.0, 1.0]),
        "margin_collapse_mean": torch.tensor([0.2, 0.2]),
        "margin_collapse_q90": torch.tensor([0.3, 0.3]),
        "js_mean": torch.tensor([0.01, 0.01]),
    }
    cis = compute_counterfactual_instability_score(parts)
    assert cis[1] > cis[0]


def test_calibrated_cis_scores_separate_validation_and_test_frames():
    validation = pd.DataFrame(
        {
            "flip_mean": [0.0, 1.0, 0.0, 1.0],
            "margin_collapse_mean": [0.1, 0.9, 0.2, 0.8],
            "margin_collapse_q90": [0.1, 0.9, 0.2, 0.8],
            "js_mean": [0.01, 0.1, 0.02, 0.09],
            "confidence_risk": [0.1, 0.4, 0.1, 0.4],
            "entropy": [0.2, 0.6, 0.2, 0.6],
            "negative_margin": [-2.0, -0.2, -1.8, -0.3],
            "failure": [0, 1, 0, 1],
        }
    )
    scored, _ = add_calibrated_cis_scores(validation, validation.copy())
    assert "calibrated_cis_score" in scored.columns
    assert scored["calibrated_cis_score"].between(0, 1).all()
