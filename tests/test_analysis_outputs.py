from pathlib import Path

import pandas as pd

from causal_reliability.analysis.main_table import build_main_table, save_outputs
from causal_reliability.analysis.sts_figure import make_figure


def test_analysis_outputs_are_written(tmp_path: Path):
    out = tmp_path / "synthetic"
    out.mkdir(parents=True)
    cert = pd.DataFrame(
        {
            "pred": [0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
            "label": [0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
            "confidence": [0.9, 0.8, 0.9, 0.7, 0.6, 0.95, 0.85, 0.9, 0.55, 0.8],
            "margin": [2.0, 1.5, 2.0, 1.0, 0.4, 2.3, 1.8, 2.1, 0.2, 1.6],
            "shift_risk": [0.1, 0.9, 0.2, 0.3, 1.0, 0.1, 0.2, 0.1, 1.2, 0.2],
            "causal_reliability": [0.9, 0.4, 0.8, 0.7, 0.3, 0.95, 0.8, 0.9, 0.2, 0.8],
        }
    )
    cert["failure"] = (cert["pred"] != cert["label"]).astype(int)
    cert.to_csv(out / "certificates.csv", index=False)

    table = build_main_table(tmp_path)
    assert not table.empty
    save_outputs(table, tmp_path)
    make_figure(tmp_path)

    assert (tmp_path / "main_results_table.csv").exists()
    assert (tmp_path / "main_results_table.md").exists()
    assert (tmp_path / "main_results_summary.json").exists()
    assert (tmp_path / "sts_main_figure.png").exists()
    assert (tmp_path / "sts_main_figure.pdf").exists()
