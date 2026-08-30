from __future__ import annotations

import json

import pandas as pd

from scripts.topic5_continuous_marked_state_h2b.build_v02_reports import (
    _primary_scoring_table,
)


def test_primary_scoring_table_exposes_eligible_and_heldout_denominators(tmp_path):
    output = tmp_path / "fits/by_subject/p1/primary"
    output.mkdir(parents=True)
    pd.DataFrame([{
        "patient_id": "p1",
        "lead_minutes": 30,
        "B_state__n_risk_sets": 2,
        "state_minus_observation_conditional_log_loss": -0.02,
        "persistent_minus_memoryless_conditional_log_loss": 0.01,
    }]).to_csv(output / "patient_median_probe_metrics.csv", index=False)
    (output / "time_label_permutation.json").write_text(json.dumps({
        "status": "COMPLETE",
        "null_q025": -0.08,
        "null_q975": 0.05,
    }), encoding="utf-8")
    table, sentence = _primary_scoring_table(tmp_path, [{
        "subject": "p1", "n": 10, "tier": "primary_chronological",
        "by_lead": {},
    }])
    assert "|p1|10|2|" in table
    assert "10 次合格发作" in sentence
    assert "2 个 held-out risk sets" in sentence
    assert "仍在置换范围内" in sentence
