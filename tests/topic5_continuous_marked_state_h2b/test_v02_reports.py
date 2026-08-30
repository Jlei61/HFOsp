from __future__ import annotations

import json

import pandas as pd

from scripts.topic5_continuous_marked_state_h2b.build_v02_reports import (
    _effect_table,
    _h1_stratum_table,
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


def test_main_effect_table_does_not_silently_mix_h1_strata():
    frame = pd.DataFrame([
        {
            "stratum": "all_checkpoint_available",
            "evaluation_tier": "descriptive_case_series",
            "lead_minutes": 30,
            "effect": "state_minus_observation_conditional_log_loss",
            "n_patients": 4, "n_favourable": 4,
            "patient_median_effect": -0.1,
            "two_sided_exact_sign_p": 0.125,
        },
        {
            "stratum": "h1_stable_stratum",
            "evaluation_tier": "descriptive_case_series",
            "lead_minutes": 30,
            "effect": "state_minus_observation_conditional_log_loss",
            "n_patients": 1, "n_favourable": 1,
            "patient_median_effect": -0.2,
            "two_sided_exact_sign_p": 1.0,
        },
    ])

    main = _effect_table(frame)
    stratified = _h1_stratum_table(frame)

    assert "|4|4/4|" in main
    assert "|1|1/1|" not in main
    assert "H1-stable" in stratified
    assert "|1|1/1|" in stratified
