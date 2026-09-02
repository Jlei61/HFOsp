from __future__ import annotations

import pytest

from scripts.publish_group_event_state_v033_evaluator_contract import build_contract
from src.topic5_group_event_state.v033_evaluator import canonical as C


def _power(fp=0.0):
    curves = []
    for view in ("count_profile", "grammar"):
        curves.append({
            "view": view,
            "cells": [
                {"kind": "D0", "n_replicates": 1,
                 "false_positive_rate_by_level": {"0": fp, "1": 0.0, "2": 0.0}},
                {"kind": "D3", "n_replicates": 1,
                 "power_by_level": {"0": 1.0, "1": 1.0, "2": 1.0},
                 "gain_by_level": {}},
            ],
        })
    return {"preset": "sentinel", "source_commit": "a" * 40, "human_targets_used": False,
            "sealed_partition_opened": False, "curves": curves}


def _boundary():
    return {
        "sealed_partition_opened": False,
        "source_commit": "abc",
        "cohort": {
            "all_kept_equals_in_target_segments": True,
            "all_matches_v032_eligibility": True,
            "all_state_events_exclude_seizure_and_postictal": True,
            "n_subjects": 27, "n_ok": 27,
        },
    }


def _discrepancy():
    return {
        "canonical_schema_version": C.SCHEMA_VERSION,
        "sealed_partition_opened": False,
        "audit": {"all_published_reproduced": True},
    }


def test_contract_separates_evaluator_assay_and_human_evidence():
    out = build_contract(power=_power(), boundary=_boundary(), discrepancy=_discrepancy(),
                         training_commit="def", evaluator_test_count=64, joint_test_count=130)
    assert out["count_profile"]["target_bins_seconds"] == [[0, 300], [300, 900], [900, 1800]]
    assert out["count_profile"]["primary_dispersion"].startswith("one H-fitted")
    assert out["implementation"]["training_commit"] == "def"
    assert out["assay"]["status"] == "SENTINEL_PASS_D0_NO_FALSE_POSITIVES"
    assert out["assay"]["source_commit"] == "a" * 40
    assert out["phase_contract"]["reported_gain_uses"] == "dev_test_only"
    assert out["phase_contract"]["estimability_uses"] == "dev_test_only"
    assert out["human_scientific_conclusion"] == "NONE"
    assert out["sealed_partition_opened"] is False


def test_contract_fails_closed_on_false_positive_or_bad_boundary():
    with pytest.raises(ValueError, match="false positive"):
        build_contract(power=_power(fp=1.0), boundary=_boundary(), discrepancy=_discrepancy(),
                       training_commit="def", evaluator_test_count=64, joint_test_count=130)
    boundary = _boundary()
    boundary["cohort"]["all_state_events_exclude_seizure_and_postictal"] = False
    with pytest.raises(ValueError, match="boundary audit failed"):
        build_contract(power=_power(), boundary=boundary, discrepancy=_discrepancy(),
                       training_commit="def", evaluator_test_count=64, joint_test_count=130)
