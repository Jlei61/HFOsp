from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = Path("/home/honglab/leijiaxin/HFOsp")
SCRIPT = ROOT / "scripts/run_topic5_rnn_full_cohort_field_transfer_v0_1.py"


def _module():
    spec = importlib.util.spec_from_file_location("topic5_rnn_full_cohort_transfer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not CANONICAL.exists(), reason="canonical workspace unavailable")
def test_denominator_contract_is_34_and_fig3_17_167():
    audit = _module().audit_inputs(CANONICAL)
    assert audit["n_interictal_subjects"] == 34
    assert audit["n_model_seeds_per_subject"] == 3
    assert audit["n_primary_ictal_subjects"] == 17
    assert audit["n_primary_ictal_seizures"] == 167


@pytest.mark.skipif(not CANONICAL.exists(), reason="canonical workspace unavailable")
def test_primary_inventory_never_routes_through_legacy_outer_cache():
    audit = _module().audit_inputs(CANONICAL)
    assert audit["legacy_outer_cache_reads"] == 0
    assert all("outer_" not in path for path in audit["source_paths"].values())
    assert audit["target_used_for_training_or_selection"] is False


def test_score_fails_closed_without_pre_target_field_manifest(tmp_path):
    with pytest.raises(RuntimeError, match="target_free_model_field_manifest_missing"):
        _module().score(CANONICAL, tmp_path)


def test_mode_to_ab_assignment_uses_frozen_interictal_ranks():
    module = _module()
    names = np.asarray(["A", "B", "C", "D"])
    train_templates = np.asarray([[3, 2, 1, 0], [0, 1, 2, 3]], float)
    record = {"interictal_field": {
        "contact_order": names.tolist(),
        "rank_a": [0, 1, 2, 3],
        "rank_b": [3, 2, 1, 0],
    }}
    mapping, correlation = module._mode_to_ab_mapping(train_templates, names, record)
    assert mapping == {"a": 1, "b": 0}
    assert correlation[1, 0] > 0.99
    assert correlation[0, 1] > 0.99
