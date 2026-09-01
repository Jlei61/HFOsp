import json

import scripts.aggregate_topic4_spatial_zm_qigk as aggregate
from scripts.aggregate_topic4_spatial_zm_qigk import (
    _compact,
    _confirmation_families,
    _rank,
)


def _row(**updates):
    row = {
        "path": "x.json",
        "mode": "hybrid",
        "full_edge": True,
        "all_checks_pass": True,
        "n_checks_pass": 7,
        "contact_fraction_consistently_rhythmic": 1.0,
        "contact_peak_mad_hz": 0.0,
        "joint_global_recruitment_duty": 1.0,
        "median_peak_power_fraction": 0.5,
        "run_role": "confirmation",
        "parameter_set_id": "locked_v1",
        "parameter_contract_sha256": "config-a",
        "seed": 1,
    }
    row.update(updates)
    return row


def test_passing_full_edge_hybrid_ranks_before_q_only_positive_control():
    hybrid = _row(path="hybrid.json")
    q_only = _row(path="q.json", mode="q_only")
    assert sorted([q_only, hybrid], key=_rank)[0] is hybrid


def test_more_global_rhythm_ranks_before_local_candidate_if_neither_passes():
    broad = _row(path="broad.json", all_checks_pass=False, n_checks_pass=5,
                 contact_fraction_consistently_rhythmic=0.9)
    local = _row(path="local.json", all_checks_pass=False, n_checks_pass=5,
                 contact_fraction_consistently_rhythmic=0.4)
    assert sorted([local, broad], key=_rank)[0] is broad


def test_confirmation_family_requires_three_unique_all_passing_seeds():
    rows = [_row(path=f"seed{seed}.json", seed=seed) for seed in (1, 2, 3)]
    family = _confirmation_families(rows, minimum_seeds=3)[0]
    assert family["eligible_multi_seed_family"] is True
    assert family["seeds"] == [1, 2, 3]

    rows[-1] = _row(path="seed3.json", seed=3, all_checks_pass=False)
    family = _confirmation_families(rows, minimum_seeds=3)[0]
    assert family["eligible_multi_seed_family"] is False


def test_discovery_runs_cannot_form_confirmation_family():
    rows = [_row(seed=seed, run_role="discovery") for seed in (1, 2, 3)]
    assert _confirmation_families(rows, minimum_seeds=3) == []


def test_confirmation_family_rejects_parameter_id_with_config_drift():
    rows = [_row(path=f"seed{seed}.json", seed=seed) for seed in (1, 2, 3)]
    rows[-1]["parameter_contract_sha256"] = "config-b"
    family = _confirmation_families(rows, minimum_seeds=3)[0]
    assert family["single_frozen_config"] is False
    assert family["eligible_multi_seed_family"] is False


def test_compact_hashes_full_edge_contract_before_return(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate, "ROOT", tmp_path)
    path = tmp_path / "seed1.json"
    path.write_text(json.dumps({
        "seed": 1,
        "candidate_id": "Joint",
        "mode": "hybrid",
        "full_edge_contract": {
            "E_to_E_dose": 1.0,
            "E_to_I_dose": 1.0,
            "learned_edges_modified": False,
        },
        "hybrid_config": {"k_q_per_ms": 0.001},
        "protocol_contract": {"duration_ms": 7000.0},
    }))
    row = _compact(path)
    assert row["full_edge"] is True
    assert len(row["parameter_contract_sha256"]) == 64
