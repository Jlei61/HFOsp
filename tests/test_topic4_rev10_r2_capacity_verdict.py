from scripts.audit_topic4_rev10_r2_shared_route_capacity import (
    fit_library_oracle,
)


def _summary():
    def support(a, b):
        return {"mode_conditioned_joint_support": {
            "A": {"n_joint_in_distribution": a},
            "B": {"n_joint_in_distribution": b},
        }}
    return {
        "network_seeds": [1, 2],
        "candidate_details": {
            "first": {"by_seed": {"1": support(1, 2), "2": support(0, 2)}},
            "second": {"by_seed": {"1": support(0, 3), "2": support(1, 3)}},
            "shared": {"by_seed": {"1": support(1, 4), "2": support(2, 4)}},
        },
    }


def test_fit_library_oracle_requires_same_candidate_across_networks():
    audit = fit_library_oracle(_summary())
    assert audit["by_seed"]["1"]["n_candidates_with_mode_A"] == 2
    assert audit["by_seed"]["2"]["n_candidates_with_mode_A"] == 2
    assert audit["shared_mode_A_candidate_ids"] == ["shared"]


def test_fit_library_oracle_does_not_confuse_per_network_oracle_with_shared():
    summary = _summary()
    summary["candidate_details"].pop("shared")
    audit = fit_library_oracle(summary)
    assert all(row["n_candidates_with_mode_A"] == 1 for row in audit["by_seed"].values())
    assert audit["n_shared_mode_A_candidates"] == 0
