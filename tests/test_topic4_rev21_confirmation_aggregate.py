from scripts.aggregate_topic4_rev21_confirmation import confirmation_robustness


def _cells(counts):
    rows = []
    for topology, count in enumerate(counts, start=1):
        for dynamics in range(4):
            eligible = dynamics < count
            rows.append({
                "topology_seed": topology,
                "dynamics_seed": dynamics,
                "model_ictal": {
                    "status": ("MODEL_ICTAL_ELIGIBLE_REV21" if eligible
                               else "MODEL_ICTAL_INELIGIBLE_REV21"),
                },
            })
    return rows


def test_confirmation_requires_eight_cells_across_two_topologies():
    passed = confirmation_robustness(_cells([4, 4, 0]))
    assert passed["eligible_cells"] == 8
    assert passed["pass"]


def test_confirmation_rejects_single_topology_and_low_total():
    concentrated = confirmation_robustness(_cells([4, 1, 1]))
    assert not concentrated["pass"]
    low_total = confirmation_robustness(_cells([3, 2, 2]))
    assert low_total["topologies_with_at_least_two_of_four"] == 3
    assert not low_total["pass"]
