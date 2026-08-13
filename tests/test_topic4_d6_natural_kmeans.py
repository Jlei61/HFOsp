import numpy as np

from src.topic4_d6_natural_kmeans import (
    best_binary_alignment,
    contact_split_folds,
    crossfit_patient_readout,
    natural_kmeans,
)


def _contract():
    return {"contacts": [
        {
            "contact_index": index,
            "shaft_id": "ICL" if index < 6 else "SCL",
            "within_shaft_order_by_shared_axis": index if index < 6 else index - 6,
        }
        for index in range(10)
    ]}


def test_contact_folds_preserve_all_contacts_and_shafts():
    left, right = contact_split_folds(_contract())
    assert set(left).isdisjoint(set(right))
    assert set(left) | set(right) == set(range(10))
    assert set(left) & set(range(6))
    assert set(left) & set(range(6, 10))
    assert set(right) & set(range(6))
    assert set(right) & set(range(6, 10))


def test_best_binary_alignment_handles_label_swap():
    result = best_binary_alignment([1, 1, 0, 0], [0, 0, 1, 1])
    assert result["purity"] == 1.0
    assert result["balanced_alignment"] == 1.0


def test_crossfit_and_natural_kmeans_recover_two_rank_modes():
    rng = np.random.default_rng(7)
    base_a = np.arange(10, dtype=float)
    base_b = base_a[::-1]
    patient = np.vstack([
        base_a + rng.normal(0, 0.05, 10) for _ in range(20)
    ] + [
        base_b + rng.normal(0, 0.05, 10) for _ in range(20)
    ])
    patient_labels = np.repeat([0, 1], 20)
    model = np.vstack([
        base_a + rng.normal(0, 0.08, 10) for _ in range(14)
    ] + [
        base_b + rng.normal(0, 0.08, 10) for _ in range(10)
    ])
    crossfit = crossfit_patient_readout(
        model, patient, patient_labels, contact_split_folds(_contract()),
    )
    assert crossfit["consensus_fraction"] > 0.9
    assert crossfit["signed_margin"] > 0.9
    result = natural_kmeans(model, crossfit["consensus_labels"])
    assert result["status"] == "OK"
    assert result["direction_purity"] > 0.9
    assert result["direction_balanced_alignment"] > 0.9
