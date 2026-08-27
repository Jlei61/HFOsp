import numpy as np
import pytest

from scripts import audit_topic4_rev14_j14_controls as audit


def _patient(n_blocks=8, events_per_mode_block=7):
    ranks = []
    labels = []
    blocks = []
    for block in range(n_blocks):
        for mode in (0, 1):
            template = (
                np.asarray([0.0, 1.0, 2.0, 3.0])
                if mode == 0 else np.asarray([3.0, 2.0, 1.0, 0.0])
            )
            for event in range(events_per_mode_block):
                ranks.append(template + 0.01 * event)
                labels.append(mode)
                blocks.append(block)
    return {
        "all_ranks": np.asarray(ranks),
        "all_labels": np.asarray(labels),
        "all_blocks": np.asarray(blocks),
        "contact_names": np.asarray(["ICL1", "ICL2", "SCL1", "SCL2"]),
    }


def _score_record(values, seeds=(11, 22, 33)):
    per_seed = [
        {"objective_seed": seed, "objective": value}
        for seed, value in zip(seeds, values)
    ]
    return {
        "objective": float(np.mean(values)),
        "per_objective_seed": per_seed,
    }


def test_block_disjoint_split_and_one_event_per_block_mode():
    patient = _patient()
    model, reference = audit._block_disjoint_control_sets(patient, seed=17)

    model_blocks = set(model["audit"]["model_blocks"])
    reference_blocks = set(model["audit"]["reference_blocks"])
    assert model["audit"]["block_disjoint"] is True
    assert model_blocks.isdisjoint(reference_blocks)
    assert set(model["blocks"]).issubset(model_blocks)
    assert set(reference["all_blocks"]).issubset(reference_blocks)

    pairs = list(zip(model["blocks"].tolist(), model["labels"].tolist()))
    assert len(pairs) == len(set(pairs))
    assert np.bincount(model["labels"], minlength=2).tolist() == [4, 4]


def test_objective_seed_contract_is_exactly_three_frozen_seeds():
    assert audit._objective_seeds(20260827) == (
        20260827, 20365556, 20470586,
    )


def test_score_runs_every_frozen_objective_seed(monkeypatch):
    seen = []

    def fake_score_once(*args, objective_seed, **kwargs):
        seen.append(objective_seed)
        return {
            "objective": float(objective_seed),
            "weakest_mode_lse": 1.0,
            "mode_means": [1.0, 1.0],
            "effective_events": [2.0, 2.0],
            "occupancy_js": 0.0,
            "ambiguity": 0.0,
            "contrast_loss": 0.0,
            "support_loss": 0.0,
            "n_events": len(args[0]),
            "objective_seed": objective_seed,
        }

    monkeypatch.setattr(audit, "_score_once", fake_score_once)
    seeds = audit._objective_seeds(101)
    result = audit._score(
        np.zeros((2, 4)), np.asarray([0.0, 1.0]), patient={},
        projections=np.empty((0, 0)), calibration={}, formal={},
        objective_seeds=seeds,
    )
    assert seen == list(seeds)
    assert result["objective_seeds"] == list(seeds)
    assert [row["objective_seed"] for row in result["per_objective_seed"]] == list(seeds)


def test_every_degenerate_control_must_be_worse_at_every_seed():
    controls = {
        "block_disjoint_patient_self": _score_record([1.0, 1.0, 1.0]),
        "always_worse": _score_record([1.1, 1.2, 1.3]),
        # Its mean is worse, but one seed is better than self.
        "mean_only_worse": _score_record([0.9, 1.3, 1.3]),
    }
    ordering = audit._strict_per_seed_ordering(controls)
    assert ordering == {"always_worse": True, "mean_only_worse": False}
    assert not all(ordering.values())


def test_strict_ordering_fails_closed_on_seed_mismatch_or_duplicates():
    baseline = _score_record([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="do not align"):
        audit._strict_per_seed_ordering({
            "block_disjoint_patient_self": baseline,
            "bad": _score_record([2.0, 2.0, 2.0], seeds=(11, 22, 44)),
        })
    duplicate = _score_record([2.0, 2.0, 2.0], seeds=(11, 11, 33))
    with pytest.raises(ValueError, match="duplicate objective seed"):
        audit._strict_per_seed_ordering({
            "block_disjoint_patient_self": baseline,
            "bad": duplicate,
        })


def test_mode_evidence_requires_three_contacts_and_excludes_injected_ood(monkeypatch):
    captured = {}

    def fake_objective(*args, **kwargs):
        captured.update(kwargs)
        return {
            "objective": 1.0,
            "weakest_mode_lse": 1.0,
            "modes": {
                "0": {"mean": 1.0, "effective_events": 1.0},
                "1": {"mean": 1.0, "effective_events": 1.0},
            },
            "occupancy_js": 0.0,
            "ambiguity": 0.0,
            "contrast": {"loss": 0.0},
            "support_loss": 0.0,
        }

    monkeypatch.setattr(audit, "rev14_objective", fake_objective)
    ranks = np.asarray([
        [0.0, 1.0, np.nan, np.nan],
        [0.0, 1.0, 2.0, np.nan],
        [0.0, 1.0, 2.0, 3.0],
    ])
    audit._score_once(
        ranks, np.asarray([0.0, 0.0, 1.0]), patient={
            "all_ranks": np.zeros((1, 4)),
            "all_labels": np.asarray([0]),
            "all_blocks": np.asarray([0]),
            "contact_names": np.asarray(["ICL1", "ICL2", "SCL1", "SCL2"]),
        }, projections=np.empty((0, 0)), calibration={}, formal={
            "sample_size_per_side": 1, "draws_per_network": 1, "tau": 0.25,
        }, objective_seed=5, ood_mask=np.asarray([False, False, True]),
    )
    np.testing.assert_array_equal(
        captured["mode_evidence_mask"], [False, True, False],
    )
    assert captured["returned_families"] == 3
    assert captured["less_than_three_contact_families"] == 1


def test_ood_mask_must_align_with_events():
    with pytest.raises(ValueError, match="OOD mask does not align"):
        audit._score_once(
            np.zeros((2, 4)), np.asarray([0.0, 1.0]), patient={},
            projections=np.empty((0, 0)), calibration={}, formal={},
            objective_seed=1, ood_mask=np.asarray([False]),
        )
