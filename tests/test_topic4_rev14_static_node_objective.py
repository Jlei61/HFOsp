from pathlib import Path

import numpy as np
import pytest

from src.topic4_node_dualmode import fixed_projection_matrix
from src.topic4_rev14_static_node_objective import (
    _weighted_unique_sample_with_missing,
    rev14_objective,
)


def _patient():
    names = np.asarray(["ICL1", "ICL2", "SCL1", "SCL2"])
    rows, labels, blocks = [], [], []
    for block in (0, 1):
        for mode in (0, 1):
            template = (
                np.asarray([0.0, 1.0, 2.0, 3.0])
                if mode == 0 else np.asarray([3.0, 2.0, 1.0, 0.0])
            )
            for _ in range(8):
                rows.append(template.copy())
                labels.append(mode)
                blocks.append(block)
    return (
        np.asarray(rows), np.asarray(labels), np.asarray(blocks), names,
    )


def _calibration():
    return {"modes": {
        str(mode): {
            key: {"floor_q95": 1.0}
            for key in ("recruitment", "precedence", "profile", "cloud")
        }
        for mode in (0, 1)
    }}


def _score(ranks, probability_b, **counts):
    patient, labels, blocks, names = _patient()
    return rev14_objective(
        np.asarray(ranks, float), np.asarray(probability_b, float),
        patient, labels, blocks, names,
        projections=fixed_projection_matrix(8, n_directions=8, seed=5),
        calibration=_calibration(), sample_size=6, draws=8, seed=9,
        returned_families=counts.get("returned", len(ranks)),
        contact_evaluable_families=counts.get("source", len(ranks)),
        overlap_excluded_families=counts.get("overlap", 0),
        less_than_three_contact_families=counts.get("under", 0),
        mode_evidence_mask=counts.get("evidence"),
    )


def test_weighted_sample_never_duplicates_one_attractive_event():
    ranks = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    sampled = _weighted_unique_sample_with_missing(
        ranks, np.asarray([1.0, 0.0]), sample_size=6,
        rng=np.random.default_rng(1),
    )
    assert np.sum(np.all(np.isfinite(sampled), axis=1)) == 1
    assert np.sum(np.all(~np.isfinite(sampled), axis=1)) == 5


def test_zero_event_result_is_finite_and_explicitly_poor():
    result = _score(np.empty((0, 4)), np.empty(0), returned=0, source=0)
    assert np.isfinite(result["objective"])
    assert result["modes"]["0"]["mean"] == 2.0
    assert result["modes"]["1"]["mean"] == 2.0
    assert result["occupancy_js"] == pytest.approx(np.log(2.0))
    assert result["contrast"]["loss"] == 1.0


def test_single_mode_cannot_hide_missing_other_mode():
    patient, labels, _, _ = _patient()
    mode_zero = patient[labels == 0][:8]
    result = _score(mode_zero, np.zeros(len(mode_zero)))
    assert result["modes"]["1"]["mean"] == 2.0
    assert result["contrast"]["loss"] == 1.0
    assert result["support"]["effective_events"][1] == 0.0


def test_all_ambiguous_events_have_zero_mode_support():
    patient, _, _, _ = _patient()
    result = _score(patient[:16], np.full(16, 0.5))
    assert result["modes"]["0"]["effective_events"] == 0.0
    assert result["modes"]["1"]["effective_events"] == 0.0
    assert result["modes"]["0"]["mean"] == 2.0
    assert result["modes"]["1"]["mean"] == 2.0


def test_sparse_or_ood_events_enter_distances_but_not_mode_evidence_support():
    patient, labels, _, _ = _patient()
    model = patient[:16].copy()
    evidence = np.zeros(len(model), dtype=bool)
    result = _score(
        model, labels[:16].astype(float), evidence=evidence, under=len(model),
    )
    assert result["modes"]["0"]["effective_events"] == 0.0
    assert result["modes"]["1"]["effective_events"] == 0.0
    assert result["modes"]["0"]["all_contact_primary_effective_events"] > 0.0
    assert result["modes"]["1"]["all_contact_primary_effective_events"] > 0.0
    assert result["modes"]["0"]["raw_draw_mean"] is not None
    assert result["modes"]["1"]["raw_draw_mean"] is not None


def test_one_repeated_attractive_event_is_worse_than_two_supported_modes():
    patient, labels, _, _ = _patient()
    supported = _score(patient[:16], labels[:16].astype(float))
    repeated = np.repeat(patient[[0]], 16, axis=0)
    repeated_score = _score(repeated, np.r_[np.zeros(8), np.ones(8)])
    assert repeated_score["objective"] > supported["objective"]


def test_scl_censoring_worsens_shaft_balanced_recruitment():
    patient, labels, _, _ = _patient()
    model = patient[:16].copy()
    probabilities = labels[:16].astype(float)
    intact = _score(model, probabilities)
    censored = model.copy()
    censored[:, 2:] = np.nan
    lost = _score(censored, probabilities)
    assert lost["modes"]["0"]["recruitment"] > intact["modes"]["0"]["recruitment"]
    assert lost["modes"]["1"]["recruitment"] > intact["modes"]["1"]["recruitment"]


def test_support_and_overlap_are_continuous_costs_not_drop_rules():
    patient, labels, _, _ = _patient()
    model = patient[:16]
    probability = labels[:16].astype(float)
    clean = _score(model, probability, returned=16, source=16)
    poor = _score(
        model, probability, returned=24, source=20, overlap=4, under=6,
    )
    assert poor["objective"] > clean["objective"]
    assert poor["overlap_fraction"] == pytest.approx(0.2)
    assert poor["support"]["contact_non_evaluable_fraction"] == pytest.approx(4 / 24)


def test_invalid_support_counts_fail_closed():
    with pytest.raises(ValueError, match="event support counts"):
        _score(np.empty((0, 4)), np.empty(0), returned=2, source=3)


def test_objective_module_has_no_kmeans_or_heldout_dependency():
    source = (Path(__file__).resolve().parents[1]
              / "src/topic4_rev14_static_node_objective.py").read_text().lower()
    assert "import sklearn" not in source
    assert "natural_kmeans(" not in source
    assert "patient_heldout" not in source
