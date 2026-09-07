import copy
import json

import numpy as np

from scripts import analyze_topic4_multievent_distribution_v2_1 as de
from scripts.run_topic4_multievent_distribution_v2_1_followup import (
    _update_population,
)
from src.topic4_multievent_condition_identity_v2_1 import condition_key


def _candidate(candidate_id, tau=18.0):
    return {
        "candidate_id": candidate_id,
        "population": 0,
        "domain": "whole_sheet",
        "node_field": {
            "field_type": "manual_dual_core_budget_matched",
            "centers_mm": [[1.0, 2.0], [10.0, 11.0]],
            "target_count": 1499,
        },
        "node_mapping": {"signed_depth_shrinkage": 1.0, "node_gain": 1.0},
        "mechanisms": {
            "g_EE": 0.0, "g_EtoI": 0.0, "Z_M": "off",
            "ellipse_angle_deg": -20.0, "ellipse_aspect_ratio": 2.0,
            "ellipse_reference_angle_deg": -20.0,
            "ellipse_reference_aspect_ratio": 2.0,
        },
        "dynamic_parameters": {
            "E_to_E_weight_scale": 1.0, "E_to_I_weight_scale": 1.0,
            "I_to_E_weight_scale": 1.0, "I_to_I_weight_scale": 1.0,
            "tau_d_GABA_ms": tau,
        },
    }


def _score(candidate_id, loss):
    return {
        "candidate_id": candidate_id,
        "ranking_eligible": loss is not None,
        "score": {"loss_off": loss},
    }


def test_condition_identity_includes_time_constant_but_not_proposal_id():
    baseline = _candidate("baseline", tau=18.0)
    alias = copy.deepcopy(baseline)
    alias["candidate_id"] = "alias"
    high_tau = _candidate("high_tau", tau=24.0)
    assert condition_key(baseline) == condition_key(alias)
    assert condition_key(baseline) != condition_key(high_tau)


def test_g2a_replacement_changes_population_seen_by_next_batch(tmp_path):
    parent = _candidate("parent")
    untouched = _candidate("untouched")
    untouched["population"] = 1
    child = _candidate("child")
    report = {"candidates": [_score("parent", 2.0), _score("child", 1.0)]}
    plan = {"batch": "A", "plans": [{
        "population": 0, "target_index": 0,
        "target_candidate_id": "parent", "candidate_id": "child",
    }]}
    path = tmp_path / "updated.json"
    result = _update_population(
        [parent, untouched], report, plan, [child], output_path=path,
    )
    assert result["replacement_affects_next_batch"] is True
    assert [row["candidate_id"] for row in result["candidates"]] == [
        "child", "untouched",
    ]
    assert json.loads(path.read_text())["updates"][0]["replace"] is True


def _population():
    rows = []
    for population in (0, 1):
        for index in range(24):
            row = _candidate(f"p{population}_{index:02d}")
            row["population"] = population
            offset = 0.08 * index + 0.02 * population
            row["node_field"]["centers_mm"] = [
                [2.0 + offset, 3.0 + offset],
                [14.0 + offset, 10.0 + offset],
            ]
            row["node_mapping"]["node_gain"] = 0.6 + 0.02 * index
            row["dynamic_parameters"].update({
                "E_to_E_weight_scale": 0.82 + 0.01 * index,
                "E_to_I_weight_scale": 0.84 + 0.01 * index,
                "I_to_E_weight_scale": 0.83 + 0.01 * index,
                "tau_d_GABA_ms": 13.0 + 0.5 * index,
            })
            row["mechanisms"]["ellipse_angle_deg"] = (
                de.base.THETA - 20.0 + 1.5 * index
            )
            row["mechanisms"]["ellipse_aspect_ratio"] = 1.1 + 0.06 * index
            rows.append(row)
    return rows


def _write_freeze_inputs(folder, candidates, report):
    folder.mkdir(parents=True)
    de.base.write(folder / "objective_qualification.json", {
        "objective_version": de.OBJECTIVE_VERSION,
    })
    de.base.write(folder / "scores.json", report)
    de.base.write(folder / "population.json", {"candidates": candidates})


def test_second_batch_parameters_depend_on_post_a_population(monkeypatch, tmp_path):
    monkeypatch.setattr(de, "review_offline_scans", lambda: {"G2_allowed": True})
    monkeypatch.setattr(de.base, "positions", lambda: np.zeros((2, 2)))
    monkeypatch.setattr(de, "geometry_allowed", lambda *args, **kwargs: True)
    monkeypatch.setattr(de, "field_descriptor", lambda centers, target_count: {
        "field_type": "manual_dual_core_budget_matched",
        "centers_mm": np.asarray(centers).tolist(),
        "target_count": target_count,
    })
    monkeypatch.setattr(de, "audit_geometry", lambda *args, **kwargs: {
        "selected_count": 1499,
    })
    parents = _population()
    g1 = {"candidates": [
        {**_score(row["candidate_id"], 10.0 + index),
         "population": row["population"]}
        for index, row in enumerate(parents)
    ]}

    adaptive = tmp_path / "adaptive"
    _write_freeze_inputs(adaptive, parents, g1)
    monkeypatch.setattr(de, "OUT", adaptive)
    plan_a, offspring_a = de.freeze_de_batch(
        g1, parents, batch="A", parent_score_path=adaptive / "scores.json",
        parent_population_path=adaptive / "population.json",
    )
    resumed_plan_a, resumed_offspring_a = de.freeze_de_batch(
        g1, parents, batch="A", parent_score_path=adaptive / "scores.json",
        parent_population_path=adaptive / "population.json",
    )
    assert resumed_plan_a == plan_a
    assert resumed_offspring_a == offspring_a
    scored_a = {"candidates": g1["candidates"] + [
        {**_score(row["candidate_id"], -1.0), "population": row["population"]}
        for row in offspring_a
    ]}
    de.base.write(adaptive / "pool.json", scored_a)
    updated = _update_population(
        parents, scored_a, plan_a, offspring_a,
        output_path=adaptive / "updated.json",
    )
    _, adaptive_b = de.freeze_de_batch(
        scored_a, updated["candidates"], batch="B",
        parent_score_path=adaptive / "pool.json",
        parent_population_path=adaptive / "updated.json",
    )

    control = tmp_path / "control"
    _write_freeze_inputs(control, parents, scored_a)
    monkeypatch.setattr(de, "OUT", control)
    _, control_b = de.freeze_de_batch(
        scored_a, parents, batch="B", parent_score_path=control / "scores.json",
        parent_population_path=control / "population.json",
    )
    adaptive_vectors = [de.vector_from_candidate(row) for row in adaptive_b]
    control_vectors = [de.vector_from_candidate(row) for row in control_b]
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(adaptive_vectors, control_vectors)
    )
