"""Synthetic and mutation tests for the seed-specific Phase-C modal layer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.adjudicate_topic4_zm_phasec as ADJ
import scripts.analyze_topic4_zm_phasec_modal as ANALYZE
import src.topic4_zm_phasec_modal as M
import src.topic4_zm_phasec_verdict as V


def _observables(seed, *, n=260, periodic=False):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4), indexing="ij")
    a = xx.ravel()
    b = yy.ravel()
    if periodic:
        phase = np.arange(n) * 2 * np.pi / 10.0 + 0.1 * seed
        c1, c2 = np.sin(phase), np.cos(phase)
    else:
        latent = np.zeros((n, 2))
        latent[0] = rng.normal(size=2)
        operator = np.asarray([[0.92, 0.18], [-0.04, 0.84]])
        for index in range(n - 1):
            latent[index + 1] = operator @ latent[index] + rng.normal(
                scale=0.03, size=2
            )
        c1, c2 = latent.T
    E = 20 + np.outer(c1, a) + np.outer(c2, b)
    I = 15 + 0.7 * np.outer(c1, b) - 0.4 * np.outer(c2, a)
    return {
        "E_rate_grid": E.reshape(n, 4, 4),
        "I_rate_grid": I.reshape(n, 4, 4),
    }


def _runs(periodic=False):
    return [
        {
            "noise": "noise_replay",
            "role": "fit",
            "observables": _observables(1, periodic=periodic),
        },
        {
            "noise": "noise_repeat",
            "role": "fit",
            "observables": _observables(3, periodic=periodic),
        },
        {
            "noise": "noise_independent",
            "role": "noise_heldout",
            "observables": _observables(8, periodic=periodic),
        },
    ]


def _route(seed=1, phenotype="balanced_AI_tonic_candidate"):
    routed = M.phenotype_route(phenotype)
    return {
        "seed": seed,
        "input_phenotype": phenotype,
        "output_phenotype": phenotype,
        "phenotype_source": "C0_seed",
        "c0_class": phenotype,
        "c1_class": None,
        **routed,
        "modal_override_allowed": False,
    }


def test_ai_operator_uses_noise_and_time_heldout_without_leakage():
    route = _route()
    runs = _runs()
    first = M.analyze_seed(
        route, runs, bin_ms=2.0, pathology_axis_deg=0.0
    )
    assert first["status"] == "identified"
    assert first["evidence_type"] == "observational_low_rank_DMD"
    assert first["fit_noise_ids"] == ["noise_repeat", "noise_replay"]
    assert first["noise_heldout_ids"] == ["noise_independent"]

    mutated = _runs()
    rng = np.random.default_rng(99)
    mutated[-1]["observables"] = {
        key: rng.normal(size=value.shape)
        for key, value in mutated[-1]["observables"].items()
    }
    second = M.analyze_seed(
        route, mutated, bin_ms=2.0, pathology_axis_deg=0.0
    )
    np.testing.assert_allclose(
        first["leading_spatial_mode"], second["leading_spatial_mode"]
    )
    assert first["training_relative_error"] == second["training_relative_error"]
    assert (
        first["noise_heldout"]["heldout_relative_error"]
        != second["noise_heldout"]["heldout_relative_error"]
    )


def test_periodic_is_stroboscopic_and_cannot_be_misrouted_to_fixed_eigen():
    route = _route(phenotype="periodic_non_tonic_carrier")
    out = M.analyze_seed(
        route,
        _runs(periodic=True),
        bin_ms=2.0,
        pathology_axis_deg=20.0,
        period_ms=20.0,
    )
    assert out["status"] == "identified"
    assert out["operator_tool"] == "stroboscopic_floquet"
    assert out["lag_bins"] == 10
    assert out["heldout_cycles_required"] is True

    bad = dict(route)
    bad["operator_tool"] = "eigen"
    failed = M.analyze_seed(
        bad,
        _runs(periodic=True),
        bin_ms=2.0,
        pathology_axis_deg=20.0,
        period_ms=20.0,
    )
    assert failed["status"] == M.NOT_IDENTIFIABLE
    assert "periodic" in failed["reason"]


def test_missing_heldout_noise_fails_closed_and_saturated_builds_no_mode():
    failed = M.analyze_seed(
        _route(), _runs()[:2], bin_ms=2.0, pathology_axis_deg=0.0
    )
    assert failed["status"] == M.NOT_IDENTIFIABLE
    saturated = M.analyze_seed(
        _route(phenotype="refractory_saturated_branch"),
        [],
        bin_ms=2.0,
        pathology_axis_deg=0.0,
        locked_sensitivity={
            "local_gain_hz_per_mV": 2.1,
            "refractory_fraction": 0.91,
            "source_sha256": "a" * 64,
        },
    )
    assert saturated["status"] == "summarized_without_operator"
    assert saturated["seizure_mode_claim"] is False
    assert "leading_spatial_mode" not in saturated


def test_cross_seed_comparison_never_pools_eigenvalues_and_marks_disagreement():
    one = M.analyze_seed(
        _route(seed=1), _runs(), bin_ms=2.0, pathology_axis_deg=0.0
    )
    three = M.analyze_seed(
        _route(seed=3), _runs(), bin_ms=2.0, pathology_axis_deg=0.0
    )
    aggregate = M.aggregate_seed_modal([one, three])
    assert aggregate["eigenvalue_pooling"] == "forbidden_not_performed"
    assert "pooled_eigenvalues" not in aggregate
    assert len(aggregate["same_class_spatial_comparisons"]) == 1
    with pytest.raises(ValueError, match="duplicate seed"):
        M.aggregate_seed_modal([one, dict(one)])

    periodic = M.analyze_seed(
        _route(seed=4, phenotype="periodic_non_tonic_carrier"),
        _runs(periodic=True),
        bin_ms=2.0,
        pathology_axis_deg=0.0,
        period_ms=20.0,
    )
    mixed = M.aggregate_seed_modal([one, periodic])
    assert mixed["class_disagreement"] is True
    assert mixed["same_class_spatial_comparisons"] == []


def test_modal_route_cannot_override_c0_c1_phenotype():
    with pytest.raises(ValueError, match="does not equal accepted"):
        M.derive_seed_route(
            seed=1,
            c0_seed_row={"klass": "balanced_AI_tonic_candidate"},
            c1_cell=None,
            selected_phenotype="periodic_non_tonic_carrier",
        )
    bad = _route()
    bad["output_phenotype"] = "periodic_non_tonic_carrier"
    with pytest.raises(ValueError, match="cannot override"):
        M.analyze_seed(
            bad, _runs(), bin_ms=2.0, pathology_axis_deg=0.0
        )


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_analyzer_rejects_mutated_representative_npz(tmp_path, monkeypatch):
    monkeypatch.setattr(ANALYZE.PCC, "validate_manifest", lambda _: None)
    producer = tmp_path / "producer.py"
    producer.write_text("# producer\n", encoding="utf-8")
    phasec = {
        "manifest_sha256": "a" * 64,
        "production_authorized": True,
        "provenance": {
            "producer_file_sha256": {str(producer): _sha(producer)}
        },
    }
    c0_gate = {"schema": "c0_gate", "verdict": "balanced_AI_tonic_candidate_supported"}
    c0_native = {
        "schema": "c0",
        "seed_rows": [{"seed": 1, "klass": "balanced_AI_tonic_candidate"}],
    }
    c1_gate = {"schema": "c1_gate", "verdict": "no_maturation"}
    c1_native = {"schema": "c1", "cells": []}
    values = {
        "phasec_manifest": phasec,
        "c0_resolution_gate": c0_gate,
        "c0_native_summary": c0_native,
        "c1_resolution_gate": c1_gate,
        "c1_native_summary": c1_native,
    }
    paths = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        _write_json(path, value)
        paths[name] = path

    runs = []
    for index, (noise, role) in enumerate((
        ("noise_replay", "fit"),
        ("noise_repeat", "fit"),
        ("noise_independent", "noise_heldout"),
    )):
        obs_path = tmp_path / f"obs{index}.npz"
        arrays = _observables(index + 1)
        np.savez(
            obs_path,
            phasec1_observables_schema=np.asarray("schema"),
            bin_ms=np.asarray(2.0),
            **arrays,
        )
        part_path = tmp_path / f"part{index}.json"
        part = {
            "status": "complete",
            "seed": 1,
            "cell_id": None,
            "phase": "rising",
            "noise": noise,
            "observables_path": str(obs_path),
            "observables_sha256": _sha(obs_path),
            "runtime_provenance": {"producer_sha256": {"runner.py": "b" * 64}},
        }
        _write_json(part_path, part)
        runs.append({
            "phase": "rising",
            "noise": noise,
            "role": role,
            "part_path": str(part_path),
            "part_file_sha256": _sha(part_path),
            "part_semantic_sha256": ANALYZE._artifact_semantic_sha(part),
            "observables_path": str(obs_path),
            "observables_file_sha256": _sha(obs_path),
            "observables_semantic_sha256": ANALYZE._semantic_npz_sha(
                obs_path
            ),
            "runtime_producer_file_sha256": {
                "runner.py": "b" * 64
            },
        })
    selection_body = {
        "schema": ANALYZE.SELECTION_SCHEMA,
        "selection_is_closed": True,
        "selection_rule": ANALYZE.SELECTION_RULE,
        "phasec_manifest_sha256": phasec["manifest_sha256"],
        "heldout_contract": M.HELDOUT_CONTRACT,
        "inputs": {
            name: {
                "path": str(path),
                "file_sha256": _sha(path),
                "semantic_sha256": ANALYZE._artifact_semantic_sha(value),
                "manifest_sha256": value.get("manifest_sha256"),
                "parent_phasec_manifest_sha256": phasec["manifest_sha256"],
            }
            for (name, path), value in zip(paths.items(), values.values())
        },
        "producer_file_sha256": ANALYZE._selection_producer_locks(),
        "seeds": {
            "1": {
                "phenotype": "balanced_AI_tonic_candidate",
                "phenotype_source": "C0_seed",
                "route": "AI_observational_DMD",
                "tier": None,
                "cell_id": None,
                "pathology_axis_deg": 0.0,
                "runs": runs,
            }
        },
    }
    selection = {
        **selection_body,
        "manifest_sha256": M.canonical_sha256(selection_body),
    }
    selection_path = tmp_path / "selection.json"
    _write_json(selection_path, selection)
    out = ANALYZE.analyze(
        phasec_manifest_path=paths["phasec_manifest"],
        c0_gate_path=paths["c0_resolution_gate"],
        c0_native_path=paths["c0_native_summary"],
        c1_gate_path=paths["c1_resolution_gate"],
        c1_native_path=paths["c1_native_summary"],
        representatives_path=selection_path,
    )
    assert out["seed_results"][0]["status"] == "identified"
    with np.load(runs[-1]["observables_path"], allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    arrays["E_rate_grid"] = arrays["E_rate_grid"] + 1.0
    np.savez(runs[-1]["observables_path"], **arrays)
    with pytest.raises(ValueError, match="observables file SHA"):
        ANALYZE.analyze(
            phasec_manifest_path=paths["phasec_manifest"],
            c0_gate_path=paths["c0_resolution_gate"],
            c0_native_path=paths["c0_native_summary"],
            c1_gate_path=paths["c1_resolution_gate"],
            c1_native_path=paths["c1_native_summary"],
            representatives_path=selection_path,
        )
    # An attacker who re-locks the changed bytes and the parent part but
    # leaves the original semantic NPZ identity must still fail closed.
    changed_selection = json.loads(selection_path.read_text())
    changed_run = changed_selection["seeds"]["1"]["runs"][-1]
    part_path = Path(changed_run["part_path"])
    part = json.loads(part_path.read_text())
    part["observables_sha256"] = _sha(Path(changed_run["observables_path"]))
    _write_json(part_path, part)
    changed_run["observables_file_sha256"] = part["observables_sha256"]
    changed_run["part_file_sha256"] = _sha(part_path)
    changed_run["part_semantic_sha256"] = (
        ANALYZE._artifact_semantic_sha(part)
    )
    changed_selection["manifest_sha256"] = M.canonical_sha256({
        key: value for key, value in changed_selection.items()
        if key != "manifest_sha256"
    })
    _write_json(selection_path, changed_selection)
    with pytest.raises(ValueError, match="observables semantic SHA"):
        ANALYZE.analyze(
            phasec_manifest_path=paths["phasec_manifest"],
            c0_gate_path=paths["c0_resolution_gate"],
            c0_native_path=paths["c0_native_summary"],
            c1_gate_path=paths["c1_resolution_gate"],
            c1_native_path=paths["c1_native_summary"],
            representatives_path=selection_path,
        )


def test_write_once_builder_is_deterministic_and_c0_ai_does_not_need_c1_positive(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ANALYZE.PCC, "validate_manifest", lambda _: None)
    producer = tmp_path / "phasec_producer.py"
    producer.write_text("# producer\n", encoding="utf-8")
    phasec = {
        "manifest_sha256": "a" * 64,
        "production_authorized": True,
        "provenance": {
            "producer_file_sha256": {
                str(producer): _sha(producer),
            }
        },
    }
    c0_gate = {
        "schema": ANALYZE.C0_GATE_SCHEMA,
        "verdict": "balanced_AI_tonic_candidate_supported",
        "resolution_gate": "passed",
    }
    c0_native = {
        "schema": "c0",
        "seed_rows": [{
            "seed": 1,
            "klass": "balanced_AI_tonic_candidate",
            "hierarchical_ci": {},
        }],
    }
    c1_gate = {
        "schema": ANALYZE.C1_GATE_SCHEMA,
        "verdict": "no_maturation_in_tested_primary_neighbourhood",
        "resolution_gate": "not_required_without_native_window",
    }
    cells = []
    for tier, cell_id in (
        ("secondary_shell", "00_shell"),
        ("primary_convex", "z_primary"),
        ("primary_convex", "a_primary"),
    ):
        run_rows = []
        for phase in ("rising", "peak"):
            for noise in ("noise_z", "noise_a", "noise_m"):
                obs_path = tmp_path / f"{cell_id}_{phase}_{noise}.npz"
                np.savez(
                    obs_path,
                    phasec1_observables_schema=np.asarray("schema"),
                    bin_ms=np.asarray(2.0),
                    **_observables(abs(hash((cell_id, phase, noise))) % 999),
                )
                part_path = tmp_path / f"{cell_id}_{phase}_{noise}.json"
                part = {
                    "status": "complete",
                    "seed": 1,
                    "cell_id": cell_id,
                    "phase": phase,
                    "noise": noise,
                    "observables_path": str(obs_path),
                    "observables_sha256": _sha(obs_path),
                    "runtime_provenance": {
                        "producer_sha256": {str(producer): _sha(producer)}
                    },
                }
                _write_json(part_path, part)
                run_rows.append({
                    "status": "complete",
                    "phase": phase,
                    "noise": noise,
                    "part_path": str(part_path),
                    "part_sha256": _sha(part_path),
                })
        cells.append({
            "seed": 1,
            "tier": tier,
            "cell_id": cell_id,
            # Deliberately opposite to C0: this is an observation source,
            # not authority to reroute the accepted C0 identity.
            "cell_class": "refractory_saturated",
            "status": "complete",
            "run_rows": run_rows,
        })
    c1_native = {
        "schema": "c1",
        "cells": cells,
        "primary_adjudication": {"status": "no_window"},
        "secondary_shell_adjudication": {"status": "no_window"},
    }
    values = {
        "phasec_manifest": phasec,
        "c0_resolution_gate": c0_gate,
        "c0_native_summary": c0_native,
        "c1_resolution_gate": c1_gate,
        "c1_native_summary": c1_native,
    }
    paths = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        _write_json(path, value)
        paths[name] = path
    kwargs = {
        "phasec_manifest_path": paths["phasec_manifest"],
        "c0_gate_path": paths["c0_resolution_gate"],
        "c0_native_path": paths["c0_native_summary"],
        "c1_gate_path": paths["c1_resolution_gate"],
        "c1_native_path": paths["c1_native_summary"],
    }
    first = ANALYZE.build_representative_manifest(**kwargs)
    second = ANALYZE.build_representative_manifest(**kwargs)
    assert first == second
    selected = first["seeds"]["1"]
    assert selected["phenotype_source"] == "C0_seed"
    assert selected["phenotype"] == "balanced_AI_tonic_candidate"
    assert selected["route"] == "AI_observational_DMD"
    assert selected["tier"] == "primary_convex"
    assert selected["cell_id"] == "a_primary"
    assert {row["phase"] for row in selected["runs"]} == {"peak"}
    assert [row["noise"] for row in selected["runs"]] == [
        "noise_a", "noise_m", "noise_z"
    ]
    assert all(
        row["part_semantic_sha256"]
        and row["observables_semantic_sha256"]
        and row["runtime_producer_file_sha256"]
        for row in selected["runs"]
    )
    output = tmp_path / "selection.json"
    ANALYZE._atomic_write(output, first)
    ANALYZE._atomic_write(output, first)
    changed = json.loads(json.dumps(first))
    changed["seeds"]["1"]["cell_id"] = "z_primary"
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        ANALYZE._atomic_write(output, changed)

    analyzed = ANALYZE.analyze(
        representatives_path=output,
        **kwargs,
    )
    assert analyzed["seed_results"][0]["input_phenotype"] == (
        "balanced_AI_tonic_candidate"
    )
    assert analyzed["seed_results"][0]["c1_class"] == (
        "refractory_saturated"
    )
    assert analyzed["seed_results"][0]["route"] == "AI_observational_DMD"
    handcrafted = json.loads(json.dumps(first))
    handcrafted["producer_file_sha256"] = {
        str(producer): _sha(producer)
    }
    handcrafted["manifest_sha256"] = M.canonical_sha256({
        key: value for key, value in handcrafted.items()
        if key != "manifest_sha256"
    })
    handcrafted_path = tmp_path / "handcrafted.json"
    _write_json(handcrafted_path, handcrafted)
    with pytest.raises(ValueError, match="locked builder"):
        ANALYZE.analyze(
            representatives_path=handcrafted_path,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("c0_gate", "c1_gate", "match"),
    (
        (
            {
                "schema": ANALYZE.C0_GATE_SCHEMA,
                "verdict": "C0_no_evidence",
            },
            {
                "schema": ANALYZE.C1_GATE_SCHEMA,
                "verdict": "no_maturation_in_tested_primary_neighbourhood",
                "resolution_gate": "not_required_without_native_window",
            },
            "C0 resolution gate",
        ),
        (
            {
                "schema": ANALYZE.C0_GATE_SCHEMA,
                "verdict": "balanced_AI_tonic_candidate_supported",
                "resolution_gate": "passed",
            },
            {
                "schema": ANALYZE.C1_GATE_SCHEMA,
                "verdict": "C1_window_pending_dt2",
            },
            "C1 resolution gate",
        ),
        (
            {
                "schema": ANALYZE.C0_GATE_SCHEMA,
                "verdict": "balanced_AI_tonic_candidate_supported",
                "resolution_gate": "passed",
            },
            {
                "schema": ANALYZE.C1_GATE_SCHEMA,
                "verdict": "C1_blocked_resolution_gate",
            },
            "C1 resolution gate",
        ),
        (
            {
                "schema": ANALYZE.C0_GATE_SCHEMA,
                "verdict": "balanced_AI_tonic_candidate_supported",
            },
            {
                "schema": ANALYZE.C1_GATE_SCHEMA,
                "verdict": "no_maturation_in_tested_primary_neighbourhood",
                "resolution_gate": "not_required_without_native_window",
            },
            "C0 resolution gate",
        ),
    ),
)
def test_modal_lock_rejects_nonterminal_resolution_gates(
    c0_gate, c1_gate, match
):
    with pytest.raises(ValueError, match=match):
        ANALYZE._validate_terminal_resolution_gates(c0_gate, c1_gate)


@pytest.mark.parametrize(
    ("c0_gate", "c1_gate"),
    (
        (
            {
                "schema": ANALYZE.C0_GATE_SCHEMA,
                "verdict": "mixed_or_indeterminate_tonic_branch",
                "resolution_gate": "not_required_without_native_positive",
            },
            {
                "schema": ANALYZE.C1_GATE_SCHEMA,
                "verdict": "seed_heterogeneous_maturation",
                "resolution_gate": "not_required_without_native_window",
            },
        ),
        (
            {
                "schema": ANALYZE.C0_GATE_SCHEMA,
                "verdict": "resolution_sensitive_identity",
            },
            {
                "schema": ANALYZE.C1_GATE_SCHEMA,
                "verdict": "resolution_sensitive_maturation",
            },
        ),
    ),
)
def test_modal_lock_accepts_explicit_terminal_resolution_gates(
    c0_gate, c1_gate
):
    ANALYZE._validate_terminal_resolution_gates(c0_gate, c1_gate)


def test_adjudication_wrapper_rehashes_inputs_and_modal_cannot_change_verdict(tmp_path):
    c0 = {"verdict": "replicated_balanced_asynchronous_tonic_candidate"}
    primary = {"verdict": "no_local_maturation_window"}
    shell = {"verdict": "no_local_maturation_window"}
    modal = {"verdict": "descriptive_only", "status": "complete"}
    coverage = {
        "c0": {"status": "complete"},
        "c1_primary": {"status": "complete"},
        "c1_shell": {"status": "complete"},
        "modal": {"status": "complete"},
    }
    artifacts = {
        "c0": c0, "c1_primary": primary, "c1_shell": shell,
        "modal": modal, "coverage": coverage,
    }
    paths = {}
    for name, value in artifacts.items():
        path = tmp_path / f"{name}.json"
        _write_json(path, value)
        paths[name] = path
    phasec_path = tmp_path / "phasec.json"
    phasec_body = {
        "per_seed": {
            "1": {"canonical_config_sha": "b" * 64},
            "3": {"canonical_config_sha": "c" * 64},
        },
        "provenance": {"producer_file_sha256": {"x.py": "d" * 64}},
    }
    _write_json(phasec_path, {
        **phasec_body,
        "manifest_sha256": ADJ._canonical_sha(phasec_body),
    })
    phasec = json.loads(phasec_path.read_text())
    trigger_body = {
        "selection_is_closed": True,
        "phasec_manifest_sha256": phasec["manifest_sha256"],
        "producer_file_sha256": {"trigger.py": "e" * 64},
    }
    trigger_path = tmp_path / "trigger.json"
    _write_json(trigger_path, {
        **trigger_body,
        "manifest_sha256": ADJ._canonical_sha(trigger_body),
    })
    first = ADJ.adjudicate_files(
        c0_path=paths["c0"],
        c1_primary_path=paths["c1_primary"],
        c1_shell_path=paths["c1_shell"],
        modal_path=paths["modal"],
        coverage_path=paths["coverage"],
        trigger_path=trigger_path,
        phasec_manifest_path=phasec_path,
    )
    assert first["entry"] == "not_tested"
    assert first["offset"] == "not_tested"
    assert first["recovery_lifecycle"] == "not_established"
    assert first["summary_authority_model"] == (
        "runtime_rehash_of_deterministic_derived_summaries_then_"
        "final_write_once_lock; summary_SHAs_were_not_preregistered"
    )
    assert first["input_file_provenance"]["modal"]["file_sha256"] == _sha(
        paths["modal"]
    )
    modal["verdict"] = "attempted_modal_override"
    _write_json(paths["modal"], modal)
    second = ADJ.adjudicate_files(
        c0_path=paths["c0"],
        c1_primary_path=paths["c1_primary"],
        c1_shell_path=paths["c1_shell"],
        modal_path=paths["modal"],
        coverage_path=paths["coverage"],
        trigger_path=trigger_path,
        phasec_manifest_path=phasec_path,
    )
    assert second["verdict"] == first["verdict"]
    assert second["next_route"] == first["next_route"]
    assert second["input_file_provenance"]["modal"]["file_sha256"] != (
        first["input_file_provenance"]["modal"]["file_sha256"]
    )

    base_kwargs = {
        "c0_path": paths["c0"],
        "c1_primary_path": paths["c1_primary"],
        "c1_shell_path": paths["c1_shell"],
        "modal_path": paths["modal"],
        "coverage_path": paths["coverage"],
        "trigger_path": trigger_path,
        "phasec_manifest_path": phasec_path,
    }
    for missing in (
        "c1_primary_path", "c1_shell_path", "modal_path",
        "coverage_path", "trigger_path",
    ):
        kwargs = dict(base_kwargs)
        kwargs[missing] = None
        blocked = ADJ.adjudicate_files(**kwargs)
        assert blocked["verdict"] == "no_evidence"
        assert blocked["next_route"] == "no_evidence"
        assert blocked["wrapper_provenance_issues"]
        assert blocked["entry"] == "not_tested"
        assert blocked["recovery_lifecycle"] == "not_established"


def test_complete_mixed_c0_does_not_hide_preregistered_primary_window(
    tmp_path,
):
    artifacts = {
        "c0": {
            "verdict": "heterogeneous_or_unresolved",
            "seed_classes": {
                "1": "balanced_asynchronous_tonic_candidate",
                "3": "refractory_limited_plateau",
            },
        },
        "c1_primary": {"verdict": "local_maturation_window"},
        "c1_shell": {"verdict": "not_tested"},
        "modal": {"verdict": "seed_specific", "status": "partial"},
        "coverage": {
            "c0": {"status": "complete"},
            "c1_primary": {"status": "complete"},
            "c1_shell": {"status": "not_tested"},
            "modal": {"status": "partial"},
        },
    }
    paths = {}
    for name, value in artifacts.items():
        path = tmp_path / f"{name}.json"
        _write_json(path, value)
        paths[name] = path
    phasec_body = {
        "per_seed": {
            "1": {"canonical_config_sha": "b" * 64},
            "3": {"canonical_config_sha": "c" * 64},
        },
        "provenance": {"producer_file_sha256": {"x.py": "d" * 64}},
    }
    phasec_path = tmp_path / "phasec.json"
    phasec = {
        **phasec_body,
        "manifest_sha256": ADJ._canonical_sha(phasec_body),
    }
    _write_json(phasec_path, phasec)
    trigger_body = {
        "selection_is_closed": True,
        "phasec_manifest_sha256": phasec["manifest_sha256"],
        "producer_file_sha256": {"trigger.py": "e" * 64},
    }
    trigger_path = tmp_path / "trigger.json"
    _write_json(trigger_path, {
        **trigger_body,
        "manifest_sha256": ADJ._canonical_sha(trigger_body),
    })
    out = ADJ.adjudicate_files(
        c0_path=paths["c0"],
        c1_primary_path=paths["c1_primary"],
        c1_shell_path=paths["c1_shell"],
        modal_path=paths["modal"],
        coverage_path=paths["coverage"],
        trigger_path=trigger_path,
        phasec_manifest_path=phasec_path,
    )
    assert out["verdict"] == "maturation_window_at_primary_convex_states"
    assert out["layers"]["source_identity"]["verdict"] == (
        "heterogeneous_or_unresolved"
    )
    assert out["entry"] == "not_tested"
    assert out["recovery_lifecycle"] == "not_established"


def _nested_final_sources(
    tmp_path,
    *,
    c1_gate_verdict="maturation_window_at_primary_convex_states",
    c1_gate_reason=None,
):
    phasec_body = {
        "production_authorized": True,
        "per_seed": {
            "1": {"canonical_config_sha": "b" * 64},
            "3": {"canonical_config_sha": "c" * 64},
        },
        "provenance": {"producer_file_sha256": {"x.py": "d" * 64}},
    }
    phasec = {
        **phasec_body,
        "manifest_sha256": ADJ._canonical_sha(phasec_body),
    }
    c0_native = {
        "schema": "zm_phasec_c0_summary_v1",
        "manifest_sha256": phasec["manifest_sha256"],
        "aggregate": {
            "verdict": "mixed_or_indeterminate_tonic_branch",
            "seed_classes": {
                "1": "balanced_AI_tonic_candidate",
                "3": "refractory_saturated_branch",
            },
        },
        "seed_rows": [],
    }
    c0_gate = {
        "schema": "zm_phasec_c0_resolution_gate_v1",
        "verdict": "mixed_or_indeterminate_tonic_branch",
        "resolution_gate": "not_required_without_native_positive",
    }
    c1_native = {
        "schema": "zm_phasec1_summary_v1_2026-07-28",
        "phasec_manifest_sha256": phasec["manifest_sha256"],
        "primary_adjudication": {
            "status": "local_maturation_window",
            "strict_negative": False,
        },
        "secondary_shell_adjudication": {
            "status": "no_window",
            "strict_negative": False,
        },
        "verdict": "primary_maturation_candidate_requires_dt2",
    }
    paths = {}
    for name, value in (
        ("phasec", phasec),
        ("c0_native", c0_native),
        ("c0_gate", c0_gate),
        ("c1_native", c1_native),
    ):
        path = tmp_path / f"{name}.json"
        _write_json(path, value)
        paths[name] = path
    c1_gate = {
        "schema": "zm_phasec1_resolution_gate_v1_2026-07-28",
        "verdict": c1_gate_verdict,
        "resolution_gate": (
            "passed"
            if c1_gate_verdict.startswith("maturation_") else None
        ),
        "reason": c1_gate_reason,
        "native_summary_path": str(paths["c1_native"]),
        "native_summary_sha256": _sha(paths["c1_native"]),
    }
    modal = {
        "schema": "zm_phasec_modal_summary_v1_2026-07-28",
        "status": "partial",
        "verdict": "seed_specific_modal_audit_partial",
        "phasec_manifest_sha256": phasec["manifest_sha256"],
    }
    for name, value in (("c1_gate", c1_gate), ("modal", modal)):
        path = tmp_path / f"{name}.json"
        _write_json(path, value)
        paths[name] = path
    return paths, phasec


def _closed_trigger(tmp_path, phasec):
    body = {
        "selection_is_closed": True,
        "phasec_manifest_sha256": phasec["manifest_sha256"],
        "producer_file_sha256": {"trigger.py": "e" * 64},
    }
    path = tmp_path / "trigger-final.json"
    _write_json(path, {
        **body,
        "manifest_sha256": ADJ._canonical_sha(body),
    })
    return path


def test_build_inputs_deterministically_splits_nested_c1_and_adjudicates(
    tmp_path,
):
    paths, phasec = _nested_final_sources(tmp_path)
    inputs = ADJ.build_final_inputs(
        phasec_manifest_path=paths["phasec"],
        c0_gate_path=paths["c0_gate"],
        c0_native_path=paths["c0_native"],
        c1_gate_path=paths["c1_gate"],
        c1_native_path=paths["c1_native"],
        modal_path=paths["modal"],
    )
    assert inputs["c0"]["verdict"] == (
        "mixed_or_indeterminate_tonic_branch"
    )
    assert inputs["c1_primary"]["verdict"] == "local_maturation_window"
    assert inputs["c1_shell"]["verdict"] == "insufficient_coverage"
    assert inputs["coverage"] == {
        **{
            key: value for key, value in inputs["coverage"].items()
            if key not in {"c0", "c1_primary", "c1_shell", "modal"}
        },
        "c0": {"status": "complete"},
        "c1_primary": {"status": "complete"},
        "c1_shell": {"status": "incomplete"},
        "modal": {"status": "partial"},
    }
    for artifact in inputs.values():
        assert artifact["phasec_manifest_sha256"] == (
            phasec["manifest_sha256"]
        )
        assert artifact["source_provenance"]["c1_native_summary"][
            "file_sha256"
        ] == _sha(paths["c1_native"])
        assert artifact["derivation"] == (
            "deterministic_runtime_rehash_then_final_write_once_lock"
        )
    output_dir = tmp_path / "final-inputs"
    written = ADJ.write_final_inputs(output_dir, inputs)
    assert ADJ.write_final_inputs(output_dir, inputs) == written
    out = ADJ.adjudicate_files(
        c0_path=written["c0"],
        c1_primary_path=written["c1_primary"],
        c1_shell_path=written["c1_shell"],
        modal_path=paths["modal"],
        coverage_path=written["coverage"],
        trigger_path=_closed_trigger(tmp_path, phasec),
        phasec_manifest_path=paths["phasec"],
    )
    assert out["verdict"] == "maturation_window_at_primary_convex_states"
    assert out["layers"]["source_identity"]["verdict"] == (
        "mixed_or_indeterminate_tonic_branch"
    )


@pytest.mark.parametrize(
    "gate_verdict",
    ("C1_window_pending_dt2", "C1_blocked_resolution_gate"),
)
def test_build_inputs_never_turns_pending_or_blocked_dt2_into_a_result(
    tmp_path, gate_verdict,
):
    paths, phasec = _nested_final_sources(
        tmp_path, c1_gate_verdict=gate_verdict
    )
    inputs = ADJ.build_final_inputs(
        phasec_manifest_path=paths["phasec"],
        c0_gate_path=paths["c0_gate"],
        c0_native_path=paths["c0_native"],
        c1_gate_path=paths["c1_gate"],
        c1_native_path=paths["c1_native"],
        modal_path=paths["modal"],
    )
    assert inputs["c1_primary"]["verdict"] == "C1_blocked_manifest"
    assert inputs["coverage"]["c1_primary"]["status"] == "blocked_manifest"
    assert inputs["c1_shell"]["verdict"] == "not_tested"
    assert inputs["coverage"]["c1_shell"]["status"] == "not_tested"
    assert "maturation" not in inputs["c1_primary"]["verdict"].lower()
    assert "no_local_maturation" not in inputs["c1_primary"]["verdict"]
    written = ADJ.write_final_inputs(tmp_path / "inputs", inputs)
    out = ADJ.adjudicate_files(
        c0_path=written["c0"],
        c1_primary_path=written["c1_primary"],
        c1_shell_path=written["c1_shell"],
        modal_path=paths["modal"],
        coverage_path=written["coverage"],
        trigger_path=_closed_trigger(tmp_path, phasec),
        phasec_manifest_path=paths["phasec"],
    )
    assert out["verdict"] == "C1_blocked_manifest"
