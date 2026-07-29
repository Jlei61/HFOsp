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


def _write_production_part(
    path,
    part,
    *,
    manifest_sha256,
    task_key,
    producer_sha256,
):
    """Publish a synthetic part with its immutable adjacent resource receipt."""

    token = "test-" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    runtime = dict(part.get("runtime_provenance", {}))
    runtime.update({
        "manifest_sha256": manifest_sha256,
        "producer_sha256": producer_sha256,
        "coordinator_run_id": "modal-test-coordinator",
        "coordinator_launch_token": token,
        "self_pid_at_publish": 12345,
        "self_vm_swap_kb_at_publish": 0,
    })
    manifest_field = (
        "manifest_sha256"
        if part.get("schema") == "zm_phasec_identity_cell_v1"
        else "phasec_manifest_sha256"
    )
    part = {
        **part,
        manifest_field: manifest_sha256,
        "runtime_provenance": runtime,
    }
    _write_json(path, part)
    audit = {
        "pid": 12345,
        "task_key": task_key,
        "coordinator_run_id": "modal-test-coordinator",
        "coordinator_launch_token": token,
        "n_samples": 1,
        "first_sample_at": 1.0,
        "last_sample_at": 1.0,
        "observed_max_kb": 0,
        "final_publish_swap_kb": 0,
    }
    receipt = ANALYZE.PRES.build_resource_receipt(
        artifact_path=path,
        artifact_root=ANALYZE.ROOT,
        artifact_sha256=_sha(path),
        manifest_sha256=manifest_sha256,
        task_key=task_key,
        run_id="modal-test-coordinator",
        launch_token=token,
        pid=12345,
        audit_row=audit,
        sampled_allowed_bytes=0,
    )
    ANALYZE.PRES.publish_resource_receipt_once(
        ANALYZE.PRES.resource_receipt_path(path), receipt
    )
    return part


def _resource_index_fixture(tmp_path):
    """Two non-modal parts with auxiliary observables and valid receipts."""
    manifest_sha = "a" * 64
    tasks = []
    for index, role in enumerate(("numerator", "denominator")):
        obs = tmp_path / f"resource_obs_{index}.npz"
        np.savez(obs, value=np.asarray([index], np.int16))
        part_path = tmp_path / f"resource_part_{index}.json"
        task_key = f"resource-test-{index}"
        part = {
            "schema": "zm_phasec1_base_part_v1_2026-07-28",
            "status": "complete",
            "seed": 1,
            "tier": "primary_convex",
            "cell_id": f"cell-{index}",
            "phase": "rising",
            "noise": f"noise-{index}",
            "resolution": "dt",
            "observables_path": str(obs),
            "observables_sha256": _sha(obs),
        }
        _write_production_part(
            part_path,
            part,
            manifest_sha256=manifest_sha,
            task_key=task_key,
            producer_sha256={"runner.py": "b" * 64},
        )
        tasks.append((task_key, part_path, role))
    index = ADJ.C1.build_resource_receipt_index(
        tasks, manifest_sha256=manifest_sha
    )
    return {"manifest_sha256": manifest_sha}, {
        "resource_receipt_index": index
    }, tasks


def test_final_resource_index_rejects_deleted_nonmodal_receipt(
    tmp_path, monkeypatch
):
    phasec, summary, tasks = _resource_index_fixture(tmp_path)
    monkeypatch.setattr(
        ADJ, "_expected_resource_tasks",
        lambda *_args, **_kwargs: tasks,
    )
    assert ADJ._resource_index_issues(
        "c1_native", summary, phasec
    ) == []
    ADJ.PRES.resource_receipt_path(tasks[1][1]).unlink()
    issues = ADJ._resource_index_issues(
        "c1_native", summary, phasec
    )
    assert any("resource_live_binding_failure" in row for row in issues)

    summary_path = tmp_path / "resource_summary.json"
    _write_json(summary_path, summary)
    audit = {
        "status": "complete",
        "issues": [],
        "summary_sources": {
            "synthetic": ADJ._resource_source_ref(
                "c1_native", summary_path, summary
            )
        },
    }
    artifacts = {
        "c0": {
            "verdict": "balanced_AI_tonic_candidate_supported",
            "resource_receipt_audit": audit,
        },
        "c1_primary": {
            "verdict": "no_local_maturation_window",
            "resource_receipt_audit": audit,
        },
        "c1_shell": {
            "verdict": "not_tested",
            "resource_receipt_audit": audit,
        },
        "modal": {"verdict": "descriptive_only", "status": "complete"},
        "coverage": {
            "c0": {"status": "complete"},
            "c1_primary": {"status": "complete"},
            "c1_shell": {"status": "not_tested"},
            "modal": {"status": "complete"},
            "resource_receipt_audit": audit,
        },
    }
    paths = {}
    for name, value in artifacts.items():
        path = tmp_path / f"final_{name}.json"
        _write_json(path, value)
        paths[name] = path
    phasec_body = {
        "per_seed": {"1": {"canonical_config_sha": "b" * 64}},
        "provenance": {"producer_file_sha256": {"x.py": "c" * 64}},
    }
    phasec_path = tmp_path / "final_phasec.json"
    _write_json(phasec_path, {
        **phasec_body,
        "manifest_sha256": ADJ._canonical_sha(phasec_body),
    })
    trigger_body = {
        "selection_is_closed": True,
        "phasec_manifest_sha256": ADJ._canonical_sha(phasec_body),
        "producer_file_sha256": {"trigger.py": "d" * 64},
    }
    trigger_path = tmp_path / "final_trigger.json"
    _write_json(trigger_path, {
        **trigger_body,
        "manifest_sha256": ADJ._canonical_sha(trigger_body),
    })
    monkeypatch.setattr(
        ADJ, "_final_resource_source_contract", lambda *_: []
    )
    final = ADJ.adjudicate_files(
        c0_path=paths["c0"],
        c1_primary_path=paths["c1_primary"],
        c1_shell_path=paths["c1_shell"],
        modal_path=paths["modal"],
        coverage_path=paths["coverage"],
        trigger_path=trigger_path,
        phasec_manifest_path=phasec_path,
    )
    assert final["verdict"] == "no_evidence"
    assert any(
        "resource_receipt_audit_live_failure" in row
        for row in final["wrapper_provenance_issues"]
    )


@pytest.mark.parametrize("mutation", ("part_swap", "aux_swap"))
def test_final_resource_index_rejects_part_or_aux_swap(
    tmp_path, monkeypatch, mutation
):
    phasec, summary, tasks = _resource_index_fixture(tmp_path)
    monkeypatch.setattr(
        ADJ, "_expected_resource_tasks",
        lambda *_args, **_kwargs: tasks,
    )
    entries = summary["resource_receipt_index"]["entries"]
    if mutation == "part_swap":
        left, right = tasks[0][1], tasks[1][1]
    else:
        left = ADJ._resolve(entries[0]["aux_observables_path"])
        right = ADJ._resolve(entries[1]["aux_observables_path"])
    left_bytes, right_bytes = left.read_bytes(), right.read_bytes()
    left.write_bytes(right_bytes)
    right.write_bytes(left_bytes)
    issues = ADJ._resource_index_issues(
        "c1_native", summary, phasec
    )
    assert any(
        "resource_live_binding" in row for row in issues
    )


def test_final_resource_index_requires_reused_denominator_role(
    tmp_path, monkeypatch
):
    phasec, summary, tasks = _resource_index_fixture(tmp_path)
    numerator_only = ADJ.C1.build_resource_receipt_index(
        tasks[:1], manifest_sha256=phasec["manifest_sha256"]
    )
    summary["resource_receipt_index"] = numerator_only
    monkeypatch.setattr(
        ADJ, "_expected_resource_tasks",
        lambda *_args, **_kwargs: tasks,
    )
    issues = ADJ._resource_index_issues(
        "c1_native", summary, phasec
    )
    assert any("resource_unique_task_set_mismatch" in row for row in issues)
    assert any(
        "resource_logical_consumption_set_mismatch" in row
        for row in issues
    )


def _modal_run_fixture(tmp_path, *, resolution="dt", suffix="one"):
    manifest_sha = "a" * 64
    obs_path = tmp_path / f"{suffix}.npz"
    np.savez(
        obs_path,
        phasec1_observables_schema=np.asarray("schema"),
        bin_ms=np.asarray(2.0),
        **_observables(7),
    )
    part_path = tmp_path / f"{suffix}.json"
    prefix = "base" if resolution == "dt" else "dt2"
    task_key = (
        f"{prefix}|s1|primary_convex|cell_a|rising|noise_replay"
    )
    part = _write_production_part(
        part_path,
        {
            "schema": "zm_phasec1_base_part_v1_2026-07-28",
            "status": "complete",
            "seed": 1,
            "tier": "primary_convex",
            "cell_id": "cell_a",
            "phase": "rising",
            "noise": "noise_replay",
            "resolution": resolution,
            "observables_path": str(obs_path),
            "observables_sha256": _sha(obs_path),
        },
        manifest_sha256=manifest_sha,
        task_key=task_key,
        producer_sha256={"runner.py": "b" * 64},
    )
    run = {
        "phase": "rising",
        "noise": "noise_replay",
        "role": "fit",
        "part_path": str(part_path),
        "part_file_sha256": _sha(part_path),
        "part_semantic_sha256": ANALYZE._artifact_semantic_sha(part),
        "observables_path": str(obs_path),
        "observables_file_sha256": _sha(obs_path),
        "observables_semantic_sha256": ANALYZE._semantic_npz_sha(obs_path),
        "runtime_producer_file_sha256": {"runner.py": "b" * 64},
        **ANALYZE._validate_and_lock_resource_receipt(
            part_path,
            part,
            phasec_manifest_sha256=manifest_sha,
        ),
    }
    return manifest_sha, part_path, run


def test_modal_task_key_supports_c0_native_c1_and_dt2_c1():
    assert ANALYZE._production_task_key({
        "schema": "zm_phasec_identity_cell_v1",
        "seed": 3,
        "state_tag": "peak__natural",
        "replicate": "noise_repeat",
    }) == "identity|s3|peak__natural|noise_repeat"
    common = {
        "schema": "zm_phasec1_base_part_v1_2026-07-28",
        "seed": 1,
        "tier": "primary_convex",
        "cell_id": "cell_a",
        "phase": "rising",
        "noise": "noise_replay",
    }
    assert ANALYZE._production_task_key({
        **common, "resolution": "dt",
    }) == "base|s1|primary_convex|cell_a|rising|noise_replay"
    assert ANALYZE._production_task_key({
        **common, "resolution": "dt2",
    }) == "dt2|s1|primary_convex|cell_a|rising|noise_replay"


@pytest.mark.parametrize("mutation", ("delete", "tamper", "swap"))
def test_modal_load_fails_closed_on_resource_receipt_mutation(
    tmp_path, mutation
):
    manifest_sha, part_path, run = _modal_run_fixture(tmp_path)
    receipt_path = ANALYZE.PRES.resource_receipt_path(part_path)
    if mutation == "delete":
        receipt_path.unlink()
    elif mutation == "tamper":
        receipt = json.loads(receipt_path.read_text())
        receipt["task_key"] = "tampered"
        _write_json(receipt_path, receipt)
    else:
        _, other_part, _ = _modal_run_fixture(
            tmp_path, resolution="dt2", suffix="other"
        )
        receipt_path.write_bytes(
            ANALYZE.PRES.resource_receipt_path(other_part).read_bytes()
        )
    with pytest.raises(ValueError, match="resource receipt invalid"):
        ANALYZE._load_observables(
            run,
            expected_seed=1,
            expected_cell="cell_a",
            phasec_manifest_sha256=manifest_sha,
        )


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
            "schema": "zm_phasec_identity_cell_v1",
            "status": "complete",
            "seed": 1,
            "cell_id": None,
            "state_tag": "rising",
            "replicate": noise,
            "resolution": "dt",
            "phase": "rising",
            "noise": noise,
            "observables_path": str(obs_path),
            "observables_sha256": _sha(obs_path),
        }
        part = _write_production_part(
            part_path,
            part,
            manifest_sha256=phasec["manifest_sha256"],
            task_key=f"identity|s1|rising|{noise}",
            producer_sha256={"runner.py": "b" * 64},
        )
        receipt_lock = ANALYZE._validate_and_lock_resource_receipt(
            part_path,
            part,
            phasec_manifest_sha256=phasec["manifest_sha256"],
        )
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
            **receipt_lock,
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
    with pytest.raises(ValueError, match="resource receipt invalid"):
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
        "primary_gate": {"status": "not_required"},
        "shell_gate": {"status": "not_required"},
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
                    "schema": "zm_phasec1_base_part_v1_2026-07-28",
                    "status": "complete",
                    "seed": 1,
                    "tier": tier,
                    "cell_id": cell_id,
                    "phase": phase,
                    "noise": noise,
                    "resolution": "dt",
                    "observables_path": str(obs_path),
                    "observables_sha256": _sha(obs_path),
                }
                part = _write_production_part(
                    part_path,
                    part,
                    manifest_sha256=phasec["manifest_sha256"],
                    task_key=(
                        f"base|s1|{tier}|{cell_id}|{phase}|{noise}"
                    ),
                    producer_sha256={str(producer): _sha(producer)},
                )
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
                "primary_gate": {"status": "not_required"},
                "shell_gate": {"status": "not_required"},
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
                "primary_gate": {"status": "indeterminate"},
                "shell_gate": {"status": "not_required"},
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
                "primary_gate": {"status": "blocked"},
                "shell_gate": {"status": "not_required"},
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
                "primary_gate": {"status": "not_required"},
                "shell_gate": {"status": "not_required"},
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
                "primary_gate": {"status": "not_required"},
                "shell_gate": {"status": "not_required"},
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
                "primary_gate": {"status": "contradicted"},
                "shell_gate": {"status": "not_required"},
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
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(ADJ, "_audit_resource_sources", lambda *_: [])
    monkeypatch.setattr(
        ADJ, "_final_resource_source_contract", lambda *_: []
    )
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
                "resource_receipt_audit": {
                    "status": "complete",
                    "issues": [],
                    "summary_sources": {"synthetic": {}},
                },
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
        "resolution": "dt",
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
    c0_native["manifest_file_sha256"] = _sha(paths["phasec"])
    _write_json(paths["c0_native"], c0_native)
    c1_native["phasec_manifest_file_sha256"] = _sha(paths["phasec"])
    _write_json(paths["c1_native"], c1_native)
    c0_gate.update({
        "native_summary": str(paths["c0_native"]),
        "native_summary_sha256": _sha(paths["c0_native"]),
        "dt2_summary": None,
        "dt2_summary_sha256": None,
    })
    _write_json(paths["c0_gate"], c0_gate)
    c1_gate = {
        "schema": "zm_phasec1_resolution_gate_v2_2026-07-29",
        "verdict": c1_gate_verdict,
        "resolution_gate": (
            "passed"
            if c1_gate_verdict.startswith("maturation_") else None
        ),
        "reason": c1_gate_reason,
        "native_summary_path": str(paths["c1_native"]),
        "native_summary_sha256": _sha(paths["c1_native"]),
    }
    if c1_gate_verdict == "maturation_window_at_primary_convex_states":
        c1_gate["primary_gate"] = {"status": "confirmed"}
        c1_gate["shell_gate"] = {"status": "not_required"}
    elif c1_gate_verdict == "maturation_candidate_in_secondary_shell":
        c1_gate["primary_gate"] = {"status": "not_required"}
        c1_gate["shell_gate"] = {"status": "confirmed"}
    elif c1_gate_verdict == "resolution_sensitive_maturation":
        if c1_gate_reason and "secondary" in c1_gate_reason:
            c1_gate["primary_gate"] = {"status": "not_required"}
            c1_gate["shell_gate"] = {"status": "contradicted"}
        else:
            c1_gate["primary_gate"] = {"status": "contradicted"}
            c1_gate["shell_gate"] = {"status": "not_required"}
    elif c1_gate_verdict == "C1_window_pending_dt2":
        c1_gate["primary_gate"] = {"status": "indeterminate"}
        c1_gate["shell_gate"] = {"status": "not_required"}
    elif c1_gate_verdict == "C1_blocked_resolution_gate":
        c1_gate["primary_gate"] = {"status": "blocked"}
        c1_gate["shell_gate"] = {"status": "not_required"}
    else:
        c1_gate["primary_gate"] = {"status": "not_required"}
        c1_gate["shell_gate"] = {"status": "not_required"}
    if c1_gate_verdict in {
        "maturation_window_at_primary_convex_states",
        "maturation_candidate_in_secondary_shell",
        "resolution_sensitive_maturation",
    }:
        dt2_summary = {
            "schema": (
                "zm_phasec1_dt2_confirmation_summary_v1_2026-07-28"
            ),
            "resolution": "dt2_confirmation_only",
            "phasec_manifest_sha256": phasec["manifest_sha256"],
            "phasec_manifest_file_sha256": _sha(paths["phasec"]),
            "verdict": c1_gate_verdict,
        }
        paths["c1_dt2"] = tmp_path / "c1_dt2.json"
        _write_json(paths["c1_dt2"], dt2_summary)
        c1_gate.update({
            "dt2_summary_path": str(paths["c1_dt2"]),
            "dt2_summary_sha256": _sha(paths["c1_dt2"]),
        })
    else:
        c1_gate.update({
            "dt2_summary_path": None,
            "dt2_summary_sha256": None,
        })
    paths["c1_gate"] = tmp_path / "c1_gate.json"
    _write_json(paths["c1_gate"], c1_gate)
    modal_inputs = {
        "phasec_manifest": (paths["phasec"], phasec),
        "c0_resolution_gate": (paths["c0_gate"], c0_gate),
        "c0_native_summary": (paths["c0_native"], c0_native),
        "c1_resolution_gate": (paths["c1_gate"], c1_gate),
        "c1_native_summary": (paths["c1_native"], c1_native),
    }
    modal = {
        "schema": "zm_phasec_modal_summary_v1_2026-07-28",
        "status": "partial",
        "verdict": "seed_specific_modal_audit_partial",
        "phasec_manifest_sha256": phasec["manifest_sha256"],
        "input_provenance": {
            name: {
                "file_sha256": _sha(path),
                "semantic_sha256": ADJ._canonical_sha(value),
                "schema": value.get("schema"),
                "parent_phasec_manifest_sha256": phasec[
                    "manifest_sha256"
                ],
            }
            for name, (path, value) in modal_inputs.items()
        },
    }
    paths["modal"] = tmp_path / "modal.json"
    _write_json(paths["modal"], modal)
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
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(ADJ, "_audit_resource_sources", lambda *_: [])
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


def test_final_inputs_consume_primary_and_shell_resolution_gates_separately():
    native = {
        "primary_adjudication": {
            "status": "local_maturation_window",
            "strict_negative": False,
        },
        "secondary_shell_adjudication": {
            "status": "maturation_candidate_in_secondary_shell",
            "strict_negative": False,
        },
    }
    gate = {
        "verdict": "maturation_window_at_primary_convex_states",
        "primary_gate": {"status": "confirmed"},
        "shell_gate": {"status": "indeterminate"},
    }
    primary, shell, primary_coverage, shell_coverage = (
        ADJ._c1_final_inputs(gate, native, common={})
    )
    assert primary["verdict"] == "local_maturation_window"
    assert primary_coverage == "complete"
    assert shell["verdict"] == "insufficient_coverage"
    assert shell_coverage == "incomplete"
    assert shell["layer_resolution_gate"]["status"] == "indeterminate"


def test_build_inputs_rejects_stale_c0_resolution_gate(tmp_path):
    paths, _phasec = _nested_final_sources(tmp_path)
    native = json.loads(paths["c0_native"].read_text(encoding="utf-8"))
    native["aggregate"]["verdict"] = "refractory_saturated_branch_supported"
    _write_json(paths["c0_native"], native)
    with pytest.raises(ValueError, match="C0 gate/native summary file SHA mismatch"):
        ADJ.build_final_inputs(
            phasec_manifest_path=paths["phasec"],
            c0_gate_path=paths["c0_gate"],
            c0_native_path=paths["c0_native"],
            c1_gate_path=paths["c1_gate"],
            c1_native_path=paths["c1_native"],
            modal_path=paths["modal"],
        )


def test_build_inputs_rejects_c0_schema_parent_and_modal_source_drift(tmp_path):
    paths, _phasec = _nested_final_sources(tmp_path)
    native = json.loads(paths["c0_native"].read_text(encoding="utf-8"))
    native["manifest_file_sha256"] = "0" * 64
    _write_json(paths["c0_native"], native)
    gate = json.loads(paths["c0_gate"].read_text(encoding="utf-8"))
    gate["native_summary_sha256"] = _sha(paths["c0_native"])
    _write_json(paths["c0_gate"], gate)
    with pytest.raises(ValueError, match="C0 native summary parent"):
        ADJ.build_final_inputs(
            phasec_manifest_path=paths["phasec"],
            c0_gate_path=paths["c0_gate"],
            c0_native_path=paths["c0_native"],
            c1_gate_path=paths["c1_gate"],
            c1_native_path=paths["c1_native"],
            modal_path=paths["modal"],
        )

    schema_dir = tmp_path / "schema"
    schema_dir.mkdir()
    paths, _phasec = _nested_final_sources(schema_dir)
    gate = json.loads(paths["c0_gate"].read_text(encoding="utf-8"))
    gate["schema"] = "wrong"
    _write_json(paths["c0_gate"], gate)
    with pytest.raises(ValueError, match="C0 resolution gate schema"):
        ADJ.build_final_inputs(
            phasec_manifest_path=paths["phasec"],
            c0_gate_path=paths["c0_gate"],
            c0_native_path=paths["c0_native"],
            c1_gate_path=paths["c1_gate"],
            c1_native_path=paths["c1_native"],
            modal_path=paths["modal"],
        )

    modal_dir = tmp_path / "modal-drift"
    modal_dir.mkdir()
    paths, _phasec = _nested_final_sources(modal_dir)
    gate = json.loads(paths["c0_gate"].read_text(encoding="utf-8"))
    gate["diagnostic_drift"] = True
    _write_json(paths["c0_gate"], gate)
    with pytest.raises(
        ValueError, match="modal/c0_resolution_gate input provenance mismatch"
    ):
        ADJ.build_final_inputs(
            phasec_manifest_path=paths["phasec"],
            c0_gate_path=paths["c0_gate"],
            c0_native_path=paths["c0_native"],
            c1_gate_path=paths["c1_gate"],
            c1_native_path=paths["c1_native"],
            modal_path=paths["modal"],
        )


def test_build_inputs_requires_locked_c1_native_and_dt2_provenance(tmp_path):
    paths, _phasec = _nested_final_sources(tmp_path)
    gate = json.loads(paths["c1_gate"].read_text(encoding="utf-8"))
    gate.pop("native_summary_sha256")
    _write_json(paths["c1_gate"], gate)
    with pytest.raises(
        ValueError, match="C1 gate lacks locked native summary provenance"
    ):
        ADJ.build_final_inputs(
            phasec_manifest_path=paths["phasec"],
            c0_gate_path=paths["c0_gate"],
            c0_native_path=paths["c0_native"],
            c1_gate_path=paths["c1_gate"],
            c1_native_path=paths["c1_native"],
            modal_path=paths["modal"],
        )

    (tmp_path / "missing-dt2").mkdir()
    paths, _phasec = _nested_final_sources(tmp_path / "missing-dt2")
    gate = json.loads(paths["c1_gate"].read_text(encoding="utf-8"))
    gate["dt2_summary_path"] = None
    gate["dt2_summary_sha256"] = None
    _write_json(paths["c1_gate"], gate)
    with pytest.raises(
        ValueError,
        match="terminal resolution verdict lacks dt2 summary provenance",
    ):
        ADJ.build_final_inputs(
            phasec_manifest_path=paths["phasec"],
            c0_gate_path=paths["c0_gate"],
            c0_native_path=paths["c0_native"],
            c1_gate_path=paths["c1_gate"],
            c1_native_path=paths["c1_native"],
            modal_path=paths["modal"],
        )

    (tmp_path / "drifted-dt2").mkdir()
    paths, _phasec = _nested_final_sources(tmp_path / "drifted-dt2")
    with paths["c1_dt2"].open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(
        ValueError, match="C1 gate/dt2 summary file SHA mismatch"
    ):
        ADJ.build_final_inputs(
            phasec_manifest_path=paths["phasec"],
            c0_gate_path=paths["c0_gate"],
            c0_native_path=paths["c0_native"],
            c1_gate_path=paths["c1_gate"],
            c1_native_path=paths["c1_native"],
            modal_path=paths["modal"],
        )


def test_build_inputs_rejects_legacy_c1_gate_without_layer_closures(tmp_path):
    paths, _phasec = _nested_final_sources(tmp_path)
    gate = json.loads(paths["c1_gate"].read_text(encoding="utf-8"))
    gate.pop("shell_gate")
    _write_json(paths["c1_gate"], gate)
    with pytest.raises(
        ValueError, match="lacks closed layer gates"
    ):
        ADJ.build_final_inputs(
            phasec_manifest_path=paths["phasec"],
            c0_gate_path=paths["c0_gate"],
            c0_native_path=paths["c0_native"],
            c1_gate_path=paths["c1_gate"],
            c1_native_path=paths["c1_native"],
            modal_path=paths["modal"],
        )


@pytest.mark.parametrize(
    "gate_verdict",
    ("C1_window_pending_dt2", "C1_blocked_resolution_gate"),
)
def test_build_inputs_never_turns_pending_or_blocked_dt2_into_a_result(
    tmp_path, gate_verdict, monkeypatch,
):
    monkeypatch.setattr(ADJ, "_audit_resource_sources", lambda *_: [])
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
    assert inputs["c1_shell"]["verdict"] == "insufficient_coverage"
    assert inputs["coverage"]["c1_shell"]["status"] == "incomplete"
    assert inputs["c1_shell"]["layer_resolution_gate"]["status"] == (
        "not_required"
    )
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
