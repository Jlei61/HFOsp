"""Phase-C immutable manifest and fail-closed drift tests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from src import topic4_zm_phasec_contract as C
from src import topic4_zm_phasec_metrics as M
from src import topic4_zm_phasec_neighbourhood as N
from src import topic4_zm_phasec_phenotype as P
import scripts.lock_topic4_zm_phasec as LOCK
import scripts.analyze_topic4_zm_phasec1 as C1
import scripts.build_topic4_zm_phasec1_neighbourhood as B


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RUNTIME_CLOSURE = {
    "scripts/run_topic4_zm_branch_decision.py",
    "scripts/run_m4_phaseplane.py",
    "scripts/run_m4_dynamic_qi.py",
    "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py",
    "scripts/run_zm_snn_native_exit.py",
    "src/topic4_zm_checkpoint.py",
    "src/topic4_zm_noise_bank.py",
    "src/topic4_zm_source_rhythm.py",
    "src/topic4_zm_ictal_carrier.py",
    "src/topic4_zm_carrier_verdict.py",
    "src/topic4_zm_fork_state.py",
    "src/topic4_zm_minimal_carrier.py",
    "src/topic4_zm_effective_rank.py",
    "src/topic4_zm_modal_operator.py",
    "src/topic4_zm_boundaries.py",
    "src/sef_hfo_heterogeneity.py",
    "src/sef_hfo_subject_placement.py",
    "src/sef_hfo_m4_metrics.py",
    "src/sef_hfo_m4_phaseplane.py",
    "src/sef_hfo_m4_termination.py",
}


@pytest.fixture(scope="module")
def manifest():
    return C.build_manifest(ROOT)


def _rehash(d):
    out = copy.deepcopy(d)
    out.pop("manifest_sha256", None)
    raw = json.dumps(
        out, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()
    out["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    return out


def test_real_inputs_build_a_deterministic_complete_contract(manifest):
    again = C.build_manifest(ROOT)
    assert again == manifest
    assert manifest["schema"] == C.PHASEC_INPUT_VERSION
    assert manifest["production_authorized"] is False
    with pytest.raises(C.ContractInputError, match="production requires"):
        C.require_production_manifest(manifest)
    assert manifest["c0"]["seeds"] == [1, 3, 4]
    assert manifest["c0"]["fast_phases"] == ["rising", "peak"]
    assert manifest["c0"]["noise_replicates"] == [
        "noise_replay", "noise_resample_1", "noise_resample_2"
    ]
    assert manifest["c0"]["duration_ms"] == 8000.0
    assert manifest["c0"]["burn_in_ms"] == 500.0
    assert manifest["c0"]["protocols"]["identity"] == {
        "burn_in_ms": 500.0,
        "measure_ms": 8000.0,
        "fine_bin_ms": 2.0,
        "current_stride_ms": 1.0,
    }
    assert manifest["c0"]["protocols"]["gain"] == {
        "burn_in_ms": 500.0,
        "measure_ms": 1000.0,
        "threshold_delta_abs_mV": [0.05, 0.10],
    }
    assert manifest["c0"]["required_identity_continuations"] == 18
    assert manifest["c0"]["threshold_perturbation"]["values"] == [
        -0.10, -0.05, 0.0, 0.05, 0.10
    ]
    assert manifest["c0"]["threshold_perturbation"]["changes_production_config"] is False
    assert len(manifest["c1"]["primary_cell_names"]) == 10
    assert len(manifest["c1"]["secondary_shell_cell_names"]) == 8
    assert manifest["c1"]["secondary_shell_step_robust_sd"] == 0.25
    assert manifest["c1"]["physical_bounds"]["z_min"] == 0.0
    assert manifest["c1"]["physical_bounds"]["z_min_source"] == (
        "Z equation clip; not qI q_min"
    )
    assert manifest["design_amendments"]["z_physical_bound"][
        "qI_q_min_not_applicable"
    ] is True
    assert "threshold diagnostic offsets" in manifest["design_amendments"][
        "gain_probe"
    ]["locked_implementation"]
    assert manifest["claim_boundary"]["recovery_lifecycle_established"] is False
    assert manifest["claim_boundary"]["actuator_authorized"] is False
    producer_locks = manifest["provenance"]["producer_file_sha256"]
    assert EXPECTED_RUNTIME_CLOSURE <= set(producer_locks)
    for path in (
        "scripts/analyze_topic4_zm_phasec_modal.py",
        "scripts/adjudicate_topic4_zm_phasec.py",
        "scripts/plot_topic4_zm_phasec.py",
        "src/topic4_zm_phasec_modal.py",
        "src/topic4_zm_phasec_plot.py",
        "src/topic4_zm_phasec_resources.py",
    ):
        assert path in producer_locks
        assert producer_locks[path] == C.sha256_file(ROOT / path)
    assert manifest["resources"] == C._resources()
    for seed in ("1", "3", "4"):
        row = manifest["per_seed"][seed]
        assert len(row["c1_source_states"]) == 6
        assert len(row["c1_primary_cells"]) == 10
        assert len(row["c1_secondary_shell_cells"]) == 8
        assert row["fixed_panels"]["activity_independent"] is True
        assert len(row["fixed_panels"]["analysis_panel_E_ids"]) == 1024
        assert len(row["fixed_panels"]["pairwise_panel_E_ids"]) == 256
        assert len(row["fixed_panels"]["panel_sha256"]) == 64
        assert row["panel_selection_config_sha"] == row["canonical_config_sha"]
        assert row["readout_kernel_width_mm"] == 0.278
        if seed in {"1", "3"}:
            dt2 = row["resolution_confirmations"]["dt2"]
            assert dt2["parent_config_sha"] == row["canonical_config_sha"]
            assert len(dt2["c1_source_states"]) == 6
            assert dt2["panel_selection_config_sha"] == row[
                "canonical_config_sha"
            ]
            assert dt2["fixed_panels"] == row["fixed_panels"]
        else:
            assert row["resolution_confirmations"] == {}
        for phase in ("rising", "peak"):
            state = row["c0_carrier_states"][phase]["state"]
            assert Path(ROOT / state["path"]).is_file()
            assert len(state["file_sha256"]) == 64
            banks = row["c0_carrier_states"][phase]["noise_banks"]
            assert len(banks) == 3 and all(x["is_paired"] for x in banks)


def test_canonical_engine_closure_is_independently_complete_and_fail_closed():
    canonical = json.loads(
        (
            ROOT / C.DEFAULT_UPSTREAM_ROOT
            / "phase0/canonical_config.json"
        ).read_text()
    )
    expected = {str(path) for path in C.CANONICAL_ENGINE_RUNTIME_PATHS}
    for seed in C.PRIMARY_SEEDS:
        seed_config = canonical["seeds"][str(seed)]
        assert set(seed_config["config"]["engine_sha256"]) == expected
        C._validate_canonical_engine_closure(
            seed_config, root=ROOT, seed=seed
        )
    mutated = copy.deepcopy(canonical["seeds"]["1"])
    mutated["config"]["engine_sha256"].pop(
        "src/snn_engine/mz_slow_vars.py"
    )
    with pytest.raises(
        C.ContractInputError, match="canonical engine closure drift"
    ):
        C._validate_canonical_engine_closure(
            mutated, root=ROOT, seed=1
        )


def test_resource_contract_drift_fails_with_recomputed_self_hash(manifest):
    changed = copy.deepcopy(manifest)
    changed["resources"]["worker_swap_poll_max_s"] = 6.0
    changed = _rehash(changed)
    with pytest.raises(C.ContractError, match="resource contract drift"):
        C.validate_manifest(changed)


def test_all_decisive_thresholds_are_manifested_and_match_producers(manifest):
    thresholds = manifest["thresholds"]
    assert thresholds == C._thresholds()
    assert thresholds["pairwise_shift_null_strata"] == list(
        M.PAIR_NULL_STRATUM_NAMES
    )
    assert thresholds["pairwise_shift_null_draws"] == 100
    assert thresholds["c1_rest_dwell_ms"] == P.DEFAULTS["rest_dwell_ms"]
    assert thresholds["c1_periodic_rest_reset_fraction_max"] == (
        P.DEFAULTS["maximum_periodic_rest_reset_fraction"]
    )
    assert thresholds["c1_periodic_source_phase_corr_min"] == (
        C1.PERIODIC_PHASE_STRUCTURE_CORR_MIN
    )
    assert thresholds[
        "c1_periodic_cross_phase_period_rel_diff_max"
    ] == C1.PERIODIC_PHASE_MEDIAN_REL_DIFF_MAX


def test_build_is_read_only_for_upstream_config_and_anchors():
    paths = [
        ROOT / C.DEFAULT_UPSTREAM_ROOT / "phase0/canonical_config.json",
        *[
            ROOT / C.DEFAULT_UPSTREAM_ROOT / f"anchors/seed{s}/anchor.json"
            for s in C.PRIMARY_SEEDS
        ],
        ROOT / C.PANEL_PATH,
    ]
    before = {p: C.sha256_file(p) for p in paths}
    C.build_manifest(ROOT)
    after = {p: C.sha256_file(p) for p in paths}
    assert before == after


def test_fixed_panel_manifest_is_self_hashed_and_seed_aligned(manifest):
    panel_path = ROOT / C.PANEL_PATH
    panels = json.loads(panel_path.read_text())
    config_shas = {
        seed: row["canonical_config_sha"]
        for seed, row in manifest["per_seed"].items()
    }
    C.validate_panel_manifest(panels, config_shas=config_shas)
    assert manifest["provenance"]["panel_manifest_sha256"] == (
        panels["manifest_sha256"]
    )
    assert manifest["provenance"]["panel_manifest_file_sha256"] == (
        C.sha256_file(panel_path)
    )
    assert set(manifest["provenance"]["producer_file_sha256"]) == {
        str(path) for path in C.PRODUCTION_PRODUCER_PATHS
    }


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda d: d["seeds"]["1"]["analysis_panel_E_ids"].pop(),
            "self-hash mismatch",
        ),
        (
            lambda d: d["seeds"]["1"].update(activity_independent=False),
            "self-hash mismatch",
        ),
        (
            lambda d: d["seeds"]["1"].update(config_sha="0" * 64),
            "self-hash mismatch",
        ),
    ],
)
def test_panel_manifest_mutations_fail_closed(manifest, mutation, match):
    panels = json.loads((ROOT / C.PANEL_PATH).read_text())
    mutation(panels)
    with pytest.raises(C.ContractInputError, match=match):
        C.validate_panel_manifest(
            panels,
            config_shas={
                seed: row["canonical_config_sha"]
                for seed, row in manifest["per_seed"].items()
            },
        )


def test_missing_required_field_fails_closed(manifest):
    broken = copy.deepcopy(manifest)
    del broken["thresholds"]
    with pytest.raises(C.ContractInputError, match="required fields"):
        C.validate_manifest(broken)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda d: d["c0"].update(duration_ms=7999.0), "duration drift"),
        (
            lambda d: d["c0"]["threshold_perturbation"].update(
                values=[-0.2, 0.0, 0.2]
            ),
            "threshold-perturbation drift",
        ),
        (
            lambda d: d["c1"].update(secondary_shell_step_robust_sd=0.5),
            "shell extent drift",
        ),
        (
            lambda d: d["c0"].update(fast_phases=["trough", "peak"]),
            "phase drift",
        ),
        (
            lambda d: d["thresholds"].update(
                c1_periodic_source_phase_corr_min=0.79
            ),
            "decisive threshold manifest drift",
        ),
    ],
)
def test_contract_drift_fails_even_with_a_recomputed_self_hash(manifest, mutator, match):
    broken = copy.deepcopy(manifest)
    mutator(broken)
    broken = _rehash(broken)
    with pytest.raises(C.ContractInputError, match=match):
        C.validate_manifest(broken)


def test_write_once_allows_exact_reuse_and_refuses_overwrite(tmp_path, manifest):
    path = tmp_path / "phasec_manifest.json"
    assert C.write_manifest_once(path, manifest) == "created"
    first = path.read_bytes()
    assert C.write_manifest_once(path, copy.deepcopy(manifest)) == "reused"
    assert path.read_bytes() == first

    changed = copy.deepcopy(manifest)
    changed["design_amendments"]["test_nonce"] = "different-valid-lock"
    changed = _rehash(changed)
    with pytest.raises(C.ImmutableManifestError, match="differs"):
        C.write_manifest_once(path, changed)
    assert path.read_bytes() == first


def test_legacy_v12_requires_explicit_recoverable_invalidation(tmp_path):
    path = tmp_path / "phasec_manifest.json"
    body = {
        "schema": LOCK.LEGACY_PRODUCTION_SCHEMA,
        "legacy": "preserve-me",
    }
    legacy = {**body, "manifest_sha256": C._object_sha(body)}
    path.write_text(json.dumps(legacy, sort_keys=True) + "\n")
    destination = LOCK.invalidate_legacy_manifest(path)
    assert not path.exists()
    assert destination.is_file()
    assert json.loads(destination.read_text()) == legacy

    unknown = tmp_path / "unknown.json"
    unknown_body = {"schema": "unknown_v9"}
    unknown.write_text(json.dumps({
        **unknown_body, "manifest_sha256": C._object_sha(unknown_body)
    }))
    with pytest.raises(C.ImmutableManifestError, match="unknown manifest"):
        LOCK.invalidate_legacy_manifest(unknown)
    assert unknown.is_file()


def test_seed_or_noise_coverage_cannot_silently_shrink(manifest):
    broken = copy.deepcopy(manifest)
    del broken["per_seed"]["4"]
    broken = _rehash(broken)
    with pytest.raises(C.ContractInputError, match="coverage drift"):
        C.validate_manifest(broken)

    broken = copy.deepcopy(manifest)
    broken["per_seed"]["1"]["c0_carrier_states"]["rising"]["noise_banks"].pop()
    broken = _rehash(broken)
    with pytest.raises(C.ContractInputError, match="noise coverage drift"):
        C.validate_manifest(broken)


def test_existing_manifest_with_valid_but_different_sha_is_rejected(manifest):
    changed = copy.deepcopy(manifest)
    changed["design_amendments"]["test_nonce"] = "different-valid-lock"
    changed = _rehash(changed)
    C.validate_manifest(changed)
    with pytest.raises(C.ImmutableManifestError, match="existing="):
        C.assert_manifest_matches(changed, manifest)


def _rehash_coordinate(payload):
    value = copy.deepcopy(payload)
    value.pop("manifest_sha256", None)
    value.pop("semantic_sha256", None)
    value["semantic_sha256"] = C._object_sha(value)
    value["manifest_sha256"] = C._object_sha(value)
    return value


def _synthetic_coordinate_manifest(root, directory, resolution, input_path, input_manifest):
    seeds = (1, 3, 4) if resolution == "dt" else (1, 3)
    rows = {}
    for seed in seeds:
        # The authoritative checkpoints carry NE in slow._I_I_last.  Production
        # validation deliberately refuses a cropped toy field.
        authority = (
            input_manifest["per_seed"][str(seed)]
            if resolution == "dt"
            else input_manifest["per_seed"][str(seed)][
                "resolution_confirmations"
            ]["dt2"]
        )
        with np.load(root / authority["c1_source_states"][0]["path"],
                     allow_pickle=False) as source_npz:
            n_e = int(np.asarray(source_npz["slow._I_I_last"]).size)
        sources, observed = C._authoritative_coordinate_sources(
            input_manifest,
            root=root,
            resolution=resolution,
            seed=seed,
            n_e=n_e,
        )
        core = np.zeros(n_e, bool)
        core[:max(1, n_e // 4)] = True
        axis = np.linspace(-1.0, 1.0, n_e, dtype=np.float64)
        perpendicular = np.sin(
            np.linspace(0.0, 4.0 * np.pi, n_e, dtype=np.float64)
        )
        coordinates = N.build_coordinate_set(
            observed,
            core_mask=core,
            axis_coord=axis,
            perpendicular_coord=perpendicular,
        )
        for cell in coordinates["primary"]:
            if cell["exact_observed_anchor"]:
                source = sources[(cell["trajectory_id"], cell["left_stage"])]
                cell["source_state_ref"] = {
                    key: source[key] for key in (
                        "path", "file_sha256", "state_hash",
                        "slow_state_sha256",
                    )
                }
        arrays = N.coordinate_array_payload(coordinates)
        npz_path = directory / f"{resolution}_seed{seed}.npz"
        npz_path.write_bytes(N.deterministic_npz_bytes(arrays))
        native = input_manifest["per_seed"][str(seed)]
        config_sha = (
            native["canonical_config_sha"]
            if resolution == "dt"
            else native["resolution_confirmations"]["dt2"]["config_sha"]
        )
        cells = list(coordinates["primary"]) + list(
            coordinates["secondary_shell"]
        )
        rows[str(seed)] = {
            "seed": seed,
            "config_sha": config_sha,
            "panel_selection_config_sha": native["canonical_config_sha"],
            "npz_path": str(npz_path.relative_to(root)),
            "npz_file_sha256": C.sha256_file(npz_path),
            "npz_semantic_sha256": C._semantic_npz_sha(npz_path),
            "n_E": n_e,
            "input_states": [
                {
                    key: sources[(phase, stage)][key]
                    for key in (
                        "phase", "stage", "path", "file_sha256",
                        "state_hash", "slow_state_sha256",
                    )
                }
                for phase in C.FAST_PHASES for stage in C.PRIMARY_STAGES
            ],
            "cells": [
                B._cell_metadata(cell, index)
                for index, cell in enumerate(cells)
            ],
        }
    semantic_payload = {
        "schema": "zm_phasec1_coordinate_manifest_v2_2026-07-28",
        "resolution": resolution,
        "parent_phasec_input_manifest_path": str(input_path.relative_to(root)),
        "parent_phasec_input_manifest_file_sha256": C.sha256_file(input_path),
        "parent_phasec_input_manifest_sha256": input_manifest["manifest_sha256"],
        "seeds": rows,
    }
    semantic_payload["coverage_attestation"] = C._expected_coordinate_attestation(
        semantic_payload, root=root, input_manifest=input_manifest
    )
    payload = _rehash_coordinate(semantic_payload)
    path = directory / f"coordinate_{resolution}.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _attach_cross_attestation(paths):
    coordinates = {
        resolution: json.loads(path.read_text())
        for resolution, path in paths.items()
    }
    required = [
        ("dt", seed) for seed in C.PRIMARY_SEEDS
    ] + [("dt2", seed) for seed in (1, 3)]
    pairs = set.intersection(*(
        C._coordinate_valid_pairs(
            coordinates[resolution]["seeds"][str(seed)]
        )
        for resolution, seed in required
    ))
    cross = {
        "schema": (
            "zm_phasec1_cross_resolution_coverage_attestation_v1_2026-07-29"
        ),
        "required_resolution_seeds": [
            {"resolution": resolution, "seed": seed}
            for resolution, seed in required
        ],
        "native_primary_valid": coordinates["dt"][
            "coverage_attestation"
        ]["primary_valid"],
        "dt2_primary_valid": coordinates["dt2"][
            "coverage_attestation"
        ]["primary_valid"],
        "homologous_adjacent_primary_pairs": [
            list(pair) for pair in sorted(pairs)
        ],
        "identifiable": bool(pairs),
    }
    for resolution, path in paths.items():
        coordinate = coordinates[resolution]
        coordinate["cross_resolution_coverage_attestation"] = cross
        coordinate = _rehash_coordinate(coordinate)
        path.write_text(json.dumps(coordinate, sort_keys=True) + "\n")


@pytest.fixture(scope="module")
def coordinate_bundle(manifest):
    temp_root = ROOT / "results/topic4_sef_hfo/zm_phase_c_tonic_identity"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phasec-lock-test-", dir=temp_root) as name:
        directory = Path(name)
        input_path = directory / "phasec_input_manifest.json"
        C.write_manifest_once(input_path, manifest)
        coordinate_paths = {
            resolution: _synthetic_coordinate_manifest(
                ROOT, directory, resolution, input_path, manifest
            )
            for resolution in ("dt", "dt2")
        }
        _attach_cross_attestation(coordinate_paths)
        final = C.build_final_manifest(
            ROOT,
            input_path=input_path,
            coordinate_paths=coordinate_paths,
        )
        yield {
            "directory": directory,
            "input_path": input_path,
            "input_manifest": manifest,
            "coordinate_paths": coordinate_paths,
            "final": final,
        }


def test_two_stage_lock_is_acyclic_and_coordinate_mutation_fails_closed(
    coordinate_bundle,
):
    manifest = coordinate_bundle["input_manifest"]
    input_path = coordinate_bundle["input_path"]
    coordinate_paths = coordinate_bundle["coordinate_paths"]
    final = coordinate_bundle["final"]
    assert final["schema"] == C.PHASEC_CONTRACT_VERSION
    assert final["production_authorized"] is True
    C.require_production_manifest(final)
    assert set(final["c1"]["coordinate_manifests"]) == {"dt", "dt2"}
    assert set(final["c1"][
        "coordinate_npz_file_sha256_by_seed_by_resolution"
    ]["dt2"]) == {"1", "3"}
    assert set(final["c1"][
        "coordinate_npz_semantic_sha256_by_seed_by_resolution"
    ]["dt2"]) == {"1", "3"}

    coordinate = json.loads(coordinate_paths["dt"].read_text())
    row = coordinate["seeds"]["1"]
    original_npz = ROOT / row["npz_path"]
    mutated_npz = coordinate_bundle["directory"] / "appended_mutation.npz"
    mutated_npz.write_bytes(original_npz.read_bytes() + b"mutation")
    row["npz_path"] = str(mutated_npz.relative_to(ROOT))
    coordinate = _rehash_coordinate(coordinate)
    with pytest.raises(C.ContractInputError, match="NPZ file hash drift"):
        C.validate_coordinate_manifest(
            coordinate,
            root=ROOT,
            input_manifest=manifest,
            input_path=input_path,
        )


def test_coordinate_authority_rejects_self_consistent_source_replacement(
    coordinate_bundle,
):
    coordinate = json.loads(
        coordinate_bundle["coordinate_paths"]["dt"].read_text()
    )
    coordinate["seeds"]["1"]["input_states"][0]["state_hash"] = "f" * 64
    coordinate = _rehash_coordinate(coordinate)
    with pytest.raises(C.ContractInputError, match="source provenance drift"):
        C.validate_coordinate_manifest(
            coordinate,
            root=ROOT,
            input_manifest=coordinate_bundle["input_manifest"],
            input_path=coordinate_bundle["input_path"],
        )


def test_coordinate_authority_rejects_duplicate_primary_inventory(
    coordinate_bundle,
):
    coordinate = json.loads(
        coordinate_bundle["coordinate_paths"]["dt"].read_text()
    )
    cells = coordinate["seeds"]["1"]["cells"]
    cells[1]["cell_id"] = cells[0]["cell_id"]
    coordinate = _rehash_coordinate(coordinate)
    with pytest.raises(C.ContractInputError, match="canonical cell inventory drift"):
        C.validate_coordinate_manifest(
            coordinate,
            root=ROOT,
            input_manifest=coordinate_bundle["input_manifest"],
            input_path=coordinate_bundle["input_path"],
        )


@pytest.mark.parametrize(
    ("array_row", "label", "error"),
    [(1, "midpoint", "canonical state drift"),
     (10, "shell", "physical gate drift")],
)
def test_coordinate_authority_recomputes_primary_and_shell_states(
    coordinate_bundle, array_row, label, error,
):
    coordinate = json.loads(
        coordinate_bundle["coordinate_paths"]["dt"].read_text()
    )
    row = coordinate["seeds"]["1"]
    source_npz = ROOT / row["npz_path"]
    with np.load(source_npz, allow_pickle=False) as data:
        arrays = {name: np.array(data[name], copy=True) for name in data.files}
    arrays["z"][array_row, 0] += 1e-5
    mutated_npz = coordinate_bundle["directory"] / f"mutated_{label}.npz"
    mutated_npz.write_bytes(N.deterministic_npz_bytes(arrays))
    row["npz_path"] = str(mutated_npz.relative_to(ROOT))
    row["npz_file_sha256"] = C.sha256_file(mutated_npz)
    row["npz_semantic_sha256"] = C._semantic_npz_sha(mutated_npz)
    coordinate = _rehash_coordinate(coordinate)
    with pytest.raises(C.ContractInputError, match=error):
        C.validate_coordinate_manifest(
            coordinate,
            root=ROOT,
            input_manifest=coordinate_bundle["input_manifest"],
            input_path=coordinate_bundle["input_path"],
        )


def test_final_authority_rejects_empty_homologous_intersection(
    coordinate_bundle,
):
    final = copy.deepcopy(coordinate_bundle["final"])
    coverage = final["c1"]["coordinate_coverage_attestation"]
    coverage["homologous_adjacent_primary_pairs"] = []
    coverage["identifiable"] = False
    final = _rehash(final)
    with pytest.raises(C.ContractInputError, match="coverage attestation drift"):
        C.require_production_manifest(final)
