import json
import os

import numpy as np

import scripts.analyze_topic4_zm_phasec0 as A


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle)


def _real_fixture(
    tmp_path, monkeypatch, seed_classes, *, corrupt_seed=None,
    null_draw_mismatch_seed=None,
):
    old_out, old_boot = A.OUT, A.N_BOOT
    A.OUT = str(tmp_path)
    A.N_BOOT = 100
    monkeypatch.setattr(A, "MANIFEST_PATH", str(tmp_path / "phasec_manifest.json"))
    monkeypatch.setattr(A, "PANELS_PATH", str(tmp_path / "phasec_panels.json"))
    monkeypatch.setattr(A.PCC, "validate_manifest", lambda value: None)
    manifest = {
        "manifest_sha256": "locked-manifest",
        "provenance": {
            "producer_file_sha256": {"producer.py": "locked-producer-sha"}
        },
        "per_seed": {},
    }
    panel_rows = {}
    for seed in A.SEEDS:
        config_sha = f"config-{seed}"
        panel = {
            "seed": seed,
            "config_sha": config_sha,
            "NE": 8,
            "analysis_panel_E_ids": [0, 1, 2, 3],
            "analysis_panel_n_core": 2,
            "analysis_panel_n_surround": 2,
            "pairwise_panel_E_ids": [0, 1, 2, 3],
            "pairwise_panel_n_core": 2,
            "pairwise_panel_n_surround": 2,
            "selection": "locked-fixture",
            "activity_independent": True,
        }
        panel["panel_sha256"] = A._object_sha(panel)
        panel_rows[str(seed)] = panel

        def source(state):
            return {
                "state": {
                    "state_hash": f"state-{seed}-{state}",
                    "file_sha256": f"state-file-{seed}-{state}",
                },
                "noise_banks": [
                    {
                        "replicate": noise,
                        "bank_sha": f"bank-{seed}-{state}-{noise}",
                    }
                    for noise in A.NOISES
                ],
            }

        manifest["per_seed"][str(seed)] = {
            "canonical_config_sha": config_sha,
            "c0_pre_entry_gain_control": source("pre_entry__natural"),
            "c0_carrier_states": {
                "rising": source("bounded_mid__rising"),
                "peak": source("bounded_mid__peak"),
            },
        }
    panels = {
        "schema": "fixture",
        "seeds": panel_rows,
    }
    panels["manifest_sha256"] = A._object_sha(panels)
    _write_json(A.MANIFEST_PATH, manifest)
    _write_json(A.PANELS_PATH, panels)
    manifest_file_sha = A._sha256(A.MANIFEST_PATH)
    runtime_provenance = {
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha,
        "producer_sha256": manifest["provenance"]["producer_file_sha256"],
    }

    for seed in A.SEEDS:
        identity = seed_classes[seed]
        sat = identity == "sat"
        rho, cv2, ref = (0.85, 0.05, 0.90) if sat else (0.05, 0.95, 0.05)
        panel = panel_rows[str(seed)]
        for phase in A.PHASES:
            source = manifest["per_seed"][str(seed)]["c0_carrier_states"][
                phase.rsplit("__", 1)[-1]
            ]
            banks = {row["replicate"]: row for row in source["noise_banks"]}
            for noise in A.NOISES:
                root = tmp_path / f"obs-{seed}-{phase}-{noise}"
                root.mkdir(parents=True)
                npz = root / "observables.npz"
                arrays = {
                    "hierarchical_schema": np.asarray(A.HIERARCHICAL_SCHEMA),
                    "block_ms": np.asarray(500.0),
                    "ceiling_window_ms": np.asarray(250.0),
                    "ceiling_stride_ms": np.asarray(50.0),
                    "active_area_window_ms": np.asarray(25.0),
                    "spatial_grid_n": np.asarray(16),
                    "spatial_grid_n_occupied_E": np.asarray(256),
                    "spatial_grid_all_E_bins_occupied": np.asarray(True),
                    "spatial_active_floor_hz": np.asarray(5.0),
                    "spatial_area_denominator": np.asarray(
                        "anatomy_occupied_E_grid_bins"
                    ),
                    "pairwise_bin_ms": np.asarray(5.0),
                    "pairwise_null_draws": np.asarray(100),
                    "rho80_active_core_by_block_window": np.full((16, 6), rho),
                    "block_isi_cv2_by_panel_neuron": np.full((16, 4), cv2),
                    "block_refractory_isi_fraction_by_panel_neuron":
                        np.full((16, 4), ref),
                    "pair_corr_by_block_and_pair": np.zeros((16, 6)),
                    "pair_null_median_by_block_and_draw": np.full(
                        (16, 3, 100), 0.05
                    ),
                    "pair_null_stratum_names": np.asarray(
                        A.PCM.PAIR_NULL_STRATUM_NAMES
                    ),
                    "active_area_fraction_by_block_window": np.full((16, 20), 0.30),
                    "analysis_panel_E_ids": np.asarray(
                        panel["analysis_panel_E_ids"], int
                    ),
                    "pairwise_panel_E_ids": np.asarray(
                        panel["pairwise_panel_E_ids"], int
                    ),
                }
                if corrupt_seed == seed and phase == A.PHASES[0] and noise == A.NOISES[0]:
                    arrays.pop("pair_corr_by_block_and_pair")
                if (
                    null_draw_mismatch_seed == seed
                    and phase == A.PHASES[0]
                    and noise == A.NOISES[0]
                ):
                    arrays["pair_null_median_by_block_and_draw"] = arrays[
                        "pair_null_median_by_block_and_draw"
                    ][:, :, :20]
                np.savez(npz, **arrays)
                _write_json(A._identity_path("dt", seed, phase, noise), {
                    "status": "complete",
                    "scientific_end_reason": None,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "panel_sha256": panel["panel_sha256"],
                    "seed": seed,
                    "resolution": "dt",
                    "state_tag": phase,
                    "replicate": noise,
                    "config_sha": manifest["per_seed"][str(seed)][
                        "canonical_config_sha"
                    ],
                    "state_hash": source["state"]["state_hash"],
                    "state_file_sha256": source["state"]["file_sha256"],
                    "noise_bank_sha": banks[noise]["bank_sha"],
                    "burn_in_ms": 500.0,
                    "measure_ms": 8000.0,
                    "evidence_value": "production",
                    "carrier_gates": {
                        "runaway": False,
                        "whole_sheet_plateau": False,
                        "empirical_rest_dwell": False,
                    },
                    "runtime_provenance": runtime_provenance,
                    "observables_path": str(npz),
                    "observables_sha256": A._sha256(npz),
                })

        for state in A.GAIN_STATES:
            source = (
                manifest["per_seed"][str(seed)]["c0_pre_entry_gain_control"]
                if state == "pre_entry__natural"
                else manifest["per_seed"][str(seed)]["c0_carrier_states"][
                    state.rsplit("__", 1)[-1]
                ]
            )
            banks = {row["replicate"]: row for row in source["noise_banks"]}
            slope = 20.0 if state == "pre_entry__natural" else (
                2.0 if sat else 18.0
            )
            for noise in A.NOISES:
                common = {
                    "status": "complete",
                    "scientific_end_reason": None,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "seed": seed,
                    "resolution": "dt",
                    "state_tag": state,
                    "replicate": noise,
                    "config_sha": manifest["per_seed"][str(seed)][
                        "canonical_config_sha"
                    ],
                    "state_hash": source["state"]["state_hash"],
                    "state_file_sha256": source["state"]["file_sha256"],
                    "noise_bank_sha": banks[noise]["bank_sha"],
                    "gain_plateau_gate_pass": True,
                    "runtime_provenance": runtime_provenance,
                }
                _write_json(
                    A._gain_path("dt", seed, state, noise, 0.0, 0),
                    {**common, "core_rate_hz": 100.0,
                     "core_rate_500ms_hz": [100.0, 100.0]},
                )
                for delta in A.DELTAS:
                    for sign in (-1, 1):
                        rate = 100.0 - sign * slope * delta
                        _write_json(
                            A._gain_path(
                                "dt", seed, state, noise, delta, sign
                            ),
                            {**common, "core_rate_hz": rate,
                             "core_rate_500ms_hz": [rate, rate]},
                        )
    try:
        return A.analyze("dt", A.SEEDS)
    finally:
        A.OUT, A.N_BOOT = old_out, old_boot


def _hier_run(*, phase, rho, gain, cv2, ref, corr=0.0, null=0.05, area=0.25):
    """Small sufficient-stat fixture with real block and sampling dimensions."""
    n_block, n_neuron, n_pair, n_null = 4, 12, 20, 16
    return {
        "phase": phase,
        "noise": "fixture",
        "hierarchical": {
            "rho80_active_core_by_block_window": np.full(
                (n_block, 6), rho, dtype=float
            ),
            "block_isi_cv2_by_panel_neuron": np.full(
                (n_block, n_neuron), cv2, dtype=float
            ),
            "block_refractory_isi_fraction_by_panel_neuron": np.full(
                (n_block, n_neuron), ref, dtype=float
            ),
            "pair_corr_by_block_and_pair": np.full(
                (n_block, n_pair), corr, dtype=float
            ),
            "pair_null_median_by_block_and_draw": np.full(
                (n_block, 3, n_null), null, dtype=float
            ),
            "active_area_fraction_by_block_window": np.full(
                (n_block, 20), area, dtype=float
            ),
            "pair_strata": np.resize(
                np.asarray([0, 1, 2], np.int8), n_pair
            ),
        },
        "gain_ratio_samples": np.full(n_block, gain, dtype=float),
        "runaway": False,
        "whole_sheet_plateau": False,
        "empirical_rest_dwell": False,
    }


def test_hierarchical_bootstrap_uses_block_neuron_pair_and_continuation_levels():
    runs = [
        _hier_run(
            phase=phase, rho=0.05, gain=0.9, cv2=0.95, ref=0.05
        )
        for phase in A.PHASES
        for _ in range(3)
    ]
    out1 = A.hierarchical_seed_bootstrap(runs, seed=13, n_boot=300)
    out2 = A.hierarchical_seed_bootstrap(runs, seed=13, n_boot=300)
    assert out1 == out2
    assert out1["structure"] == (
        "500ms_blocks_then_locked_neurons_or_pairs_then_continuations"
    )
    assert out1["rho80_active_core"]["hi"] < 0.20
    assert out1["gain_relative_to_preentry"]["lo"] > 0.50
    assert out1["isi_cv2_median"]["lo"] > 0.70


def test_missing_hierarchical_npz_field_fails_closed(tmp_path):
    arrays = tmp_path / "observables.npz"
    np.savez(
        arrays,
        hierarchical_schema=np.asarray(A.HIERARCHICAL_SCHEMA),
        block_ms=np.asarray(500.0),
        rho80_active_core_by_block_window=np.ones((2, 6)),
    )
    row = {
        "observables_path": str(arrays),
        "observables_sha256": A._sha256(arrays),
    }
    out = A._load_hierarchical_npz(row)
    assert out["status"] == "blocked"
    assert "missing_npz_fields" in out["reason"]


def test_mismatched_pairwise_null_array_draw_count_fails_closed(
    tmp_path, monkeypatch
):
    blocked = _real_fixture(
        tmp_path, monkeypatch, {1: "sat", 3: "sat", 4: "sat"},
        null_draw_mismatch_seed=1,
    )
    seed1 = next(row for row in blocked["seed_rows"] if row["seed"] == 1)
    assert blocked["aggregate"]["verdict"] == "C0_no_evidence"
    assert seed1["klass"] == "C0_blocked_observables"
    assert "pairwise_null_draw_count_mismatch" in seed1["reason"]


def test_run_level_conjunction_prevents_disjoint_saturation_evidence():
    high_rho_only = _hier_run(
        phase=A.PHASES[0], rho=0.8, gain=0.9, cv2=0.1, ref=0.1
    )
    low_rho_ref_locked = _hier_run(
        phase=A.PHASES[0], rho=0.1, gain=0.9, cv2=0.1, ref=0.95
    )
    assert A.classify_run_joint(high_rho_only)["klass"] == "mixed"
    assert A.classify_run_joint(low_rho_ref_locked)["klass"] == "mixed"


def test_ai_gate_uses_spatial_active_area_and_rejects_whole_sheet_extent():
    local = _hier_run(
        phase=A.PHASES[0], rho=0.05, gain=0.9, cv2=0.95, ref=0.05,
        area=0.25,
    )
    whole_sheet = _hier_run(
        phase=A.PHASES[0], rho=0.05, gain=0.9, cv2=0.95, ref=0.05,
        area=0.75,
    )
    assert A.classify_run_joint(local)["klass"] == (
        "balanced_AI_tonic_candidate"
    )
    assert A.classify_run_joint(whole_sheet)["klass"] == "mixed"


def test_ai_pair_null_is_matched_within_each_core_surround_stratum():
    run = _hier_run(
        phase=A.PHASES[0], rho=0.05, gain=0.9, cv2=0.95, ref=0.05,
        corr=0.0, null=0.05, area=0.25,
    )
    strata = run["hierarchical"]["pair_strata"]
    pair = run["hierarchical"]["pair_corr_by_block_and_pair"]
    # The pooled median remains harmless, but core-surround pairs exceed their
    # own matched shift null and must block the AI label.
    pair[:, strata == 1] = 0.08
    run["hierarchical"]["pair_null_median_by_block_and_draw"][:, 1, :] = 0.01
    point = A._continuation_point(run)
    assert abs(point["pairwise_observed_median"]) < 0.10
    assert point["pairwise_stratum_max_excess"] > 0.0
    assert A.classify_run_joint(run)["klass"] == "mixed"


def test_phase_rule_is_two_of_three_complete_joint_runs():
    rows = []
    for phase in A.PHASES:
        rows.extend([
            {**_hier_run(phase=phase, rho=0.05, gain=0.9, cv2=0.95, ref=0.05),
             "noise": "n1"},
            {**_hier_run(phase=phase, rho=0.05, gain=0.9, cv2=0.95, ref=0.05),
             "noise": "n2"},
            {**_hier_run(phase=phase, rho=0.35, gain=0.4, cv2=0.4, ref=0.4),
             "noise": "n3"},
        ])
    out = A.phase_support_from_runs(rows)
    assert out["balanced_AI_tonic_candidate"]["passes"]
    assert not out["refractory_saturated_branch"]["passes"]


def test_one_terminal_scientific_failure_per_phase_does_not_erase_two_of_three():
    rows = []
    for phase in A.PHASES:
        rows.extend([
            {**_hier_run(phase=phase, rho=0.05, gain=0.9, cv2=0.95, ref=0.05),
             "noise": "n1"},
            {**_hier_run(phase=phase, rho=0.05, gain=0.9, cv2=0.95, ref=0.05),
             "noise": "n2"},
            {
                "phase": phase,
                "noise": "n3",
                "scientific_failure": "runaway",
                "runaway": True,
            },
        ])
    out = A._seed_class_from_hierarchy(rows, seed=9)
    assert out["klass"] == "balanced_AI_tonic_candidate"
    assert out["hierarchical_ci"]["n_numeric_continuations"] == 4


def test_fast_phase_is_a_fixed_bootstrap_stratum():
    runs = []
    for phase, rho in zip(A.PHASES, (0.01, 0.19)):
        runs.extend([
            _hier_run(phase=phase, rho=rho, gain=0.9, cv2=0.95, ref=0.05)
            for _ in range(2)
        ])
    out = A.hierarchical_seed_bootstrap(runs, seed=21, n_boot=200)
    assert out["n_numeric_continuations"] == 4
    assert out["n_drawn_continuations"] == 6


def test_scientific_failure_is_not_technical_block():
    assert A._part_failure_kind({
        "status": "scientific_failure", "scientific_end_reason": "runaway"
    }) == "scientific"
    assert A._part_failure_kind({
        "status": "scientific_failure",
        "scientific_end_reason": "truncated_or_missing_observable",
    }) == "technical"
    assert A._gain_failure_kind("nonlinear_or_nonmonotone") == "scientific"
    assert A._gain_failure_kind("gain_manifest_mismatch") == "technical"


def test_analyzer_fails_closed_on_missing_or_mutated_runtime_producer_map():
    manifest = {
        "manifest_sha256": "semantic",
        "provenance": {"producer_file_sha256": {"producer.py": "locked"}},
    }
    exact = {
        "runtime_provenance": {
            "manifest_sha256": "semantic",
            "manifest_file_sha256": "file",
            "producer_sha256": {"producer.py": "locked"},
        }
    }
    assert A._runtime_provenance_failure(exact, manifest, "file") is None
    missing = {}
    assert A._runtime_provenance_failure(
        missing, manifest, "file"
    ) == "missing_runtime_provenance"
    mutated = {
        "runtime_provenance": {
            **exact["runtime_provenance"],
            "producer_sha256": {"producer.py": "mutated"},
        }
    }
    assert A._runtime_provenance_failure(
        mutated, manifest, "file"
    ) == "runtime_producer_hash_mismatch"


def test_resolution_glue_requires_two_homologous_supporting_seeds():
    native = {
        "aggregate": {
            "verdict": "balanced_AI_tonic_candidate_supported",
            "supporting_seeds": [1, 3],
        },
        "seed_rows": [
            {"seed": 1, "klass": "balanced_AI_tonic_candidate"},
            {"seed": 3, "klass": "balanced_AI_tonic_candidate"},
            {"seed": 4, "klass": "mixed_or_indeterminate_tonic_branch"},
        ],
    }
    dt2 = {
        "aggregate": {
            "verdict": "balanced_AI_tonic_candidate_supported",
            "supporting_seeds": [1, 3],
        },
        "seed_rows": [
            {"seed": 1, "klass": "balanced_AI_tonic_candidate",
             "homologous_anchor_validated": True},
            {"seed": 3, "klass": "balanced_AI_tonic_candidate",
             "homologous_anchor_validated": True},
        ],
    }
    assert A.combine_resolution_summaries(native, dt2)["verdict"] == (
        "balanced_AI_tonic_candidate_supported"
    )
    dt2["seed_rows"][1]["klass"] = "refractory_saturated_branch"
    assert A.combine_resolution_summaries(native, dt2)["verdict"] == (
        "resolution_sensitive_identity"
    )


def test_dt2_gain_provenance_uses_independent_source_and_native_lineage():
    def family(prefix):
        return {
            "state": {
                "state_hash": f"{prefix}-state",
                "file_sha256": f"{prefix}-file",
            },
            "noise_banks": [{
                "replicate": "noise_replay",
                "bank_sha": f"{prefix}-noise",
            }],
        }

    manifest = {
        "manifest_sha256": "manifest",
        "provenance": {"producer_file_sha256": {"p.py": "p"}},
        "per_seed": {
            "1": {
                "canonical_config_sha": "native-config",
                "c0_carrier_states": {"rising": family("native")},
                "c0_pre_entry_gain_control": family("native-pre"),
                "resolution_confirmations": {
                    "dt2": {
                        "parent_config_sha": "native-config",
                        "config_sha": "dt2-config",
                        "c0_carrier_states": {"rising": family("dt2")},
                        "c0_pre_entry_gain_control": family("dt2-pre"),
                    }
                },
            }
        },
    }
    expected = A._manifest_source(
        manifest, 1, "bounded_mid__rising", "noise_replay",
        resolution="dt2",
    )
    assert expected["state_hash"] == "dt2-state"
    assert expected["native_state_hash"] == "native-state"
    row = {
        "manifest_sha256": "manifest",
        "seed": 1,
        "state_tag": "bounded_mid__rising",
        "replicate": "noise_replay",
        "resolution": "dt2",
        "state_hash": "dt2-state",
        "state_file_sha256": "dt2-file",
        "noise_bank_sha": "dt2-noise",
        "config_sha": "dt2-config",
        "homologous_anchor_validated": True,
        "homologous_native_state_hash": "native-state",
        "homologous_parent_config_sha": "native-config",
        "runtime_provenance": {
            "manifest_sha256": "manifest",
            "manifest_file_sha256": "manifest-file",
            "producer_sha256": {"p.py": "p"},
        },
        "status": "complete",
        "scientific_end_reason": None,
        "gain_plateau_gate_pass": True,
        "core_rate_500ms_hz": [1.0, 1.0],
    }
    assert A._validate_gain_payload(
        row, manifest=manifest, manifest_file_sha="manifest-file",
        seed=1, state="bounded_mid__rising", noise="noise_replay",
        resolution="dt2", expected=expected,
    ) == "ok"
    mutated = {**row, "state_hash": "native-state"}
    assert A._validate_gain_payload(
        mutated, manifest=manifest, manifest_file_sha="manifest-file",
        seed=1, state="bounded_mid__rising", noise="noise_replay",
        resolution="dt2", expected=expected,
    ) == "gain_provenance_mismatch"
    mutated = {**row, "state_file_sha256": "wrong-file"}
    assert A._validate_gain_payload(
        mutated, manifest=manifest, manifest_file_sha="manifest-file",
        seed=1, state="bounded_mid__rising", noise="noise_replay",
        resolution="dt2", expected=expected,
    ) == "gain_provenance_mismatch"
    mutated = {**row, "homologous_native_state_hash": "wrong"}
    assert A._validate_gain_payload(
        mutated, manifest=manifest, manifest_file_sha="manifest-file",
        seed=1, state="bounded_mid__rising", noise="noise_replay",
        resolution="dt2", expected=expected,
    ) == "dt2_gain_homologous_anchor_unvalidated"


def test_real_hierarchical_artifacts_support_replicated_identity(
    tmp_path, monkeypatch
):
    out = _real_fixture(
        tmp_path, monkeypatch, {1: "sat", 3: "sat", 4: "sat"}
    )
    assert out["aggregate"]["verdict"] == "refractory_saturated_branch_supported"
    assert all(
        row["hierarchical_ci"]["structure"].startswith("500ms_blocks")
        for row in out["seed_rows"]
    )


def test_real_hierarchical_artifacts_preserve_opposite_seed(
    tmp_path, monkeypatch
):
    out = _real_fixture(
        tmp_path, monkeypatch, {1: "sat", 3: "sat", 4: "ai"}
    )
    assert out["aggregate"]["verdict"] == "seed_heterogeneous_identity"


def test_real_hierarchical_artifact_schema_failure_is_technical_block(
    tmp_path, monkeypatch
):
    out = _real_fixture(
        tmp_path, monkeypatch, {1: "sat", 3: "sat", 4: "sat"},
        corrupt_seed=3,
    )
    assert out["aggregate"]["verdict"] == "C0_no_evidence"
    seed3 = next(row for row in out["seed_rows"] if row["seed"] == 3)
    assert seed3["klass"] == "C0_blocked_observables"
